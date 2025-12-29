"""
Emotional Memory Repository Layer
Handles all database operations for emotional memory management
Following clean architecture and SOLID principles
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import hashlib
import json
import re
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.dialects.postgresql import insert
import numpy as np
from redis import asyncio as redis_async
from ..db.emotional_memory_models import (
    EmotionalMemory,
    EmotionalEvent,
    UserEmotionalPreference,
    EmotionalTriggerPattern,
    MemoryAccessLog,
    EmotionalMemoryCache,
    StorageLayerType,
    PrivacyLevelType
)
from ..core.utils.timezone_utils import utc_now
from ..core.security.encryption import EncryptionService
from ..core.monitoring.metrics_collector import MetricsCollector
from ..core.exceptions import RepositoryException, DataIntegrityException

class EmotionalMemoryRepository:
    """Repository for emotional memory operations with multi-tier storage"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis_async.Redis,
        encryption_service: EncryptionService,
        metrics_collector: MetricsCollector
    ):
        self.db = db_session
        self.redis = redis_client
        self.encryption = encryption_service
        self.metrics = metrics_collector
        
        # Storage tier thresholds (in hours)
        self.HOT_STORAGE_THRESHOLD = 24
        self.WARM_STORAGE_THRESHOLD = 168  # 7 days
        
    async def create_memory(
        self,
        user_id: str,
        emotion_data: Dict[str, Any],
        encrypt: bool = False
    ) -> EmotionalMemory:
        """Create a new emotional memory with proper storage tier assignment"""
        try:
            # Start metrics tracking
            with self.metrics.timer('repository.emotional_memory.create'):
                
                # Prepare memory data
                memory_data = {
                    'user_id': user_id,
                    'session_id': emotion_data.get('session_id'),
                    'emotion_type': emotion_data['emotion_type'],
                    'intensity': emotion_data['intensity'],
                    'valence': emotion_data.get('valence', 0),
                    'arousal': emotion_data.get('arousal', 0.5),
                    'content': emotion_data['content'],
                    'tags': emotion_data.get('tags', []),
                    'context_data': emotion_data.get('context', {}),
                    'source': emotion_data.get('source', 'user_input'),
                    'storage_layer': StorageLayerType.HOT,
                    'importance_score': self._calculate_importance(emotion_data)
                }
                
                # Handle encryption if requested
                if encrypt:
                    encrypted_content, key_id = await self.encryption.encrypt_data(
                        memory_data['content']
                    )
                    memory_data['content'] = encrypted_content
                    memory_data['encryption_key_id'] = key_id
                    memory_data['is_encrypted'] = True
                
                # Generate content embedding for semantic search
                if emotion_data.get('content'):
                    memory_data['content_embedding'] = await self._generate_embedding(
                        emotion_data['content']
                    )
                
                # Create memory record
                memory = EmotionalMemory(**memory_data)
                self.db.add(memory)
                await self.db.flush()
                
                # Cache in Redis for hot storage
                await self._cache_memory(memory)
                
                # Log access
                await self._log_memory_access(
                    memory.id,
                    user_id,
                    'create',
                    emotion_data.get('session_id')
                )
                
                await self.db.commit()
                
                # Track metrics
                self.metrics.increment('emotional_memories.created')
                self.metrics.increment(f'emotional_memories.{emotion_data["emotion_type"]}')
                
                return memory
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('emotional_memories.create_error')
            raise RepositoryException(f"Failed to create emotional memory: {str(e)}")
    
    async def get_memory_by_id(
        self,
        memory_id: UUID,
        user_id: str,
        increment_access: bool = True
    ) -> Optional[EmotionalMemory]:
        """Retrieve memory by ID with access tracking and cache optimization"""
        try:
            with self.metrics.timer('repository.emotional_memory.get_by_id'):
                
                # Try hot cache first
                cached = await self._get_cached_memory(memory_id)
                if cached:
                    self.metrics.increment('emotional_memories.cache_hit')
                    if increment_access:
                        await self._increment_access_count(memory_id)
                    return cached
                
                # Query database
                query = select(EmotionalMemory).where(
                    and_(
                        EmotionalMemory.id == memory_id,
                        EmotionalMemory.user_id == user_id,
                        EmotionalMemory.deleted_at.is_(None)
                    )
                )
                
                result = await self.db.execute(query)
                memory = result.scalar_one_or_none()
                
                if memory:
                    # Decrypt if necessary
                    if memory.is_encrypted:
                        memory.content = await self.encryption.decrypt_data(
                            memory.content,
                            memory.encryption_key_id
                        )
                    
                    # Update access tracking
                    if increment_access:
                        await self._increment_access_count(memory_id)
                        await self._log_memory_access(
                            memory_id,
                            user_id,
                            'read'
                        )
                    
                    # Re-cache if in hot storage
                    if memory.storage_layer == StorageLayerType.HOT:
                        await self._cache_memory(memory)
                    
                    self.metrics.increment('emotional_memories.cache_miss')
                
                return memory
                
        except Exception as e:
            self.metrics.increment('emotional_memories.get_error')
            raise RepositoryException(f"Failed to retrieve memory: {str(e)}")
    
    async def search_memories(
        self,
        user_id: str,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[EmotionalMemory], int]:
        """Advanced search with filtering, pagination, and relevance ranking"""
        try:
            with self.metrics.timer('repository.emotional_memory.search'):
                
                # Build base query
                query = select(EmotionalMemory).where(
                    and_(
                        EmotionalMemory.user_id == user_id,
                        EmotionalMemory.deleted_at.is_(None)
                    )
                )
                
                # Apply filters
                if filters.get('emotion_type'):
                    query = query.where(
                        EmotionalMemory.emotion_type == filters['emotion_type']
                    )
                
                if filters.get('storage_layer'):
                    storage_layer = filters['storage_layer']
                    if isinstance(storage_layer, str):
                        storage_layer = StorageLayerType(storage_layer)
                    query = query.where(
                        EmotionalMemory.storage_layer == storage_layer
                    )
                
                if filters.get('min_intensity'):
                    query = query.where(
                        EmotionalMemory.intensity >= filters['min_intensity']
                    )
                
                if filters.get('date_from'):
                    query = query.where(
                        EmotionalMemory.timestamp >= filters['date_from']
                    )
                
                if filters.get('date_to'):
                    query = query.where(
                        EmotionalMemory.timestamp <= filters['date_to']
                    )
                
                if filters.get('tags'):
                    query = query.where(
                        EmotionalMemory.tags.contains(filters['tags'])
                    )
                
                search_query = filters.get('search_query')
                if search_query:
                    query_embedding = await self._generate_embedding(search_query)
                    q = np.asarray(query_embedding, dtype=np.float32)

                    candidate_limit = max(limit * 50, 500)
                    candidate_query = query.where(
                        EmotionalMemory.content_embedding.is_not(None)
                    ).order_by(
                        EmotionalMemory.timestamp.desc()
                    ).limit(candidate_limit)

                    result = await self.db.execute(candidate_query)
                    candidates = result.scalars().all()

                    scored = []
                    for memory in candidates:
                        emb = memory.content_embedding
                        if not emb or len(emb) != len(query_embedding):
                            continue
                        v = np.asarray(emb, dtype=np.float32)
                        dist = float(np.linalg.norm(v - q))
                        scored.append((dist, memory))

                    scored.sort(key=lambda x: x[0])
                    total = len(scored)
                    memories = [m for _, m in scored[offset:offset + limit]]
                else:
                    # Default ordering by importance and recency
                    query = query.order_by(
                        EmotionalMemory.importance_score.desc(),
                        EmotionalMemory.timestamp.desc()
                    )

                    # Get total count
                    count_query = select(func.count()).select_from(query.subquery())
                    total_result = await self.db.execute(count_query)
                    total = total_result.scalar()

                    # Apply pagination
                    query = query.limit(limit).offset(offset)

                    # Execute query
                    result = await self.db.execute(query)
                    memories = result.scalars().all()
                
                # Decrypt encrypted memories
                for memory in memories:
                    if memory.is_encrypted:
                        memory.content = await self.encryption.decrypt_data(
                            memory.content,
                            memory.encryption_key_id
                        )
                
                self.metrics.increment('emotional_memories.search', tags={'count': len(memories)})
                
                return memories, total
                
        except Exception as e:
            self.metrics.increment('emotional_memories.search_error')
            raise RepositoryException(f"Failed to search memories: {str(e)}")
    
    async def update_storage_tier(
        self,
        memory_id: UUID,
        new_tier: str
    ) -> bool:
        """Move memory between storage tiers based on access patterns"""
        try:
            with self.metrics.timer('repository.emotional_memory.update_tier'):
                
                # Update storage tier
                query = update(EmotionalMemory).where(
                    EmotionalMemory.id == memory_id
                ).values(
                    storage_layer=StorageLayerType(new_tier),
                    updated_at=utc_now()
                )
                
                result = await self.db.execute(query)
                await self.db.commit()
                
                # Handle cache updates
                if new_tier == 'hot':
                    # Load into hot cache
                    memory = await self.get_memory_by_id(memory_id, increment_access=False)
                    if memory:
                        await self._cache_memory(memory)
                elif new_tier in ['warm', 'cold']:
                    # Remove from hot cache
                    await self._remove_from_cache(memory_id)
                
                self.metrics.increment(f'emotional_memories.tier_migration.{new_tier}')
                
                return result.rowcount > 0
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('emotional_memories.tier_update_error')
            raise RepositoryException(f"Failed to update storage tier: {str(e)}")
    
    async def archive_old_memories(
        self,
        days_threshold: int = 30
    ) -> int:
        """Archive old memories to cold storage"""
        try:
            with self.metrics.timer('repository.emotional_memory.archive'):
                
                cutoff_date = utc_now() - timedelta(days=days_threshold)
                
                # Find memories to archive
                query = update(EmotionalMemory).where(
                    and_(
                        EmotionalMemory.storage_layer.in_([StorageLayerType.HOT, StorageLayerType.WARM]),
                        EmotionalMemory.last_accessed < cutoff_date,
                        EmotionalMemory.importance_score < 0.7,  # Don't archive important memories
                        EmotionalMemory.deleted_at.is_(None)
                    )
                ).values(
                    storage_layer=StorageLayerType.COLD,
                    updated_at=utc_now()
                )
                
                result = await self.db.execute(query)
                await self.db.commit()
                
                archived_count = result.rowcount
                
                self.metrics.increment(
                    'emotional_memories.archived',
                    value=archived_count
                )
                
                return archived_count
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('emotional_memories.archive_error')
            raise RepositoryException(f"Failed to archive memories: {str(e)}")

    async def optimize_storage_tiers(
        self,
        hot_cutoff: datetime,
        warm_cutoff: datetime,
    ) -> Dict[str, int]:
        """根据访问与重要性规则批量调整存储层级"""
        try:
            with self.metrics.timer("repository.emotional_memory.optimize_tiers"):
                results: Dict[str, int] = {
                    "hot_to_warm": 0,
                    "warm_to_hot": 0,
                    "cold_to_warm": 0,
                }

                q1 = (
                    update(EmotionalMemory)
                    .where(
                        and_(
                            EmotionalMemory.storage_layer == StorageLayerType.HOT,
                            EmotionalMemory.last_accessed < hot_cutoff,
                            EmotionalMemory.importance_score < 0.8,
                            EmotionalMemory.deleted_at.is_(None),
                        )
                    )
                    .values(storage_layer=StorageLayerType.WARM, updated_at=utc_now())
                )
                r1 = await self.db.execute(q1)
                results["hot_to_warm"] = r1.rowcount or 0

                q2 = (
                    update(EmotionalMemory)
                    .where(
                        and_(
                            EmotionalMemory.storage_layer == StorageLayerType.WARM,
                            or_(
                                EmotionalMemory.last_accessed >= hot_cutoff,
                                EmotionalMemory.access_count >= 10,
                            ),
                            EmotionalMemory.deleted_at.is_(None),
                        )
                    )
                    .values(storage_layer=StorageLayerType.HOT, updated_at=utc_now())
                )
                r2 = await self.db.execute(q2)
                results["warm_to_hot"] = r2.rowcount or 0

                q3 = (
                    update(EmotionalMemory)
                    .where(
                        and_(
                            EmotionalMemory.storage_layer == StorageLayerType.COLD,
                            EmotionalMemory.last_accessed >= warm_cutoff,
                            EmotionalMemory.importance_score >= 0.7,
                            EmotionalMemory.deleted_at.is_(None),
                        )
                    )
                    .values(storage_layer=StorageLayerType.WARM, updated_at=utc_now())
                )
                r3 = await self.db.execute(q3)
                results["cold_to_warm"] = r3.rowcount or 0

                await self.db.commit()
                return results
        except Exception as e:
            await self.db.rollback()
            raise RepositoryException(f"Failed to optimize storage tiers: {str(e)}")
    
    async def delete_memory(
        self,
        memory_id: UUID,
        user_id: str,
        hard_delete: bool = False
    ) -> bool:
        """Delete memory (soft delete by default)"""
        try:
            with self.metrics.timer('repository.emotional_memory.delete'):
                
                if hard_delete:
                    # Permanent deletion
                    query = delete(EmotionalMemory).where(
                        and_(
                            EmotionalMemory.id == memory_id,
                            EmotionalMemory.user_id == user_id
                        )
                    )
                else:
                    # Soft delete
                    query = update(EmotionalMemory).where(
                        and_(
                            EmotionalMemory.id == memory_id,
                            EmotionalMemory.user_id == user_id
                        )
                    ).values(
                        deleted_at=utc_now(),
                        is_active=False
                    )
                
                result = await self.db.execute(query)
                await self.db.commit()
                
                # Remove from cache
                await self._remove_from_cache(memory_id)
                
                # Log access
                await self._log_memory_access(
                    memory_id,
                    user_id,
                    'delete' if hard_delete else 'soft_delete'
                )
                
                self.metrics.increment('emotional_memories.deleted')
                
                return result.rowcount > 0
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('emotional_memories.delete_error')
            raise RepositoryException(f"Failed to delete memory: {str(e)}")
    
    # Helper methods
    
    async def _cache_memory(self, memory: EmotionalMemory) -> None:
        """Cache memory in Redis for fast access"""
        cache_key = f"emotional_memory:{memory.id}"
        cache_data = {
            'id': str(memory.id),
            'user_id': memory.user_id,
            'emotion_type': memory.emotion_type,
            'intensity': memory.intensity,
            'content': memory.content,
            'timestamp': memory.timestamp.isoformat(),
            'storage_layer': memory.storage_layer.value
        }
        
        # Set with expiration based on storage tier
        ttl = 3600 if memory.storage_layer == StorageLayerType.HOT else 300  # 1 hour for hot, 5 min for others
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(cache_data)
        )
    
    async def _get_cached_memory(self, memory_id: UUID) -> Optional[Dict[str, Any]]:
        """Retrieve memory from cache"""
        cache_key = f"emotional_memory:{memory_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    async def _remove_from_cache(self, memory_id: UUID) -> None:
        """Remove memory from cache"""
        cache_key = f"emotional_memory:{memory_id}"
        await self.redis.delete(cache_key)
    
    async def _increment_access_count(self, memory_id: UUID) -> None:
        """Increment memory access count and update last accessed time"""
        query = update(EmotionalMemory).where(
            EmotionalMemory.id == memory_id
        ).values(
            access_count=EmotionalMemory.access_count + 1,
            last_accessed=utc_now()
        )
        
        await self.db.execute(query)
        await self.db.commit()
    
    async def _log_memory_access(
        self,
        memory_id: UUID,
        user_id: str,
        access_type: str,
        session_id: Optional[str] = None
    ) -> None:
        """Log memory access for analytics"""
        log = MemoryAccessLog(
            memory_id=memory_id,
            user_id=user_id,
            access_type=access_type,
            session_id=session_id
        )
        
        self.db.add(log)
        await self.db.flush()
    
    def _calculate_importance(self, emotion_data: Dict[str, Any]) -> float:
        """Calculate importance score for memory prioritization"""
        importance = 0.5  # Base score
        
        # Adjust based on intensity
        importance += emotion_data.get('intensity', 0) * 0.2
        
        # Adjust based on valence extremity
        valence = emotion_data.get('valence', 0)
        importance += abs(valence) * 0.15
        
        # Adjust based on arousal
        arousal = emotion_data.get('arousal', 0.5)
        importance += arousal * 0.15
        
        # Cap at 1.0
        return min(importance, 1.0)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for semantic search"""
        tokens = re.findall(r"[\\w]+", text.lower())
        dim = 128
        vec = np.zeros(dim, dtype=np.float32)
        for token in tokens:
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % dim
            vec[idx] += 1.0 if (h[4] & 1) else -1.0

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.astype(float).tolist()

class EmotionalEventRepository:
    """Repository for emotional event tracking and analysis"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        metrics_collector: MetricsCollector
    ):
        self.db = db_session
        self.metrics = metrics_collector
    
    async def create_event(
        self,
        memory_id: UUID,
        user_id: str,
        event_data: Dict[str, Any]
    ) -> EmotionalEvent:
        """Create a new emotional event"""
        try:
            with self.metrics.timer('repository.emotional_event.create'):
                
                event = EmotionalEvent(
                    memory_id=memory_id,
                    user_id=user_id,
                    event_type=event_data['event_type'],
                    trigger_source=event_data['trigger_source'],
                    trigger_context=event_data.get('context', {}),
                    parent_event_id=event_data.get('parent_event_id'),
                    causal_strength=event_data.get('causal_strength', 0.5),
                    impact_score=event_data.get('impact_score', 0.5),
                    duration_seconds=event_data.get('duration'),
                    affected_emotions=event_data.get('affected_emotions', [])
                )
                
                self.db.add(event)
                await self.db.flush()
                
                # Update causal chain if parent exists
                if event.parent_event_id:
                    await self._update_causal_chain(event)
                
                await self.db.commit()
                
                self.metrics.increment('emotional_events.created')
                
                return event
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('emotional_events.create_error')
            raise RepositoryException(f"Failed to create emotional event: {str(e)}")
    
    async def get_user_events(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[EmotionalEvent]:
        """Get events for a user with pagination"""
        try:
            with self.metrics.timer('repository.emotional_event.get_user_events'):
                
                query = select(EmotionalEvent).where(
                    EmotionalEvent.user_id == user_id
                ).order_by(
                    EmotionalEvent.timestamp.desc()
                ).limit(limit).offset(offset)
                
                result = await self.db.execute(query)
                events = result.scalars().all()
                
                return events
                
        except Exception as e:
            self.metrics.increment('emotional_events.get_error')
            raise RepositoryException(f"Failed to get user events: {str(e)}")
    
    async def analyze_causal_relationships(
        self,
        user_id: str,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Analyze causal relationships between events"""
        try:
            with self.metrics.timer('repository.emotional_event.analyze_causal'):
                
                cutoff_date = utc_now() - time_window
                
                # Get events within time window
                query = select(EmotionalEvent).where(
                    and_(
                        EmotionalEvent.user_id == user_id,
                        EmotionalEvent.timestamp >= cutoff_date
                    )
                ).options(
                    selectinload(EmotionalEvent.parent_event)
                )
                
                result = await self.db.execute(query)
                events = result.scalars().all()
                
                # Build causal graph
                causal_graph = {}
                for event in events:
                    if event.parent_event_id:
                        if event.parent_event_id not in causal_graph:
                            causal_graph[str(event.parent_event_id)] = []
                        causal_graph[str(event.parent_event_id)].append({
                            'event_id': str(event.id),
                            'event_type': event.event_type,
                            'causal_strength': event.causal_strength,
                            'impact_score': event.impact_score
                        })
                
                return {
                    'total_events': len(events),
                    'causal_chains': len(causal_graph),
                    'graph': causal_graph,
                    'avg_causal_strength': np.mean([e.causal_strength for e in events if e.causal_strength])
                }
                
        except Exception as e:
            self.metrics.increment('emotional_events.analyze_error')
            raise RepositoryException(f"Failed to analyze causal relationships: {str(e)}")
    
    async def _update_causal_chain(self, event: EmotionalEvent) -> None:
        """Update causal chain for an event"""
        # Get parent's causal chain
        parent_query = select(EmotionalEvent).where(
            EmotionalEvent.id == event.parent_event_id
        )
        parent_result = await self.db.execute(parent_query)
        parent = parent_result.scalar_one_or_none()
        
        if parent:
            # Build chain from parent
            chain = parent.causal_chain or []
            chain.append(parent.id)
            
            # Update event's causal chain
            event.causal_chain = chain
            await self.db.flush()

class UserPreferenceRepository:
    """Repository for user emotional preference management"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        metrics_collector: MetricsCollector
    ):
        self.db = db_session
        self.metrics = metrics_collector
    
    async def get_or_create_preferences(
        self,
        user_id: str
    ) -> UserEmotionalPreference:
        """Get or create user preferences"""
        try:
            with self.metrics.timer('repository.user_preference.get_or_create'):
                
                # Try to get existing
                query = select(UserEmotionalPreference).where(
                    UserEmotionalPreference.user_id == user_id
                )
                
                result = await self.db.execute(query)
                preference = result.scalar_one_or_none()
                
                if not preference:
                    # Create new preference profile
                    preference = UserEmotionalPreference(
                        user_id=user_id,
                        dominant_emotions=[],
                        emotion_weights={},
                        preferred_responses={},
                        avoided_triggers=[],
                        response_patterns={},
                        communication_preferences={}
                    )
                    
                    self.db.add(preference)
                    await self.db.commit()
                    
                    self.metrics.increment('user_preferences.created')
                
                return preference
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('user_preferences.error')
            raise RepositoryException(f"Failed to get/create preferences: {str(e)}")
    
    async def update_preferences(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> UserEmotionalPreference:
        """Update user preferences with new learning"""
        try:
            with self.metrics.timer('repository.user_preference.update'):
                
                query = update(UserEmotionalPreference).where(
                    UserEmotionalPreference.user_id == user_id
                ).values(
                    **updates,
                    updated_at=utc_now(),
                    version=UserEmotionalPreference.version + 1
                )
                
                await self.db.execute(query)
                await self.db.commit()
                
                # Get updated preferences
                preference = await self.get_or_create_preferences(user_id)
                
                self.metrics.increment('user_preferences.updated')
                
                return preference
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('user_preferences.update_error')
            raise RepositoryException(f"Failed to update preferences: {str(e)}")

class TriggerPatternRepository:
    """Repository for emotional trigger pattern management"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        metrics_collector: MetricsCollector
    ):
        self.db = db_session
        self.metrics = metrics_collector
    
    async def create_or_update_pattern(
        self,
        user_id: str,
        pattern_data: Dict[str, Any]
    ) -> EmotionalTriggerPattern:
        """Create or update a trigger pattern"""
        try:
            with self.metrics.timer('repository.trigger_pattern.create_or_update'):
                
                # Check if pattern exists
                pattern_name = pattern_data['pattern_name']
                
                query = select(EmotionalTriggerPattern).where(
                    and_(
                        EmotionalTriggerPattern.user_id == user_id,
                        EmotionalTriggerPattern.pattern_name == pattern_name
                    )
                )
                
                result = await self.db.execute(query)
                pattern = result.scalar_one_or_none()
                
                if pattern:
                    # Update existing pattern
                    for key, value in pattern_data.items():
                        setattr(pattern, key, value)
                    pattern.frequency += 1
                    pattern.last_triggered = utc_now()
                else:
                    # Create new pattern
                    pattern = EmotionalTriggerPattern(
                        user_id=user_id,
                        **pattern_data
                    )
                    self.db.add(pattern)
                
                await self.db.commit()
                
                self.metrics.increment('trigger_patterns.updated')
                
                return pattern
                
        except Exception as e:
            await self.db.rollback()
            self.metrics.increment('trigger_patterns.error')
            raise RepositoryException(f"Failed to create/update trigger pattern: {str(e)}")
    
    async def get_active_patterns(
        self,
        user_id: str,
        min_confidence: float = 0.5
    ) -> List[EmotionalTriggerPattern]:
        """Get active trigger patterns for a user"""
        try:
            with self.metrics.timer('repository.trigger_pattern.get_active'):
                
                query = select(EmotionalTriggerPattern).where(
                    and_(
                        EmotionalTriggerPattern.user_id == user_id,
                        EmotionalTriggerPattern.is_active == True,
                        EmotionalTriggerPattern.confidence >= min_confidence
                    )
                ).order_by(
                    EmotionalTriggerPattern.frequency.desc(),
                    EmotionalTriggerPattern.confidence.desc()
                )
                
                result = await self.db.execute(query)
                patterns = result.scalars().all()
                
                return patterns
                
        except Exception as e:
            self.metrics.increment('trigger_patterns.get_error')
            raise RepositoryException(f"Failed to get active patterns: {str(e)}")
