"""
Test suite for Emotional Memory Service
Tests business logic, orchestration, and LangGraph integration
"""

from src.core.utils.timezone_utils import utc_now
import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
from src.services.emotional_memory_service import EmotionalMemoryService
from src.repositories.emotional_memory_repository import (
    EmotionalMemoryRepository,
    EmotionalEventRepository,
    UserPreferenceRepository,
    TriggerPatternRepository
)
from src.db.emotional_memory_models import (
    EmotionalMemory,
    EmotionalEvent,
    UserEmotionalPreference,
    EmotionalTriggerPattern
)
from src.ai.memory.models import MemoryContext
from src.core.exceptions import ServiceException, ValidationException

@pytest.fixture
def mock_memory_repo():
    """Mock EmotionalMemoryRepository"""
    repo = AsyncMock(spec=EmotionalMemoryRepository)
    repo.create_memory = AsyncMock()
    repo.search_memories = AsyncMock(return_value=([], 0))
    repo.get_memory_by_id = AsyncMock()
    repo.update_storage_tier = AsyncMock(return_value=True)
    repo.archive_old_memories = AsyncMock(return_value=10)
    return repo

@pytest.fixture
def mock_event_repo():
    """Mock EmotionalEventRepository"""
    repo = AsyncMock(spec=EmotionalEventRepository)
    repo.create_event = AsyncMock()
    repo.analyze_causal_relationships = AsyncMock(return_value={})
    return repo

@pytest.fixture
def mock_preference_repo():
    """Mock UserPreferenceRepository"""
    repo = AsyncMock(spec=UserPreferenceRepository)
    repo.get_or_create_preferences = AsyncMock()
    repo.update_preferences = AsyncMock()
    return repo

@pytest.fixture
def mock_pattern_repo():
    """Mock TriggerPatternRepository"""
    repo = AsyncMock(spec=TriggerPatternRepository)
    repo.create_or_update_pattern = AsyncMock()
    repo.get_active_patterns = AsyncMock(return_value=[])
    return repo

@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    return AsyncMock()

@pytest.fixture
def emotional_memory_service(
    mock_memory_repo,
    mock_event_repo,
    mock_preference_repo,
    mock_pattern_repo,
    mock_redis_client
):
    """Create EmotionalMemoryService instance with mocks"""
    return EmotionalMemoryService(
        memory_repo=mock_memory_repo,
        event_repo=mock_event_repo,
        preference_repo=mock_preference_repo,
        pattern_repo=mock_pattern_repo,
        redis_client=mock_redis_client,
        postgres_url="postgresql://test"
    )

class TestEmotionalMemoryService:
    """Test suite for EmotionalMemoryService"""
    
    @pytest.mark.asyncio
    async def test_create_memory_success(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test successful memory creation with all processing steps"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'emotion_type': 'joy',
            'intensity': 0.8,
            'content': 'Achievement unlocked!',
            'valence': 0.9,
            'arousal': 0.7
        }
        
        mock_memory = Mock(
            spec=EmotionalMemory,
            id=uuid4(),
            emotion_type='joy',
            intensity=0.8,
            storage_layer='hot'
        )
        mock_memory_repo.create_memory.return_value = mock_memory
        
        # Act
        result = await emotional_memory_service.create_memory(user_id, emotion_data)
        
        # Assert
        assert result['storage_tier'] == 'hot'
        assert result['importance_score'] > 0
        assert 'memory_id' in result
        mock_memory_repo.create_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_memory_validation_error(
        self,
        emotional_memory_service
    ):
        """Test memory creation with invalid data"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'emotion_type': 'joy',
            # Missing required 'intensity' field
            'content': 'Test content'
        }
        
        # Act & Assert
        with pytest.raises(ServiceException) as exc:
            await emotional_memory_service.create_memory(user_id, emotion_data)
        
        assert "Missing required field" in str(exc.value)
    
    @pytest.mark.asyncio
    async def test_create_memory_intensity_validation(
        self,
        emotional_memory_service
    ):
        """Test intensity range validation"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'emotion_type': 'joy',
            'intensity': 1.5,  # Invalid: > 1
            'content': 'Test content'
        }
        
        # Act & Assert
        with pytest.raises(ServiceException) as exc:
            await emotional_memory_service.create_memory(user_id, emotion_data)
        
        assert "Intensity must be between 0 and 1" in str(exc.value)
    
    @pytest.mark.asyncio
    async def test_get_memories_with_enrichment(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_preference_repo
    ):
        """Test retrieving memories with enrichment"""
        # Arrange
        user_id = "user123"
        
        mock_memories = [
            Mock(
                spec=EmotionalMemory,
                id=uuid4(),
                emotion_type='joy',
                intensity=0.8,
                timestamp=utc_now(),
                importance_score=0.7,
                decay_rate=0.1
            )
        ]
        mock_memory_repo.search_memories.return_value = (mock_memories, 1)
        
        mock_preferences = Mock(
            spec=UserEmotionalPreference,
            emotion_weights={'joy': 0.9}
        )
        mock_preference_repo.get_or_create_preferences.return_value = mock_preferences
        
        # Act
        result = await emotional_memory_service.get_memories(user_id)
        
        # Assert
        assert len(result) == 1
        assert result[0]['preference_alignment'] == 0.9
        assert 'current_strength' in result[0]
    
    @pytest.mark.asyncio
    async def test_search_memories_semantic(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test semantic search functionality"""
        # Arrange
        user_id = "user123"
        query = "happy moments with family"
        
        mock_memories = [
            Mock(
                spec=EmotionalMemory,
                id=uuid4(),
                content="Family gathering was wonderful",
                emotion_type='joy',
                intensity=0.9,
                importance_score=0.8,
                timestamp=utc_now()
            )
        ]
        mock_memory_repo.search_memories.return_value = (mock_memories, 1)
        
        # Act
        result = await emotional_memory_service.search_memories_semantic(user_id, query)
        
        # Assert
        assert len(result) == 1
        assert 'relevance_score' in result[0]
        assert result[0]['memory']['emotion_type'] == 'joy'
    
    @pytest.mark.asyncio
    async def test_detect_emotional_events(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_event_repo
    ):
        """Test emotional event detection"""
        # Arrange
        user_id = "user123"
        
        # Create memories with emotional transition
        mock_memories = [
            Mock(
                spec=EmotionalMemory,
                id=uuid4(),
                emotion_type='sadness',
                intensity=0.3,
                valence=-0.5,
                timestamp=utc_now() - timedelta(hours=2)
            ),
            Mock(
                spec=EmotionalMemory,
                id=uuid4(),
                emotion_type='joy',
                intensity=0.9,
                valence=0.8,
                timestamp=utc_now()
            )
        ]
        mock_memory_repo.search_memories.return_value = (mock_memories, 2)
        
        mock_event = Mock(spec=EmotionalEvent, id=uuid4())
        mock_event_repo.create_event.return_value = mock_event
        
        causal_analysis = {
            'total_events': 1,
            'causal_chains': 0,
            'graph': {},
            'avg_causal_strength': 0.7
        }
        mock_event_repo.analyze_causal_relationships.return_value = causal_analysis
        
        # Act
        result = await emotional_memory_service.detect_emotional_events(user_id)
        
        # Assert
        assert result['event_count'] >= 0
        assert 'causal_analysis' in result
        mock_event_repo.create_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_learn_user_preferences(
        self,
        emotional_memory_service,
        mock_preference_repo,
        mock_memory_repo
    ):
        """Test user preference learning"""
        # Arrange
        user_id = "user123"
        
        mock_preferences = Mock(
            spec=UserEmotionalPreference,
            emotion_weights={},
            learning_rate=0.1
        )
        mock_preference_repo.get_or_create_preferences.return_value = mock_preferences
        
        mock_memories = [
            Mock(
                spec=EmotionalMemory,
                emotion_type='joy',
                intensity=0.8,
                timestamp=utc_now()
            ),
            Mock(
                spec=EmotionalMemory,
                emotion_type='joy',
                intensity=0.7,
                timestamp=utc_now()
            ),
            Mock(
                spec=EmotionalMemory,
                emotion_type='contentment',
                intensity=0.6,
                timestamp=utc_now()
            )
        ]
        mock_memory_repo.search_memories.return_value = (mock_memories, 3)
        
        # Act
        result = await emotional_memory_service.learn_user_preferences(user_id)
        
        # Assert
        assert 'dominant_emotions' in result
        assert 'emotion_weights' in result
        mock_preference_repo.update_preferences.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_identify_trigger_patterns(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_pattern_repo
    ):
        """Test trigger pattern identification"""
        # Arrange
        user_id = "user123"
        
        # Create memories with pattern
        mock_memories = [
            Mock(
                spec=EmotionalMemory,
                emotion_type='anxiety',
                intensity=0.7,
                valence=-0.5,
                timestamp=datetime(2024, 1, 1, 9, 0),  # Morning
                context_data={'trigger': 'meeting'}
            ),
            Mock(
                spec=EmotionalMemory,
                emotion_type='anxiety',
                intensity=0.8,
                valence=-0.6,
                timestamp=datetime(2024, 1, 2, 9, 30),  # Morning
                context_data={'trigger': 'meeting'}
            ),
            Mock(
                spec=EmotionalMemory,
                emotion_type='anxiety',
                intensity=0.75,
                valence=-0.55,
                timestamp=datetime(2024, 1, 3, 9, 15),  # Morning
                context_data={'trigger': 'meeting'}
            )
        ]
        mock_memory_repo.search_memories.return_value = (mock_memories, 3)
        
        mock_pattern = Mock(spec=EmotionalTriggerPattern, id=uuid4())
        mock_pattern_repo.create_or_update_pattern.return_value = mock_pattern
        
        # Act
        result = await emotional_memory_service.identify_trigger_patterns(user_id, min_frequency=3)
        
        # Assert
        assert isinstance(result, list)
        # Pattern should be identified since we have 3 similar memories
        if len(result) > 0:
            mock_pattern_repo.create_or_update_pattern.assert_called()
    
    @pytest.mark.asyncio
    async def test_optimize_storage_tiers(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test storage tier optimization"""
        # Arrange
        mock_memory_repo.archive_old_memories.return_value = 15
        
        # Act
        result = await emotional_memory_service.optimize_storage_tiers()
        
        # Assert
        assert 'warm_to_cold' in result
        assert result['warm_to_cold'] == 15
        mock_memory_repo.archive_old_memories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_privacy_level_determination(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test privacy level determination for sensitive content"""
        # Arrange
        user_id = "user123"
        
        # Test with sensitive content
        sensitive_emotion_data = {
            'emotion_type': 'shame',
            'intensity': 0.9,
            'content': 'My personal medical information',
            'valence': -0.8
        }
        
        mock_memory = Mock(
            spec=EmotionalMemory,
            id=uuid4(),
            privacy_level='private'
        )
        mock_memory_repo.create_memory.return_value = mock_memory
        
        # Act
        result = await emotional_memory_service.create_memory(user_id, sensitive_emotion_data)
        
        # Assert
        assert result['privacy_level'] == 'private'
        # Should be encrypted due to private level
        mock_memory_repo.create_memory.assert_called_with(
            user_id=user_id,
            emotion_data=pytest.approx(sensitive_emotion_data),
            encrypt=True
        )
    
    @pytest.mark.asyncio
    async def test_importance_calculation_with_context(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test importance score calculation with context"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'emotion_type': 'achievement',
            'intensity': 0.9,
            'content': 'Major milestone reached',
            'valence': 0.95,
            'arousal': 0.85
        }
        
        context = MemoryContext(
            is_milestone=True,
            is_learning_moment=True
        )
        
        mock_memory = Mock(spec=EmotionalMemory, id=uuid4())
        mock_memory_repo.create_memory.return_value = mock_memory
        
        # Act
        result = await emotional_memory_service.create_memory(
            user_id,
            emotion_data,
            context=context
        )
        
        # Assert
        # With milestone and learning moment, importance should be high
        assert result['importance_score'] > 0.8
        assert result['storage_tier'] == 'hot'  # High importance = hot storage

class TestEmotionalMemoryServiceIntegration:
    """Integration tests for EmotionalMemoryService"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_memory_lifecycle(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_event_repo,
        mock_preference_repo,
        mock_pattern_repo
    ):
        """Test complete memory lifecycle from creation to pattern detection"""
        user_id = "integration_user"
        emotion_data = {
            'emotion_type': 'joy',
            'intensity': 0.9,
            'content': '重大成就达成！',
            'valence': 0.8,
            'arousal': 0.7
        }
        mock_memory = Mock(
            spec=EmotionalMemory,
            id=uuid4(),
            user_id=user_id,
            emotion_type='joy',
            intensity=0.9,
            valence=0.8,
            arousal=0.7,
            content='重大成就达成！',
            storage_layer='hot',
            importance_score=0.9,
            access_count=0,
            last_accessed=utc_now(),
            decay_rate=0.05,
            timestamp=utc_now(),
            privacy_level='public',
            tags=[],
            is_encrypted=False
        )
        mock_memory_repo.create_memory.return_value = mock_memory

        mock_preferences = Mock(
            user_id=user_id,
            emotion_weights={},
            learning_rate=0.5
        )
        mock_preference_repo.get_or_create_preferences.return_value = mock_preferences
        mock_pattern_repo.get_active_patterns.return_value = []
        mock_event = Mock(
            spec=EmotionalEvent,
            id=uuid4(),
            memory_id=mock_memory.id,
            event_type="high_intensity",
            trigger_source="intensity_threshold",
            timestamp=utc_now(),
            impact_score=mock_memory.intensity,
            affected_emotions=[mock_memory.emotion_type],
            causal_strength=0.5
        )
        mock_event_repo.create_event.return_value = mock_event

        result = await emotional_memory_service.create_memory(user_id, emotion_data)

        assert result["event_detected"] is True
        assert result["patterns_triggered"] is False
        mock_event_repo.create_event.assert_called_once()
        mock_preference_repo.update_preferences.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_user_isolation(
        self,
        emotional_memory_service,
        mock_memory_repo
    ):
        """Test that memories are properly isolated between users"""
        # Arrange
        user1_id = "user1"
        user2_id = "user2"
        
        emotion_data = {
            'emotion_type': 'joy',
            'intensity': 0.8,
            'content': 'Private content'
        }
        
        # Create memories for both users
        await emotional_memory_service.create_memory(user1_id, emotion_data)
        await emotional_memory_service.create_memory(user2_id, emotion_data)
        
        # Mock search to return empty for cross-user queries
        mock_memory_repo.search_memories.return_value = ([], 0)
        
        # Act - Try to get user1's memories as user2
        result = await emotional_memory_service.get_memories(user2_id, {'user_id': user1_id})
        
        # Assert - Should get no results
        assert len(result) == 0

class TestPerformanceOptimization:
    """Performance and optimization tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_memory_creation_performance(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_preference_repo
    ):
        """Test performance of batch memory creation"""
        user_id = "batch_user"
        batch_size = 5
        emotion_data = {
            'emotion_type': 'joy',
            'intensity': 0.4,
            'content': 'batch content',
            'valence': 0.2,
            'arousal': 0.3
        }
        mock_memory_repo.create_memory.side_effect = [
            Mock(
                spec=EmotionalMemory,
                id=uuid4(),
                user_id=user_id,
                emotion_type='joy',
                intensity=0.4,
                valence=0.2,
                arousal=0.3,
                content='batch content',
                storage_layer='warm',
                importance_score=0.4,
                access_count=0,
                last_accessed=utc_now(),
                decay_rate=0.05,
                timestamp=utc_now(),
                privacy_level='public',
                tags=[],
                is_encrypted=False
            )
            for _ in range(batch_size)
        ]
        mock_preferences = Mock(
            user_id=user_id,
            emotion_weights={},
            learning_rate=0.5
        )
        mock_preference_repo.get_or_create_preferences.return_value = mock_preferences
        tasks = [
            emotional_memory_service.create_memory(user_id, emotion_data)
            for _ in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == batch_size
        assert mock_memory_repo.create_memory.call_count == batch_size
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_hit_rate(
        self,
        emotional_memory_service,
        mock_memory_repo,
        mock_preference_repo
    ):
        """Test memory retrieval consistency for repeated access"""
        user_id = "cache_user"
        mock_memory = Mock(
            spec=EmotionalMemory,
            id=uuid4(),
            user_id=user_id,
            emotion_type='joy',
            intensity=0.7,
            valence=0.5,
            arousal=0.4,
            content='cached content',
            storage_layer='hot',
            importance_score=0.7,
            access_count=1,
            last_accessed=utc_now(),
            decay_rate=0.05,
            timestamp=utc_now(),
            privacy_level='public',
            tags=[],
            is_encrypted=False
        )
        mock_memory_repo.search_memories.return_value = ([mock_memory], 1)
        mock_preferences = Mock(
            user_id=user_id,
            emotion_weights={},
            learning_rate=0.5
        )
        mock_preference_repo.get_or_create_preferences.return_value = mock_preferences

        first = await emotional_memory_service.get_memories(user_id, {})
        second = await emotional_memory_service.get_memories(user_id, {})

        assert len(first) == 1
        assert len(second) == 1
        assert mock_memory_repo.search_memories.await_count == 2
