"""
Test suite for Emotional Memory Repository
Tests all database operations with proper mocking and assertions
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession
from redis import asyncio as aioredis

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
    EmotionalTriggerPattern,
    StorageLayerType,
    PrivacyLevelType
)
from src.core.security.encryption import EncryptionService
from src.core.monitoring.metrics_collector import MetricsCollector


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    client = AsyncMock(spec=aioredis.Redis)
    client.setex = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_encryption_service():
    """Mock encryption service"""
    service = AsyncMock(spec=EncryptionService)
    service.encrypt_data = AsyncMock(return_value=("encrypted_data", "key_id"))
    service.decrypt_data = AsyncMock(return_value="decrypted_data")
    return service


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector"""
    collector = Mock(spec=MetricsCollector)
    collector.timer = Mock()
    collector.timer.return_value.__enter__ = Mock()
    collector.timer.return_value.__exit__ = Mock()
    collector.increment = Mock()
    return collector


@pytest.fixture
def memory_repository(
    mock_db_session,
    mock_redis_client,
    mock_encryption_service,
    mock_metrics_collector
):
    """Create EmotionalMemoryRepository instance with mocks"""
    return EmotionalMemoryRepository(
        db_session=mock_db_session,
        redis_client=mock_redis_client,
        encryption_service=mock_encryption_service,
        metrics_collector=mock_metrics_collector
    )


class TestEmotionalMemoryRepository:
    """Test suite for EmotionalMemoryRepository"""
    
    @pytest.mark.asyncio
    async def test_create_memory_success(self, memory_repository, mock_db_session):
        """Test successful memory creation"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'session_id': 'session456',
            'emotion_type': 'joy',
            'intensity': 0.8,
            'valence': 0.9,
            'arousal': 0.7,
            'content': 'Test emotional content',
            'tags': ['test', 'positive'],
            'context': {'location': 'home'}
        }
        
        # Act
        result = await memory_repository.create_memory(user_id, emotion_data)
        
        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.flush.called
        assert mock_db_session.commit.called
        assert isinstance(result, EmotionalMemory)
        assert result.user_id == user_id
        assert result.emotion_type == 'joy'
        assert result.intensity == 0.8
        assert result.storage_layer == 'hot'
    
    @pytest.mark.asyncio
    async def test_create_memory_with_encryption(
        self,
        memory_repository,
        mock_db_session,
        mock_encryption_service
    ):
        """Test memory creation with encryption"""
        # Arrange
        user_id = "user123"
        emotion_data = {
            'session_id': 'session456',
            'emotion_type': 'shame',
            'intensity': 0.9,
            'content': 'Sensitive content',
        }
        
        # Act
        result = await memory_repository.create_memory(user_id, emotion_data, encrypt=True)
        
        # Assert
        mock_encryption_service.encrypt_data.assert_called_once()
        assert result.is_encrypted is True
        assert result.encryption_key_id == "key_id"
    
    @pytest.mark.asyncio
    async def test_get_memory_by_id_cache_hit(
        self,
        memory_repository,
        mock_redis_client
    ):
        """Test retrieving memory from cache"""
        # Arrange
        memory_id = uuid4()
        user_id = "user123"
        cached_data = {
            'id': str(memory_id),
            'user_id': user_id,
            'emotion_type': 'joy',
            'intensity': 0.8
        }
        mock_redis_client.get.return_value = str(cached_data).encode()
        
        # Act
        result = await memory_repository.get_memory_by_id(memory_id, user_id)
        
        # Assert
        mock_redis_client.get.assert_called_once()
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_search_memories_with_filters(
        self,
        memory_repository,
        mock_db_session
    ):
        """Test searching memories with various filters"""
        # Arrange
        user_id = "user123"
        filters = {
            'emotion_type': 'joy',
            'storage_layer': 'hot',
            'min_intensity': 0.5,
            'date_from': datetime.utcnow() - timedelta(days=7),
            'tags': ['positive']
        }
        
        mock_memories = [
            Mock(spec=EmotionalMemory, id=uuid4(), emotion_type='joy', intensity=0.8),
            Mock(spec=EmotionalMemory, id=uuid4(), emotion_type='joy', intensity=0.6)
        ]
        
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_memories
        mock_db_session.execute.return_value.scalar.return_value = 2
        
        # Act
        memories, total = await memory_repository.search_memories(user_id, filters)
        
        # Assert
        assert len(memories) == 2
        assert total == 2
        assert mock_db_session.execute.called
    
    @pytest.mark.asyncio
    async def test_update_storage_tier(
        self,
        memory_repository,
        mock_db_session,
        mock_redis_client
    ):
        """Test updating memory storage tier"""
        # Arrange
        memory_id = uuid4()
        new_tier = 'warm'
        
        mock_db_session.execute.return_value.rowcount = 1
        
        # Act
        result = await memory_repository.update_storage_tier(memory_id, new_tier)
        
        # Assert
        assert result is True
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called
        mock_redis_client.delete.assert_called()  # Should remove from cache
    
    @pytest.mark.asyncio
    async def test_archive_old_memories(
        self,
        memory_repository,
        mock_db_session
    ):
        """Test archiving old memories to cold storage"""
        # Arrange
        mock_db_session.execute.return_value.rowcount = 5
        
        # Act
        archived_count = await memory_repository.archive_old_memories(days_threshold=30)
        
        # Assert
        assert archived_count == 5
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_delete_memory_soft(
        self,
        memory_repository,
        mock_db_session
    ):
        """Test soft deleting a memory"""
        # Arrange
        memory_id = uuid4()
        user_id = "user123"
        
        mock_db_session.execute.return_value.rowcount = 1
        
        # Act
        result = await memory_repository.delete_memory(memory_id, user_id, hard_delete=False)
        
        # Assert
        assert result is True
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called


class TestEmotionalEventRepository:
    """Test suite for EmotionalEventRepository"""
    
    @pytest.fixture
    def event_repository(self, mock_db_session, mock_metrics_collector):
        """Create EmotionalEventRepository instance"""
        return EmotionalEventRepository(
            db_session=mock_db_session,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_create_event(
        self,
        event_repository,
        mock_db_session
    ):
        """Test creating an emotional event"""
        # Arrange
        memory_id = uuid4()
        user_id = "user123"
        event_data = {
            'event_type': 'intensity_spike',
            'trigger_source': 'user_input',
            'context': {'reason': 'achievement'},
            'impact_score': 0.9,
            'affected_emotions': ['joy', 'pride']
        }
        
        # Act
        result = await event_repository.create_event(memory_id, user_id, event_data)
        
        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.flush.called
        assert mock_db_session.commit.called
        assert isinstance(result, EmotionalEvent)
    
    @pytest.mark.asyncio
    async def test_analyze_causal_relationships(
        self,
        event_repository,
        mock_db_session
    ):
        """Test analyzing causal relationships between events"""
        # Arrange
        user_id = "user123"
        
        mock_events = [
            Mock(
                spec=EmotionalEvent,
                id=uuid4(),
                parent_event_id=None,
                event_type='trigger',
                causal_strength=0.8
            ),
            Mock(
                spec=EmotionalEvent,
                id=uuid4(),
                parent_event_id=uuid4(),
                event_type='response',
                causal_strength=0.6
            )
        ]
        
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_events
        
        # Act
        result = await event_repository.analyze_causal_relationships(user_id)
        
        # Assert
        assert 'total_events' in result
        assert 'causal_chains' in result
        assert 'graph' in result
        assert 'avg_causal_strength' in result
        assert result['total_events'] == 2


class TestUserPreferenceRepository:
    """Test suite for UserPreferenceRepository"""
    
    @pytest.fixture
    def preference_repository(self, mock_db_session, mock_metrics_collector):
        """Create UserPreferenceRepository instance"""
        return UserPreferenceRepository(
            db_session=mock_db_session,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_get_or_create_preferences_existing(
        self,
        preference_repository,
        mock_db_session
    ):
        """Test getting existing user preferences"""
        # Arrange
        user_id = "user123"
        mock_preference = Mock(spec=UserEmotionalPreference, user_id=user_id)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_preference
        
        # Act
        result = await preference_repository.get_or_create_preferences(user_id)
        
        # Assert
        assert result == mock_preference
        assert not mock_db_session.add.called
    
    @pytest.mark.asyncio
    async def test_get_or_create_preferences_new(
        self,
        preference_repository,
        mock_db_session
    ):
        """Test creating new user preferences"""
        # Arrange
        user_id = "user123"
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Act
        result = await preference_repository.get_or_create_preferences(user_id)
        
        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        assert isinstance(result, UserEmotionalPreference)
        assert result.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_update_preferences(
        self,
        preference_repository,
        mock_db_session
    ):
        """Test updating user preferences"""
        # Arrange
        user_id = "user123"
        updates = {
            'dominant_emotions': ['joy', 'contentment'],
            'model_accuracy': 0.85,
            'training_samples': 100
        }
        
        mock_preference = Mock(spec=UserEmotionalPreference)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_preference
        
        # Act
        result = await preference_repository.update_preferences(user_id, updates)
        
        # Assert
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called


class TestTriggerPatternRepository:
    """Test suite for TriggerPatternRepository"""
    
    @pytest.fixture
    def pattern_repository(self, mock_db_session, mock_metrics_collector):
        """Create TriggerPatternRepository instance"""
        return TriggerPatternRepository(
            db_session=mock_db_session,
            metrics_collector=mock_metrics_collector
        )
    
    @pytest.mark.asyncio
    async def test_create_pattern(
        self,
        pattern_repository,
        mock_db_session
    ):
        """Test creating a new trigger pattern"""
        # Arrange
        user_id = "user123"
        pattern_data = {
            'pattern_name': 'morning_anxiety',
            'pattern_type': 'temporal',
            'trigger_conditions': {'time': 'morning', 'day': 'weekday'},
            'triggered_emotions': ['anxiety', 'stress'],
            'frequency': 1,
            'confidence': 0.7
        }
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Act
        result = await pattern_repository.create_or_update_pattern(user_id, pattern_data)
        
        # Assert
        assert mock_db_session.add.called
        assert mock_db_session.commit.called
        assert isinstance(result, EmotionalTriggerPattern)
    
    @pytest.mark.asyncio
    async def test_get_active_patterns(
        self,
        pattern_repository,
        mock_db_session
    ):
        """Test getting active trigger patterns"""
        # Arrange
        user_id = "user123"
        min_confidence = 0.6
        
        mock_patterns = [
            Mock(spec=EmotionalTriggerPattern, confidence=0.8),
            Mock(spec=EmotionalTriggerPattern, confidence=0.7)
        ]
        
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_patterns
        
        # Act
        result = await pattern_repository.get_active_patterns(user_id, min_confidence)
        
        # Assert
        assert len(result) == 2
        assert mock_db_session.execute.called


# Integration tests
class TestRepositoryIntegration:
    """Integration tests for repository interactions"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_event_cascade(
        self,
        memory_repository,
        mock_db_session
    ):
        """Test cascading operations between memories and events"""
        # This would test the interaction between creating a memory
        # and automatically generating events
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_preference_learning_flow(
        self,
        memory_repository,
        preference_repository,
        mock_db_session
    ):
        """Test the flow of learning preferences from memories"""
        # This would test how memories update user preferences
        pass