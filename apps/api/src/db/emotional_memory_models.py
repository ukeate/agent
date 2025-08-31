"""
SQLAlchemy ORM Models for Emotional Memory Management System
Defines database schema for emotional memory storage and analysis
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey,
    Enum, Index, CheckConstraint, UniqueConstraint, JSON
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
import enum

Base = declarative_base()


class StorageLayerType(enum.Enum):
    """Storage tier enumeration"""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class PrivacyLevelType(enum.Enum):
    """Privacy level enumeration"""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


class EmotionalMemory(Base):
    """
    Core emotional memory storage model
    Implements multi-tier storage with encryption support
    """
    __tablename__ = 'emotional_memories'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # User and session tracking
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Emotional content
    emotion_type = Column(String(50), nullable=False, index=True)
    intensity = Column(Float, nullable=False)
    valence = Column(Float, nullable=False, default=0.0)  # -1 to 1
    arousal = Column(Float, nullable=False, default=0.5)  # 0 to 1
    content = Column(Text, nullable=False)
    content_embedding = Column(ARRAY(Float), nullable=True)  # Vector embedding for similarity search
    
    # Storage management
    storage_layer = Column(Enum(StorageLayerType), nullable=False, default=StorageLayerType.HOT)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    importance_score = Column(Float, nullable=False, default=0.5)
    decay_rate = Column(Float, nullable=False, default=0.1)
    
    # Privacy and security
    privacy_level = Column(Enum(PrivacyLevelType), nullable=False, default=PrivacyLevelType.PROTECTED)
    encryption_key_id = Column(String(255), nullable=True)
    is_encrypted = Column(Boolean, nullable=False, default=False)
    
    # Metadata
    tags = Column(ARRAY(String), nullable=True)
    context_data = Column(JSONB, nullable=True)
    source = Column(String(100), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    events = relationship("EmotionalEvent", back_populates="memory", cascade="all, delete-orphan")
    access_logs = relationship("MemoryAccessLog", back_populates="memory", cascade="all, delete-orphan")
    cache_entries = relationship("EmotionalMemoryCache", back_populates="memory", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('intensity >= 0 AND intensity <= 1', name='check_intensity_range'),
        CheckConstraint('valence >= -1 AND valence <= 1', name='check_valence_range'),
        CheckConstraint('arousal >= 0 AND arousal <= 1', name='check_arousal_range'),
        CheckConstraint('importance_score >= 0 AND importance_score <= 1', name='check_importance_range'),
        CheckConstraint('decay_rate >= 0 AND decay_rate <= 1', name='check_decay_range'),
        Index('idx_emotional_memories_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_emotional_memories_storage_layer', 'storage_layer', 'last_accessed'),
        Index('idx_emotional_memories_emotion_intensity', 'emotion_type', 'intensity'),
        Index('idx_emotional_memories_importance', 'importance_score', 'access_count'),
    )
    
    @hybrid_property
    def is_hot(self) -> bool:
        """Check if memory is in hot storage"""
        return self.storage_layer == StorageLayerType.HOT
    
    @hybrid_property
    def is_encrypted_private(self) -> bool:
        """Check if memory is encrypted and private"""
        return self.is_encrypted and self.privacy_level == PrivacyLevelType.PRIVATE
    
    def __repr__(self):
        return f"<EmotionalMemory(id={self.id}, user={self.user_id}, emotion={self.emotion_type}, intensity={self.intensity})>"


class EmotionalEvent(Base):
    """
    Emotional event tracking model
    Records significant emotional events and their causal relationships
    """
    __tablename__ = 'emotional_events'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    memory_id = Column(UUID(as_uuid=True), ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    trigger_source = Column(String(255), nullable=False)
    trigger_context = Column(JSONB, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Causal relationships
    parent_event_id = Column(UUID(as_uuid=True), ForeignKey('emotional_events.id', ondelete='SET NULL'), nullable=True)
    causal_chain = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    causal_strength = Column(Float, nullable=False, default=0.5)
    
    # Impact metrics
    impact_score = Column(Float, nullable=False, default=0.5)
    duration_seconds = Column(Integer, nullable=True)
    affected_emotions = Column(ARRAY(String), nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, nullable=False, default=False)
    processing_timestamp = Column(DateTime(timezone=True), nullable=True)
    processing_result = Column(JSONB, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    memory = relationship("EmotionalMemory", back_populates="events")
    parent_event = relationship("EmotionalEvent", remote_side=[id], backref="child_events")
    
    # Indexes
    __table_args__ = (
        Index('idx_emotional_events_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_emotional_events_type_impact', 'event_type', 'impact_score'),
        Index('idx_emotional_events_causal', 'parent_event_id'),
    )
    
    def __repr__(self):
        return f"<EmotionalEvent(id={self.id}, type={self.event_type}, impact={self.impact_score})>"


class UserEmotionalPreference(Base):
    """
    User emotional preference model
    Stores learned preferences and response patterns
    """
    __tablename__ = 'user_emotional_preferences'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Preference model
    dominant_emotions = Column(ARRAY(String), nullable=True)
    emotion_weights = Column(JSONB, nullable=True)  # {emotion: weight}
    preferred_responses = Column(JSONB, nullable=True)
    avoided_triggers = Column(ARRAY(String), nullable=True)
    
    # Learning parameters
    learning_rate = Column(Float, nullable=False, default=0.01)
    adaptation_speed = Column(Float, nullable=False, default=0.1)
    stability_threshold = Column(Float, nullable=False, default=0.8)
    
    # Model metrics
    model_accuracy = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=False, default=0)
    last_training = Column(DateTime(timezone=True), nullable=True)
    
    # Response patterns
    response_patterns = Column(JSONB, nullable=True)
    interaction_style = Column(String(50), nullable=True)
    communication_preferences = Column(JSONB, nullable=True)
    
    # Privacy settings
    share_preferences = Column(Boolean, nullable=False, default=False)
    anonymous_learning = Column(Boolean, nullable=False, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(Integer, nullable=False, default=1)
    
    def __repr__(self):
        return f"<UserEmotionalPreference(user={self.user_id}, accuracy={self.model_accuracy})>"


class EmotionalTriggerPattern(Base):
    """
    Emotional trigger pattern model
    Identifies and tracks recurring emotional triggers
    """
    __tablename__ = 'emotional_trigger_patterns'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Pattern identification
    pattern_name = Column(String(255), nullable=False)
    pattern_type = Column(String(100), nullable=False, index=True)
    trigger_conditions = Column(JSONB, nullable=False)
    trigger_keywords = Column(ARRAY(String), nullable=True)
    
    # Pattern metrics
    frequency = Column(Integer, nullable=False, default=0)
    avg_intensity = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False, default=0.5)
    reliability = Column(Float, nullable=False, default=0.5)
    
    # Temporal patterns
    time_patterns = Column(JSONB, nullable=True)  # {hour: frequency, day: frequency}
    seasonal_variation = Column(JSONB, nullable=True)
    circadian_alignment = Column(Float, nullable=True)
    
    # Associated emotions
    triggered_emotions = Column(ARRAY(String), nullable=False)
    emotion_probabilities = Column(JSONB, nullable=True)
    
    # Response strategy
    recommended_responses = Column(JSONB, nullable=True)
    avoidance_strategies = Column(JSONB, nullable=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_triggered = Column(DateTime(timezone=True), nullable=True)
    next_expected = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_trigger_patterns_user_type', 'user_id', 'pattern_type'),
        Index('idx_trigger_patterns_frequency', 'frequency', 'confidence'),
    )
    
    def __repr__(self):
        return f"<EmotionalTriggerPattern(name={self.pattern_name}, frequency={self.frequency})>"


class MemoryAccessLog(Base):
    """
    Memory access log model
    Tracks all access patterns for analytics and optimization
    """
    __tablename__ = 'memory_access_logs'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys
    memory_id = Column(UUID(as_uuid=True), ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Access details
    access_type = Column(String(50), nullable=False)  # read, write, update, delete
    access_timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    access_duration_ms = Column(Integer, nullable=True)
    access_source = Column(String(100), nullable=True)
    
    # Context
    session_id = Column(String(255), nullable=True)
    request_context = Column(JSONB, nullable=True)
    response_context = Column(JSONB, nullable=True)
    
    # Performance metrics
    retrieval_latency_ms = Column(Integer, nullable=True)
    processing_latency_ms = Column(Integer, nullable=True)
    storage_layer_accessed = Column(String(20), nullable=True)
    
    # Security
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    authentication_method = Column(String(50), nullable=True)
    
    # Relationships
    memory = relationship("EmotionalMemory", back_populates="access_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_access_logs_memory', 'memory_id', 'access_timestamp'),
        Index('idx_access_logs_user', 'user_id', 'access_timestamp'),
    )
    
    def __repr__(self):
        return f"<MemoryAccessLog(memory={self.memory_id}, type={self.access_type})>"


class EmotionalMemoryCache(Base):
    """
    Memory cache model
    Manages cached data for performance optimization
    """
    __tablename__ = 'emotional_memory_cache'
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    cache_key = Column(String(255), nullable=False, unique=True, index=True)
    
    # Foreign key
    memory_id = Column(UUID(as_uuid=True), ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False)
    
    # Cache data
    cached_data = Column(JSONB, nullable=False)
    cache_type = Column(String(50), nullable=False)  # query, embedding, aggregate
    compression_type = Column(String(20), nullable=True)
    
    # TTL management
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    hit_count = Column(Integer, nullable=False, default=0)
    last_hit = Column(DateTime(timezone=True), nullable=True)
    
    # Performance
    size_bytes = Column(Integer, nullable=True)
    compute_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    memory = relationship("EmotionalMemory", back_populates="cache_entries")
    
    # Indexes
    __table_args__ = (
        Index('idx_cache_key_expires', 'cache_key', 'expires_at'),
        Index('idx_cache_memory', 'memory_id'),
    )
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def __repr__(self):
        return f"<EmotionalMemoryCache(key={self.cache_key}, expires={self.expires_at})>"


# Create additional indexes for vector similarity search
# Note: These would be created using PostgreSQL's pgvector extension
"""
CREATE INDEX idx_emotional_memories_embedding_vector 
ON emotional_memories 
USING ivfflat (content_embedding vector_l2_ops)
WITH (lists = 100);
"""