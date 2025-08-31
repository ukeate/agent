"""
Create emotional memory management system tables
Migration: 004
Date: 2024-01-15
Author: AI Agent System Team
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# revision identifiers
revision = '004_emotional_memory'
down_revision = '003_distributed_security_tables'
branch_labels = None
depends_on = None

def upgrade():
    """Create emotional memory management system tables with proper indexing and constraints"""
    
    # 1. Create emotional_memories table - Core memory storage
    op.create_table(
        'emotional_memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('session_id', sa.String(255), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        
        # Emotional content
        sa.Column('emotion_type', sa.String(50), nullable=False, index=True),
        sa.Column('intensity', sa.Float(), nullable=False),
        sa.Column('valence', sa.Float(), nullable=False),  # -1 to 1 (negative to positive)
        sa.Column('arousal', sa.Float(), nullable=False),  # 0 to 1 (calm to excited)
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_embedding', postgresql.ARRAY(sa.Float), nullable=True),  # For semantic search
        
        # Storage management
        sa.Column('storage_layer', sa.Enum('hot', 'warm', 'cold', name='storage_layer_type'), 
                  nullable=False, server_default='hot'),
        sa.Column('access_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_accessed', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('importance_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('decay_rate', sa.Float(), nullable=False, server_default='0.1'),
        
        # Privacy and security
        sa.Column('privacy_level', sa.Enum('public', 'protected', 'private', name='privacy_level_type'), 
                  nullable=False, server_default='protected'),
        sa.Column('encryption_key_id', sa.String(255), nullable=True),
        sa.Column('is_encrypted', sa.Boolean(), nullable=False, server_default='false'),
        
        # Metadata
        sa.Column('tags', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('context_data', postgresql.JSONB(), nullable=True),
        sa.Column('source', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        
        # Audit fields
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        
        # Constraints
        sa.CheckConstraint('intensity >= 0 AND intensity <= 1', name='check_intensity_range'),
        sa.CheckConstraint('valence >= -1 AND valence <= 1', name='check_valence_range'),
        sa.CheckConstraint('arousal >= 0 AND arousal <= 1', name='check_arousal_range'),
        sa.CheckConstraint('importance_score >= 0 AND importance_score <= 1', name='check_importance_range'),
        sa.CheckConstraint('decay_rate >= 0 AND decay_rate <= 1', name='check_decay_range')
    )
    
    # Create indexes for performance
    op.create_index('idx_emotional_memories_user_timestamp', 'emotional_memories', 
                    ['user_id', 'timestamp'], postgresql_using='btree')
    op.create_index('idx_emotional_memories_storage_layer', 'emotional_memories', 
                    ['storage_layer', 'last_accessed'], postgresql_using='btree')
    op.create_index('idx_emotional_memories_emotion_intensity', 'emotional_memories', 
                    ['emotion_type', 'intensity'], postgresql_using='btree')
    op.create_index('idx_emotional_memories_importance', 'emotional_memories', 
                    ['importance_score', 'access_count'], postgresql_using='btree')
    op.create_index('idx_emotional_memories_embedding', 'emotional_memories',
                    ['content_embedding'], postgresql_using='ivfflat', postgresql_ops={'content_embedding': 'vector_l2_ops'})
    
    # 2. Create emotional_events table - Event detection and tracking
    op.create_table(
        'emotional_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        
        # Event details
        sa.Column('event_type', sa.String(100), nullable=False, index=True),
        sa.Column('trigger_source', sa.String(255), nullable=False),
        sa.Column('trigger_context', postgresql.JSONB(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        
        # Causal relationships
        sa.Column('parent_event_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('emotional_events.id', ondelete='SET NULL'), nullable=True),
        sa.Column('causal_chain', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('causal_strength', sa.Float(), nullable=False, server_default='0.5'),
        
        # Impact metrics
        sa.Column('impact_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('affected_emotions', postgresql.ARRAY(sa.String), nullable=True),
        
        # Processing status
        sa.Column('is_processed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('processing_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_result', postgresql.JSONB(), nullable=True),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create indexes for event queries
    op.create_index('idx_emotional_events_user_timestamp', 'emotional_events', 
                    ['user_id', 'timestamp'], postgresql_using='btree')
    op.create_index('idx_emotional_events_type_impact', 'emotional_events', 
                    ['event_type', 'impact_score'], postgresql_using='btree')
    op.create_index('idx_emotional_events_causal', 'emotional_events', 
                    ['parent_event_id'], postgresql_using='btree')
    
    # 3. Create user_emotional_preferences table - Preference learning
    op.create_table(
        'user_emotional_preferences',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.String(255), nullable=False, unique=True, index=True),
        
        # Preference model
        sa.Column('dominant_emotions', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('emotion_weights', postgresql.JSONB(), nullable=True),  # {emotion: weight}
        sa.Column('preferred_responses', postgresql.JSONB(), nullable=True),
        sa.Column('avoided_triggers', postgresql.ARRAY(sa.String), nullable=True),
        
        # Learning parameters
        sa.Column('learning_rate', sa.Float(), nullable=False, server_default='0.01'),
        sa.Column('adaptation_speed', sa.Float(), nullable=False, server_default='0.1'),
        sa.Column('stability_threshold', sa.Float(), nullable=False, server_default='0.8'),
        
        # Model metrics
        sa.Column('model_accuracy', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('training_samples', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_training', sa.DateTime(timezone=True), nullable=True),
        
        # Response patterns
        sa.Column('response_patterns', postgresql.JSONB(), nullable=True),
        sa.Column('interaction_style', sa.String(50), nullable=True),
        sa.Column('communication_preferences', postgresql.JSONB(), nullable=True),
        
        # Privacy settings
        sa.Column('share_preferences', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('anonymous_learning', sa.Boolean(), nullable=False, server_default='true'),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1')
    )
    
    # 4. Create emotional_trigger_patterns table - Pattern recognition
    op.create_table(
        'emotional_trigger_patterns',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        
        # Pattern identification
        sa.Column('pattern_name', sa.String(255), nullable=False),
        sa.Column('pattern_type', sa.String(100), nullable=False, index=True),
        sa.Column('trigger_conditions', postgresql.JSONB(), nullable=False),
        sa.Column('trigger_keywords', postgresql.ARRAY(sa.String), nullable=True),
        
        # Pattern metrics
        sa.Column('frequency', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_intensity', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('reliability', sa.Float(), nullable=False, server_default='0.5'),
        
        # Temporal patterns
        sa.Column('time_patterns', postgresql.JSONB(), nullable=True),  # {hour: frequency, day: frequency}
        sa.Column('seasonal_variation', postgresql.JSONB(), nullable=True),
        sa.Column('circadian_alignment', sa.Float(), nullable=True),
        
        # Associated emotions
        sa.Column('triggered_emotions', postgresql.ARRAY(sa.String), nullable=False),
        sa.Column('emotion_probabilities', postgresql.JSONB(), nullable=True),
        
        # Response strategy
        sa.Column('recommended_responses', postgresql.JSONB(), nullable=True),
        sa.Column('avoidance_strategies', postgresql.JSONB(), nullable=True),
        
        # Status
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_triggered', sa.DateTime(timezone=True), nullable=True),
        sa.Column('next_expected', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create indexes for pattern queries
    op.create_index('idx_trigger_patterns_user_type', 'emotional_trigger_patterns', 
                    ['user_id', 'pattern_type'], postgresql_using='btree')
    op.create_index('idx_trigger_patterns_frequency', 'emotional_trigger_patterns', 
                    ['frequency', 'confidence'], postgresql_using='btree')
    
    # 5. Create memory_access_logs table - Access tracking and analytics
    op.create_table(
        'memory_access_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        
        # Access details
        sa.Column('access_type', sa.String(50), nullable=False),  # read, write, update, delete
        sa.Column('access_timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('access_duration_ms', sa.Integer(), nullable=True),
        sa.Column('access_source', sa.String(100), nullable=True),
        
        # Context
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('request_context', postgresql.JSONB(), nullable=True),
        sa.Column('response_context', postgresql.JSONB(), nullable=True),
        
        # Performance metrics
        sa.Column('retrieval_latency_ms', sa.Integer(), nullable=True),
        sa.Column('processing_latency_ms', sa.Integer(), nullable=True),
        sa.Column('storage_layer_accessed', sa.String(20), nullable=True),
        
        # Security
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(255), nullable=True),
        sa.Column('authentication_method', sa.String(50), nullable=True)
    )
    
    # Create indexes for access log queries
    op.create_index('idx_access_logs_memory', 'memory_access_logs', 
                    ['memory_id', 'access_timestamp'], postgresql_using='btree')
    op.create_index('idx_access_logs_user', 'memory_access_logs', 
                    ['user_id', 'access_timestamp'], postgresql_using='btree')
    
    # 6. Create emotional_memory_cache table - Cache management
    op.create_table(
        'emotional_memory_cache',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('cache_key', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('emotional_memories.id', ondelete='CASCADE'), nullable=False),
        
        # Cache data
        sa.Column('cached_data', postgresql.JSONB(), nullable=False),
        sa.Column('cache_type', sa.String(50), nullable=False),  # query, embedding, aggregate
        sa.Column('compression_type', sa.String(20), nullable=True),
        
        # TTL management
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('hit_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_hit', sa.DateTime(timezone=True), nullable=True),
        
        # Performance
        sa.Column('size_bytes', sa.Integer(), nullable=True),
        sa.Column('compute_time_ms', sa.Integer(), nullable=True)
    )
    
    # Create indexes for cache
    op.create_index('idx_cache_key_expires', 'emotional_memory_cache', 
                    ['cache_key', 'expires_at'], postgresql_using='btree')
    op.create_index('idx_cache_memory', 'emotional_memory_cache', 
                    ['memory_id'], postgresql_using='btree')
    
    # Create update trigger for updated_at columns
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Apply trigger to all tables with updated_at
    for table in ['emotional_memories', 'emotional_events', 'user_emotional_preferences', 
                  'emotional_trigger_patterns']:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at 
            BEFORE UPDATE ON {table} 
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();
        """)

def downgrade():
    """Drop all emotional memory management tables"""
    
    # Drop triggers first
    for table in ['emotional_memories', 'emotional_events', 'user_emotional_preferences', 
                  'emotional_trigger_patterns']:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
    
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('emotional_memory_cache')
    op.drop_table('memory_access_logs')
    op.drop_table('emotional_trigger_patterns')
    op.drop_table('user_emotional_preferences')
    op.drop_table('emotional_events')
    op.drop_table('emotional_memories')
    
    # Drop custom types
    op.execute("DROP TYPE IF EXISTS storage_layer_type")
    op.execute("DROP TYPE IF EXISTS privacy_level_type")