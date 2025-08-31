"""
创建情感状态建模系统数据表

Revision ID: 003_emotion_modeling
Revises: 002_create_evaluation_tables
Create Date: 2025-01-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


def upgrade():
    """创建情感建模相关表"""
    
    # 创建情感状态表
    op.create_table(
        'emotion_states',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('emotion', sa.String(50), nullable=False),
        sa.Column('intensity', sa.Float(), nullable=False),
        sa.Column('valence', sa.Float(), nullable=False),
        sa.Column('arousal', sa.Float(), nullable=False),
        sa.Column('dominance', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('duration', sa.Interval(), nullable=True),
        sa.Column('triggers', sa.JSON(), nullable=True),
        sa.Column('context', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(50), nullable=False, default='manual'),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建索引
    op.create_index('idx_emotion_states_user_time', 'emotion_states', ['user_id', 'timestamp'])
    op.create_index('idx_emotion_states_emotion', 'emotion_states', ['emotion'])
    op.create_index('idx_emotion_states_timestamp', 'emotion_states', ['timestamp'])
    op.create_index('idx_emotion_states_session', 'emotion_states', ['session_id'])
    
    # 创建个性画像表
    op.create_table(
        'personality_profiles',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column('emotional_traits', sa.JSON(), nullable=False),
        sa.Column('baseline_emotions', sa.JSON(), nullable=False),
        sa.Column('emotion_volatility', sa.Float(), nullable=False),
        sa.Column('recovery_rate', sa.Float(), nullable=False),
        sa.Column('dominant_emotions', sa.JSON(), nullable=False),
        sa.Column('trigger_patterns', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False, default=0),
        sa.Column('confidence_score', sa.Float(), nullable=False, default=0.0),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建情感转换表
    op.create_table(
        'emotion_transitions',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_emotion', sa.String(50), nullable=False),
        sa.Column('to_emotion', sa.String(50), nullable=False),
        sa.Column('transition_probability', sa.Float(), nullable=False),
        sa.Column('occurrence_count', sa.Integer(), nullable=False),
        sa.Column('avg_duration', sa.Interval(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('context_factors', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建转换表索引
    op.create_index('idx_emotion_transitions_user', 'emotion_transitions', ['user_id'])
    op.create_index('idx_emotion_transitions_from_to', 'emotion_transitions', ['from_emotion', 'to_emotion'])
    
    # 创建情感聚类表 (用于存储聚类结果)
    op.create_table(
        'emotion_clusters',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('centroid_valence', sa.Float(), nullable=False),
        sa.Column('centroid_arousal', sa.Float(), nullable=False),
        sa.Column('centroid_dominance', sa.Float(), nullable=False),
        sa.Column('cluster_size', sa.Integer(), nullable=False),
        sa.Column('dominant_emotions', sa.JSON(), nullable=False),
        sa.Column('characteristic_features', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建聚类表索引
    op.create_index('idx_emotion_clusters_user', 'emotion_clusters', ['user_id'])
    
    # 创建情感预测历史表
    op.create_table(
        'emotion_predictions',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('current_emotion', sa.String(50), nullable=False),
        sa.Column('predicted_emotions', sa.JSON(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('time_horizon_seconds', sa.Integer(), nullable=False),
        sa.Column('prediction_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('factors', sa.JSON(), nullable=True),
        sa.Column('actual_emotion', sa.String(50), nullable=True),  # 用于验证预测准确性
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建预测表索引
    op.create_index('idx_emotion_predictions_user_time', 'emotion_predictions', ['user_id', 'prediction_time'])
    
    # 创建情感模式表 (存储识别出的情感模式)
    op.create_table(
        'emotion_patterns',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),  # daily, weekly, trigger_based等
        sa.Column('pattern_name', sa.String(100), nullable=False),
        sa.Column('pattern_data', sa.JSON(), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),  # 模式强度 [0,1]
        sa.Column('support', sa.Float(), nullable=False),   # 支持度 [0,1] 
        sa.Column('confidence', sa.Float(), nullable=False), # 置信度 [0,1]
        sa.Column('discovered_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=False),
        sa.Column('occurrence_count', sa.Integer(), nullable=False, default=1),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建模式表索引
    op.create_index('idx_emotion_patterns_user', 'emotion_patterns', ['user_id'])
    op.create_index('idx_emotion_patterns_type', 'emotion_patterns', ['pattern_type'])


def downgrade():
    """删除情感建模相关表"""
    op.drop_table('emotion_patterns')
    op.drop_table('emotion_predictions') 
    op.drop_table('emotion_clusters')
    op.drop_table('emotion_transitions')
    op.drop_table('personality_profiles')
    op.drop_table('emotion_states')