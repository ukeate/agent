"""
创建事件追踪表

Revision ID: 003
Revises: 002
Create Date: 2024-01-15 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone

# revision identifiers
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None

def upgrade() -> None:
    """创建事件追踪相关表"""
    
    # 1. 创建事件流表 (主要的事件存储表)
    op.create_table(
        'event_streams',
        # 主键和基本信息
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('event_id', sa.String(128), nullable=False, index=True),
        
        # 实验相关
        sa.Column('experiment_id', sa.String(128), nullable=False, index=True),
        sa.Column('variant_id', sa.String(128), nullable=True, index=True),
        sa.Column('user_id', sa.String(128), nullable=False, index=True),
        sa.Column('session_id', sa.String(128), nullable=True, index=True),
        
        # 事件分类
        sa.Column('event_type', sa.String(32), nullable=False, index=True),
        sa.Column('event_name', sa.String(128), nullable=False, index=True),
        sa.Column('event_category', sa.String(64), nullable=True, index=True),
        
        # 时间信息
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('server_timestamp', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now(), index=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        
        # 事件属性 (使用JSONB以获得更好的性能)
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('user_properties', postgresql.JSONB, nullable=True),
        sa.Column('experiment_context', postgresql.JSONB, nullable=True),
        
        # 技术信息
        sa.Column('client_info', postgresql.JSONB, nullable=True),
        sa.Column('device_info', postgresql.JSONB, nullable=True),
        sa.Column('geo_info', postgresql.JSONB, nullable=True),
        
        # 数据质量
        sa.Column('status', sa.String(32), nullable=False, default='pending', index=True),
        sa.Column('data_quality', sa.String(16), nullable=False, default='high', index=True),
        sa.Column('validation_errors', postgresql.JSONB, nullable=True),
        
        # 处理信息
        sa.Column('processing_metadata', postgresql.JSONB, nullable=True),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('last_retry_at', sa.DateTime(timezone=True), nullable=True),
        
        # 数据分区键
        sa.Column('partition_key', sa.String(7), nullable=False, index=True),
        
        # 审计字段
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
    )
    
    # 为event_streams创建唯一约束
    op.create_unique_constraint('uq_event_id', 'event_streams', ['event_id'])
    
    # 创建复合索引
    op.create_index('idx_experiment_variant_time', 'event_streams', 
                   ['experiment_id', 'variant_id', 'event_timestamp'])
    op.create_index('idx_user_experiment_time', 'event_streams', 
                   ['user_id', 'experiment_id', 'event_timestamp'])
    op.create_index('idx_event_type_name_time', 'event_streams', 
                   ['event_type', 'event_name', 'event_timestamp'])
    op.create_index('idx_partition_time', 'event_streams', 
                   ['partition_key', 'event_timestamp'])
    op.create_index('idx_status_partition', 'event_streams', 
                   ['status', 'partition_key'])
    op.create_index('idx_quality_time', 'event_streams', 
                   ['data_quality', 'event_timestamp'])
    
    # 创建GIN索引用于JSONB字段搜索
    op.create_index('idx_properties_gin', 'event_streams', ['properties'], 
                   postgresql_using='gin')
    op.create_index('idx_user_properties_gin', 'event_streams', ['user_properties'], 
                   postgresql_using='gin')
    
    # 2. 创建事件聚合表
    op.create_table(
        'event_aggregations',
        sa.Column('id', sa.String(36), primary_key=True),
        
        # 聚合维度
        sa.Column('experiment_id', sa.String(128), nullable=False, index=True),
        sa.Column('variant_id', sa.String(128), nullable=True, index=True),
        sa.Column('event_type', sa.String(32), nullable=False, index=True),
        sa.Column('event_name', sa.String(128), nullable=False, index=True),
        
        # 时间维度
        sa.Column('aggregation_period', sa.String(16), nullable=False, index=True),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False, index=True),
        
        # 聚合指标
        sa.Column('event_count', sa.BigInteger, nullable=False, default=0),
        sa.Column('unique_users', sa.Integer, nullable=False, default=0),
        sa.Column('unique_sessions', sa.Integer, nullable=False, default=0),
        
        # 数值型指标统计
        sa.Column('total_value', sa.Float, nullable=True),
        sa.Column('avg_value', sa.Float, nullable=True),
        sa.Column('min_value', sa.Float, nullable=True),
        sa.Column('max_value', sa.Float, nullable=True),
        sa.Column('std_value', sa.Float, nullable=True),
        
        # 分位数统计
        sa.Column('p50_value', sa.Float, nullable=True),
        sa.Column('p90_value', sa.Float, nullable=True),
        sa.Column('p95_value', sa.Float, nullable=True),
        sa.Column('p99_value', sa.Float, nullable=True),
        
        # 自定义指标
        sa.Column('custom_metrics', postgresql.JSONB, nullable=True),
        
        # 元数据
        sa.Column('aggregated_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('source_events_count', sa.BigInteger, nullable=False),
        sa.Column('data_completeness', sa.Float, nullable=True),
    )
    
    # 创建聚合表索引
    op.create_index('idx_exp_variant_period_start', 'event_aggregations', 
                   ['experiment_id', 'variant_id', 'aggregation_period', 'period_start'])
    op.create_index('idx_event_type_period', 'event_aggregations', 
                   ['event_type', 'event_name', 'aggregation_period', 'period_start'])
    op.create_index('idx_period_start_end', 'event_aggregations', 
                   ['period_start', 'period_end'])
    
    # 唯一约束确保不重复聚合
    op.create_unique_constraint('uq_aggregation_unique', 'event_aggregations',
                               ['experiment_id', 'variant_id', 'event_type', 'event_name', 
                                'aggregation_period', 'period_start'])
    
    # 3. 创建事件指标表
    op.create_table(
        'event_metrics',
        sa.Column('id', sa.String(36), primary_key=True),
        
        # 实验信息
        sa.Column('experiment_id', sa.String(128), nullable=False, index=True),
        sa.Column('metric_name', sa.String(128), nullable=False, index=True),
        sa.Column('metric_type', sa.String(32), nullable=False, index=True),
        
        # 变体数据
        sa.Column('variant_metrics', postgresql.JSONB, nullable=False),
        
        # 统计信息
        sa.Column('calculation_method', sa.String(64), nullable=False),
        sa.Column('confidence_level', sa.Float, nullable=False, default=0.95),
        sa.Column('sample_size', postgresql.JSONB, nullable=False),
        
        # 统计检验结果
        sa.Column('statistical_test', sa.String(64), nullable=True),
        sa.Column('p_value', sa.Float, nullable=True),
        sa.Column('effect_size', sa.Float, nullable=True),
        sa.Column('confidence_interval', postgresql.JSONB, nullable=True),
        sa.Column('is_significant', sa.Boolean, nullable=True),
        sa.Column('statistical_power', sa.Float, nullable=True),
        
        # 时间范围
        sa.Column('calculation_start_time', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('calculation_end_time', sa.DateTime(timezone=True), nullable=False, index=True),
        
        # 元数据
        sa.Column('calculated_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('calculation_duration_ms', sa.Integer, nullable=True),
        sa.Column('data_freshness_score', sa.Float, nullable=True),
        
        # 质量控制
        sa.Column('quality_checks', postgresql.JSONB, nullable=True),
        sa.Column('warnings', postgresql.JSONB, nullable=True),
    )
    
    # 创建指标表索引
    op.create_index('idx_exp_metric_time', 'event_metrics', 
                   ['experiment_id', 'metric_name', 'calculation_end_time'])
    op.create_index('idx_metric_calculated_at', 'event_metrics', 
                   ['metric_name', 'calculated_at'])
    op.create_index('idx_exp_time_range', 'event_metrics', 
                   ['experiment_id', 'calculation_start_time', 'calculation_end_time'])
    
    # 4. 创建事件批次表
    op.create_table(
        'event_batches',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('batch_id', sa.String(128), nullable=False, unique=True, index=True),
        
        # 批次信息
        sa.Column('batch_type', sa.String(32), nullable=False, index=True),
        sa.Column('status', sa.String(32), nullable=False, default='pending', index=True),
        
        # 处理范围
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('experiment_ids', postgresql.JSONB, nullable=True),
        
        # 处理统计
        sa.Column('total_events', sa.BigInteger, nullable=False, default=0),
        sa.Column('processed_events', sa.BigInteger, nullable=False, default=0),
        sa.Column('failed_events', sa.BigInteger, nullable=False, default=0),
        sa.Column('duplicate_events', sa.BigInteger, nullable=False, default=0),
        
        # 执行信息
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer, nullable=True),
        sa.Column('worker_id', sa.String(128), nullable=True),
        
        # 错误信息
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_details', postgresql.JSONB, nullable=True),
        sa.Column('retry_count', sa.Integer, default=0),
        
        # 元数据
        sa.Column('configuration', postgresql.JSONB, nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB, nullable=True),
        
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
    )
    
    # 创建批次表索引
    op.create_index('idx_batch_type_status', 'event_batches', ['batch_type', 'status'])
    op.create_index('idx_start_end_time', 'event_batches', ['start_time', 'end_time'])
    op.create_index('idx_status_created', 'event_batches', ['status', 'created_at'])
    
    # 5. 创建事件Schema表
    op.create_table(
        'event_schemas',
        sa.Column('id', sa.String(36), primary_key=True),
        
        # Schema信息
        sa.Column('schema_name', sa.String(128), nullable=False, index=True),
        sa.Column('schema_version', sa.String(16), nullable=False, index=True),
        sa.Column('event_type', sa.String(32), nullable=False, index=True),
        sa.Column('event_name', sa.String(128), nullable=False, index=True),
        
        # Schema定义
        sa.Column('schema_definition', postgresql.JSONB, nullable=False),
        sa.Column('validation_rules', postgresql.JSONB, nullable=True),
        
        # 字段映射
        sa.Column('field_mappings', postgresql.JSONB, nullable=True),
        sa.Column('derived_fields', postgresql.JSONB, nullable=True),
        
        # 元数据
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('owner', sa.String(128), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        
        # 使用统计
        sa.Column('events_validated', sa.BigInteger, nullable=False, default=0),
        sa.Column('validation_success_rate', sa.Float, nullable=True),
        sa.Column('last_validation_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
    )
    
    # 创建Schema表索引和约束
    op.create_index('idx_schema_event_type', 'event_schemas', 
                   ['event_type', 'event_name', 'schema_version'])
    op.create_index('idx_active_schemas', 'event_schemas', ['is_active', 'event_type'])
    op.create_unique_constraint('uq_schema_unique', 'event_schemas',
                               ['schema_name', 'schema_version'])
    
    # 6. 创建事件去重表
    op.create_table(
        'event_deduplication',
        # 使用事件指纹作为主键
        sa.Column('event_fingerprint', sa.String(64), primary_key=True),
        
        # 原始事件信息
        sa.Column('original_event_id', sa.String(128), nullable=False, index=True),
        sa.Column('experiment_id', sa.String(128), nullable=False, index=True),
        sa.Column('user_id', sa.String(128), nullable=False, index=True),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        
        # 去重信息
        sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now()),
        sa.Column('duplicate_count', sa.Integer, nullable=False, default=0),
        sa.Column('last_duplicate_at', sa.DateTime(timezone=True), nullable=True),
        
        # TTL字段
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, index=True),
    )
    
    # 创建去重表索引
    op.create_index('idx_exp_user_time', 'event_deduplication', 
                   ['experiment_id', 'user_id', 'event_timestamp'])
    op.create_index('idx_expires_at', 'event_deduplication', ['expires_at'])
    
    # 7. 创建事件错误表
    op.create_table(
        'event_errors',
        sa.Column('id', sa.String(36), primary_key=True),
        
        # 错误事件信息
        sa.Column('failed_event_id', sa.String(128), nullable=True, index=True),
        sa.Column('raw_event_data', postgresql.JSONB, nullable=False),
        
        # 错误分类
        sa.Column('error_type', sa.String(64), nullable=False, index=True),
        sa.Column('error_code', sa.String(32), nullable=True, index=True),
        sa.Column('error_message', sa.Text, nullable=False),
        sa.Column('error_details', postgresql.JSONB, nullable=True),
        
        # 处理信息
        sa.Column('processing_stage', sa.String(64), nullable=False, index=True),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('last_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_recoverable', sa.Boolean, nullable=True),
        
        # 元数据
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False, 
                 default=lambda: utc_now(), index=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_method', sa.String(64), nullable=True),
    )
    
    # 创建错误表索引
    op.create_index('idx_error_type_stage', 'event_errors', ['error_type', 'processing_stage'])
    op.create_index('idx_retry_count_occurred', 'event_errors', ['retry_count', 'occurred_at'])
    op.create_index('idx_recoverable_occurred', 'event_errors', ['is_recoverable', 'occurred_at'])
    
    # 8. 创建用于分区的函数和触发器（可选，用于自动分区管理）
    # 注意：这部分在生产环境中可能需要更复杂的分区策略
    
    # 创建更新updated_at字段的触发器函数
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # 为需要的表创建触发器
    op.execute("""
        CREATE TRIGGER update_event_streams_updated_at 
        BEFORE UPDATE ON event_streams 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_event_batches_updated_at 
        BEFORE UPDATE ON event_batches 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)
    
    op.execute("""
        CREATE TRIGGER update_event_schemas_updated_at 
        BEFORE UPDATE ON event_schemas 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """)

def downgrade() -> None:
    """删除事件追踪表"""
    
    # 删除触发器
    op.execute("DROP TRIGGER IF EXISTS update_event_streams_updated_at ON event_streams;")
    op.execute("DROP TRIGGER IF EXISTS update_event_batches_updated_at ON event_batches;")
    op.execute("DROP TRIGGER IF EXISTS update_event_schemas_updated_at ON event_schemas;")
    
    # 删除函数
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    
    # 删除表（按依赖关系逆序删除）
    op.drop_table('event_errors')
    op.drop_table('event_deduplication')
    op.drop_table('event_schemas')
    op.drop_table('event_batches')
    op.drop_table('event_metrics')
    op.drop_table('event_aggregations')
    op.drop_table('event_streams')
