"""
反馈系统数据库迁移
创建用户反馈学习系统所需的所有数据表
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, Index, ForeignKey, create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()

class FeedbackEvent(Base):
    """反馈事件表"""
    __tablename__ = 'feedback_events'
    
    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 业务字段
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    item_id = Column(String(255), nullable=True, index=True)
    
    # 反馈内容
    feedback_type = Column(String(50), nullable=False, index=True)
    value = Column(Text, nullable=False)  # JSON字符串存储
    raw_value = Column(JSONB, nullable=True)
    context = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=True)
    
    # 时间戳
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)
    
    # 状态标识
    processed = Column(Boolean, default=False, nullable=False, index=True)
    valid = Column(Boolean, default=True, nullable=False, index=True)
    
    # 质量评分
    quality_score = Column(Float, nullable=True)
    quality_factors = Column(JSONB, nullable=True)
    
    # 批次关联
    batch_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # 创建索引
    __table_args__ = (
        Index('idx_feedback_user_time', user_id, timestamp),
        Index('idx_feedback_item_time', item_id, timestamp),
        Index('idx_feedback_type_time', feedback_type, timestamp),
        Index('idx_feedback_batch_processed', batch_id, processed),
    )

class FeedbackBatch(Base):
    """反馈批次表"""
    __tablename__ = 'feedback_batches'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False)
    
    event_count = Column(Integer, default=0)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    processed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserFeedbackProfile(Base):
    """用户反馈档案表"""
    __tablename__ = 'user_feedback_profiles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 统计数据
    total_feedbacks = Column(Integer, default=0)
    total_explicit_feedbacks = Column(Integer, default=0)
    total_implicit_feedbacks = Column(Integer, default=0)
    
    # 评分和质量
    average_rating = Column(Float, nullable=True)
    average_quality_score = Column(Float, default=0.0)
    trust_score = Column(Float, default=1.0)
    consistency_score = Column(Float, default=1.0)
    
    # 偏好向量
    preference_vector = Column(JSONB, nullable=True)
    category_preferences = Column(JSONB, nullable=True)
    
    # 行为模式
    activity_patterns = Column(JSONB, nullable=True)
    engagement_metrics = Column(JSONB, nullable=True)
    
    # 时间戳
    first_feedback_at = Column(DateTime, nullable=True)
    last_feedback_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ItemFeedbackSummary(Base):
    """推荐项反馈汇总表"""
    __tablename__ = 'item_feedback_summaries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 统计数据
    total_feedbacks = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    
    # 评分统计
    average_rating = Column(Float, nullable=True)
    rating_count = Column(Integer, default=0)
    rating_distribution = Column(JSONB, nullable=True)  # {1: count, 2: count, ...}
    
    # 点赞统计
    like_count = Column(Integer, default=0)
    dislike_count = Column(Integer, default=0)
    like_ratio = Column(Float, default=0.0)
    
    # 其他交互
    comment_count = Column(Integer, default=0)
    bookmark_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    
    # 质量指标
    average_quality_score = Column(Float, default=0.0)
    quality_distribution = Column(JSONB, nullable=True)
    
    # 参与度指标
    engagement_metrics = Column(JSONB, nullable=True)
    behavioral_signals = Column(JSONB, nullable=True)
    
    # 时间戳
    first_feedback_at = Column(DateTime, nullable=True)
    last_feedback_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RewardSignal(Base):
    """奖励信号表"""
    __tablename__ = 'reward_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    item_id = Column(String(255), nullable=False, index=True)
    
    # 奖励值
    reward_value = Column(Float, nullable=False)
    normalized_reward = Column(Float, nullable=False)
    confidence_score = Column(Float, default=1.0)
    
    # 计算参数
    calculation_strategy = Column(String(100), nullable=False)
    strategy_params = Column(JSONB, nullable=True)
    time_window_seconds = Column(Integer, nullable=False)
    
    # 组成因素
    feedback_components = Column(JSONB, nullable=True)  # 各种反馈的贡献
    temporal_factors = Column(JSONB, nullable=True)     # 时间因素
    context_factors = Column(JSONB, nullable=True)      # 上下文因素
    
    # 有效性
    calculated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # 索引
    __table_args__ = (
        Index('idx_reward_user_item', user_id, item_id),
        Index('idx_reward_calculated_at', calculated_at),
    )

class FeedbackQualityLog(Base):
    """反馈质量评估日志表"""
    __tablename__ = 'feedback_quality_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feedback_event_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # 质量评分
    quality_score = Column(Float, nullable=False)
    quality_factors = Column(JSONB, nullable=False)  # 各项质量因子评分
    
    # 检测结果
    is_valid = Column(Boolean, nullable=False)
    anomaly_detected = Column(Boolean, default=False)
    anomaly_reasons = Column(JSONB, nullable=True)
    
    # 处理信息
    assessment_method = Column(String(100), nullable=False)  # 评估方法
    assessment_config = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class FeedbackAggregation(Base):
    """反馈聚合数据表"""
    __tablename__ = 'feedback_aggregations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 聚合维度
    aggregation_type = Column(String(50), nullable=False)  # daily, weekly, monthly
    dimension = Column(String(50), nullable=False)         # user, item, type
    dimension_value = Column(String(255), nullable=False)  # 维度值
    
    # 时间范围
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # 聚合数据
    metrics = Column(JSONB, nullable=False)  # 包含各种聚合指标
    metadata = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 复合索引
    __table_args__ = (
        Index('idx_aggregation_type_dim', aggregation_type, dimension, dimension_value),
        Index('idx_aggregation_period', period_start, period_end),
    )


def upgrade(connection):
    """执行迁移升级"""
    print("创建反馈系统数据表...")
    
    # 创建所有表
    Base.metadata.create_all(connection)
    
    # 添加额外的索引和约束
    connection.execute("""
        -- 为时间序列查询优化的部分索引
        CREATE INDEX IF NOT EXISTS idx_feedback_events_recent 
        ON feedback_events (timestamp DESC) 
        WHERE timestamp > NOW() - INTERVAL '30 days';
        
        -- 为用户活跃度查询优化
        CREATE INDEX IF NOT EXISTS idx_feedback_events_user_recent 
        ON feedback_events (user_id, timestamp DESC) 
        WHERE valid = true AND timestamp > NOW() - INTERVAL '7 days';
        
        -- 为推荐项热度查询优化
        CREATE INDEX IF NOT EXISTS idx_feedback_events_item_type 
        ON feedback_events (item_id, feedback_type) 
        WHERE valid = true;
        
        -- 为质量分析查询优化
        CREATE INDEX IF NOT EXISTS idx_feedback_events_quality 
        ON feedback_events (quality_score DESC) 
        WHERE quality_score IS NOT NULL;
    """)
    
    print("反馈系统数据表创建完成!")


def downgrade(connection):
    """执行迁移降级"""
    print("删除反馈系统数据表...")
    
    # 删除额外索引
    connection.execute("""
        DROP INDEX IF EXISTS idx_feedback_events_recent;
        DROP INDEX IF EXISTS idx_feedback_events_user_recent;
        DROP INDEX IF EXISTS idx_feedback_events_item_type;
        DROP INDEX IF EXISTS idx_feedback_events_quality;
    """)
    
    # 删除所有表（按照依赖关系的相反顺序）
    tables_to_drop = [
        'feedback_aggregations',
        'feedback_quality_logs', 
        'reward_signals',
        'item_feedback_summaries',
        'user_feedback_profiles',
        'feedback_events',
        'feedback_batches'
    ]
    
    for table in tables_to_drop:
        connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    
    print("反馈系统数据表删除完成!")


if __name__ == "__main__":
    # 直接运行时的测试代码
    from sqlalchemy import create_engine
    
    # 这里可以添加测试连接字符串
    # engine = create_engine('postgresql://user:pass@localhost/dbname')
    # upgrade(engine)
    print("反馈系统迁移脚本已准备就绪")