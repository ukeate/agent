"""
多步推理工作流持久化模型
扩展现有工作流模型以支持多步推理和任务分解
"""

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.core.database import Base


class WorkflowDefinitionModel(Base):
    """工作流定义持久化模型"""
    __tablename__ = "workflow_definitions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="工作流定义唯一标识"
    )
    
    name = Column(
        String(255),
        nullable=False,
        comment="工作流名称"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="工作流描述"
    )
    
    version = Column(
        String(20),
        nullable=False,
        default="1.0",
        comment="版本号"
    )
    
    # 步骤定义
    steps_definition = Column(
        JSONB,
        nullable=False,
        comment="步骤定义列表"
    )
    
    # 执行配置
    execution_mode = Column(
        String(50),
        nullable=False,
        default="sequential",
        comment="执行模式: sequential, parallel, hybrid"
    )
    
    max_parallel_steps = Column(
        Integer,
        nullable=False,
        default=5,
        comment="最大并行步骤数"
    )
    
    total_timeout_seconds = Column(
        Integer,
        nullable=True,
        comment="总超时时间(秒)"
    )
    
    # 错误处理配置
    error_handling = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="错误处理配置"
    )
    
    # 元数据
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="元数据"
    )
    
    tags = Column(
        ARRAY(String),
        nullable=False,
        default=list,
        comment="标签列表"
    )
    
    # 版本控制
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="是否激活"
    )
    
    created_by = Column(
        String(255),
        nullable=True,
        comment="创建者"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )
    
    # 关联关系
    executions = relationship("WorkflowExecutionModel", back_populates="definition")
    
    def __repr__(self):
        return f"<WorkflowDefinitionModel(id='{self.id}', name='{self.name}', version='{self.version}')>"


class WorkflowExecutionModel(Base):
    """工作流执行持久化模型"""
    __tablename__ = "workflow_executions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="工作流执行唯一标识"
    )
    
    workflow_definition_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_definitions.id"),
        nullable=False,
        comment="工作流定义ID"
    )
    
    session_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="会话ID"
    )
    
    # 执行状态
    status = Column(
        String(50),
        nullable=False,
        default="pending",
        comment="执行状态: pending, running, paused, completed, failed, cancelled"
    )
    
    current_step_id = Column(
        String(255),
        nullable=True,
        comment="当前步骤ID"
    )
    
    # 执行时间
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始时间"
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="完成时间"
    )
    
    # 执行结果
    final_result = Column(
        JSONB,
        nullable=True,
        comment="最终结果"
    )
    
    execution_context = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="执行上下文"
    )
    
    # 统计信息
    total_steps = Column(
        Integer,
        nullable=False,
        default=0,
        comment="总步骤数"
    )
    
    completed_steps = Column(
        Integer,
        nullable=False,
        default=0,
        comment="已完成步骤数"
    )
    
    failed_steps = Column(
        Integer,
        nullable=False,
        default=0,
        comment="失败步骤数"
    )
    
    # 性能指标
    total_execution_time_ms = Column(
        Integer,
        nullable=True,
        comment="总执行时间(毫秒)"
    )
    
    parallel_efficiency = Column(
        Float,
        nullable=True,
        comment="并行执行效率"
    )
    
    # 关联关系
    definition = relationship("WorkflowDefinitionModel", back_populates="executions")
    step_executions = relationship("WorkflowStepExecutionModel", back_populates="execution", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<WorkflowExecutionModel(id='{self.id}', status='{self.status}')>"
    
    @property
    def progress_percentage(self):
        """计算进度百分比"""
        if self.total_steps == 0:
            return 0
        return (self.completed_steps / self.total_steps) * 100


class WorkflowStepExecutionModel(Base):
    """工作流步骤执行持久化模型"""
    __tablename__ = "workflow_step_executions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="步骤执行唯一标识"
    )
    
    execution_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.id"),
        nullable=False,
        comment="工作流执行ID"
    )
    
    step_id = Column(
        String(255),
        nullable=False,
        comment="步骤ID"
    )
    
    status = Column(
        String(50),
        nullable=False,
        default="pending",
        comment="执行状态: pending, running, completed, failed, skipped, cancelled"
    )
    
    # 执行时间
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始时间"
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="完成时间"
    )
    
    duration_ms = Column(
        Integer,
        nullable=True,
        comment="执行时长(毫秒)"
    )
    
    # 执行结果
    input_data = Column(
        JSONB,
        nullable=True,
        comment="输入数据"
    )
    
    output_data = Column(
        JSONB,
        nullable=True,
        comment="输出数据"
    )
    
    error_message = Column(
        Text,
        nullable=True,
        comment="错误信息"
    )
    
    error_code = Column(
        String(100),
        nullable=True,
        comment="错误代码"
    )
    
    # 执行统计
    retry_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="重试次数"
    )
    
    memory_usage_mb = Column(
        Float,
        nullable=True,
        comment="内存使用(MB)"
    )
    
    cpu_usage_percent = Column(
        Float,
        nullable=True,
        comment="CPU使用率"
    )
    
    # 推理相关(如果是推理步骤)
    reasoning_chain_id = Column(
        String(255),
        nullable=True,
        comment="推理链ID"
    )
    
    confidence_score = Column(
        Float,
        nullable=True,
        comment="置信度分数"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    execution = relationship("WorkflowExecutionModel", back_populates="step_executions")
    
    def __repr__(self):
        return f"<WorkflowStepExecutionModel(id='{self.id}', step_id='{self.step_id}', status='{self.status}')>"


class TaskDecompositionModel(Base):
    """任务分解持久化模型"""
    __tablename__ = "task_decompositions"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="任务分解唯一标识"
    )
    
    problem_statement = Column(
        Text,
        nullable=False,
        comment="原始问题陈述"
    )
    
    context = Column(
        Text,
        nullable=True,
        comment="上下文信息"
    )
    
    # 分解配置
    max_depth = Column(
        Integer,
        nullable=False,
        default=3,
        comment="最大分解深度"
    )
    
    target_complexity = Column(
        String(50),
        nullable=False,
        default="medium",
        comment="目标复杂度"
    )
    
    reasoning_strategy = Column(
        String(100),
        nullable=False,
        default="zero_shot",
        comment="推理策略"
    )
    
    # 分解结果
    task_dag = Column(
        JSONB,
        nullable=True,
        comment="任务依赖图(DAG)"
    )
    
    total_tasks = Column(
        Integer,
        nullable=False,
        default=0,
        comment="总任务数"
    )
    
    max_depth_achieved = Column(
        Integer,
        nullable=False,
        default=0,
        comment="实际达到的最大深度"
    )
    
    # 质量指标
    decomposition_quality_score = Column(
        Float,
        nullable=True,
        comment="分解质量评分"
    )
    
    completeness_score = Column(
        Float,
        nullable=True,
        comment="完整性评分"
    )
    
    # 元数据
    decomposition_strategy = Column(
        String(100),
        nullable=True,
        comment="分解策略"
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="元数据"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    def __repr__(self):
        return f"<TaskDecompositionModel(id='{self.id}', total_tasks={self.total_tasks})>"


class WorkflowResultModel(Base):
    """工作流结果持久化模型"""
    __tablename__ = "workflow_results"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="工作流结果唯一标识"
    )
    
    workflow_execution_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflow_executions.id"),
        nullable=False,
        unique=True,
        comment="工作流执行ID"
    )
    
    final_result = Column(
        JSONB,
        nullable=False,
        comment="最终结果"
    )
    
    confidence_score = Column(
        Float,
        nullable=False,
        comment="整体置信度"
    )
    
    # 步骤结果
    step_results = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="各步骤结果"
    )
    
    aggregation_strategy = Column(
        String(100),
        nullable=False,
        comment="聚合策略"
    )
    
    # 质量指标
    consistency_score = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="一致性评分"
    )
    
    completeness_score = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="完整性评分"
    )
    
    accuracy_estimate = Column(
        Float,
        nullable=True,
        comment="准确性估计"
    )
    
    # 性能统计
    total_execution_time_ms = Column(
        Integer,
        nullable=False,
        comment="总执行时间(毫秒)"
    )
    
    parallel_efficiency = Column(
        Float,
        nullable=True,
        comment="并行执行效率"
    )
    
    resource_usage = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="资源使用情况"
    )
    
    # 元数据
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="元数据"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    # 关联关系
    workflow_execution = relationship("WorkflowExecutionModel")
    
    def __repr__(self):
        return f"<WorkflowResultModel(id='{self.id}', confidence_score={self.confidence_score})>"


class WorkflowCacheModel(Base):
    """工作流缓存持久化模型"""
    __tablename__ = "workflow_cache"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="缓存唯一标识"
    )
    
    cache_key = Column(
        String(512),
        unique=True,
        nullable=False,
        index=True,
        comment="缓存键"
    )
    
    workflow_type = Column(
        String(100),
        nullable=False,
        comment="工作流类型"
    )
    
    input_hash = Column(
        String(128),
        nullable=False,
        index=True,
        comment="输入数据哈希"
    )
    
    result_data = Column(
        JSONB,
        nullable=False,
        comment="缓存结果数据"
    )
    
    confidence_score = Column(
        Float,
        nullable=False,
        comment="结果置信度"
    )
    
    hit_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="命中次数"
    )
    
    ttl_seconds = Column(
        Integer,
        nullable=False,
        default=3600,
        comment="存活时间(秒)"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        comment="过期时间"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间"
    )
    
    last_accessed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="最后访问时间"
    )
    
    def __repr__(self):
        return f"<WorkflowCacheModel(id='{self.id}', cache_key='{self.cache_key}', hit_count={self.hit_count})>"
    
    @property
    def is_expired(self):
        """检查是否已过期"""
        return datetime.utcnow() > self.expires_at
    
    def increment_hit_count(self):
        """增加命中次数"""
        self.hit_count += 1
        self.last_accessed_at = datetime.utcnow()