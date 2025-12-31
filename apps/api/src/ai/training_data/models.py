"""
训练数据管理系统的核心数据模型

这个模块定义了训练数据管理系统的SQLAlchemy数据模型，包括：
- 数据源管理
- 数据记录存储
- 标注任务管理
- 数据版本控制
"""

import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import Index

class Base(DeclarativeBase):
    ...

class SourceType(Enum):
    """数据源类型枚举"""
    API = "api"
    FILE = "file" 
    DATABASE = "database"
    WEB = "web"
    MANUAL = "manual"

class DataStatus(Enum):
    """数据处理状态枚举"""
    RAW = "raw"
    PROCESSED = "processed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ERROR = "error"

class AnnotationTaskType(Enum):
    """标注任务类型枚举"""
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_LABELING = "sequence_labeling"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class AnnotationStatus(Enum):
    """标注状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"

class DataSourceModel(Base):
    """数据源模型"""
    __tablename__ = "data_sources"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utc_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: utc_now(),
        onupdate=lambda: utc_now()
    )
    
    # 关系
    records: Mapped[List["DataRecordModel"]] = relationship("DataRecordModel", back_populates="source")

class DataRecordModel(Base):
    """数据记录模型"""
    __tablename__ = "data_records"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    record_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(255), ForeignKey("data_sources.source_id"), nullable=False)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    processed_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default=DataStatus.RAW.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utc_now())
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: utc_now(),
        onupdate=lambda: utc_now()
    )
    
    # 关系
    source: Mapped["DataSourceModel"] = relationship("DataSourceModel", back_populates="records")
    annotations: Mapped[List["AnnotationModel"]] = relationship("AnnotationModel", back_populates="data_record")

class AnnotationTaskModel(Base):
    """标注任务模型"""
    __tablename__ = "annotation_tasks"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    data_records: Mapped[List[str]] = mapped_column(JSONB, default=lambda: [])  # record IDs列表
    annotation_schema: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    guidelines: Mapped[Optional[str]] = mapped_column(Text)
    assignees: Mapped[List[str]] = mapped_column(JSONB, default=lambda: [])  # user IDs列表
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default=AnnotationStatus.PENDING.value)
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utc_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: utc_now(),
        onupdate=lambda: utc_now()
    )
    
    # 关系
    annotations: Mapped[List["AnnotationModel"]] = relationship("AnnotationModel", back_populates="task")

class AnnotationModel(Base):
    """标注结果模型"""
    __tablename__ = "annotations"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    annotation_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(255), ForeignKey("annotation_tasks.task_id"), nullable=False)
    record_id: Mapped[str] = mapped_column(String(255), ForeignKey("data_records.record_id"), nullable=False)
    annotator_id: Mapped[str] = mapped_column(String(255), nullable=False)
    annotation_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    time_spent: Mapped[Optional[int]] = mapped_column(Integer)  # 秒数
    status: Mapped[str] = mapped_column(String(20), default='submitted')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utc_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: utc_now(),
        onupdate=lambda: utc_now()
    )
    
    # 关系
    task: Mapped["AnnotationTaskModel"] = relationship("AnnotationTaskModel", back_populates="annotations")
    data_record: Mapped["DataRecordModel"] = relationship("DataRecordModel", back_populates="annotations")
    
    # 复合唯一约束
    __table_args__ = (
        # 确保每个标注员对每条记录在同一个任务中只能标注一次
        {"sqlite_on_conflict": "REPLACE"}  # 对于SQLite
    )

class DataVersionModel(Base):
    """数据版本模型"""
    __tablename__ = "data_versions"
    
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    version_number: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_version: Mapped[Optional[str]] = mapped_column(String(255), ForeignKey("data_versions.version_id"))
    changes_summary: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    data_path: Mapped[Optional[str]] = mapped_column(String(500))  # 数据文件路径
    data_hash: Mapped[Optional[str]] = mapped_column(String(64))   # 数据内容哈希
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: utc_now())
    
    # 自引用关系
    parent: Mapped[Optional["DataVersionModel"]] = relationship("DataVersionModel", remote_side=[version_id])

# 创建数据库索引

# 为常用查询字段创建索引
Index('idx_data_sources_type', DataSourceModel.source_type)
Index('idx_data_sources_active', DataSourceModel.is_active)
Index('idx_data_records_source', DataRecordModel.source_id)
Index('idx_data_records_status', DataRecordModel.status)
Index('idx_data_records_quality', DataRecordModel.quality_score)
Index('idx_data_records_created', DataRecordModel.created_at)
Index('idx_annotation_tasks_status', AnnotationTaskModel.status)
Index('idx_annotation_tasks_created_by', AnnotationTaskModel.created_by)
Index('idx_annotations_task', AnnotationModel.task_id)
Index('idx_annotations_record', AnnotationModel.record_id)
Index('idx_annotations_annotator', AnnotationModel.annotator_id)
Index('idx_data_versions_dataset', DataVersionModel.dataset_name)
Index('idx_data_versions_parent', DataVersionModel.parent_version)
Index('idx_data_versions_created', DataVersionModel.created_at)
