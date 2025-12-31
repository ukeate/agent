"""
训练数据管理系统数据模型
"""

from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Boolean, Index
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

class Base(DeclarativeBase):
    ...

@dataclass
class DataSource:
    """数据源配置"""
    source_id: str
    source_type: str  # 'api', 'file', 'database', 'web', 'manual'
    name: str
    description: str
    config: Dict[str, Any]
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()

@dataclass
class DataRecord:
    """数据记录"""
    record_id: str
    source_id: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    status: str = 'raw'  # 'raw', 'processed', 'validated', 'rejected'
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()
        if self.metadata is None:
            self.metadata = {}

class AnnotationTaskType(Enum):
    """标注任务类型"""
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_LABELING = "sequence_labeling"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

class AnnotationStatus(Enum):
    """标注状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"

class AnnotationTaskStatus(Enum):
    """标注任务状态"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class AnnotationTask:
    """标注任务"""
    task_id: str
    name: str
    description: str
    task_type: str  # AnnotationTaskType value
    record_ids: List[str]  # record IDs 
    schema: Dict[str, Any]  # annotation schema
    annotators: List[str]  # user IDs
    created_by: str
    guidelines: str
    status: AnnotationTaskStatus = AnnotationTaskStatus.DRAFT
    created_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()

@dataclass
class Annotation:
    """标注结果"""
    annotation_id: str
    task_id: str
    record_id: str
    annotator_id: str
    annotation_data: Dict[str, Any]
    confidence: Optional[float] = None
    time_spent: Optional[int] = None  # seconds
    status: str = 'submitted'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()

@dataclass
class DataVersion:
    """数据版本"""
    version_id: str
    dataset_name: str
    version_number: str
    description: str
    created_by: str
    parent_version: Optional[str] = None
    changes_summary: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()
        if self.changes_summary is None:
            self.changes_summary = {}
        if self.metadata is None:
            self.metadata = {}

# SQLAlchemy模型

class DataSourceModel(Base):
    """数据源表模型"""
    __tablename__ = "data_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(255), unique=True, nullable=False)
    source_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    config = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: utc_now())
    updated_at = Column(DateTime(timezone=True), default=lambda: utc_now(), onupdate=lambda: utc_now())
    
    # 性能优化索引
    __table_args__ = (
        Index('idx_data_sources_source_type', 'source_type'),
        Index('idx_data_sources_is_active', 'is_active'),
        Index('idx_data_sources_created_at', 'created_at'),
    )

class DataRecordModel(Base):
    """数据记录表模型"""
    __tablename__ = "data_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    record_id = Column(String(255), unique=True, nullable=False)
    source_id = Column(String(255), nullable=False)
    raw_data = Column(JSONB)
    processed_data = Column(JSONB)
    record_metadata = Column('metadata', JSONB, default={})
    quality_score = Column(Float)
    status = Column(String(20), default='raw')
    created_at = Column(DateTime(timezone=True), default=lambda: utc_now())
    processed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), default=lambda: utc_now(), onupdate=lambda: utc_now())
    
    # 性能优化索引
    __table_args__ = (
        Index('idx_data_records_source_id', 'source_id'),
        Index('idx_data_records_status', 'status'),
        Index('idx_data_records_quality_score', 'quality_score'),
        Index('idx_data_records_created_at', 'created_at'),
        Index('idx_data_records_processed_at', 'processed_at'),
        Index('idx_data_records_composite', 'source_id', 'status', 'created_at'),
    )

class AnnotationTaskModel(Base):
    """标注任务表模型"""
    __tablename__ = "annotation_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    task_type = Column(String(50), nullable=False)
    data_records = Column(JSONB)  # List of record IDs
    annotation_schema = Column(JSONB)
    guidelines = Column(Text)
    assignees = Column(JSONB)  # List of user IDs
    created_by = Column(String(255), nullable=False)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime(timezone=True), default=utc_now)
    deadline = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

class AnnotationModel(Base):
    """标注结果表模型"""
    __tablename__ = "annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    annotation_id = Column(String(255), unique=True, nullable=False)
    task_id = Column(String(255), nullable=False)
    record_id = Column(String(255), nullable=False)
    annotator_id = Column(String(255), nullable=False)
    annotation_data = Column(JSONB)
    confidence = Column(Float)
    time_spent = Column(Integer)
    status = Column(String(20), default='submitted')
    created_at = Column(DateTime(timezone=True), default=lambda: utc_now())
    updated_at = Column(DateTime(timezone=True), default=lambda: utc_now(), onupdate=lambda: utc_now())

class DataVersionModel(Base):
    """数据版本表模型"""
    __tablename__ = "data_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(String(255), unique=True, nullable=False)
    dataset_name = Column(String(255), nullable=False)
    version_number = Column(String(50), nullable=False)
    description = Column(Text)
    created_by = Column(String(255), nullable=False)
    parent_version = Column(String(255))
    changes_summary = Column(JSONB, default={})
    version_metadata = Column('metadata', JSONB, default={})
    data_path = Column(String(500))  # 数据文件路径
    data_hash = Column(String(64))   # 数据内容哈希
    record_count = Column(Integer, default=0)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utc_now)
