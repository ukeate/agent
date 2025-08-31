"""
训练数据管理系统核心数据结构和实体类

这个模块定义了系统的核心数据结构，包括：
- 数据源配置
- 数据记录实体
- 标注任务和结果
- 版本管理实体
"""

from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, Any, Optional, List, AsyncIterator
from abc import ABC, abstractmethod
from enum import Enum

from .models import SourceType, DataStatus, AnnotationTaskType, AnnotationStatus


@dataclass
class DataSource:
    """数据源配置"""
    source_id: str
    source_type: SourceType
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
    """数据记录实体"""
    record_id: str
    source_id: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    status: DataStatus = DataStatus.RAW
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()
        if not self.metadata:
            self.metadata = {}


@dataclass
class AnnotationTask:
    """标注任务实体"""
    task_id: str
    name: str
    description: str
    task_type: AnnotationTaskType
    data_records: List[str]  # record IDs
    annotation_schema: Dict[str, Any]
    guidelines: str
    assignees: List[str]  # user IDs
    created_by: str
    status: AnnotationStatus = AnnotationStatus.PENDING
    created_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()


@dataclass
class Annotation:
    """标注结果实体"""
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
    """数据版本实体"""
    version_id: str
    dataset_name: str
    version_number: str
    description: str
    created_by: str
    parent_version: Optional[str] = None
    changes_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()


# 数据收集器抽象基类
class DataCollector(ABC):
    """数据收集器抽象基类"""
    
    def __init__(self, source: DataSource):
        self.source = source
    
    @abstractmethod
    async def collect_data(self) -> AsyncIterator[DataRecord]:
        """收集数据的抽象方法"""
        pass
    
    def generate_record_id(self, data: Dict[str, Any]) -> str:
        """生成唯一的记录ID"""
        import json
        import hashlib
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()


# 数据处理规则抽象基类
class ProcessingRule(ABC):
    """数据处理规则抽象基类"""
    
    @abstractmethod
    async def apply(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """应用处理规则"""
        pass


# 质量评估器抽象基类
class QualityAssessor(ABC):
    """质量评估器抽象基类"""
    
    @abstractmethod
    def assess(self, data: Dict[str, Any]) -> float:
        """评估数据质量"""
        pass


# 数据导出格式枚举
class ExportFormat(Enum):
    """数据导出格式"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"


# 任务统计信息
@dataclass
class TaskStatistics:
    """任务统计信息"""
    task_id: str
    total_records: int
    annotated_records: int
    progress_percentage: float
    status_counts: Dict[str, int]
    annotator_stats: List[Dict[str, Any]]


# 版本对比结果
@dataclass
class VersionComparison:
    """版本对比结果"""
    version1_id: str
    version2_id: str
    summary: Dict[str, Any]
    added_records: List[Dict[str, Any]]
    removed_records: List[Dict[str, Any]]
    modified_records: List[Dict[str, Any]]


# 收集统计信息
@dataclass
class CollectionStats:
    """收集统计信息"""
    total_collected: int = 0
    total_processed: int = 0
    total_stored: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


# 数据过滤器
@dataclass
class DataFilter:
    """数据过滤器"""
    source_id: Optional[str] = None
    status: Optional[DataStatus] = None
    min_quality_score: Optional[float] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: Optional[int] = 1000
    offset: int = 0


# 标注进度信息
@dataclass 
class AnnotationProgress:
    """标注进度信息"""
    task_id: str
    total_records: int
    annotated_records: int
    progress_percentage: float
    status_distribution: Dict[str, int]
    annotator_performance: List[Dict[str, Any]]
    estimated_completion: Optional[datetime] = None