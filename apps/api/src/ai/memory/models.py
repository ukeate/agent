"""记忆系统数据模型定义"""
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class MemoryType(Enum):
    """记忆类型枚举"""
    WORKING = "working"      # 工作记忆(短期)
    EPISODIC = "episodic"    # 情景记忆(事件)
    SEMANTIC = "semantic"    # 语义记忆(知识)


class MemoryStatus(Enum):
    """记忆状态枚举"""
    ACTIVE = "active"        # 活跃
    ARCHIVED = "archived"    # 已归档
    COMPRESSED = "compressed" # 已压缩
    DELETED = "deleted"      # 已删除


class Memory(BaseModel):
    """记忆数据模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MemoryType
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)  # 重要性评分 0-1
    access_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    decay_factor: float = Field(default=0.5, ge=0.0, le=1.0)  # 遗忘系数
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # 关联信息
    related_memories: List[str] = Field(default_factory=list)  # 相关记忆ID列表
    tags: List[str] = Field(default_factory=list)  # 标签
    source: Optional[str] = None  # 记忆来源
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryCreateRequest(BaseModel):
    """创建记忆请求模型"""
    type: MemoryType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    importance: Optional[float] = 0.5
    tags: Optional[List[str]] = None
    source: Optional[str] = None


class MemoryUpdateRequest(BaseModel):
    """更新记忆请求模型"""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: Optional[float] = None
    tags: Optional[List[str]] = None
    status: Optional[MemoryStatus] = None


class MemoryResponse(BaseModel):
    """记忆响应模型"""
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any]
    importance: float
    access_count: int
    created_at: datetime
    last_accessed: datetime
    status: MemoryStatus
    tags: List[str]
    relevance_score: Optional[float] = None  # 相关性评分(查询时使用)
    
    @classmethod
    def from_memory(cls, memory: Memory, relevance_score: Optional[float] = None):
        """从Memory对象创建响应"""
        return cls(
            id=memory.id,
            type=memory.type,
            content=memory.content,
            metadata=memory.metadata,
            importance=memory.importance,
            access_count=memory.access_count,
            created_at=memory.created_at,
            last_accessed=memory.last_accessed,
            status=memory.status,
            tags=memory.tags,
            relevance_score=relevance_score
        )


class MemoryQuery(BaseModel):
    """记忆查询模型"""
    query: str
    memory_types: Optional[List[MemoryType]] = None
    limit: int = Field(default=10, ge=1, le=100)
    time_range: Optional[Dict[str, datetime]] = None
    min_importance: Optional[float] = None
    tags: Optional[List[str]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class MemoryFilters(BaseModel):
    """记忆过滤器"""
    memory_types: Optional[List[MemoryType]] = None
    status: Optional[List[MemoryStatus]] = None
    min_importance: Optional[float] = None
    max_importance: Optional[float] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class MemoryAnalytics(BaseModel):
    """记忆分析统计"""
    total_memories: int
    memories_by_type: Dict[str, int]
    memories_by_status: Dict[str, int]
    avg_importance: float
    total_access_count: int
    avg_access_count: float
    most_accessed_memories: List[MemoryResponse]
    recent_memories: List[MemoryResponse]
    memory_growth_rate: float  # 记忆增长率
    storage_usage_mb: float  # 存储使用量(MB)
    
    
class ImportResult(BaseModel):
    """导入结果模型"""
    success_count: int
    failed_count: int
    errors: List[str]
    imported_ids: List[str]