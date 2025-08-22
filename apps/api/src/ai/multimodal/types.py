"""
多模态处理类型定义
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List


class ContentType(str, Enum):
    """内容类型枚举"""
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class ProcessingStatus(str, Enum):
    """处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class ModelPriority(str, Enum):
    """模型选择优先级"""
    COST = "cost"
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"


class ModelComplexity(str, Enum):
    """任务复杂度"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class MultimodalContent:
    """多模态内容数据类"""
    content_id: str
    content_type: ContentType
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    

@dataclass
class ProcessingResult:
    """处理结果数据类"""
    content_id: str
    status: ProcessingStatus
    extracted_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: Optional[str] = None
    tokens_used: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class ProcessingOptions:
    """处理选项配置"""
    priority: ModelPriority = ModelPriority.BALANCED
    complexity: ModelComplexity = ModelComplexity.MEDIUM
    max_tokens: int = 1000
    temperature: float = 0.1
    enable_cache: bool = True
    extract_text: bool = True
    extract_objects: bool = True
    extract_sentiment: bool = False