"""
行为分析数据模型

定义行为事件、用户会话、行为模式和异常检测的数据结构。
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from src.core.utils.timezone_utils import utc_factory
from enum import Enum
from pydantic import BaseModel, Field
import uuid

class EventType(str, Enum):
    """事件类型枚举"""
    USER_ACTION = "user_action"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"
    FEEDBACK_EVENT = "feedback_event"

class BehaviorEvent(BaseModel):
    """行为事件数据模型"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    event_type: EventType
    event_name: str = Field(..., description="如: click_button, send_message, view_page")
    event_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict, description="设备、位置、渠道等上下文")
    timestamp: datetime = Field(default_factory=utc_factory)
    duration_ms: Optional[int] = Field(None, description="事件持续时间(毫秒)")

class UserSession(BaseModel):
    """用户会话数据模型"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_time: datetime = Field(default_factory=utc_factory)
    end_time: Optional[datetime] = None
    events_count: int = 0
    interaction_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    anomaly_flags: List[str] = Field(default_factory=list)

class BehaviorPattern(BaseModel):
    """行为模式数据模型"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_name: str
    pattern_type: str = Field(..., description="sequence, frequency, temporal")
    pattern_definition: Dict[str, Any]
    support: float = Field(..., ge=0.0, le=1.0, description="支持度")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    users_count: int = Field(..., ge=0)
    examples: List[str] = Field(default_factory=list, description="示例session_ids")
    
    created_at: datetime = Field(default_factory=utc_factory)
    updated_at: datetime = Field(default_factory=utc_factory)

class AnomalyType(str, Enum):
    """异常类型枚举"""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"

class SeverityLevel(str, Enum):
    """异常严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AnomalyDetection(BaseModel):
    """异常检测数据模型"""
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    anomaly_score: float = Field(..., description="异常分数,范围根据检测方法而定")
    detection_method: str = Field(..., description="检测方法名称")
    details: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=utc_factory)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

class TrendMetric(BaseModel):
    """趋势指标数据模型"""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str
    metric_type: str = Field(..., description="count, average, sum, rate")
    value: float
    timestamp: datetime = Field(default_factory=utc_factory)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ForecastResult(BaseModel):
    """预测结果数据模型"""
    forecast_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str
    forecast_horizon: int = Field(..., description="预测时间范围(天数)")
    predicted_values: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    forecast_dates: List[datetime]
    model_type: str = Field(..., description="预测模型类型: arima, prophet, lstm")
    model_params: Dict[str, Any] = Field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_factory)

class DashboardConfig(BaseModel):
    """仪表板配置数据模型"""
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    dashboard_name: str
    layout: Dict[str, Any] = Field(default_factory=dict)
    widgets: List[Dict[str, Any]] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    refresh_interval: int = Field(30, description="刷新间隔(秒)")
    is_public: bool = False
    created_at: datetime = Field(default_factory=utc_factory)
    updated_at: datetime = Field(default_factory=utc_factory)

# 批量操作模型

class BulkEventRequest(BaseModel):
    """批量事件请求模型"""
    events: List[BehaviorEvent]
    batch_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    compression: Optional[str] = Field(None, description="压缩方式: gzip, lz4")
    
class EventQueryFilter(BaseModel):
    """事件查询过滤器"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    event_types: Optional[List[EventType]] = None
    event_names: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=10000)
    offset: int = Field(0, ge=0)

class PatternQueryFilter(BaseModel):
    """模式查询过滤器"""
    pattern_types: Optional[List[str]] = None
    min_support: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_users_count: Optional[int] = Field(None, ge=1)
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)

class AnomalyQueryFilter(BaseModel):
    """异常查询过滤器"""
    user_id: Optional[str] = None
    anomaly_types: Optional[List[AnomalyType]] = None
    severity_levels: Optional[List[SeverityLevel]] = None
    resolved: Optional[bool] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)
