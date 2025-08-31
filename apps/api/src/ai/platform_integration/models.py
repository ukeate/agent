"""平台集成数据模型"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, ConfigDict


class ComponentType(str, Enum):
    """组件类型枚举"""
    FINE_TUNING = "fine_tuning"
    COMPRESSION = "compression"
    HYPERPARAMETER = "hyperparameter"
    EVALUATION = "evaluation"
    DATA_MANAGEMENT = "data_management"
    MODEL_SERVICE = "model_service"
    CUSTOM = "custom"


class ComponentStatus(str, Enum):
    """组件状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class WorkflowStatus(str, Enum):
    """工作流状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """组件信息"""
    component_id: str
    component_type: ComponentType
    name: str
    version: str
    status: ComponentStatus
    health_endpoint: str
    api_endpoint: str
    metadata: Dict[str, Any]
    registered_at: datetime
    last_heartbeat: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['component_type'] = self.component_type.value
        data['status'] = self.status.value
        data['registered_at'] = self.registered_at.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


class ComponentRegistration(BaseModel):
    """组件注册请求模型"""
    component_id: str = Field(..., description="组件唯一标识")
    component_type: ComponentType = Field(..., description="组件类型")
    name: str = Field(..., description="组件名称")
    version: str = Field(..., description="组件版本")
    health_endpoint: str = Field(..., description="健康检查端点")
    api_endpoint: str = Field(..., description="API端点")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="组件元数据")
    
    model_config = ConfigDict(use_enum_values=True)


class WorkflowRequest(BaseModel):
    """工作流请求模型"""
    workflow_type: str = Field(..., description="工作流类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工作流参数")
    priority: int = Field(default=0, description="优先级")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_type": "full_fine_tuning",
                "parameters": {
                    "model_name": "gpt-4",
                    "data_config": {
                        "dataset": "custom_dataset",
                        "batch_size": 32
                    }
                },
                "priority": 1
            }
        }
    )


class WorkflowStep(BaseModel):
    """工作流步骤"""
    step_name: str = Field(..., description="步骤名称")
    status: WorkflowStatus = Field(..., description="步骤状态")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    result: Optional[Dict[str, Any]] = Field(None, description="步骤结果")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = ConfigDict(use_enum_values=True)


class WorkflowState(BaseModel):
    """工作流状态"""
    workflow_id: str = Field(..., description="工作流ID")
    workflow_type: str = Field(..., description="工作流类型")
    status: WorkflowStatus = Field(..., description="工作流状态")
    steps: List[WorkflowStep] = Field(default_factory=list, description="工作流步骤")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="工作流参数")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    current_step: Optional[str] = Field(None, description="当前步骤")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = ConfigDict(use_enum_values=True)


class PlatformHealthStatus(BaseModel):
    """平台健康状态"""
    overall_status: str = Field(..., description="整体状态")
    healthy_components: int = Field(..., description="健康组件数")
    total_components: int = Field(..., description="总组件数")
    components: Dict[str, Dict[str, Any]] = Field(..., description="组件状态详情")
    timestamp: datetime = Field(..., description="时间戳")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "overall_status": "healthy",
                "healthy_components": 5,
                "total_components": 6,
                "components": {
                    "fine_tuning_service": {
                        "status": "healthy",
                        "last_heartbeat": "2025-08-23T10:00:00Z",
                        "component_type": "fine_tuning"
                    }
                },
                "timestamp": "2025-08-23T10:00:00Z"
            }
        }
    )


class PerformanceMetrics(BaseModel):
    """性能指标"""
    cpu_percent: float = Field(..., description="CPU使用率")
    memory_percent: float = Field(..., description="内存使用率")
    disk_usage: Dict[str, Any] = Field(..., description="磁盘使用情况")
    network_usage: Dict[str, Any] = Field(..., description="网络使用情况")
    bottlenecks: List[str] = Field(default_factory=list, description="性能瓶颈")
    timestamp: datetime = Field(..., description="时间戳")


class MonitoringConfig(BaseModel):
    """监控配置"""
    prometheus_enabled: bool = Field(default=True, description="是否启用Prometheus")
    grafana_enabled: bool = Field(default=True, description="是否启用Grafana")
    alert_rules: List[Dict[str, Any]] = Field(default_factory=list, description="告警规则")
    health_check_interval: int = Field(default=30, description="健康检查间隔(秒)")
    metrics_retention_days: int = Field(default=30, description="指标保留天数")