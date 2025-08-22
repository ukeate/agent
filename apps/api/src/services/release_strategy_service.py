"""
发布策略配置管理服务

管理实验的发布策略和流程配置
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import json

from ..core.database import async_session_manager


class ReleaseType(str, Enum):
    """发布类型"""
    CANARY = "canary"  # 金丝雀发布
    BLUE_GREEN = "blue_green"  # 蓝绿发布
    ROLLING = "rolling"  # 滚动发布
    FEATURE_FLAG = "feature_flag"  # 功能开关
    GRADUAL = "gradual"  # 渐进发布
    SHADOW = "shadow"  # 影子发布


class ApprovalLevel(str, Enum):
    """审批级别"""
    NONE = "none"  # 无需审批
    SINGLE = "single"  # 单人审批
    MULTIPLE = "multiple"  # 多人审批
    TIERED = "tiered"  # 分级审批


class Environment(str, Enum):
    """环境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ReleaseStage:
    """发布阶段"""
    name: str
    environment: Environment
    traffic_percentage: float
    duration_hours: float
    success_criteria: Dict[str, Any]
    rollback_criteria: Dict[str, Any]
    approval_required: bool
    approvers: List[str] = field(default_factory=list)
    
    
@dataclass
class ReleaseStrategy:
    """发布策略"""
    id: str
    name: str
    description: str
    experiment_id: str
    release_type: ReleaseType
    stages: List[ReleaseStage]
    approval_level: ApprovalLevel
    auto_promote: bool  # 是否自动晋级
    auto_rollback: bool  # 是否自动回滚
    monitoring_config: Dict[str, Any]
    notification_config: Dict[str, Any]
    schedule: Optional[Dict[str, Any]] = None  # 发布计划
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReleaseExecution:
    """发布执行"""
    strategy_id: str
    experiment_id: str
    current_stage: int
    status: str  # pending, in_progress, completed, failed, rolled_back
    started_at: datetime
    completed_at: Optional[datetime] = None
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ReleaseStrategyService:
    """发布策略服务"""
    
    def __init__(self):
        self.strategies: Dict[str, ReleaseStrategy] = {}
        self.executions: Dict[str, ReleaseExecution] = {}
        self.templates: Dict[str, ReleaseStrategy] = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, ReleaseStrategy]:
        """初始化策略模板"""
        return {
            "safe_canary": self._create_safe_canary_template(),
            "fast_rollout": self._create_fast_rollout_template(),
            "blue_green": self._create_blue_green_template(),
            "progressive": self._create_progressive_template()
        }
        
    def _create_safe_canary_template(self) -> ReleaseStrategy:
        """创建安全金丝雀模板"""
        return ReleaseStrategy(
            id="template_safe_canary",
            name="安全金丝雀发布",
            description="小流量验证，逐步扩大",
            experiment_id="",
            release_type=ReleaseType.CANARY,
            stages=[
                ReleaseStage(
                    name="金丝雀验证",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=1.0,
                    duration_hours=24,
                    success_criteria={
                        "error_rate": {"max": 0.01},
                        "latency_p99": {"max": 1000},
                        "min_sample_size": 1000
                    },
                    rollback_criteria={
                        "error_rate": {"max": 0.05},
                        "latency_increase": {"max": 2.0}
                    },
                    approval_required=False
                ),
                ReleaseStage(
                    name="小规模推广",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=10.0,
                    duration_hours=48,
                    success_criteria={
                        "error_rate": {"max": 0.01},
                        "conversion_rate": {"min": 0.09}
                    },
                    rollback_criteria={
                        "error_rate": {"max": 0.03}
                    },
                    approval_required=True,
                    approvers=["tech_lead"]
                ),
                ReleaseStage(
                    name="大规模验证",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=50.0,
                    duration_hours=72,
                    success_criteria={
                        "all_metrics": "positive"
                    },
                    rollback_criteria={
                        "any_metric": "degraded"
                    },
                    approval_required=True,
                    approvers=["tech_lead", "product_manager"]
                ),
                ReleaseStage(
                    name="全量发布",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=100.0,
                    duration_hours=0,
                    success_criteria={},
                    rollback_criteria={},
                    approval_required=True,
                    approvers=["director"]
                )
            ],
            approval_level=ApprovalLevel.TIERED,
            auto_promote=False,
            auto_rollback=True,
            monitoring_config={
                "metrics": ["error_rate", "latency", "conversion_rate"],
                "alert_channels": ["email", "slack"],
                "check_interval_minutes": 5
            },
            notification_config={
                "on_stage_complete": True,
                "on_rollback": True,
                "on_approval_needed": True,
                "channels": ["email", "slack"]
            }
        )
        
    def _create_fast_rollout_template(self) -> ReleaseStrategy:
        """创建快速发布模板"""
        return ReleaseStrategy(
            id="template_fast_rollout",
            name="快速发布",
            description="快速推广到全量",
            experiment_id="",
            release_type=ReleaseType.ROLLING,
            stages=[
                ReleaseStage(
                    name="初始验证",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=20.0,
                    duration_hours=2,
                    success_criteria={
                        "error_rate": {"max": 0.02}
                    },
                    rollback_criteria={
                        "error_rate": {"max": 0.1}
                    },
                    approval_required=False
                ),
                ReleaseStage(
                    name="扩大范围",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=60.0,
                    duration_hours=4,
                    success_criteria={
                        "error_rate": {"max": 0.02}
                    },
                    rollback_criteria={
                        "error_rate": {"max": 0.05}
                    },
                    approval_required=False
                ),
                ReleaseStage(
                    name="全量发布",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=100.0,
                    duration_hours=0,
                    success_criteria={},
                    rollback_criteria={},
                    approval_required=False
                )
            ],
            approval_level=ApprovalLevel.NONE,
            auto_promote=True,
            auto_rollback=True,
            monitoring_config={
                "metrics": ["error_rate", "latency"],
                "check_interval_minutes": 1
            },
            notification_config={
                "on_complete": True,
                "channels": ["slack"]
            }
        )
        
    def _create_blue_green_template(self) -> ReleaseStrategy:
        """创建蓝绿发布模板"""
        return ReleaseStrategy(
            id="template_blue_green",
            name="蓝绿发布",
            description="完整切换，快速回滚",
            experiment_id="",
            release_type=ReleaseType.BLUE_GREEN,
            stages=[
                ReleaseStage(
                    name="绿色环境准备",
                    environment=Environment.STAGING,
                    traffic_percentage=0.0,
                    duration_hours=1,
                    success_criteria={
                        "health_check": "passed",
                        "smoke_test": "passed"
                    },
                    rollback_criteria={},
                    approval_required=False
                ),
                ReleaseStage(
                    name="流量切换",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=100.0,
                    duration_hours=0.5,
                    success_criteria={
                        "error_rate": {"max": 0.01}
                    },
                    rollback_criteria={
                        "error_rate": {"max": 0.05}
                    },
                    approval_required=True,
                    approvers=["ops_team"]
                ),
                ReleaseStage(
                    name="监控验证",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=100.0,
                    duration_hours=2,
                    success_criteria={
                        "all_metrics": "stable"
                    },
                    rollback_criteria={
                        "any_alert": "critical"
                    },
                    approval_required=False
                )
            ],
            approval_level=ApprovalLevel.SINGLE,
            auto_promote=False,
            auto_rollback=True,
            monitoring_config={
                "metrics": ["error_rate", "latency", "throughput"],
                "alert_channels": ["pagerduty"],
                "check_interval_minutes": 1
            },
            notification_config={
                "on_switch": True,
                "on_rollback": True,
                "channels": ["email", "slack", "pagerduty"]
            }
        )
        
    def _create_progressive_template(self) -> ReleaseStrategy:
        """创建渐进式发布模板"""
        return ReleaseStrategy(
            id="template_progressive",
            name="渐进式发布",
            description="基于指标自动调整流量",
            experiment_id="",
            release_type=ReleaseType.GRADUAL,
            stages=[
                ReleaseStage(
                    name="试点",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=5.0,
                    duration_hours=12,
                    success_criteria={
                        "conversion_rate": {"min": 0.1},
                        "error_rate": {"max": 0.01}
                    },
                    rollback_criteria={
                        "conversion_drop": {"max": 0.2}
                    },
                    approval_required=False
                ),
                ReleaseStage(
                    name="扩展",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=25.0,
                    duration_hours=24,
                    success_criteria={
                        "positive_metrics": {"min": 2}
                    },
                    rollback_criteria={
                        "negative_metrics": {"max": 1}
                    },
                    approval_required=False
                ),
                ReleaseStage(
                    name="推广",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=75.0,
                    duration_hours=48,
                    success_criteria={
                        "statistical_significance": True
                    },
                    rollback_criteria={
                        "confidence_drop": {"max": 0.1}
                    },
                    approval_required=True,
                    approvers=["data_scientist"]
                ),
                ReleaseStage(
                    name="完成",
                    environment=Environment.PRODUCTION,
                    traffic_percentage=100.0,
                    duration_hours=0,
                    success_criteria={},
                    rollback_criteria={},
                    approval_required=True,
                    approvers=["product_manager", "tech_lead"]
                )
            ],
            approval_level=ApprovalLevel.MULTIPLE,
            auto_promote=True,
            auto_rollback=False,
            monitoring_config={
                "metrics": ["conversion_rate", "revenue", "user_satisfaction"],
                "ml_models": ["anomaly_detection", "trend_prediction"],
                "check_interval_minutes": 10
            },
            notification_config={
                "on_milestone": True,
                "daily_summary": True,
                "channels": ["email", "dashboard"]
            }
        )
        
    async def create_strategy(
        self,
        experiment_id: str,
        name: str,
        release_type: ReleaseType,
        stages: List[ReleaseStage],
        **kwargs
    ) -> ReleaseStrategy:
        """创建发布策略"""
        strategy = ReleaseStrategy(
            id=f"strategy_{experiment_id}_{datetime.utcnow().timestamp()}",
            name=name,
            description=kwargs.get("description", ""),
            experiment_id=experiment_id,
            release_type=release_type,
            stages=stages,
            approval_level=kwargs.get("approval_level", ApprovalLevel.SINGLE),
            auto_promote=kwargs.get("auto_promote", False),
            auto_rollback=kwargs.get("auto_rollback", True),
            monitoring_config=kwargs.get("monitoring_config", {}),
            notification_config=kwargs.get("notification_config", {}),
            schedule=kwargs.get("schedule"),
            metadata=kwargs.get("metadata", {})
        )
        
        self.strategies[strategy.id] = strategy
        return strategy
        
    async def create_from_template(
        self,
        experiment_id: str,
        template_name: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> ReleaseStrategy:
        """从模板创建策略"""
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 不存在")
            
        template = self.templates[template_name]
        
        # 复制模板
        strategy = ReleaseStrategy(
            id=f"strategy_{experiment_id}_{datetime.utcnow().timestamp()}",
            name=template.name,
            description=template.description,
            experiment_id=experiment_id,
            release_type=template.release_type,
            stages=template.stages.copy(),
            approval_level=template.approval_level,
            auto_promote=template.auto_promote,
            auto_rollback=template.auto_rollback,
            monitoring_config=template.monitoring_config.copy(),
            notification_config=template.notification_config.copy()
        )
        
        # 应用自定义配置
        if customizations:
            for key, value in customizations.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
                    
        self.strategies[strategy.id] = strategy
        return strategy
        
    async def execute_strategy(self, strategy_id: str) -> ReleaseExecution:
        """执行发布策略"""
        if strategy_id not in self.strategies:
            raise ValueError(f"策略 {strategy_id} 不存在")
            
        strategy = self.strategies[strategy_id]
        
        execution = ReleaseExecution(
            strategy_id=strategy_id,
            experiment_id=strategy.experiment_id,
            current_stage=0,
            status="in_progress",
            started_at=datetime.utcnow()
        )
        
        exec_id = f"exec_{strategy_id}_{datetime.utcnow().timestamp()}"
        self.executions[exec_id] = execution
        
        # 这里应该启动异步任务执行各个阶段
        # 简化示例
        
        return execution
        
    async def approve_stage(
        self,
        exec_id: str,
        stage_index: int,
        approver: str,
        approved: bool,
        comments: Optional[str] = None
    ) -> bool:
        """审批阶段"""
        if exec_id not in self.executions:
            return False
            
        execution = self.executions[exec_id]
        strategy = self.strategies[execution.strategy_id]
        
        if stage_index >= len(strategy.stages):
            return False
            
        stage = strategy.stages[stage_index]
        
        if not stage.approval_required:
            return True
            
        # 记录审批
        approval = {
            "stage_index": stage_index,
            "stage_name": stage.name,
            "approver": approver,
            "approved": approved,
            "comments": comments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        execution.approvals.append(approval)
        
        # 检查是否满足审批要求
        if strategy.approval_level == ApprovalLevel.SINGLE:
            return approved
        elif strategy.approval_level == ApprovalLevel.MULTIPLE:
            # 需要所有审批人批准
            required_approvers = set(stage.approvers)
            approved_by = {
                a["approver"] for a in execution.approvals
                if a["stage_index"] == stage_index and a["approved"]
            }
            return required_approvers.issubset(approved_by)
            
        return False
        
    async def get_execution_status(self, exec_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态"""
        if exec_id not in self.executions:
            return None
            
        execution = self.executions[exec_id]
        strategy = self.strategies[execution.strategy_id]
        
        current_stage = None
        if execution.current_stage < len(strategy.stages):
            current_stage = strategy.stages[execution.current_stage]
            
        return {
            "exec_id": exec_id,
            "strategy_name": strategy.name,
            "experiment_id": execution.experiment_id,
            "status": execution.status,
            "current_stage_index": execution.current_stage,
            "current_stage_name": current_stage.name if current_stage else None,
            "total_stages": len(strategy.stages),
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "approvals_pending": self._get_pending_approvals(execution, strategy),
            "metrics": execution.metrics
        }
        
    def _get_pending_approvals(
        self,
        execution: ReleaseExecution,
        strategy: ReleaseStrategy
    ) -> List[Dict[str, Any]]:
        """获取待审批项"""
        pending = []
        
        if execution.current_stage < len(strategy.stages):
            stage = strategy.stages[execution.current_stage]
            if stage.approval_required:
                approved_by = {
                    a["approver"] for a in execution.approvals
                    if a["stage_index"] == execution.current_stage and a["approved"]
                }
                
                for approver in stage.approvers:
                    if approver not in approved_by:
                        pending.append({
                            "stage": stage.name,
                            "approver": approver,
                            "required_by": datetime.utcnow() + timedelta(hours=24)
                        })
                        
        return pending
        
    async def validate_strategy(self, strategy: ReleaseStrategy) -> List[str]:
        """验证策略配置"""
        errors = []
        
        # 验证阶段
        if not strategy.stages:
            errors.append("至少需要一个发布阶段")
            
        # 验证流量百分比
        for i, stage in enumerate(strategy.stages):
            if stage.traffic_percentage < 0 or stage.traffic_percentage > 100:
                errors.append(f"阶段{i+1}的流量百分比无效")
                
            if stage.approval_required and not stage.approvers:
                errors.append(f"阶段{i+1}需要审批但未指定审批人")
                
        # 验证监控配置
        if strategy.auto_rollback and not strategy.monitoring_config:
            errors.append("自动回滚需要配置监控")
            
        return errors