"""
流量渐进调整服务

实现实验流量的渐进式调整和安全发布
"""
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass, field
import math

from ..core.database import get_db_session
from ..services.anomaly_detection_service import AnomalyDetectionService
from ..services.realtime_metrics_service import RealtimeMetricsService
from ..services.alert_rules_service import AlertRulesEngine


class RampStrategy(str, Enum):
    """流量爬坡策略"""
    LINEAR = "linear"  # 线性增长
    EXPONENTIAL = "exponential"  # 指数增长
    LOGARITHMIC = "logarithmic"  # 对数增长
    STEP = "step"  # 阶梯增长
    CUSTOM = "custom"  # 自定义曲线


class RampStatus(str, Enum):
    """爬坡状态"""
    SCHEDULED = "scheduled"  # 已计划
    RUNNING = "running"  # 进行中
    PAUSED = "paused"  # 已暂停
    COMPLETED = "completed"  # 已完成
    ROLLED_BACK = "rolled_back"  # 已回滚
    FAILED = "failed"  # 失败


class RolloutPhase(str, Enum):
    """发布阶段"""
    CANARY = "canary"  # 金丝雀发布 (1-5%)
    PILOT = "pilot"  # 试点发布 (5-20%)
    BETA = "beta"  # Beta发布 (20-50%)
    GRADUAL = "gradual"  # 渐进发布 (50-95%)
    FULL = "full"  # 全量发布 (95-100%)


@dataclass
class RampStep:
    """爬坡步骤"""
    step_number: int
    target_percentage: float
    duration_minutes: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_percentage: Optional[float] = None
    metrics_snapshot: Optional[Dict[str, Any]] = None
    health_check_passed: Optional[bool] = None
    
    
@dataclass
class RampPlan:
    """爬坡计划"""
    experiment_id: str
    variant: str
    strategy: RampStrategy
    start_percentage: float
    target_percentage: float
    total_duration_hours: float
    steps: List[RampStep]
    health_checks: Dict[str, Any]
    rollback_conditions: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    
@dataclass
class RampExecution:
    """爬坡执行记录"""
    plan_id: str
    experiment_id: str
    status: RampStatus
    current_step: int
    current_percentage: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    alerts_triggered: List[str] = field(default_factory=list)


class TrafficRampService:
    """流量爬坡服务"""
    
    def __init__(self):
        self.ramp_plans: Dict[str, RampPlan] = {}
        self.executions: Dict[str, RampExecution] = {}
        self.active_ramps: Dict[str, asyncio.Task] = {}
        self.anomaly_service = AnomalyDetectionService()
        self.metrics_service = RealtimeMetricsService()
        self.alert_engine = AlertRulesEngine()
        
    async def create_ramp_plan(
        self,
        experiment_id: str,
        variant: str = "treatment",
        strategy: RampStrategy = RampStrategy.LINEAR,
        start_percentage: float = 0.0,
        target_percentage: float = 100.0,
        duration_hours: float = 24.0,
        num_steps: int = 10,
        health_checks: Optional[Dict[str, Any]] = None,
        rollback_conditions: Optional[Dict[str, Any]] = None
    ) -> RampPlan:
        """
        创建流量爬坡计划
        
        Args:
            experiment_id: 实验ID
            variant: 变体名称
            strategy: 爬坡策略
            start_percentage: 起始流量百分比
            target_percentage: 目标流量百分比
            duration_hours: 总持续时间(小时)
            num_steps: 步骤数量
            health_checks: 健康检查配置
            rollback_conditions: 回滚条件
            
        Returns:
            爬坡计划
        """
        # 生成步骤
        steps = self._generate_ramp_steps(
            strategy,
            start_percentage,
            target_percentage,
            duration_hours,
            num_steps
        )
        
        # 默认健康检查
        if health_checks is None:
            health_checks = {
                "min_sample_size": 1000,
                "max_error_rate": 0.01,
                "max_latency_p99": 1000,
                "min_success_rate": 0.99
            }
            
        # 默认回滚条件
        if rollback_conditions is None:
            rollback_conditions = {
                "error_rate_threshold": 0.05,
                "latency_increase": 2.0,
                "alert_severity": "critical",
                "manual_trigger": True
            }
            
        plan = RampPlan(
            experiment_id=experiment_id,
            variant=variant,
            strategy=strategy,
            start_percentage=start_percentage,
            target_percentage=target_percentage,
            total_duration_hours=duration_hours,
            steps=steps,
            health_checks=health_checks,
            rollback_conditions=rollback_conditions
        )
        
        plan_id = f"ramp_{experiment_id}_{utc_now().timestamp()}"
        self.ramp_plans[plan_id] = plan
        
        return plan
        
    def _generate_ramp_steps(
        self,
        strategy: RampStrategy,
        start: float,
        target: float,
        duration_hours: float,
        num_steps: int
    ) -> List[RampStep]:
        """生成爬坡步骤"""
        steps = []
        step_duration = int(duration_hours * 60 / num_steps)  # 每步持续时间(分钟)
        
        if strategy == RampStrategy.LINEAR:
            # 线性增长
            for i in range(num_steps):
                progress = (i + 1) / num_steps
                percentage = start + (target - start) * progress
                steps.append(RampStep(
                    step_number=i + 1,
                    target_percentage=round(percentage, 2),
                    duration_minutes=step_duration
                ))
                
        elif strategy == RampStrategy.EXPONENTIAL:
            # 指数增长
            for i in range(num_steps):
                progress = (i + 1) / num_steps
                # 使用指数函数: y = a * e^(b*x)
                exp_progress = (math.exp(progress * 2) - 1) / (math.exp(2) - 1)
                percentage = start + (target - start) * exp_progress
                steps.append(RampStep(
                    step_number=i + 1,
                    target_percentage=round(percentage, 2),
                    duration_minutes=step_duration
                ))
                
        elif strategy == RampStrategy.LOGARITHMIC:
            # 对数增长
            for i in range(num_steps):
                progress = (i + 1) / num_steps
                # 使用对数函数: y = log(1 + x * (e - 1))
                log_progress = math.log(1 + progress * (math.e - 1))
                percentage = start + (target - start) * log_progress
                steps.append(RampStep(
                    step_number=i + 1,
                    target_percentage=round(percentage, 2),
                    duration_minutes=step_duration
                ))
                
        elif strategy == RampStrategy.STEP:
            # 阶梯增长
            step_size = (target - start) / num_steps
            for i in range(num_steps):
                percentage = start + step_size * (i + 1)
                steps.append(RampStep(
                    step_number=i + 1,
                    target_percentage=round(percentage, 2),
                    duration_minutes=step_duration
                ))
                
        else:  # CUSTOM
            # 自定义曲线，这里使用S曲线作为示例
            for i in range(num_steps):
                progress = (i + 1) / num_steps
                # S曲线 (sigmoid)
                s_progress = 1 / (1 + math.exp(-10 * (progress - 0.5)))
                percentage = start + (target - start) * s_progress
                steps.append(RampStep(
                    step_number=i + 1,
                    target_percentage=round(percentage, 2),
                    duration_minutes=step_duration
                ))
                
        return steps
        
    async def start_ramp(self, plan_id: str) -> RampExecution:
        """
        开始执行爬坡计划
        
        Args:
            plan_id: 计划ID
            
        Returns:
            执行记录
        """
        if plan_id not in self.ramp_plans:
            raise ValueError(f"计划 {plan_id} 不存在")
            
        plan = self.ramp_plans[plan_id]
        
        # 创建执行记录
        execution = RampExecution(
            plan_id=plan_id,
            experiment_id=plan.experiment_id,
            status=RampStatus.RUNNING,
            current_step=0,
            current_percentage=plan.start_percentage,
            started_at=utc_now()
        )
        
        exec_id = f"exec_{plan_id}_{utc_now().timestamp()}"
        self.executions[exec_id] = execution
        
        # 启动异步任务
        task = asyncio.create_task(self._execute_ramp(exec_id, plan, execution))
        self.active_ramps[exec_id] = task
        
        return execution
        
    async def _execute_ramp(
        self,
        exec_id: str,
        plan: RampPlan,
        execution: RampExecution
    ):
        """执行爬坡计划"""
        try:
            for step in plan.steps:
                if execution.status != RampStatus.RUNNING:
                    break
                    
                # 更新当前步骤
                execution.current_step = step.step_number
                step.start_time = utc_now()
                
                # 调整流量
                await self._adjust_traffic(
                    plan.experiment_id,
                    plan.variant,
                    step.target_percentage
                )
                
                execution.current_percentage = step.target_percentage
                step.actual_percentage = step.target_percentage
                
                # 等待步骤持续时间
                await asyncio.sleep(step.duration_minutes * 60)
                
                # 执行健康检查
                health_passed = await self._perform_health_check(
                    plan.experiment_id,
                    plan.health_checks
                )
                step.health_check_passed = health_passed
                
                if not health_passed:
                    # 健康检查失败，触发回滚
                    await self._rollback(exec_id, "健康检查失败")
                    break
                    
                # 检查回滚条件
                should_rollback, reason = await self._check_rollback_conditions(
                    plan.experiment_id,
                    plan.rollback_conditions
                )
                
                if should_rollback:
                    await self._rollback(exec_id, reason)
                    break
                    
                # 记录指标快照
                metrics = await self.metrics_service.get_experiment_metrics(
                    plan.experiment_id
                )
                step.metrics_snapshot = metrics
                execution.metrics_history.append({
                    "step": step.step_number,
                    "percentage": step.target_percentage,
                    "timestamp": utc_now().isoformat(),
                    "metrics": metrics
                })
                
                step.end_time = utc_now()
                
            # 完成爬坡
            if execution.status == RampStatus.RUNNING:
                execution.status = RampStatus.COMPLETED
                execution.completed_at = utc_now()
                
        except Exception as e:
            execution.status = RampStatus.FAILED
            execution.rollback_reason = str(e)
            await self._rollback(exec_id, f"执行失败: {e}")
            
    async def _adjust_traffic(
        self,
        experiment_id: str,
        variant: str,
        percentage: float
    ):
        """调整流量百分比"""
        # 这里应该调用实际的流量分配服务
        # 更新实验配置中的流量分配
        print(f"调整实验 {experiment_id} 变体 {variant} 流量到 {percentage}%")
        
    async def _perform_health_check(
        self,
        experiment_id: str,
        health_checks: Dict[str, Any]
    ) -> bool:
        """执行健康检查"""
        # 获取实时指标
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查样本量
        if health_checks.get("min_sample_size"):
            total_samples = sum(
                m.get("count", 0) 
                for m in metrics.values()
            )
            if total_samples < health_checks["min_sample_size"]:
                return False
                
        # 检查错误率
        if health_checks.get("max_error_rate"):
            error_rate = metrics.get("error_rate", {}).get("value", 0)
            if error_rate > health_checks["max_error_rate"]:
                return False
                
        # 检查延迟
        if health_checks.get("max_latency_p99"):
            latency_p99 = metrics.get("latency", {}).get("p99", 0)
            if latency_p99 > health_checks["max_latency_p99"]:
                return False
                
        # 检查成功率
        if health_checks.get("min_success_rate"):
            success_rate = metrics.get("success_rate", {}).get("value", 1.0)
            if success_rate < health_checks["min_success_rate"]:
                return False
                
        return True
        
    async def _check_rollback_conditions(
        self,
        experiment_id: str,
        rollback_conditions: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """检查回滚条件"""
        # 获取实时指标
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查错误率阈值
        if rollback_conditions.get("error_rate_threshold"):
            error_rate = metrics.get("error_rate", {}).get("value", 0)
            if error_rate > rollback_conditions["error_rate_threshold"]:
                return True, f"错误率 {error_rate:.2%} 超过阈值"
                
        # 检查延迟增加
        if rollback_conditions.get("latency_increase"):
            baseline_latency = metrics.get("latency", {}).get("baseline", 100)
            current_latency = metrics.get("latency", {}).get("current", 100)
            if current_latency > baseline_latency * rollback_conditions["latency_increase"]:
                return True, f"延迟增加 {current_latency/baseline_latency:.1f}x"
                
        # 检查告警
        if rollback_conditions.get("alert_severity"):
            active_alerts = await self.alert_engine.get_active_alerts(
                experiment_id=experiment_id,
                severity=rollback_conditions["alert_severity"]
            )
            if active_alerts:
                return True, f"触发 {rollback_conditions['alert_severity']} 级别告警"
                
        return False, None
        
    async def _rollback(self, exec_id: str, reason: str):
        """执行回滚"""
        if exec_id not in self.executions:
            return
            
        execution = self.executions[exec_id]
        plan = self.ramp_plans[execution.plan_id]
        
        # 更新状态
        execution.status = RampStatus.ROLLED_BACK
        execution.rolled_back_at = utc_now()
        execution.rollback_reason = reason
        
        # 恢复到初始流量
        await self._adjust_traffic(
            plan.experiment_id,
            plan.variant,
            plan.start_percentage
        )
        
        # 发送告警
        alert_data = {
            "experiment_id": plan.experiment_id,
            "variant": plan.variant,
            "rollback_reason": reason,
            "current_percentage": execution.current_percentage,
            "timestamp": utc_now().isoformat()
        }
        
        await self.alert_engine.evaluate_rules(alert_data)
        
    async def pause_ramp(self, exec_id: str) -> bool:
        """暂停爬坡"""
        if exec_id not in self.executions:
            return False
            
        execution = self.executions[exec_id]
        
        if execution.status == RampStatus.RUNNING:
            execution.status = RampStatus.PAUSED
            execution.paused_at = utc_now()
            
            # 取消异步任务
            if exec_id in self.active_ramps:
                self.active_ramps[exec_id].cancel()
                del self.active_ramps[exec_id]
                
            return True
            
        return False
        
    async def resume_ramp(self, exec_id: str) -> bool:
        """恢复爬坡"""
        if exec_id not in self.executions:
            return False
            
        execution = self.executions[exec_id]
        
        if execution.status == RampStatus.PAUSED:
            execution.status = RampStatus.RUNNING
            plan = self.ramp_plans[execution.plan_id]
            
            # 从当前步骤继续
            remaining_steps = plan.steps[execution.current_step:]
            plan.steps = remaining_steps
            
            # 重新启动任务
            task = asyncio.create_task(
                self._execute_ramp(exec_id, plan, execution)
            )
            self.active_ramps[exec_id] = task
            
            return True
            
        return False
        
    async def get_ramp_status(self, exec_id: str) -> Optional[Dict[str, Any]]:
        """获取爬坡状态"""
        if exec_id not in self.executions:
            return None
            
        execution = self.executions[exec_id]
        plan = self.ramp_plans[execution.plan_id]
        
        # 计算进度
        progress = (execution.current_step / len(plan.steps)) * 100 if plan.steps else 0
        
        # 估算剩余时间
        if execution.status == RampStatus.RUNNING and execution.current_step < len(plan.steps):
            remaining_steps = len(plan.steps) - execution.current_step
            avg_step_duration = plan.total_duration_hours * 60 / len(plan.steps)
            eta_minutes = remaining_steps * avg_step_duration
            eta = utc_now() + timedelta(minutes=eta_minutes)
        else:
            eta = None
            
        return {
            "exec_id": exec_id,
            "plan_id": execution.plan_id,
            "experiment_id": execution.experiment_id,
            "status": execution.status,
            "progress": progress,
            "current_step": execution.current_step,
            "total_steps": len(plan.steps),
            "current_percentage": execution.current_percentage,
            "target_percentage": plan.target_percentage,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "eta": eta.isoformat() if eta else None,
            "metrics_history": execution.metrics_history[-10:],  # 最近10个数据点
            "alerts_triggered": execution.alerts_triggered
        }
        
    async def get_recommended_plan(
        self,
        experiment_id: str,
        risk_level: str = "medium"
    ) -> RampPlan:
        """
        获取推荐的爬坡计划
        
        Args:
            experiment_id: 实验ID
            risk_level: 风险等级 (low, medium, high)
            
        Returns:
            推荐的爬坡计划
        """
        # 根据风险等级推荐参数
        if risk_level == "low":
            # 低风险：快速爬坡
            return await self.create_ramp_plan(
                experiment_id=experiment_id,
                strategy=RampStrategy.EXPONENTIAL,
                start_percentage=5.0,
                target_percentage=100.0,
                duration_hours=12.0,
                num_steps=5
            )
        elif risk_level == "high":
            # 高风险：缓慢谨慎
            return await self.create_ramp_plan(
                experiment_id=experiment_id,
                strategy=RampStrategy.LOGARITHMIC,
                start_percentage=1.0,
                target_percentage=100.0,
                duration_hours=72.0,
                num_steps=20
            )
        else:  # medium
            # 中等风险：平衡方案
            return await self.create_ramp_plan(
                experiment_id=experiment_id,
                strategy=RampStrategy.LINEAR,
                start_percentage=5.0,
                target_percentage=100.0,
                duration_hours=24.0,
                num_steps=10
            )
            
    def get_phase_recommendation(self, current_percentage: float) -> RolloutPhase:
        """获取当前发布阶段建议"""
        if current_percentage <= 5:
            return RolloutPhase.CANARY
        elif current_percentage <= 20:
            return RolloutPhase.PILOT
        elif current_percentage <= 50:
            return RolloutPhase.BETA
        elif current_percentage <= 95:
            return RolloutPhase.GRADUAL
        else:
            return RolloutPhase.FULL