"""
自动扩缩容引擎

基于负载和资源使用情况自动调整智能体集群规模。
实现基于Kubernetes HPA/VPA模式的智能扩缩容策略。
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from .topology import AgentInfo, AgentGroup, AgentStatus, ResourceUsage, ResourceSpec
from .state_manager import ClusterStateManager
from .lifecycle_manager import LifecycleManager, BatchOperationResult
from .metrics_collector import MetricsCollector

from src.core.logging import get_logger
class ScalingAction(Enum):
    """扩缩容动作"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

class ScalingReason(Enum):
    """扩缩容原因"""
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_RESPONSE_TIME = "high_response_time"
    LOW_UTILIZATION = "low_utilization"
    MANUAL_TRIGGER = "manual_trigger"
    POLICY_VIOLATION = "policy_violation"

@dataclass
class ScalingPolicy:
    """扩缩容策略配置"""
    policy_id: str = "default"
    name: str = "Default Scaling Policy"
    
    # 目标指标
    target_cpu_percent: float = 70.0       # 目标CPU使用率
    target_memory_percent: float = 75.0    # 目标内存使用率
    target_response_time_ms: float = 2000.0 # 目标响应时间
    max_error_rate: float = 0.05           # 最大错误率
    
    # 扩容阈值
    scale_up_cpu_threshold: float = 80.0    # CPU超过80%扩容
    scale_up_memory_threshold: float = 85.0  # 内存超过85%扩容
    scale_up_response_threshold: float = 3000.0 # 响应时间超过3s扩容
    
    # 缩容阈值
    scale_down_cpu_threshold: float = 30.0   # CPU低于30%缩容
    scale_down_memory_threshold: float = 35.0 # 内存低于35%缩容
    scale_down_response_threshold: float = 1000.0 # 响应时间低于1s缩容
    
    # 扩缩容参数
    scale_up_factor: float = 1.5            # 扩容倍数
    scale_down_factor: float = 0.8          # 缩容倍数
    min_instances: int = 1                  # 最小实例数
    max_instances: int = 10                 # 最大实例数
    
    # 稳定性参数
    stabilization_window_seconds: int = 300  # 稳定时间窗口（5分钟）
    cooldown_period_seconds: int = 180      # 冷却时间（3分钟）
    evaluation_window_seconds: int = 120    # 评估窗口（2分钟）
    
    # 策略开关
    enabled: bool = True
    scale_up_enabled: bool = True
    scale_down_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "target_metrics": {
                "cpu_percent": self.target_cpu_percent,
                "memory_percent": self.target_memory_percent,
                "response_time_ms": self.target_response_time_ms,
                "max_error_rate": self.max_error_rate
            },
            "scale_up_thresholds": {
                "cpu": self.scale_up_cpu_threshold,
                "memory": self.scale_up_memory_threshold,
                "response_time": self.scale_up_response_threshold
            },
            "scale_down_thresholds": {
                "cpu": self.scale_down_cpu_threshold,
                "memory": self.scale_down_memory_threshold,
                "response_time": self.scale_down_response_threshold
            },
            "scaling_factors": {
                "up": self.scale_up_factor,
                "down": self.scale_down_factor
            },
            "instance_limits": {
                "min": self.min_instances,
                "max": self.max_instances
            },
            "timings": {
                "stabilization_window": self.stabilization_window_seconds,
                "cooldown_period": self.cooldown_period_seconds,
                "evaluation_window": self.evaluation_window_seconds
            },
            "enabled": self.enabled,
            "scale_up_enabled": self.scale_up_enabled,
            "scale_down_enabled": self.scale_down_enabled
        }

@dataclass
class ScalingDecision:
    """扩缩容决策"""
    action: ScalingAction
    reason: ScalingReason
    current_instances: int
    target_instances: int
    confidence: float  # 0-1，决策置信度
    metrics: Dict[str, float]  # 触发决策的指标
    policy_id: str
    timestamp: float = field(default_factory=time.time)
    
    @property
    def instance_change(self) -> int:
        """实例数变化"""
        return self.target_instances - self.current_instances
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "instance_change": self.instance_change,
            "confidence": self.confidence,
            "metrics": self.metrics,
            "policy_id": self.policy_id,
            "timestamp": self.timestamp
        }

@dataclass
class ScalingEvent:
    """扩缩容事件记录"""
    event_id: str
    decision: ScalingDecision
    group_id: str
    execution_result: Optional[BatchOperationResult] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """执行时长"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def complete(self, success: bool, result: Optional[BatchOperationResult] = None, error: Optional[str] = None):
        """完成事件"""
        self.success = success
        self.execution_result = result
        self.error_message = error
        self.end_time = time.time()

class AutoScaler:
    """自动扩缩容引擎
    
    提供基于负载的智能扩缩容功能，包括：
    - 实时监控集群负载和资源使用情况
    - 基于配置策略自动决策扩缩容操作
    - 支持多种扩缩容策略和触发条件
    - 提供扩缩容历史和性能分析
    """
    
    def __init__(
        self,
        cluster_manager: ClusterStateManager,
        lifecycle_manager: LifecycleManager,
        metrics_collector: MetricsCollector,
        default_policy: Optional[ScalingPolicy] = None
    ):
        self.cluster_manager = cluster_manager
        self.lifecycle_manager = lifecycle_manager
        self.metrics_collector = metrics_collector
        self.logger = get_logger(__name__)
        
        # 扩缩容策略
        self.policies: Dict[str, ScalingPolicy] = {}
        self.group_policies: Dict[str, str] = {}  # 分组ID -> 策略ID映射
        
        # 设置默认策略
        if default_policy:
            self.policies[default_policy.policy_id] = default_policy
        else:
            self.policies["default"] = ScalingPolicy()
        
        # 扩缩容状态跟踪
        self.last_scaling_time: Dict[str, float] = {}  # 分组ID -> 最后扩缩容时间
        self.scaling_history: List[ScalingEvent] = []
        self.max_history_size = 1000
        
        # 扩缩容任务
        self.scaling_task: Optional[asyncio.Task] = None
        self.evaluation_interval = 60.0  # 每60秒评估一次
        
        # 扩缩容锁，防止并发操作
        self.scaling_locks: Dict[str, asyncio.Lock] = {}
        
        # 性能指标
        self.scaler_metrics = {
            "evaluations_performed": 0,
            "scale_up_actions": 0,
            "scale_down_actions": 0,
            "scaling_events": 0,
            "successful_scalings": 0,
            "failed_scalings": 0
        }
        
        self.logger.info("AutoScaler initialized")
    
    async def start(self):
        """启动自动扩缩容引擎"""
        try:
            # 启动扩缩容评估任务
            self.scaling_task = asyncio.create_task(self._scaling_loop())
            
            self.logger.info("AutoScaler started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start AutoScaler: {e}")
            raise
    
    async def stop(self):
        """停止自动扩缩容引擎"""
        try:
            # 停止扩缩容任务
            if self.scaling_task:
                self.scaling_task.cancel()
                try:
                    await self.scaling_task
                except asyncio.CancelledError:
                    raise
            
            self.logger.info("AutoScaler stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping AutoScaler: {e}")
    
    # 策略管理
    def add_policy(self, policy: ScalingPolicy):
        """添加扩缩容策略"""
        self.policies[policy.policy_id] = policy
        self.logger.info(f"Scaling policy added: {policy.name} ({policy.policy_id})")
    
    def remove_policy(self, policy_id: str):
        """移除扩缩容策略"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            
            # 清理分组策略映射
            groups_to_remove = [
                group_id for group_id, pid in self.group_policies.items() 
                if pid == policy_id
            ]
            for group_id in groups_to_remove:
                del self.group_policies[group_id]
            
            self.logger.info(f"Scaling policy removed: {policy_id}")
    
    def get_policies(self) -> List[ScalingPolicy]:
        """获取所有策略"""
        return list(self.policies.values())
    
    def set_group_policy(self, group_id: str, policy_id: str):
        """为分组设置扩缩容策略"""
        if policy_id in self.policies:
            self.group_policies[group_id] = policy_id
            self.logger.info(f"Group {group_id} assigned policy {policy_id}")
        else:
            raise ValueError(f"Policy {policy_id} not found")
    
    def get_group_policy(self, group_id: str) -> ScalingPolicy:
        """获取分组的扩缩容策略"""
        policy_id = self.group_policies.get(group_id, "default")
        return self.policies.get(policy_id, self.policies["default"])
    
    # 扩缩容决策
    async def evaluate_scaling_need(self, group_id: str) -> ScalingDecision:
        """评估分组的扩缩容需求"""
        
        try:
            # 获取分组信息
            topology = await self.cluster_manager.get_cluster_topology()
            group = topology.groups.get(group_id)
            
            if not group:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    reason=ScalingReason.POLICY_VIOLATION,
                    current_instances=0,
                    target_instances=0,
                    confidence=0.0,
                    metrics={},
                    policy_id="none"
                )
            
            # 获取策略
            policy = self.get_group_policy(group_id)
            if not policy.enabled:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    reason=ScalingReason.POLICY_VIOLATION,
                    current_instances=group.agent_count,
                    target_instances=group.agent_count,
                    confidence=0.0,
                    metrics={},
                    policy_id=policy.policy_id
                )
            
            # 获取分组中的智能体
            group_agents = topology.get_group_agents(group_id)
            healthy_agents = [agent for agent in group_agents if agent.is_healthy]
            
            current_instances = len(healthy_agents)
            
            # 如果没有健康的智能体，无法评估
            if current_instances == 0:
                return ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    reason=ScalingReason.POLICY_VIOLATION,
                    current_instances=0,
                    target_instances=max(1, policy.min_instances),
                    confidence=0.0,
                    metrics={},
                    policy_id=policy.policy_id
                )
            
            # 收集分组的聚合指标
            metrics = await self._collect_group_metrics(group_agents)
            
            # 评估扩缩容需求
            decision = await self._make_scaling_decision(
                policy, current_instances, metrics
            )
            
            # 应用实例限制
            decision.target_instances = max(
                policy.min_instances,
                min(policy.max_instances, decision.target_instances)
            )
            
            # 检查冷却期
            if await self._is_in_cooldown(group_id, policy):
                decision.action = ScalingAction.NO_ACTION
                decision.confidence *= 0.5  # 降低置信度
            
            self.logger.debug(
                f"Scaling evaluation for group {group_id}: "
                f"{decision.action.value} ({decision.current_instances} -> {decision.target_instances})"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating scaling need for group {group_id}: {e}")
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                reason=ScalingReason.POLICY_VIOLATION,
                current_instances=0,
                target_instances=0,
                confidence=0.0,
                metrics={},
                policy_id="error"
            )
    
    async def execute_scaling_decision(self, group_id: str, decision: ScalingDecision) -> ScalingEvent:
        """执行扩缩容决策"""
        
        event_id = f"scaling-{group_id}-{int(time.time())}"
        event = ScalingEvent(
            event_id=event_id,
            decision=decision,
            group_id=group_id
        )
        
        try:
            # 获取扩缩容锁
            if group_id not in self.scaling_locks:
                self.scaling_locks[group_id] = asyncio.Lock()
            
            async with self.scaling_locks[group_id]:
                if decision.action == ScalingAction.SCALE_UP:
                    result = await self._scale_up_group(group_id, decision)
                elif decision.action == ScalingAction.SCALE_DOWN:
                    result = await self._scale_down_group(group_id, decision)
                else:
                    # 无操作
                    result = None
                
                # 记录结果
                if result:
                    event.complete(result.success_rate >= 0.8, result)
                    
                    # 更新最后扩缩容时间
                    self.last_scaling_time[group_id] = time.time()
                    
                    # 更新指标
                    if decision.action == ScalingAction.SCALE_UP:
                        self.scaler_metrics["scale_up_actions"] += 1
                    else:
                        self.scaler_metrics["scale_down_actions"] += 1
                    
                    if event.success:
                        self.scaler_metrics["successful_scalings"] += 1
                    else:
                        self.scaler_metrics["failed_scalings"] += 1
                else:
                    event.complete(True)  # 无操作视为成功
                
                # 记录事件
                self.scaling_history.append(event)
                if len(self.scaling_history) > self.max_history_size:
                    self.scaling_history = self.scaling_history[-self.max_history_size:]
                
                self.scaler_metrics["scaling_events"] += 1
                
                self.logger.info(
                    f"Scaling event {event_id} completed: "
                    f"{decision.action.value} for group {group_id}, "
                    f"success: {event.success}"
                )
                
                return event
                
        except Exception as e:
            self.logger.error(f"Error executing scaling decision: {e}")
            event.complete(False, error=str(e))
            self.scaler_metrics["failed_scalings"] += 1
            return event
    
    # 手动扩缩容
    async def manual_scale(
        self, 
        group_id: str, 
        target_instances: int,
        reason: str = "Manual scaling operation"
    ) -> ScalingEvent:
        """手动扩缩容"""
        
        try:
            # 获取当前实例数
            topology = await self.cluster_manager.get_cluster_topology()
            group = topology.groups.get(group_id)
            
            if not group:
                raise ValueError(f"Group {group_id} not found")
            
            current_instances = group.agent_count
            
            # 创建手动扩缩容决策
            if target_instances > current_instances:
                action = ScalingAction.SCALE_UP
            elif target_instances < current_instances:
                action = ScalingAction.SCALE_DOWN
            else:
                action = ScalingAction.NO_ACTION
            
            decision = ScalingDecision(
                action=action,
                reason=ScalingReason.MANUAL_TRIGGER,
                current_instances=current_instances,
                target_instances=target_instances,
                confidence=1.0,
                metrics={},
                policy_id="manual"
            )
            
            # 执行扩缩容
            return await self.execute_scaling_decision(group_id, decision)
            
        except Exception as e:
            self.logger.error(f"Error in manual scaling: {e}")
            # 创建失败事件
            event = ScalingEvent(
                event_id=f"manual-{group_id}-{int(time.time())}",
                decision=ScalingDecision(
                    action=ScalingAction.NO_ACTION,
                    reason=ScalingReason.MANUAL_TRIGGER,
                    current_instances=0,
                    target_instances=target_instances,
                    confidence=0.0,
                    metrics={},
                    policy_id="manual"
                ),
                group_id=group_id
            )
            event.complete(False, error=str(e))
            return event
    
    # 查询接口
    async def get_scaling_recommendations(self) -> Dict[str, ScalingDecision]:
        """获取所有分组的扩缩容建议"""
        
        recommendations = {}
        
        try:
            topology = await self.cluster_manager.get_cluster_topology()
            
            for group_id in topology.groups.keys():
                decision = await self.evaluate_scaling_need(group_id)
                recommendations[group_id] = decision
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting scaling recommendations: {e}")
            return {}
    
    def get_scaling_history(
        self, 
        group_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ScalingEvent]:
        """获取扩缩容历史"""
        
        filtered_history = self.scaling_history
        
        # 按分组过滤
        if group_id:
            filtered_history = [
                event for event in filtered_history 
                if event.group_id == group_id
            ]
        
        # 按时间降序排序并限制数量
        filtered_history.sort(key=lambda x: x.start_time, reverse=True)
        return filtered_history[:limit]
    
    def get_scaler_metrics(self) -> Dict[str, Any]:
        """获取扩缩容器指标"""
        return {
            **self.scaler_metrics,
            "active_policies": len(self.policies),
            "group_policy_mappings": len(self.group_policies),
            "scaling_history_size": len(self.scaling_history),
            "success_rate": (
                self.scaler_metrics["successful_scalings"] / 
                max(1, self.scaler_metrics["scaling_events"])
            )
        }
    
    # 内部方法
    async def _scaling_loop(self):
        """扩缩容评估循环"""
        
        while True:
            try:
                await asyncio.sleep(self.evaluation_interval)
                await self._perform_scaling_evaluation()
                
            except asyncio.CancelledError:
                self.logger.info("Scaling evaluation loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in scaling evaluation loop: {e}")
    
    async def _perform_scaling_evaluation(self):
        """执行扩缩容评估"""
        
        try:
            # 获取所有分组
            topology = await self.cluster_manager.get_cluster_topology()
            
            for group_id in topology.groups.keys():
                try:
                    # 评估扩缩容需求
                    decision = await self.evaluate_scaling_need(group_id)
                    
                    # 如果需要扩缩容，执行操作
                    if decision.action != ScalingAction.NO_ACTION and decision.confidence > 0.7:
                        await self.execute_scaling_decision(group_id, decision)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating group {group_id}: {e}")
                    continue
            
            self.scaler_metrics["evaluations_performed"] += 1
            
        except Exception as e:
            self.logger.error(f"Error performing scaling evaluation: {e}")
    
    async def _collect_group_metrics(self, agents: List[AgentInfo]) -> Dict[str, float]:
        """收集分组的聚合指标"""
        
        if not agents:
            return {}
        
        metrics = {
            "cpu_usage_percent": 0.0,
            "memory_usage_percent": 0.0,
            "storage_usage_percent": 0.0,
            "gpu_usage_percent": 0.0,
            "active_tasks": 0.0,
            "total_requests": 0.0,
            "failed_requests": 0.0,
            "avg_response_time": 0.0,
            "error_rate": 0.0
        }
        
        # 聚合所有智能体的指标
        total_requests = 0
        total_failed = 0
        weighted_response_time = 0.0
        weight_sum = 0.0
        
        for agent in agents:
            usage = agent.resource_usage
            
            metrics["cpu_usage_percent"] += usage.cpu_usage_percent
            metrics["memory_usage_percent"] += usage.memory_usage_percent
            metrics["storage_usage_percent"] += usage.storage_usage_percent
            metrics["gpu_usage_percent"] += usage.gpu_usage_percent
            metrics["active_tasks"] += usage.active_tasks
            
            total_requests += usage.total_requests
            total_failed += usage.failed_requests
            
            # 加权平均响应时间
            if usage.total_requests > 0:
                weight = usage.total_requests
                weighted_response_time += usage.avg_response_time * weight
                weight_sum += weight
        
        # 计算平均值
        agent_count = len(agents)
        if agent_count > 0:
            metrics["cpu_usage_percent"] /= agent_count
            metrics["memory_usage_percent"] /= agent_count
            metrics["storage_usage_percent"] /= agent_count
            metrics["gpu_usage_percent"] /= agent_count
        
        metrics["total_requests"] = total_requests
        metrics["failed_requests"] = total_failed
        
        # 计算错误率和平均响应时间
        if total_requests > 0:
            metrics["error_rate"] = total_failed / total_requests
        
        if weight_sum > 0:
            metrics["avg_response_time"] = weighted_response_time / weight_sum
        
        return metrics
    
    async def _make_scaling_decision(
        self, 
        policy: ScalingPolicy, 
        current_instances: int,
        metrics: Dict[str, float]
    ) -> ScalingDecision:
        """制定扩缩容决策"""
        
        action = ScalingAction.NO_ACTION
        reason = ScalingReason.POLICY_VIOLATION
        confidence = 0.0
        target_instances = current_instances
        
        # 检查扩容条件
        scale_up_signals = []
        
        if policy.scale_up_enabled:
            # CPU使用率过高
            if metrics.get("cpu_usage_percent", 0) > policy.scale_up_cpu_threshold:
                scale_up_signals.append(("cpu", metrics["cpu_usage_percent"] / policy.scale_up_cpu_threshold))
                reason = ScalingReason.HIGH_CPU
            
            # 内存使用率过高
            if metrics.get("memory_usage_percent", 0) > policy.scale_up_memory_threshold:
                scale_up_signals.append(("memory", metrics["memory_usage_percent"] / policy.scale_up_memory_threshold))
                reason = ScalingReason.HIGH_MEMORY
            
            # 响应时间过长
            if metrics.get("avg_response_time", 0) > policy.scale_up_response_threshold:
                scale_up_signals.append(("response_time", metrics["avg_response_time"] / policy.scale_up_response_threshold))
                reason = ScalingReason.HIGH_RESPONSE_TIME
            
            # 错误率过高
            if metrics.get("error_rate", 0) > policy.max_error_rate:
                scale_up_signals.append(("error_rate", metrics["error_rate"] / policy.max_error_rate))
                reason = ScalingReason.HIGH_ERROR_RATE
        
        # 检查缩容条件
        scale_down_signals = []
        
        if policy.scale_down_enabled:
            # CPU使用率过低
            if (metrics.get("cpu_usage_percent", 100) < policy.scale_down_cpu_threshold and
                metrics.get("memory_usage_percent", 100) < policy.scale_down_memory_threshold and
                metrics.get("avg_response_time", 10000) < policy.scale_down_response_threshold):
                
                utilization_score = min(
                    metrics.get("cpu_usage_percent", 0) / policy.scale_down_cpu_threshold,
                    metrics.get("memory_usage_percent", 0) / policy.scale_down_memory_threshold
                )
                scale_down_signals.append(("utilization", 1.0 - utilization_score))
                reason = ScalingReason.LOW_UTILIZATION
        
        # 决策逻辑
        if scale_up_signals and current_instances < policy.max_instances:
            action = ScalingAction.SCALE_UP
            
            # 计算扩容实例数
            max_signal = max(signal[1] for signal in scale_up_signals)
            scale_factor = min(policy.scale_up_factor, 1.0 + max_signal * 0.5)
            target_instances = math.ceil(current_instances * scale_factor)
            
            # 置信度基于信号强度
            confidence = min(1.0, max_signal - 0.5)  # 超过阈值50%时置信度为1
            
        elif scale_down_signals and current_instances > policy.min_instances:
            action = ScalingAction.SCALE_DOWN
            
            # 计算缩容实例数
            max_signal = max(signal[1] for signal in scale_down_signals)
            scale_factor = max(policy.scale_down_factor, 1.0 - max_signal * 0.3)
            target_instances = math.floor(current_instances * scale_factor)
            
            # 缩容更保守，置信度要求更高
            confidence = min(0.8, max_signal)
        
        return ScalingDecision(
            action=action,
            reason=reason,
            current_instances=current_instances,
            target_instances=target_instances,
            confidence=confidence,
            metrics=metrics.copy(),
            policy_id=policy.policy_id
        )
    
    async def _is_in_cooldown(self, group_id: str, policy: ScalingPolicy) -> bool:
        """检查是否在冷却期内"""
        
        last_scaling = self.last_scaling_time.get(group_id, 0)
        cooldown_end = last_scaling + policy.cooldown_period_seconds
        
        return time.time() < cooldown_end
    
    async def _scale_up_group(self, group_id: str, decision: ScalingDecision) -> BatchOperationResult:
        """扩容分组"""
        
        try:
            instances_to_add = decision.target_instances - decision.current_instances
            
            self.logger.info(f"Scaling up group {group_id}: adding {instances_to_add} instances")
            
            # 创建新的智能体实例
            # 这里简化实现，实际应该根据分组配置创建智能体
            new_agents = []
            for i in range(instances_to_add):
                agent_id = f"{group_id}-scale-{int(time.time())}-{i}"
                # 这里需要根据分组模板创建智能体
                # 暂时创建一个占位符，实际实现需要集成智能体工厂
                new_agents.append(agent_id)
            
            # 批量启动智能体（模拟）
            # 实际实现需要调用lifecycle_manager的批量操作
            result = BatchOperationResult(
                total_count=instances_to_add,
                success_count=instances_to_add,
                failed_count=0,
                results={agent_id: None for agent_id in new_agents},
                operation_type="start",
                started_at=time.time(),
                completed_at=time.time(),
                batch_id=f"scaleup-{group_id}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error scaling up group {group_id}: {e}")
            raise
    
    async def _scale_down_group(self, group_id: str, decision: ScalingDecision) -> BatchOperationResult:
        """缩容分组"""
        
        try:
            instances_to_remove = decision.current_instances - decision.target_instances
            
            self.logger.info(f"Scaling down group {group_id}: removing {instances_to_remove} instances")
            
            # 获取分组中的智能体，选择要移除的实例
            topology = await self.cluster_manager.get_cluster_topology()
            group_agents = topology.get_group_agents(group_id)
            
            # 选择移除策略：优先移除利用率最低的实例
            agents_with_utilization = []
            for agent in group_agents:
                utilization = (
                    agent.resource_usage.cpu_usage_percent + 
                    agent.resource_usage.memory_usage_percent
                ) / 2.0
                agents_with_utilization.append((agent, utilization))
            
            # 按利用率排序，选择最低利用率的智能体
            agents_with_utilization.sort(key=lambda x: x[1])
            agents_to_remove = [
                agent.agent_id for agent, _ in agents_with_utilization[:instances_to_remove]
            ]
            
            # 批量停止智能体
            from .lifecycle_manager import AgentOperation
            result = await self.lifecycle_manager.batch_operation(
                agents_to_remove,
                AgentOperation.STOP,
                {"graceful": True}
            )
            
            return BatchOperationResult(
                total_count=len(agents_to_remove),
                success_count=sum(1 for r in result.values() if r),
                failed_count=sum(1 for r in result.values() if not r),
                results={agent_id: result.get(agent_id) for agent_id in agents_to_remove},
                operation_type="stop",
                started_at=time.time(),
                completed_at=time.time(),
                batch_id=f"scaledown-{group_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error scaling down group {group_id}: {e}")
            raise
