from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from .fault_detector import FaultEvent, FaultType, FaultSeverity
from src.ai.cluster.topology import AgentStatus
from src.ai.distributed_task.models import TaskStatus

from src.core.logging import get_logger
class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    IMMEDIATE_RESTART = "immediate_restart"      # 立即重启
    GRACEFUL_RESTART = "graceful_restart"       # 优雅重启
    TASK_MIGRATION = "task_migration"           # 任务迁移
    SERVICE_DEGRADATION = "service_degradation"  # 服务降级
    MANUAL_INTERVENTION = "manual_intervention"  # 手动干预

class RecoveryManager:
    """恢复管理器"""
    
    def __init__(
        self, 
        cluster_manager, 
        task_coordinator, 
        lifecycle_manager,
        config: Dict[str, Any]
    ):
        self.cluster_manager = cluster_manager
        self.task_coordinator = task_coordinator
        self.lifecycle_manager = lifecycle_manager
        self.config = config
        self.logger = get_logger(__name__)
        
        # 恢复策略配置
        self.recovery_strategies = {
            FaultType.AGENT_UNRESPONSIVE: [
                RecoveryStrategy.GRACEFUL_RESTART,
                RecoveryStrategy.IMMEDIATE_RESTART,
                RecoveryStrategy.TASK_MIGRATION
            ],
            FaultType.AGENT_ERROR: [
                RecoveryStrategy.GRACEFUL_RESTART,
                RecoveryStrategy.TASK_MIGRATION
            ],
            FaultType.PERFORMANCE_DEGRADATION: [
                RecoveryStrategy.TASK_MIGRATION,
                RecoveryStrategy.SERVICE_DEGRADATION
            ],
            FaultType.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.TASK_MIGRATION,
                RecoveryStrategy.GRACEFUL_RESTART
            ],
            FaultType.NETWORK_PARTITION: [
                RecoveryStrategy.SERVICE_DEGRADATION,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            FaultType.NODE_FAILURE: [
                RecoveryStrategy.TASK_MIGRATION,
                RecoveryStrategy.SERVICE_DEGRADATION
            ]
        }
        
        # 恢复历史
        self.recovery_history: List[Dict[str, Any]] = []
        
        # 恢复任务队列
        self.recovery_queue = asyncio.Queue()
        
        # 运行控制
        self.running = False
    
    async def start(self):
        """启动恢复管理器"""
        self.running = True
        
        # 启动恢复处理循环
        create_task_with_logging(self._recovery_processing_loop())
        
        self.logger.info("Recovery manager started")
    
    async def stop(self):
        """停止恢复管理器"""
        self.running = False
        self.logger.info("Recovery manager stopped")
    
    async def handle_fault_event(self, fault_event: FaultEvent):
        """处理故障事件"""
        
        try:
            self.logger.info(f"Handling fault event: {fault_event.fault_id}")
            
            # 加入恢复队列
            await self.recovery_queue.put(fault_event)
            
        except Exception as e:
            self.logger.error(f"Error handling fault event {fault_event.fault_id}: {e}")
    
    async def _recovery_processing_loop(self):
        """恢复处理循环"""
        
        while self.running:
            try:
                # 等待故障事件
                fault_event = await self.recovery_queue.get()
                
                # 执行恢复操作
                await self._execute_recovery(fault_event)
                
            except Exception as e:
                self.logger.error(f"Error in recovery processing loop: {e}")
    
    async def _execute_recovery(self, fault_event: FaultEvent):
        """执行恢复操作"""
        
        start_time = utc_now()
        recovery_success = False
        recovery_actions = []
        
        try:
            # 获取恢复策略
            strategies = self.recovery_strategies.get(
                fault_event.fault_type,
                [RecoveryStrategy.MANUAL_INTERVENTION]
            )
            
            # 按顺序尝试恢复策略
            for strategy in strategies:
                self.logger.info(f"Trying recovery strategy: {strategy.value} for fault {fault_event.fault_id}")
                
                success = await self._apply_recovery_strategy(fault_event, strategy)
                recovery_actions.append({
                    "strategy": strategy.value,
                    "success": success,
                    "timestamp": utc_now().isoformat()
                })
                
                if success:
                    recovery_success = True
                    self.logger.info(f"Recovery successful with strategy: {strategy.value}")
                    break
                else:
                    self.logger.warning(f"Recovery strategy {strategy.value} failed, trying next")
            
            # 更新故障事件状态
            if recovery_success:
                fault_event.resolved = True
                fault_event.resolved_at = utc_now()
                fault_event.recovery_actions = [action["strategy"] for action in recovery_actions if action["success"]]
            
            # 记录恢复历史
            recovery_record = {
                "fault_id": fault_event.fault_id,
                "fault_type": fault_event.fault_type.value,
                "affected_components": fault_event.affected_components,
                "recovery_start": start_time.isoformat(),
                "recovery_end": utc_now().isoformat(),
                "recovery_success": recovery_success,
                "recovery_actions": recovery_actions,
                "recovery_time": (utc_now() - start_time).total_seconds()
            }
            
            self.recovery_history.append(recovery_record)
            
            # 限制历史记录大小
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-500:]
            
            if not recovery_success:
                self.logger.error(f"All recovery strategies failed for fault {fault_event.fault_id}")
            
        except Exception as e:
            self.logger.error(f"Recovery execution failed for fault {fault_event.fault_id}: {e}")
    
    async def _apply_recovery_strategy(
        self, 
        fault_event: FaultEvent, 
        strategy: RecoveryStrategy
    ) -> bool:
        """应用恢复策略"""
        
        try:
            if strategy == RecoveryStrategy.IMMEDIATE_RESTART:
                return await self._immediate_restart_recovery(fault_event)
            
            elif strategy == RecoveryStrategy.GRACEFUL_RESTART:
                return await self._graceful_restart_recovery(fault_event)
            
            elif strategy == RecoveryStrategy.TASK_MIGRATION:
                return await self._task_migration_recovery(fault_event)
            
            elif strategy == RecoveryStrategy.SERVICE_DEGRADATION:
                return await self._service_degradation_recovery(fault_event)
            
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return await self._manual_intervention_recovery(fault_event)
            
            else:
                self.logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery strategy {strategy.value} execution error: {e}")
            return False
    
    async def _immediate_restart_recovery(self, fault_event: FaultEvent) -> bool:
        """立即重启恢复"""
        
        success_count = 0
        total_count = len(fault_event.affected_components)
        
        for component_id in fault_event.affected_components:
            if component_id == "system" or component_id == "network":
                continue  # 跳过系统级组件
            
            try:
                # 停止智能体
                stop_result = await self.lifecycle_manager.stop_agent(component_id, graceful=False)
                if not stop_result.success:
                    self.logger.warning(f"Failed to stop agent {component_id}: {stop_result.message}")
                    continue
                await asyncio.sleep(2)
                
                # 启动智能体
                start_result = await self.lifecycle_manager.start_agent(component_id)
                
                if start_result.success:
                    success_count += 1
                    self.logger.info(f"Successfully restarted agent {component_id}")
                else:
                    self.logger.warning(f"Failed to restart agent {component_id}: {start_result.message}")
                
            except Exception as e:
                self.logger.error(f"Immediate restart failed for {component_id}: {e}")
        
        # 如果大部分组件重启成功，认为恢复成功
        return success_count >= (total_count * 0.7)
    
    async def _graceful_restart_recovery(self, fault_event: FaultEvent) -> bool:
        """优雅重启恢复"""
        
        success_count = 0
        total_count = len(fault_event.affected_components)
        
        for component_id in fault_event.affected_components:
            if component_id == "system" or component_id == "network":
                continue
            
            try:
                # 优雅停止智能体
                stop_result = await self.lifecycle_manager.stop_agent(component_id, graceful=True)
                if not stop_result.success:
                    self.logger.warning(f"Failed to stop agent {component_id}: {stop_result.message}")
                    continue
                await asyncio.sleep(5)  # 等待更长时间
                
                # 启动智能体
                start_result = await self.lifecycle_manager.start_agent(component_id)
                
                if start_result.success:
                    success_count += 1
                    self.logger.info(f"Successfully gracefully restarted agent {component_id}")
                else:
                    self.logger.warning(f"Failed to gracefully restart agent {component_id}: {start_result.message}")
                
            except Exception as e:
                self.logger.error(f"Graceful restart failed for {component_id}: {e}")
        
        return success_count >= (total_count * 0.7)
    
    async def _task_migration_recovery(self, fault_event: FaultEvent) -> bool:
        """任务迁移恢复"""
        
        migration_success = True
        
        for component_id in fault_event.affected_components:
            if component_id == "system" or component_id == "network":
                continue
            
            try:
                # 获取智能体的活跃任务
                agent_tasks = await self._get_agent_active_tasks(component_id)
                
                if agent_tasks:
                    # 迁移任务到其他智能体
                    for task_id in agent_tasks:
                        success = await self.task_coordinator.reassign_task(task_id)
                        if not success:
                            migration_success = False
                            self.logger.warning(f"Failed to migrate task {task_id} from {component_id}")
                        else:
                            self.logger.info(f"Successfully migrated task {task_id} from {component_id}")
                
                # 将智能体标记为维护状态
                await self.cluster_manager.update_agent_status(
                    component_id, 
                    AgentStatus.MAINTENANCE
                )
                
            except Exception as e:
                self.logger.error(f"Task migration failed for {component_id}: {e}")
                migration_success = False
        
        return migration_success
    
    async def _service_degradation_recovery(self, fault_event: FaultEvent) -> bool:
        """服务降级恢复"""
        
        try:
            # 实现服务降级逻辑
            degradation_config = {
                "reduced_capacity": True,
                "non_critical_features_disabled": True,
                "request_rate_limiting": True,
                "fault_components": fault_event.affected_components
            }
            
            # 应用降级配置到集群
            await self._apply_degradation_config(degradation_config)
            
            self.logger.info(f"Service degradation applied for fault {fault_event.fault_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Service degradation failed: {e}")
            return False
    
    async def _manual_intervention_recovery(self, fault_event: FaultEvent) -> bool:
        """手动干预恢复"""
        
        # 创建手动干预任务
        intervention_task = {
            "fault_id": fault_event.fault_id,
            "fault_type": fault_event.fault_type.value,
            "severity": fault_event.severity.value,
            "affected_components": fault_event.affected_components,
            "description": fault_event.description,
            "created_at": utc_now().isoformat(),
            "status": "pending_manual_intervention"
        }
        
        # 这里应该将任务发送到运维系统或工单系统
        # 简化实现中只记录日志
        self.logger.critical(f"Manual intervention required for fault {fault_event.fault_id}: {intervention_task}")
        
        # 手动干预总是返回False，因为需要人工处理
        return False
    
    async def _get_agent_active_tasks(self, agent_id: str) -> List[str]:
        """获取智能体的活跃任务"""
        
        try:
            # 从任务协调器获取活跃任务
            tasks = await self.task_coordinator.get_agent_tasks(agent_id)
            active_statuses = {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.RETRY}
            return [task.task_id for task in tasks if task.status in active_statuses]
        except Exception as e:
            self.logger.error(f"Failed to get active tasks for agent {agent_id}: {e}")
            return []
    
    async def _apply_degradation_config(self, config: Dict[str, Any]):
        """应用服务降级配置"""
        
        try:
            # 这里应该实现具体的降级逻辑
            # 例如：限制请求速率、禁用非关键功能等
            self.logger.info(f"Applying degradation config: {config}")
            
            # 通知集群管理器应用降级配置
            if hasattr(self.cluster_manager, 'apply_degradation'):
                await self.cluster_manager.apply_degradation(config)
            
        except Exception as e:
            self.logger.error(f"Failed to apply degradation config: {e}")
            raise
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        
        if not self.recovery_history:
            return {
                "total_recoveries": 0,
                "success_rate": 0.0,
                "avg_recovery_time": 0.0,
                "strategy_success_rates": {}
            }
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = len([r for r in self.recovery_history if r["recovery_success"]])
        success_rate = successful_recoveries / total_recoveries
        
        total_recovery_time = sum(r["recovery_time"] for r in self.recovery_history)
        avg_recovery_time = total_recovery_time / total_recoveries
        
        # 统计各策略成功率
        strategy_stats = {}
        for record in self.recovery_history:
            for action in record["recovery_actions"]:
                strategy = action["strategy"]
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {"total": 0, "success": 0}
                
                strategy_stats[strategy]["total"] += 1
                if action["success"]:
                    strategy_stats[strategy]["success"] += 1
        
        strategy_success_rates = {}
        for strategy, stats in strategy_stats.items():
            strategy_success_rates[strategy] = stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "total_recoveries": total_recoveries,
            "success_rate": success_rate,
            "avg_recovery_time": avg_recovery_time,
            "strategy_success_rates": strategy_success_rates,
            "recent_recoveries": self.recovery_history[-10:]  # 最近10次恢复
        }
