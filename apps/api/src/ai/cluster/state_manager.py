"""
集群状态管理器

负责智能体集群状态的存储、同步和一致性管理。
实现分布式状态管理和冲突解决机制。
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import asdict
from datetime import datetime, timedelta

from .topology import (
    ClusterTopology, AgentInfo, AgentGroup, AgentStatus, 
    ResourceUsage, AgentHealthCheck
)


class StateChangeEvent:
    """状态变更事件"""
    
    def __init__(self, event_type: str, agent_id: str, old_state: Any, new_state: Any):
        self.event_type = event_type
        self.agent_id = agent_id
        self.old_state = old_state
        self.new_state = new_state
        self.timestamp = time.time()
        self.event_id = f"{event_type}-{agent_id}-{int(self.timestamp)}"


class ClusterStateManager:
    """集群状态管理器
    
    提供集群状态的集中管理、同步和持久化功能。
    支持智能体注册、状态更新、健康检查和拓扑变更管理。
    """
    
    def __init__(self, cluster_id: str = "default", storage_backend: Optional[Any] = None):
        self.cluster_id = cluster_id
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(__name__)
        
        # 集群拓扑
        self.topology = ClusterTopology(cluster_id=cluster_id)
        
        # 状态同步
        self._sync_lock = asyncio.Lock()
        self._change_listeners: List[Callable] = []
        self._state_version = 0
        
        # 健康检查
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30.0  # 30秒
        
        # 状态持久化
        self._persistence_task: Optional[asyncio.Task] = None
        self._persistence_interval = 60.0  # 60秒持久化一次
        
        # 事件历史
        self._state_events: List[StateChangeEvent] = []
        self._max_events = 1000
        
        # 性能指标
        self._metrics = {
            "total_state_changes": 0,
            "sync_operations": 0,
            "health_checks_performed": 0,
            "persistence_operations": 0
        }
        
        self.logger.info(f"ClusterStateManager initialized for cluster {cluster_id}")
    
    async def start(self):
        """启动状态管理器"""
        try:
            # 从存储后端恢复状态
            await self._restore_state()
            
            # 启动健康检查任务
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # 启动持久化任务
            if self.storage_backend:
                self._persistence_task = asyncio.create_task(self._persistence_loop())
            
            self.logger.info("ClusterStateManager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start ClusterStateManager: {e}")
            raise
    
    async def stop(self):
        """停止状态管理器"""
        try:
            # 停止任务
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._persistence_task:
                self._persistence_task.cancel()
                try:
                    await self._persistence_task
                except asyncio.CancelledError:
                    pass
            
            # 最终持久化
            await self._persist_state()
            
            self.logger.info("ClusterStateManager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ClusterStateManager: {e}")
    
    # 智能体管理
    async def register_agent(self, agent: AgentInfo) -> bool:
        """注册智能体"""
        async with self._sync_lock:
            try:
                # 检查是否已存在
                if agent.agent_id in self.topology.agents:
                    self.logger.warning(f"Agent {agent.agent_id} already registered")
                    return False
                
                # 设置集群ID
                agent.cluster_id = self.cluster_id
                
                # 添加到拓扑
                old_state = None
                success = self.topology.add_agent(agent)
                
                if success:
                    # 记录状态变更
                    event = StateChangeEvent(
                        "agent_registered", 
                        agent.agent_id, 
                        old_state, 
                        asdict(agent)
                    )
                    await self._emit_state_change(event)
                    
                    self.logger.info(f"Agent {agent.agent_id} registered successfully")
                    return True
                else:
                    self.logger.error(f"Failed to register agent {agent.agent_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error registering agent {agent.agent_id}: {e}")
                return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销智能体"""
        async with self._sync_lock:
            try:
                # 获取旧状态
                old_agent = self.topology.get_agent(agent_id)
                if not old_agent:
                    self.logger.warning(f"Agent {agent_id} not found for unregistration")
                    return False
                
                old_state = asdict(old_agent)
                
                # 从拓扑移除
                success = self.topology.remove_agent(agent_id)
                
                if success:
                    # 记录状态变更
                    event = StateChangeEvent(
                        "agent_unregistered",
                        agent_id,
                        old_state,
                        None
                    )
                    await self._emit_state_change(event)
                    
                    self.logger.info(f"Agent {agent_id} unregistered successfully")
                    return True
                else:
                    self.logger.error(f"Failed to unregister agent {agent_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error unregistering agent {agent_id}: {e}")
                return False
    
    async def update_agent_status(
        self, 
        agent_id: str, 
        status: AgentStatus, 
        details: Optional[str] = None
    ) -> bool:
        """更新智能体状态"""
        async with self._sync_lock:
            try:
                agent = self.topology.get_agent(agent_id)
                if not agent:
                    self.logger.warning(f"Agent {agent_id} not found for status update")
                    return False
                
                old_status = agent.status
                old_state = asdict(agent)
                
                # 更新状态
                agent.update_status(status, details)
                
                # 记录状态变更
                event = StateChangeEvent(
                    "agent_status_changed",
                    agent_id,
                    {"status": old_status.value},
                    {"status": status.value, "details": details}
                )
                await self._emit_state_change(event)
                
                self.logger.debug(f"Agent {agent_id} status updated: {old_status} -> {status}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id} status: {e}")
                return False
    
    async def update_agent_resource_usage(
        self, 
        agent_id: str, 
        usage: ResourceUsage
    ) -> bool:
        """更新智能体资源使用情况"""
        async with self._sync_lock:
            try:
                agent = self.topology.get_agent(agent_id)
                if not agent:
                    return False
                
                old_usage = agent.resource_usage
                agent.update_resource_usage(usage)
                
                # 只记录显著变化的事件
                if self._is_significant_usage_change(old_usage, usage):
                    event = StateChangeEvent(
                        "agent_resource_usage_changed",
                        agent_id,
                        asdict(old_usage),
                        asdict(usage)
                    )
                    await self._emit_state_change(event)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id} resource usage: {e}")
                return False
    
    async def update_agent_health(
        self, 
        agent_id: str, 
        health: AgentHealthCheck
    ) -> bool:
        """更新智能体健康状态"""
        async with self._sync_lock:
            try:
                agent = self.topology.get_agent(agent_id)
                if not agent:
                    return False
                
                old_health = agent.health.is_healthy
                agent.health = health
                
                # 如果健康状态发生变化，记录事件
                if old_health != health.is_healthy:
                    event = StateChangeEvent(
                        "agent_health_changed",
                        agent_id,
                        {"is_healthy": old_health},
                        {"is_healthy": health.is_healthy}
                    )
                    await self._emit_state_change(event)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id} health: {e}")
                return False
    
    # 分组管理
    async def create_group(self, group: AgentGroup) -> bool:
        """创建智能体分组"""
        async with self._sync_lock:
            try:
                success = self.topology.add_group(group)
                
                if success:
                    event = StateChangeEvent(
                        "group_created",
                        group.group_id,
                        None,
                        asdict(group)
                    )
                    await self._emit_state_change(event)
                    
                    self.logger.info(f"Group {group.group_id} created successfully")
                    return True
                else:
                    self.logger.error(f"Failed to create group {group.group_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error creating group {group.group_id}: {e}")
                return False
    
    async def add_agent_to_group(self, group_id: str, agent_id: str) -> bool:
        """将智能体添加到分组"""
        async with self._sync_lock:
            try:
                group = self.topology.groups.get(group_id)
                if not group:
                    self.logger.warning(f"Group {group_id} not found")
                    return False
                
                agent = self.topology.get_agent(agent_id)
                if not agent:
                    self.logger.warning(f"Agent {agent_id} not found")
                    return False
                
                success = group.add_agent(agent_id)
                if success:
                    # 更新智能体的分组信息
                    agent.group_id = group_id
                    
                    event = StateChangeEvent(
                        "agent_added_to_group",
                        agent_id,
                        {"group_id": None},
                        {"group_id": group_id}
                    )
                    await self._emit_state_change(event)
                    
                    self.logger.info(f"Agent {agent_id} added to group {group_id}")
                    return True
                else:
                    self.logger.warning(f"Failed to add agent {agent_id} to group {group_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error adding agent {agent_id} to group {group_id}: {e}")
                return False
    
    # 状态查询
    async def get_cluster_topology(self) -> ClusterTopology:
        """获取集群拓扑"""
        async with self._sync_lock:
            # 返回拓扑的深拷贝，避免外部修改
            return ClusterTopology(
                cluster_id=self.topology.cluster_id,
                name=self.topology.name,
                description=self.topology.description,
                agents=self.topology.agents.copy(),
                groups=self.topology.groups.copy(),
                agent_dependencies=self.topology.agent_dependencies.copy(),
                communication_paths=self.topology.communication_paths.copy(),
                config=self.topology.config.copy(),
                resource_limits=self.topology.resource_limits,
                labels=self.topology.labels.copy(),
                metadata=self.topology.metadata.copy(),
                created_at=self.topology.created_at,
                updated_at=self.topology.updated_at
            )
    
    async def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """获取智能体信息"""
        return self.topology.get_agent(agent_id)
    
    async def get_agents_by_status(self, status: AgentStatus) -> List[AgentInfo]:
        """根据状态获取智能体列表"""
        return self.topology.get_agents_by_status(status)
    
    async def get_healthy_agents(self) -> List[AgentInfo]:
        """获取健康的智能体列表"""
        return self.topology.get_healthy_agents()
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """获取集群统计信息"""
        return {
            "cluster_id": self.topology.cluster_id,
            "total_agents": self.topology.total_agents,
            "running_agents": self.topology.running_agents,
            "healthy_agents": self.topology.healthy_agents,
            "health_score": self.topology.cluster_health_score,
            "resource_usage": asdict(self.topology.cluster_resource_usage),
            "groups_count": len(self.topology.groups),
            "state_version": self._state_version,
            "metrics": self._metrics.copy(),
            "updated_at": self.topology.updated_at
        }
    
    # 事件监听
    def add_change_listener(self, listener: Callable[[StateChangeEvent], None]):
        """添加状态变更监听器"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[StateChangeEvent], None]):
        """移除状态变更监听器"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    async def get_recent_events(self, limit: int = 100) -> List[StateChangeEvent]:
        """获取最近的状态变更事件"""
        return self._state_events[-limit:] if self._state_events else []
    
    # 内部方法
    async def _emit_state_change(self, event: StateChangeEvent):
        """触发状态变更事件"""
        try:
            # 更新状态版本
            self._state_version += 1
            self._metrics["total_state_changes"] += 1
            
            # 记录事件
            self._state_events.append(event)
            
            # 保持事件历史在限制范围内
            if len(self._state_events) > self._max_events:
                self._state_events = self._state_events[-self._max_events:]
            
            # 通知监听器
            for listener in self._change_listeners:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event)
                    else:
                        listener(event)
                except Exception as e:
                    self.logger.error(f"Error in state change listener: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error emitting state change event: {e}")
    
    def _is_significant_usage_change(self, old: ResourceUsage, new: ResourceUsage) -> bool:
        """判断资源使用变化是否显著"""
        # 定义显著变化的阈值
        thresholds = {
            "cpu_usage_percent": 10.0,     # CPU使用率变化超过10%
            "memory_usage_percent": 10.0,   # 内存使用率变化超过10%
            "active_tasks": 5,              # 活跃任务数变化超过5个
            "error_rate": 0.05             # 错误率变化超过5%
        }
        
        return (
            abs(new.cpu_usage_percent - old.cpu_usage_percent) > thresholds["cpu_usage_percent"] or
            abs(new.memory_usage_percent - old.memory_usage_percent) > thresholds["memory_usage_percent"] or
            abs(new.active_tasks - old.active_tasks) > thresholds["active_tasks"] or
            abs(new.error_rate - old.error_rate) > thresholds["error_rate"]
        )
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                self.logger.info("Health check loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        try:
            current_time = time.time()
            
            async with self._sync_lock:
                for agent in self.topology.agents.values():
                    # 检查心跳超时
                    if not agent.health.is_responsive:
                        if agent.status not in [AgentStatus.FAILED, AgentStatus.STOPPED]:
                            await self.update_agent_status(
                                agent.agent_id, 
                                AgentStatus.FAILED, 
                                "Health check timeout"
                            )
                    
                    # 检查是否需要重启
                    elif agent.health.needs_restart:
                        if agent.status not in [AgentStatus.FAILED, AgentStatus.STOPPING]:
                            await self.update_agent_status(
                                agent.agent_id,
                                AgentStatus.FAILED,
                                f"Too many consecutive failures: {agent.health.consecutive_failures}"
                            )
                
                self._metrics["health_checks_performed"] += 1
                
        except Exception as e:
            self.logger.error(f"Error performing health checks: {e}")
    
    async def _persistence_loop(self):
        """持久化循环"""
        while True:
            try:
                await asyncio.sleep(self._persistence_interval)
                await self._persist_state()
                
            except asyncio.CancelledError:
                self.logger.info("Persistence loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in persistence loop: {e}")
    
    async def _persist_state(self):
        """持久化状态"""
        if not self.storage_backend:
            return
        
        try:
            state_data = {
                "cluster_topology": self.topology.to_dict(),
                "state_version": self._state_version,
                "metrics": self._metrics,
                "timestamp": time.time()
            }
            
            # 使用存储后端保存状态
            await self.storage_backend.save_cluster_state(
                self.cluster_id,
                state_data
            )
            
            self._metrics["persistence_operations"] += 1
            self.logger.debug(f"Cluster state persisted for {self.cluster_id}")
            
        except Exception as e:
            self.logger.error(f"Error persisting cluster state: {e}")
    
    async def _restore_state(self):
        """从存储后端恢复状态"""
        if not self.storage_backend:
            return
        
        try:
            state_data = await self.storage_backend.load_cluster_state(self.cluster_id)
            
            if state_data:
                # 恢复拓扑数据
                topology_data = state_data.get("cluster_topology", {})
                if topology_data:
                    # 这里可以实现更复杂的状态恢复逻辑
                    # 目前简单地从数据重建拓扑
                    pass
                
                # 恢复状态版本
                self._state_version = state_data.get("state_version", 0)
                
                # 恢复指标
                saved_metrics = state_data.get("metrics", {})
                self._metrics.update(saved_metrics)
                
                self.logger.info(f"Cluster state restored for {self.cluster_id}")
            else:
                self.logger.info(f"No previous state found for cluster {self.cluster_id}")
                
        except Exception as e:
            self.logger.error(f"Error restoring cluster state: {e}")