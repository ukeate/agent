"""冲突解决器实现"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from .models import Task, TaskStatus, Conflict, ConflictType


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, state_manager=None, task_coordinator=None):
        self.state_manager = state_manager
        self.task_coordinator = task_coordinator
        self.logger = logging.getLogger(__name__)
        
        # 冲突检测器
        self.conflict_detectors = {
            ConflictType.RESOURCE_CONFLICT: self._detect_resource_conflicts,
            ConflictType.STATE_CONFLICT: self._detect_state_conflicts,
            ConflictType.ASSIGNMENT_CONFLICT: self._detect_assignment_conflicts,
            ConflictType.DEPENDENCY_CONFLICT: self._detect_dependency_conflicts
        }
        
        # 冲突解决策略
        self.resolution_strategies = {
            "priority_based": self._priority_based_resolution,
            "resource_optimization": self._resource_optimization_resolution,
            "load_balancing": self._load_balancing_resolution,
            "fairness": self._fairness_based_resolution
        }
        
        # 冲突历史
        self.conflict_history: List[Conflict] = []
        
        # 启动冲突检测循环
        asyncio.create_task(self._start_conflict_detection_loop())
    
    async def detect_conflicts(self) -> List[Conflict]:
        """检测所有类型的冲突"""
        
        all_conflicts = []
        
        for conflict_type, detector in self.conflict_detectors.items():
            try:
                conflicts = await detector()
                all_conflicts.extend(conflicts)
                
            except Exception as e:
                self.logger.error(f"Error detecting {conflict_type.value} conflicts: {e}")
        
        return all_conflicts
    
    async def resolve_conflict(
        self, 
        conflict: Conflict, 
        strategy: str = "priority_based"
    ) -> bool:
        """解决单个冲突"""
        
        try:
            if strategy not in self.resolution_strategies:
                self.logger.warning(f"Unknown resolution strategy: {strategy}, using priority_based")
                strategy = "priority_based"
            
            resolution_result = await self.resolution_strategies[strategy](conflict)
            
            if resolution_result:
                conflict.resolved = True
                conflict.resolution_strategy = strategy
                conflict.resolution_result = resolution_result
                
                # 应用解决方案
                success = await self._apply_resolution(conflict, resolution_result)
                
                if success:
                    self.logger.info(f"Conflict {conflict.conflict_id} resolved successfully")
                    self.conflict_history.append(conflict)
                    return True
                else:
                    self.logger.error(f"Failed to apply resolution for conflict {conflict.conflict_id}")
                    conflict.resolved = False
                    return False
            else:
                self.logger.warning(f"No resolution found for conflict {conflict.conflict_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
            return False
    
    async def _detect_resource_conflicts(self) -> List[Conflict]:
        """检测资源冲突"""
        
        conflicts = []
        
        # 获取所有活跃任务
        active_tasks = await self._get_active_tasks()
        
        # 按智能体分组
        agent_tasks: Dict[str, List[Task]] = {}
        for task in active_tasks:
            if task.assigned_to:
                if task.assigned_to not in agent_tasks:
                    agent_tasks[task.assigned_to] = []
                agent_tasks[task.assigned_to].append(task)
        
        # 检查每个智能体的资源冲突
        for agent_id, tasks in agent_tasks.items():
            total_requirements = {"cpu": 0, "memory": 0, "disk_space": 0}
            
            for task in tasks:
                for resource, amount in task.resource_requirements.items():
                    if resource in total_requirements:
                        total_requirements[resource] += amount
            
            # 获取智能体资源容量（简化实现）
            agent_capacity = {"cpu": 1.0, "memory": 8192, "disk_space": 102400}  # 示例值
            
            # 检查资源超限
            for resource, required in total_requirements.items():
                available = agent_capacity.get(resource, 0)
                if required > available:
                    conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type=ConflictType.RESOURCE_CONFLICT,
                        description=f"Agent {agent_id} resource {resource} overcommitted: {required} > {available}",
                        involved_tasks=[task.task_id for task in tasks],
                        involved_agents=[agent_id],
                        timestamp=datetime.now()
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_state_conflicts(self) -> List[Conflict]:
        """检测状态冲突"""
        
        conflicts = []
        
        # 获取所有任务状态
        all_tasks = await self._get_all_tasks()
        
        for task in all_tasks:
            # 检查状态一致性
            state_issues = []
            
            # 检查分配状态一致性
            if task.status == TaskStatus.ASSIGNED and not task.assigned_to:
                state_issues.append("Task marked as assigned but no agent assigned")
            
            if task.assigned_to and task.status == TaskStatus.PENDING:
                state_issues.append("Agent assigned but task still pending")
            
            # 检查时间逻辑一致性
            if task.started_at and task.completed_at:
                if task.started_at >= task.completed_at:
                    state_issues.append("Start time is after completion time")
            
            # 如果发现问题，创建冲突
            if state_issues:
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.STATE_CONFLICT,
                    description=f"Task {task.task_id} state inconsistencies: {', '.join(state_issues)}",
                    involved_tasks=[task.task_id],
                    involved_agents=[task.assigned_to] if task.assigned_to else [],
                    timestamp=datetime.now()
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_assignment_conflicts(self) -> List[Conflict]:
        """检测分配冲突"""
        
        conflicts = []
        
        # 检查重复分配
        assignment_map: Dict[str, List[str]] = {}  # task_id -> [agent_ids]
        
        active_tasks = await self._get_active_tasks()
        
        for task in active_tasks:
            if task.assigned_to:
                if task.task_id not in assignment_map:
                    assignment_map[task.task_id] = []
                assignment_map[task.task_id].append(task.assigned_to)
        
        for task_id, agent_ids in assignment_map.items():
            if len(agent_ids) > 1:
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.ASSIGNMENT_CONFLICT,
                    description=f"Task {task_id} assigned to multiple agents: {agent_ids}",
                    involved_tasks=[task_id],
                    involved_agents=agent_ids,
                    timestamp=datetime.now()
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_dependency_conflicts(self) -> List[Conflict]:
        """检测依赖冲突"""
        
        conflicts = []
        
        all_tasks = await self._get_all_tasks()
        task_map = {task.task_id: task for task in all_tasks}
        
        for task in all_tasks:
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in task_map:
                        conflict = Conflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                            description=f"Task {task.task_id} depends on non-existent task {dep_id}",
                            involved_tasks=[task.task_id],
                            involved_agents=[task.assigned_to] if task.assigned_to else [],
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
                        continue
                    
                    # 检查循环依赖
                    if self._has_circular_dependency(task, task_map, set()):
                        conflict = Conflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                            description=f"Circular dependency detected involving task {task.task_id}",
                            involved_tasks=[task.task_id] + task.dependencies,
                            involved_agents=[],
                            timestamp=datetime.now()
                        )
                        conflicts.append(conflict)
                        break
        
        return conflicts
    
    def _has_circular_dependency(
        self, 
        task: Task, 
        task_map: Dict[str, Task], 
        visited: Set[str]
    ) -> bool:
        """检查循环依赖"""
        
        if task.task_id in visited:
            return True
        
        visited.add(task.task_id)
        
        for dep_id in task.dependencies:
            if dep_id in task_map:
                if self._has_circular_dependency(task_map[dep_id], task_map, visited.copy()):
                    return True
        
        return False
    
    async def _priority_based_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """基于优先级的解决策略"""
        
        # 简化实现：选择优先级最高的任务
        resolution = {
            "action": "reassign",
            "details": "Reassign lower priority tasks"
        }
        
        return resolution
    
    async def _resource_optimization_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """资源优化解决策略"""
        
        resolution = {
            "action": "reschedule",
            "details": "Reschedule tasks to optimize resource usage"
        }
        
        return resolution
    
    async def _load_balancing_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """负载均衡解决策略"""
        
        resolution = {
            "action": "redistribute",
            "details": "Redistribute tasks across agents"
        }
        
        return resolution
    
    async def _fairness_based_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """公平性解决策略"""
        
        resolution = {
            "action": "fair_allocation",
            "details": "Allocate resources fairly among tasks"
        }
        
        return resolution
    
    async def _apply_resolution(self, conflict: Conflict, resolution: Dict[str, Any]) -> bool:
        """应用解决方案"""
        
        try:
            action = resolution.get("action")
            
            if action == "reassign":
                # 重新分配任务
                for task_id in conflict.involved_tasks:
                    # 实际实现需要调用任务协调器重新分配
                    self.logger.info(f"Reassigning task {task_id}")
            
            elif action == "reschedule":
                # 重新调度任务
                self.logger.info("Rescheduling tasks")
            
            elif action == "redistribute":
                # 重新分配任务到不同智能体
                self.logger.info("Redistributing tasks")
            
            elif action == "fair_allocation":
                # 公平分配资源
                self.logger.info("Fair allocation of resources")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply resolution: {e}")
            return False
    
    async def _get_active_tasks(self) -> List[Task]:
        """获取活跃任务（简化实现）"""
        
        # 实际实现应该从状态管理器获取
        return []
    
    async def _get_all_tasks(self) -> List[Task]:
        """获取所有任务（简化实现）"""
        
        # 实际实现应该从状态管理器获取
        return []
    
    async def _start_conflict_detection_loop(self):
        """启动冲突检测循环"""
        
        detection_interval = 10.0  # 秒
        
        while True:
            try:
                conflicts = await self.detect_conflicts()
                
                for conflict in conflicts:
                    # 自动尝试解决冲突
                    await self.resolve_conflict(conflict)
                
                await asyncio.sleep(detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in conflict detection loop: {e}")
                await asyncio.sleep(detection_interval)