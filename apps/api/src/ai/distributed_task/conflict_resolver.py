"""冲突解决器实现"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from .models import Task, TaskStatus, Conflict, ConflictType
from src.core.utils.timezone_utils import utc_now
from src.core.utils.async_utils import create_task_with_logging

class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, state_manager=None, task_coordinator=None):
        self.state_manager = state_manager
        self.task_coordinator = task_coordinator
        self.logger = get_logger(__name__)
        
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
        create_task_with_logging(self._start_conflict_detection_loop())
    
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
            
            service_registry = getattr(self.task_coordinator, "service_registry", None) if self.task_coordinator else None
            if not service_registry:
                continue
            agent_meta = await service_registry.get_agent(agent_id)
            if not agent_meta or not getattr(agent_meta, "resources", None):
                continue
            resources = agent_meta.resources
            available = {
                "cpu": 1.0 - float(resources.get("cpu_usage", 0.0) or 0.0),
                "memory": float(resources.get("memory_total", 0) or 0) - float(resources.get("memory_used", 0) or 0),
                "disk_space": float(resources.get("disk_total", 0) or 0) - float(resources.get("disk_used", 0) or 0),
            }
            
            # 检查资源超限
            for resource, required in total_requirements.items():
                if required > available.get(resource, 0):
                    conflict = Conflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type=ConflictType.RESOURCE_CONFLICT,
                        description=f"Agent {agent_id} resource {resource} overcommitted: {required} > {available.get(resource, 0)}",
                        involved_tasks=[task.task_id for task in tasks],
                        involved_agents=[agent_id],
                        timestamp=utc_now()
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
                    timestamp=utc_now()
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
                    timestamp=utc_now()
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
                            timestamp=utc_now()
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
                            timestamp=utc_now()
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
        tasks = [t for t in await self._get_all_tasks() if t.task_id in set(conflict.involved_tasks)]
        if not tasks:
            return {}
        tasks.sort(key=lambda t: t.priority.value)
        keep_task_id = tasks[0].task_id
        return {
            "action": "reassign",
            "keep_task_id": keep_task_id,
            "tasks": [t.task_id for t in tasks[1:]],
        }
    
    async def _resource_optimization_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """资源优化解决策略"""
        tasks = [t for t in await self._get_all_tasks() if t.task_id in set(conflict.involved_tasks)]
        if not tasks:
            return {}
        tasks.sort(key=lambda t: sum(float(v or 0) for v in t.resource_requirements.values()), reverse=True)
        return {
            "action": "reassign",
            "tasks": [tasks[0].task_id],
        }
    
    async def _load_balancing_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """负载均衡解决策略"""
        tasks = [t for t in await self._get_all_tasks() if t.task_id in set(conflict.involved_tasks)]
        if not tasks:
            return {}
        agent_load: Dict[str, int] = {}
        for t in tasks:
            if t.assigned_to:
                agent_load[t.assigned_to] = agent_load.get(t.assigned_to, 0) + 1
        if not agent_load:
            return {"action": "reassign", "tasks": [t.task_id for t in tasks]}
        busiest = max(agent_load.items(), key=lambda x: x[1])[0]
        return {
            "action": "reassign",
            "tasks": [t.task_id for t in tasks if t.assigned_to == busiest],
        }
    
    async def _fairness_based_resolution(self, conflict: Conflict) -> Dict[str, Any]:
        """公平性解决策略"""
        tasks = [t for t in await self._get_all_tasks() if t.task_id in set(conflict.involved_tasks)]
        if not tasks:
            return {}
        tasks.sort(key=lambda t: t.created_at)
        return {"action": "reassign", "tasks": [tasks[-1].task_id]}
    
    async def _apply_resolution(self, conflict: Conflict, resolution: Dict[str, Any]) -> bool:
        """应用解决方案"""
        
        try:
            action = resolution.get("action")
            
            if action == "reassign":
                task_ids = resolution.get("tasks") or conflict.involved_tasks
                if not self.task_coordinator:
                    return False
                for task_id in task_ids:
                    ok = await self.task_coordinator.reassign_task(task_id)
                    if not ok:
                        return False
                return True

            return False
            
        except Exception as e:
            self.logger.error(f"Failed to apply resolution: {e}")
            return False
    
    async def _get_active_tasks(self) -> List[Task]:
        """获取活跃任务（简化实现）"""
        if self.task_coordinator:
            return list(self.task_coordinator.active_tasks.values())
        if not self.state_manager:
            return []
        tasks = []
        for v in getattr(self.state_manager, "global_state", {}).values():
            if isinstance(v, dict) and v.get("task_id"):
                tasks.append(Task.from_dict(v.copy()))
        return [t for t in tasks if t.status in {TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS}]
    
    async def _get_all_tasks(self) -> List[Task]:
        """获取所有任务（简化实现）"""
        if self.task_coordinator:
            tasks = []
            tasks.extend(self.task_coordinator.task_queue)
            tasks.extend(self.task_coordinator.active_tasks.values())
            tasks.extend(self.task_coordinator.completed_tasks.values())
            uniq: Dict[str, Task] = {}
            for t in tasks:
                uniq[t.task_id] = t
            return list(uniq.values())
        if not self.state_manager:
            return []
        tasks = []
        for v in getattr(self.state_manager, "global_state", {}).values():
            if isinstance(v, dict) and v.get("task_id"):
                tasks.append(Task.from_dict(v.copy()))
        return tasks
    
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
from src.core.logging import get_logger
