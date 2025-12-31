"""分布式任务协调引擎"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from .models import Task, TaskStatus, TaskPriority
from .raft_consensus import RaftConsensusEngine
from .task_decomposer import TaskDecomposer
from .intelligent_assigner import IntelligentAssigner
from .state_manager import DistributedStateManager
from .conflict_resolver import ConflictResolver
from src.core.utils.timezone_utils import utc_now
from src.core.utils.async_utils import create_task_with_logging

class DistributedTaskCoordinationEngine:
    """分布式任务协调引擎"""
    
    def __init__(
        self,
        node_id: str,
        cluster_nodes: List[str],
        message_bus=None,
        service_registry=None,
        load_balancer=None
    ):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.message_bus = message_bus
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.logger = get_logger(__name__)
        
        # 初始化各个组件
        self.raft_consensus = RaftConsensusEngine(
            node_id=node_id,
            cluster_nodes=cluster_nodes,
            message_bus=message_bus
        )
        
        self.task_decomposer = TaskDecomposer()
        
        self.intelligent_assigner = IntelligentAssigner(
            service_registry=service_registry,
            load_balancer=load_balancer
        )
        
        self.state_manager = DistributedStateManager(
            node_id=node_id,
            raft_consensus=self.raft_consensus
        )
        
        self.conflict_resolver = ConflictResolver(
            state_manager=self.state_manager,
            task_coordinator=self
        )
        
        # 任务队列和管理
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # 系统状态
        self.is_running = False
        self.processing_loop_task = None
        
        # 性能统计
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "average_processing_time": 0.0
        }
        
        # 设置Raft回调
        self.raft_consensus.apply_callback = self._handle_raft_command
        self.raft_consensus.state_change_callback = self._handle_state_change
    
    async def start(self):
        """启动协调引擎"""
        
        self.logger.info(f"Starting distributed task coordination engine on node {self.node_id}")
        
        # 启动Raft共识
        await self.raft_consensus.start()
        
        # 启动任务处理循环
        self.is_running = True
        self.processing_loop_task = create_task_with_logging(self._task_processing_loop())
        
        self.logger.info("Task coordination engine started")
    
    async def stop(self):
        """停止协调引擎"""
        
        self.logger.info("Stopping distributed task coordination engine")
        
        self.is_running = False
        
        # 停止处理循环
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                raise
        
        # 停止Raft共识
        await self.raft_consensus.stop()
        
        self.logger.info("Task coordination engine stopped")
    
    async def submit_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        requirements: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """提交任务"""
        requirements = requirements or {}
        resource_keys = {"cpu", "memory", "disk_space", "gpu"}
        resource_requirements = {k: requirements[k] for k in resource_keys if k in requirements}
        task_requirements = requirements.copy()
        for k in resource_requirements:
            task_requirements.pop(k, None)

        # 创建任务
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            task_type=task_type,
            data=task_data,
            requirements=task_requirements,
            priority=priority,
            created_at=utc_now(),
            resource_requirements=resource_requirements,
        )
        
        # 通过Raft共识添加任务
        command = {
            "action": "submit_task",
            "task": task.to_dict(),
            "client_id": self.node_id,
            "sequence_number": self.stats["tasks_submitted"]
        }
        
        success = await self.raft_consensus.append_entry(command)
        
        if success:
            self.stats["tasks_submitted"] += 1
            self.logger.info(f"Task {task_id} submitted successfully")
            return task_id
        else:
            self.logger.error(f"Failed to submit task {task_id}")
            return ""
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        
        # 通过Raft共识取消任务
        command = {
            "action": "cancel_task",
            "task_id": task_id,
            "client_id": self.node_id
        }
        
        return await self.raft_consensus.append_entry(command)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        
        # 从active_tasks或completed_tasks获取
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
        else:
            # 从状态管理器获取
            task_state = await self.state_manager.get_global_state(f"task_{task_id}")
            if task_state:
                return task_state
            return {"status": "not_found"}
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error
        }
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        
        return {
            "node_id": self.node_id,
            "raft_state": self.raft_consensus.state.value,
            "leader_id": self.raft_consensus.leader_id,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queued_tasks": len(self.task_queue),
            "stats": self.stats,
            "state_summary": await self.state_manager.get_state_summary()
        }

    async def get_agent_tasks(self, agent_id: str) -> List[Task]:
        return [task for task in self.active_tasks.values() if task.assigned_to == agent_id]

    async def reassign_task(self, task_id: str) -> bool:
        task = self.active_tasks.get(task_id)
        if not task:
            return False
        new_agent_id = await self.intelligent_assigner.reassign_task(task, "failure")
        if not new_agent_id:
            return False
        await self.state_manager.set_global_state(f"task_{task.task_id}", task.to_dict())
        return True
    
    async def _task_processing_loop(self):
        """任务处理循环"""
        
        while self.is_running:
            try:
                # 处理队列中的任务
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    await self._process_task(task)
                
                # 检查超时任务
                await self._check_task_timeouts()
                
                # 小延迟避免空转
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
    
    async def _process_task(self, task: Task):
        """处理一个任务"""
        
        try:
            # 分解任务
            if task.requirements.get("decompose", False):
                subtasks = await self.task_decomposer.decompose_task(task)
                
                if subtasks:
                    # 将子任务加入队列
                    for subtask in subtasks:
                        self.task_queue.append(subtask)
                        await self.state_manager.set_global_state(
                            f"task_{subtask.task_id}",
                            subtask.to_dict()
                        )
                    
                    # 更新父任务状态
                    task.status = TaskStatus.DECOMPOSED
                    await self.state_manager.set_global_state(
                        f"task_{task.task_id}",
                        task.to_dict()
                    )
                    return
            
            # 分配任务
            agent_id = await self.intelligent_assigner.assign_task(
                task,
                strategy=task.requirements.get("assignment_strategy", "capability_based")
            )
            
            if agent_id:
                # 更新任务状态
                task.assigned_to = agent_id
                task.status = TaskStatus.ASSIGNED
                task.started_at = utc_now()
                
                # 移动到active_tasks
                self.active_tasks[task.task_id] = task
                
                # 更新全局状态
                await self.state_manager.set_global_state(
                    f"task_{task.task_id}",
                    task.to_dict()
                )
                
                self.logger.info(f"Task {task.task_id} assigned to agent {agent_id}")
            else:
                # 分配失败，重新加入队列
                self.task_queue.append(task)
                self.logger.warning(f"Failed to assign task {task.task_id}, will retry")
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.completed_tasks[task.task_id] = task
            self.stats["tasks_failed"] += 1
    
    async def _check_task_timeouts(self):
        """检查任务超时"""
        
        current_time = utc_now()
        
        for task_id, task in list(self.active_tasks.items()):
            if task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                
                if elapsed > task.timeout_seconds:
                    self.logger.warning(f"Task {task_id} timed out")
                    
                    # 重新分配任务
                    await self.intelligent_assigner.reassign_task(task, "timeout")
    
    async def _handle_task_completion(self, task_id: str, result: Dict[str, Any]):
        """处理任务完成"""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = utc_now()
            task.result = result
            
            # 移动到completed_tasks
            del self.active_tasks[task_id]
            self.completed_tasks[task_id] = task
            
            # 更新统计
            self.stats["tasks_completed"] += 1
            
            # 更新平均处理时间
            if task.started_at:
                processing_time = (task.completed_at - task.started_at).total_seconds()
                current_avg = self.stats["average_processing_time"]
                total_completed = self.stats["tasks_completed"]
                self.stats["average_processing_time"] = (
                    (current_avg * (total_completed - 1) + processing_time) / total_completed
                )
            
            # 更新全局状态
            await self.state_manager.set_global_state(
                f"task_{task_id}",
                task.to_dict()
            )
            
            # 释放智能体
            if task.assigned_to:
                await self.intelligent_assigner.release_agent(task.assigned_to)
            
            self.logger.info(f"Task {task_id} completed successfully")
    
    async def _handle_raft_command(self, entry):
        """处理Raft命令"""
        
        command = entry.command_data
        action = command.get("action")
        
        if action == "submit_task":
            # 添加任务到队列
            task_dict = command["task"]
            task = Task.from_dict(task_dict)
            self.task_queue.append(task)
            
            # 存储到状态管理器
            await self.state_manager.set_global_state(
                f"task_{task.task_id}",
                task_dict
            )
            
        elif action == "cancel_task":
            # 取消任务
            task_id = command["task_id"]
            
            # 从队列中移除
            self.task_queue = [t for t in self.task_queue if t.task_id != task_id]
            
            # 如果在active_tasks中，标记为取消
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self.active_tasks[task_id]
                self.completed_tasks[task_id] = task
                self.stats["tasks_cancelled"] += 1
            
            # 更新全局状态
            await self.state_manager.set_global_state(
                f"task_{task_id}_cancelled",
                {"cancelled": True, "timestamp": utc_now().isoformat()}
            )
        
        elif action == "update_task_status":
            # 更新任务状态
            task_id = command["task_id"]
            new_status = command["status"]
            
            if task_id in self.active_tasks:
                self.active_tasks[task_id].status = TaskStatus(new_status)
    
    async def _handle_state_change(self, new_state):
        """处理Raft状态变化"""
        
        self.logger.info(f"Raft state changed to {new_state.value}")
        
        # 如果成为Leader，可能需要恢复任务处理
        # 如果成为Follower，可能需要暂停某些操作
from src.core.logging import get_logger
