"""
工作流任务调度器
基于Redis实现分布式任务队列和调度
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import uuid4
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from pydantic import BaseModel

from models.schemas.workflow import (
    WorkflowExecution, WorkflowStep, WorkflowStepExecution, 
    WorkflowStepStatus, TaskPriority
)
from src.ai.dag.task_planner import TaskPlanner, SchedulingStrategy, ExecutionPlan
from src.core.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """任务状态"""
    QUEUED = "queued"
    PENDING = "pending" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class QueuePriority(int, Enum):
    """队列优先级"""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


@dataclass
class ScheduledTask:
    """调度任务"""
    id: str
    execution_id: str
    step_id: str
    workflow_definition_id: str
    priority: int
    dependencies: List[str]
    estimated_duration: int  # 分钟
    max_retries: int
    retry_count: int
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.QUEUED
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理datetime序列化
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        """从字典创建实例"""
        # 处理datetime反序列化
        for key in ['scheduled_at', 'started_at', 'completed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


class TaskQueue:
    """Redis任务队列"""
    
    def __init__(self, redis_client: redis.Redis, queue_name: str = "workflow_tasks"):
        self.redis = redis_client
        self.queue_name = queue_name
        self.priority_queue_name = f"{queue_name}:priority"
        self.processing_queue_name = f"{queue_name}:processing"
        self.completed_queue_name = f"{queue_name}:completed"
        self.failed_queue_name = f"{queue_name}:failed"
        
        # 键前缀
        self.task_key_prefix = f"{queue_name}:task:"
        self.execution_key_prefix = f"{queue_name}:execution:"
        self.worker_key_prefix = f"{queue_name}:worker:"
    
    async def enqueue_task(self, task: ScheduledTask) -> bool:
        """将任务加入队列"""
        try:
            # 保存任务详情
            task_key = f"{self.task_key_prefix}{task.id}"
            await self.redis.hset(task_key, mapping=task.to_dict())
            
            # 根据优先级加入不同队列
            if task.priority >= QueuePriority.HIGH:
                await self.redis.lpush(f"{self.priority_queue_name}:high", task.id)
            elif task.priority >= QueuePriority.NORMAL:
                await self.redis.lpush(f"{self.priority_queue_name}:normal", task.id)
            else:
                await self.redis.lpush(f"{self.priority_queue_name}:low", task.id)
            
            # 更新执行统计
            execution_key = f"{self.execution_key_prefix}{task.execution_id}"
            await self.redis.hincrby(execution_key, "total_tasks", 1)
            await self.redis.hincrby(execution_key, "queued_tasks", 1)
            
            logger.info(f"任务已入队: {task.id}, 优先级: {task.priority}")
            return True
            
        except Exception as e:
            logger.error(f"任务入队失败: {task.id}, 错误: {e}")
            return False
    
    async def dequeue_task(self, worker_id: str, timeout: int = 10) -> Optional[ScheduledTask]:
        """从队列中取出任务"""
        try:
            # 按优先级从高到低尝试获取任务
            queue_names = [
                f"{self.priority_queue_name}:high",
                f"{self.priority_queue_name}:normal", 
                f"{self.priority_queue_name}:low"
            ]
            
            for queue_name in queue_names:
                # 使用阻塞弹出
                result = await self.redis.brpop(queue_name, timeout=timeout)
                if result:
                    _, task_id = result
                    task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                    
                    # 获取任务详情
                    task = await self.get_task(task_id)
                    if task:
                        # 标记任务为运行中
                        task.status = TaskStatus.RUNNING
                        task.worker_id = worker_id
                        task.started_at = utc_now()
                        
                        # 更新任务状态
                        await self.update_task(task)
                        
                        # 移动到处理队列
                        await self.redis.lpush(self.processing_queue_name, task_id)
                        
                        # 更新执行统计
                        execution_key = f"{self.execution_key_prefix}{task.execution_id}"
                        await self.redis.hincrby(execution_key, "queued_tasks", -1)
                        await self.redis.hincrby(execution_key, "running_tasks", 1)
                        
                        logger.info(f"任务出队: {task_id}, worker: {worker_id}")
                        return task
            
            return None
            
        except Exception as e:
            logger.error(f"任务出队失败, worker: {worker_id}, 错误: {e}")
            return None
    
    async def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """标记任务完成"""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.completed_at = utc_now()
            if result:
                task.metadata = task.metadata or {}
                task.metadata['result'] = result
            
            await self.update_task(task)
            
            # 从处理队列移动到完成队列
            await self.redis.lrem(self.processing_queue_name, 1, task_id)
            await self.redis.lpush(self.completed_queue_name, task_id)
            
            # 更新执行统计
            execution_key = f"{self.execution_key_prefix}{task.execution_id}"
            await self.redis.hincrby(execution_key, "running_tasks", -1)
            await self.redis.hincrby(execution_key, "completed_tasks", 1)
            
            logger.info(f"任务完成: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"标记任务完成失败: {task_id}, 错误: {e}")
            return False
    
    async def fail_task(self, task_id: str, error_message: str) -> bool:
        """标记任务失败"""
        try:
            task = await self.get_task(task_id)
            if not task:
                return False
            
            task.retry_count += 1
            task.error_message = error_message
            
            # 检查是否需要重试
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                # 重新入队，延迟执行
                delay_seconds = min(2 ** task.retry_count, 300)  # 指数退避，最大5分钟
                task.scheduled_at = utc_now() + timedelta(seconds=delay_seconds)
                
                await self.update_task(task)
                await self.redis.lrem(self.processing_queue_name, 1, task_id)
                
                # 延迟重新入队
                await asyncio.sleep(1)  # 短暂延迟
                await self.enqueue_task(task)
                
                logger.info(f"任务重试: {task_id}, 第{task.retry_count}次, 延迟{delay_seconds}秒")
            else:
                # 超过最大重试次数，标记为失败
                task.status = TaskStatus.FAILED
                task.completed_at = utc_now()
                
                await self.update_task(task)
                await self.redis.lrem(self.processing_queue_name, 1, task_id)
                await self.redis.lpush(self.failed_queue_name, task_id)
                
                # 更新执行统计
                execution_key = f"{self.execution_key_prefix}{task.execution_id}"
                await self.redis.hincrby(execution_key, "running_tasks", -1)
                await self.redis.hincrby(execution_key, "failed_tasks", 1)
                
                logger.error(f"任务失败: {task_id}, 超过最大重试次数")
            
            return True
            
        except Exception as e:
            logger.error(f"标记任务失败失败: {task_id}, 错误: {e}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务详情"""
        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            task_data = await self.redis.hgetall(task_key)
            
            if not task_data:
                return None
            
            # 转换字节串为字符串
            task_dict = {}
            for key, value in task_data.items():
                key = key.decode() if isinstance(key, bytes) else key
                value = value.decode() if isinstance(value, bytes) else value
                
                # 尝试解析JSON
                if key in ['metadata', 'dependencies']:
                    try:
                        value = json.loads(value) if value else ([] if key == 'dependencies' else {})
                    except json.JSONDecodeError:
                        value = [] if key == 'dependencies' else {}
                elif key in ['priority', 'estimated_duration', 'max_retries', 'retry_count']:
                    value = int(value)
                
                task_dict[key] = value
            
            return ScheduledTask.from_dict(task_dict)
            
        except Exception as e:
            logger.error(f"获取任务失败: {task_id}, 错误: {e}")
            return None
    
    async def update_task(self, task: ScheduledTask) -> bool:
        """更新任务状态"""
        try:
            task_key = f"{self.task_key_prefix}{task.id}"
            task_data = task.to_dict()
            
            # 序列化复杂字段
            for key in ['metadata', 'dependencies']:
                if key in task_data:
                    task_data[key] = json.dumps(task_data[key])
            
            await self.redis.hset(task_key, mapping=task_data)
            return True
            
        except Exception as e:
            logger.error(f"更新任务失败: {task.id}, 错误: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """获取队列统计信息"""
        try:
            stats = {}
            
            # 各优先级队列长度
            stats['high_priority'] = await self.redis.llen(f"{self.priority_queue_name}:high")
            stats['normal_priority'] = await self.redis.llen(f"{self.priority_queue_name}:normal")
            stats['low_priority'] = await self.redis.llen(f"{self.priority_queue_name}:low")
            
            # 处理状态队列长度
            stats['processing'] = await self.redis.llen(self.processing_queue_name)
            stats['completed'] = await self.redis.llen(self.completed_queue_name)
            stats['failed'] = await self.redis.llen(self.failed_queue_name)
            
            # 总计
            stats['total_queued'] = stats['high_priority'] + stats['normal_priority'] + stats['low_priority']
            stats['total_tasks'] = sum(stats.values())
            
            return stats
            
        except Exception as e:
            logger.error(f"获取队列统计失败: {e}")
            return {}


class WorkflowScheduler:
    """工作流调度器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.task_queue = TaskQueue(redis_client)
        self.task_planner = TaskPlanner()
        self.active_workers: Dict[str, Dict[str, Any]] = {}
        self.scheduling_strategies = {
            'fifo': self._schedule_fifo,
            'priority': self._schedule_priority,
            'critical_path': self._schedule_critical_path,
            'resource_aware': self._schedule_resource_aware
        }
    
    async def schedule_workflow_execution(
        self, 
        execution: WorkflowExecution,
        steps: List[WorkflowStep],
        strategy: str = 'critical_path'
    ) -> List[ScheduledTask]:
        """
        调度工作流执行
        
        Args:
            execution: 工作流执行实例
            steps: 工作流步骤列表
            strategy: 调度策略
            
        Returns:
            调度任务列表
        """
        try:
            logger.info(f"开始调度工作流: {execution.id}, 策略: {strategy}")
            
            # 构建DAG和执行计划
            from models.schemas.workflow import TaskDAG, TaskNode
            
            # 将WorkflowStep转换为TaskNode
            task_nodes = []
            for step in steps:
                task_node = TaskNode(
                    id=step.id,
                    name=step.name,
                    description=step.description or "",
                    task_type=step.step_type,
                    dependencies=step.dependencies,
                    complexity_score=5.0,  # 默认复杂度
                    estimated_duration_minutes=step.timeout_seconds // 60 if step.timeout_seconds else 10,
                    priority=TaskPriority.NORMAL
                )
                task_nodes.append(task_node)
            
            # 创建TaskDAG
            dag = TaskDAG(
                id=str(uuid4()),
                name=f"Workflow {execution.id}",
                description="Auto-generated from workflow execution",
                nodes=task_nodes
            )
            
            # 构建图并创建执行计划
            graph = self.task_planner.build_graph(dag)
            execution_plan = self.task_planner.create_execution_plan(
                graph, 
                SchedulingStrategy.CRITICAL_PATH
            )
            
            # 根据策略调度任务
            scheduler_func = self.scheduling_strategies.get(strategy, self._schedule_priority)
            scheduled_tasks = await scheduler_func(execution, steps, execution_plan)
            
            # 将任务加入队列
            for task in scheduled_tasks:
                await self.task_queue.enqueue_task(task)
            
            logger.info(f"工作流调度完成: {execution.id}, 共{len(scheduled_tasks)}个任务")
            return scheduled_tasks
            
        except Exception as e:
            logger.error(f"工作流调度失败: {execution.id}, 错误: {e}")
            return []
    
    async def _schedule_fifo(
        self, 
        execution: WorkflowExecution,
        steps: List[WorkflowStep],
        execution_plan: ExecutionPlan
    ) -> List[ScheduledTask]:
        """先进先出调度"""
        scheduled_tasks = []
        base_time = utc_now()
        
        for i, step in enumerate(steps):
            task = ScheduledTask(
                id=str(uuid4()),
                execution_id=execution.id,
                step_id=step.id,
                workflow_definition_id=execution.workflow_definition_id,
                priority=QueuePriority.NORMAL,
                dependencies=step.dependencies,
                estimated_duration=step.timeout_seconds // 60 if step.timeout_seconds else 10,
                max_retries=step.retry_count or 3,
                retry_count=0,
                scheduled_at=base_time + timedelta(seconds=i)  # 简单的时间偏移
            )
            scheduled_tasks.append(task)
        
        return scheduled_tasks
    
    async def _schedule_priority(
        self, 
        execution: WorkflowExecution,
        steps: List[WorkflowStep],
        execution_plan: ExecutionPlan
    ) -> List[ScheduledTask]:
        """优先级调度"""
        scheduled_tasks = []
        base_time = utc_now()
        
        # 根据步骤类型确定优先级
        type_priority_map = {
            'reasoning': QueuePriority.HIGH,
            'tool_call': QueuePriority.NORMAL,
            'validation': QueuePriority.NORMAL,
            'aggregation': QueuePriority.HIGH,
            'decision': QueuePriority.HIGH
        }
        
        for step in steps:
            priority = type_priority_map.get(step.step_type.value, QueuePriority.NORMAL)
            
            task = ScheduledTask(
                id=str(uuid4()),
                execution_id=execution.id,
                step_id=step.id,
                workflow_definition_id=execution.workflow_definition_id,
                priority=priority,
                dependencies=step.dependencies,
                estimated_duration=step.timeout_seconds // 60 if step.timeout_seconds else 10,
                max_retries=step.retry_count or 3,
                retry_count=0,
                scheduled_at=base_time
            )
            scheduled_tasks.append(task)
        
        return scheduled_tasks
    
    async def _schedule_critical_path(
        self, 
        execution: WorkflowExecution,
        steps: List[WorkflowStep],
        execution_plan: ExecutionPlan
    ) -> List[ScheduledTask]:
        """关键路径优先调度"""
        scheduled_tasks = []
        base_time = utc_now()
        
        critical_path_steps = set(execution_plan.critical_path)
        
        for step in steps:
            # 关键路径上的任务获得更高优先级
            if step.id in critical_path_steps:
                priority = QueuePriority.CRITICAL
            else:
                priority = QueuePriority.NORMAL
            
            task = ScheduledTask(
                id=str(uuid4()),
                execution_id=execution.id,
                step_id=step.id,
                workflow_definition_id=execution.workflow_definition_id,
                priority=priority,
                dependencies=step.dependencies,
                estimated_duration=step.timeout_seconds // 60 if step.timeout_seconds else 10,
                max_retries=step.retry_count or 3,
                retry_count=0,
                scheduled_at=base_time
            )
            scheduled_tasks.append(task)
        
        return scheduled_tasks
    
    async def _schedule_resource_aware(
        self, 
        execution: WorkflowExecution,
        steps: List[WorkflowStep],
        execution_plan: ExecutionPlan
    ) -> List[ScheduledTask]:
        """资源感知调度"""
        scheduled_tasks = []
        base_time = utc_now()
        
        # 分析资源需求
        resource_heavy_steps = []
        for step in steps:
            if step.step_type.value in ['reasoning', 'tool_call']:
                resource_heavy_steps.append(step.id)
        
        for step in steps:
            # 资源密集型任务降低优先级避免冲突
            if step.id in resource_heavy_steps:
                priority = QueuePriority.LOW
                # 增加调度延迟
                delay_minutes = len([s for s in resource_heavy_steps if s <= step.id]) * 2
                scheduled_at = base_time + timedelta(minutes=delay_minutes)
            else:
                priority = QueuePriority.NORMAL
                scheduled_at = base_time
            
            task = ScheduledTask(
                id=str(uuid4()),
                execution_id=execution.id,
                step_id=step.id,
                workflow_definition_id=execution.workflow_definition_id,
                priority=priority,
                dependencies=step.dependencies,
                estimated_duration=step.timeout_seconds // 60 if step.timeout_seconds else 10,
                max_retries=step.retry_count or 3,
                retry_count=0,
                scheduled_at=scheduled_at
            )
            scheduled_tasks.append(task)
        
        return scheduled_tasks
    
    async def register_worker(self, worker_id: str, capabilities: List[str], max_concurrent: int = 1) -> bool:
        """注册工作器"""
        try:
            worker_info = {
                'id': worker_id,
                'capabilities': capabilities,
                'max_concurrent': max_concurrent,
                'current_tasks': 0,
                'registered_at': utc_now().isoformat(),
                'last_heartbeat': utc_now().isoformat(),
                'status': 'active'
            }
            
            worker_key = f"{self.task_queue.worker_key_prefix}{worker_id}"
            await self.redis.hset(worker_key, mapping=worker_info)
            
            # 设置过期时间
            await self.redis.expire(worker_key, 3600)  # 1小时
            
            self.active_workers[worker_id] = worker_info
            logger.info(f"工作器注册: {worker_id}, 能力: {capabilities}")
            return True
            
        except Exception as e:
            logger.error(f"工作器注册失败: {worker_id}, 错误: {e}")
            return False
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """注销工作器"""
        try:
            worker_key = f"{self.task_queue.worker_key_prefix}{worker_id}"
            await self.redis.delete(worker_key)
            
            if worker_id in self.active_workers:
                del self.active_workers[worker_id]
            
            logger.info(f"工作器注销: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"工作器注销失败: {worker_id}, 错误: {e}")
            return False
    
    async def update_worker_heartbeat(self, worker_id: str) -> bool:
        """更新工作器心跳"""
        try:
            worker_key = f"{self.task_queue.worker_key_prefix}{worker_id}"
            await self.redis.hset(worker_key, 'last_heartbeat', utc_now().isoformat())
            
            if worker_id in self.active_workers:
                self.active_workers[worker_id]['last_heartbeat'] = utc_now().isoformat()
            
            return True
            
        except Exception as e:
            logger.error(f"更新工作器心跳失败: {worker_id}, 错误: {e}")
            return False
    
    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        try:
            queue_stats = await self.task_queue.get_queue_stats()
            
            stats = {
                'queue_stats': queue_stats,
                'active_workers': len(self.active_workers),
                'worker_details': list(self.active_workers.values()),
                'scheduling_strategies': list(self.scheduling_strategies.keys()),
                'timestamp': utc_now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取调度器统计失败: {e}")
            return {}
    
    async def cleanup_expired_tasks(self, max_age_hours: int = 24) -> int:
        """清理过期任务"""
        try:
            cutoff_time = utc_now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            # 清理完成和失败的任务
            for queue_name in [self.task_queue.completed_queue_name, self.task_queue.failed_queue_name]:
                while True:
                    task_id = await self.redis.rpop(queue_name)
                    if not task_id:
                        break
                    
                    task_id = task_id.decode() if isinstance(task_id, bytes) else task_id
                    task = await self.task_queue.get_task(task_id)
                    
                    if task and task.completed_at and task.completed_at < cutoff_time:
                        # 删除过期任务
                        task_key = f"{self.task_queue.task_key_prefix}{task_id}"
                        await self.redis.delete(task_key)
                        cleaned_count += 1
                    else:
                        # 如果任务还没过期，放回队列
                        if task:
                            await self.redis.lpush(queue_name, task_id)
                        break
            
            logger.info(f"清理过期任务完成: {cleaned_count} 个")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理过期任务失败: {e}")
            return 0


# 工作器基类
class WorkflowWorker:
    """工作流工作器基类"""
    
    def __init__(self, worker_id: str, scheduler: WorkflowScheduler, capabilities: List[str]):
        self.worker_id = worker_id
        self.scheduler = scheduler
        self.capabilities = capabilities
        self.running = False
        self.current_task: Optional[ScheduledTask] = None
    
    async def start(self):
        """启动工作器"""
        self.running = True
        await self.scheduler.register_worker(self.worker_id, self.capabilities)
        
        logger.info(f"工作器启动: {self.worker_id}")
        
        # 主工作循环
        while self.running:
            try:
                # 更新心跳
                await self.scheduler.update_worker_heartbeat(self.worker_id)
                
                # 获取任务
                task = await self.scheduler.task_queue.dequeue_task(self.worker_id, timeout=5)
                
                if task:
                    self.current_task = task
                    await self._process_task(task)
                    self.current_task = None
                
            except Exception as e:
                logger.error(f"工作器执行错误: {self.worker_id}, 错误: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """停止工作器"""
        self.running = False
        await self.scheduler.unregister_worker(self.worker_id)
        logger.info(f"工作器停止: {self.worker_id}")
    
    async def _process_task(self, task: ScheduledTask):
        """处理任务（子类需要实现）"""
        try:
            logger.info(f"开始处理任务: {task.id}")
            
            # 模拟任务处理
            await asyncio.sleep(1)
            
            # 标记任务完成
            await self.scheduler.task_queue.complete_task(task.id, {"processed_by": self.worker_id})
            
        except Exception as e:
            logger.error(f"任务处理失败: {task.id}, 错误: {e}")
            await self.scheduler.task_queue.fail_task(task.id, str(e))