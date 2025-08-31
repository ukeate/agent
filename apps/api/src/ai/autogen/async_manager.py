"""
AutoGen异步智能体管理器
实现异步智能体生命周期管理、任务调度和协作
"""
import asyncio
import uuid
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import structlog

from .agents import BaseAutoGenAgent, create_agent_from_config
from .config import AgentConfig, AgentRole
from .events import (
    Event, EventType, EventPriority, EventBus, EventHandler,
    MessageQueue, StateManager
)

logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    """智能体状态枚举"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """智能体任务"""
    id: str
    agent_id: str
    task_type: str
    description: str
    input_data: Dict[str, Any]
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "description": self.description,
            "input_data": self.input_data,
            "priority": self.priority,
            "status": self.status.value if hasattr(self.status, 'value') else str(self.status),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class AgentInfo:
    """智能体信息"""
    id: str
    name: str
    role: AgentRole
    status: AgentStatus
    agent: BaseAutoGenAgent
    created_at: datetime
    last_activity: datetime
    current_task_id: Optional[str] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value if hasattr(self.role, 'value') else str(self.role),
            "status": self.status.value if hasattr(self.status, 'value') else str(self.status),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "current_task_id": self.current_task_id,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "average_task_time": self.average_task_time
        }


class AsyncAgentManager:
    """异步智能体管理器"""
    
    def __init__(
        self, 
        event_bus: EventBus,
        message_queue: MessageQueue,
        state_manager: StateManager,
        max_concurrent_tasks: int = 10
    ):
        self.event_bus = event_bus
        self.message_queue = message_queue
        self.state_manager = state_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # 智能体管理
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # 任务管理
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # 运行状态
        self.running = False
        self._task_processor_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # 注册事件处理器
        self._register_event_handlers()
        
        logger.info("异步智能体管理器初始化完成", max_concurrent_tasks=max_concurrent_tasks)
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 注册智能体状态变化处理器
        self.state_manager.add_state_change_callback(self._on_agent_state_change)
    
    async def start(self) -> None:
        """启动管理器"""
        if self.running:
            logger.warning("异步智能体管理器已在运行")
            return
        
        self.running = True
        
        # 启动任务处理器
        self._task_processor_task = asyncio.create_task(self._task_processor())
        
        # 启动健康监控
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        
        # 发布系统启动事件
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS_CHANGED,
            source="agent_manager",
            data={"status": "started", "timestamp": utc_now().isoformat()}
        ))
        
        logger.info("异步智能体管理器启动")
    
    async def stop(self) -> None:
        """停止管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止所有运行中的任务
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                logger.info("取消运行中的任务", task_id=task_id)
        
        # 等待任务完成
        if self.running_tasks:
            await asyncio.gather(
                *self.running_tasks.values(), 
                return_exceptions=True
            )
            self.running_tasks.clear()
        
        # 停止管理器任务
        if self._task_processor_task:
            self._task_processor_task.cancel()
            try:
                await self._task_processor_task
            except asyncio.CancelledError:
                pass
        
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # 发布系统停止事件
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STATUS_CHANGED,
            source="agent_manager",
            data={"status": "stopped", "timestamp": utc_now().isoformat()}
        ))
        
        logger.info("异步智能体管理器停止")
    
    async def create_agent(self, config: AgentConfig, agent_id: Optional[str] = None) -> str:
        """创建新的智能体"""
        try:
            agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
            
            # 检查是否已存在
            if agent_id in self.agents:
                raise ValueError(f"智能体ID已存在: {agent_id}")
            
            logger.info("开始创建智能体", agent_id=agent_id, role=config.role)
            
            # 创建智能体实例
            agent = create_agent_from_config(config)
            
            # 创建智能体信息
            agent_info = AgentInfo(
                id=agent_id,
                name=config.name,
                role=config.role,
                status=AgentStatus.INITIALIZING,
                agent=agent,
                created_at=utc_now(),
                last_activity=utc_now()
            )
            
            # 注册到管理器
            self.agents[agent_id] = agent_info
            self.agent_configs[agent_id] = config
            
            # 更新状态
            await self.state_manager.update_agent_state(agent_id, {
                "status": AgentStatus.INITIALIZING.value,
                "created_at": agent_info.created_at.isoformat(),
                "config": {
                    "name": config.name,
                    "role": config.role.value if hasattr(config.role, 'value') else str(config.role),
                    "capabilities": config.capabilities
                }
            })
            
            # 标记为空闲状态
            await self._update_agent_status(agent_id, AgentStatus.IDLE)
            
            # 发布智能体创建事件
            await self.event_bus.publish(Event(
                type=EventType.AGENT_CREATED,
                source="agent_manager",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "name": config.name,
                    "role": config.role.value if hasattr(config.role, 'value') else str(config.role),
                    "capabilities": config.capabilities
                }
            ))
            
            logger.info("智能体创建成功", agent_id=agent_id, name=config.name)
            return agent_id
            
        except Exception as e:
            logger.error("创建智能体失败", agent_id=agent_id, error=str(e))
            raise
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """销毁智能体"""
        try:
            if agent_id not in self.agents:
                logger.warning("智能体不存在", agent_id=agent_id)
                return False
            
            agent_info = self.agents[agent_id]
            
            # 如果有正在执行的任务，先取消
            if agent_info.current_task_id:
                await self.cancel_task(agent_info.current_task_id)
            
            # 更新状态为关闭
            await self._update_agent_status(agent_id, AgentStatus.SHUTDOWN)
            
            # 删除状态
            await self.state_manager.delete_agent_state(agent_id)
            
            # 从管理器移除
            del self.agents[agent_id]
            if agent_id in self.agent_configs:
                del self.agent_configs[agent_id]
            
            # 发布智能体销毁事件
            await self.event_bus.publish(Event(
                type=EventType.AGENT_DESTROYED,
                source="agent_manager",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "name": agent_info.name,
                    "total_tasks": agent_info.total_tasks,
                    "completed_tasks": agent_info.completed_tasks
                }
            ))
            
            logger.info("智能体销毁成功", agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error("销毁智能体失败", agent_id=agent_id, error=str(e))
            return False
    
    async def submit_task(
        self, 
        agent_id: str,
        task_type: str,
        description: str,
        input_data: Dict[str, Any],
        priority: int = 0,
        timeout_seconds: int = 300
    ) -> str:
        """提交任务给智能体"""
        try:
            # 检查智能体是否存在
            if agent_id not in self.agents:
                raise ValueError(f"智能体不存在: {agent_id}")
            
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            
            # 创建任务
            task = AgentTask(
                id=task_id,
                agent_id=agent_id,
                task_type=task_type,
                description=description,
                input_data=input_data,
                priority=priority,
                timeout_seconds=timeout_seconds
            )
            
            # 保存任务
            self.tasks[task_id] = task
            
            # 加入任务队列（优先级队列，数值越小优先级越高）
            await self.task_queue.put((-priority, task_id))
            
            # 发布任务分配事件
            await self.event_bus.publish(Event(
                type=EventType.TASK_ASSIGNED,
                source="agent_manager",
                target=agent_id,
                data={
                    "task_id": task_id,
                    "task_type": task_type,
                    "priority": priority,
                    "agent_id": agent_id
                }
            ))
            
            logger.info("任务提交成功", task_id=task_id, agent_id=agent_id, task_type=task_type)
            return task_id
            
        except Exception as e:
            logger.error("任务提交失败", agent_id=agent_id, error=str(e))
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            if task_id not in self.tasks:
                logger.warning("任务不存在", task_id=task_id)
                return False
            
            task = self.tasks[task_id]
            
            # 如果任务正在运行，取消执行
            if task_id in self.running_tasks:
                running_task = self.running_tasks[task_id]
                if not running_task.done():
                    running_task.cancel()
                del self.running_tasks[task_id]
            
            # 更新任务状态
            task.status = TaskStatus.CANCELLED
            task.completed_at = utc_now()
            
            # 如果智能体正在执行此任务，更新智能体状态
            agent_info = self.agents.get(task.agent_id)
            if agent_info and agent_info.current_task_id == task_id:
                agent_info.current_task_id = None
                await self._update_agent_status(task.agent_id, AgentStatus.IDLE)
            
            logger.info("任务取消成功", task_id=task_id)
            return True
            
        except Exception as e:
            logger.error("任务取消失败", task_id=task_id, error=str(e))
            return False
    
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取智能体信息"""
        if agent_id not in self.agents:
            return None
        
        agent_info = self.agents[agent_id]
        state = await self.state_manager.get_agent_state(agent_id)
        
        return {
            **agent_info.to_dict(),
            "state": state
        }
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有智能体"""
        agents_info = []
        for agent_id, agent_info in self.agents.items():
            state = await self.state_manager.get_agent_state(agent_id)
            agents_info.append({
                **agent_info.to_dict(),
                "state": state
            })
        return agents_info
    
    async def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None
    
    async def list_tasks(
        self, 
        agent_id: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[Dict[str, Any]]:
        """列出任务"""
        tasks = []
        for task in self.tasks.values():
            if agent_id and task.agent_id != agent_id:
                continue
            if status and task.status != status:
                continue
            tasks.append(task.to_dict())
        return tasks
    
    async def _task_processor(self) -> None:
        """任务处理器"""
        logger.info("任务处理器启动")
        
        while self.running:
            try:
                # 检查并发任务限制
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # 从队列获取任务
                try:
                    priority, task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 检查任务是否仍然有效
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                # 检查任务状态
                if task.status != TaskStatus.PENDING:
                    continue
                
                # 检查智能体是否可用
                agent_info = self.agents.get(task.agent_id)
                if not agent_info or agent_info.status != AgentStatus.IDLE:
                    # 重新放回队列
                    await self.task_queue.put((priority, task_id))
                    await asyncio.sleep(0.1)
                    continue
                
                # 执行任务
                task_coroutine = self._execute_task(task_id)
                self.running_tasks[task_id] = asyncio.create_task(task_coroutine)
                
            except Exception as e:
                logger.error("任务处理器异常", error=str(e))
                await asyncio.sleep(1.0)
        
        logger.info("任务处理器停止")
    
    async def _execute_task(self, task_id: str) -> None:
        """执行单个任务"""
        task = self.tasks[task_id]
        agent_info = self.agents[task.agent_id]
        
        try:
            logger.info("开始执行任务", task_id=task_id, agent_id=task.agent_id)
            
            # 更新任务状态
            task.status = TaskStatus.RUNNING
            task.started_at = utc_now()
            
            # 更新智能体状态
            agent_info.current_task_id = task_id
            await self._update_agent_status(task.agent_id, AgentStatus.BUSY)
            
            # 发布任务开始事件
            await self.event_bus.publish(Event(
                type=EventType.TASK_STARTED,
                source=task.agent_id,
                data={
                    "task_id": task_id,
                    "task_type": task.task_type,
                    "agent_id": task.agent_id
                }
            ))
            
            # 执行任务
            result = await asyncio.wait_for(
                agent_info.agent.execute_task(
                    task.description,
                    task.task_type,
                    task.input_data
                ),
                timeout=task.timeout_seconds
            )
            
            # 任务完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = utc_now()
            task.result = result
            
            # 更新智能体统计
            agent_info.completed_tasks += 1
            agent_info.total_tasks += 1
            
            # 计算平均执行时间
            execution_time = (task.completed_at - task.started_at).total_seconds()
            if agent_info.average_task_time == 0:
                agent_info.average_task_time = execution_time
            else:
                agent_info.average_task_time = (
                    agent_info.average_task_time * (agent_info.completed_tasks - 1) + execution_time
                ) / agent_info.completed_tasks
            
            # 发布任务完成事件
            await self.event_bus.publish(Event(
                type=EventType.TASK_COMPLETED,
                source=task.agent_id,
                data={
                    "task_id": task_id,
                    "task_type": task.task_type,
                    "agent_id": task.agent_id,
                    "execution_time": execution_time,
                    "result": result
                }
            ))
            
            logger.info("任务执行成功", task_id=task_id, execution_time=execution_time)
            
        except asyncio.TimeoutError:
            # 任务超时
            task.status = TaskStatus.FAILED
            task.error = "Task timeout"
            task.completed_at = utc_now()
            
            agent_info.failed_tasks += 1
            agent_info.total_tasks += 1
            
            await self.event_bus.publish(Event(
                type=EventType.TASK_FAILED,
                source=task.agent_id,
                data={
                    "task_id": task_id,
                    "agent_id": task.agent_id,
                    "error": "timeout",
                    "timeout_seconds": task.timeout_seconds
                }
            ))
            
            logger.error("任务执行超时", task_id=task_id, timeout=task.timeout_seconds)
            
        except Exception as e:
            # 任务失败
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = utc_now()
            
            agent_info.failed_tasks += 1
            agent_info.total_tasks += 1
            
            # 检查是否需要重试
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                await self.task_queue.put((-task.priority, task_id))
                logger.info("任务重试", task_id=task_id, retry_count=task.retry_count)
            else:
                await self.event_bus.publish(Event(
                    type=EventType.TASK_FAILED,
                    source=task.agent_id,
                    data={
                        "task_id": task_id,
                        "agent_id": task.agent_id,
                        "error": str(e),
                        "retry_count": task.retry_count
                    }
                ))
                
                logger.error("任务执行失败", task_id=task_id, error=str(e))
        
        finally:
            # 清理
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # 重置智能体状态
            agent_info.current_task_id = None
            agent_info.last_activity = utc_now()
            await self._update_agent_status(task.agent_id, AgentStatus.IDLE)
    
    async def _update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """更新智能体状态"""
        if agent_id not in self.agents:
            return
        
        agent_info = self.agents[agent_id]
        old_status = agent_info.status
        agent_info.status = status
        agent_info.last_activity = utc_now()
        
        # 更新状态管理器
        await self.state_manager.update_agent_state(agent_id, {
            "status": status.value,
            "last_activity": agent_info.last_activity.isoformat()
        })
        
        # 发布状态变化事件
        if old_status != status:
            await self.event_bus.publish(Event(
                type=EventType.AGENT_STATUS_CHANGED,
                source="agent_manager",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "old_status": old_status.value,
                    "new_status": status.value
                }
            ))
    
    async def _on_agent_state_change(
        self, 
        agent_id: str, 
        old_state: Dict[str, Any], 
        new_state: Dict[str, Any]
    ) -> None:
        """智能体状态变化回调"""
        logger.debug("智能体状态变化", agent_id=agent_id, new_status=new_state.get("status"))
    
    async def _health_monitor(self) -> None:
        """健康监控"""
        logger.info("健康监控启动")
        
        while self.running:
            try:
                current_time = utc_now()
                
                # 检查智能体健康状态
                for agent_id, agent_info in self.agents.items():
                    # 检查是否长时间无活动
                    inactive_time = (current_time - agent_info.last_activity).total_seconds()
                    if inactive_time > 3600:  # 1小时无活动
                        logger.warning("智能体长时间无活动", agent_id=agent_id, inactive_hours=inactive_time/3600)
                
                # 检查任务队列大小
                queue_size = self.task_queue.qsize()
                if queue_size > 100:
                    logger.warning("任务队列积压严重", queue_size=queue_size)
                
                # 等待下次检查
                await asyncio.sleep(300)  # 5分钟检查一次
                
            except Exception as e:
                logger.error("健康监控异常", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("健康监控停止")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        total_agents = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.IDLE)
        busy_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.BUSY)
        
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        running_tasks = len(self.running_tasks)
        pending_tasks = self.task_queue.qsize()
        
        return {
            "running": self.running,
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "busy": busy_agents,
                "error": sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ERROR)
            },
            "tasks": {
                "total": total_tasks,
                "completed": completed_tasks,
                "running": running_tasks,
                "pending": pending_tasks,
                "failed": sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
            },
            "queue": {
                "size": pending_tasks,
                "max_concurrent": self.max_concurrent_tasks
            }
        }