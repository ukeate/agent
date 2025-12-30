"""
AutoGen异步事件驱动架构
实现事件总线、消息队列和状态管理
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict

from src.core.logging import get_logger
logger = get_logger(__name__)

class EventType(str, Enum):
    """事件类型枚举"""
    # 智能体生命周期事件
    AGENT_CREATED = "agent.created"
    AGENT_DESTROYED = "agent.destroyed"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    
    # 消息通信事件
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_PROCESSED = "message.processed"
    MESSAGE_FAILED = "message.failed"
    
    # 对话管理事件
    CONVERSATION_STARTED = "conversation.started"
    CONVERSATION_PAUSED = "conversation.paused"
    CONVERSATION_RESUMED = "conversation.resumed"
    CONVERSATION_ENDED = "conversation.ended"
    
    # 任务执行事件
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # 系统事件
    ERROR_OCCURRED = "error.occurred"
    SYSTEM_STATUS_CHANGED = "system.status_changed"

class EventPriority(str, Enum):
    """事件优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Event:
    """事件数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = field(default=EventType.MESSAGE_SENT)
    timestamp: datetime = field(default_factory=lambda: utc_now())
    source: str = ""  # 事件源（智能体名称、系统组件名等）
    target: Optional[str] = None  # 事件目标
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: EventPriority = field(default=EventPriority.NORMAL)
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None  # 关联ID，用于追踪相关事件
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'priority': self.priority.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['type'] = EventType(data['type'])
        data['priority'] = EventPriority(data['priority'])
        return cls(**data)

class EventHandler(ABC):
    """事件处理器抽象基类"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """处理事件"""
        ...
    
    @property
    @abstractmethod
    def supported_events(self) -> List[EventType]:
        """支持的事件类型"""
        ...

class EventBus:
    """异步事件总线"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.subscribers: Dict[EventType, List[EventHandler]] = {}
        self.wildcard_handlers: List[EventHandler] = []  # 监听所有事件的处理器
        self.running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._processed_events = 0
        self._failed_events = 0
        
        # 事件过滤器
        self.event_filters: List[Callable[[Event], bool]] = []
        
        logger.info("事件总线初始化完成", max_queue_size=max_queue_size)
    
    async def start(self, worker_count: int = 3) -> None:
        """启动事件处理循环"""
        if self.running:
            logger.warning("事件总线已在运行")
            return
        
        self.running = True
        
        # 启动多个工作协程处理事件
        for i in range(worker_count):
            task = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        logger.info("事件总线启动", worker_count=worker_count)
    
    async def stop(self) -> None:
        """停止事件处理"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待所有工作任务完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
        
        logger.info(
            "事件总线停止",
            processed_events=self._processed_events,
            failed_events=self._failed_events
        )
    
    async def publish(self, event: Event) -> bool:
        """发布事件"""
        try:
            # 应用事件过滤器
            for filter_func in self.event_filters:
                if not filter_func(event):
                    logger.debug("事件被过滤器拒绝", event_type=event.type, event_id=event.id)
                    return False
            
            # 如果队列满了，尝试丢弃低优先级事件
            if self.event_queue.full():
                await self._handle_queue_overflow(event)
            
            await self.event_queue.put(event)
            
            logger.debug(
                "事件发布成功",
                event_type=event.type,
                event_id=event.id,
                source=event.source,
                queue_size=self.event_queue.qsize()
            )
            
            return True
            
        except asyncio.QueueFull:
            logger.error(
                "事件队列已满，丢弃事件",
                event_type=event.type,
                event_id=event.id,
                queue_size=self.event_queue.qsize()
            )
            return False
        except Exception as e:
            logger.error("事件发布失败", event_id=event.id, error=str(e))
            return False
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """订阅特定类型事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
        logger.info(
            "添加事件订阅",
            event_type=event_type,
            handler=handler.__class__.__name__
        )
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """订阅所有事件"""
        self.wildcard_handlers.append(handler)
        logger.info("添加通配符事件订阅", handler=handler.__class__.__name__)
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(handler)
                logger.info("取消事件订阅", event_type=event_type)
            except ValueError:
                logger.warning("事件订阅不存在，无法取消", event_type=event_type, exc_info=True)
    
    def add_filter(self, filter_func: Callable[[Event], bool]) -> None:
        """添加事件过滤器"""
        self.event_filters.append(filter_func)
    
    async def _event_worker(self, worker_name: str) -> None:
        """事件处理工作协程"""
        logger.info("事件处理工作协程启动", worker=worker_name)
        
        while self.running:
            try:
                # 等待事件，设置超时避免永久阻塞
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                await self._process_event(event, worker_name)
                self.event_queue.task_done()
                self._processed_events += 1
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                logger.error("事件处理工作协程异常", worker=worker_name, error=str(e))
                self._failed_events += 1
        
        logger.info("事件处理工作协程停止", worker=worker_name)
    
    async def _process_event(self, event: Event, worker_name: str) -> None:
        """处理单个事件"""
        start_time = utc_now()
        
        try:
            # 收集所有需要处理此事件的处理器
            handlers = []
            
            # 特定类型订阅者
            if event.type in self.subscribers:
                handlers.extend(self.subscribers[event.type])
            
            # 通配符订阅者
            handlers.extend(self.wildcard_handlers)
            
            if not handlers:
                logger.debug("没有找到事件处理器", event_type=event.type)
                return
            
            # 并行处理所有处理器
            tasks = []
            for handler in handlers:
                task = asyncio.create_task(
                    self._safe_handle_event(handler, event)
                )
                tasks.append(task)
            
            # 等待所有处理器完成
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            processing_time = (utc_now() - start_time).total_seconds()
            
            logger.debug(
                "事件处理完成",
                event_type=event.type,
                event_id=event.id,
                handler_count=len(handlers),
                processing_time_ms=processing_time * 1000,
                worker=worker_name
            )
            
        except Exception as e:
            logger.error(
                "事件处理失败",
                event_type=event.type,
                event_id=event.id,
                error=str(e),
                worker=worker_name
            )
    
    async def _safe_handle_event(self, handler: EventHandler, event: Event) -> None:
        """安全处理事件（捕获处理器异常）"""
        try:
            await handler.handle(event)
        except Exception as e:
            logger.error(
                "事件处理器异常",
                handler=handler.__class__.__name__,
                event_type=event.type,
                event_id=event.id,
                error=str(e)
            )
    
    async def _handle_queue_overflow(self, new_event: Event) -> None:
        """处理队列溢出"""
        # 如果新事件是高优先级或关键优先级，尝试清理低优先级事件
        if new_event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
            # 简单实现：这里可以添加更复杂的优先级管理逻辑
            logger.warning(
                "队列溢出，高优先级事件强制入队",
                event_type=new_event.type,
                priority=new_event.priority
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计信息"""
        return {
            "running": self.running,
            "queue_size": self.event_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "processed_events": self._processed_events,
            "failed_events": self._failed_events,
            "subscriber_count": sum(len(handlers) for handlers in self.subscribers.values()),
            "wildcard_handler_count": len(self.wildcard_handlers),
            "worker_count": len(self._worker_tasks)
        }

class MessageQueue:
    """消息队列系统"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_queues: Dict[str, asyncio.Queue] = {}
        self._message_handlers: Dict[str, Callable] = {}
        
    async def send_message(
        self, 
        queue_name: str, 
        message: Dict[str, Any],
        priority: int = 0
    ) -> bool:
        """发送消息到队列"""
        try:
            # 如果有Redis，使用Redis队列
            if self.redis_client:
                await self._send_to_redis_queue(queue_name, message, priority)
            else:
                # 否则使用本地队列
                await self._send_to_local_queue(queue_name, message)
            
            logger.debug("消息发送成功", queue=queue_name, message_id=message.get('id'))
            return True
            
        except Exception as e:
            logger.error("消息发送失败", queue=queue_name, error=str(e))
            return False
    
    async def _send_to_local_queue(self, queue_name: str, message: Dict[str, Any]) -> None:
        """发送到本地队列"""
        if queue_name not in self.local_queues:
            self.local_queues[queue_name] = asyncio.Queue()
        
        await self.local_queues[queue_name].put(message)
    
    async def _send_to_redis_queue(
        self, 
        queue_name: str, 
        message: Dict[str, Any], 
        priority: int
    ) -> None:
        """发送到Redis队列"""
        # Redis实现（使用redis.asyncio）
        serialized_message = json.dumps(message)
        await self.redis_client.lpush(f"queue:{queue_name}", serialized_message)
    
    async def consume_messages(
        self, 
        queue_name: str, 
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """消费消息"""
        if self.redis_client:
            await self._consume_from_redis(queue_name, handler)
        else:
            await self._consume_from_local(queue_name, handler)
    
    async def _consume_from_local(
        self, 
        queue_name: str, 
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """从本地队列消费消息"""
        if queue_name not in self.local_queues:
            self.local_queues[queue_name] = asyncio.Queue()
        
        queue = self.local_queues[queue_name]
        
        while True:
            try:
                message = await queue.get()
                await handler(message)
                queue.task_done()
            except Exception as e:
                logger.error("消息处理失败", queue=queue_name, error=str(e))
    
    async def _consume_from_redis(
        self, 
        queue_name: str, 
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """从Redis队列消费消息"""
        while True:
            try:
                result = await self.redis_client.brpop(f"queue:{queue_name}", timeout=1)
                if result:
                    _, message_data = result
                    message = json.loads(message_data)
                    await handler(message)
            except Exception as e:
                logger.error("Redis消息处理失败", queue=queue_name, error=str(e))
                await asyncio.sleep(1)  # 错误时等待一秒再重试

class StateManager:
    """分布式状态管理器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_state: Dict[str, Any] = {}
        self.state_locks: Dict[str, asyncio.Lock] = {}
        self._state_change_callbacks: List[Callable[[str, Any, Any], None]] = []
    
    async def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        try:
            # 先从本地缓存获取
            if agent_id in self.local_state:
                return self.local_state[agent_id].copy()
            
            # 从Redis获取
            if self.redis_client:
                state_data = await self.redis_client.get(f"agent_state:{agent_id}")
                if state_data:
                    state = json.loads(state_data)
                    self.local_state[agent_id] = state
                    return state.copy()
            
            # 返回默认状态
            default_state = {
                "status": "idle",
                "last_activity": utc_now().isoformat(),
                "current_task": None,
                "capabilities": [],
                "load": 0.0
            }
            return default_state
            
        except Exception as e:
            logger.error("获取智能体状态失败", agent_id=agent_id, error=str(e))
            return {}
    
    async def update_agent_state(
        self, 
        agent_id: str, 
        state_update: Dict[str, Any]
    ) -> bool:
        """更新智能体状态"""
        try:
            # 获取锁
            if agent_id not in self.state_locks:
                self.state_locks[agent_id] = asyncio.Lock()
            
            async with self.state_locks[agent_id]:
                # 获取当前状态
                current_state = await self.get_agent_state(agent_id)
                old_state = current_state.copy()
                
                # 应用更新
                current_state.update(state_update)
                current_state["last_updated"] = utc_now().isoformat()
                
                # 同时更新本地和Redis
                self.local_state[agent_id] = current_state
                
                if self.redis_client:
                    await self.redis_client.setex(
                        f"agent_state:{agent_id}",
                        3600,  # 1小时TTL
                        json.dumps(current_state)
                    )
                
                # 通知状态变化回调
                for callback in self._state_change_callbacks:
                    try:
                        await callback(agent_id, old_state, current_state)
                    except Exception as e:
                        logger.error("状态变化回调失败", callback=callback.__name__, error=str(e))
                
                logger.debug("智能体状态更新成功", agent_id=agent_id)
                return True
                
        except Exception as e:
            logger.error("更新智能体状态失败", agent_id=agent_id, error=str(e))
            return False
    
    def add_state_change_callback(
        self, 
        callback: Callable[[str, Any, Any], None]
    ) -> None:
        """添加状态变化回调"""
        self._state_change_callbacks.append(callback)
    
    async def delete_agent_state(self, agent_id: str) -> bool:
        """删除智能体状态"""
        try:
            # 从本地删除
            if agent_id in self.local_state:
                del self.local_state[agent_id]
            
            # 从Redis删除
            if self.redis_client:
                await self.redis_client.delete(f"agent_state:{agent_id}")
            
            # 清理锁
            if agent_id in self.state_locks:
                del self.state_locks[agent_id]
            
            logger.info("智能体状态删除成功", agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error("删除智能体状态失败", agent_id=agent_id, error=str(e))
            return False
    
    async def list_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """列出所有智能体状态"""
        try:
            if self.redis_client:
                # 从Redis获取所有状态
                keys = await self.redis_client.keys("agent_state:*")
                states = {}
                
                for key in keys:
                    agent_id = key.decode().replace("agent_state:", "")
                    state_data = await self.redis_client.get(key)
                    if state_data:
                        states[agent_id] = json.loads(state_data)
                
                return states
            else:
                # 返回本地状态
                return self.local_state.copy()
                
        except Exception as e:
            logger.error("列出智能体状态失败", error=str(e))
            return {}

# 预定义的事件处理器
class LoggingEventHandler(EventHandler):
    """日志事件处理器"""
    
    @property
    def supported_events(self) -> List[EventType]:
        return list(EventType)  # 支持所有事件类型
    
    async def handle(self, event: Event) -> None:
        """记录事件日志"""
        logger.info(
            "事件处理",
            event_type=event.type,
            event_id=event.id,
            source=event.source,
            target=event.target,
            timestamp=event.timestamp.isoformat()
        )

class MetricsEventHandler(EventHandler):
    """指标收集事件处理器"""
    
    def __init__(self):
        self.metrics = {
            "event_counts": {},
            "processing_times": [],
            "error_counts": 0
        }
    
    @property
    def supported_events(self) -> List[EventType]:
        return list(EventType)
    
    async def handle(self, event: Event) -> None:
        """收集事件指标"""
        # 统计事件数量
        event_type = event.type.value
        if event_type not in self.metrics["event_counts"]:
            self.metrics["event_counts"][event_type] = 0
        self.metrics["event_counts"][event_type] += 1
        
        # 统计错误
        if event.type == EventType.ERROR_OCCURRED:
            self.metrics["error_counts"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标数据"""
        return self.metrics.copy()
