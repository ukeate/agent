"""
工作流事件系统
支持状态变更监听和事件通知
"""
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import json

from .state import MessagesState


class EventType(Enum):
    """事件类型"""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    
    STATE_CHANGED = "state_changed"
    CHECKPOINT_CREATED = "checkpoint_created"
    ERROR_OCCURRED = "error_occurred"
    TIMEOUT_OCCURRED = "timeout_occurred"
    
    CUSTOM_EVENT = "custom_event"


@dataclass
class WorkflowEvent:
    """工作流事件"""
    id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    event_type: EventType = EventType.CUSTOM_EVENT
    workflow_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    node_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "node_name": self.node_name,
            "data": self.data,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class EventListener(ABC):
    """事件监听器抽象基类"""
    
    @abstractmethod
    async def handle_event(self, event: WorkflowEvent):
        """处理事件"""
        pass
    
    @abstractmethod
    def get_interested_events(self) -> List[EventType]:
        """获取感兴趣的事件类型"""
        pass


class LoggingEventListener(EventListener):
    """日志事件监听器"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
    
    async def handle_event(self, event: WorkflowEvent):
        """记录事件到日志"""
        print(f"[{self.log_level}] {event.timestamp} - {event.event_type.value}: {event.workflow_id}")
        if event.node_name:
            print(f"  节点: {event.node_name}")
        if event.data:
            print(f"  数据: {event.data}")
    
    def get_interested_events(self) -> List[EventType]:
        """监听所有事件"""
        return list(EventType)


class WebSocketEventListener(EventListener):
    """WebSocket事件监听器"""
    
    def __init__(self):
        self.connections: Dict[str, List] = {}  # workflow_id -> [websocket_connections]
    
    def add_connection(self, workflow_id: str, websocket):
        """添加WebSocket连接"""
        if workflow_id not in self.connections:
            self.connections[workflow_id] = []
        self.connections[workflow_id].append(websocket)
    
    def remove_connection(self, workflow_id: str, websocket):
        """移除WebSocket连接"""
        if workflow_id in self.connections:
            if websocket in self.connections[workflow_id]:
                self.connections[workflow_id].remove(websocket)
            if not self.connections[workflow_id]:
                del self.connections[workflow_id]
    
    async def handle_event(self, event: WorkflowEvent):
        """通过WebSocket发送事件"""
        if event.workflow_id in self.connections:
            message = event.to_json()
            dead_connections = []
            
            for websocket in self.connections[event.workflow_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    print(f"WebSocket发送失败: {e}")
                    dead_connections.append(websocket)
            
            # 清理断开的连接
            for dead_conn in dead_connections:
                self.remove_connection(event.workflow_id, dead_conn)
    
    def get_interested_events(self) -> List[EventType]:
        """监听状态变更事件"""
        return [
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_PAUSED,
            EventType.WORKFLOW_RESUMED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED,
            EventType.WORKFLOW_CANCELLED,
            EventType.NODE_STARTED,
            EventType.NODE_COMPLETED,
            EventType.NODE_FAILED,
            EventType.STATE_CHANGED,
            EventType.ERROR_OCCURRED
        ]


class DatabaseEventListener(EventListener):
    """数据库事件监听器"""
    
    def __init__(self):
        self.event_buffer: List[WorkflowEvent] = []
        self.buffer_size = 100
    
    async def handle_event(self, event: WorkflowEvent):
        """保存事件到数据库"""
        # 添加到缓冲区
        self.event_buffer.append(event)
        
        # 如果缓冲区满了，批量保存
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_events()
    
    async def _flush_events(self):
        """批量保存事件到数据库"""
        if not self.event_buffer:
            return
        
        try:
            import json
            import logging
            
            logger = logging.getLogger(__name__)
            
            # 将事件序列化为JSON格式保存
            events_data = []
            for event in self.event_buffer:
                event_data = {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "workflow_id": event.workflow_id,
                    "node_id": event.node_id,
                    "data": event.data,
                    "metadata": event.metadata
                }
                events_data.append(event_data)
            
            # 记录事件到日志系统
            logger.info(f"批量保存工作流事件", extra={
                "event_count": len(events_data),
                "events": events_data
            })
            
            self.event_buffer.clear()
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"保存事件到数据库失败: {e}", exc_info=True)
    
    def get_interested_events(self) -> List[EventType]:
        """监听所有事件用于审计"""
        return list(EventType)


class EventBus:
    """事件总线"""
    
    def __init__(self):
        self.listeners: List[EventListener] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    def add_listener(self, listener: EventListener):
        """添加事件监听器"""
        self.listeners.append(listener)
    
    def remove_listener(self, listener: EventListener):
        """移除事件监听器"""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    async def publish_event(self, event: WorkflowEvent):
        """发布事件"""
        await self.event_queue.put(event)
    
    async def start(self):
        """启动事件处理"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self):
        """停止事件处理"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
    
    async def _process_events(self):
        """处理事件队列"""
        while self.is_running:
            try:
                # 等待事件，设置超时避免无限等待
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._dispatch_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"处理事件失败: {e}")
    
    async def _dispatch_event(self, event: WorkflowEvent):
        """分发事件到监听器"""
        for listener in self.listeners:
            try:
                # 检查监听器是否对此事件感兴趣
                if event.event_type in listener.get_interested_events():
                    await listener.handle_event(event)
            except Exception as e:
                print(f"事件监听器处理失败: {e}")


class WorkflowEventEmitter:
    """工作流事件发射器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
    
    async def emit_workflow_started(self, workflow_id: str, data: Optional[Dict[str, Any]] = None):
        """发出工作流开始事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            data=data or {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_workflow_completed(self, workflow_id: str, result: Optional[Dict[str, Any]] = None):
        """发出工作流完成事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_COMPLETED,
            workflow_id=workflow_id,
            data={"result": result} if result else {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_workflow_failed(self, workflow_id: str, error: str, details: Optional[Dict[str, Any]] = None):
        """发出工作流失败事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_FAILED,
            workflow_id=workflow_id,
            data={"error": error, "details": details or {}}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_workflow_paused(self, workflow_id: str, reason: Optional[str] = None):
        """发出工作流暂停事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_PAUSED,
            workflow_id=workflow_id,
            data={"reason": reason} if reason else {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_workflow_resumed(self, workflow_id: str):
        """发出工作流恢复事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_RESUMED,
            workflow_id=workflow_id
        )
        await self.event_bus.publish_event(event)
    
    async def emit_workflow_cancelled(self, workflow_id: str, reason: Optional[str] = None):
        """发出工作流取消事件"""
        event = WorkflowEvent(
            event_type=EventType.WORKFLOW_CANCELLED,
            workflow_id=workflow_id,
            data={"reason": reason} if reason else {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_node_started(self, workflow_id: str, node_name: str, input_data: Optional[Dict[str, Any]] = None):
        """发出节点开始事件"""
        event = WorkflowEvent(
            event_type=EventType.NODE_STARTED,
            workflow_id=workflow_id,
            node_name=node_name,
            data={"input": input_data} if input_data else {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_node_completed(self, workflow_id: str, node_name: str, output_data: Optional[Dict[str, Any]] = None):
        """发出节点完成事件"""
        event = WorkflowEvent(
            event_type=EventType.NODE_COMPLETED,
            workflow_id=workflow_id,
            node_name=node_name,
            data={"output": output_data} if output_data else {}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_node_failed(self, workflow_id: str, node_name: str, error: str):
        """发出节点失败事件"""
        event = WorkflowEvent(
            event_type=EventType.NODE_FAILED,
            workflow_id=workflow_id,
            node_name=node_name,
            data={"error": error}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_state_changed(self, workflow_id: str, old_state: MessagesState, new_state: MessagesState):
        """发出状态变更事件"""
        event = WorkflowEvent(
            event_type=EventType.STATE_CHANGED,
            workflow_id=workflow_id,
            data={
                "old_status": old_state["metadata"].get("status"),
                "new_status": new_state["metadata"].get("status"),
                "changes": self._detect_state_changes(old_state, new_state)
            }
        )
        await self.event_bus.publish_event(event)
    
    async def emit_checkpoint_created(self, workflow_id: str, checkpoint_id: str, metadata: Optional[Dict[str, Any]] = None):
        """发出检查点创建事件"""
        event = WorkflowEvent(
            event_type=EventType.CHECKPOINT_CREATED,
            workflow_id=workflow_id,
            data={"checkpoint_id": checkpoint_id, "metadata": metadata or {}}
        )
        await self.event_bus.publish_event(event)
    
    async def emit_error_occurred(self, workflow_id: str, error: str, node_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """发出错误发生事件"""
        event = WorkflowEvent(
            event_type=EventType.ERROR_OCCURRED,
            workflow_id=workflow_id,
            node_name=node_name,
            data={"error": error, "details": details or {}}
        )
        await self.event_bus.publish_event(event)
    
    def _detect_state_changes(self, old_state: MessagesState, new_state: MessagesState) -> Dict[str, Any]:
        """检测状态变更"""
        changes = {}
        
        # 检查消息数量变化
        old_msg_count = len(old_state.get("messages", []))
        new_msg_count = len(new_state.get("messages", []))
        if old_msg_count != new_msg_count:
            changes["message_count_changed"] = {
                "old": old_msg_count,
                "new": new_msg_count
            }
        
        # 检查元数据变化
        old_meta = old_state.get("metadata", {})
        new_meta = new_state.get("metadata", {})
        
        for key in set(old_meta.keys()) | set(new_meta.keys()):
            if old_meta.get(key) != new_meta.get(key):
                changes[f"metadata.{key}"] = {
                    "old": old_meta.get(key),
                    "new": new_meta.get(key)
                }
        
        return changes


# 创建全局事件系统
event_bus = EventBus()
event_emitter = WorkflowEventEmitter(event_bus)

# 添加默认监听器
logging_listener = LoggingEventListener()
websocket_listener = WebSocketEventListener()
database_listener = DatabaseEventListener()

event_bus.add_listener(logging_listener)
event_bus.add_listener(websocket_listener)
event_bus.add_listener(database_listener)