"""
情感智能系统模块间通信协议实现
实现高效、可靠的模块间数据传输和事件通知机制
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor

from .core_interfaces import UnifiedEmotionalData, EmotionalIntelligenceResponse


class MessageType(str, Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class ModuleType(str, Enum):
    """模块类型"""
    EMOTION_RECOGNITION = "emotion_recognition"
    STATE_MODELING = "state_modeling"
    EMPATHY_GENERATION = "empathy_generation"
    MEMORY_MANAGEMENT = "memory_management"
    DECISION_ENGINE = "decision_engine"
    SOCIAL_ANALYSIS = "social_analysis"
    API_GATEWAY = "api_gateway"
    DATA_FLOW_MANAGER = "data_flow_manager"
    SYSTEM_MONITOR = "system_monitor"


class Priority(int, Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MessageHeader:
    """消息头部"""
    message_id: str
    message_type: MessageType
    source_module: ModuleType
    target_module: Optional[ModuleType]
    priority: Priority
    timestamp: datetime
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class Message:
    """通信消息"""
    header: MessageHeader
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "header": asdict(self.header),
            "payload": self.payload,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        header_data = data["header"]
        header = MessageHeader(
            message_id=header_data["message_id"],
            message_type=MessageType(header_data["message_type"]),
            source_module=ModuleType(header_data["source_module"]),
            target_module=ModuleType(header_data["target_module"]) if header_data.get("target_module") else None,
            priority=Priority(header_data["priority"]),
            timestamp=datetime.fromisoformat(header_data["timestamp"]),
            correlation_id=header_data.get("correlation_id"),
            reply_to=header_data.get("reply_to"),
            expires_at=datetime.fromisoformat(header_data["expires_at"]) if header_data.get("expires_at") else None
        )
        return cls(
            header=header,
            payload=data["payload"],
            metadata=data.get("metadata", {})
        )


class MessageHandler(ABC):
    """消息处理器抽象基类"""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> Optional[Message]:
        """处理消息"""
        pass
    
    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """判断是否能处理消息"""
        pass


class EventBus:
    """事件总线"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger(__name__)
    
    def subscribe(self, event_type: str, handler: Callable):
        """订阅事件"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
    
    async def publish(self, event_type: str, event_data: Any):
        """发布事件"""
        if event_type in self._subscribers:
            tasks = []
            for handler in self._subscribers[event_type]:
                try:
                    task = handler(event_data)
                    if asyncio.iscoroutine(task):
                        tasks.append(task)
                    else:
                        # 同步处理器
                        pass
                except Exception as e:
                    self._logger.error(f"Event handler error: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class MessageQueue:
    """消息队列"""
    
    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._dead_letter_queue: List[Message] = []
        self._logger = logging.getLogger(__name__)
    
    async def enqueue(self, message: Message) -> bool:
        """入队消息"""
        try:
            # 检查消息是否过期
            if message.header.expires_at and message.header.expires_at < datetime.now():
                self._logger.warning(f"Message {message.header.message_id} expired")
                return False
            
            priority = -message.header.priority.value  # 负数用于优先级队列
            await self._queue.put((priority, message.header.timestamp, message))
            return True
        except asyncio.QueueFull:
            self._logger.error("Message queue full")
            return False
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[Message]:
        """出队消息"""
        try:
            priority, timestamp, message = await asyncio.wait_for(
                self._queue.get(), timeout=timeout
            )
            return message
        except asyncio.TimeoutError:
            return None
    
    def move_to_dead_letter(self, message: Message, reason: str):
        """移动到死信队列"""
        message.metadata = message.metadata or {}
        message.metadata["dead_letter_reason"] = reason
        message.metadata["dead_letter_timestamp"] = datetime.now().isoformat()
        self._dead_letter_queue.append(message)
        self._logger.warning(f"Message {message.header.message_id} moved to dead letter queue: {reason}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()


class CommunicationProtocol:
    """通信协议管理器"""
    
    def __init__(self, module_type: ModuleType):
        self.module_type = module_type
        self._message_handlers: List[MessageHandler] = []
        self._message_queue = MessageQueue()
        self._event_bus = EventBus()
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._heartbeat_interval = 30  # 秒
        self._is_running = False
        self._logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # 性能指标
        self._metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_processed": 0,
            "processing_errors": 0,
            "average_processing_time": 0.0
        }
    
    def add_message_handler(self, handler: MessageHandler):
        """添加消息处理器"""
        self._message_handlers.append(handler)
    
    def remove_message_handler(self, handler: MessageHandler):
        """移除消息处理器"""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    async def start(self):
        """启动协议服务"""
        self._is_running = True
        await asyncio.gather(
            self._message_processor(),
            self._heartbeat_sender(),
            return_exceptions=True
        )
    
    async def stop(self):
        """停止协议服务"""
        self._is_running = False
        self._executor.shutdown(wait=True)
    
    async def send_request(
        self, 
        target_module: ModuleType,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """发送请求并等待响应"""
        message_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        message = Message(
            header=MessageHeader(
                message_id=message_id,
                message_type=MessageType.REQUEST,
                source_module=self.module_type,
                target_module=target_module,
                priority=priority,
                timestamp=datetime.now(),
                correlation_id=correlation_id,
                expires_at=datetime.now() + timedelta(seconds=timeout)
            ),
            payload=payload
        )
        
        # 创建响应Future
        response_future = asyncio.Future()
        self._pending_responses[correlation_id] = response_future
        
        try:
            # 发送消息
            await self._send_message(message)
            self._metrics["messages_sent"] += 1
            
            # 等待响应
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._logger.error(f"Request timeout: {message_id}")
            return None
        finally:
            self._pending_responses.pop(correlation_id, None)
    
    async def send_response(
        self, 
        request_message: Message,
        payload: Dict[str, Any]
    ):
        """发送响应"""
        response_message = Message(
            header=MessageHeader(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RESPONSE,
                source_module=self.module_type,
                target_module=request_message.header.source_module,
                priority=request_message.header.priority,
                timestamp=datetime.now(),
                correlation_id=request_message.header.correlation_id
            ),
            payload=payload
        )
        
        await self._send_message(response_message)
        self._metrics["messages_sent"] += 1
    
    async def send_event(
        self, 
        event_type: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL
    ):
        """发送事件"""
        message = Message(
            header=MessageHeader(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.EVENT,
                source_module=self.module_type,
                target_module=None,
                priority=priority,
                timestamp=datetime.now()
            ),
            payload={
                "event_type": event_type,
                "data": payload
            }
        )
        
        await self._send_message(message)
        await self._event_bus.publish(event_type, payload)
        self._metrics["messages_sent"] += 1
    
    async def _send_message(self, message: Message):
        """发送消息到队列"""
        await self._message_queue.enqueue(message)
    
    async def _message_processor(self):
        """消息处理循环"""
        while self._is_running:
            try:
                message = await self._message_queue.dequeue(timeout=1.0)
                if message:
                    start_time = datetime.now()
                    await self._process_message(message)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # 更新性能指标
                    self._metrics["messages_processed"] += 1
                    self._metrics["average_processing_time"] = (
                        self._metrics["average_processing_time"] * 0.9 + processing_time * 0.1
                    )
            except Exception as e:
                self._logger.error(f"Message processing error: {e}")
                self._metrics["processing_errors"] += 1
    
    async def _process_message(self, message: Message):
        """处理单个消息"""
        try:
            self._metrics["messages_received"] += 1
            
            # 处理响应消息
            if message.header.message_type == MessageType.RESPONSE:
                correlation_id = message.header.correlation_id
                if correlation_id in self._pending_responses:
                    self._pending_responses[correlation_id].set_result(message.payload)
                return
            
            # 处理事件消息
            if message.header.message_type == MessageType.EVENT:
                event_type = message.payload.get("event_type")
                event_data = message.payload.get("data")
                if event_type:
                    await self._event_bus.publish(event_type, event_data)
                return
            
            # 处理请求消息
            handler_found = False
            for handler in self._message_handlers:
                if handler.can_handle(message):
                    try:
                        response = await handler.handle_message(message)
                        if response:
                            await self.send_response(message, response.payload)
                        handler_found = True
                        break
                    except Exception as e:
                        self._logger.error(f"Handler error: {e}")
                        error_response = {
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        await self.send_response(message, error_response)
            
            if not handler_found:
                self._logger.warning(f"No handler found for message: {message.header.message_id}")
                self._message_queue.move_to_dead_letter(message, "No handler found")
                
        except Exception as e:
            self._logger.error(f"Message processing error: {e}")
            self._message_queue.move_to_dead_letter(message, f"Processing error: {e}")
    
    async def _heartbeat_sender(self):
        """心跳发送循环"""
        while self._is_running:
            try:
                heartbeat_message = Message(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.HEARTBEAT,
                        source_module=self.module_type,
                        target_module=None,
                        priority=Priority.LOW,
                        timestamp=datetime.now()
                    ),
                    payload={
                        "module_type": self.module_type.value,
                        "status": "healthy",
                        "metrics": self._metrics.copy()
                    }
                )
                
                await self._event_bus.publish("system.heartbeat", heartbeat_message.payload)
                await asyncio.sleep(self._heartbeat_interval)
            except Exception as e:
                self._logger.error(f"Heartbeat error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self._metrics,
            "queue_size": self._message_queue.get_queue_size(),
            "dead_letter_count": len(self._message_queue._dead_letter_queue),
            "pending_responses": len(self._pending_responses),
            "uptime": datetime.now().isoformat(),
            "module_type": self.module_type.value
        }


# 具体的消息处理器实现示例

class EmotionRecognitionHandler(MessageHandler):
    """情感识别处理器"""
    
    def __init__(self, recognition_engine):
        self.recognition_engine = recognition_engine
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """处理情感识别请求"""
        if message.payload.get("action") == "recognize_emotion":
            input_data = message.payload.get("input_data", {})
            try:
                result = await self.recognition_engine.recognize_emotion(input_data)
                return Message(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.RESPONSE,
                        source_module=ModuleType.EMOTION_RECOGNITION,
                        target_module=message.header.source_module,
                        priority=message.header.priority,
                        timestamp=datetime.now(),
                        correlation_id=message.header.correlation_id
                    ),
                    payload={"result": result, "success": True}
                )
            except Exception as e:
                return Message(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        message_type=MessageType.ERROR,
                        source_module=ModuleType.EMOTION_RECOGNITION,
                        target_module=message.header.source_module,
                        priority=message.header.priority,
                        timestamp=datetime.now(),
                        correlation_id=message.header.correlation_id
                    ),
                    payload={"error": str(e), "success": False}
                )
        return None
    
    def can_handle(self, message: Message) -> bool:
        """判断是否能处理消息"""
        return (message.header.target_module == ModuleType.EMOTION_RECOGNITION and
                message.header.message_type == MessageType.REQUEST)


class DataSyncHandler(MessageHandler):
    """数据同步处理器"""
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """处理数据同步请求"""
        action = message.payload.get("action")
        
        if action == "sync_emotional_data":
            emotional_data = UnifiedEmotionalData(**message.payload.get("data", {}))
            # 实际的数据同步逻辑
            success = await self._sync_data(emotional_data)
            
            return Message(
                header=MessageHeader(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    source_module=ModuleType.DATA_FLOW_MANAGER,
                    target_module=message.header.source_module,
                    priority=message.header.priority,
                    timestamp=datetime.now(),
                    correlation_id=message.header.correlation_id
                ),
                payload={"success": success}
            )
        return None
    
    def can_handle(self, message: Message) -> bool:
        """判断是否能处理消息"""
        return (message.header.target_module == ModuleType.DATA_FLOW_MANAGER and
                message.header.message_type == MessageType.REQUEST)
    
    async def _sync_data(self, emotional_data: UnifiedEmotionalData) -> bool:
        """同步情感数据"""
        # 这里实现实际的数据同步逻辑
        return True


# 协议工厂
class ProtocolFactory:
    """协议工厂"""
    
    @staticmethod
    def create_protocol(module_type: ModuleType) -> CommunicationProtocol:
        """创建通信协议实例"""
        protocol = CommunicationProtocol(module_type)
        
        # 根据模块类型添加特定的处理器
        if module_type == ModuleType.EMOTION_RECOGNITION:
            # protocol.add_message_handler(EmotionRecognitionHandler(recognition_engine))
            pass
        elif module_type == ModuleType.DATA_FLOW_MANAGER:
            protocol.add_message_handler(DataSyncHandler())
        
        return protocol