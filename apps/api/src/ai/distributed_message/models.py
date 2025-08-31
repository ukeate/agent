"""
分布式消息通信模块 - 数据模型
实现智能体间的标准化消息格式和通信协议
"""

import asyncio
import json
import uuid
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型枚举"""
    # 基础通信
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"
    
    # 任务协调
    TASK_REQUEST = "task_request"
    TASK_ACCEPT = "task_accept"
    TASK_REJECT = "task_reject"
    TASK_RESULT = "task_result"
    TASK_STATUS = "task_status"
    TASK_CANCEL = "task_cancel"
    
    # 协作通信
    COLLABORATION_INVITE = "collaboration_invite"
    COLLABORATION_JOIN = "collaboration_join"
    COLLABORATION_LEAVE = "collaboration_leave"
    COLLABORATION_UPDATE = "collaboration_update"
    
    # 资源管理
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_OFFER = "resource_offer"
    RESOURCE_RELEASE = "resource_release"
    RESOURCE_STATUS = "resource_status"
    
    # 系统事件
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_UPDATE = "config_update"
    
    # 数据传输
    DATA_CHUNK = "data_chunk"
    DATA_STREAM_START = "data_stream_start"
    DATA_STREAM_CHUNK = "data_stream_chunk"
    DATA_STREAM_END = "data_stream_end"
    DATA_SYNC = "data_sync"
    
    # 广播和多播
    BROADCAST = "broadcast"
    MULTICAST = "multicast"


class MessagePriority(int, Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class DeliveryMode(str, Enum):
    """消息投递模式"""
    AT_MOST_ONCE = "at_most_once"      # 最多一次
    AT_LEAST_ONCE = "at_least_once"    # 至少一次
    EXACTLY_ONCE = "exactly_once"      # 恰好一次


@dataclass
class MessageHeader:
    """消息头"""
    message_id: str
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None  # 生存时间(秒)
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    retry_count: int = 0
    max_retries: int = 3
    
    # 高级通信字段
    stream_id: Optional[str] = None  # 数据流ID
    multicast_group: Optional[str] = None  # 多播组ID


@dataclass
class Message:
    """智能体消息"""
    header: MessageHeader
    sender_id: str
    receiver_id: Optional[str]  # None表示广播消息
    message_type: MessageType
    payload: Dict[str, Any]
    
    # 路由信息
    topic: Optional[str] = None
    routing_key: Optional[str] = None
    
    # 元数据
    schema_version: str = "1.0"
    encoding: str = "json"
    
    def to_bytes(self) -> bytes:
        """序列化为字节"""
        try:
            data = {
                "header": asdict(self.header),
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id,
                "message_type": self.message_type.value,
                "payload": self.payload,
                "topic": self.topic,
                "routing_key": self.routing_key,
                "schema_version": self.schema_version,
                "encoding": self.encoding
            }
            
            # 处理datetime序列化
            data["header"]["timestamp"] = self.header.timestamp.isoformat()
            
            return json.dumps(data, ensure_ascii=False).encode('utf-8')
            
        except Exception as e:
            logger.error(f"消息序列化失败: {e}")
            raise
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """从字节反序列化"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            
            # 重建MessageHeader
            header_data = json_data["header"]
            header_data["timestamp"] = datetime.fromisoformat(header_data["timestamp"])
            header_data["priority"] = MessagePriority(header_data["priority"])
            header_data["delivery_mode"] = DeliveryMode(header_data["delivery_mode"])
            
            header = MessageHeader(**header_data)
            
            return cls(
                header=header,
                sender_id=json_data["sender_id"],
                receiver_id=json_data["receiver_id"],
                message_type=MessageType(json_data["message_type"]),
                payload=json_data["payload"],
                topic=json_data.get("topic"),
                routing_key=json_data.get("routing_key"),
                schema_version=json_data.get("schema_version", "1.0"),
                encoding=json_data.get("encoding", "json")
            )
            
        except Exception as e:
            logger.error(f"消息反序列化失败: {e}")
            raise


class MessageHandler:
    """消息处理器"""
    
    def __init__(
        self, 
        message_type: MessageType,
        handler: Callable[[Message], Any],
        is_async: bool = True,
        max_concurrent: int = 10
    ):
        self.message_type = message_type
        self.handler = handler
        self.is_async = is_async
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            "handled": 0,
            "errors": 0,
            "last_handled": None
        }
    
    async def handle_message(self, message: Message):
        """处理消息"""
        async with self.semaphore:
            try:
                start_time = time.time()
                
                if self.is_async:
                    result = await self.handler(message)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.handler, message)
                
                # 更新统计信息
                self.stats["handled"] += 1
                self.stats["last_handled"] = datetime.now()
                
                processing_time = time.time() - start_time
                logger.debug(f"消息 {message.header.message_id} 处理完成, 耗时: {processing_time:.3f}s")
                
                return result
                
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"消息处理失败 {message.header.message_id}: {e}")
                raise


@dataclass
class StreamConfig:
    """JetStream流配置"""
    
    name: str
    subjects: List[str]
    retention_policy: str = "limits"
    max_messages: int = 1000000
    max_bytes: int = 1024*1024*1024  # 1GB
    max_age: int = 7*24*3600  # 7天
    storage: str = "file"
    replicas: int = 3
    
    def to_nats_config(self) -> Dict[str, Any]:
        """转换为NATS配置格式"""
        return {
            "name": self.name,
            "subjects": self.subjects,
            "retention": self.retention_policy,
            "max_msgs": self.max_messages,
            "max_bytes": self.max_bytes,
            "max_age": self.max_age,
            "storage": self.storage,
            "num_replicas": self.replicas
        }


class ConnectionState(str, Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass
class ConnectionMetrics:
    """连接指标"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    active_subscriptions: int = 0
    pending_requests: int = 0
    connection_reconnects: int = 0
    connection_errors: int = 0
    last_heartbeat: Optional[datetime] = None
    
    def reset(self):
        """重置指标"""
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class TopicConfig:
    """主题配置"""
    direct_messages: str
    broadcast: str
    group_messages: str
    task_coordination: str
    resource_management: str
    system_events: str
    data_streams: str
    
    @classmethod
    def create_for_agent(cls, agent_id: str, cluster_name: str) -> 'TopicConfig':
        """为智能体创建主题配置"""
        return cls(
            direct_messages=f"agents.direct.{agent_id}",
            broadcast=f"agents.broadcast.{cluster_name}",
            group_messages=f"agents.group.{cluster_name}",
            task_coordination=f"agents.tasks.{cluster_name}",
            resource_management=f"agents.resources.{cluster_name}",
            system_events=f"agents.events.{cluster_name}",
            data_streams=f"agents.streams.{cluster_name}"
        )