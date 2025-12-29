"""
分布式消息通信框架
基于NATS JetStream实现智能体间的高可靠消息通信
"""

from .models import (
    Message,
    MessageHeader,
    MessageType,
    MessagePriority,
    DeliveryMode,
    MessageHandler,
    StreamConfig,
    ConnectionState,
    ConnectionMetrics,
    TopicConfig
)
from .protocol import MessageProtocol

__all__ = [
    # 核心模型
    "Message",
    "MessageHeader", 
    "MessageType",
    "MessagePriority",
    "DeliveryMode",
    "MessageHandler",
    "StreamConfig",
    "ConnectionState",
    "ConnectionMetrics",
    "TopicConfig",
    
    # 核心组件
    "MessageProtocol"
]

__version__ = "1.0.0"
