"""
数据库模型定义
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class ConversationStatus(str, Enum):
    """对话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"

class SenderType(str, Enum):
    """发送者类型枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Conversation:
    """对话模型"""
    id: str
    user_id: str
    title: str
    agent_type: str = "react"
    status: ConversationStatus = ConversationStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Message:
    """消息模型"""
    id: str
    conversation_id: str
    content: str
    sender_type: SenderType
    message_type: MessageType = MessageType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Task:
    """任务模型"""
    id: str
    conversation_id: str
    description: str
    task_type: str = "general"
    status: str = "pending"
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None