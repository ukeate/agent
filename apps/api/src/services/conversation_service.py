"""
对话会话管理服务
负责ReAct智能体的对话上下文管理和持久化
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import structlog

from ..core.config import get_settings
from ..db.models import Conversation, Message, Task

logger = structlog.get_logger(__name__)

class ConversationService:
    """对话会话管理服务"""

    def __init__(self):
        self.settings = get_settings()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.max_context_length = self.settings.MAX_CONTEXT_LENGTH
        self.session_timeout = self.settings.SESSION_TIMEOUT_MINUTES * 60  # 转换为秒

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        agent_type: str = "react",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建新的对话会话"""
        conversation_id = str(uuid.uuid4())
        
        try:
            # 创建数据库记录
            conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                title=title or f"ReAct对话 - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}",
                agent_type=agent_type,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # 这里应该保存到数据库，但由于没有数据库连接，我们先用内存存储
            logger.info(
                "创建对话会话",
                conversation_id=conversation_id,
                user_id=user_id,
                agent_type=agent_type
            )
            
            # 在内存中创建会话状态
            self.active_sessions[conversation_id] = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "title": conversation.title,
                "agent_type": agent_type,
                "metadata": metadata or {},
                "messages": [],
                "context": {},
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "status": "active"
            }
            
            return conversation_id
            
        except Exception as e:
            logger.error(
                "创建对话会话失败",
                error=str(e),
                user_id=user_id
            )
            raise

    async def add_message(
        self,
        conversation_id: str,
        content: str,
        sender_type: str = "user",
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """添加消息到对话"""
        message_id = str(uuid.uuid4())
        
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                raise ValueError(f"对话会话不存在: {conversation_id}")
            
            message = {
                "id": message_id,
                "conversation_id": conversation_id,
                "content": content,
                "sender_type": sender_type,  # user, assistant, system
                "message_type": message_type,  # text, tool_call, tool_result
                "metadata": metadata or {},
                "tool_calls": tool_calls or [],
                "created_at": datetime.now(timezone.utc),
                "timestamp": datetime.now(timezone.utc).timestamp()
            }
            
            # 添加到会话
            session["messages"].append(message)
            session["updated_at"] = datetime.now(timezone.utc)
            
            # 检查上下文长度限制
            await self._manage_context_length(conversation_id)
            
            logger.info(
                "添加消息到对话",
                conversation_id=conversation_id,
                message_id=message_id,
                sender_type=sender_type,
                content_length=len(content)
            )
            
            return message_id
            
        except Exception as e:
            logger.error(
                "添加消息失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        include_system: bool = True
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                # 尝试从数据库加载
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")
            
            messages = session["messages"]
            
            # 过滤系统消息
            if not include_system:
                messages = [msg for msg in messages if msg["sender_type"] != "system"]
            
            # 应用限制
            if limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(
                "获取对话历史失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")
            
            # 构建OpenAI格式的消息列表
            openai_messages = []
            
            for message in session["messages"]:
                if message["sender_type"] == "user":
                    openai_messages.append({
                        "role": "user",
                        "content": message["content"]
                    })
                elif message["sender_type"] == "assistant":
                    if message["message_type"] == "tool_call":
                        # 工具调用消息
                        openai_messages.append({
                            "role": "assistant",
                            "content": message["content"],
                            "tool_calls": message.get("tool_calls", [])
                        })
                    else:
                        # 普通助手消息
                        openai_messages.append({
                            "role": "assistant", 
                            "content": message["content"]
                        })
                elif message["sender_type"] == "system":
                    openai_messages.append({
                        "role": "system",
                        "content": message["content"]
                    })
                elif message["message_type"] == "tool_result":
                    # 工具结果消息
                    openai_messages.append({
                        "role": "tool",
                        "content": message["content"],
                        "tool_call_id": message["metadata"].get("tool_call_id")
                    })
            
            return {
                "conversation_id": conversation_id,
                "messages": openai_messages,
                "session_context": session["context"],
                "metadata": session["metadata"],
                "message_count": len(session["messages"]),
                "updated_at": session["updated_at"]
            }
            
        except Exception as e:
            logger.error(
                "获取对话上下文失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def update_conversation_context(
        self,
        conversation_id: str,
        context_updates: Dict[str, Any]
    ) -> None:
        """更新对话上下文"""
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                raise ValueError(f"对话会话不存在: {conversation_id}")
            
            session["context"].update(context_updates)
            session["updated_at"] = datetime.now(timezone.utc)
            
            logger.info(
                "更新对话上下文",
                conversation_id=conversation_id,
                updates=list(context_updates.keys())
            )
            
        except Exception as e:
            logger.error(
                "更新对话上下文失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def close_conversation(self, conversation_id: str) -> None:
        """关闭对话会话"""
        try:
            session = self.active_sessions.get(conversation_id)
            if session:
                session["status"] = "closed"
                session["updated_at"] = datetime.now(timezone.utc)
                
                # 保存到数据库
                await self._save_conversation_to_db(conversation_id)
                
                # 从内存中移除
                del self.active_sessions[conversation_id]
                
                logger.info(
                    "关闭对话会话",
                    conversation_id=conversation_id
                )
            
        except Exception as e:
            logger.error(
                "关闭对话会话失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话摘要"""
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")
            
            messages = session["messages"]
            
            # 统计消息类型
            message_stats = {
                "total": len(messages),
                "user": len([m for m in messages if m["sender_type"] == "user"]),
                "assistant": len([m for m in messages if m["sender_type"] == "assistant"]),
                "system": len([m for m in messages if m["sender_type"] == "system"]),
                "tool_calls": len([m for m in messages if m["message_type"] == "tool_call"]),
                "tool_results": len([m for m in messages if m["message_type"] == "tool_result"])
            }
            
            # 计算会话时长
            if messages:
                start_time = messages[0]["created_at"]
                end_time = messages[-1]["created_at"]
                duration_seconds = (end_time - start_time).total_seconds()
            else:
                duration_seconds = 0
            
            return {
                "conversation_id": conversation_id,
                "title": session["title"],
                "agent_type": session["agent_type"],
                "status": session["status"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "duration_seconds": duration_seconds,
                "message_stats": message_stats,
                "context_keys": list(session["context"].keys()),
                "metadata": session["metadata"]
            }
            
        except Exception as e:
            logger.error(
                "获取对话摘要失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def _manage_context_length(self, conversation_id: str) -> None:
        """管理上下文长度，防止超出限制"""
        session = self.active_sessions.get(conversation_id)
        if not session:
            return
        
        messages = session["messages"]
        
        # 简单的token估算：每个字符约1个token
        total_tokens = sum(len(msg["content"]) for msg in messages)
        
        if total_tokens > self.max_context_length:
            # 移除最早的非系统消息，保留最近的对话
            keep_system_messages = [msg for msg in messages if msg["sender_type"] == "system"]
            non_system_messages = [msg for msg in messages if msg["sender_type"] != "system"]
            
            # 保留最近的消息，直到token数量在限制内
            truncated_messages = []
            current_tokens = 0
            
            for message in reversed(non_system_messages):
                msg_tokens = len(message["content"])
                if current_tokens + msg_tokens > (self.max_context_length * 0.8):  # 留20%缓冲
                    break
                truncated_messages.insert(0, message)
                current_tokens += msg_tokens
            
            # 重新组合消息
            session["messages"] = keep_system_messages + truncated_messages
            
            logger.info(
                "截断对话上下文",
                conversation_id=conversation_id,
                original_messages=len(messages),
                truncated_messages=len(session["messages"]),
                estimated_tokens=current_tokens
            )

    async def _load_conversation_from_db(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """从数据库加载对话会话"""
        try:
            from sqlalchemy.ext.asyncio import AsyncSession
            from ..core.database import async_session_factory
            
            if not async_session_factory:
                logger.warning("数据库会话工厂未初始化")
                return None
                
            async with async_session_factory() as session:
                # 加载对话记录
                conversation = await session.get(Conversation, conversation_id)
                if not conversation:
                    return None
                
                # 加载消息记录
                from sqlalchemy import select
                messages_result = await session.execute(
                    select(Message).where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                )
                messages = messages_result.scalars().all()
                
                # 构建会话数据
                session_data = {
                    "conversation_id": conversation_id,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                    "metadata": conversation.metadata or {},
                    "messages": [
                        {
                            "id": str(msg.id),
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.created_at.isoformat(),
                            "metadata": msg.metadata or {}
                        }
                        for msg in messages
                    ]
                }
                
                logger.info(
                    "成功从数据库加载对话会话",
                    conversation_id=conversation_id,
                    message_count=len(messages)
                )
                
                return session_data
                
        except Exception as e:
            logger.error(
                "从数据库加载对话失败",
                conversation_id=conversation_id,
                error=str(e)
            )
            return None

    async def _save_conversation_to_db(self, conversation_id: str) -> None:
        """保存对话会话到数据库"""
        # 这里应该保存到数据库
        logger.warning(
            "数据库保存功能未实现",
            conversation_id=conversation_id
        )

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出用户的对话会话"""
        try:
            # 从内存中过滤用户的会话
            user_sessions = [
                session for session in self.active_sessions.values()
                if session["user_id"] == user_id
            ]
            
            # 按更新时间排序
            user_sessions.sort(key=lambda x: x["updated_at"], reverse=True)
            
            # 应用分页
            paginated_sessions = user_sessions[offset:offset + limit]
            
            # 返回摘要信息
            summaries = []
            for session in paginated_sessions:
                summary = await self.get_conversation_summary(session["conversation_id"])
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(
                "列出对话会话失败",
                error=str(e),
                user_id=user_id
            )
            raise

    async def cleanup_expired_sessions(self) -> int:
        """清理过期的内存会话"""
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            expired_sessions = []
            
            for conversation_id, session in self.active_sessions.items():
                last_updated = session["updated_at"].timestamp()
                if current_time - last_updated > self.session_timeout:
                    expired_sessions.append(conversation_id)
            
            # 保存并移除过期会话
            for conversation_id in expired_sessions:
                await self._save_conversation_to_db(conversation_id)
                del self.active_sessions[conversation_id]
            
            logger.info(
                "清理过期会话",
                expired_count=len(expired_sessions)
            )
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(
                "清理过期会话失败",
                error=str(e)
            )
            return 0


# 单例模式的服务实例
_conversation_service: Optional[ConversationService] = None

async def get_conversation_service() -> ConversationService:
    """获取对话服务实例"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service