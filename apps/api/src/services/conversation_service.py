"""
对话会话管理服务
负责ReAct智能体的对话上下文管理和持久化
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from src.core.utils.timezone_utils import utc_now, parse_iso_string, from_timestamp, to_utc
from src.core.config import get_settings
from src.core.database import get_db_session
from src.models.database.session import Session as SessionModel

from src.core.logging import get_logger
logger = get_logger(__name__)

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
            created_at = utc_now()
            session_title = title or f"ReAct对话 - {created_at.strftime('%Y-%m-%d %H:%M')}"
            async with get_db_session() as db:
                db.add(
                    SessionModel(
                        id=uuid.UUID(conversation_id),
                        user_id=user_id,
                        title=session_title,
                        status="active",
                        context={"metadata": metadata or {}, "context": {}, "messages": []},
                        agent_config={"agent_type": agent_type},
                        message_count=0,
                        is_active=True,
                        last_activity_at=created_at,
                    )
                )
                await db.commit()

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
                "title": session_title,
                "agent_type": agent_type,
                "metadata": metadata or {},
                "messages": [],
                "context": {},
                "created_at": created_at,
                "updated_at": created_at,
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
                session = await self._load_conversation_from_db(conversation_id)
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
                "created_at": utc_now(),
                "timestamp": utc_now().timestamp()
            }
            
            # 添加到会话
            session["messages"].append(message)
            session["updated_at"] = utc_now()
            
            # 检查上下文长度限制
            await self._manage_context_length(conversation_id)
            await self._save_conversation_to_db(conversation_id)
            
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

    async def update_tool_call_result(
        self,
        conversation_id: str,
        step_id: Optional[str],
        result: Any,
        status: str = "success"
    ) -> bool:
        """更新工具调用结果"""
        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")

            def normalize_result(value: Any) -> Any:
                if value is None:
                    return None
                try:
                    json.dumps(value, ensure_ascii=False)
                    return value
                except TypeError:
                    return str(value)

            safe_result = normalize_result(result)
            target_tool_call = None

            if step_id:
                for message in reversed(session["messages"]):
                    for tool_call in message.get("tool_calls", []):
                        if str(tool_call.get("step_id")) == str(step_id):
                            target_tool_call = tool_call
                            break
                    if target_tool_call:
                        break

            if not target_tool_call:
                for message in reversed(session["messages"]):
                    tool_calls = message.get("tool_calls", [])
                    if not tool_calls:
                        continue
                    for tool_call in reversed(tool_calls):
                        if tool_call.get("status") == "pending" or (
                            tool_call.get("result") is None
                            and tool_call.get("tool_result") is None
                        ):
                            target_tool_call = tool_call
                            break
                    if target_tool_call:
                        break

            if not target_tool_call:
                return False

            target_tool_call["result"] = safe_result
            target_tool_call["tool_result"] = safe_result
            target_tool_call["status"] = status

            session["updated_at"] = utc_now()
            await self._save_conversation_to_db(conversation_id)
            return True

        except Exception as e:
            logger.error(
                "更新工具调用结果失败",
                error=str(e),
                conversation_id=conversation_id,
                step_id=step_id
            )
            return False

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
                    tool_call_id = (
                        message["metadata"].get("tool_call_id")
                        or message["metadata"].get("step_id")
                    )
                    openai_messages.append({
                        "role": "tool",
                        "content": message["content"],
                        "tool_call_id": tool_call_id
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
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")
            
            session["context"].update(context_updates)
            session["updated_at"] = utc_now()
            await self._save_conversation_to_db(conversation_id)
            
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
            if not session:
                session = await self._load_conversation_from_db(conversation_id)
            if not session:
                return

            session["status"] = "closed"
            session["updated_at"] = utc_now()

            await self._save_conversation_to_db(conversation_id)
            self.active_sessions.pop(conversation_id, None)

            logger.info("关闭对话会话", conversation_id=conversation_id)
            
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
            
            return self._build_conversation_summary(session)
            
        except Exception as e:
            logger.error(
                "获取对话摘要失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    def _parse_message_created_at(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return to_utc(value)
        if isinstance(value, (int, float)):
            return from_timestamp(value)
        if isinstance(value, str):
            return parse_iso_string(value)
        return None

    def _normalize_message_created_at(self, value: Any) -> Optional[str]:
        parsed = self._parse_message_created_at(value)
        if parsed:
            return parsed.isoformat()
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return None

    def _build_conversation_summary_from_model(self, model: SessionModel) -> Dict[str, Any]:
        raw = model.context or {}
        raw_messages = raw.get("messages") or []
        if not isinstance(raw_messages, list):
            raw_messages = []

        message_stats = {
            "total": len(raw_messages),
            "user": 0,
            "assistant": 0,
            "system": 0,
            "tool_calls": 0,
            "tool_results": 0,
        }

        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            sender_type = message.get("sender_type")
            if sender_type in ("user", "assistant", "system"):
                message_stats[sender_type] += 1
            message_type = message.get("message_type")
            if message_type == "tool_call":
                message_stats["tool_calls"] += 1
            elif message_type == "tool_result":
                message_stats["tool_results"] += 1

        last_message = None
        for message in reversed(raw_messages):
            if not isinstance(message, dict):
                continue
            if message.get("sender_type") == "system":
                continue
            last_message = {
                "id": message.get("id"),
                "content": message.get("content", ""),
                "sender_type": message.get("sender_type", "assistant"),
                "created_at": self._normalize_message_created_at(message.get("created_at")),
            }
            break

        agent_config = model.agent_config or {}
        agent_type = agent_config.get("agent_type") or "react"
        created_at = model.created_at or utc_now()
        updated_at = model.last_activity_at or model.updated_at or created_at
        context_payload = raw.get("context") or {}
        duration_seconds = 0
        if raw_messages:
            first_message = raw_messages[0] if isinstance(raw_messages[0], dict) else None
            last_raw_message = raw_messages[-1] if isinstance(raw_messages[-1], dict) else None
            start_dt = self._parse_message_created_at(first_message.get("created_at") if first_message else None)
            end_dt = self._parse_message_created_at(last_raw_message.get("created_at") if last_raw_message else None)
            if start_dt and end_dt:
                duration_seconds = max(0, (end_dt - start_dt).total_seconds())
            else:
                duration_seconds = max(0, (updated_at - created_at).total_seconds())

        return {
            "conversation_id": str(model.id),
            "title": model.title,
            "agent_type": agent_type,
            "status": model.status,
            "created_at": created_at,
            "updated_at": updated_at,
            "duration_seconds": duration_seconds,
            "message_stats": message_stats,
            "last_message": last_message,
            "context_keys": list(context_payload.keys()),
            "metadata": raw.get("metadata") or {},
        }

    def _build_conversation_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        messages = session["messages"]
        last_message = None
        for message in reversed(messages):
            if message.get("sender_type") == "system":
                continue
            last_message = {
                "id": message.get("id"),
                "content": message.get("content", ""),
                "sender_type": message.get("sender_type", "assistant"),
                "created_at": message.get("created_at"),
            }
            break

        message_stats = {
            "total": len(messages),
            "user": len([m for m in messages if m["sender_type"] == "user"]),
            "assistant": len([m for m in messages if m["sender_type"] == "assistant"]),
            "system": len([m for m in messages if m["sender_type"] == "system"]),
            "tool_calls": len([m for m in messages if m["message_type"] == "tool_call"]),
            "tool_results": len([m for m in messages if m["message_type"] == "tool_result"])
        }

        if messages:
            start_time = messages[0]["created_at"]
            end_time = messages[-1]["created_at"]
            duration_seconds = (end_time - start_time).total_seconds()
        else:
            duration_seconds = 0

        return {
            "conversation_id": session["conversation_id"],
            "title": session["title"],
            "agent_type": session["agent_type"],
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "duration_seconds": duration_seconds,
            "message_stats": message_stats,
            "last_message": last_message,
            "context_keys": list(session["context"].keys()),
            "metadata": session["metadata"]
        }

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str
    ) -> Dict[str, Any]:
        """更新对话标题"""
        normalized = title.strip()
        if not normalized:
            raise ValueError("对话标题不能为空")

        try:
            session = self.active_sessions.get(conversation_id)
            if not session:
                session = await self._load_conversation_from_db(conversation_id)
                if not session:
                    raise ValueError(f"对话会话不存在: {conversation_id}")

            session["title"] = normalized
            session["updated_at"] = utc_now()
            await self._save_conversation_to_db(conversation_id)

            logger.info(
                "更新对话标题",
                conversation_id=conversation_id,
                title=normalized
            )

            return {
                "conversation_id": conversation_id,
                "title": normalized,
                "updated_at": session["updated_at"]
            }
        except Exception as e:
            logger.error(
                "更新对话标题失败",
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

    async def get_conversation_for_user(
        self,
        conversation_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """校验并获取指定用户的对话会话"""
        session = self.active_sessions.get(conversation_id)
        if session:
            if session.get("user_id") == user_id:
                return session
            logger.warning(
                "对话访问被拒绝",
                conversation_id=conversation_id,
                user_id=user_id
            )
            return None
        return await self._load_conversation_from_db(conversation_id, user_id=user_id)

    async def _load_conversation_from_db(
        self,
        conversation_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """从数据库加载对话会话"""
        try:
            conversation_uuid = uuid.UUID(conversation_id)
            async with get_db_session() as db:
                model = await db.get(SessionModel, conversation_uuid)
                if not model:
                    return None
                if user_id and str(model.user_id) != str(user_id):
                    logger.warning(
                        "对话访问被拒绝",
                        conversation_id=conversation_id,
                        user_id=user_id
                    )
                    return None

                session_data = self._build_session_data_from_model(model)
                self.active_sessions[conversation_id] = session_data

                logger.info(
                    "成功从数据库加载对话会话",
                    conversation_id=conversation_id,
                    message_count=len(session_data["messages"]),
                )

                return session_data
                
        except Exception as e:
            logger.error(
                "从数据库加载对话失败",
                conversation_id=conversation_id,
                error=str(e)
            )
            return None

    def _build_session_data_from_model(self, model: SessionModel) -> Dict[str, Any]:
        conversation_id = str(model.id)
        raw = model.context or {}
        raw_messages = raw.get("messages") or []

        messages: List[Dict[str, Any]] = []
        for msg in raw_messages:
            raw_created_at = msg.get("created_at")
            created_at = parse_iso_string(raw_created_at) if isinstance(raw_created_at, str) else None
            if not created_at:
                created_at = utc_now()

            raw_timestamp = msg.get("timestamp")
            timestamp = raw_timestamp if isinstance(raw_timestamp, (int, float)) else created_at.timestamp()

            messages.append(
                {
                    "id": msg.get("id") or str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "content": msg.get("content", ""),
                    "sender_type": msg.get("sender_type", "assistant"),
                    "message_type": msg.get("message_type", "text"),
                    "metadata": msg.get("metadata") or {},
                    "tool_calls": msg.get("tool_calls") or [],
                    "created_at": created_at,
                    "timestamp": timestamp,
                }
            )

        agent_config = model.agent_config or {}
        agent_type = agent_config.get("agent_type") or "react"

        created_at = model.created_at or utc_now()
        updated_at = model.last_activity_at or model.updated_at or created_at

        return {
            "conversation_id": conversation_id,
            "user_id": model.user_id,
            "title": model.title,
            "agent_type": agent_type,
            "metadata": raw.get("metadata") or {},
            "messages": messages,
            "context": raw.get("context") or {},
            "created_at": created_at,
            "updated_at": updated_at,
            "status": model.status,
        }

    async def _save_conversation_to_db(self, conversation_id: str) -> None:
        """保存对话会话到数据库"""
        session_data = self.active_sessions.get(conversation_id)
        if not session_data:
            return

        try:
            conversation_uuid = uuid.UUID(conversation_id)
            async with get_db_session() as db:
                model = await db.get(SessionModel, conversation_uuid)
                if not model:
                    raise ValueError(f"对话会话不存在: {conversation_id}")

                messages_payload = []
                for msg in session_data["messages"]:
                    created_at = msg.get("created_at")
                    created_at_iso = created_at.isoformat() if hasattr(created_at, "isoformat") else None
                    timestamp = msg.get("timestamp")
                    if not isinstance(timestamp, (int, float)) and hasattr(created_at, "timestamp"):
                        timestamp = created_at.timestamp()

                    messages_payload.append(
                        {
                            "id": msg.get("id"),
                            "content": msg.get("content"),
                            "sender_type": msg.get("sender_type"),
                            "message_type": msg.get("message_type"),
                            "metadata": msg.get("metadata") or {},
                            "tool_calls": msg.get("tool_calls") or [],
                            "created_at": created_at_iso,
                            "timestamp": timestamp,
                        }
                    )

                model.title = session_data.get("title") or model.title
                model.status = session_data.get("status") or model.status
                model.context = {
                    "metadata": session_data.get("metadata") or {},
                    "context": session_data.get("context") or {},
                    "messages": messages_payload,
                }

                agent_config = model.agent_config or {}
                if session_data.get("agent_type"):
                    agent_config["agent_type"] = session_data["agent_type"]
                model.agent_config = agent_config

                model.message_count = len(session_data["messages"])
                model.is_active = model.status == "active"
                model.last_activity_at = session_data.get("updated_at") or utc_now()

                await db.commit()

        except Exception as e:
            logger.error("保存对话会话失败", conversation_id=conversation_id, error=str(e))
            raise

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出用户的对话会话"""
        try:
            from sqlalchemy import select, desc, func
            normalized_query = (query or "").strip()
            max_search_messages = 20
            max_search_chars = 4000

            def normalize_text(value: str) -> str:
                return " ".join(
                    str(value).lower()
                    .replace("/", " ")
                    .replace("_", " ")
                    .replace("-", " ")
                    .split()
                )

            def build_search_target(model: SessionModel) -> str:
                raw = model.context or {}
                raw_messages = raw.get("messages") if isinstance(raw, dict) else []
                if not isinstance(raw_messages, list):
                    raw_messages = []
                parts: List[str] = []
                total_length = 0

                def push(value: Any) -> None:
                    nonlocal total_length
                    if value is None:
                        return
                    text = str(value)
                    if not text:
                        return
                    if max_search_chars and total_length >= max_search_chars:
                        return
                    parts.append(text)
                    total_length += len(text) + 1

                push(model.title or "")
                message_count = 0
                for message in reversed(raw_messages):
                    if message_count >= max_search_messages or (
                        max_search_chars and total_length >= max_search_chars
                    ):
                        break
                    if not isinstance(message, dict):
                        continue
                    content = message.get("content")
                    if content:
                        push(content)
                        message_count += 1
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if max_search_chars and total_length >= max_search_chars:
                                break
                            if not isinstance(tool_call, dict):
                                continue
                            name = tool_call.get("tool_name") or tool_call.get("name")
                            if name:
                                push(name)
                            result = (
                                tool_call.get("result")
                                or tool_call.get("tool_result")
                                or tool_call.get("output")
                                or tool_call.get("error")
                            )
                            if result:
                                push(result)

                return " ".join(parts)

            tokens = []
            if normalized_query:
                normalized_query = normalize_text(normalized_query)
                tokens = normalized_query.split()

            if tokens:
                async with get_db_session() as db:
                    result = await db.execute(
                        select(SessionModel)
                        .where(SessionModel.user_id == user_id)
                        .order_by(desc(SessionModel.updated_at))
                    )
                    models = result.scalars().all()

                summaries: List[Dict[str, Any]] = []
                matched_total = 0
                for model in models:
                    search_target = normalize_text(build_search_target(model))
                    if not all(token in search_target for token in tokens):
                        continue
                    if matched_total >= offset and len(summaries) < limit:
                        summary = self._build_conversation_summary_from_model(model)
                        summaries.append(summary)
                    matched_total += 1

                return summaries, matched_total

            async with get_db_session() as db:
                total_result = await db.execute(
                    select(func.count(SessionModel.id)).where(SessionModel.user_id == user_id)
                )
                total = total_result.scalar_one()
                result = await db.execute(
                    select(SessionModel)
                    .where(SessionModel.user_id == user_id)
                    .order_by(desc(SessionModel.updated_at))
                    .offset(offset)
                    .limit(limit)
                )
                models = result.scalars().all()

            summaries = []
            for model in models:
                summaries.append(self._build_conversation_summary_from_model(model))
            return summaries, total
            
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
            current_time = utc_now().timestamp()
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
