"""
多智能体协作服务
"""

from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
import json
import re
from src.ai.autogen import (
    GroupChatManager,
    ConversationSession,
    create_default_agents,
    create_agent_from_config,
    AgentConfig,
    AgentRole,
    ConversationConfig,
)
from src.ai.autogen.config import AGENT_CONFIGS
from src.ai.autogen.agents import BaseAutoGenAgent

from src.core.logging import get_logger
logger = get_logger(__name__)

class MultiAgentService:
    """多智能体协作服务"""
    
    def __init__(self):
        self.group_chat_manager = GroupChatManager()
        self._active_sessions: Dict[str, ConversationSession] = {}
        self._conversation_ws_sessions: Dict[str, str] = {}

    def bind_ws_session(self, conversation_id: str, ws_session_id: str):
        self._conversation_ws_sessions[conversation_id] = ws_session_id

    def get_ws_session_id(self, conversation_id: str) -> Optional[str]:
        return self._conversation_ws_sessions.get(conversation_id)
    
    async def create_multi_agent_conversation(
        self,
        initial_message: str,
        agent_roles: Optional[List[AgentRole]] = None,
        conversation_config: Optional[ConversationConfig] = None,
        user_context: Optional[str] = None,
        websocket_callback = None,
    ) -> Dict[str, Any]:
        """创建多智能体对话"""
        try:
            # 创建参与的智能体
            participants = await self._create_participants(agent_roles)
            
            # 创建对话会话
            session = await self.group_chat_manager.create_session(
                participants=participants,
                config=conversation_config or ConversationConfig(),
                initial_topic=user_context,
            )
            
            # 缓存活跃会话
            self._active_sessions[session.session_id] = session
            
            # 启动对话（立即返回，不等待完成）
            result = await session.start_conversation(initial_message, websocket_callback)
            
            logger.info(
                "多智能体对话创建成功",
                session_id=session.session_id,
                participant_count=len(participants),
                initial_message_length=len(initial_message),
            )

            return {
                "conversation_id": session.session_id,
                "status": "active",  # 立即返回active状态
                "participants": [
                    {
                        "name": agent.config.name,
                        "role": agent.config.role,
                        "capabilities": agent.config.capabilities,
                    }
                    for agent in participants
                ],
                "created_at": session.created_at.isoformat(),
                "config": {
                    "max_rounds": session.config.max_rounds,
                    "timeout_seconds": session.config.timeout_seconds,
                },
                "initial_status": result,
            }
            
        except Exception as e:
            logger.error(
                "创建多智能体对话失败",
                error=str(e),
                initial_message=initial_message[:100] if initial_message else None,
            )
            raise
    
    async def _create_participants(
        self, 
        agent_roles: Optional[List[AgentRole]] = None
    ) -> List[BaseAutoGenAgent]:
        """创建参与对话的智能体"""
        if not agent_roles:
            # 使用默认智能体组合
            return create_default_agents()
        
        participants = []
        for role in agent_roles:
            try:
                config = AGENT_CONFIGS.get(role)
                agent = create_agent_from_config(
                    config
                    or AgentConfig(
                        name=f"{role.value}_agent",
                        role=role,
                        system_prompt=self._get_role_system_prompt(role),
                        capabilities=self._get_role_capabilities(role),
                    )
                )
                participants.append(agent)
            except Exception as e:
                logger.error(
                    "创建智能体失败",
                    role=role,
                    error=str(e)
                )
                continue
        
        if not participants:
            # 如果没有成功创建任何智能体，使用默认组合
            logger.warning("使用默认智能体组合")
            return create_default_agents()
        
        return participants
    
    def _get_role_system_prompt(self, role: AgentRole) -> str:
        """获取角色系统提示词"""
        config = AGENT_CONFIGS.get(role)
        return config.system_prompt if config else "你是一位AI助手，请协助完成任务。"
    
    def _get_role_capabilities(self, role: AgentRole) -> List[str]:
        """获取角色能力列表"""
        config = AGENT_CONFIGS.get(role)
        return config.capabilities if config else ["通用AI助手"]
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            # 尝试从管理器获取
            session = await self.group_chat_manager.get_session(conversation_id)
        
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        status = session.get_status()
        status.pop("session_id", None)
        status["conversation_id"] = conversation_id
        
        # 添加实时统计信息
        status.update({
            "real_time_stats": {
                "active_speaker": self._get_current_speaker(session),
                "next_expected_speaker": self._get_next_speaker(session),
                "conversation_progress": self._calculate_progress(session),
                "estimated_completion": self._estimate_completion(session),
            }
        })
        
        return status
    
    def _get_current_speaker(self, session: ConversationSession) -> Optional[str]:
        """获取当前发言者"""
        if session.messages:
            last_message = session.messages[-1]
            return last_message.get("sender")
        return None
    
    def _get_next_speaker(self, session: ConversationSession) -> Optional[str]:
        """预测下一个发言者"""
        # 简单轮转逻辑
        if session.participants:
            next_index = (session.current_speaker_index + 1) % len(session.participants)
            return session.participants[next_index].config.name
        return None
    
    def _calculate_progress(self, session: ConversationSession) -> float:
        """计算对话进度百分比"""
        if session.config.max_rounds == 0:
            return 0.0
        return min(session.round_count / session.config.max_rounds * 100, 100.0)
    
    def _estimate_completion(self, session: ConversationSession) -> Optional[str]:
        """估算完成时间"""
        if session.round_count == 0:
            return None
        
        # 基于当前进度估算剩余时间
        elapsed = utc_now() - session.created_at
        avg_time_per_round = elapsed.total_seconds() / session.round_count
        remaining_rounds = max(0, session.config.max_rounds - session.round_count)
        estimated_seconds = remaining_rounds * avg_time_per_round
        
        if estimated_seconds < 60:
            return f"{int(estimated_seconds)}秒"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds / 60)}分钟"
        else:
            return f"{int(estimated_seconds / 3600)}小时"
    
    async def pause_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """暂停对话"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        result = await session.pause_conversation()
        
        logger.info(
            "对话已暂停",
            conversation_id=conversation_id,
            round_count=session.round_count,
        )
        
        return result
    
    async def resume_conversation(self, conversation_id: str, websocket_callback=None) -> Dict[str, Any]:
        """恢复对话"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        result = await session.resume_conversation(websocket_callback)
        
        logger.info(
            "对话已恢复",
            conversation_id=conversation_id,
            round_count=session.round_count,
        )
        
        return result
    
    async def terminate_conversation(
        self, 
        conversation_id: str, 
        reason: str = "用户终止"
    ) -> Dict[str, Any]:
        """终止对话"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        result = await session.terminate_conversation(reason)
        
        # 从活跃会话中移除
        if conversation_id in self._active_sessions:
            del self._active_sessions[conversation_id]
        self._conversation_ws_sessions.pop(conversation_id, None)
        
        logger.info(
            "对话已终止",
            conversation_id=conversation_id,
            reason=reason,
            message_count=result.get("message_count", 0),
        )
        
        return result
    
    async def get_conversation_messages(
        self, 
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """获取对话消息"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            session = await self.group_chat_manager.get_session(conversation_id)
        
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        messages = session.messages[offset:]
        if limit:
            messages = messages[:limit]
        
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "total_count": len(session.messages),
            "returned_count": len(messages),
            "offset": offset,
        }

    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """生成对话摘要"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            session = await self.group_chat_manager.get_session(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")

        messages = session.messages
        key_points = self._collect_snippets(messages, 3)
        decisions = self._collect_snippets(
            messages,
            3,
            keywords=["决定", "结论", "建议", "需要", "should", "decide"],
        )
        action_items = self._collect_snippets(
            messages,
            3,
            keywords=["行动", "下一步", "计划", "执行", "follow", "todo", "task"],
        )
        participants_summary = self._build_participants_summary(messages)

        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "round_count": session.round_count,
            "key_points": key_points,
            "decisions_made": decisions,
            "action_items": action_items,
            "participants_summary": participants_summary,
        }

    async def analyze_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """分析对话内容"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            session = await self.group_chat_manager.get_session(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")

        messages = session.messages
        topic_distribution = self._build_topic_distribution(messages)
        sentiment_analysis = self._build_sentiment_analysis(messages)
        sender_counts = self._build_sender_counts(messages)
        recommendations = self._build_recommendations(session, messages, sender_counts)

        return {
            "conversation_id": conversation_id,
            "sentiment_analysis": sentiment_analysis,
            "topic_distribution": topic_distribution,
            "interaction_patterns": [
                {
                    "type": "message_count",
                    "total_messages": len(messages),
                    "by_sender": sender_counts,
                }
            ],
            "recommendations": recommendations,
        }

    async def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """导出对话内容"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            session = await self.group_chat_manager.get_session(conversation_id)
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")

        participants = [
            {
                "name": agent.config.name,
                "role": agent.config.role,
                "capabilities": agent.config.capabilities,
            }
            for agent in session.participants
        ]

        return {
            "conversation_id": conversation_id,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "round_count": session.round_count,
            "participants": participants,
            "config": {
                "max_rounds": session.config.max_rounds,
                "timeout_seconds": session.config.timeout_seconds,
                "auto_reply": session.config.auto_reply,
            },
            "messages": session.messages,
        }
    
    async def list_active_conversations(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """列出所有活跃对话"""
        conversations = []

        for session_id, session in self._active_sessions.items():
            status = session.get_status()
            conversations.append({
                "conversation_id": session_id,
                "status": status["status"],
                "created_at": status["created_at"],
                "message_count": status["message_count"],
                "round_count": status["round_count"],
                "participants": status["participants"],
            })

        conversations.sort(
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )
        if offset > 0:
            conversations = conversations[offset:]
        if limit is not None:
            conversations = conversations[:limit]

        return conversations

    def _normalize_content(self, content: str) -> str:
        return " ".join(content.strip().split())

    def _to_snippet(self, content: str, limit: int = 160) -> str:
        normalized = self._normalize_content(content)
        if len(normalized) > limit:
            return normalized[:limit].rstrip() + "..."
        return normalized

    def _collect_snippets(
        self,
        messages: List[Dict[str, Any]],
        limit: int,
        keywords: Optional[List[str]] = None,
    ) -> List[str]:
        snippets: List[str] = []
        for message in reversed(messages):
            content = message.get("content") or ""
            normalized = self._normalize_content(content)
            if not normalized:
                continue
            if keywords:
                normalized_lower = normalized.lower()
                if not any(keyword.lower() in normalized_lower for keyword in keywords):
                    continue
            snippet = self._to_snippet(normalized)
            if snippet in snippets:
                continue
            snippets.append(snippet)
            if len(snippets) >= limit:
                break
        return list(reversed(snippets))

    def _build_participants_summary(
        self,
        messages: List[Dict[str, Any]],
        limit: int = 8,
    ) -> Dict[str, str]:
        summary: Dict[str, str] = {}
        for message in reversed(messages):
            sender = message.get("sender")
            if not sender or sender in summary:
                continue
            content = message.get("content") or ""
            normalized = self._normalize_content(content)
            if not normalized:
                continue
            summary[sender] = self._to_snippet(normalized)
            if len(summary) >= limit:
                break
        return summary

    def _extract_tokens(self, text: str) -> List[str]:
        tokens: List[str] = []
        for token in re.findall(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]{2,}", text):
            normalized = token.lower()
            if len(normalized) < 2:
                continue
            tokens.append(normalized)
        return tokens

    def _build_topic_distribution(
        self,
        messages: List[Dict[str, Any]],
        limit: int = 5,
    ) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for message in messages:
            content = message.get("content") or ""
            for token in self._extract_tokens(content):
                counts[token] = counts.get(token, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return {}
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        return {token: count / total for token, count in top}

    def _build_sentiment_analysis(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        positive_words = [
            "成功",
            "稳定",
            "清晰",
            "优秀",
            "提升",
            "满意",
            "有效",
            "完成",
            "good",
            "great",
            "success",
            "stable",
        ]
        negative_words = [
            "失败",
            "错误",
            "问题",
            "异常",
            "风险",
            "延迟",
            "崩溃",
            "不足",
            "bug",
            "error",
            "fail",
            "issue",
        ]
        positive = 0
        negative = 0
        for message in messages:
            content = message.get("content") or ""
            content_lower = content.lower()
            for word in positive_words:
                if word in content_lower:
                    positive += 1
            for word in negative_words:
                if word in content_lower:
                    negative += 1
        total = positive + negative
        if total == 0:
            return {"neutral": 1.0}
        return {
            "positive": positive / total,
            "negative": negative / total,
        }

    def _build_sender_counts(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for message in messages:
            sender = message.get("sender") or "未知"
            counts[sender] = counts.get(sender, 0) + 1
        return counts

    def _build_recommendations(
        self,
        session: ConversationSession,
        messages: List[Dict[str, Any]],
        sender_counts: Dict[str, int],
    ) -> List[str]:
        recommendations: List[str] = []
        if not messages:
            return ["暂无消息，建议先启动对话。"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            recommendations.append("建议补充用户目标或约束，便于智能体对齐。")
        if sender_counts:
            max_sender = max(sender_counts.items(), key=lambda item: item[1])
            total_messages = sum(sender_counts.values())
            if total_messages > 0 and max_sender[1] / total_messages > 0.6:
                recommendations.append("建议引导其他智能体参与，避免单一观点主导。")
        if session.config.max_rounds and session.round_count >= session.config.max_rounds:
            recommendations.append("对话已接近最大轮次，可考虑收敛结论。")
        if not recommendations:
            recommendations.append("对话节奏正常，可继续推进任务拆解。")
        return recommendations
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        total_sessions = len(self._active_sessions)
        total_messages = sum(
            len(session.messages) for session in self._active_sessions.values()
        )
        
        # 按角色统计智能体使用情况
        role_usage = {}
        for session in self._active_sessions.values():
            for agent in session.participants:
                role = agent.config.role
                if role not in role_usage:
                    role_usage[role] = {
                        "session_count": 0,
                        "message_count": 0,
                    }
                role_usage[role]["session_count"] += 1
                
                # 统计该智能体的消息数量
                agent_messages = [
                    msg for msg in session.messages 
                    if msg["sender"] == agent.config.name
                ]
                role_usage[role]["message_count"] += len(agent_messages)
        
        return {
            "total_active_sessions": total_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": (
                total_messages / total_sessions if total_sessions > 0 else 0
            ),
            "role_usage_statistics": role_usage,
            "session_status_distribution": self._get_status_distribution(),
        }
    
    def _get_status_distribution(self) -> Dict[str, int]:
        """获取会话状态分布"""
        distribution = {}
        for session in self._active_sessions.values():
            status = session.status.value
            distribution[status] = distribution.get(status, 0) + 1
        return distribution
    
    async def cleanup_inactive_sessions(self) -> Dict[str, Any]:
        """清理非活跃会话"""
        # 清理已完成或终止的会话
        inactive_sessions = []
        for session_id, session in list(self._active_sessions.items()):
            if session.status.value in ["completed", "terminated", "error"]:
                inactive_sessions.append(session_id)
                del self._active_sessions[session_id]
                self._conversation_ws_sessions.pop(session_id, None)
        
        # 清理管理器中的会话
        cleaned_count = await self.group_chat_manager.cleanup_completed_sessions()
        
        logger.info(
            "清理非活跃会话",
            local_cleaned=len(inactive_sessions),
            manager_cleaned=cleaned_count,
            remaining_active=len(self._active_sessions),
        )
        
        return {
            "cleaned_sessions": len(inactive_sessions) + cleaned_count,
            "remaining_active_sessions": len(self._active_sessions),
            "cleanup_timestamp": utc_now().isoformat(),
        }
