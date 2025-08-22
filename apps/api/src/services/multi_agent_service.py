"""
多智能体协作服务
"""
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
import json

from src.ai.autogen import (
    GroupChatManager,
    ConversationSession,
    create_default_agents,
    create_agent_from_config,
    AgentConfig,
    AgentRole,
    ConversationConfig,
)
from src.ai.autogen.agents import BaseAutoGenAgent
import structlog

logger = structlog.get_logger(__name__)


class MultiAgentService:
    """多智能体协作服务"""
    
    def __init__(self):
        self.group_chat_manager = GroupChatManager()
        self._active_sessions: Dict[str, ConversationSession] = {}
    
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
                "messages": [session.messages[0]] if session.messages else [],  # 只返回初始消息
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
                # 为每个角色创建智能体实例
                agent = create_agent_from_config(
                    AgentConfig(
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
        prompts = {
            AgentRole.CODE_EXPERT: """你是一位专业的软件开发专家，擅长代码编写、重构和优化。
请用简洁、技术准确的语言回答，专注于实际的代码实现方案。""",
            
            AgentRole.ARCHITECT: """你是一位资深的软件架构师，专长于系统架构设计和技术选型。
请从架构角度分析问题，提供高层次的设计方案和技术建议。""",
            
            AgentRole.DOC_EXPERT: """你是一位专业的技术文档专家，擅长技术文档撰写和优化。
请提供清晰、结构化的文档建议，确保信息准确且易于理解。""",
        }
        return prompts.get(role, "你是一位AI助手，请协助完成任务。")
    
    def _get_role_capabilities(self, role: AgentRole) -> List[str]:
        """获取角色能力列表"""
        capabilities = {
            AgentRole.CODE_EXPERT: ["代码生成", "代码审查", "性能优化", "问题调试"],
            AgentRole.ARCHITECT: ["架构设计", "技术选型", "性能架构", "系统建模"],
            AgentRole.DOC_EXPERT: ["文档撰写", "API文档", "知识管理", "信息架构"],
        }
        return capabilities.get(role, ["通用AI助手"])
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话状态"""
        session = self._active_sessions.get(conversation_id)
        if not session:
            # 尝试从管理器获取
            session = await self.group_chat_manager.get_session(conversation_id)
        
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        
        status = session.get_status()
        
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
        elapsed = datetime.now(timezone.utc) - session.created_at
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
    
    async def list_active_conversations(self) -> List[Dict[str, Any]]:
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
        
        return conversations
    
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
            "cleanup_timestamp": datetime.now(timezone.utc).isoformat(),
        }