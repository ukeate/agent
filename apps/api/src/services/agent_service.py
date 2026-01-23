"""
智能体服务层
整合ReAct智能体和对话管理功能
"""

import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from src.ai.agents.react_agent import ReActAgent, ReActStep, ReActStepType
from src.services.conversation_service import get_conversation_service
from src.core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

class AgentService:
    """智能体服务类"""

    def __init__(self):
        self.settings = get_settings()
        self.agents: Dict[str, ReActAgent] = {}
        self.conversation_service = None

    async def initialize(self):
        """初始化服务"""
        self.conversation_service = await get_conversation_service()
        logger.info("智能体服务初始化完成")

    async def _ensure_conversation_access(
        self,
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        if not self.conversation_service:
            await self.initialize()
        session = await self.conversation_service.get_conversation_for_user(
            conversation_id,
            user_id
        )
        if not session:
            raise ValueError(f"对话会话不存在: {conversation_id}")
        return session

    async def create_agent_session(
        self,
        user_id: str,
        agent_type: str = "react",
        agent_config: Optional[Dict[str, Any]] = None,
        conversation_title: Optional[str] = None
    ) -> Dict[str, str]:
        """创建智能体会话"""
        try:
            # 确保服务已初始化
            if not self.conversation_service:
                await self.initialize()

            # 创建对话会话
            conversation_id = await self.conversation_service.create_conversation(
                user_id=user_id,
                title=conversation_title,
                agent_type=agent_type,
                metadata=agent_config or {}
            )

            # 创建智能体实例
            agent = await self._create_agent(agent_type, agent_config or {})
            await agent.initialize()

            # 存储智能体实例
            agent_id = f"{agent_type}_{conversation_id}"
            self.agents[agent_id] = agent

            logger.info(
                "创建智能体会话",
                user_id=user_id,
                conversation_id=conversation_id,
                agent_id=agent_id,
                agent_type=agent_type
            )

            return {
                "conversation_id": conversation_id,
                "agent_id": agent_id,
                "agent_type": agent_type
            }

        except Exception as e:
            logger.error(
                "创建智能体会话失败",
                error=str(e),
                user_id=user_id,
                agent_type=agent_type
            )
            raise

    async def update_conversation_title(
        self,
        conversation_id: str,
        user_id: str,
        title: str
    ) -> Dict[str, Any]:
        """更新对话标题"""
        if not self.conversation_service:
            await self.initialize()
        await self._ensure_conversation_access(conversation_id, user_id)
        return await self.conversation_service.update_conversation_title(
            conversation_id,
            title
        )

    def chat_with_agent(
        self,
        conversation_id: str,
        user_input: str,
        user_id: str,
        stream: bool = False
    ):
        """与智能体对话"""
        if stream:
            # 流式处理
            return self._chat_with_agent_stream(conversation_id, user_input, user_id)
        else:
            # 非流式处理 - 返回协程
            return self._chat_with_agent_regular(conversation_id, user_input, user_id)

    async def _chat_with_agent_regular(
        self,
        conversation_id: str,
        user_input: str,
        user_id: str
    ):
        """非流式对话处理"""
        try:
            await self._ensure_conversation_access(conversation_id, user_id)
            # 获取智能体实例
            agent = await self._get_agent_for_conversation(conversation_id)
            if not agent:
                raise ValueError(f"未找到对话的智能体: {conversation_id}")

            # 添加用户消息到对话历史
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                content=user_input,
                sender_type="user"
            )

            # 获取对话上下文
            context = await self.conversation_service.get_conversation_context(conversation_id)

            # 非流式处理
            return await self._handle_regular_chat(
                agent=agent,
                conversation_id=conversation_id,
                user_input=user_input,
                context=context["session_context"]
            )
        except Exception as e:
            logger.error(
                "智能体对话失败",
                error=str(e),
                conversation_id=conversation_id,
                user_id=user_id
            )
            raise

    async def _chat_with_agent_stream(
        self,
        conversation_id: str,
        user_input: str,
        user_id: str
    ):
        """流式对话处理"""
        try:
            await self._ensure_conversation_access(conversation_id, user_id)
            # 获取智能体实例
            agent = await self._get_agent_for_conversation(conversation_id)
            if not agent:
                raise ValueError(f"未找到对话的智能体: {conversation_id}")

            # 添加用户消息到对话历史
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                content=user_input,
                sender_type="user"
            )

            # 获取对话上下文
            context = await self.conversation_service.get_conversation_context(conversation_id)

            # 流式处理
            async for step_data in self._handle_streaming_chat(
                agent=agent,
                conversation_id=conversation_id,
                user_input=user_input,
                context=context["session_context"]
            ):
                yield step_data
        except Exception as e:
            logger.error(
                "流式智能体对话失败",
                error=str(e),
                conversation_id=conversation_id,
                user_id=user_id
            )
            yield {
                "conversation_id": conversation_id,
                "error": str(e),
                "step_type": "error"
            }

    async def _handle_regular_chat(
        self,
        agent: ReActAgent,
        conversation_id: str,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理非流式对话"""
        try:
            # 运行ReAct会话
            react_session = await agent.run_session(
                user_input=user_input,
                session_id=conversation_id,
                context=context
            )

            # 处理智能体的推理步骤
            response_content = []
            tool_calls = []
            last_action_step: Optional[ReActStep] = None

            for step in react_session.steps:
                if step.step_type == ReActStepType.THOUGHT:
                    # 添加思考过程到消息
                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=f"思考: {step.content}",
                        sender_type="assistant",
                        message_type="thought",
                        metadata={"step_id": step.step_id}
                    )
                    response_content.append(f"思考: {step.content}")

                elif step.step_type == ReActStepType.ACTION:
                    # 记录工具调用
                    tool_call_info = {
                        "tool_name": step.tool_name,
                        "tool_args": step.tool_args,
                        "step_id": step.step_id,
                        "status": "pending",
                        "timestamp": step.timestamp
                    }
                    tool_calls.append(tool_call_info)
                    last_action_step = step

                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=f"行动: {step.content}",
                        sender_type="assistant",
                        message_type="tool_call",
                        tool_calls=[tool_call_info],
                        metadata={"step_id": step.step_id}
                    )

                elif step.step_type == ReActStepType.OBSERVATION:
                    # 记录观察结果
                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=f"观察: {step.content}",
                        sender_type="assistant",
                        message_type="tool_result",
                        metadata={"step_id": step.step_id}
                    )
                    if last_action_step:
                        result = (
                            last_action_step.tool_result
                            if last_action_step.tool_result is not None
                            else step.content
                        )
                        status = (
                            "error"
                            if isinstance(result, dict) and result.get("error")
                            else "success"
                        )
                        await self.conversation_service.update_tool_call_result(
                            conversation_id=conversation_id,
                            step_id=last_action_step.step_id,
                            result=result,
                            status=status
                        )
                        last_action_step = None

                elif step.step_type == ReActStepType.FINAL_ANSWER:
                    # 最终答案
                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=step.content,
                        sender_type="assistant",
                        message_type="text",
                        metadata={"step_id": step.step_id, "final_answer": True}
                    )
                    response_content.append(step.content)

            # 更新对话上下文
            await self.conversation_service.update_conversation_context(
                conversation_id=conversation_id,
                context_updates={
                    "last_react_session": react_session.session_id,
                    "total_steps": len(react_session.steps),
                    "completed": any(step.step_type == ReActStepType.FINAL_ANSWER for step in react_session.steps)
                }
            )

            # 获取最终答案
            final_answer = None
            for step in reversed(react_session.steps):
                if step.step_type == ReActStepType.FINAL_ANSWER:
                    final_answer = step.content
                    break

            return {
                "conversation_id": conversation_id,
                "response": final_answer or "处理完成",
                "steps": len(react_session.steps),
                "tool_calls": tool_calls,
                "completed": final_answer is not None,
                "session_summary": agent.get_session_summary(react_session.session_id)
            }

        except Exception as e:
            logger.error(
                "处理智能体对话失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def _handle_streaming_chat(
        self,
        agent: ReActAgent,
        conversation_id: str,
        user_input: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """处理流式对话"""
        try:
            last_action_step: Optional[ReActStep] = None
            async for step in agent.run_streaming_session(
                user_input=user_input,
                session_id=conversation_id,
                context=context
            ):
                # 实时处理每个推理步骤
                step_data = {
                    "conversation_id": conversation_id,
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "content": step.content,
                    "timestamp": step.timestamp
                }

                if step.step_type == ReActStepType.ACTION:
                    step_data.update({
                        "tool_name": step.tool_name,
                        "tool_args": step.tool_args
                    })

                    # 记录工具调用
                    last_action_step = step
                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=f"行动: {step.content}",
                        sender_type="assistant",
                        message_type="tool_call",
                        tool_calls=[{
                            "tool_name": step.tool_name,
                            "tool_args": step.tool_args,
                            "step_id": step.step_id,
                            "status": "pending",
                            "timestamp": step.timestamp
                        }],
                        metadata={"step_id": step.step_id}
                    )

                elif step.step_type == ReActStepType.OBSERVATION:
                    if last_action_step:
                        result = (
                            last_action_step.tool_result
                            if last_action_step.tool_result is not None
                            else step.content
                        )
                        status = (
                            "error"
                            if isinstance(result, dict) and result.get("error")
                            else "success"
                        )
                        await self.conversation_service.update_tool_call_result(
                            conversation_id=conversation_id,
                            step_id=last_action_step.step_id,
                            result=result,
                            status=status
                        )
                        last_action_step = None

                elif step.step_type == ReActStepType.FINAL_ANSWER:
                    # 记录最终答案
                    await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        content=step.content,
                        sender_type="assistant",
                        message_type="text",
                        metadata={"step_id": step.step_id, "final_answer": True}
                    )

                yield step_data
            summary = agent.get_session_summary(conversation_id)
            if summary:
                if not self.conversation_service:
                    await self.initialize()
                await self.conversation_service.update_conversation_context(
                    conversation_id=conversation_id,
                    context_updates={
                        "last_react_session": summary.get("session_id"),
                        "total_steps": summary.get("total_steps", 0),
                        "completed": summary.get("completed", False),
                    }
                )

        except Exception as e:
            logger.error(
                "流式对话处理失败",
                error=str(e),
                conversation_id=conversation_id
            )
            yield {
                "conversation_id": conversation_id,
                "error": str(e),
                "step_type": "error"
            }

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取对话历史"""
        try:
            if not self.conversation_service:
                await self.initialize()

            await self._ensure_conversation_access(conversation_id, user_id)
            # 获取对话历史
            messages = await self.conversation_service.get_conversation_history(
                conversation_id=conversation_id,
                limit=limit,
                include_system=False
            )

            # 获取对话摘要
            summary = await self.conversation_service.get_conversation_summary(conversation_id)

            return {
                "conversation_id": conversation_id,
                "messages": messages,
                "summary": summary
            }

        except Exception as e:
            logger.error(
                "获取对话历史失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def get_agent_status(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        try:
            await self._ensure_conversation_access(conversation_id, user_id)
            agent = await self._get_agent_for_conversation(conversation_id)
            if not agent:
                return {
                    "conversation_id": conversation_id,
                    "status": "not_found",
                    "session_summary": None,
                    "agent_type": "react",
                }

            # 获取会话摘要
            session = agent.get_session(conversation_id)
            if session:
                summary = agent.get_session_summary(conversation_id)
            else:
                summary = None

            return {
                "conversation_id": conversation_id,
                "status": "active" if session else "inactive",
                "session_summary": summary,
                "agent_type": "react"
            }

        except Exception as e:
            logger.error(
                "获取智能体状态失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def close_agent_session(self, conversation_id: str, user_id: str) -> None:
        """关闭智能体会话"""
        try:
            if not self.conversation_service:
                await self.initialize()

            await self._ensure_conversation_access(conversation_id, user_id)
            # 清理智能体会话
            agent = await self._get_agent_for_conversation(conversation_id)
            if agent:
                agent.clear_session(conversation_id)

            # 关闭对话会话
            await self.conversation_service.close_conversation(conversation_id)

            # 移除智能体实例
            agent_ids_to_remove = [
                agent_id for agent_id in self.agents.keys()
                if conversation_id in agent_id
            ]
            for agent_id in agent_ids_to_remove:
                del self.agents[agent_id]

            logger.info(
                "关闭智能体会话",
                conversation_id=conversation_id
            )

        except Exception as e:
            logger.error(
                "关闭智能体会话失败",
                error=str(e),
                conversation_id=conversation_id
            )
            raise

    async def _create_agent(self, agent_type: str, config: Dict[str, Any]) -> ReActAgent:
        """创建智能体实例"""
        if agent_type == "react":
            return ReActAgent(
                model=config.get("model", "gpt-4o-mini"),
                max_steps=config.get("max_steps", 10),
                temperature=config.get("temperature", 0.1)
            )
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")

    async def _get_agent_for_conversation(self, conversation_id: str) -> Optional[ReActAgent]:
        """获取对话对应的智能体"""
        for agent_id, agent in self.agents.items():
            if conversation_id in agent_id:
                return agent
        return None

    async def list_user_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出用户的对话会话"""
        try:
            # 确保服务已初始化
            if not self.conversation_service:
                await self.initialize()
                
            return await self.conversation_service.list_conversations(
                user_id=user_id,
                limit=limit,
                offset=offset,
                query=query
            )

        except Exception as e:
            logger.error(
                "列出用户对话失败",
                error=str(e),
                user_id=user_id
            )
            raise

# 单例模式的服务实例
_agent_service: Optional[AgentService] = None

async def get_agent_service() -> AgentService:
    """获取智能体服务实例"""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
