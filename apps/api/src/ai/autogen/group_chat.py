"""
AutoGen 0.7.x 多智能体对话实现
使用新版本的autogen-agentchat和autogen-core API
"""
import asyncio
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from autogen_agentchat.teams import BaseGroupChat, RoundRobinGroupChat
from autogen_core import CancellationToken
import structlog

from .agents import BaseAutoGenAgent, create_default_agents
from .config import ConversationConfig
from src.core.config import get_settings
from src.core.constants import TimeoutConstants

logger = structlog.get_logger(__name__)


class ConversationStatus(str, Enum):
    """对话状态枚举"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"


class ConversationSession:
    """多智能体对话会话 - 使用新版AutoGen API"""
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        participants: Optional[List[BaseAutoGenAgent]] = None,
        config: Optional[ConversationConfig] = None,
        initial_topic: Optional[str] = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())
        self.participants = participants or create_default_agents()
        self.config = config or ConversationConfig()
        self.initial_topic = initial_topic
        
        self.status = ConversationStatus.CREATED
        self.created_at = utc_now()
        self.updated_at = self.created_at
        self.messages: List[Dict[str, Any]] = []
        self.current_speaker_index = 0
        self.round_count = 0
        
        # AutoGen 0.7.x 对象
        self._group_chat: Optional[RoundRobinGroupChat] = None
        self._cancellation_token = CancellationToken()
        
        # 任务管理
        self._active_tasks: List[asyncio.Task] = []  # 保存运行中的任务
        
        self._initialize_group_chat()
    
    def _initialize_group_chat(self):
        """初始化GroupChat"""
        try:
            # 提取AutoGen智能体实例
            autogen_agents = [agent.agent for agent in self.participants]
            
            # 创建RoundRobinGroupChat (新版API)
            self._group_chat = RoundRobinGroupChat(autogen_agents)
            
            logger.info(
                "GroupChat初始化成功",
                session_id=self.session_id,
                participant_count=len(self.participants),
                max_rounds=self.config.max_rounds,
            )
            
        except Exception as e:
            logger.error(
                "GroupChat初始化失败",
                session_id=self.session_id,
                error=str(e)
            )
            self.status = ConversationStatus.ERROR
            raise
    
    async def start_conversation(self, initial_message: str, websocket_callback=None) -> Dict[str, Any]:
        """启动多智能体对话"""
        if self.status != ConversationStatus.CREATED:
            raise ValueError(f"会话状态不正确: {self.status}")
        
        try:
            self.status = ConversationStatus.ACTIVE
            self.updated_at = utc_now()
            
            # 记录初始消息
            self._add_message("user", "系统", initial_message)
            
            logger.info(
                "开始多智能体对话",
                session_id=self.session_id,
                initial_message=initial_message[:100] + "..." if len(initial_message) > 100 else initial_message,
            )
            
            # 启动GroupChat对话（异步执行，不等待完成）
            async def run_with_error_handling():
                try:
                    await self._run_group_chat_with_websocket(initial_message, websocket_callback)
                except Exception as e:
                    logger.error(f"群组对话执行异常: {e}", session_id=self.session_id)
                    if websocket_callback:
                        await websocket_callback({
                            "type": "conversation_error",
                            "session_id": self.session_id,
                            "error": f"对话执行异常: {str(e)}"
                        })
            
            # 创建任务并保存引用以便后续取消
            task = asyncio.create_task(run_with_error_handling())
            self._active_tasks.append(task)
            
            return {
                "session_id": self.session_id,
                "status": self.status.value,
                "message_count": len(self.messages),
                "current_round": self.round_count,
            }
            
        except Exception as e:
            logger.error(
                "启动多智能体对话失败",
                session_id=self.session_id,
                error=str(e)
            )
            self.status = ConversationStatus.ERROR
            raise
    
    async def _run_group_chat(self, initial_message: str):
        """运行群组对话 - 使用新的API"""
        try:
            if not self._group_chat:
                raise ValueError("GroupChat未初始化")
            
            # 限制轮数和时间避免无限循环
            max_rounds = min(self.config.max_rounds, 3)  # 进一步限制轮数
            timeout_seconds = min(self.config.timeout_seconds, TimeoutConstants.CONVERSATION_TIMEOUT_SECONDS)  # 限制总超时时间
            
            current_message = initial_message
            for round_num in range(max_rounds):
                self.round_count += 1
                
                # 轮流让每个智能体响应
                for i, participant in enumerate(self.participants):
                    try:
                        # 添加单个智能体响应的超时控制
                        response = await asyncio.wait_for(
                            participant.generate_response(
                                current_message, 
                                self._cancellation_token
                            ),
                            timeout=float(TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS)  # 单个智能体响应超时
                        )
                        
                        if response and response.strip():
                            # 记录响应
                            self._add_message("assistant", participant.config.name, response)
                            current_message = response  # 下一个智能体使用这个响应
                        else:
                            # 如果响应为空，使用默认响应
                            default_response = f"我是{participant.config.name}，我已收到您的消息并正在处理中。"
                            self._add_message("assistant", participant.config.name, default_response)
                            current_message = default_response
                        
                        # 检查是否应该终止
                        if self._should_terminate():
                            self.status = ConversationStatus.COMPLETED
                            return
                            
                    except asyncio.TimeoutError:
                        logger.warning(
                            "智能体响应超时",
                            agent_name=participant.config.name,
                            round=round_num,
                            timeout_seconds=TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS
                        )
                        # 添加超时响应
                        timeout_response = f"我是{participant.config.name}，正在思考中，请稍等..."
                        self._add_message("assistant", participant.config.name, timeout_response)
                        continue
                        
                    except Exception as e:
                        logger.error(
                            "智能体响应失败",
                            agent_name=participant.config.name,
                            round=round_num,
                            error=str(e)
                        )
                        # 添加错误响应
                        error_response = f"我是{participant.config.name}，遇到了一些技术问题，正在恢复中。"
                        self._add_message("assistant", participant.config.name, error_response)
                        continue
            
            # 如果达到最大轮数
            self.status = ConversationStatus.COMPLETED
            self.updated_at = utc_now()
            
        except Exception as e:
            logger.error(
                "群组对话执行失败",
                session_id=self.session_id,
                error=str(e)
            )
            self.status = ConversationStatus.ERROR
            raise
    
    async def _run_group_chat_with_websocket(self, initial_message: str, websocket_callback=None):
        """运行群组对话并通过WebSocket实时推送消息"""
        try:
            if not self._group_chat:
                raise ValueError("GroupChat未初始化")
            
            # 限制轮数和时间避免无限循环
            max_rounds = min(self.config.max_rounds, 3)
            
            current_message = initial_message
            for round_num in range(max_rounds):
                self.round_count += 1
                
                # 轮流让每个智能体响应
                for i, participant in enumerate(self.participants):
                    try:
                        # 通知前端当前发言者
                        if websocket_callback:
                            await websocket_callback({
                                "type": "speaker_change",
                                "session_id": self.session_id,
                                "current_speaker": participant.config.name,
                                "round": self.round_count
                            })
                        
                        logger.info(f"开始生成智能体流式响应 - 智能体: {participant.config.name}, 轮次: {self.round_count}, 消息: {current_message[:100]}...")
                        
                        # 创建流式响应的WebSocket回调
                        full_response = ""
                        message_id = str(uuid.uuid4())
                        
                        async def stream_callback(chunk_data):
                            nonlocal full_response
                            
                            if chunk_data["type"] == "token":
                                # 每个token实时推送
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "streaming_token",
                                        "session_id": self.session_id,
                                        "message_id": message_id,
                                        "agent_name": participant.config.name,
                                        "token": chunk_data["content"],
                                        "full_content": chunk_data["full_content"],
                                        "round": self.round_count,
                                        "is_complete": False
                                    })
                            elif chunk_data["type"] == "complete":
                                # 响应完成
                                full_response = chunk_data["full_content"]
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "streaming_complete",
                                        "session_id": self.session_id,
                                        "message_id": message_id,
                                        "agent_name": participant.config.name,
                                        "full_content": full_response,
                                        "round": self.round_count,
                                        "is_complete": True
                                    })
                            elif chunk_data["type"] == "error":
                                # 错误处理
                                full_response = chunk_data["full_content"]
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "streaming_error", 
                                        "session_id": self.session_id,
                                        "message_id": message_id,
                                        "agent_name": participant.config.name,
                                        "error": chunk_data["error"],
                                        "full_content": full_response,
                                        "round": self.round_count,
                                        "is_complete": True
                                    })
                        
                        try:
                            # 使用流式响应生成
                            response = await asyncio.wait_for(
                                participant.generate_streaming_response(
                                    current_message,
                                    stream_callback=stream_callback,
                                    cancellation_token=self._cancellation_token
                                ),
                                timeout=float(TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS)
                            )
                            
                            logger.info(f"智能体流式响应生成完成 - 智能体: {participant.config.name}, 响应长度: {len(response) if response else 0}")
                            
                            if response and response.strip():
                                # 记录完整响应到历史记录
                                message = self._add_message_to_history("assistant", participant.config.name, response)
                                current_message = response
                                
                                # 发送消息已保存通知
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "new_message",
                                        "session_id": self.session_id,
                                        "data": {
                                            "id": message["id"],
                                            "role": message["role"],
                                            "sender": message["sender"],
                                            "content": message["content"],
                                            "timestamp": message["timestamp"],
                                            "round": message["round"]
                                        }
                                    })
                            else:
                                # 如果响应为空，使用默认响应
                                default_response = f"我是{participant.config.name}，我已收到您的消息并正在处理中。"
                                message = self._add_message_and_notify("assistant", participant.config.name, default_response, websocket_callback)
                                current_message = default_response
                        
                        except asyncio.TimeoutError:
                            # 流式响应超时处理
                            if full_response and full_response.strip():
                                # 如果已经有部分响应，使用部分响应
                                message = self._add_message_to_history("assistant", participant.config.name, full_response)
                                current_message = full_response
                                
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "new_message",
                                        "session_id": self.session_id,
                                        "data": {
                                            "id": message["id"],
                                            "role": message["role"],
                                            "sender": message["sender"],  
                                            "content": message["content"],
                                            "timestamp": message["timestamp"],
                                            "round": message["round"]
                                        }
                                    })
                            else:
                                # 没有响应，抛出超时异常让外层处理
                                raise
                        
                        # 检查是否应该终止
                        if self._should_terminate():
                            self.status = ConversationStatus.COMPLETED
                            if websocket_callback:
                                await websocket_callback({
                                    "type": "conversation_completed",
                                    "session_id": self.session_id,
                                    "total_messages": len(self.messages),
                                    "total_rounds": self.round_count
                                })
                            return
                            
                    except asyncio.TimeoutError:
                        logger.warning(
                            "智能体响应超时",
                            agent_name=participant.config.name,
                            round=round_num,
                            timeout_seconds=TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS
                        )
                        # 添加超时响应并通过WebSocket推送
                        timeout_response = f"我是{participant.config.name}，正在思考中，请稍等..."
                        logger.info(f"准备推送超时消息: {timeout_response}, websocket_callback存在: {websocket_callback is not None}")
                        self._add_message_and_notify("assistant", participant.config.name, timeout_response, websocket_callback)
                        continue
                        
                    except Exception as e:
                        logger.error(
                            "智能体响应失败",
                            agent_name=participant.config.name,
                            round=round_num,
                            error=str(e)
                        )
                        # 添加错误响应
                        error_response = f"我是{participant.config.name}，遇到了一些技术问题，正在恢复中。"
                        self._add_message_and_notify("assistant", participant.config.name, error_response, websocket_callback)
                        continue
            
            # 如果达到最大轮数
            self.status = ConversationStatus.COMPLETED
            self.updated_at = utc_now()
            
            if websocket_callback:
                await websocket_callback({
                    "type": "conversation_completed",
                    "session_id": self.session_id,
                    "total_messages": len(self.messages),
                    "total_rounds": self.round_count
                })
            
        except Exception as e:
            logger.error(
                "群组对话执行失败",
                session_id=self.session_id,
                error=str(e)
            )
            self.status = ConversationStatus.ERROR
            if websocket_callback:
                await websocket_callback({
                    "type": "conversation_error",
                    "session_id": self.session_id,
                    "error": str(e)
                })
    
    async def _continue_group_chat_from_resume(self, current_message: str, websocket_callback=None):
        """从暂停点继续群组对话"""
        try:
            if not self._group_chat:
                raise ValueError("GroupChat未初始化")
            
            # 计算剩余轮数
            remaining_rounds = max(0, self.config.max_rounds - self.round_count)
            if remaining_rounds == 0:
                # 如果已经达到最大轮数，直接完成
                self.status = ConversationStatus.COMPLETED
                if websocket_callback:
                    await websocket_callback({
                        "type": "conversation_completed",
                        "session_id": self.session_id,
                        "total_messages": len(self.messages),
                        "total_rounds": self.round_count
                    })
                return
            
            # 通知恢复开始
            if websocket_callback:
                await websocket_callback({
                    "type": "conversation_resumed",
                    "session_id": self.session_id,
                    "remaining_rounds": remaining_rounds,
                    "current_round": self.round_count
                })
            
            # 继续对话流程
            for round_num in range(remaining_rounds):
                self.round_count += 1
                
                # 检查是否被取消
                if self._cancellation_token.is_cancelled():
                    break
                
                # 轮流让每个智能体响应
                for i, participant in enumerate(self.participants):
                    try:
                        # 检查是否被取消
                        if self._cancellation_token.is_cancelled():
                            break
                        
                        # 通知前端当前发言者
                        if websocket_callback:
                            await websocket_callback({
                                "type": "speaker_change",
                                "session_id": self.session_id,
                                "current_speaker": participant.config.name,
                                "round": self.round_count
                            })
                        
                        logger.info(f"恢复后继续生成智能体流式响应 - 智能体: {participant.config.name}, 轮次: {self.round_count}")
                        
                        # 创建流式响应的WebSocket回调（与原始流程相同）
                        full_response = ""
                        message_id = str(uuid.uuid4())
                        
                        async def stream_callback(chunk_data):
                            nonlocal full_response
                            
                            if chunk_data["type"] == "token":
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "streaming_token",
                                        "session_id": self.session_id,
                                        "message_id": message_id,
                                        "agent_name": participant.config.name,
                                        "token": chunk_data["content"],
                                        "full_content": chunk_data["full_content"],
                                        "round": self.round_count,
                                        "is_complete": False
                                    })
                            elif chunk_data["type"] == "complete":
                                full_response = chunk_data["full_content"]
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "streaming_complete",
                                        "session_id": self.session_id,
                                        "message_id": message_id,
                                        "agent_name": participant.config.name,
                                        "full_content": full_response,
                                        "round": self.round_count,
                                        "is_complete": True
                                    })
                        
                        try:
                            # 使用流式响应生成
                            response = await asyncio.wait_for(
                                participant.generate_streaming_response(
                                    current_message,
                                    stream_callback=stream_callback,
                                    cancellation_token=self._cancellation_token
                                ),
                                timeout=float(TimeoutConstants.AGENT_RESPONSE_TIMEOUT_SECONDS)
                            )
                            
                            if response and response.strip():
                                # 记录完整响应到历史记录
                                message = self._add_message_to_history("assistant", participant.config.name, response)
                                current_message = response
                                
                                # 发送消息已保存通知
                                if websocket_callback:
                                    await websocket_callback({
                                        "type": "new_message",
                                        "session_id": self.session_id,
                                        "data": {
                                            "id": message["id"],
                                            "role": message["role"],
                                            "sender": message["sender"],
                                            "content": message["content"],
                                            "timestamp": message["timestamp"],
                                            "round": message["round"]
                                        }
                                    })
                            
                        except asyncio.TimeoutError:
                            # 超时处理
                            if full_response and full_response.strip():
                                message = self._add_message_to_history("assistant", participant.config.name, full_response)
                                current_message = full_response
                        
                        # 检查是否应该终止
                        if self._should_terminate():
                            self.status = ConversationStatus.COMPLETED
                            if websocket_callback:
                                await websocket_callback({
                                    "type": "conversation_completed",
                                    "session_id": self.session_id,
                                    "total_messages": len(self.messages),
                                    "total_rounds": self.round_count
                                })
                            return
                            
                    except Exception as e:
                        logger.error(
                            "恢复后智能体响应失败",
                            agent_name=participant.config.name,
                            round=round_num,
                            error=str(e)
                        )
                        continue
            
            # 如果达到最大轮数
            self.status = ConversationStatus.COMPLETED
            self.updated_at = utc_now()
            
            if websocket_callback:
                await websocket_callback({
                    "type": "conversation_completed",
                    "session_id": self.session_id,
                    "total_messages": len(self.messages),
                    "total_rounds": self.round_count
                })
            
        except Exception as e:
            logger.error(
                "恢复对话执行失败",
                session_id=self.session_id,
                error=str(e)
            )
            self.status = ConversationStatus.ERROR
            if websocket_callback:
                await websocket_callback({
                    "type": "conversation_error",
                    "session_id": self.session_id,
                    "error": str(e)
                })
    
    def _add_message_and_notify(self, role: str, sender: str, content: str, websocket_callback=None):
        """添加消息并通过WebSocket通知"""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "sender": sender,
            "content": content,
            "timestamp": utc_now().isoformat(),
            "round": self.round_count,
        }
        self.messages.append(message)
        
        # 通过WebSocket实时推送消息
        if websocket_callback:
            asyncio.create_task(websocket_callback({
                "type": "new_message",
                "session_id": self.session_id,
                "message": message
            }))
        
        return message
    
    def _add_message(self, role: str, sender: str, content: str):
        """添加消息到历史记录"""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "sender": sender,
            "content": content,
            "timestamp": utc_now().isoformat(),
            "round": self.round_count,
        }
        self.messages.append(message)
        return message
    
    def _add_message_to_history(self, role: str, sender: str, content: str):
        """添加消息到历史记录（用于流式响应完成后的最终消息）"""
        return self._add_message(role, sender, content)
    
    def _should_terminate(self) -> bool:
        """检查是否应该终止对话"""
        # 检查轮数限制（只在轮次完成后检查）
        if self.round_count > self.config.max_rounds:
            return True
        
        # 检查最后几条消息是否包含终止关键词
        if len(self.messages) >= 2:
            recent_messages = self.messages[-2:]
            for msg in recent_messages:
                content = msg["content"].lower()
                if any(keyword in content for keyword in [
                    "会话结束", "讨论完成", "结论达成", 
                    "terminate", "session_end", "完成讨论"
                ]):
                    return True
        
        return False
    
    async def pause_conversation(self) -> Dict[str, Any]:
        """暂停对话"""
        if self.status != ConversationStatus.ACTIVE:
            raise ValueError(f"无法暂停非活跃状态的对话: {self.status}")
        
        self.status = ConversationStatus.PAUSED
        self.updated_at = utc_now()
        
        # 取消当前的操作
        self._cancellation_token.cancel()
        
        # 关键修复：取消所有活跃任务
        cancelled_count = 0
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        
        logger.info(
            "对话已暂停",
            session_id=self.session_id,
            cancelled_tasks=cancelled_count,
            total_tasks=len(self._active_tasks)
        )
        
        return {"session_id": self.session_id, "status": self.status.value}
    
    async def resume_conversation(self, websocket_callback=None) -> Dict[str, Any]:
        """恢复对话"""
        if self.status != ConversationStatus.PAUSED:
            raise ValueError(f"无法恢复非暂停状态的对话: {self.status}")
        
        self.status = ConversationStatus.ACTIVE
        self.updated_at = utc_now()
        
        # 创建新的取消令牌
        self._cancellation_token = CancellationToken()
        
        # 清理之前被取消的任务
        self._active_tasks = [task for task in self._active_tasks if not task.done()]
        
        # 重新启动对话流程
        if self.messages:
            # 获取最后一条消息作为继续点
            last_message = self.messages[-1]
            current_message = last_message.get("content", "继续之前的讨论")
            
            # 启动GroupChat对话（异步执行，不等待完成）
            async def resume_with_error_handling():
                try:
                    await self._continue_group_chat_from_resume(current_message, websocket_callback)
                except Exception as e:
                    logger.error(f"恢复对话执行异常: {e}", session_id=self.session_id)
                    if websocket_callback:
                        await websocket_callback({
                            "type": "conversation_error",
                            "session_id": self.session_id,
                            "error": f"恢复对话异常: {str(e)}"
                        })
            
            # 创建任务并保存引用以便后续取消
            task = asyncio.create_task(resume_with_error_handling())
            self._active_tasks.append(task)
        
        logger.info("对话已恢复并重新启动", session_id=self.session_id)
        
        return {"session_id": self.session_id, "status": self.status.value}
    
    async def terminate_conversation(self, reason: str = "用户终止") -> Dict[str, Any]:
        """终止对话"""
        self.status = ConversationStatus.TERMINATED
        self.updated_at = utc_now()
        
        # 取消所有操作
        self._cancellation_token.cancel()
        
        # 生成对话总结
        summary = await self._generate_conversation_summary(reason)
        
        logger.info(
            "对话已终止",
            session_id=self.session_id,
            reason=reason,
            message_count=len(self.messages),
        )
        
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "summary": summary,
            "message_count": len(self.messages),
            "round_count": self.round_count,
        }
    
    async def _generate_conversation_summary(self, termination_reason: str) -> Dict[str, Any]:
        """生成对话总结"""
        return {
            "termination_reason": termination_reason,
            "duration_minutes": (
                self.updated_at - self.created_at
            ).total_seconds() / 60,
            "total_messages": len(self.messages),
            "total_rounds": self.round_count,
            "participants": [
                {
                    "name": agent.config.name,
                    "role": agent.config.role,
                    "message_count": len([
                        msg for msg in self.messages 
                        if msg["sender"] == agent.config.name
                    ])
                }
                for agent in self.participants
            ],
            "key_topics": self._extract_key_topics(),
        }
    
    def _extract_key_topics(self) -> List[str]:
        """提取关键话题（简单实现）"""
        if self.initial_topic:
            return [self.initial_topic]
        return ["技术讨论", "多智能体协作"]
    
    def get_status(self) -> Dict[str, Any]:
        """获取会话状态"""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "round_count": self.round_count,
            "participants": [
                {
                    "name": agent.config.name,
                    "role": agent.config.role,
                    "status": agent.get_status()["status"],
                }
                for agent in self.participants
            ],
            "config": {
                "max_rounds": self.config.max_rounds,
                "timeout_seconds": self.config.timeout_seconds,
                "auto_reply": self.config.auto_reply,
            }
        }


class GroupChatManager:
    """多智能体群组会话管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}
        self.settings = get_settings()
    
    async def create_session(
        self,
        participants: Optional[List[BaseAutoGenAgent]] = None,
        config: Optional[ConversationConfig] = None,
        initial_topic: Optional[str] = None,
    ) -> ConversationSession:
        """创建新的对话会话"""
        session = ConversationSession(
            participants=participants,
            config=config,
            initial_topic=initial_topic,
        )
        
        self.sessions[session.session_id] = session
        
        logger.info(
            "创建新的多智能体会话",
            session_id=session.session_id,
            participant_count=len(session.participants),
        )
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取指定会话"""
        return self.sessions.get(session_id)
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        return [session.get_status() for session in self.sessions.values()]
    
    async def cleanup_completed_sessions(self) -> int:
        """清理已完成的会话"""
        completed_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.status in [ConversationStatus.COMPLETED, ConversationStatus.TERMINATED]
        ]
        
        for session_id in completed_sessions:
            del self.sessions[session_id]
        
        logger.info(
            "清理已完成会话",
            cleaned_count=len(completed_sessions),
            remaining_count=len(self.sessions),
        )
        
        return len(completed_sessions)