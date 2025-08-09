"""
ReAct (Reasoning + Acting) 智能体实现
支持思考、行动、观察的循环推理模式
"""

import json
import uuid
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
import structlog

from ...ai.openai_client import get_openai_client
from ...ai.mcp.client import get_mcp_client_manager

logger = structlog.get_logger(__name__)


class ReActStepType(Enum):
    """ReAct步骤类型"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"
    STREAMING_TOKEN = "streaming_token"


@dataclass
class ReActStep:
    """ReAct步骤数据结构"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: ReActStepType = ReActStepType.THOUGHT
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReActSession:
    """ReAct会话状态"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[ReActStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10
    current_step: int = 0


class ReActAgent:
    """ReAct智能体类"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_steps: int = 10,
        temperature: float = 0.1,
    ):
        """初始化ReAct智能体"""
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.openai_client = None
        self.mcp_client = None
        self.sessions: Dict[str, ReActSession] = {}

    async def initialize(self):
        """初始化客户端连接"""
        self.openai_client = await get_openai_client()
        self.mcp_client = await get_mcp_client_manager()
        logger.info("ReAct智能体初始化完成")

    def _get_system_prompt(self) -> str:
        """获取ReAct系统提示词"""
        return """你是一个智能助手，使用ReAct(Reasoning + Acting)模式来解决问题。

你的工作流程是：
1. **Thought**: 分析问题，思考需要做什么
2. **Action**: 选择并调用合适的工具
3. **Observation**: 观察工具的执行结果
4. 重复1-3直到能够提供最终答案
5. **Final Answer**: 提供完整的答案

可用的响应格式：

对于思考：
```
Thought: [你的分析和思考过程]
```

对于行动：
```
Action: [tool_name]
Action Input: [JSON格式的工具参数]
```

对于最终答案：
```
Final Answer: [你的完整答案]
```

重要规则：
- 每次只能进行一个Action
- Action Input必须是有效的JSON格式
- 仔细分析Observation的结果
- 如果工具调用失败，分析原因并尝试其他方法
- 最终答案要完整、准确、有用
- 当提供代码时，必须使用标准markdown代码块格式：```语言\\n代码\\n```
- 确保代码有正确的缩进和换行格式
- 每个代码块只能有一对```标记，绝对不要在回答最后单独添加```"""

    def _parse_response(self, response: str) -> ReActStep:
        """解析AI响应，识别步骤类型和内容"""
        response = response.strip()
        
        # 检查是否包含最终答案（可能在多行响应中）
        if "Final Answer:" in response:
            # 找到Final Answer的位置并提取内容
            final_answer_index = response.find("Final Answer:")
            content = response[final_answer_index + len("Final Answer:"):].strip()
            # 修复markdown格式
            content = self._fix_markdown_format(content)
            return ReActStep(
                step_type=ReActStepType.FINAL_ANSWER,
                content=content
            )
        
        # 检查是否是简单的最终答案格式
        if response.startswith("Final Answer:"):
            content = response[len("Final Answer:"):].strip()
            # 修复markdown格式
            content = self._fix_markdown_format(content)
            return ReActStep(
                step_type=ReActStepType.FINAL_ANSWER,
                content=content
            )
        
        # 检查是否是思考
        if response.startswith("Thought:"):
            content = response[len("Thought:"):].strip()
            return ReActStep(
                step_type=ReActStepType.THOUGHT,
                content=content
            )
        
        # 检查是否是行动
        if "Action:" in response and "Action Input:" in response:
            lines = response.split('\n')
            action_line = ""
            action_input_line = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Action:"):
                    action_line = line[len("Action:"):].strip()
                elif line.startswith("Action Input:"):
                    action_input_line = line[len("Action Input:"):].strip()
            
            if action_line and action_input_line:
                try:
                    # 解析工具参数
                    tool_args = json.loads(action_input_line)
                    return ReActStep(
                        step_type=ReActStepType.ACTION,
                        content=f"调用工具: {action_line}",
                        tool_name=action_line,
                        tool_args=tool_args
                    )
                except json.JSONDecodeError as e:
                    logger.error("工具参数JSON解析失败", error=str(e), input=action_input_line)
                    return ReActStep(
                        step_type=ReActStepType.THOUGHT,
                        content=f"工具参数格式错误: {str(e)}"
                    )
        
        # 默认作为思考处理
        return ReActStep(
            step_type=ReActStepType.THOUGHT,
            content=response
        )


    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """执行工具调用"""
        try:
            logger.info(
                "执行工具调用",
                tool_name=tool_name,
                args=tool_args
            )
            
            # 需要确定工具属于哪个服务器类型
            server_type = await self._determine_server_type(tool_name)
            if not server_type:
                return {"error": f"未找到工具 {tool_name} 对应的服务器类型"}
            
            # 通过MCP客户端调用工具
            result = await self.mcp_client.call_tool(server_type, tool_name, tool_args)
            
            logger.info(
                "工具调用成功",
                tool_name=tool_name,
                server_type=server_type,
                result_type=type(result).__name__
            )
            
            return result
            
        except Exception as e:
            error_msg = f"工具调用失败: {str(e)}"
            logger.error(
                "工具调用错误",
                tool_name=tool_name,
                error=str(e)
            )
            return {"error": error_msg}

    async def _determine_server_type(self, tool_name: str) -> Optional[str]:
        """确定工具对应的服务器类型"""
        try:
            # 获取所有可用工具
            all_tools = await self.mcp_client.get_available_tools()
            
            # 搜索工具属于哪个服务器类型
            for server_type, tools in all_tools.items():
                for tool in tools:
                    if tool.get("name") == tool_name:
                        return server_type
            
            logger.warning(f"未找到工具 {tool_name} 对应的服务器类型")
            return None
            
        except Exception as e:
            logger.error(f"确定工具服务器类型时出错: {str(e)}")
            return None

    async def _build_conversation_history(self, session: ReActSession) -> List[Dict[str, str]]:
        """构建对话历史"""
        messages = []
        
        # 添加系统提示
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })
        
        # 添加可用工具信息
        if self.mcp_client:
            try:
                all_tools = await self.mcp_client.get_available_tools()
                if all_tools:
                    tools_desc = "可用工具:\n"
                    for server_type, tools in all_tools.items():
                        tools_desc += f"\n{server_type}服务器工具:\n"
                        for tool in tools:
                            tools_desc += f"- {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}\n"
                    
                    messages.append({
                        "role": "system",
                        "content": tools_desc
                    })
            except Exception as e:
                logger.warning("获取工具信息失败", error=str(e))
        
        # 添加步骤历史
        for step in session.steps:
            if step.step_type == ReActStepType.THOUGHT:
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {step.content}"
                })
            elif step.step_type == ReActStepType.ACTION:
                messages.append({
                    "role": "assistant",
                    "content": f"Action: {step.tool_name}\nAction Input: {json.dumps(step.tool_args, ensure_ascii=False)}"
                })
            elif step.step_type == ReActStepType.OBSERVATION:
                messages.append({
                    "role": "user",
                    "content": f"Observation: {step.content}"
                })
            elif step.step_type == ReActStepType.FINAL_ANSWER:
                messages.append({
                    "role": "assistant",
                    "content": f"Final Answer: {step.content}"
                })
        
        return messages

    async def _get_next_step(self, session: ReActSession) -> ReActStep:
        """获取下一个推理步骤"""
        messages = await self._build_conversation_history(session)
        
        # 如果没有用户输入，需要添加
        if not any(msg["role"] == "user" for msg in messages):
            # 从上下文获取用户问题
            user_question = session.context.get("user_question", "请开始思考")
            messages.append({
                "role": "user", 
                "content": user_question
            })
        
        try:
            response = await self.openai_client.create_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            content = response.get("content", "").strip()
            if not content:
                logger.warning("AI响应为空")
                return ReActStep(
                    step_type=ReActStepType.THOUGHT,
                    content="需要重新思考问题"
                )
            
            return self._parse_response(content)
            
        except Exception as e:
            logger.error("获取AI响应失败", error=str(e))
            return ReActStep(
                step_type=ReActStepType.THOUGHT,
                content=f"处理错误: {str(e)}"
            )

    async def _get_next_step_streaming(self, session: ReActSession):
        """获取下一个推理步骤的流式版本"""
        messages = await self._build_conversation_history(session)
        
        # 如果没有用户输入，需要添加
        if not any(msg["role"] == "user" for msg in messages):
            # 从上下文获取用户问题
            user_question = session.context.get("user_question", "请开始思考")
            messages.append({
                "role": "user", 
                "content": user_question
            })
        
        try:
            full_content = ""
            should_stream_content = False  # 控制是否输出内容的标志
            
            async for chunk in self.openai_client.create_streaming_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            ):
                if chunk.get("content"):
                    token_content = chunk["content"]
                    full_content += token_content
                    
                    # 检查是否到达Final Answer部分
                    if "Final Answer:" in full_content and not should_stream_content:
                        should_stream_content = True
                        # 提取Final Answer后的内容
                        final_answer_index = full_content.find("Final Answer:")
                        clean_content = full_content[final_answer_index + len("Final Answer:"):].strip()
                        # 修复代码块格式
                        clean_content = self._fix_markdown_format(clean_content)
                        
                        # 输出已有的clean内容
                        if clean_content:
                            yield ReActStep(
                                step_type=ReActStepType.STREAMING_TOKEN,
                                content=clean_content,
                                metadata={
                                    "is_streaming": True,
                                    "full_content_so_far": clean_content
                                }
                            )
                        continue
                    
                    # 如果已经在Final Answer阶段，直接输出token
                    if should_stream_content:
                        # 保持原始token内容，不进行任何清理
                        if token_content:
                            yield ReActStep(
                                step_type=ReActStepType.STREAMING_TOKEN,
                                content=token_content,
                                metadata={
                                    "is_streaming": True,
                                    "full_content_so_far": full_content
                                }
                            )
            
            # 推理步骤完成，解析最终内容
            if full_content.strip():
                final_step = self._parse_response(full_content)
                yield final_step
            else:
                logger.warning("AI流式响应为空")
                yield ReActStep(
                    step_type=ReActStepType.THOUGHT,
                    content="需要重新思考问题"
                )
            
        except Exception as e:
            logger.error("获取AI流式响应失败", error=str(e))
            yield ReActStep(
                step_type=ReActStepType.THOUGHT,
                content=f"处理错误: {str(e)}"
            )

    def _fix_markdown_format(self, content: str) -> str:
        """修复markdown格式，确保代码块正确显示"""
        if not content:
            return content
        
        # 移除末尾单独的```标记
        content = content.strip()
        if content.endswith('\n```'):
            content = content[:-4].rstrip()
        elif content.endswith('```'):
            content = content[:-3].rstrip()
        
        return content

    async def run_session(
        self, 
        user_input: str, 
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReActSession:
        """运行ReAct会话"""
        # 获取或创建会话
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = ReActSession(max_steps=self.max_steps)
            if session_id:
                session.session_id = session_id
            self.sessions[session.session_id] = session
        
        # 更新上下文
        if context:
            session.context.update(context)
        session.context["user_question"] = user_input
        
        logger.info(
            "开始ReAct会话",
            session_id=session.session_id,
            user_input=user_input,
            max_steps=session.max_steps
        )
        
        # 确保初始化完成
        if not self.openai_client:
            await self.initialize()
        
        # 推理循环
        while session.current_step < session.max_steps:
            session.current_step += 1
            
            logger.info(
                "执行推理步骤",
                session_id=session.session_id,
                step=session.current_step
            )
            
            # 获取下一步
            step = await self._get_next_step(session)
            session.steps.append(step)
            
            logger.info(
                "推理步骤完成",
                session_id=session.session_id,
                step_type=step.step_type.value,
                content=step.content[:100] + "..." if len(step.content) > 100 else step.content
            )
            
            # 如果是最终答案，结束循环
            if step.step_type == ReActStepType.FINAL_ANSWER:
                break
            
            # 如果是行动，执行工具调用
            if step.step_type == ReActStepType.ACTION and step.tool_name:
                result = await self._execute_tool(step.tool_name, step.tool_args or {})
                
                # 添加观察步骤
                observation = ReActStep(
                    step_type=ReActStepType.OBSERVATION,
                    content=str(result)
                )
                session.steps.append(observation)
                
                # 更新原步骤的结果
                step.tool_result = result
        
        # 如果达到最大步骤数还没有最终答案，添加一个
        if session.current_step >= session.max_steps:
            last_step = session.steps[-1] if session.steps else None
            if not last_step or last_step.step_type != ReActStepType.FINAL_ANSWER:
                final_step = ReActStep(
                    step_type=ReActStepType.FINAL_ANSWER,
                    content="已达到最大推理步骤数，基于当前信息提供答案。"
                )
                session.steps.append(final_step)
        
        logger.info(
            "ReAct会话完成",
            session_id=session.session_id,
            total_steps=len(session.steps),
            final_answer_provided=any(step.step_type == ReActStepType.FINAL_ANSWER for step in session.steps)
        )
        
        return session

    async def _stream_final_answer(self, final_answer_content: str, session: ReActSession):
        """流式输出最终答案"""
        try:
            # 构建消息历史，包含所有推理步骤
            messages = [{"role": "system", "content": self._get_system_prompt()}]
            
            # 添加对话历史
            context = session.context
            if context.get("user_question"):
                messages.append({
                    "role": "user", 
                    "content": context["user_question"]
                })
            
            # 添加所有推理步骤作为上下文
            reasoning_context = []
            for step in session.steps[:-1]:  # 排除最后一步（当前的final_answer）
                if step.step_type == ReActStepType.THOUGHT:
                    reasoning_context.append(f"思考: {step.content}")
                elif step.step_type == ReActStepType.ACTION:
                    reasoning_context.append(f"行动: {step.tool_name}({step.tool_args})")
                elif step.step_type == ReActStepType.OBSERVATION:
                    reasoning_context.append(f"观察: {step.content}")
            
            if reasoning_context:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(reasoning_context) + f"\n\n基于以上分析，我的最终回答是: {final_answer_content[:50]}..."
                })
            
            # 要求重新生成更详细的最终答案
            messages.append({
                "role": "user",
                "content": "请基于你的分析，提供一个完整、详细的最终答案。直接回答，不要重复推理过程。"
            })
            
            # 使用流式API重新生成答案
            full_content = ""
            async for chunk in self.openai_client.create_streaming_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            ):
                if chunk.get("content"):
                    token_content = chunk["content"]
                    full_content += token_content
                    
                    # 创建流式token步骤
                    token_step = ReActStep(
                        step_type=ReActStepType.STREAMING_TOKEN,
                        content=token_content,
                        metadata={"is_streaming": True, "full_content_so_far": full_content}
                    )
                    yield token_step
            
            # 发送完成标记
            completion_step = ReActStep(
                step_type=ReActStepType.FINAL_ANSWER,
                content=full_content,
                metadata={"is_complete": True, "streaming_finished": True}
            )
            yield completion_step
            
        except Exception as e:
            logger.error(f"流式答案生成失败: {str(e)}")
            # 如果流式失败，返回原始答案
            fallback_step = ReActStep(
                step_type=ReActStepType.FINAL_ANSWER,
                content=final_answer_content,
                metadata={"fallback": True}
            )
            yield fallback_step

    async def run_streaming_session(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ReActStep, None]:
        """流式运行ReAct会话"""
        # 获取或创建会话
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = ReActSession(max_steps=self.max_steps)
            if session_id:
                session.session_id = session_id
            self.sessions[session.session_id] = session
        
        # 更新上下文
        if context:
            session.context.update(context)
        session.context["user_question"] = user_input
        
        # 确保初始化完成
        if not self.openai_client:
            await self.initialize()
        
        # 推理循环
        while session.current_step < session.max_steps:
            session.current_step += 1
            
            # 使用流式获取下一步，实时输出每个推理token
            final_step = None
            async for step_chunk in self._get_next_step_streaming(session):
                if step_chunk.step_type == ReActStepType.STREAMING_TOKEN:
                    # 实时流式输出推理过程
                    yield step_chunk
                else:
                    # 完整的推理步骤
                    final_step = step_chunk
                    session.steps.append(step_chunk)
                    
                    # 如果不是流式token，输出完整步骤
                    if step_chunk.step_type != ReActStepType.STREAMING_TOKEN:
                        yield step_chunk
                    break
            
            if not final_step:
                logger.warning("未获得有效的推理步骤")
                break
            
            # 如果是最终答案，结束循环
            if final_step.step_type == ReActStepType.FINAL_ANSWER:
                break
            
            # 如果是行动，执行工具调用
            if final_step.step_type == ReActStepType.ACTION and final_step.tool_name:
                result = await self._execute_tool(final_step.tool_name, final_step.tool_args or {})
                
                # 添加观察步骤
                observation = ReActStep(
                    step_type=ReActStepType.OBSERVATION,
                    content=str(result)
                )
                session.steps.append(observation)
                
                # 流式返回观察
                yield observation
                
                # 更新原步骤的结果
                final_step.tool_result = result

    def get_session(self, session_id: str) -> Optional[ReActSession]:
        """获取会话状态"""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话摘要"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        # 统计步骤类型
        step_counts = {}
        for step in session.steps:
            step_type = step.step_type.value
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        # 获取最终答案
        final_answer = None
        for step in reversed(session.steps):
            if step.step_type == ReActStepType.FINAL_ANSWER:
                final_answer = step.content
                break
        
        return {
            "session_id": session.session_id,
            "total_steps": len(session.steps),
            "current_step": session.current_step,
            "max_steps": session.max_steps,
            "step_counts": step_counts,
            "final_answer": final_answer,
            "completed": final_answer is not None,
            "context": session.context
        }