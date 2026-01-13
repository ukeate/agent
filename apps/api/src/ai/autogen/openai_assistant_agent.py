"""
AutoGen 0.7.1 OpenAIAssistantAgent实现
这是AutoGen 0.7.0回归的功能，支持OpenAI Assistants API
"""
import asyncio
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import structlog

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat_assistants import OpenAIAssistantAgent
    from autogen_core.models import ChatCompletionClient
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core import CancellationToken
except ImportError as e:
    OpenAIAssistantAgent = None
    AssistantAgent = None
    CancellationToken = None

from src.ai.openai_client import get_openai_client
from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class AssistantToolType(str, Enum):
    """助手工具类型"""
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"
    FUNCTION = "function"


class AssistantToolConfig(BaseModel):
    """助手工具配置"""
    type: AssistantToolType = Field(..., description="工具类型")
    function_name: Optional[str] = Field(None, description="函数名称（仅function类型）")
    function_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="函数参数")

    class Config:
        use_enum_values = True


class OpenAIAssistantConfig(BaseModel):
    """OpenAI Assistant配置"""
    name: str = Field(..., description="助手名称")
    instructions: str = Field(default="You are a helpful assistant.", description="系统指令")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    tools: List[AssistantToolConfig] = Field(default_factory=list, description="工具列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="top_p参数")
    max_prompt_tokens: Optional[int] = Field(None, ge=1, description="最大prompt tokens")
    max_completion_tokens: Optional[int] = Field(None, ge=1, description="最大completion tokens")

    class Config:
        use_enum_values = True


class OpenAIAssistantAgentWrapper:
    """
    OpenAIAssistantAgent包装器
    AutoGen 0.7.0重新引入了对OpenAI Assistants API的支持
    """

    def __init__(self, config: OpenAIAssistantConfig):
        if OpenAIAssistantAgent is None:
            raise ImportError(
                "autogen-agentchat-assistants未安装。请安装: pip install autogen-agentchat-assistants"
            )

        self.config = config
        self._agent: Optional[OpenAIAssistantAgent] = None
        self._assistant_id: Optional[str] = None
        self._settings = get_settings()
        self._initialize_agent()

    def _initialize_agent(self):
        """初始化OpenAI Assistant Agent"""
        try:
            # 创建模型客户端
            model_client = OpenAIChatCompletionClient(
                model=self.config.model,
                api_key=self._settings.OPENAI_API_KEY,
            )

            # 转换工具配置
            tools_config = []
            for tool in self.config.tools:
                if tool.type == AssistantToolType.CODE_INTERPRETER:
                    tools_config.append({"type": "code_interpreter"})
                elif tool.type == AssistantToolType.FILE_SEARCH:
                    tools_config.append({"type": "file_search"})
                elif tool.type == AssistantToolType.FUNCTION:
                    tools_config.append({
                        "type": "function",
                        "function": {
                            "name": tool.function_name,
                            "parameters": tool.function_params
                        }
                    })

            # 创建OpenAIAssistantAgent
            self._agent = OpenAIAssistantAgent(
                name=self.config.name,
                description=self.config.instructions,
                model_client=model_client,
                instructions=self.config.instructions,
                tools=tools_config if tools_config else None,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_prompt_tokens=self.config.max_prompt_tokens,
                max_completion_tokens=self.config.max_completion_tokens,
            )

            logger.info(
                "OpenAIAssistantAgent初始化成功",
                assistant_name=self.config.name,
                model=self.config.model,
                tools_count=len(tools_config),
            )

        except Exception as e:
            logger.error(
                "OpenAIAssistantAgent初始化失败",
                assistant_name=self.config.name,
                error=str(e),
            )
            raise

    @property
    def agent(self) -> OpenAIAssistantAgent:
        """获取Agent实例"""
        if self._agent is None:
            raise ValueError("Agent未初始化")
        return self._agent

    async def create_assistant(self) -> str:
        """
        创建OpenAI Assistant并返回assistant_id
        AutoGen 0.7.0新特性
        """
        try:
            # 这里需要调用OpenAI API创建assistant
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)

            # 转换工具格式
            tools = []
            for tool in self.config.tools:
                if tool.type == AssistantToolType.CODE_INTERPRETER:
                    tools.append({"type": "code_interpreter"})
                elif tool.type == AssistantToolType.FILE_SEARCH:
                    tools.append({"type": "file_search"})
                elif tool.type == AssistantToolType.FUNCTION:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.function_name,
                            "description": tool.function_params.get("description", ""),
                            "parameters": tool.function_params.get("parameters", {})
                        }
                    })

            assistant = await client.beta.assistants.create(
                name=self.config.name,
                instructions=self.config.instructions,
                model=self.config.model,
                tools=tools if tools else None,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )

            self._assistant_id = assistant.id

            logger.info(
                "OpenAI Assistant创建成功",
                assistant_id=self._assistant_id,
                name=self.config.name,
            )

            return self._assistant_id

        except Exception as e:
            logger.error("创建OpenAI Assistant失败", error=str(e))
            raise

    async def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """
        与助手对话
        AutoGen 0.7.0新特性：支持线程管理
        """
        try:
            from openai import AsyncOpenAI
            from autogen_agentchat.messages import TextMessage

            client = AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)

            # 创建或获取线程
            if not thread_id:
                thread = await client.beta.threads.create()
                thread_id = thread.id
                logger.info("创建新线程", thread_id=thread_id)

            # 添加消息到线程
            await client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message
            )

            # 创建运行
            run = await client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self._assistant_id or await self.create_assistant()
            )

            # 等待运行完成
            import time
            while True:
                run_status = await client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )

                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Run失败: {run_status.status}")

                await asyncio.sleep(0.5)

            # 获取响应
            messages = await client.beta.threads.messages.list(thread_id=thread_id)
            latest_message = messages.data[0]

            response_text = ""
            for content in latest_message.content:
                if content.type == "text":
                    response_text = content.text.value

            return {
                "response": response_text,
                "thread_id": thread_id,
                "run_id": run.id,
                "assistant_id": self._assistant_id,
            }

        except Exception as e:
            logger.error("Assistant对话失败", error=str(e))
            raise

    async def generate_response(
        self,
        message: str,
        thread_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """生成响应（简化接口）"""
        result = await self.chat(message, thread_id, cancellation_token)
        return result["response"]


def create_openai_assistant_agent(
    name: str,
    instructions: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    tools: Optional[List[AssistantToolConfig]] = None,
    temperature: float = 0.7,
    **kwargs
) -> OpenAIAssistantAgentWrapper:
    """
    创建OpenAI Assistant Agent的工厂函数
    AutoGen 0.7.0新特性
    """
    config = OpenAIAssistantConfig(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools or [],
        temperature=temperature,
        **kwargs
    )

    return OpenAIAssistantAgentWrapper(config)


def create_code_interpreter_assistant(
    name: str = "code_interpreter",
    instructions: str = "You are a helpful assistant with code execution capabilities.",
    model: str = "gpt-4o",
    **kwargs
) -> OpenAIAssistantAgentWrapper:
    """创建带代码解释器的助手"""
    tools = [AssistantToolConfig(type=AssistantToolType.CODE_INTERPRETER)]
    return create_openai_assistant_agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        **kwargs
    )


def create_file_search_assistant(
    name: str = "file_search",
    instructions: str = "You are a helpful assistant with file search capabilities.",
    model: str = "gpt-4o",
    **kwargs
) -> OpenAIAssistantAgentWrapper:
    """创建带文件搜索的助手"""
    tools = [AssistantToolConfig(type=AssistantToolType.FILE_SEARCH)]
    return create_openai_assistant_agent(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools,
        **kwargs
    )
