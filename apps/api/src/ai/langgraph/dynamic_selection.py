"""
LangGraph 0.6.5 动态模型和工具选择
支持运行时动态选择不同的LLM模型和工具集
"""
from typing import Any, Dict, List, Optional, Callable, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    # 如果langchain_openai未安装，使用备用方案
    ChatOpenAI = None
    ChatAnthropic = None
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState as LangGraphMessagesState

from .state import MessagesState
from .context import AgentContext
from src.core.config import get_settings
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


class ModelProvider(str, Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """模型配置"""
    provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="模型提供商")
    model_name: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=2000, ge=1, description="最大token数")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    base_url: Optional[str] = Field(default=None, description="API基础URL")
    max_retries: int = Field(default=3, ge=0, le=10, description="最大重试次数")
    timeout: float = Field(default=60.0, ge=1.0, description="超时时间(秒)")

    class Config:
        use_enum_values = True


class ToolBindingStrategy(str, Enum):
    """工具绑定策略"""
    ALL = "all"  # 绑定所有工具
    SELECTED = "selected"  # 绑定选定的工具
    DYNAMIC = "dynamic"  # 根据上下文动态绑定
    NONE = "none"  # 不绑定工具


class DynamicToolConfig(BaseModel):
    """动态工具配置"""
    strategy: ToolBindingStrategy = Field(default=ToolBindingStrategy.ALL, description="工具绑定策略")
    tool_names: List[str] = Field(default_factory=list, description="选定的工具名称列表")
    max_tools: int = Field(default=20, ge=0, le=128, description="最大工具数量")
    exclude_tools: List[str] = Field(default_factory=list, description="排除的工具名称列表")

    class Config:
        use_enum_values = True


class DynamicModelSelector:
    """动态模型选择器 - LangGraph 0.6.5新特性"""

    def __init__(self):
        self._model_cache: Dict[str, BaseChatModel] = {}
        self._default_config = ModelConfig()

    def get_model(self, config: Optional[ModelConfig] = None) -> BaseChatModel:
        """获取模型实例"""
        model_config = config or self._default_config
        cache_key = f"{model_config.provider}_{model_config.model_name}_{model_config.temperature}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = self._create_model(model_config)
        self._model_cache[cache_key] = model
        return model

    def _create_model(self, config: ModelConfig) -> BaseChatModel:
        """根据配置创建模型实例"""
        if config.provider == ModelProvider.OPENAI:
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key or settings.OPENAI_API_KEY,
                base_url=config.base_url,
                max_retries=config.max_retries,
                timeout=config.timeout,
            )
        elif config.provider == ModelProvider.ANTHROPIC:
            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key or settings.ANTHROPIC_API_KEY if hasattr(settings, 'ANTHROPIC_API_KEY') else None,
                max_retries=config.max_retries,
                timeout=config.timeout,
            )
        else:
            raise ValueError(f"不支持的模型提供商: {config.provider}")

    def update_model_config(self, state: MessagesState, config: ModelConfig) -> MessagesState:
        """更新状态中的模型配置"""
        if "model_config" not in state:
            state["model_config"] = {}
        state["model_config"] = config.model_dump()
        return state


class DynamicToolSelector:
    """动态工具选择器 - LangGraph 0.6.5新特性"""

    def __init__(self, available_tools: Dict[str, BaseTool]):
        self._available_tools = available_tools
        self._tool_configs: Dict[str, DynamicToolConfig] = {}

    def select_tools(
        self,
        config: Optional[DynamicToolConfig] = None,
        context: Optional[AgentContext] = None
    ) -> List[BaseTool]:
        """根据配置选择工具"""
        tool_config = config or DynamicToolConfig()

        if tool_config.strategy == ToolBindingStrategy.NONE:
            return []

        if tool_config.strategy == ToolBindingStrategy.ALL:
            return self._get_all_tools(tool_config)

        if tool_config.strategy == ToolBindingStrategy.SELECTED:
            return self._get_selected_tools(tool_config)

        if tool_config.strategy == ToolBindingStrategy.DYNAMIC:
            return self._get_dynamic_tools(tool_config, context)

        return []

    def _get_all_tools(self, config: DynamicToolConfig) -> List[BaseTool]:
        """获取所有工具（排除指定工具）"""
        tools = []
        for name, tool in self._available_tools.items():
            if name not in config.exclude_tools:
                tools.append(tool)
                if len(tools) >= config.max_tools:
                    break
        return tools

    def _get_selected_tools(self, config: DynamicToolConfig) -> List[BaseTool]:
        """获取选定的工具"""
        tools = []
        for name in config.tool_names:
            if name in self._available_tools and name not in config.exclude_tools:
                tools.append(self._available_tools[name])
                if len(tools) >= config.max_tools:
                    break
        return tools

    def _get_dynamic_tools(self, config: DynamicToolConfig, context: Optional[AgentContext]) -> List[BaseTool]:
        """根据上下文动态选择工具"""
        if not context:
            return self._get_all_tools(config)

        tools = []
        # 根据用户偏好、任务类型等上下文信息选择工具
        user_preferences = context.user_preferences
        metadata = context.metadata

        # 示例：根据任务类型选择工具
        task_type = metadata.get("task_type", "general")
        if task_type == "search":
            search_tools = [k for k in self._available_tools.keys() if "search" in k.lower()]
            for name in search_tools:
                if name not in config.exclude_tools:
                    tools.append(self._available_tools[name])
        elif task_type == "calculation":
            calc_tools = [k for k in self._available_tools.keys() if "calc" in k.lower() or "math" in k.lower()]
            for name in calc_tools:
                if name not in config.exclude_tools:
                    tools.append(self._available_tools[name])
        else:
            tools = self._get_all_tools(config)

        return tools[:config.max_tools]


def create_react_agent_with_dynamic_selection(
    state: MessagesState,
    model_config: Optional[ModelConfig] = None,
    tool_config: Optional[DynamicToolConfig] = None,
    tools: Optional[Dict[str, BaseTool]] = None,
    context: Optional[AgentContext] = None,
) -> Dict[str, Any]:
    """
    创建支持动态模型和工具选择的ReAct Agent
    这是LangGraph 0.6.5的新特性实现
    """
    try:
        # 动态选择模型
        model_selector = DynamicModelSelector()
        model = model_selector.get_model(model_config)

        # 动态选择工具
        selected_tools = []
        if tools:
            tool_selector = DynamicToolSelector(tools)
            selected_tools = tool_selector.select_tools(tool_config, context)

        # 绑定工具到模型
        if selected_tools:
            model = model.bind_tools(selected_tools)

        # 记录选择信息
        selection_info = {
            "model": {
                "provider": model_config.provider if model_config else "openai",
                "model_name": model_config.model_name if model_config else "gpt-4o-mini",
                "temperature": model_config.temperature if model_config else 0.7,
            },
            "tools": {
                "strategy": tool_config.strategy if tool_config else "all",
                "count": len(selected_tools),
                "tool_names": [tool.name for tool in selected_tools],
            }
        }

        if "selection_log" not in state["context"]:
            state["context"]["selection_log"] = []
        state["context"]["selection_log"].append({
            "timestamp": "2025-01-08T22:57:00Z",
            "selection": selection_info
        })

        state["model"] = model
        state["tools"] = selected_tools

        logger.info("动态模型和工具选择完成", **selection_info)

        return state

    except Exception as e:
        logger.error("动态模型和工具选择失败", error=str(e))
        raise


def model_selector_node(
    state: MessagesState,
    context: Optional[AgentContext] = None,
) -> MessagesState:
    """
    模型选择节点 - 根据状态动态选择模型
    LangGraph 0.6.5新特性
    """
    try:
        # 从状态中提取模型配置或使用默认配置
        model_config_data = state.get("model_config", {})
        model_config = ModelConfig(**model_config_data)

        selector = DynamicModelSelector()
        model = selector.get_model(model_config)

        state["model"] = model
        state["metadata"]["selected_model"] = model_config.model_name

        logger.info(
            "模型选择完成",
            model=model_config.model_name,
            provider=model_config.provider,
        )

        return state

    except Exception as e:
        logger.error("模型选择失败", error=str(e))
        state["metadata"]["error"] = f"模型选择失败: {str(e)}"
        return state


def tool_selector_node(
    state: MessagesState,
    context: Optional[AgentContext] = None,
    tools: Optional[Dict[str, BaseTool]] = None,
) -> MessagesState:
    """
    工具选择节点 - 根据状态和上下文动态选择工具
    LangGraph 0.6.5新特性
    """
    try:
        if not tools:
            state["tools"] = []
            return state

        # 从状态中提取工具配置或使用默认配置
        tool_config_data = state.get("tool_config", {})
        tool_config = DynamicToolConfig(**tool_config_data)

        selector = DynamicToolSelector(tools)
        selected_tools = selector.select_tools(tool_config, context)

        state["tools"] = selected_tools
        state["metadata"]["selected_tools"] = [tool.name for tool in selected_tools]

        logger.info(
            "工具选择完成",
            strategy=tool_config.strategy,
            count=len(selected_tools),
        )

        return state

    except Exception as e:
        logger.error("工具选择失败", error=str(e))
        state["metadata"]["error"] = f"工具选择失败: {str(e)}"
        state["tools"] = []
        return state


def create_dynamic_workflow(
    tools: Dict[str, BaseTool],
    default_model_config: Optional[ModelConfig] = None,
    default_tool_config: Optional[DynamicToolConfig] = None,
) -> StateGraph:
    """
    创建支持动态模型和工具选择的工作流
    LangGraph 0.6.5新特性
    """
    workflow = StateGraph(MessagesState)

    # 添加模型选择节点
    workflow.add_node("model_selector", lambda state: model_selector_node(state, None))

    # 添加工具选择节点
    workflow.add_node(
        "tool_selector",
        lambda state: tool_selector_node(state, None, tools)
    )

    # 添加Agent节点
    async def agent_node(state: MessagesState) -> MessagesState:
        model = state.get("model")
        tools = state.get("tools", [])

        if tools:
            model = model.bind_tools(tools)

        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    workflow.add_node("agent", agent_node)

    # 设置边
    workflow.add_edge(START, "model_selector")
    workflow.add_edge("model_selector", "tool_selector")
    workflow.add_edge("tool_selector", "agent")
    workflow.add_edge("agent", END)

    return workflow
