"""
LangGraph节点定义
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import asyncio
from .state import GraphState, GraphExecutionState

from src.core.logging import get_logger
@dataclass
class NodeConfig:
    """节点配置"""
    name: str
    description: str = ""
    timeout: int = 300
    retry_count: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseNode(ABC):
    """基础节点抽象类"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.name}")
    
    @abstractmethod
    async def execute(self, state: GraphState) -> GraphState:
        """执行节点逻辑"""
        raise NotImplementedError
    
    async def run_with_error_handling(self, state: GraphState) -> GraphState:
        """带错误处理的执行"""
        try:
            self.logger.info(f"开始执行节点: {self.config.name}")
            
            # 执行超时控制
            result = await asyncio.wait_for(
                self.execute(state),
                timeout=self.config.timeout
            )
            
            self.logger.info(f"节点执行成功: {self.config.name}")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"节点执行超时: {self.config.name}"
            self.logger.error(error_msg)
            state["error"] = error_msg
            state["is_complete"] = True
            return state
            
        except Exception as e:
            error_msg = f"节点执行失败: {self.config.name}, 错误: {str(e)}"
            self.logger.error(error_msg)
            state["error"] = error_msg
            state["is_complete"] = True
            return state

class AgentNode(BaseNode):
    """智能体节点"""
    
    def __init__(self, config: NodeConfig, agent_handler: Callable):
        super().__init__(config)
        self.agent_handler = agent_handler
    
    async def execute(self, state: GraphState) -> GraphState:
        """执行智能体逻辑"""
        # 更新当前智能体
        state["current_agent"] = self.config.name
        state["step_count"] += 1
        
        # 获取最新消息作为输入
        messages = state.get("messages", [])
        if not messages:
            state["error"] = "没有可处理的消息"
            state["is_complete"] = True
            return state
        
        latest_message = messages[-1]["content"] if messages else ""
        
        try:
            # 调用智能体处理器
            if asyncio.iscoroutinefunction(self.agent_handler):
                response = await self.agent_handler(latest_message, state["context"])
            else:
                response = self.agent_handler(latest_message, state["context"])
            
            # 添加智能体响应到消息历史
            state["messages"].append({
                "id": f"msg-{len(messages)}",
                "role": "assistant",
                "content": response,
                "timestamp": None,  # 将在状态管理器中设置
                "metadata": {
                    "agent": self.config.name,
                    "step": state["step_count"]
                }
            })
            
            # 更新上下文
            state["context"][f"{self.config.name}_last_response"] = response
            
        except Exception as e:
            error_msg = f"智能体处理失败: {str(e)}"
            state["error"] = error_msg
            state["is_complete"] = True
        
        return state

class ToolNode(BaseNode):
    """工具调用节点"""
    
    def __init__(self, config: NodeConfig, tool_handler: Callable):
        super().__init__(config)
        self.tool_handler = tool_handler
    
    async def execute(self, state: GraphState) -> GraphState:
        """执行工具调用"""
        state["step_count"] += 1
        
        # 获取工具参数
        tool_params = self.config.metadata.get("params", {})
        
        # 从上下文中获取动态参数
        for key, value in state["context"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # 简单的变量替换
                var_name = value[2:-1]
                if var_name in state["context"]:
                    tool_params[key] = state["context"][var_name]
        
        try:
            # 执行工具调用
            if asyncio.iscoroutinefunction(self.tool_handler):
                result = await self.tool_handler(tool_params)
            else:
                result = self.tool_handler(tool_params)
            
            # 将结果添加到上下文
            state["context"][f"{self.config.name}_result"] = result
            
            # 添加工具执行消息
            state["messages"].append({
                "id": f"tool-{state['step_count']}",
                "role": "tool",
                "content": f"工具 {self.config.name} 执行完成",
                "timestamp": None,
                "metadata": {
                    "tool": self.config.name,
                    "result": result,
                    "params": tool_params
                }
            })
            
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            state["error"] = error_msg
            state["is_complete"] = True
        
        return state

class ConditionNode(BaseNode):
    """条件判断节点"""
    
    def __init__(self, config: NodeConfig, condition_func: Callable[[GraphState], str]):
        super().__init__(config)
        self.condition_func = condition_func
    
    async def execute(self, state: GraphState) -> GraphState:
        """执行条件判断"""
        state["step_count"] += 1
        
        try:
            # 执行条件判断
            next_node = self.condition_func(state)
            
            # 更新路由信息
            state["context"]["next_node"] = next_node
            state["context"]["routing_reason"] = f"条件节点 {self.config.name} 选择路径: {next_node}"
            
            self.logger.info(f"条件判断完成: {self.config.name} -> {next_node}")
            
        except Exception as e:
            error_msg = f"条件判断失败: {str(e)}"
            state["error"] = error_msg
            state["is_complete"] = True
        
        return state

class StartNode(BaseNode):
    """开始节点"""
    
    def __init__(self):
        super().__init__(NodeConfig(name="__start__", description="图执行开始节点"))
    
    async def execute(self, state: GraphState) -> GraphState:
        """初始化图执行"""
        state["step_count"] = 0
        state["is_complete"] = False
        state["error"] = None
        
        # 添加开始消息
        if not state.get("messages"):
            state["messages"] = []
        
        state["messages"].append({
            "id": "start",
            "role": "system", 
            "content": "图执行开始",
            "timestamp": None,
            "metadata": {"node": "__start__"}
        })
        
        self.logger.info("图执行已开始")
        return state

class EndNode(BaseNode):
    """结束节点"""
    
    def __init__(self, completion_message: str = "图执行完成"):
        config = NodeConfig(name="__end__", description="图执行结束节点")
        super().__init__(config)
        self.completion_message = completion_message
    
    async def execute(self, state: GraphState) -> GraphState:
        """完成图执行"""
        state["is_complete"] = True
        
        # 添加完成消息
        state["messages"].append({
            "id": "end",
            "role": "system",
            "content": self.completion_message,
            "timestamp": None,
            "metadata": {"node": "__end__"}
        })
        
        self.logger.info("图执行已完成")
        return state

# 预定义的条件函数
def message_count_condition(max_messages: int = 10) -> Callable[[GraphState], str]:
    """基于消息数量的条件判断"""
    def condition(state: GraphState) -> str:
        message_count = len(state.get("messages", []))
        if message_count >= max_messages:
            return "__end__"
        return "continue"
    return condition

def error_condition() -> Callable[[GraphState], str]:
    """错误状态条件判断"""
    def condition(state: GraphState) -> str:
        if state.get("error"):
            return "__end__"
        return "continue"
    return condition

def step_limit_condition(max_steps: int = 50) -> Callable[[GraphState], str]:
    """步数限制条件判断"""
    def condition(state: GraphState) -> str:
        if state.get("step_count", 0) >= max_steps:
            return "__end__"
        return "continue"
    return condition
