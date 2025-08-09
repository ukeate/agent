"""
LangGraph StateGraph核心实现
基于LangGraph框架的状态图管理和执行引擎
"""
from typing import Any, Dict, List, Optional, Callable, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState as LangGraphMessagesState
import asyncio
from datetime import datetime

from .state import MessagesState, create_initial_state, validate_state
from .checkpoints import CheckpointManager, checkpoint_manager
from ...core.config import get_settings

settings = get_settings()


class WorkflowNode:
    """工作流节点基类"""
    
    def __init__(self, name: str, handler: Callable[[MessagesState], MessagesState]):
        self.name = name
        self.handler = handler
    
    async def execute(self, state: MessagesState) -> MessagesState:
        """执行节点逻辑"""
        try:
            # 更新元数据
            if "step_count" not in state["metadata"]:
                state["metadata"]["step_count"] = 0
            state["metadata"]["step_count"] += 1
            state["metadata"]["current_node"] = self.name
            state["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # 执行处理逻辑
            result = await self.handler(state) if asyncio.iscoroutinefunction(self.handler) else self.handler(state)
            
            # 添加执行日志
            if "execution_log" not in result["context"]:
                result["context"]["execution_log"] = []
            
            result["context"]["execution_log"].append({
                "node": self.name,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            return result
        except Exception as e:
            # 错误处理
            state["metadata"]["error"] = str(e)
            state["metadata"]["status"] = "failed"
            state["context"]["execution_log"] = state["context"].get("execution_log", [])
            state["context"]["execution_log"].append({
                "node": self.name,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            raise


class ConditionalRouter:
    """条件路由器"""
    
    def __init__(self, name: str, condition_func: Callable[[MessagesState], str]):
        self.name = name
        self.condition_func = condition_func
    
    def route(self, state: MessagesState) -> str:
        """根据状态决定下一个节点"""
        try:
            next_node = self.condition_func(state)
            
            # 记录路由决策
            if "routing_log" not in state["context"]:
                state["context"]["routing_log"] = []
            
            state["context"]["routing_log"].append({
                "router": self.name,
                "timestamp": datetime.now().isoformat(),
                "decision": next_node,
                "state_snapshot": {
                    "step_count": state["metadata"].get("step_count", 0),
                    "status": state["metadata"].get("status", "unknown")
                }
            })
            
            return next_node
        except Exception as e:
            state["metadata"]["error"] = f"路由失败: {str(e)}"
            return END


class LangGraphWorkflowBuilder:
    """LangGraph工作流构建器"""
    
    def __init__(self, checkpoint_manager: CheckpointManager = None):
        self.checkpoint_manager = checkpoint_manager
        self.nodes: Dict[str, WorkflowNode] = {}
        self.routers: Dict[str, ConditionalRouter] = {}
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None
    
    def add_node(self, name: str, handler: Callable[[MessagesState], MessagesState]) -> 'LangGraphWorkflowBuilder':
        """添加工作流节点"""
        self.nodes[name] = WorkflowNode(name, handler)
        return self
    
    def add_conditional_edge(self, from_node: str, router_name: str, condition_func: Callable[[MessagesState], str], path_map: Dict[str, str]) -> 'LangGraphWorkflowBuilder':
        """添加条件边"""
        self.routers[router_name] = ConditionalRouter(router_name, condition_func)
        return self
    
    def build(self) -> StateGraph:
        """构建StateGraph"""
        if self.graph is not None:
            return self.graph
        
        # 创建StateGraph实例
        self.graph = StateGraph(MessagesState)
        
        # 添加所有节点
        for name, node in self.nodes.items():
            self.graph.add_node(name, node.execute)
        
        return self.graph
    
    def compile(self, checkpointer: Optional[Any] = None) -> Any:
        """编译工作流"""
        if self.compiled_graph is not None:
            return self.compiled_graph
        
        if self.graph is None:
            self.build()
        
        # 使用PostgreSQL检查点保存器
        if checkpointer is None and hasattr(settings, 'database_url') and settings.database_url:
            try:
                # checkpointer = PostgresSaver.from_conn_string(settings.database_url)
                print("PostgreSQL检查点保存器暂时禁用")
                checkpointer = None
            except Exception as e:
                print(f"创建PostgreSQL检查点保存器失败: {e}")
                checkpointer = None
        
        self.compiled_graph = self.graph.compile(checkpointer=checkpointer)
        return self.compiled_graph
    
    async def execute(self, initial_state: MessagesState, config: Optional[Dict[str, Any]] = None) -> MessagesState:
        """执行工作流"""
        if self.compiled_graph is None:
            self.compile()
        
        try:
            # 验证初始状态
            if not validate_state(initial_state):
                raise ValueError("无效的初始状态结构")
            
            # 设置执行配置
            execution_config = config or {}
            if "configurable" not in execution_config:
                execution_config["configurable"] = {}
            
            # 添加工作流ID到配置
            thread_id = initial_state.get("workflow_id", f"workflow_{datetime.now().timestamp()}")
            execution_config["configurable"]["thread_id"] = thread_id
            
            # 执行工作流
            result = await self.compiled_graph.ainvoke(initial_state, config=execution_config)
            
            # 更新最终状态
            result["metadata"]["status"] = "completed"
            result["metadata"]["completed_at"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            # 保存失败状态
            initial_state["metadata"]["status"] = "failed"
            initial_state["metadata"]["error"] = str(e)
            initial_state["metadata"]["failed_at"] = datetime.now().isoformat()
            
            # 创建错误检查点
            if self.checkpoint_manager:
                await self.checkpoint_manager.create_checkpoint(
                    workflow_id=initial_state["workflow_id"],
                    state=initial_state,
                    metadata={"error": str(e), "type": "failure_checkpoint"}
                )
            
            raise
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流执行"""
        try:
            # 获取当前状态并创建暂停检查点
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(workflow_id)
            if latest_checkpoint:
                latest_checkpoint.state["metadata"]["status"] = "paused"
                latest_checkpoint.state["metadata"]["paused_at"] = datetime.now().isoformat()
                
                await self.checkpoint_manager.create_checkpoint(
                    workflow_id=workflow_id,
                    state=latest_checkpoint.state,
                    metadata={"type": "pause_checkpoint"}
                )
                return True
            return False
        except Exception as e:
            print(f"暂停工作流失败: {e}")
            return False
    
    async def resume_workflow(self, workflow_id: str, config: Optional[Dict[str, Any]] = None) -> Optional[MessagesState]:
        """恢复工作流执行"""
        try:
            # 从最新检查点恢复
            state = await self.checkpoint_manager.restore_from_checkpoint(workflow_id, "latest")
            if not state:
                return None
            
            # 更新状态为运行中
            state["metadata"]["status"] = "running"
            state["metadata"]["resumed_at"] = datetime.now().isoformat()
            
            # 继续执行
            return await self.execute(state, config)
            
        except Exception as e:
            print(f"恢复工作流失败: {e}")
            return None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流执行"""
        try:
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(workflow_id)
            if latest_checkpoint:
                latest_checkpoint.state["metadata"]["status"] = "cancelled"
                latest_checkpoint.state["metadata"]["cancelled_at"] = datetime.now().isoformat()
                
                await self.checkpoint_manager.create_checkpoint(
                    workflow_id=workflow_id,
                    state=latest_checkpoint.state,
                    metadata={"type": "cancellation_checkpoint"}
                )
                return True
            return False
        except Exception as e:
            print(f"取消工作流失败: {e}")
            return False


def create_simple_workflow() -> LangGraphWorkflowBuilder:
    """创建简单线性工作流示例"""
    builder = LangGraphWorkflowBuilder()
    
    def start_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "system",
            "content": "工作流开始执行",
            "timestamp": datetime.now().isoformat()
        })
        return state
    
    def process_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "assistant", 
            "content": "正在处理任务...",
            "timestamp": datetime.now().isoformat()
        })
        state["context"]["processed"] = True
        return state
    
    def end_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "system",
            "content": "工作流执行完成",
            "timestamp": datetime.now().isoformat()
        })
        return state
    
    # 构建工作流
    builder.add_node("start", start_node)
    builder.add_node("process", process_node)
    builder.add_node("end", end_node)
    
    # 构建图并添加边
    graph = builder.build()
    graph.add_edge(START, "start")
    graph.add_edge("start", "process") 
    graph.add_edge("process", "end")
    graph.add_edge("end", END)
    
    return builder


def create_conditional_workflow() -> LangGraphWorkflowBuilder:
    """创建带条件分支的工作流示例"""
    builder = LangGraphWorkflowBuilder()
    
    def analyze_node(state: MessagesState) -> MessagesState:
        import random
        state["context"]["analysis_result"] = "success" if random.random() > 0.3 else "failure"
        state["messages"].append({
            "role": "system",
            "content": f"分析完成，结果: {state['context']['analysis_result']}",
            "timestamp": datetime.now().isoformat()
        })
        return state
    
    def success_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "assistant",
            "content": "执行成功分支处理",
            "timestamp": datetime.now().isoformat()
        })
        return state
    
    def failure_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "assistant", 
            "content": "执行失败分支处理",
            "timestamp": datetime.now().isoformat()
        })
        return state
    
    def route_condition(state: MessagesState) -> str:
        result = state["context"].get("analysis_result", "failure")
        return "success" if result == "success" else "failure"
    
    # 构建工作流
    builder.add_node("analyze", analyze_node)
    builder.add_node("success", success_node)
    builder.add_node("failure", failure_node)
    
    # 构建图并添加边
    graph = builder.build()
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_condition,
        {
            "success": "success",
            "failure": "failure"
        }
    )
    graph.add_edge("success", END)
    graph.add_edge("failure", END)
    
    return builder