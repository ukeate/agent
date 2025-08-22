"""
LangGraph StateGraph核心实现
基于LangGraph 0.6.5框架的状态图管理和执行引擎
支持新Context API和durability控制
"""
from typing import Any, Dict, List, Optional, Callable, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState as LangGraphMessagesState
from langgraph.types import RunnableConfig
from langgraph.runtime import Runtime, get_runtime
import asyncio
from datetime import datetime, timezone

from .state import MessagesState, create_initial_state, validate_state
from .checkpoints import CheckpointManager, checkpoint_manager
from .context import AgentContext, create_default_context, validate_context, LangGraphContextSchema
from src.core.config import get_settings

settings = get_settings()


class WorkflowNode:
    """工作流节点基类"""
    
    def __init__(self, name: str, handler: Callable[[MessagesState], MessagesState], node_type: str = "standard"):
        self.name = name
        self.handler = handler
        self.node_type = node_type  # 支持: standard, reasoning, validation, branching
    
    async def execute(self, state: MessagesState, config: Optional[RunnableConfig] = None, runtime: Optional[Runtime[LangGraphContextSchema]] = None) -> MessagesState:
        """执行节点逻辑 - 支持新旧两种Context API"""
        try:
            # 优先使用新的Runtime Context API (LangGraph 0.6.5+)
            context = None
            if runtime and hasattr(runtime, 'context') and runtime.context is not None:
                # 新Context API
                ctx_schema = runtime.context
                context = ctx_schema.to_agent_context()
                context.update_step(self.name)
            elif config and 'configurable' in config:
                # 向后兼容：旧的config['configurable']模式
                context_data = config['configurable']
                # 只提取我们的上下文相关字段
                context_fields = {
                    'user_id': context_data.get('user_id', 'unknown'),
                    'session_id': context_data.get('session_id', 'default'),
                    'conversation_id': context_data.get('conversation_id'),
                    'agent_id': context_data.get('agent_id'),
                    'workflow_id': context_data.get('workflow_id'),
                    'thread_id': context_data.get('thread_id'),
                }
                context = AgentContext.from_dict(context_fields)
                context.update_step(self.name)
            
            # 更新元数据
            if "step_count" not in state["metadata"]:
                state["metadata"]["step_count"] = 0
            state["metadata"]["step_count"] += 1
            state["metadata"]["current_node"] = self.name
            state["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # 执行处理逻辑，检查handler是否接受context参数
            import inspect
            sig = inspect.signature(self.handler)
            params = list(sig.parameters.keys())
            
            if asyncio.iscoroutinefunction(self.handler):
                if len(params) > 1 and 'context' in params:
                    result = await self.handler(state, context)
                else:
                    result = await self.handler(state)
            else:
                if len(params) > 1 and 'context' in params:
                    result = self.handler(state, context)
                else:
                    result = self.handler(state)
            
            # 添加执行日志
            if "execution_log" not in result["context"]:
                result["context"]["execution_log"] = []
            
            result["context"]["execution_log"].append({
                "node": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": str(e)
            })
            raise


class ConditionalRouter:
    """条件路由器"""
    
    def __init__(self, name: str, condition_func: Callable[[MessagesState], str]):
        self.name = name
        self.condition_func = condition_func
    
    def route(self, state: MessagesState, config: Optional[RunnableConfig] = None, runtime: Optional[Runtime[LangGraphContextSchema]] = None) -> str:
        """根据状态决定下一个节点 - 支持新旧两种Context API"""
        try:
            # 检查condition_func是否接受context参数
            import inspect
            sig = inspect.signature(self.condition_func)
            params = list(sig.parameters.keys())
            
            if len(params) > 1 and 'context' in params:
                # 提取上下文并传递
                context = None
                if runtime and hasattr(runtime, 'context'):
                    # 新Context API
                    ctx_schema = runtime.context
                    context = ctx_schema.to_agent_context()
                elif config and 'configurable' in config:
                    # 向后兼容：旧的config['configurable']模式
                    context_data = config['configurable']
                    context_fields = {
                        'user_id': context_data.get('user_id', 'unknown'),
                        'session_id': context_data.get('session_id', 'default'),
                        'conversation_id': context_data.get('conversation_id'),
                        'agent_id': context_data.get('agent_id'),
                        'workflow_id': context_data.get('workflow_id'),
                        'thread_id': context_data.get('thread_id'),
                    }
                    context = AgentContext.from_dict(context_fields)
                next_node = self.condition_func(state, context)
            else:
                next_node = self.condition_func(state)
            
            # 记录路由决策
            if "routing_log" not in state["context"]:
                state["context"]["routing_log"] = []
            
            state["context"]["routing_log"].append({
                "router": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
    """LangGraph 0.6.5工作流构建器 - 支持新Context API"""
    
    def __init__(self, checkpoint_manager: CheckpointManager = None, use_context_api: bool = True):
        self.checkpoint_manager = checkpoint_manager
        self.nodes: Dict[str, WorkflowNode] = {}
        self.routers: Dict[str, ConditionalRouter] = {}
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None
        self.use_context_api = use_context_api  # 控制是否使用新Context API
    
    def add_node(self, name: str, handler: Callable[[MessagesState], MessagesState], node_type: str = "standard") -> 'LangGraphWorkflowBuilder':
        """添加工作流节点
        
        Args:
            name: 节点名称
            handler: 节点处理函数
            node_type: 节点类型 (standard, reasoning, validation, branching)
        """
        self.nodes[name] = WorkflowNode(name, handler, node_type)
        return self
    
    def add_reasoning_node(self, name: str, handler: Callable[[MessagesState], MessagesState]) -> 'LangGraphWorkflowBuilder':
        """添加推理节点"""
        return self.add_node(name, handler, "reasoning")
    
    def add_conditional_edge(self, from_node: str, router_name: str, condition_func: Callable[[MessagesState], str], path_map: Dict[str, str]) -> 'LangGraphWorkflowBuilder':
        """添加条件边"""
        self.routers[router_name] = ConditionalRouter(router_name, condition_func)
        return self
    
    def build(self) -> StateGraph:
        """构建StateGraph - 支持LangGraph 0.6.5新Context API"""
        if self.graph is not None:
            return self.graph
        
        # 创建StateGraph实例，支持context_schema
        if self.use_context_api:
            # 使用新的Context API (LangGraph 0.6.5+)
            self.graph = StateGraph(MessagesState, context_schema=LangGraphContextSchema)
        else:
            # 向后兼容旧版本
            self.graph = StateGraph(MessagesState)
        
        # 添加所有节点，包装以支持新旧两种API
        for name, node in self.nodes.items():
            if self.use_context_api:
                # 新Context API包装函数
                def create_node_wrapper_new(node_instance):
                    async def node_wrapper(state: MessagesState, *, config: RunnableConfig = None) -> MessagesState:
                        # 尝试获取runtime context
                        runtime = None
                        try:
                            runtime = get_runtime()
                        except Exception:
                            pass  # runtime可能不可用，使用fallback
                        return await node_instance.execute(state, config, runtime)
                    return node_wrapper
                
                self.graph.add_node(name, create_node_wrapper_new(node))
            else:
                # 旧版本包装函数
                def create_node_wrapper_old(node_instance):
                    async def node_wrapper(state: MessagesState, config: RunnableConfig = None) -> MessagesState:
                        return await node_instance.execute(state, config)
                    return node_wrapper
                
                self.graph.add_node(name, create_node_wrapper_old(node))
        
        return self.graph
    
    def compile(self, checkpointer: Optional[Any] = None, durability_mode: Literal["exit", "async", "sync"] = "async") -> Any:
        """编译工作流 - 支持durability控制
        
        Args:
            checkpointer: 检查点保存器
            durability_mode: 默认持久化模式
        """
        if self.compiled_graph is not None:
            return self.compiled_graph
        
        if self.graph is None:
            self.build()
        
        # 存储默认durability模式
        self.default_durability = durability_mode
        
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
    
    async def execute(self, initial_state: MessagesState, context: Optional[AgentContext] = None, config: Optional[Dict[str, Any]] = None, durability: Literal["exit", "async", "sync"] = "async") -> MessagesState:
        """执行工作流 - 支持LangGraph 0.6.5新特性
        
        Args:
            initial_state: 初始状态
            context: 智能体上下文（新Context API）
            config: 额外配置（向后兼容）
            durability: 持久化策略 ("exit", "async", "sync")
        """
        if self.compiled_graph is None:
            self.compile()
        
        try:
            # 验证初始状态
            if not validate_state(initial_state):
                raise ValueError("无效的初始状态结构")
            
            # 创建或验证上下文
            if context is None:
                # 向后兼容：从config中提取上下文数据
                if config and "configurable" in config:
                    context = AgentContext.from_dict(config["configurable"])
                else:
                    context = create_default_context()
            
            if not validate_context(context):
                raise ValueError("无效的上下文结构")
            
            # 准备执行参数
            if self.use_context_api:
                # 使用新Context API
                ctx_schema = LangGraphContextSchema.from_agent_context(context)
                thread_id = context.workflow_id or initial_state.get("workflow_id", f"workflow_{datetime.now(timezone.utc).timestamp()}")
                ctx_schema.thread_id = thread_id
                context.thread_id = thread_id
                
                # 使用真实的durability控制 (LangGraph 0.6.5+)
                # 根据Context7文档，durability参数需要通过stream配置传递
                config_dict = {
                    "configurable": {
                        "thread_id": thread_id,
                        **ctx_schema.to_dict()
                    }
                }
                
                # 使用stream方法执行以支持durability控制
                final_result = None
                async for chunk in self.compiled_graph.astream(
                    initial_state,
                    config=config_dict,
                    # durability参数控制持久化策略:
                    # - "exit": 仅在图完成时持久化（最佳性能）
                    # - "async": 异步持久化（良好性能和持久性）
                    # - "sync": 同步持久化（最高持久性）
                    durability=durability
                ):
                    final_result = chunk
                result = final_result or initial_state
            else:
                # 向后兼容：使用旧的config模式
                execution_config = config or {}
                if "configurable" not in execution_config:
                    execution_config["configurable"] = {}
                
                # 将上下文数据合并到configurable中
                execution_config["configurable"].update(context.to_dict())
                
                # 添加工作流ID
                thread_id = context.workflow_id or initial_state.get("workflow_id", f"workflow_{datetime.now(timezone.utc).timestamp()}")
                execution_config["configurable"]["thread_id"] = thread_id
                context.thread_id = thread_id
                
                # 执行工作流
                result = await self.compiled_graph.ainvoke(initial_state, config=execution_config)
            
            # 更新最终状态
            result["metadata"]["status"] = "completed"
            result["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            return result
            
        except Exception as e:
            # 保存失败状态
            initial_state["metadata"]["status"] = "failed"
            initial_state["metadata"]["error"] = str(e)
            initial_state["metadata"]["failed_at"] = datetime.now(timezone.utc).isoformat()
            
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
                latest_checkpoint.state["metadata"]["paused_at"] = datetime.now(timezone.utc).isoformat()
                
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
    
    async def resume_workflow(self, workflow_id: str, context: Optional[AgentContext] = None, config: Optional[Dict[str, Any]] = None) -> Optional[MessagesState]:
        """恢复工作流执行"""
        try:
            # 从最新检查点恢复
            state = await self.checkpoint_manager.restore_from_checkpoint(workflow_id, "latest")
            if not state:
                return None
            
            # 更新状态为运行中
            state["metadata"]["status"] = "running"
            state["metadata"]["resumed_at"] = datetime.now(timezone.utc).isoformat()
            
            # 继续执行
            return await self.execute(state, context, config)
            
        except Exception as e:
            print(f"恢复工作流失败: {e}")
            return None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流执行"""
        try:
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(workflow_id)
            if latest_checkpoint:
                latest_checkpoint.state["metadata"]["status"] = "cancelled"
                latest_checkpoint.state["metadata"]["cancelled_at"] = datetime.now(timezone.utc).isoformat()
                
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return state
    
    def process_node(state: MessagesState) -> MessagesState:
        import random
        import json
        
        # 生成真实的模拟数据处理结果
        sample_data = []
        categories = ['用户行为', '交易记录', '系统日志', '设备状态', '网络流量']
        
        # 生成随机数量的数据记录
        record_count = random.randint(120, 200)
        
        for i in range(record_count):
            record = {
                'id': f'record_{i+1:04d}',
                'category': random.choice(categories),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'value': round(random.uniform(10.5, 999.9), 2),
                'status': random.choice(['success', 'warning', 'info']),
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            sample_data.append(record)
        
        # 计算处理统计
        processing_stats = {
            'total_records': len(sample_data),
            'by_category': {},
            'by_status': {},
            'avg_value': round(sum(r['value'] for r in sample_data) / len(sample_data), 2),
            'processing_time_ms': random.randint(200, 500)
        }
        
        # 统计各类别数量
        for record in sample_data:
            category = record['category']
            status = record['status']
            processing_stats['by_category'][category] = processing_stats['by_category'].get(category, 0) + 1
            processing_stats['by_status'][status] = processing_stats['by_status'].get(status, 0) + 1
        
        # 保存处理结果到状态
        state["context"]["processed_data"] = sample_data
        state["context"]["processing_stats"] = processing_stats
        state["context"]["processed"] = True
        
        # 添加处理消息
        state["messages"].append({
            "role": "assistant", 
            "content": f"数据处理完成，共处理 {processing_stats['total_records']} 条记录",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "processing_stats": processing_stats,
                "sample_records": sample_data[:5]  # 只包含前5条作为示例
            }
        })
        
        return state
    
    def end_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "system",
            "content": "工作流执行完成",
            "timestamp": datetime.now(timezone.utc).isoformat()
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
    """创建完整的条件分支工作流：开始 → 数据处理 → 条件判断 → 路径A/B → 结束"""
    builder = LangGraphWorkflowBuilder()
    
    def start_node(state: MessagesState) -> MessagesState:
        state["messages"].append({
            "role": "system",
            "content": "工作流开始执行",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return state
    
    def process_node(state: MessagesState) -> MessagesState:
        import random
        import json
        
        # 生成真实的模拟数据处理结果
        sample_data = []
        categories = ['用户行为', '交易记录', '系统日志', '设备状态', '网络流量']
        
        # 生成随机数量的数据记录
        record_count = random.randint(120, 200)
        
        for i in range(record_count):
            record = {
                'id': f'record_{i+1:04d}',
                'category': random.choice(categories),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'value': round(random.uniform(10.5, 999.9), 2),
                'status': random.choice(['success', 'warning', 'info']),
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            sample_data.append(record)
        
        # 计算处理统计
        processing_stats = {
            'total_records': len(sample_data),
            'by_category': {},
            'by_status': {},
            'avg_value': round(sum(r['value'] for r in sample_data) / len(sample_data), 2),
            'processing_time_ms': random.randint(200, 500)
        }
        
        # 统计各类别数量
        for record in sample_data:
            category = record['category']
            status = record['status']
            processing_stats['by_category'][category] = processing_stats['by_category'].get(category, 0) + 1
            processing_stats['by_status'][status] = processing_stats['by_status'].get(status, 0) + 1
        
        # 保存处理结果到状态
        state["context"]["processed_data"] = sample_data
        state["context"]["processing_stats"] = processing_stats
        state["context"]["processed"] = True
        
        # 根据数据质量决定路径（基于错误率）
        error_rate = processing_stats['by_status'].get('warning', 0) / processing_stats['total_records']
        state["context"]["data_quality"] = "high" if error_rate < 0.3 else "low"
        
        # 添加处理消息
        state["messages"].append({
            "role": "assistant", 
            "content": f"数据处理完成，共处理 {processing_stats['total_records']} 条记录，错误率: {error_rate:.1%}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "processing_stats": processing_stats,
                "sample_records": sample_data[:5]  # 只包含前5条作为示例
            }
        })
        
        return state
    
    def decision_node(state: MessagesState) -> MessagesState:
        data_quality = state["context"].get("data_quality", "low")
        state["messages"].append({
            "role": "system",
            "content": f"条件判断完成，数据质量: {data_quality}，选择对应处理路径",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        state["context"]["decision_result"] = data_quality
        return state
    
    def path_a_node(state: MessagesState) -> MessagesState:
        # 高质量数据处理路径
        state["messages"].append({
            "role": "assistant",
            "content": "执行路径A：高质量数据优化处理",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        state["context"]["optimization_applied"] = True
        return state
    
    def path_b_node(state: MessagesState) -> MessagesState:
        # 低质量数据处理路径
        state["messages"].append({
            "role": "assistant", 
            "content": "执行路径B：低质量数据清理处理",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        state["context"]["cleaning_applied"] = True
        return state
    
    def end_node(state: MessagesState) -> MessagesState:
        path_taken = "路径A" if state["context"].get("optimization_applied") else "路径B"
        state["messages"].append({
            "role": "system",
            "content": f"工作流执行完成，通过{path_taken}处理",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return state
    
    def route_condition(state: MessagesState) -> str:
        decision = state["context"].get("decision_result", "low")
        return "path_a" if decision == "high" else "path_b"
    
    # 构建工作流
    builder.add_node("start", start_node)
    builder.add_node("process", process_node)
    builder.add_node("decision", decision_node)
    builder.add_node("path_a", path_a_node)
    builder.add_node("path_b", path_b_node)
    builder.add_node("end", end_node)
    
    # 构建图并添加边
    graph = builder.build()
    graph.add_edge(START, "start")
    graph.add_edge("start", "process")
    graph.add_edge("process", "decision")
    graph.add_conditional_edges(
        "decision",
        route_condition,
        {
            "path_a": "path_a",
            "path_b": "path_b"
        }
    )
    graph.add_edge("path_a", "end")
    graph.add_edge("path_b", "end")
    graph.add_edge("end", END)
    
    return builder