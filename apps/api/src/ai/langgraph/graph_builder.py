"""
LangGraph图构建器
提供图构建和执行能力
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from .state import MessagesState
from .nodes import BaseNode, StartNode, EndNode, NodeConfig

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class Edge:
    """图边定义"""
    from_node: str
    to_node: str
    condition: Optional[Callable[[GraphState], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentGraph:
    """智能体图定义"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[Edge] = []
        self.start_node = "__start__"
        self.end_nodes = {"__end__"}
        
        # 添加默认的开始和结束节点
        self.add_node("__start__", StartNode())
        self.add_node("__end__", EndNode())
    
    def add_node(self, node_id: str, node: BaseNode):
        """添加节点到图"""
        self.nodes[node_id] = node
        logger.debug(f"添加节点到图: {node_id}")
    
    def add_edge(self, from_node: str, to_node: str, condition: Optional[Callable] = None):
        """添加边到图"""
        edge = Edge(from_node=from_node, to_node=to_node, condition=condition)
        self.edges.append(edge)
        logger.debug(f"添加边到图: {from_node} -> {to_node}")
    
    def add_conditional_edges(self, from_node: str, condition_map: Dict[str, str]):
        """添加条件边"""
        for condition_result, to_node in condition_map.items():
            def make_condition(expected_result):
                return lambda state: state.get("context", {}).get("next_node") == expected_result
            
            self.add_edge(from_node, to_node, make_condition(condition_result))
    
    def get_next_nodes(self, current_node: str, state: GraphState) -> List[str]:
        """获取下一个可执行的节点"""
        next_nodes = []
        
        for edge in self.edges:
            if edge.from_node == current_node:
                # 检查边的条件
                if edge.condition is None or edge.condition(state):
                    next_nodes.append(edge.to_node)
        
        return next_nodes
    
    def validate_graph(self) -> bool:
        """验证图的有效性"""
        # 检查开始节点存在
        if self.start_node not in self.nodes:
            logger.error("图缺少开始节点")
            return False
        
        # 检查至少有一个结束节点
        if not any(end_node in self.nodes for end_node in self.end_nodes):
            logger.error("图缺少结束节点")
            return False
        
        # 检查所有边引用的节点都存在
        for edge in self.edges:
            if edge.from_node not in self.nodes:
                logger.error(f"边引用了不存在的起始节点: {edge.from_node}")
                return False
            if edge.to_node not in self.nodes:
                logger.error(f"边引用了不存在的目标节点: {edge.to_node}")
                return False
        
        # 检查从开始节点是否可达结束节点
        visited = set()
        queue = [self.start_node]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            # 获取所有邻接节点
            for edge in self.edges:
                if edge.from_node == current and edge.to_node not in visited:
                    queue.append(edge.to_node)
        
        # 检查是否能到达任何结束节点
        if not any(end_node in visited for end_node in self.end_nodes):
            logger.error("从开始节点无法到达任何结束节点")
            return False
        
        return True

class GraphBuilder:
    """图构建器"""
    
    def __init__(self):
        self.graphs: Dict[str, AgentGraph] = {}
    
    def create_graph(self, name: str, description: str = "") -> AgentGraph:
        """创建新的图"""
        graph = AgentGraph(name, description)
        self.graphs[name] = graph
        return graph
    
    def get_graph(self, name: str) -> Optional[AgentGraph]:
        """获取图"""
        return self.graphs.get(name)
    
    async def execute_graph(
        self, 
        graph_name: str, 
        initial_message: str,
        session_id: Optional[str] = None,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """执行图"""
        graph = self.get_graph(graph_name)
        if not graph:
            raise ValueError(f"图不存在: {graph_name}")
        
        if not graph.validate_graph():
            raise ValueError(f"图验证失败: {graph_name}")
        
        # 创建执行状态
        state = state_manager.create_state(session_id, max_steps)
        state.add_message("user", initial_message)
        
        logger.info(f"开始执行图: {graph_name}, 会话: {state.session_id}")
        
        try:
            # 转换为LangGraph状态格式
            graph_state = state.to_graph_state()
            
            # 从开始节点开始执行
            current_node = graph.start_node
            
            while not graph_state["is_complete"] and graph_state["step_count"] < max_steps:
                # 执行当前节点
                node = graph.nodes[current_node]
                graph_state = await node.run_with_error_handling(graph_state)
                
                # 检查执行错误
                if graph_state.get("error"):
                    logger.error(f"图执行错误: {graph_state['error']}")
                    break
                
                # 检查是否到达结束节点
                if current_node in graph.end_nodes:
                    graph_state["is_complete"] = True
                    break
                
                # 获取下一个节点
                next_nodes = graph.get_next_nodes(current_node, graph_state)
                
                if not next_nodes:
                    # 没有下一个节点，自动结束
                    graph_state["is_complete"] = True
                    logger.info("图执行完成：没有更多可执行节点")
                    break
                elif len(next_nodes) == 1:
                    # 单一路径
                    current_node = next_nodes[0]
                else:
                    # 多路径，需要条件判断(这里简化处理，选择第一个)
                    current_node = next_nodes[0]
                    logger.warning(f"检测到多个可能路径，选择第一个: {current_node}")
                
                # 防止无限循环
                if graph_state["step_count"] >= max_steps:
                    graph_state["error"] = "达到最大执行步数"
                    graph_state["is_complete"] = True
                    break
            
            # 更新状态管理器中的状态
            state.messages = [
                state.MessageState(
                    id=msg["id"],
                    role=msg["role"],
                    content=msg["content"],
                    metadata=msg.get("metadata", {})
                )
                for msg in graph_state["messages"]
            ]
            state.current_agent = graph_state["current_agent"]
            state.context = graph_state["context"]
            state.step_count = graph_state["step_count"]
            state.is_complete = graph_state["is_complete"]
            state.error = graph_state["error"]
            
            state_manager.update_state(state.session_id, state)
            
            # 构建执行结果
            result = {
                "session_id": state.session_id,
                "graph_name": graph_name,
                "status": "completed" if graph_state["is_complete"] and not graph_state["error"] else "failed",
                "steps_executed": graph_state["step_count"],
                "messages": graph_state["messages"],
                "context": graph_state["context"],
                "error": graph_state["error"]
            }
            
            logger.info(f"图执行完成: {graph_name}, 状态: {result['status']}")
            return result
            
        except Exception as e:
            error_msg = f"图执行异常: {str(e)}"
            logger.error(error_msg)
            state.fail(error_msg)
            raise
    
    def create_simple_chain_graph(
        self, 
        name: str, 
        node_configs: List[Dict[str, Any]]
    ) -> AgentGraph:
        """创建简单的链式图"""
        graph = self.create_graph(name, f"简单链式图: {name}")
        
        previous_node = "__start__"
        
        for i, config in enumerate(node_configs):
            node_id = config.get("id", f"node_{i}")
            node_type = config.get("type", "agent")
            
            if node_type == "agent":
                from .nodes import AgentNode
                node = AgentNode(
                    NodeConfig(
                        name=node_id,
                        description=config.get("description", ""),
                        metadata=config.get("metadata", {})
                    ),
                    config.get("handler")
                )
            elif node_type == "tool":
                from .nodes import ToolNode
                node = ToolNode(
                    NodeConfig(
                        name=node_id,
                        description=config.get("description", ""),
                        metadata=config.get("metadata", {})
                    ),
                    config.get("handler")
                )
            else:
                raise ValueError(f"不支持的节点类型: {node_type}")
            
            graph.add_node(node_id, node)
            graph.add_edge(previous_node, node_id)
            previous_node = node_id
        
        # 连接到结束节点
        graph.add_edge(previous_node, "__end__")
        
        return graph

# 全局图构建器实例
graph_builder = GraphBuilder()
