"""
LangGraph StateGraph测试
"""
import pytest
from datetime import datetime
from src.ai.langgraph.state import MessagesState, create_initial_state, validate_state
from src.ai.langgraph.state_graph import (
    LangGraphWorkflowBuilder,
    WorkflowNode,
    ConditionalRouter,
    create_simple_workflow,
    create_conditional_workflow
)


class TestMessagesState:
    """测试MessagesState数据结构"""
    
    def test_create_initial_state(self):
        """测试创建初始状态"""
        state = create_initial_state("test-workflow-123")
        
        assert state["workflow_id"] == "test-workflow-123"
        assert state["messages"] == []
        assert "created_at" in state["metadata"]
        assert state["metadata"]["status"] == "pending"
        assert state["context"] == {}
    
    def test_validate_state(self):
        """测试状态验证"""
        # 有效状态
        valid_state = create_initial_state("test")
        assert validate_state(valid_state) is True
        
        # 无效状态 - 缺少必要字段
        invalid_state = {"messages": []}
        assert validate_state(invalid_state) is False
        
        # 无效状态 - 字段类型错误
        invalid_state2 = {
            "messages": "not a list",
            "metadata": {},
            "context": {},
            "workflow_id": "test"
        }
        assert validate_state(invalid_state2) is False


class TestWorkflowNode:
    """测试工作流节点"""
    
    @pytest.mark.asyncio
    async def test_node_execution(self):
        """测试节点执行"""
        def handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "system",
                "content": "节点执行成功"
            })
            return state
        
        node = WorkflowNode("test_node", handler)
        initial_state = create_initial_state("test")
        
        result = await node.execute(initial_state)
        
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "节点执行成功"
        assert result["metadata"]["current_node"] == "test_node"
        assert result["metadata"]["step_count"] == 1
        assert "last_updated" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """测试节点错误处理"""
        def error_handler(state: MessagesState) -> MessagesState:
            raise ValueError("测试错误")
        
        node = WorkflowNode("error_node", error_handler)
        initial_state = create_initial_state("test")
        
        with pytest.raises(ValueError):
            await node.execute(initial_state)
        
        # 检查错误状态
        assert initial_state["metadata"]["error"] == "测试错误"
        assert initial_state["metadata"]["status"] == "failed"


class TestConditionalRouter:
    """测试条件路由器"""
    
    def test_routing_decision(self):
        """测试路由决策"""
        def condition_func(state: MessagesState) -> str:
            if state["context"].get("success"):
                return "success_path"
            return "failure_path"
        
        router = ConditionalRouter("test_router", condition_func)
        
        # 测试成功路径
        state_success = create_initial_state("test")
        state_success["context"]["success"] = True
        assert router.route(state_success) == "success_path"
        
        # 测试失败路径
        state_failure = create_initial_state("test")
        state_failure["context"]["success"] = False
        assert router.route(state_failure) == "failure_path"
        
        # 检查路由日志
        assert "routing_log" in state_success["context"]
        assert len(state_success["context"]["routing_log"]) == 1
        assert state_success["context"]["routing_log"][0]["decision"] == "success_path"


class TestLangGraphWorkflowBuilder:
    """测试工作流构建器"""
    
    def test_add_nodes(self):
        """测试添加节点"""
        builder = LangGraphWorkflowBuilder()
        
        def node1_handler(state): return state
        def node2_handler(state): return state
        
        builder.add_node("node1", node1_handler)
        builder.add_node("node2", node2_handler)
        
        assert "node1" in builder.nodes
        assert "node2" in builder.nodes
        assert len(builder.nodes) == 2
    
    def test_build_graph(self):
        """测试构建图"""
        builder = LangGraphWorkflowBuilder()
        
        def handler(state): return state
        builder.add_node("test_node", handler)
        
        graph = builder.build()
        assert graph is not None
        assert builder.graph is not None
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self):
        """测试简单工作流执行"""
        builder = create_simple_workflow()
        initial_state = create_initial_state("simple-workflow")
        
        # 编译并执行
        builder.compile()
        result = await builder.execute(initial_state)
        
        # 验证执行结果
        assert result["metadata"]["status"] == "completed"
        assert len(result["messages"]) >= 3  # 至少有开始、处理、结束消息
        assert "completed_at" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_conditional_workflow_execution(self):
        """测试条件工作流执行"""
        builder = create_conditional_workflow()
        initial_state = create_initial_state("conditional-workflow")
        
        # 编译并执行
        builder.compile()
        result = await builder.execute(initial_state)
        
        # 验证执行结果
        assert result["metadata"]["status"] == "completed"
        assert "analysis_result" in result["context"]
        assert result["context"]["analysis_result"] in ["success", "failure"]
    
    @pytest.mark.asyncio
    async def test_workflow_pause_resume(self):
        """测试工作流暂停和恢复"""
        builder = create_simple_workflow()
        workflow_id = "pause-test-workflow"
        initial_state = create_initial_state(workflow_id)
        
        # 模拟暂停操作
        success = await builder.pause_workflow(workflow_id)
        # 由于没有实际执行，暂停应该失败
        assert success is False
    
    @pytest.mark.asyncio
    async def test_workflow_cancel(self):
        """测试工作流取消"""
        builder = create_simple_workflow()
        workflow_id = "cancel-test-workflow"
        
        # 模拟取消操作
        success = await builder.cancel_workflow(workflow_id)
        # 由于没有检查点，取消应该失败
        assert success is False
    
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self):
        """测试工作流错误恢复"""
        builder = LangGraphWorkflowBuilder()
        
        def error_node(state: MessagesState) -> MessagesState:
            raise RuntimeError("模拟错误")
        
        builder.add_node("error", error_node)
        graph = builder.build()
        graph.add_edge("__start__", "error")
        graph.add_edge("error", "__end__")
        
        initial_state = create_initial_state("error-workflow")
        
        with pytest.raises(RuntimeError):
            await builder.execute(initial_state)
        
        # 验证错误状态
        assert initial_state["metadata"]["status"] == "failed"
        assert "模拟错误" in initial_state["metadata"]["error"]
        assert "failed_at" in initial_state["metadata"]