"""
LangGraph StateGraph测试
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ai.langgraph.state import MessagesState, create_initial_state, validate_state, serialize_state, deserialize_state
from ai.langgraph.state_graph import (
    LangGraphWorkflowBuilder, WorkflowNode, ConditionalRouter,
    create_simple_workflow, create_conditional_workflow
)
from ai.langgraph.checkpoints import CheckpointManager, Checkpoint
from ai.langgraph.error_handling import WorkflowErrorRecovery, ErrorType
from ai.langgraph.timeout_control import TimeoutManager, TimeoutConfig, CancellationToken


class TestMessagesState:
    """MessagesState测试"""
    
    def test_create_initial_state(self):
        """测试创建初始状态"""
        workflow_id = "test-workflow-123"
        state = create_initial_state(workflow_id)
        
        assert state["workflow_id"] == workflow_id
        assert state["messages"] == []
        assert "created_at" in state["metadata"]
        assert state["metadata"]["step_count"] == 0
        assert state["metadata"]["status"] == "pending"
        assert state["context"] == {}
    
    def test_validate_state(self):
        """测试状态验证"""
        # 有效状态
        valid_state = create_initial_state("test-id")
        assert validate_state(valid_state) == True
        
        # 无效状态
        invalid_state = {"messages": [], "metadata": {}}  # 缺少required字段
        assert validate_state(invalid_state) == False
    
    def test_serialize_deserialize_state(self):
        """测试状态序列化和反序列化"""
        original_state = create_initial_state("test-workflow")
        original_state["messages"].append({
            "role": "user",
            "content": "测试消息",
            "timestamp": datetime.now().isoformat()
        })
        original_state["context"]["test_key"] = "test_value"
        
        # 序列化
        serialized = serialize_state(original_state)
        assert isinstance(serialized, str)
        
        # 反序列化
        deserialized = deserialize_state(serialized)
        assert deserialized["workflow_id"] == original_state["workflow_id"]
        assert len(deserialized["messages"]) == 1
        assert deserialized["messages"][0]["content"] == "测试消息"
        assert deserialized["context"]["test_key"] == "test_value"


class TestWorkflowNode:
    """WorkflowNode测试"""
    
    @pytest.mark.asyncio
    async def test_node_execution(self):
        """测试节点执行"""
        def test_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "节点处理完成"
            })
            return state
        
        node = WorkflowNode("test_node", test_handler)
        state = create_initial_state("test-workflow")
        
        result = await node.execute(state)
        
        assert result["metadata"]["current_node"] == "test_node"
        assert result["metadata"]["step_count"] == 1
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "节点处理完成"
        assert len(result["context"]["execution_log"]) == 1
        assert result["context"]["execution_log"][0]["node"] == "test_node"
        assert result["context"]["execution_log"][0]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """测试节点错误处理"""
        def error_handler(state: MessagesState) -> MessagesState:
            raise ValueError("测试错误")
        
        node = WorkflowNode("error_node", error_handler)
        state = create_initial_state("test-workflow")
        
        with pytest.raises(ValueError):
            await node.execute(state)
        
        assert state["metadata"]["error"] == "测试错误"
        assert state["metadata"]["status"] == "failed"
        assert len(state["context"]["execution_log"]) == 1
        assert state["context"]["execution_log"][0]["status"] == "failed"


class TestConditionalRouter:
    """ConditionalRouter测试"""
    
    def test_routing_decision(self):
        """测试路由决策"""
        def condition_func(state: MessagesState) -> str:
            if state["context"].get("success", False):
                return "success_node"
            else:
                return "failure_node"
        
        router = ConditionalRouter("test_router", condition_func)
        
        # 成功路径
        success_state = create_initial_state("test-workflow")
        success_state["context"]["success"] = True
        result = router.route(success_state)
        assert result == "success_node"
        
        # 失败路径
        failure_state = create_initial_state("test-workflow")
        failure_state["context"]["success"] = False
        result = router.route(failure_state)
        assert result == "failure_node"
        
        # 检查路由日志
        assert len(success_state["context"]["routing_log"]) == 1
        assert success_state["context"]["routing_log"][0]["decision"] == "success_node"


class TestLangGraphWorkflowBuilder:
    """LangGraphWorkflowBuilder测试"""
    
    def test_workflow_builder_creation(self):
        """测试工作流构建器创建"""
        builder = LangGraphWorkflowBuilder()
        assert builder is not None
        assert builder.nodes == {}
        assert builder.routers == {}
        assert builder.graph is None
    
    def test_add_node(self):
        """测试添加节点"""
        builder = LangGraphWorkflowBuilder()
        
        def test_handler(state: MessagesState) -> MessagesState:
            return state
        
        result = builder.add_node("test_node", test_handler)
        assert result == builder  # 支持链式调用
        assert "test_node" in builder.nodes
        assert builder.nodes["test_node"].name == "test_node"
    
    def test_build_graph(self):
        """测试构建图"""
        builder = LangGraphWorkflowBuilder()
        
        def dummy_handler(state: MessagesState) -> MessagesState:
            return state
        
        builder.add_node("node1", dummy_handler)
        builder.add_node("node2", dummy_handler)
        
        graph = builder.build()
        assert graph is not None
        assert builder.graph == graph
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """测试工作流执行"""
        builder = create_simple_workflow()
        initial_state = create_initial_state("test-workflow")
        
        # Mock checkpointer to avoid database dependency
        with patch('ai.langgraph.state_graph.PostgresSaver') as mock_saver:
            mock_saver.from_conn_string.return_value = None
            compiled_graph = builder.compile()
            
            # Mock the compiled graph execution
            compiled_graph.ainvoke = AsyncMock(return_value=initial_state)
            
            result = await builder.execute(initial_state)
            assert result["metadata"]["status"] == "completed"
            assert "completed_at" in result["metadata"]


class TestSimpleWorkflow:
    """简单工作流测试"""
    
    def test_create_simple_workflow(self):
        """测试创建简单工作流"""
        builder = create_simple_workflow()
        assert len(builder.nodes) == 3
        assert "start" in builder.nodes
        assert "process" in builder.nodes
        assert "end" in builder.nodes
    
    @pytest.mark.asyncio
    async def test_simple_workflow_nodes(self):
        """测试简单工作流节点执行"""
        builder = create_simple_workflow()
        state = create_initial_state("test-workflow")
        
        # 测试start节点
        start_result = await builder.nodes["start"].execute(state)
        assert len(start_result["messages"]) == 1
        assert "工作流开始执行" in start_result["messages"][0]["content"]
        
        # 测试process节点
        process_result = await builder.nodes["process"].execute(start_result)
        assert len(process_result["messages"]) == 2
        assert process_result["context"]["processed"] == True
        
        # 测试end节点
        end_result = await builder.nodes["end"].execute(process_result)
        assert len(end_result["messages"]) == 3
        assert "工作流执行完成" in end_result["messages"][-1]["content"]


class TestConditionalWorkflow:
    """条件工作流测试"""
    
    def test_create_conditional_workflow(self):
        """测试创建条件工作流"""
        builder = create_conditional_workflow()
        assert len(builder.nodes) == 3
        assert "analyze" in builder.nodes
        assert "success" in builder.nodes
        assert "failure" in builder.nodes
    
    @pytest.mark.asyncio
    async def test_conditional_workflow_execution(self):
        """测试条件工作流执行"""
        builder = create_conditional_workflow()
        state = create_initial_state("test-workflow")
        
        # 执行分析节点
        analyze_result = await builder.nodes["analyze"].execute(state)
        assert "analysis_result" in analyze_result["context"]
        assert analyze_result["context"]["analysis_result"] in ["success", "failure"]
        
        # 根据结果执行对应节点
        if analyze_result["context"]["analysis_result"] == "success":
            success_result = await builder.nodes["success"].execute(analyze_result)
            assert "执行成功分支处理" in success_result["messages"][-1]["content"]
        else:
            failure_result = await builder.nodes["failure"].execute(analyze_result)
            assert "执行失败分支处理" in failure_result["messages"][-1]["content"]


class TestWorkflowControl:
    """工作流控制测试"""
    
    @pytest.mark.asyncio
    async def test_pause_workflow(self):
        """测试暂停工作流"""
        builder = LangGraphWorkflowBuilder()
        
        with patch.object(builder.checkpoint_manager, 'get_latest_checkpoint') as mock_get, \
             patch.object(builder.checkpoint_manager, 'create_checkpoint') as mock_create:
            
            # Mock checkpoint
            mock_checkpoint = Mock()
            mock_checkpoint.state = create_initial_state("test-workflow")
            mock_get.return_value = mock_checkpoint
            mock_create.return_value = None
            
            result = await builder.pause_workflow("test-workflow")
            assert result == True
            assert mock_checkpoint.state["metadata"]["status"] == "paused"
            assert "paused_at" in mock_checkpoint.state["metadata"]
    
    @pytest.mark.asyncio
    async def test_resume_workflow(self):
        """测试恢复工作流"""
        builder = LangGraphWorkflowBuilder()
        
        with patch.object(builder.checkpoint_manager, 'restore_from_checkpoint') as mock_restore, \
             patch.object(builder, 'execute') as mock_execute:
            
            # Mock state restoration
            restored_state = create_initial_state("test-workflow")
            restored_state["metadata"]["status"] = "paused"
            mock_restore.return_value = restored_state
            
            final_state = create_initial_state("test-workflow")
            final_state["metadata"]["status"] = "completed"
            mock_execute.return_value = final_state
            
            result = await builder.resume_workflow("test-workflow")
            assert result["metadata"]["status"] == "completed"
            assert "resumed_at" in restored_state["metadata"]
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(self):
        """测试取消工作流"""
        builder = LangGraphWorkflowBuilder()
        
        with patch.object(builder.checkpoint_manager, 'get_latest_checkpoint') as mock_get, \
             patch.object(builder.checkpoint_manager, 'create_checkpoint') as mock_create:
            
            mock_checkpoint = Mock()
            mock_checkpoint.state = create_initial_state("test-workflow")
            mock_get.return_value = mock_checkpoint
            mock_create.return_value = None
            
            result = await builder.cancel_workflow("test-workflow")
            assert result == True
            assert mock_checkpoint.state["metadata"]["status"] == "cancelled"
            assert "cancelled_at" in mock_checkpoint.state["metadata"]


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """端到端工作流测试"""
        builder = create_simple_workflow()
        initial_state = create_initial_state("integration-test")
        
        # Mock external dependencies
        with patch('ai.langgraph.state_graph.PostgresSaver') as mock_saver, \
             patch.object(builder.checkpoint_manager, 'create_checkpoint') as mock_checkpoint:
            
            mock_saver.from_conn_string.return_value = None
            mock_checkpoint.return_value = None
            
            # Mock compiled graph execution
            compiled_graph = builder.compile()
            
            # Create expected final state
            expected_state = initial_state.copy()
            expected_state["metadata"]["status"] = "completed"
            expected_state["metadata"]["completed_at"] = datetime.now().isoformat()
            expected_state["messages"] = [
                {"role": "system", "content": "工作流开始执行"},
                {"role": "assistant", "content": "正在处理任务..."},
                {"role": "system", "content": "工作流执行完成"}
            ]
            expected_state["context"]["processed"] = True
            
            compiled_graph.ainvoke = AsyncMock(return_value=expected_state)
            
            result = await builder.execute(initial_state)
            
            assert result["metadata"]["status"] == "completed"
            assert "completed_at" in result["metadata"]
            assert len(result["messages"]) == 3
            assert result["context"]["processed"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])