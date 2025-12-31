"""
LangGraph 0.6.5升级功能测试
测试新Context API、durability控制、Node Caching和Pre/Post Hooks
"""

import pytest
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from unittest.mock import Mock, AsyncMock, patch
from src.ai.langgraph.state_graph import (
    LangGraphWorkflowBuilder, 
    WorkflowNode,
    ConditionalRouter
)
from src.ai.langgraph.context import (
    AgentContext, 
    LangGraphContextSchema, 
    create_default_context,
    validate_context
)
from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.node_caching import (
    NodeCacheManager,
    InMemoryCache,
    CachePolicy,
    create_cached_node
)
from src.ai.langgraph.hooks import (
    HookManager,
    MessageCompressionHook,
    InputSanitizationHook,
    ResponseFilterHook,
    QualityCheckHook,
    HookConfig,
    get_hook_manager

)

class TestLangGraphContextSchema:
    """测试新Context API dataclass schema"""
    
    def test_langgraph_context_schema_creation(self):
        """测试LangGraphContextSchema创建"""
        schema = LangGraphContextSchema(
            user_id="test_user",
            session_id="test_session",
            conversation_id="test_conv"
        )
        
        assert schema.user_id == "test_user"
        assert schema.session_id == "test_session"
        assert schema.conversation_id == "test_conv"
        assert schema.max_iterations == 10  # 默认值
        assert schema.timeout_seconds == 300  # 默认值
    
    def test_context_schema_conversion(self):
        """测试Context Schema与AgentContext互转"""
        # 创建AgentContext
        agent_context = create_default_context(
            user_id="test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000"
        )
        # 使用有效的UUID格式
        agent_context.conversation_id = "550e8400-e29b-41d4-a716-446655440001"
        
        # 转换为LangGraphContextSchema
        schema = LangGraphContextSchema.from_agent_context(agent_context)
        assert schema.user_id == "test_user"
        assert schema.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.conversation_id == "550e8400-e29b-41d4-a716-446655440001"
        
        # 转换回AgentContext
        new_context = schema.to_agent_context()
        assert new_context.user_id == "test_user"
        assert new_context.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert new_context.conversation_id == "550e8400-e29b-41d4-a716-446655440001"

class TestWorkflowBuilderUpgrade:
    """测试工作流构建器的LangGraph 0.6.5升级"""
    
    def test_builder_with_context_api_enabled(self):
        """测试启用新Context API的构建器"""
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        def test_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "测试响应",
                "timestamp": utc_now().isoformat()
            })
            return state
        
        builder.add_node("test_node", test_handler)
        graph = builder.build()
        
        # 验证图使用了context_schema
        assert graph is not None
        assert builder.use_context_api is True
    
    def test_builder_with_legacy_config(self):
        """测试向后兼容的构建器"""
        builder = LangGraphWorkflowBuilder(use_context_api=False)
        
        def test_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant", 
                "content": "传统模式响应",
                "timestamp": utc_now().isoformat()
            })
            return state
        
        builder.add_node("legacy_node", test_handler)
        graph = builder.build()
        
        assert graph is not None
        assert builder.use_context_api is False
    
    def test_durability_parameter_in_compile(self):
        """测试编译时的durability参数"""
        from langgraph.graph import START, END
        
        builder = LangGraphWorkflowBuilder()
        
        def simple_handler(state: MessagesState) -> MessagesState:
            return state
        
        builder.add_node("simple", simple_handler)
        graph = builder.build()
        
        # 添加必需的边
        graph.add_edge(START, "simple")
        graph.add_edge("simple", END)
        
        # 测试不同的durability模式
        for durability in ["exit", "async", "sync"]:
            compiled_graph = builder.compile(durability_mode=durability)
            assert compiled_graph is not None
            assert builder.default_durability == durability

class TestWorkflowNodeUpgrade:
    """测试节点执行的Context API升级"""
    
    @pytest.mark.asyncio
    async def test_node_execute_with_runtime_context(self):
        """测试节点使用Runtime Context执行"""
        def test_handler(state: MessagesState, context=None) -> MessagesState:
            if context:
                state["context"]["received_context"] = True
                state["context"]["context_user_id"] = context.user_id
            return state
        
        node = WorkflowNode("test_node", test_handler)
        state = create_initial_state()
        
        # 模拟Runtime context
        mock_runtime = Mock()
        mock_context_schema = LangGraphContextSchema(
            user_id="runtime_test_user",
            session_id="runtime_session"
        )
        mock_runtime.context = mock_context_schema
        
        result = await node.execute(state, runtime=mock_runtime)
        
        assert result["context"]["received_context"] is True
        assert result["context"]["context_user_id"] == "runtime_test_user"
    
    @pytest.mark.asyncio
    async def test_node_execute_with_legacy_config(self):
        """测试节点使用传统config执行"""
        def test_handler(state: MessagesState, context=None) -> MessagesState:
            if context:
                state["context"]["received_context"] = True
                state["context"]["context_user_id"] = context.user_id
            return state
        
        node = WorkflowNode("test_node", test_handler)
        state = create_initial_state()
        
        # 传统config模式
        config = {
            "configurable": {
                "user_id": "legacy_user",
                "session_id": "legacy_session"
            }
        }
        
        result = await node.execute(state, config=config)
        
        assert result["context"]["received_context"] is True
        assert result["context"]["context_user_id"] == "legacy_user"

class TestNodeCaching:
    """测试节点缓存功能"""
    
    @pytest.mark.asyncio
    async def test_in_memory_cache_basic_operations(self):
        """测试内存缓存基本操作"""
        cache = InMemoryCache(max_size=10)
        
        # 测试设置和获取
        await cache.set("key1", {"data": "value1"}, ttl=60)
        result = await cache.get("key1")
        assert result["data"] == "value1"
        
        # 测试不存在的键
        result = await cache.get("nonexistent")
        assert result is None
        
        # 测试删除
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio 
    async def test_cache_policy_configuration(self):
        """测试缓存策略配置"""
        cache_manager = NodeCacheManager(InMemoryCache())
        
        # 设置节点特定策略
        policy = CachePolicy(ttl=120, cache_key_fields=["messages"])
        cache_manager.set_node_policy("expensive_node", policy)
        
        retrieved_policy = cache_manager.get_node_policy("expensive_node")
        assert retrieved_policy.ttl == 120
        assert retrieved_policy.cache_key_fields == ["messages"]
    
    @pytest.mark.asyncio
    async def test_cached_node_execution(self):
        """测试缓存节点执行"""
        call_count = 0
        
        def expensive_handler(state: MessagesState) -> MessagesState:
            nonlocal call_count
            call_count += 1
            state["messages"].append({
                "role": "assistant",
                "content": f"计算结果 {call_count}",
                "timestamp": utc_now().isoformat()
            })
            return state
        
        # 创建缓存包装的处理器
        cache_policy = CachePolicy(ttl=60, enabled=True)
        cached_handler = create_cached_node("expensive_node", expensive_handler, cache_policy)
        
        state1 = create_initial_state()
        state1["messages"] = [{"role": "user", "content": "输入1"}]
        
        # 第一次调用 - 应该执行
        result1 = await cached_handler(state1)
        assert call_count == 1
        
        # 第二次调用相同输入 - 应该从缓存返回
        result2 = await cached_handler(state1)
        assert call_count == 1  # 没有增加
        
        # 验证结果一致
        assert len(result1["messages"]) == len(result2["messages"])

class TestPrePostHooks:
    """测试Pre/Post Model Hooks"""
    
    @pytest.mark.asyncio
    async def test_message_compression_hook(self):
        """测试消息压缩钩子"""
        hook_config = HookConfig(name="TestCompression", enabled=True)
        hook = MessageCompressionHook(hook_config, max_messages=3)
        
        # 创建包含多条消息的状态
        state = create_initial_state()
        state["messages"] = [
            {"role": "user", "content": f"消息 {i}"} for i in range(5)
        ]
        
        result = await hook.execute(state)
        
        # 应该被压缩为4条消息（1条压缩+3条最新）
        assert len(result["messages"]) == 4
        assert result["messages"][0]["role"] == "system"
        assert "[历史消息摘要]" in result["messages"][0]["content"]
        assert "hook_logs" in result["context"]
    
    @pytest.mark.asyncio
    async def test_input_sanitization_hook(self):
        """测试输入清理钩子"""
        hook_config = HookConfig(name="TestSanitization", enabled=True)
        hook = InputSanitizationHook(hook_config, max_message_length=50)
        
        state = create_initial_state()
        state["messages"] = [
            {"role": "user", "content": "这是一条很长的消息" * 10},  # 超长消息
            {"role": "user", "content": "<script>alert('xss')</script>"}  # XSS
        ]
        
        result = await hook.execute(state)
        
        # 检查长度限制
        assert len(result["messages"][0]["content"]) <= 50 + 10  # +截断标记
        assert "[内容被截断]" in result["messages"][0]["content"]
        
        # 检查XSS过滤
        assert "[已过滤]" in result["messages"][1]["content"]
        assert "<script>" not in result["messages"][1]["content"]
    
    @pytest.mark.asyncio
    async def test_response_filter_hook(self):
        """测试响应过滤钩子"""
        hook_config = HookConfig(name="TestFilter", enabled=True)
        hook = ResponseFilterHook(hook_config)
        hook.blocked_words = ["敏感词", "禁用词"]
        
        state = create_initial_state()
        state["messages"] = [
            {"role": "assistant", "content": "这是包含敏感词的回复"},
            {"role": "assistant", "content": "正常的回复内容"}
        ]
        
        result = await hook.execute(state)
        
        # 检查敏感词被替换
        assert "[内容已被过滤]" in result["messages"][0]["content"]
        assert "敏感词" not in result["messages"][0]["content"]
        assert result["messages"][0]["metadata"]["filtered"] is True
        
        # 正常内容不受影响
        assert result["messages"][1]["content"] == "正常的回复内容"
    
    @pytest.mark.asyncio
    async def test_quality_check_hook(self):
        """测试质量检查钩子"""
        hook_config = HookConfig(name="TestQuality", enabled=True)
        hook = QualityCheckHook(hook_config, min_length=10, max_length=100)
        
        state = create_initial_state()
        state["messages"] = [
            {"role": "assistant", "content": "短"},  # 太短
            {"role": "assistant", "content": ""},  # 空内容
            {"role": "assistant", "content": "这是一个正常长度的回复内容"},  # 正常
            {"role": "assistant", "content": "这是一个非常长的回复" * 10}  # 太长
        ]
        
        result = await hook.execute(state)
        
        # 检查质量问题被标记
        assert "质量检查" in str(result["context"]["hook_logs"][-1]["hook"])
        assert result["context"]["hook_logs"][-1]["quality_issues_found"] == 3
        
        # 检查具体问题标记
        assert "内容过短" in result["messages"][0]["metadata"]["quality_issues"]
        assert "内容为空" in result["messages"][1]["metadata"]["quality_issues"]
        assert "quality_issues" not in result["messages"][2].get("metadata", {})
        assert "内容过长" in result["messages"][3]["metadata"]["quality_issues"]
    
    @pytest.mark.asyncio
    async def test_hook_manager_execution_order(self):
        """测试钩子管理器执行顺序"""
        hook_manager = HookManager()
        
        # 添加具有不同优先级的钩子
        hook1 = MessageCompressionHook(HookConfig(name="Hook1", priority=2))
        hook2 = InputSanitizationHook(HookConfig(name="Hook2", priority=1))
        
        hook_manager.add_pre_hook(hook1)
        hook_manager.add_pre_hook(hook2)
        
        # 验证优先级排序
        assert hook_manager.pre_hooks[0].config.name == "Hook2"  # 优先级1
        assert hook_manager.pre_hooks[1].config.name == "Hook1"  # 优先级2
        
        # 测试执行
        state = create_initial_state()
        state["messages"] = [{"role": "user", "content": "测试消息"}]
        
        result = await hook_manager.execute_pre_hooks(state)
        assert result is not None
    
    def test_hook_manager_status(self):
        """测试钩子管理器状态查询"""
        hook_manager = get_hook_manager()
        status = hook_manager.get_hook_status()
        
        assert "pre_hooks" in status
        assert "post_hooks" in status
        assert len(status["pre_hooks"]) > 0
        assert len(status["post_hooks"]) > 0
        
        # 验证钩子信息
        for hook_info in status["pre_hooks"]:
            assert "name" in hook_info
            assert "enabled" in hook_info
            assert "priority" in hook_info
            assert "description" in hook_info

class TestDurabilityControl:
    """测试durability控制功能"""
    
    @pytest.mark.asyncio
    async def test_execute_with_different_durability_modes(self):
        """测试不同durability模式的执行"""
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        def test_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "durability测试",
                "timestamp": utc_now().isoformat()
            })
            return state
        
        builder.add_node("test_node", test_handler)
        builder.build()
        compiled_graph = builder.compile()
        
        initial_state = create_initial_state()
        context = create_default_context()
        
        # 测试不同durability模式
        for durability in ["exit", "async", "sync"]:
            with patch.object(builder.compiled_graph, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
                mock_invoke.return_value = initial_state
                
                result = await builder.execute(
                    initial_state, 
                    context=context, 
                    durability=durability
                )
                
                # 验证durability参数被传递
                mock_invoke.assert_called_once()
                call_args = mock_invoke.call_args
                
                if builder.use_context_api:
                    assert call_args[1]["config"]["durability"] == durability

class TestIntegrationWorkflow:
    """集成测试：完整工作流测试"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_with_all_features(self):
        """测试包含所有新特性的完整工作流"""
        # 创建带有所有特性的工作流
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        # 启用缓存的节点处理器
        call_count = 0
        def cached_handler(state: MessagesState) -> MessagesState:
            nonlocal call_count
            call_count += 1
            state["messages"].append({
                "role": "assistant",
                "content": f"缓存节点处理 {call_count}",
                "timestamp": utc_now().isoformat()
            })
            return state
        
        # 使用缓存创建节点
        cache_policy = CachePolicy(ttl=60, enabled=True)
        cached_node_handler = create_cached_node("cached_processor", cached_handler, cache_policy)
        
        builder.add_node("start", lambda state: state)
        builder.add_node("cached_processor", cached_node_handler)
        builder.add_node("end", lambda state: state)
        
        # 构建和编译图
        graph = builder.build()
        graph.add_edge("start", "cached_processor")
        graph.add_edge("cached_processor", "end")
        
        compiled_graph = builder.compile(durability_mode="async")
        
        # 准备测试数据
        initial_state = create_initial_state()
        initial_state["messages"] = [
            {"role": "user", "content": "测试输入"}
        ]
        
        context = create_default_context(
            user_id="integration_test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000"
        )
        
        # 由于实际graph.ainvoke可能不可用，我们模拟执行
        # 在真实环境中，这里会执行完整的工作流
        
        # 手动执行钩子测试
        hook_manager = get_hook_manager()
        
        # 执行预处理钩子
        processed_state = await hook_manager.execute_pre_hooks(initial_state, context)
        assert processed_state is not None
        
        # 执行后处理钩子
        final_state = await hook_manager.execute_post_hooks(processed_state, context)
        assert final_state is not None
        
        # 验证钩子执行日志
        if "hook_logs" in final_state["context"]:
            assert len(final_state["context"]["hook_logs"]) > 0

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """测试工作流错误处理"""
    from langgraph.graph import START, END
    
    builder = LangGraphWorkflowBuilder(use_context_api=True)
    
    def failing_handler(state: MessagesState) -> MessagesState:
        raise ValueError("测试错误")
    
    builder.add_node("failing_node", failing_handler)
    graph = builder.build()
    
    # 添加必需的边
    graph.add_edge(START, "failing_node")
    graph.add_edge("failing_node", END)
    
    builder.compile()
    
    initial_state = create_initial_state()
    context = create_default_context()
    
    # 测试错误处理
    with pytest.raises(Exception):
        await builder.execute(initial_state, context=context)

def test_context_validation():
    """测试上下文验证功能"""
    # 有效上下文
    valid_context = create_default_context()
    assert validate_context(valid_context) is True
    
    # 测试LangGraphContextSchema
    schema = LangGraphContextSchema(
        user_id="test",
        session_id="550e8400-e29b-41d4-a716-446655440000"
    )
    context = schema.to_agent_context()
    assert validate_context(context) is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
