"""
Context API单元测试
测试LangGraph v0.6.5 Context API的类型安全和功能
包含新Context API、durability控制、Node Caching和Hooks测试
"""

import pytest
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import RunnableConfig
from src.ai.langgraph.context import (
    AgentContext,
    create_default_context,
    validate_context,
    LangGraphContextSchema
)
from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import (
    WorkflowNode,
    LangGraphWorkflowBuilder,
    create_simple_workflow,
    create_conditional_workflow
)
from src.ai.langgraph.node_caching import (
    NodeCacheManager,
    CachePolicy,
    create_cached_node,
    InMemoryCache
)
from src.ai.langgraph.hooks import (
    HookManager,
    MessageCompressionHook,
    ResponseFilterHook,
    with_hooks

)

class TestAgentContext:
    """AgentContext类型测试"""
    
    def test_create_context(self):
        """测试创建上下文"""
        from src.ai.langgraph.context import SessionContext
        session_context = SessionContext(session_id="550e8400-e29b-41d4-a716-446655440000")
        
        context = AgentContext(
            user_id="test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            conversation_id="550e8400-e29b-41d4-a716-446655440001",
            agent_id="550e8400-e29b-41d4-a716-446655440002",
            session_context=session_context
        )
        
        assert context.user_id == "test_user"
        assert context.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert context.conversation_id == "550e8400-e29b-41d4-a716-446655440001"
        assert context.agent_id == "550e8400-e29b-41d4-a716-446655440002"
        assert context.step_count == 0
        assert context.status == "running"
    
    def test_context_to_dict(self):
        """测试上下文转换为字典"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440001"
        context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id),
            metadata={"custom": "data"}
        )
        
        data = context.to_dict()
        assert data["user_id"] == "test_user"
        assert data["session_id"] == session_id
        assert data["metadata"] == {"custom": "data"}
    
    def test_context_from_dict(self):
        """测试从字典创建上下文"""
        session_id = "550e8400-e29b-41d4-a716-446655440002"
        data = {
            "user_id": "test_user",
            "session_id": session_id,
            "step_count": 5,
            "status": "paused"
        }
        
        context = AgentContext.from_dict(data)
        assert context.user_id == "test_user"
        assert context.session_id == session_id
        assert context.step_count == 5
        assert context.status == "paused"
    
    def test_update_step(self):
        """测试更新步骤信息"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440003"
        context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        context.update_step("node1")
        assert context.step_count == 1
        assert context.current_node == "node1"
        assert context.last_updated is not None
        
        context.update_step("node2")
        assert context.step_count == 2
        assert context.current_node == "node2"
    
    def test_is_max_iterations_reached(self):
        """测试最大迭代次数检查"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440004"
        context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id),
            max_iterations=3
        )
        
        assert not context.is_max_iterations_reached()
        
        context.step_count = 3
        assert context.is_max_iterations_reached()
        
        context.step_count = 4
        assert context.is_max_iterations_reached()
    
    def test_validate_context(self):
        """测试上下文验证"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440005"
        
        # 有效上下文
        valid_context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        assert validate_context(valid_context)
        
        # 无效上下文 - 缺少user_id  
        with pytest.raises(ValueError):
            AgentContext(
                user_id="",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id)
            )
        
        # 无效上下文 - 无效状态（这会在创建时就失败）
        with pytest.raises(ValueError):
            AgentContext(
                user_id="test_user",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id),
                status="invalid_status"
            )
    
    def test_create_default_context(self):
        """测试创建默认上下文"""
        context = create_default_context()
        assert context.user_id == "default_user"
        assert context.session_id == "550e8400-e29b-41d4-a716-446655440000"
        
        context = create_default_context(
            user_id="custom_user",
            workflow_id="550e8400-e29b-41d4-a716-446655440006"
        )
        assert context.user_id == "custom_user"
        assert context.workflow_id == "550e8400-e29b-41d4-a716-446655440006"

class TestWorkflowNodeWithContext:
    """测试工作流节点的Context支持"""
    
    @pytest.mark.asyncio
    async def test_node_with_context(self):
        """测试节点接收上下文"""
        received_context = None
        
        async def test_handler(state: MessagesState, context: AgentContext = None) -> MessagesState:
            nonlocal received_context
            received_context = context
            state["messages"].append({
                "role": "assistant",
                "content": f"User: {context.user_id if context else 'unknown'}"
            })
            return state
        
        node = WorkflowNode("test_node", test_handler)
        state = create_initial_state()
        
        # 创建配置包含上下文信息（使用有效的UUID格式）
        session_id = "550e8400-e29b-41d4-a716-446655440007"
        config: RunnableConfig = {
            "configurable": {
                "user_id": "test_user_123",
                "session_id": session_id
            }
        }
        
        result = await node.execute(state, config)
        
        assert received_context is not None
        assert received_context.user_id == "test_user_123"
        assert received_context.session_id == session_id
        assert len(result["messages"]) > 0
    
    @pytest.mark.asyncio
    async def test_node_without_context(self):
        """测试节点不需要上下文时的兼容性"""
        def simple_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "Simple response"
            })
            return state
        
        node = WorkflowNode("simple_node", simple_handler)
        state = create_initial_state()
        
        result = await node.execute(state)
        
        assert len(result["messages"]) > 0
        assert result["messages"][-1]["content"] == "Simple response"

class TestLangGraphWorkflowBuilderWithContext:
    """测试工作流构建器的Context支持"""
    
    @pytest.mark.asyncio
    async def test_workflow_with_context(self):
        """测试工作流执行时传递上下文"""
        builder = LangGraphWorkflowBuilder()
        
        context_info = {}
        
        def capture_context(state: MessagesState, context: AgentContext = None) -> MessagesState:
            if context:
                context_info["user_id"] = context.user_id
                context_info["session_id"] = context.session_id
            state["messages"].append({
                "role": "assistant",
                "content": f"Processed by {context.user_id if context else 'unknown'}"
            })
            return state
        
        builder.add_node("process", capture_context)
        graph = builder.build()
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        
        # 使用上下文执行
        initial_state = create_initial_state()
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440008"
        context = AgentContext(
            user_id="workflow_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        result = await builder.execute(initial_state, context)
        
        assert context_info["user_id"] == "workflow_user"
        assert context_info["session_id"] == session_id
        assert result["metadata"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """测试向后兼容性 - 使用config传递上下文"""
        builder = LangGraphWorkflowBuilder()
        
        context_info = {}
        
        def capture_context(state: MessagesState, context: AgentContext = None) -> MessagesState:
            if context:
                context_info["user_id"] = context.user_id
            state["messages"].append({"role": "assistant", "content": "Done"})
            return state
        
        builder.add_node("process", capture_context)
        graph = builder.build()
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        
        # 使用旧方式传递上下文（通过config）- 使用有效的UUID格式
        initial_state = create_initial_state()
        session_id = "550e8400-e29b-41d4-a716-446655440009"
        config = {
            "configurable": {
                "user_id": "legacy_user",
                "session_id": session_id
            }
        }
        
        result = await builder.execute(initial_state, config=config)
        
        assert context_info["user_id"] == "legacy_user"
        assert result["metadata"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_context_priority(self):
        """测试上下文优先级 - 直接传递的context优先于config"""
        builder = LangGraphWorkflowBuilder()
        
        context_info = {}
        
        def capture_context(state: MessagesState, context: AgentContext = None) -> MessagesState:
            if context:
                context_info["user_id"] = context.user_id
            state["messages"].append({"role": "assistant", "content": "Done"})
            return state
        
        builder.add_node("process", capture_context)
        graph = builder.build()
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        
        # 同时提供context和config
        initial_state = create_initial_state()
        from src.ai.langgraph.context import SessionContext
        session_id_direct = "550e8400-e29b-41d4-a716-446655440010"
        session_id_config = "550e8400-e29b-41d4-a716-446655440011"
        context = AgentContext(
            user_id="direct_user",
            session_id=session_id_direct,
            session_context=SessionContext(session_id=session_id_direct)
        )
        config = {
            "configurable": {
                "user_id": "config_user",
                "session_id": session_id_config
            }
        }
        
        result = await builder.execute(initial_state, context, config)
        
        # 直接传递的context应该优先
        assert context_info["user_id"] == "direct_user"
        assert result["metadata"]["status"] == "completed"

class TestTypeSafety:
    """类型安全测试"""
    
    def test_context_type_hints(self):
        """测试上下文类型提示"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440012"
        context = AgentContext(
            user_id="test",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        # 类型应该正确
        assert isinstance(context.user_id, str)
        assert isinstance(context.session_id, str)
        assert isinstance(context.step_count, int)
        assert isinstance(context.max_iterations, int)
        assert isinstance(context.enable_checkpoints, bool)
        assert isinstance(context.metadata, dict)
    
    def test_optional_fields(self):
        """测试可选字段"""
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440013"
        context = AgentContext(
            user_id="test",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        # 可选字段应该有默认值或None
        assert context.conversation_id is None
        assert context.agent_id is None
        assert context.workflow_id is None
        assert context.thread_id is None
        assert context.current_node is None
        assert context.last_updated is None

class TestLangGraphContextSchema:
    """测试LangGraph 0.6.5新Context Schema"""
    
    def test_create_context_schema(self):
        """测试创建Context Schema"""
        schema = LangGraphContextSchema(
            user_id="test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            conversation_id="conv123"
        )
        
        assert schema.user_id == "test_user"
        assert schema.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.conversation_id == "conv123"
        assert schema.max_iterations == 10  # 默认值
        assert schema.enable_checkpoints is True
    
    def test_schema_to_agent_context(self):
        """测试Schema转换为AgentContext"""
        schema = LangGraphContextSchema(
            user_id="test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            workflow_id="workflow123",
            max_iterations=15
        )
        
        context = schema.to_agent_context()
        
        assert isinstance(context, AgentContext)
        assert context.user_id == "test_user"
        assert context.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert context.max_iterations == 15
    
    def test_schema_from_agent_context(self):
        """测试从AgentContext创建Schema"""
        from src.ai.langgraph.context import SessionContext
        
        session_context = SessionContext(session_id="550e8400-e29b-41d4-a716-446655440000")
        context = AgentContext(
            user_id="test_user",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            session_context=session_context,
            timeout_seconds=600
        )
        
        schema = LangGraphContextSchema.from_agent_context(context)
        
        assert schema.user_id == "test_user"
        assert schema.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.timeout_seconds == 600

class TestNewContextAPI:
    """测试LangGraph 0.6.5新Context API"""
    
    @pytest.mark.asyncio
    async def test_new_context_api_workflow(self):
        """测试新Context API工作流"""
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        received_context = None
        
        def test_handler(state: MessagesState, context=None) -> MessagesState:
            nonlocal received_context
            received_context = context
            state["messages"].append({
                "role": "assistant",
                "content": "New Context API test"
            })
            return state
        
        builder.add_node("test", test_handler)
        graph = builder.build()
        graph.add_edge(START, "test")
        graph.add_edge("test", END)
        
        # 使用新Context API执行
        initial_state = create_initial_state()
        from src.ai.langgraph.context import SessionContext
        session_id = "550e8400-e29b-41d4-a716-446655440014"
        context = AgentContext(
            user_id="new_api_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        result = await builder.execute(initial_state, context, durability="async")
        
        assert result["metadata"]["status"] == "completed"
        assert len(result["messages"]) > 0
    
    @pytest.mark.asyncio
    async def test_durability_modes(self):
        """测试durability模式"""
        builder = LangGraphWorkflowBuilder()
        
        def simple_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "Durability test"
            })
            return state
        
        builder.add_node("simple", simple_handler)
        graph = builder.build()
        graph.add_edge(START, "simple")
        graph.add_edge("simple", END)
        
        initial_state = create_initial_state()
        
        # 测试不同durability模式
        for durability in ["exit", "async", "sync"]:
            result = await builder.execute(initial_state, durability=durability)
            assert result["metadata"]["status"] == "completed"

class TestNodeCaching:
    """测试Node Caching功能"""
    
    @pytest.mark.asyncio
    async def test_cache_manager(self):
        """测试缓存管理器"""
        from src.ai.langgraph.node_caching import NodeCacheManager, InMemoryCache
        
        # 创建缓存管理器，使用内存缓存后端
        backend = InMemoryCache()
        cache_manager = NodeCacheManager(backend)
        
        state = create_initial_state()
        state["messages"].append({"role": "user", "content": "test"})
        
        result_state = state.copy()
        result_state["messages"].append({"role": "assistant", "content": "cached result"})
        
        # 第一次调用应该是cache miss
        cached = await cache_manager.get_cached_result("test_node", state)
        assert cached is None
        
        # 缓存结果
        await cache_manager.cache_result("test_node", state, result_state)
        
        # 第二次调用应该是cache hit
        cached = await cache_manager.get_cached_result("test_node", state)
        assert cached is not None
        assert len(cached["messages"]) == 2
        
        # 验证缓存内容正确
        assert cached["messages"][-1]["content"] == "cached result"
    
    @pytest.mark.asyncio
    async def test_cached_node_decorator(self):
        """测试cached_node装饰器"""
        from src.ai.langgraph.node_caching import create_cached_node, CachePolicy
        
        call_count = 0
        
        async def expensive_handler(state: MessagesState, context=None) -> MessagesState:
            nonlocal call_count
            call_count += 1
            state["messages"].append({
                "role": "assistant",
                "content": f"Expensive computation {call_count}"
            })
            return state
        
        # 创建缓存节点
        cache_policy = CachePolicy(ttl=60)
        expensive_node = create_cached_node("expensive_node", expensive_handler, cache_policy)
        
        state = create_initial_state()
        
        # 第一次调用
        result1 = await expensive_node(state)
        assert call_count == 1
        assert "Expensive computation 1" in result1["messages"][-1]["content"]
        
        # 第二次调用应该使用缓存
        result2 = await expensive_node(state)
        assert call_count == 1  # 应该没有增加
        assert result2 == result1
    
    def test_cache_presets(self):
        """测试缓存预设"""
        from src.ai.langgraph.node_caching import CachePolicy
        
        # 测试快速缓存策略
        fast = CachePolicy(ttl=60, max_size=100)
        assert fast.ttl == 60
        assert fast.max_size == 100
        
        # 测试标准缓存策略
        standard = CachePolicy(ttl=300, max_size=500)
        assert standard.ttl == 300
        assert standard.max_size == 500
        
        # 测试长期缓存策略
        long_term = CachePolicy(ttl=3600, max_size=1000)
        assert long_term.ttl == 3600
        assert long_term.max_size == 1000
        
        # 测试上下文感知缓存策略
        context_aware = CachePolicy(ttl=300, cache_key_fields=["user_id", "session_id"])
        assert context_aware.cache_key_fields == ["user_id", "session_id"]
        assert context_aware.enabled is True

class TestHooks:
    """测试Pre/Post Model Hooks"""
    
    @pytest.mark.asyncio
    async def test_message_compression_hook(self):
        """测试消息压缩Hook"""
        hook = MessageCompressionHook(max_messages=3)
        
        # 创建包含多条消息的状态
        state = create_initial_state()
        for i in range(10):
            state["messages"].append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}"
            })
        
        assert len(state["messages"]) == 10
        
        # 执行压缩Hook
        compressed_state = await hook.execute(state)
        
        # 消息应该被压缩（包含压缩通知）
        assert len(compressed_state["messages"]) <= 4  # 3条消息 + 1条压缩通知
        assert "hook_logs" in compressed_state["context"]
        # 验证压缩日志记录
        hook_logs = compressed_state["context"]["hook_logs"]
        assert len(hook_logs) > 0
        assert any(log["action"] == "compressed_messages" for log in hook_logs)
    
    @pytest.mark.asyncio
    async def test_guardrails_hook(self):
        """测试Guardrails Hook"""
        from src.ai.langgraph.hooks import ResponseFilterHook, HookConfig
        
        # 创建响应过滤Hook，使用自定义配置
        config = HookConfig(enabled=True, name="guardrails")
        hook = ResponseFilterHook(config)
        # 设置自定义的禁用词列表
        hook.blocked_words = ["禁止词"]
        hook.replacement_text = "抱歉，内容已被过滤"
        
        state = create_initial_state()
        state["messages"].append({
            "role": "assistant",
            "content": "这里包含禁止词内容"
        })
        
        # 执行过滤检查
        filtered_state = await hook.execute(state)
        
        # 内容应该被替换
        assert "抱歉" in filtered_state["messages"][-1]["content"]
        # 检查过滤标记
        assert filtered_state["messages"][-1]["metadata"]["filtered"] is True
    
    @pytest.mark.asyncio
    async def test_hook_manager(self):
        """测试Hook管理器"""
        manager = HookManager()
        compression_hook = MessageCompressionHook(max_messages=5)
        manager.add_pre_hook(compression_hook)
        
        # 创建包含多条消息的状态
        state = create_initial_state()
        for i in range(10):
            state["messages"].append({
                "role": "user",
                "content": f"Message {i}"
            })
        
        # 执行Pre Hooks
        processed_state = await manager.execute_pre_hooks(state)
        
        # 消息应该被压缩
        assert len(processed_state["messages"]) <= 6  # 5条消息 + 1条压缩通知
    
    @pytest.mark.asyncio
    async def test_with_hooks_decorator(self):
        """测试with_hooks装饰器"""
        from src.ai.langgraph.hooks import with_hooks, set_hook_manager
        
        manager = HookManager()
        compression_hook = MessageCompressionHook(max_messages=3)
        manager.add_pre_hook(compression_hook)
        
        # 设置全局钩子管理器
        set_hook_manager(manager)
        
        @with_hooks()
        async def ai_node(state: MessagesState, context=None) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "AI response"
            })
            return state
        
        # 创建包含多条消息的状态
        state = create_initial_state()
        for i in range(8):
            state["messages"].append({
                "role": "user",
                "content": f"Message {i}"
            })
        
        result = await ai_node(state)
        
        # Pre Hook应该已经压缩了消息
        assert len(result["messages"]) <= 5  # 被压缩的消息 + AI响应

class TestIntegration:
    """集成测试 - 测试所有新特性一起工作"""
    
    @pytest.mark.asyncio
    async def test_full_integration(self):
        """测试所有LangGraph 0.6.5特性的集成"""
        from src.ai.langgraph.node_caching import create_cached_node, CachePolicy
        from src.ai.langgraph.hooks import with_hooks
        from src.ai.langgraph.context import SessionContext
        
        # 创建带缓存和Hooks的工作流
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        async def processing_handler(state: MessagesState, context=None) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": "Processed with caching and hooks"
            })
            return state
        
        # 创建缓存节点
        cache_policy = CachePolicy(ttl=300)
        cached_processing_node = create_cached_node("processing_node", processing_handler, cache_policy)
        
        @with_hooks()
        async def processing_node(state: MessagesState, context=None) -> MessagesState:
            return await cached_processing_node(state, context)
        
        builder.add_node("process", processing_node)
        graph = builder.build()
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        
        # 使用新Context API和durability控制
        initial_state = create_initial_state()
        session_id = "550e8400-e29b-41d4-a716-446655440015"
        context = AgentContext(
            user_id="integration_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id)
        )
        
        result = await builder.execute(
            initial_state, 
            context, 
            durability="async"
        )
        
        assert result["metadata"]["status"] == "completed"
        assert len(result["messages"]) > 0
