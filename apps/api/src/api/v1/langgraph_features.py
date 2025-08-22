"""
LangGraph 0.6.5新特性演示API端点
提供Context API、durability控制、Node Caching和Pre/Post Hooks演示
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime
import asyncio
import json

from src.ai.langgraph.state_graph import (
    LangGraphWorkflowBuilder,
    create_simple_workflow,
    create_conditional_workflow
)
from src.ai.langgraph.context import (
    LangGraphContextSchema,
    create_default_context,
    AgentContext
)
from src.ai.langgraph.state import create_initial_state, MessagesState
from src.ai.langgraph.node_caching import (
    NodeCacheManager,
    InMemoryCache,
    CachePolicy,
    create_cached_node,
    get_cache_manager
)
from src.ai.langgraph.hooks import (
    get_hook_manager,
    HookConfig,
    MessageCompressionHook,
    InputSanitizationHook,
    ResponseFilterHook,
    QualityCheckHook
)

router = APIRouter(prefix="/langgraph", tags=["LangGraph 0.6.5特性"])


# 请求/响应模型
class ContextAPIRequest(BaseModel):
    user_id: str = "demo_user"
    session_id: str = "550e8400-e29b-41d4-a716-446655440000"
    conversation_id: Optional[str] = None
    message: str = "测试新Context API"
    use_new_api: bool = True


class DurabilityRequest(BaseModel):
    message: str = "测试durability控制"
    durability_mode: Literal["exit", "async", "sync"] = "async"


class CachingRequest(BaseModel):
    message: str = "计算密集型任务"
    enable_cache: bool = True
    cache_ttl: int = 300


class HooksRequest(BaseModel):
    messages: List[Dict[str, Any]]
    enable_pre_hooks: bool = True
    enable_post_hooks: bool = True


class WorkflowResponse(BaseModel):
    success: bool
    execution_time_ms: float
    result: Dict[str, Any]
    metadata: Dict[str, Any]


# Context API演示端点
@router.post("/context-api/demo", response_model=WorkflowResponse)
async def demo_context_api(request: ContextAPIRequest):
    """演示新Context API vs 旧config模式"""
    start_time = datetime.now()
    
    try:
        # 创建工作流构建器
        builder = LangGraphWorkflowBuilder(use_context_api=request.use_new_api)
        
        def context_demo_handler(state: MessagesState, context=None) -> MessagesState:
            api_type = "新Context API" if context else "旧config模式"
            user_info = context.user_id if context else "unknown"
            
            state["messages"].append({
                "role": "assistant",
                "content": f"使用{api_type}处理消息。用户: {user_info}。消息: {request.message}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "api_type": api_type,
                    "context_available": context is not None
                }
            })
            return state
        
        builder.add_node("context_demo", context_demo_handler)
        graph = builder.build()
        
        # 添加图边
        from langgraph.graph import START, END
        graph.add_edge(START, "context_demo")
        graph.add_edge("context_demo", END)
        
        # 编译图
        compiled_graph = builder.compile()
        
        # 准备执行状态和上下文
        initial_state = create_initial_state()
        initial_state["messages"] = [
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        ]
        
        if request.use_new_api:
            # 使用新Context API
            context = create_default_context(
                user_id=request.user_id,
                session_id=request.session_id
            )
            if request.conversation_id:
                context.conversation_id = request.conversation_id
            
            result = await builder.execute(initial_state, context=context)
        else:
            # 使用旧config模式
            config = {
                "configurable": {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "conversation_id": request.conversation_id
                }
            }
            result = await builder.execute(initial_state, config=config)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return WorkflowResponse(
            success=True,
            execution_time_ms=execution_time,
            result=result,
            metadata={
                "api_type": "新Context API" if request.use_new_api else "旧config模式",
                "context_schema": "LangGraphContextSchema" if request.use_new_api else "dict"
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Context API演示失败: {str(e)}"
        )


# Durability控制演示端点
@router.post("/durability/demo", response_model=WorkflowResponse)
async def demo_durability_control(request: DurabilityRequest):
    """演示durability参数控制"""
    start_time = datetime.now()
    
    try:
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        def durability_handler(state: MessagesState) -> MessagesState:
            state["messages"].append({
                "role": "assistant",
                "content": f"使用durability模式: {request.durability_mode}处理消息: {request.message}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "durability_mode": request.durability_mode,
                    "checkpoint_info": "根据durability模式决定检查点保存策略"
                }
            })
            return state
        
        builder.add_node("durability_demo", durability_handler)
        graph = builder.build()
        
        from langgraph.graph import START, END
        graph.add_edge(START, "durability_demo")
        graph.add_edge("durability_demo", END)
        
        # 使用指定的durability模式编译
        compiled_graph = builder.compile(durability_mode=request.durability_mode)
        
        initial_state = create_initial_state()
        initial_state["messages"] = [
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        ]
        
        context = create_default_context()
        result = await builder.execute(
            initial_state, 
            context=context, 
            durability=request.durability_mode
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return WorkflowResponse(
            success=True,
            execution_time_ms=execution_time,
            result=result,
            metadata={
                "durability_mode": request.durability_mode,
                "checkpoint_strategy": {
                    "exit": "仅在工作流结束时保存检查点",
                    "async": "异步保存检查点，平衡性能和持久性",
                    "sync": "同步保存检查点，最高持久性保证"
                }[request.durability_mode]
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Durability演示失败: {str(e)}"
        )


# Node Caching演示端点
@router.post("/caching/demo", response_model=WorkflowResponse)
async def demo_node_caching(request: CachingRequest):
    """演示节点缓存功能"""
    start_time = datetime.now()
    
    try:
        builder = LangGraphWorkflowBuilder(use_context_api=True)
        
        # 模拟计算密集型任务
        call_count = 0
        def expensive_computation(state: MessagesState) -> MessagesState:
            nonlocal call_count
            call_count += 1
            
            # 模拟计算延迟
            import time
            time.sleep(0.1)  # 100ms 延迟
            
            state["messages"].append({
                "role": "assistant",
                "content": f"完成计算密集型任务 #{call_count}。消息: {request.message}",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "computation_count": call_count,
                    "cache_enabled": request.enable_cache,
                    "simulated_delay_ms": 100
                }
            })
            return state
        
        if request.enable_cache:
            # 使用缓存包装节点
            cache_policy = CachePolicy(
                ttl=request.cache_ttl,
                enabled=True,
                cache_key_fields=["messages"]
            )
            cached_handler = create_cached_node("expensive_node", expensive_computation, cache_policy)
            builder.add_node("expensive_node", cached_handler)
        else:
            builder.add_node("expensive_node", expensive_computation)
        
        graph = builder.build()
        
        from langgraph.graph import START, END
        graph.add_edge(START, "expensive_node")
        graph.add_edge("expensive_node", END)
        
        compiled_graph = builder.compile()
        
        # 执行多次相同请求测试缓存
        # 使用相同的初始状态和固定时间戳来确保缓存能够正确命中
        fixed_timestamp = datetime.now().isoformat()
        results = []
        for i in range(3):
            initial_state = create_initial_state()
            initial_state["messages"] = [
                {"role": "user", "content": request.message, "timestamp": fixed_timestamp}
            ]
            
            context = create_default_context()
            result = await builder.execute(initial_state, context=context)
            results.append(result)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 获取缓存统计
        cache_manager = get_cache_manager()
        cache_stats = {
            "total_executions": 3,
            "actual_computations": call_count,
            "cache_hits": 3 - call_count if request.enable_cache else 0,
            "cache_enabled": request.enable_cache
        }
        
        return WorkflowResponse(
            success=True,
            execution_time_ms=execution_time,
            result=results[-1],  # 返回最后一次执行结果
            metadata={
                "cache_statistics": cache_stats,
                "performance_improvement": f"{((3 - call_count) / 3 * 100):.1f}%" if request.enable_cache and call_count < 3 else "0%"
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Node Caching演示失败: {str(e)}"
        )


# Pre/Post Hooks演示端点
@router.post("/hooks/demo", response_model=WorkflowResponse)
async def demo_hooks(request: HooksRequest):
    """演示Pre/Post Model Hooks"""
    start_time = datetime.now()
    
    try:
        # 准备测试状态
        state = create_initial_state()
        state["messages"] = request.messages
        
        context = create_default_context()
        hook_manager = get_hook_manager()
        
        original_state = state.copy()
        
        # 执行预处理钩子
        if request.enable_pre_hooks:
            state = await hook_manager.execute_pre_hooks(state, context)
        
        # 模拟模型处理
        state["messages"].append({
            "role": "assistant",
            "content": "这是AI模型生成的响应内容，可能包含需要处理的内容。",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"generated_by": "demo_model"}
        })
        
        # 执行后处理钩子
        if request.enable_post_hooks:
            state = await hook_manager.execute_post_hooks(state, context)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 分析钩子效果
        hook_effects = []
        if "hook_logs" in state["context"]:
            hook_effects = state["context"]["hook_logs"]
        
        # 计算实际执行的hooks数量
        executed_hooks_count = 0
        if request.enable_pre_hooks:
            # 检查pre hooks的执行痕迹
            messages = state.get("messages", [])
            for msg in messages:
                if msg.get("metadata", {}).get("hook") == "ContextEnrichmentHook":
                    executed_hooks_count += 1
                    break
            
        if request.enable_post_hooks:
            # 检查post hooks的执行痕迹
            messages = state.get("messages", [])
            for msg in messages:
                if msg.get("metadata", {}).get("enhancement_hook") == "ResponseEnhancementHook":
                    executed_hooks_count += 1
                    break
        
        return WorkflowResponse(
            success=True,
            execution_time_ms=execution_time,
            result=state,
            metadata={
                "original_message_count": len(original_state["messages"]),
                "final_message_count": len(state["messages"]),
                "hooks_executed": executed_hooks_count,
                "hook_effects": hook_effects,
                "pre_hooks_enabled": request.enable_pre_hooks,
                "post_hooks_enabled": request.enable_post_hooks
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Hooks演示失败: {str(e)}"
        )


# 钩子配置管理端点
@router.get("/hooks/status")
async def get_hooks_status():
    """获取钩子状态"""
    hook_manager = get_hook_manager()
    return hook_manager.get_hook_status()


@router.post("/hooks/configure")
async def configure_hook(hook_name: str, enabled: bool):
    """配置钩子启用状态"""
    hook_manager = get_hook_manager()
    
    # 查找并配置钩子
    for hook in hook_manager.pre_hooks + hook_manager.post_hooks:
        if hook.config.name == hook_name:
            hook.config.enabled = enabled
            return {"success": True, "message": f"钩子 {hook_name} 已{'启用' if enabled else '禁用'}"}
    
    raise HTTPException(status_code=404, detail=f"钩子 {hook_name} 未找到")


# 缓存管理端点
@router.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息"""
    cache_manager = get_cache_manager()
    
    # 简化的缓存统计
    return {
        "cache_backend": type(cache_manager.backend).__name__,
        "default_policy": {
            "ttl": cache_manager.default_policy.ttl,
            "max_size": cache_manager.default_policy.max_size,
            "enabled": cache_manager.default_policy.enabled
        },
        "node_policies_count": len(cache_manager.node_policies)
    }


@router.post("/cache/clear")
async def clear_cache():
    """清空缓存"""
    cache_manager = get_cache_manager()
    await cache_manager.backend.clear()
    return {"success": True, "message": "缓存已清空"}


# 完整工作流演示端点
@router.post("/complete-demo", response_model=WorkflowResponse)
async def complete_feature_demo():
    """演示所有LangGraph 0.6.5新特性的完整工作流"""
    start_time = datetime.now()
    
    try:
        # 创建使用所有新特性的工作流
        builder = create_conditional_workflow()
        compiled_graph = builder.compile(durability_mode="async")
        
        # 准备测试数据
        initial_state = create_initial_state()
        initial_state["messages"] = [
            {"role": "user", "content": "请演示LangGraph 0.6.5的所有新特性", "timestamp": datetime.now().isoformat()}
        ]
        
        # 使用新Context API
        context = create_default_context(
            user_id="demo_user_complete",
            session_id="550e8400-e29b-41d4-a716-446655440000"
        )
        
        # 执行工作流
        result = await builder.execute(
            initial_state, 
            context=context, 
            durability="async"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return WorkflowResponse(
            success=True,
            execution_time_ms=execution_time,
            result=result,
            metadata={
                "features_demonstrated": [
                    "新Context API (LangGraphContextSchema)",
                    "Durability控制 (async模式)",
                    "条件分支工作流",
                    "类型安全的上下文传递"
                ],
                "workflow_type": "conditional_workflow",
                "context_api_version": "0.6.5"
            }
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=500,
            detail=f"完整演示失败: {str(e)}"
        )