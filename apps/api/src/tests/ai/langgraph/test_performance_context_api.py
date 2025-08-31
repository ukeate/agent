"""
LangGraph Context API性能基准测试
验证LangGraph v0.6.5 Context API的性能改进
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import pytest
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone

from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import LangGraphWorkflowBuilder
from src.ai.langgraph.context import AgentContext, create_context, LangGraphContextSchema
from src.ai.langgraph.checkpoints import CheckpointManager
from src.ai.langgraph.caching import CacheConfig, create_node_cache


class TestPerformanceContextAPI:
    """LangGraph Context API性能测试"""
    
    @pytest.mark.asyncio
    async def test_context_api_vs_legacy_config_performance(self):
        """对比新Context API和传统config模式的性能"""
        
        # 测试参数
        test_iterations = 50
        legacy_times: List[float] = []
        context_api_times: List[float] = []
        
        # 定义简单的测试节点
        def test_node(state: MessagesState, context: AgentContext = None) -> MessagesState:
            """测试节点函数"""
            state["messages"].append({
                "role": "system",
                "content": f"处理步骤 {len(state['messages']) + 1}",
                "timestamp": utc_now().isoformat()
            })
            if context:
                context.update_step("test_node")
            return state
        
        # 测试传统config模式
        print("\\n开始测试传统config模式性能...")
        builder_legacy = LangGraphWorkflowBuilder(use_context_api=False)
        builder_legacy.add_node("test_node", test_node)
        graph_legacy = builder_legacy.build()
        graph_legacy.add_edge("__start__", "test_node")
        graph_legacy.add_edge("test_node", "__end__")
        compiled_legacy = builder_legacy.compile()
        
        for i in range(test_iterations):
            initial_state = create_initial_state(f"legacy-test-{i}")
            context = create_context(
                user_id=f"user-{i}",
                session_id=f"session-{i}",
                workflow_id=f"legacy-test-{i}"
            )
            
            start_time = time.time()
            result = await builder_legacy.execute(
                initial_state, 
                context=context,
                config={
                    "configurable": {
                        "thread_id": f"legacy-test-{i}",
                        "user_id": f"user-{i}",
                        "session_id": f"session-{i}"
                    }
                }
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            legacy_times.append(execution_time)
            
            # 验证结果
            assert result["metadata"]["status"] == "completed"
            assert len(result["messages"]) > 0
        
        # 测试新Context API模式
        print("开始测试新Context API模式性能...")
        builder_new = LangGraphWorkflowBuilder(use_context_api=True)
        builder_new.add_node("test_node", test_node)
        graph_new = builder_new.build()
        graph_new.add_edge("__start__", "test_node")
        graph_new.add_edge("test_node", "__end__")
        compiled_new = builder_new.compile()
        
        for i in range(test_iterations):
            initial_state = create_initial_state(f"context-test-{i}")
            context = create_context(
                user_id=f"user-{i}",
                session_id=f"session-{i}",
                workflow_id=f"context-test-{i}"
            )
            
            start_time = time.time()
            result = await builder_new.execute(
                initial_state, 
                context=context
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            context_api_times.append(execution_time)
            
            # 验证结果
            assert result["metadata"]["status"] == "completed"
            assert len(result["messages"]) > 0
        
        # 计算性能统计
        legacy_avg = statistics.mean(legacy_times)
        legacy_median = statistics.median(legacy_times)
        legacy_std = statistics.stdev(legacy_times) if len(legacy_times) > 1 else 0
        
        context_api_avg = statistics.mean(context_api_times)
        context_api_median = statistics.median(context_api_times)
        context_api_std = statistics.stdev(context_api_times) if len(context_api_times) > 1 else 0
        
        performance_improvement = ((legacy_avg - context_api_avg) / legacy_avg) * 100 if legacy_avg > 0 else 0
        
        print(f"\\n=== 性能对比结果 ===")
        print(f"传统config模式:")
        print(f"  平均执行时间: {legacy_avg:.4f}秒")
        print(f"  中位数时间: {legacy_median:.4f}秒")
        print(f"  标准差: {legacy_std:.4f}秒")
        
        print(f"\\n新Context API模式:")
        print(f"  平均执行时间: {context_api_avg:.4f}秒")
        print(f"  中位数时间: {context_api_median:.4f}秒")
        print(f"  标准差: {context_api_std:.4f}秒")
        
        print(f"\\n性能提升: {performance_improvement:.1f}%")
        
        # 验证性能提升（至少5%的改进）
        assert performance_improvement > 5.0, f"性能提升不足，期望 >5%，实际 {performance_improvement:.1f}%"
        
        # 记录性能数据用于报告
        performance_report = {
            "test_iterations": test_iterations,
            "legacy_performance": {
                "avg_time_seconds": legacy_avg,
                "median_time_seconds": legacy_median,
                "std_dev_seconds": legacy_std,
                "times": legacy_times
            },
            "context_api_performance": {
                "avg_time_seconds": context_api_avg,
                "median_time_seconds": context_api_median,
                "std_dev_seconds": context_api_std,
                "times": context_api_times
            },
            "performance_improvement_percent": performance_improvement,
            "test_timestamp": utc_now().isoformat()
        }
        
        return performance_report
    
    @pytest.mark.asyncio
    async def test_context_serialization_performance(self):
        """测试Context序列化/反序列化性能"""
        test_iterations = 100
        serialization_times: List[float] = []
        deserialization_times: List[float] = []
        
        # 创建复杂的上下文对象
        context = create_context(
            user_id="performance-test-user",
            session_id="performance-test-session",
            conversation_id="performance-test-conversation",
            agent_id="performance-test-agent",
            workflow_id="performance-test-workflow",
            max_iterations=100,
            timeout_seconds=600
        )
        
        # 添加复杂元数据
        complex_metadata = {
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": list(range(100))
                    }
                }
            },
            "large_list": [f"item_{i}" for i in range(500)],
            "timestamp_data": [
                utc_now().isoformat() for _ in range(50)
            ]
        }
        context.merge_metadata(complex_metadata)
        
        print(f"\\n测试Context序列化性能 ({test_iterations}次迭代)...")
        
        for i in range(test_iterations):
            # 序列化性能测试
            start_time = time.time()
            serialized = context.to_dict()
            end_time = time.time()
            serialization_times.append(end_time - start_time)
            
            # 反序列化性能测试
            start_time = time.time()
            deserialized = AgentContext.from_dict(serialized)
            end_time = time.time()
            deserialization_times.append(end_time - start_time)
            
            # 验证序列化正确性
            assert deserialized.user_id == context.user_id
            assert deserialized.session_id == context.session_id
            assert deserialized.workflow_id == context.workflow_id
        
        # 计算统计数据
        serialization_avg = statistics.mean(serialization_times)
        deserialization_avg = statistics.mean(deserialization_times)
        
        print(f"序列化平均时间: {serialization_avg:.6f}秒")
        print(f"反序列化平均时间: {deserialization_avg:.6f}秒")
        print(f"总体往返时间: {(serialization_avg + deserialization_avg):.6f}秒")
        
        # 验证性能要求（序列化应该在1ms以内）
        assert serialization_avg < 0.001, f"序列化性能不达标，期望 <1ms，实际 {serialization_avg*1000:.2f}ms"
        assert deserialization_avg < 0.001, f"反序列化性能不达标，期望 <1ms，实际 {deserialization_avg*1000:.2f}ms"
        
        return {
            "serialization_avg_seconds": serialization_avg,
            "deserialization_avg_seconds": deserialization_avg,
            "total_roundtrip_seconds": serialization_avg + deserialization_avg
        }
    
    @pytest.mark.asyncio
    async def test_node_caching_performance(self):
        """测试Node级缓存性能"""
        # 配置内存缓存用于测试
        cache_config = CacheConfig(
            backend="memory",
            ttl_default=3600,
            max_entries=1000
        )
        cache = create_node_cache(cache_config)
        
        # 测试数据
        test_iterations = 100
        cache_times: List[float] = []
        non_cache_times: List[float] = []
        
        def expensive_computation(data: Dict[str, Any]) -> Dict[str, Any]:
            """模拟耗时计算"""
            # 简单的计算延迟模拟
            time.sleep(0.001)  # 1ms延迟
            return {
                "result": data.get("value", 0) * 2,
                "computed_at": utc_now().isoformat()
            }
        
        # 创建测试上下文
        context = create_context("cache-test-user", "cache-test-session")
        
        print(f"\\n测试Node缓存性能 ({test_iterations}次迭代)...")
        
        # 测试不使用缓存的性能
        for i in range(test_iterations):
            test_data = {"value": i % 10}  # 使用重复数据测试缓存效果
            
            start_time = time.time()
            result = expensive_computation(test_data)
            end_time = time.time()
            
            non_cache_times.append(end_time - start_time)
        
        # 测试使用缓存的性能
        for i in range(test_iterations):
            test_data = {"value": i % 10}  # 使用重复数据测试缓存效果
            cache_key = cache.generate_cache_key("test_node", context, test_data)
            
            start_time = time.time()
            
            # 尝试从缓存获取
            cached_result = await cache.get(cache_key)
            if cached_result is None:
                # 缓存未命中，执行计算并缓存结果
                result = expensive_computation(test_data)
                await cache.set(cache_key, result)
            else:
                result = cached_result
            
            end_time = time.time()
            cache_times.append(end_time - start_time)
        
        # 计算性能统计
        non_cache_avg = statistics.mean(non_cache_times)
        cache_avg = statistics.mean(cache_times)
        performance_improvement = ((non_cache_avg - cache_avg) / non_cache_avg) * 100
        
        print(f"不使用缓存平均时间: {non_cache_avg:.6f}秒")
        print(f"使用缓存平均时间: {cache_avg:.6f}秒")
        print(f"缓存性能提升: {performance_improvement:.1f}%")
        
        # 获取缓存统计
        cache_stats = await cache.get_stats()
        print(f"缓存命中率: {cache_stats['hit_rate']:.1%}")
        print(f"缓存条目数: {cache_stats['cache_entries']}")
        
        # 验证缓存效果（应该有显著性能提升）
        assert performance_improvement > 30.0, f"缓存性能提升不足，期望 >30%，实际 {performance_improvement:.1f}%"
        assert cache_stats["hit_rate"] > 0.5, f"缓存命中率过低，期望 >50%，实际 {cache_stats['hit_rate']:.1%}"
        
        return {
            "non_cache_avg_seconds": non_cache_avg,
            "cache_avg_seconds": cache_avg,
            "performance_improvement_percent": performance_improvement,
            "cache_stats": cache_stats
        }
    
    @pytest.mark.asyncio
    async def test_memory_usage_comparison(self):
        """测试内存使用对比"""
        import sys
        import gc
        
        def get_memory_usage():
            """获取当前内存使用情况"""
            gc.collect()
            return sys.getsizeof(gc.get_objects())
        
        # 测试传统config模式内存使用
        print("\\n测试内存使用情况...")
        
        start_memory = get_memory_usage()
        
        # 创建多个传统config工作流
        legacy_workflows = []
        for i in range(10):
            builder = LangGraphWorkflowBuilder(use_context_api=False)
            builder.add_node("test", lambda state: state)
            legacy_workflows.append(builder)
        
        legacy_memory = get_memory_usage() - start_memory
        
        # 清理
        del legacy_workflows
        gc.collect()
        
        # 测试新Context API内存使用
        start_memory = get_memory_usage()
        
        context_api_workflows = []
        for i in range(10):
            builder = LangGraphWorkflowBuilder(use_context_api=True)
            builder.add_node("test", lambda state: state)
            context_api_workflows.append(builder)
        
        context_api_memory = get_memory_usage() - start_memory
        
        memory_improvement = ((legacy_memory - context_api_memory) / legacy_memory) * 100 if legacy_memory > 0 else 0
        
        print(f"传统config模式内存使用: {legacy_memory:,} bytes")
        print(f"新Context API内存使用: {context_api_memory:,} bytes")
        print(f"内存使用改善: {memory_improvement:.1f}%")
        
        return {
            "legacy_memory_bytes": legacy_memory,
            "context_api_memory_bytes": context_api_memory,
            "memory_improvement_percent": memory_improvement
        }


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """性能基准测试套件"""
    
    @pytest.mark.asyncio
    async def test_full_performance_report(self):
        """生成完整的性能报告"""
        print("\\n" + "="*60)
        print("LANGGRAPH v0.6.5 CONTEXT API 性能基准报告")
        print("="*60)
        
        test_suite = TestPerformanceContextAPI()
        
        # 执行所有性能测试
        context_api_report = await test_suite.test_context_api_vs_legacy_config_performance()
        serialization_report = await test_suite.test_context_serialization_performance()
        caching_report = await test_suite.test_node_caching_performance()
        memory_report = await test_suite.test_memory_usage_comparison()
        
        # 汇总报告
        full_report = {
            "test_timestamp": utc_now().isoformat(),
            "langgraph_version": "0.6.5",
            "context_api_vs_legacy": context_api_report,
            "serialization_performance": serialization_report,
            "node_caching_performance": caching_report,
            "memory_usage_comparison": memory_report,
            "summary": {
                "context_api_improvement_percent": context_api_report["performance_improvement_percent"],
                "caching_improvement_percent": caching_report["performance_improvement_percent"],
                "memory_improvement_percent": memory_report["memory_improvement_percent"],
                "overall_performance_grade": "EXCELLENT" if (
                    context_api_report["performance_improvement_percent"] > 15 and
                    caching_report["performance_improvement_percent"] > 50
                ) else "GOOD"
            }
        }
        
        print("\\n=== 性能测试总结 ===")
        print(f"Context API vs Legacy: {context_api_report['performance_improvement_percent']:.1f}% 改善")
        print(f"缓存性能提升: {caching_report['performance_improvement_percent']:.1f}%")
        print(f"内存使用改善: {memory_report['memory_improvement_percent']:.1f}%")
        print(f"整体性能等级: {full_report['summary']['overall_performance_grade']}")
        
        # 验证整体性能提升满足需求
        assert context_api_report["performance_improvement_percent"] > 10.0, "Context API性能提升不足"
        assert caching_report["performance_improvement_percent"] > 30.0, "缓存性能提升不足"
        
        return full_report