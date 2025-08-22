"""
LangGraph缓存集成测试
测试缓存与实际工作流的集成功能
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from src.ai.langgraph.caching import CacheConfig, create_node_cache
from src.ai.langgraph.context import AgentContext
from src.ai.langgraph.cached_node import (
    cached_node,
    invalidate_node_cache,
    cache_warmup
)
from src.ai.langgraph.cache_factory import CacheFactory
from src.ai.langgraph.cache_monitor import (
    CacheMonitor,
    CacheHealthChecker
)


class TestWorkflowCacheIntegration:
    """测试工作流缓存集成"""
    
    @pytest.fixture
    async def cache_setup(self):
        """设置缓存测试环境"""
        config = CacheConfig(
            backend="memory",
            ttl_default=3600,
            max_entries=1000,
            monitoring=True
        )
        cache = create_node_cache(config)
        monitor = CacheMonitor(cache)
        
        yield cache, monitor
        
        # 清理
        await cache.clear()
    
    @pytest.fixture
    def workflow_context(self):
        """工作流上下文fixture"""
        import uuid
        session_id = str(uuid.uuid4())
        return AgentContext(
            user_id="workflow_user",
            session_id=session_id,
            workflow_id="test_workflow",
            agent_id=str(uuid.uuid4()),
            session_context={"session_id": session_id}
        )
    
    @pytest.mark.asyncio
    async def test_multi_node_workflow(self, cache_setup, workflow_context):
        """测试多节点工作流缓存"""
        cache, monitor = cache_setup
        
        # 模拟工作流节点
        @cached_node(name="data_extraction", cache=cache, ttl=1800)
        async def extract_data(context: AgentContext, source: str):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return {
                "extracted_data": f"data_from_{source}",
                "timestamp": datetime.now().isoformat(),
                "node": "data_extraction"
            }
        
        @cached_node(name="data_analysis", cache=cache, ttl=1800)
        async def analyze_data(context: AgentContext, data: Dict[str, Any]):
            await asyncio.sleep(0.05)  # 模拟分析时间
            return {
                "analysis_result": f"analyzed_{data.get('extracted_data', 'unknown')}",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat(),
                "node": "data_analysis"
            }
        
        @cached_node(name="result_formatting", cache=cache, ttl=900)
        async def format_result(context: AgentContext, analysis: Dict[str, Any]):
            await asyncio.sleep(0.02)  # 模拟格式化时间
            return {
                "formatted_result": f"formatted_{analysis.get('analysis_result', 'unknown')}",
                "format": "json",
                "timestamp": datetime.now().isoformat(),
                "node": "result_formatting"
            }
        
        # 第一次执行完整工作流
        start_time = time.time()
        
        extracted = await extract_data(workflow_context, source="database")
        analyzed = await analyze_data(workflow_context, data=extracted)
        formatted = await format_result(workflow_context, analysis=analyzed)
        
        first_execution_time = time.time() - start_time
        
        # 第二次执行（应该全部命中缓存）
        start_time = time.time()
        
        extracted2 = await extract_data(workflow_context, source="database")
        analyzed2 = await analyze_data(workflow_context, data=extracted2)
        formatted2 = await format_result(workflow_context, analysis=analyzed2)
        
        second_execution_time = time.time() - start_time
        
        # 验证结果一致性
        assert extracted == extracted2
        assert analyzed == analyzed2
        assert formatted == formatted2
        
        # 验证缓存加速效果
        assert second_execution_time < first_execution_time * 0.5
        
        # 验证缓存统计
        stats = await cache.get_stats()
        assert stats["hit_count"] >= 3  # 三个节点都应该命中
        assert stats["set_count"] >= 3  # 三个节点都应该被缓存
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_workflow(self, cache_setup, workflow_context):
        """测试缓存失效工作流"""
        cache, monitor = cache_setup
        
        @cached_node(name="user_profile", cache=cache)
        async def get_user_profile(context: AgentContext, user_id: str):
            return {
                "user_id": user_id,
                "name": f"User_{user_id}",
                "last_updated": datetime.now().isoformat()
            }
        
        user_id = "test_user_123"
        
        # 第一次获取用户资料
        profile1 = await get_user_profile(workflow_context, user_id)
        
        # 第二次获取（应该命中缓存）
        profile2 = await get_user_profile(workflow_context, user_id)
        assert profile1 == profile2
        
        # 手动失效缓存
        success = await invalidate_node_cache(
            node_name="user_profile",
            context=workflow_context,
            inputs={"user_id": user_id},
            cache=cache
        )
        assert success is True
        
        # 第三次获取（应该重新计算）
        await asyncio.sleep(0.01)  # 确保时间戳不同
        profile3 = await get_user_profile(workflow_context, user_id)
        
        # 验证缓存失效后重新计算
        assert profile3["user_id"] == profile1["user_id"]
        assert profile3["name"] == profile1["name"]
        # 时间戳应该不同（重新计算）
        assert profile3["last_updated"] != profile1["last_updated"]
    
    @pytest.mark.asyncio
    async def test_cache_warmup_workflow(self, cache_setup, workflow_context):
        """测试缓存预热工作流"""
        cache, monitor = cache_setup
        
        @cached_node(name="recommendation", cache=cache)
        async def generate_recommendation(context: AgentContext, user_type: str, category: str):
            await asyncio.sleep(0.1)  # 模拟复杂计算
            return {
                "recommendations": [f"item_{user_type}_{category}_{i}" for i in range(5)],
                "user_type": user_type,
                "category": category,
                "generated_at": datetime.now().isoformat()
            }
        
        # 准备预热数据
        warmup_contexts = [workflow_context] * 3
        warmup_inputs = [
            {"user_type": "premium", "category": "electronics"},
            {"user_type": "basic", "category": "books"},
            {"user_type": "premium", "category": "clothing"}
        ]
        
        # 执行缓存预热
        warmup_stats = await cache_warmup(
            node_func=generate_recommendation.__wrapped__,  # 获取原始函数
            contexts=warmup_contexts,
            inputs_list=warmup_inputs,
            node_name="recommendation",
            cache=cache
        )
        
        # 验证预热统计
        assert warmup_stats["success_count"] == 3
        assert warmup_stats["error_count"] == 0
        assert warmup_stats["total_count"] == 3
        
        # 验证预热效果：调用预热的组合应该很快
        start_time = time.time()
        result = await generate_recommendation(
            workflow_context, 
            user_type="premium", 
            category="electronics"
        )
        execution_time = time.time() - start_time
        
        # 应该很快（命中缓存）
        assert execution_time < 0.05
        assert result["user_type"] == "premium"
        assert result["category"] == "electronics"


class TestCacheFactoryIntegration:
    """测试缓存工厂集成"""
    
    @pytest.mark.asyncio
    async def test_factory_singleton(self):
        """测试工厂单例模式"""
        factory1 = CacheFactory()
        factory2 = CacheFactory()
        
        # 应该是同一个实例
        assert factory1 is factory2
        
        # 缓存实例也应该是单例
        cache1 = factory1.get_cache()
        cache2 = factory2.get_cache()
        
        assert cache1 is cache2
    
    @pytest.mark.asyncio
    async def test_factory_config_override(self):
        """测试配置覆盖"""
        factory = CacheFactory()
        
        # 重置工厂以测试新配置
        factory.reset_cache()
        
        # 获取缓存（应该使用默认配置）
        cache = factory.get_cache()
        assert cache is not None
        
        # 清理
        await factory.close_cache()


class TestCacheMonitoringIntegration:
    """测试缓存监控集成"""
    
    @pytest.fixture
    async def monitored_cache_setup(self):
        """设置带监控的缓存"""
        config = CacheConfig(
            backend="memory",
            monitoring=True,
            ttl_default=3600
        )
        cache = create_node_cache(config)
        monitor = CacheMonitor(cache)
        health_checker = CacheHealthChecker(cache)
        
        # 启动监控
        await monitor.start_monitoring()
        
        yield cache, monitor, health_checker
        
        # 清理
        await monitor.stop_monitoring()
        await cache.clear()
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, monitored_cache_setup):
        """测试实时监控"""
        cache, monitor, health_checker = monitored_cache_setup
        context = AgentContext(user_id="monitor_user", session_id="monitor_session")
        
        @cached_node(name="monitored_node", cache=cache)
        async def monitored_function(context: AgentContext, data: str):
            await asyncio.sleep(0.05)
            return {"processed": data, "timestamp": datetime.now().isoformat()}
        
        # 执行一系列操作
        for i in range(10):
            await monitored_function(context, data=f"data_{i}")
        
        # 重复一些调用以产生缓存命中
        for i in range(5):
            await monitored_function(context, data=f"data_{i}")
        
        # 等待监控收集数据
        await asyncio.sleep(0.1)
        
        # 检查监控指标
        stats = await monitor.get_detailed_stats()
        assert stats["set_count"] >= 10  # 至少10次设置
        assert stats["hit_count"] >= 5   # 至少5次命中
        assert stats["hit_rate"] > 0     # 有命中率
        assert "avg_get_latency_ms" in stats
        assert "avg_set_latency_ms" in stats
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, monitored_cache_setup):
        """测试健康监控"""
        cache, monitor, health_checker = monitored_cache_setup
        
        # 执行健康检查
        health = await health_checker.health_check()
        
        # 验证健康状态
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in health
        assert "timestamp" in health
        
        # 验证具体检查项
        if health["status"] == "healthy":
            assert all(check["status"] == "pass" for check in health["checks"].values())
        
        # 执行性能检查
        performance = await health_checker.performance_check()
        
        # 验证性能数据
        assert "latency_ms" in performance
        assert "timestamp" in performance
        assert "status" in performance


class TestCacheErrorHandling:
    """测试缓存错误处理"""
    
    @pytest.mark.asyncio
    async def test_cache_failure_fallback(self):
        """测试缓存失败时的降级处理"""
        # 创建一个会失败的缓存配置
        config = CacheConfig(
            backend="redis",
            redis_url="redis://nonexistent:6379/0"  # 不存在的Redis
        )
        
        # 创建缓存实例（连接会失败）
        cache = create_node_cache(config)
        context = AgentContext(user_id="error_user", session_id="error_session")
        
        @cached_node(name="error_prone_node", cache=cache)
        async def error_prone_function(context: AgentContext, data: str):
            return {"result": f"processed_{data}"}
        
        # 调用函数（缓存失败但函数应该正常工作）
        result = await error_prone_function(context, data="test")
        
        # 验证函数正常执行（即使缓存失败）
        assert result["result"] == "processed_test"
        
        # 验证错误统计
        assert cache.stats.error_count > 0
    
    @pytest.mark.asyncio
    async def test_serialization_error_handling(self):
        """测试序列化错误处理"""
        config = CacheConfig(backend="memory", serialize_method="json")
        cache = create_node_cache(config)
        context = AgentContext(user_id="ser_user", session_id="ser_session")
        
        # 创建不可JSON序列化的数据
        class NonSerializable:
            def __init__(self):
                self.data = "test"
        
        @cached_node(name="serialization_node", cache=cache)
        async def serialization_function(context: AgentContext):
            return NonSerializable()  # 不可JSON序列化
        
        # 调用函数（序列化会失败但应该降级处理）
        result = await serialization_function(context)
        
        # 验证函数正常执行
        assert isinstance(result, NonSerializable)
        assert result.data == "test"


@pytest.mark.asyncio
async def test_end_to_end_caching_workflow():
    """端到端缓存工作流测试"""
    # 设置完整的缓存环境
    config = CacheConfig(
        backend="memory",
        ttl_default=3600,
        max_entries=100,
        monitoring=True
    )
    
    cache = create_node_cache(config)
    monitor = CacheMonitor(cache)
    health_checker = CacheHealthChecker(cache)
    
    # 启动监控
    await monitor.start_monitoring()
    
    try:
        # 创建模拟的AI工作流
        context = AgentContext(
            user_id="e2e_user",
            session_id="e2e_session",
            workflow_id="e2e_workflow",
            agent_id="e2e_agent"
        )
        
        @cached_node(name="context_analysis", cache=cache, ttl=1800)
        async def analyze_context(context: AgentContext, text: str):
            await asyncio.sleep(0.1)  # 模拟LLM调用
            return {
                "entities": ["person", "location"],
                "sentiment": "positive",
                "confidence": 0.9,
                "text_length": len(text)
            }
        
        @cached_node(name="response_generation", cache=cache, ttl=900)
        async def generate_response(context: AgentContext, analysis: Dict[str, Any]):
            await asyncio.sleep(0.15)  # 模拟LLM生成
            return {
                "response": f"Based on analysis: {analysis['sentiment']} sentiment detected",
                "entities_count": len(analysis['entities']),
                "generated_at": datetime.now().isoformat()
            }
        
        # 执行工作流多次以测试缓存效果
        test_inputs = [
            "Hello, how are you doing today?",
            "I'm feeling great about this project!",
            "Hello, how are you doing today?",  # 重复
            "What's the weather like in New York?"
        ]
        
        execution_times = []
        results = []
        
        for text in test_inputs:
            start_time = time.time()
            
            # 执行两阶段工作流
            analysis = await analyze_context(context, text)
            response = await generate_response(context, analysis)
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            results.append(response)
        
        # 验证缓存加速效果
        # 第3次调用（重复输入）应该比第1次快
        assert execution_times[2] < execution_times[0] * 0.5
        
        # 验证结果一致性（相同输入应该得到相同结果）
        assert results[0]["response"] == results[2]["response"]
        
        # 检查监控数据
        stats = await monitor.get_detailed_stats()
        assert stats["hit_count"] >= 2  # 至少有一些缓存命中
        assert stats["set_count"] >= 6  # 至少6次缓存设置（4个输入，每个2个节点，去重后）
        assert stats["hit_rate"] > 0
        
        # 检查健康状态
        health = await health_checker.health_check()
        assert health["status"] in ["healthy", "degraded"]
        
        # 检查性能
        performance = await health_checker.performance_check()
        assert performance["status"] in ["fast", "slow"]
        assert "latency_ms" in performance
        
        print(f"工作流性能统计:")
        print(f"  - 总执行次数: {len(test_inputs)}")
        print(f"  - 缓存命中率: {stats['hit_rate']:.2%}")
        print(f"  - 平均获取延迟: {stats['avg_get_latency_ms']:.2f}ms")
        print(f"  - 执行时间范围: {min(execution_times)*1000:.2f}-{max(execution_times)*1000:.2f}ms")
        
    finally:
        # 清理
        await monitor.stop_monitoring()
        await cache.clear()