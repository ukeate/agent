"""
LangGraph缓存功能单元测试
测试Node级缓存的核心功能和性能
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.ai.langgraph.caching import (
    CacheConfig,
    CacheStats, 
    NodeCache,
    RedisNodeCache,
    MemoryNodeCache,
    create_node_cache
)
from src.ai.langgraph.context import AgentContext
from src.ai.langgraph.cached_node import (
    CachedNodeWrapper,
    cached_node,
    invalidate_node_cache
)
from src.ai.langgraph.cache_monitor import (
    CacheMonitor,
    CacheHealthChecker,
    CacheMetrics
)


class TestCacheConfig:
    """测试缓存配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "redis"
        assert config.ttl_default == 3600
        assert config.max_entries == 10000
        assert config.key_prefix == "langgraph:cache"
        assert config.serialize_method == "pickle"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = CacheConfig(
            enabled=False,
            backend="memory",
            ttl_default=1800,
            max_entries=5000,
            key_prefix="test:cache"
        )
        assert config.enabled is False
        assert config.backend == "memory"
        assert config.ttl_default == 1800
        assert config.max_entries == 5000
        assert config.key_prefix == "test:cache"


class TestCacheStats:
    """测试缓存统计"""
    
    def test_initial_stats(self):
        """测试初始统计数据"""
        stats = CacheStats()
        assert stats.hit_count == 0
        assert stats.miss_count == 0
        assert stats.set_count == 0
        assert stats.error_count == 0
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
    
    def test_hit_rate_calculation(self):
        """测试命中率计算"""
        stats = CacheStats()
        stats.hit_count = 7
        stats.miss_count = 3
        
        assert stats.hit_rate == 0.7
        assert stats.miss_rate == 0.3
    
    def test_latency_calculation(self):
        """测试延迟计算"""
        stats = CacheStats()
        stats.get_latency_total = 0.5  # 0.5秒
        stats.get_operations = 10
        
        assert stats.avg_get_latency_ms == 50.0  # 50毫秒
    
    def test_to_dict(self):
        """测试转换为字典"""
        stats = CacheStats()
        stats.hit_count = 5
        stats.miss_count = 2
        
        data = stats.to_dict()
        assert data["hit_count"] == 5
        assert data["miss_count"] == 2
        assert data["hit_rate"] == 5/7
        assert "uptime_seconds" in data


class TestMemoryNodeCache:
    """测试内存缓存实现"""
    
    @pytest.fixture
    def memory_cache(self):
        """内存缓存fixture"""
        config = CacheConfig(backend="memory", max_entries=3)
        return MemoryNodeCache(config)
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, memory_cache):
        """测试设置和获取"""
        key = "test:key"
        value = {"data": "test_value", "timestamp": time.time()}
        
        # 设置缓存
        success = await memory_cache.set(key, value, ttl=3600)
        assert success is True
        
        # 获取缓存
        retrieved_value = await memory_cache.get(key)
        assert retrieved_value == value
        
        # 统计验证
        assert memory_cache.stats.set_count == 1
        assert memory_cache.stats.hit_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, memory_cache):
        """测试缓存未命中"""
        key = "nonexistent:key"
        
        value = await memory_cache.get(key)
        assert value is None
        assert memory_cache.stats.miss_count == 1
    
    @pytest.mark.asyncio
    async def test_ttl_expiry(self, memory_cache):
        """测试TTL过期"""
        key = "test:ttl"
        value = {"data": "expire_test"}
        
        # 设置1秒TTL
        await memory_cache.set(key, value, ttl=1)
        
        # 立即获取应该成功
        result = await memory_cache.get(key)
        assert result == value
        
        # 等待过期
        await asyncio.sleep(1.1)
        
        # 现在应该获取不到
        result = await memory_cache.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, memory_cache):
        """测试LRU淘汰策略"""
        # 填满缓存（max_entries=3）
        for i in range(3):
            await memory_cache.set(f"key_{i}", f"value_{i}")
        
        # 访问key_0使其成为最近使用
        await memory_cache.get("key_0")
        
        # 添加新键，应该淘汰key_1（最久未使用）
        await memory_cache.set("key_3", "value_3")
        
        # 验证key_1被淘汰
        assert await memory_cache.get("key_1") is None
        # 验证key_0仍存在
        assert await memory_cache.get("key_0") == "value_0"
        # 验证新键存在
        assert await memory_cache.get("key_3") == "value_3"
    
    @pytest.mark.asyncio
    async def test_delete(self, memory_cache):
        """测试删除缓存"""
        key = "test:delete"
        value = {"data": "to_delete"}
        
        await memory_cache.set(key, value)
        assert await memory_cache.exists(key) is True
        
        success = await memory_cache.delete(key)
        assert success is True
        assert await memory_cache.exists(key) is False
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self, memory_cache):
        """测试模式清理"""
        # 设置多个测试键
        await memory_cache.set("test:1", "value1")
        await memory_cache.set("test:2", "value2")
        await memory_cache.set("other:1", "value3")
        
        # 清理test:*模式
        count = await memory_cache.clear("test:*")
        assert count == 2
        
        # 验证清理结果
        assert await memory_cache.get("test:1") is None
        assert await memory_cache.get("test:2") is None
        assert await memory_cache.get("other:1") == "value3"


@pytest.mark.asyncio 
class TestRedisNodeCache:
    """测试Redis缓存实现（需要Mock）"""
    
    @pytest.fixture
    def redis_cache_config(self):
        """Redis缓存配置fixture"""
        return CacheConfig(
            backend="redis",
            redis_url="redis://localhost:6379/1",
            ttl_default=3600
        )
    
    @patch('redis.asyncio.from_url')
    async def test_redis_connection(self, mock_redis, redis_cache_config):
        """测试Redis连接"""
        # 模拟Redis客户端
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping = AsyncMock(return_value=True)
        
        cache = RedisNodeCache(redis_cache_config)
        redis_client = await cache._get_redis()
        
        # 验证连接创建
        mock_redis.assert_called_once()
        assert redis_client == mock_redis_client
    
    @patch('redis.asyncio.from_url')
    async def test_redis_set_get(self, mock_redis, redis_cache_config):
        """测试Redis设置和获取"""
        # 模拟Redis客户端
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.ping = AsyncMock(return_value=True)
        
        test_data = {"test": "data"}
        serialized_data = cache._serialize_value(test_data)
        
        mock_redis_client.set = AsyncMock(return_value=True)
        mock_redis_client.get = AsyncMock(return_value=serialized_data)
        
        cache = RedisNodeCache(redis_cache_config)
        
        # 测试设置
        success = await cache.set("test:key", test_data, ttl=1800)
        assert success is True
        mock_redis_client.set.assert_called_once_with(
            "test:key", serialized_data, ex=1800
        )
        
        # 测试获取
        result = await cache.get("test:key")
        assert result == test_data
        mock_redis_client.get.assert_called_once_with("test:key")
    
    @patch('redis.asyncio.from_url')
    async def test_redis_connection_failure(self, mock_redis, redis_cache_config):
        """测试Redis连接失败处理"""
        # 模拟连接失败
        mock_redis.side_effect = Exception("Connection failed")
        
        cache = RedisNodeCache(redis_cache_config)
        
        # 连接失败时get应该返回None
        result = await cache.get("test:key")
        assert result is None
        assert cache.stats.error_count > 0


class TestCacheKeyGeneration:
    """测试缓存键生成"""
    
    @pytest.fixture
    def cache(self):
        """缓存实例fixture"""
        config = CacheConfig()
        return MemoryNodeCache(config)
    
    @pytest.fixture
    def context(self):
        """上下文fixture"""
        return AgentContext(
            user_id="test_user",
            session_id="test_session",
            workflow_id="test_workflow"
        )
    
    def test_cache_key_generation(self, cache, context):
        """测试缓存键生成"""
        node_name = "test_node"
        inputs = {"param1": "value1", "param2": 42}
        
        key = cache.generate_cache_key(node_name, context, inputs)
        
        # 验证键包含必要组件
        assert "langgraph:cache" in key  # 前缀
        assert "test_node" in key  # 节点名
        assert "user:test_user" in key  # 用户ID
        assert "session:test_session" in key  # 会话ID
        assert "workflow:test_workflow" in key  # 工作流ID
        assert "inputs:" in key  # 输入哈希
        assert "ctx:" in key  # 上下文哈希
    
    def test_cache_key_consistency(self, cache, context):
        """测试缓存键一致性"""
        node_name = "test_node"
        inputs = {"param1": "value1", "param2": 42}
        
        key1 = cache.generate_cache_key(node_name, context, inputs)
        key2 = cache.generate_cache_key(node_name, context, inputs)
        
        # 相同参数应该生成相同的键
        assert key1 == key2
    
    def test_cache_key_uniqueness(self, cache, context):
        """测试缓存键唯一性"""
        node_name = "test_node"
        inputs1 = {"param1": "value1"}
        inputs2 = {"param1": "value2"}
        
        key1 = cache.generate_cache_key(node_name, context, inputs1)
        key2 = cache.generate_cache_key(node_name, context, inputs2)
        
        # 不同输入应该生成不同的键
        assert key1 != key2


class TestCachedNodeWrapper:
    """测试缓存节点包装器"""
    
    @pytest.fixture
    def mock_cache(self):
        """模拟缓存fixture"""
        cache = AsyncMock(spec=NodeCache)
        cache.config = MagicMock()
        cache.config.enabled = True
        return cache
    
    @pytest.fixture
    def context(self):
        """上下文fixture"""
        return AgentContext(
            user_id="test_user",
            session_id="test_session"
        )
    
    @pytest.mark.asyncio
    async def test_cached_node_hit(self, mock_cache, context):
        """测试缓存命中"""
        # 模拟缓存命中
        cached_result = {"cached": "data"}
        mock_cache.get.return_value = cached_result
        mock_cache.generate_cache_key.return_value = "test:cache:key"
        
        # 创建测试函数
        @cached_node(cache=mock_cache)
        async def test_function(context: AgentContext, param1: str):
            return {"computed": "result", "param1": param1}
        
        # 调用函数
        result = await test_function(context, param1="test")
        
        # 验证返回缓存结果
        assert result == cached_result
        
        # 验证缓存被查询但未设置
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cached_node_miss(self, mock_cache, context):
        """测试缓存未命中"""
        # 模拟缓存未命中
        mock_cache.get.return_value = None
        mock_cache.generate_cache_key.return_value = "test:cache:key"
        mock_cache.set.return_value = True
        
        # 创建测试函数
        @cached_node(cache=mock_cache)
        async def test_function(context: AgentContext, param1: str):
            return {"computed": "result", "param1": param1}
        
        # 调用函数
        result = await test_function(context, param1="test")
        
        # 验证返回计算结果
        expected_result = {"computed": "result", "param1": "test"}
        assert result == expected_result
        
        # 验证缓存被查询和设置
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once_with(
            "test:cache:key", expected_result, ttl=None
        )
    
    @pytest.mark.asyncio
    async def test_cached_node_disabled(self, mock_cache, context):
        """测试缓存禁用"""
        # 禁用缓存
        mock_cache.config.enabled = False
        
        # 创建测试函数
        @cached_node(cache=mock_cache)
        async def test_function(context: AgentContext, param1: str):
            return {"computed": "result", "param1": param1}
        
        # 调用函数
        result = await test_function(context, param1="test")
        
        # 验证直接返回计算结果，不使用缓存
        expected_result = {"computed": "result", "param1": "test"}
        assert result == expected_result
        
        # 验证缓存未被调用
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cached_node_no_context(self, mock_cache):
        """测试无上下文的节点"""
        mock_cache.generate_cache_key.return_value = None
        
        # 创建无上下文的测试函数
        @cached_node(cache=mock_cache)
        async def test_function(param1: str):
            return {"computed": "result", "param1": param1}
        
        # 调用函数
        result = await test_function(param1="test")
        
        # 验证直接返回计算结果
        expected_result = {"computed": "result", "param1": "test"}
        assert result == expected_result
        
        # 验证缓存未被使用
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()


class TestCacheMonitor:
    """测试缓存监控器"""
    
    @pytest.fixture
    def mock_cache(self):
        """模拟缓存fixture"""
        cache = AsyncMock(spec=NodeCache)
        cache.get_stats = AsyncMock(return_value={
            "cache_entries": 100,
            "redis_used_memory": 1024000
        })
        return cache
    
    @pytest.fixture
    def monitor(self, mock_cache):
        """缓存监控器fixture"""
        return CacheMonitor(mock_cache)
    
    def test_metrics_recording(self, monitor):
        """测试指标记录"""
        # 记录各种操作
        monitor.record_hit()
        monitor.record_hit()
        monitor.record_miss()
        monitor.record_set()
        monitor.record_error()
        
        # 验证统计
        assert monitor.metrics.hit_count == 2
        assert monitor.metrics.miss_count == 1
        assert monitor.metrics.set_count == 1
        assert monitor.metrics.error_count == 1
        assert monitor.metrics.hit_rate == 2/3
    
    @pytest.mark.asyncio
    async def test_detailed_stats(self, monitor):
        """测试详细统计信息"""
        # 记录一些操作
        monitor.record_hit()
        monitor.record_miss()
        
        # 获取详细统计
        stats = await monitor.get_detailed_stats()
        
        # 验证统计内容
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5
        assert "cache_backend" in stats
        assert "cache_entries" in stats
    
    def test_summary(self, monitor):
        """测试监控摘要"""
        monitor.record_hit()
        monitor.record_miss()
        
        summary = monitor.get_summary()
        
        assert "hit_rate" in summary
        assert "total_operations" in summary
        assert "current_entries" in summary


class TestCacheHealthChecker:
    """测试缓存健康检查器"""
    
    @pytest.fixture
    def mock_cache(self):
        """模拟缓存fixture"""
        cache = AsyncMock(spec=NodeCache)
        return cache
    
    @pytest.fixture
    def health_checker(self, mock_cache):
        """健康检查器fixture"""
        return CacheHealthChecker(mock_cache)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, health_checker, mock_cache):
        """测试健康检查成功"""
        # 模拟成功的缓存操作
        mock_cache.set.return_value = True
        mock_cache.get.return_value = {"timestamp": time.time()}
        mock_cache.delete.return_value = True
        
        health = await health_checker.health_check()
        
        assert health["status"] == "healthy"
        assert all(check["status"] == "pass" for check in health["checks"].values())
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, health_checker, mock_cache):
        """测试健康检查失败"""
        # 模拟缓存操作失败
        mock_cache.set.side_effect = Exception("Redis connection failed")
        
        health = await health_checker.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["checks"]["connection"]["status"] == "fail"
    
    @pytest.mark.asyncio
    async def test_performance_check(self, health_checker, mock_cache):
        """测试性能检查"""
        # 模拟快速响应
        mock_cache.set.return_value = True
        mock_cache.get.return_value = {"test": "data"}
        mock_cache.delete.return_value = True
        
        performance = await health_checker.performance_check()
        
        assert "latency_ms" in performance
        assert "set" in performance["latency_ms"]
        assert "get" in performance["latency_ms"]
        assert performance["status"] in ["fast", "slow"]


class TestCacheFactory:
    """测试缓存工厂"""
    
    def test_create_redis_cache(self):
        """测试创建Redis缓存"""
        config = CacheConfig(backend="redis")
        cache = create_node_cache(config)
        
        assert isinstance(cache, RedisNodeCache)
    
    def test_create_memory_cache(self):
        """测试创建内存缓存"""
        config = CacheConfig(backend="memory")
        cache = create_node_cache(config)
        
        assert isinstance(cache, MemoryNodeCache)
    
    def test_create_unknown_backend(self):
        """测试创建未知后端缓存"""
        config = CacheConfig(backend="unknown")
        cache = create_node_cache(config)
        
        # 应该降级到内存缓存
        assert isinstance(cache, MemoryNodeCache)


@pytest.mark.asyncio
async def test_cache_integration():
    """集成测试：完整的缓存工作流"""
    # 创建内存缓存用于测试
    config = CacheConfig(backend="memory", ttl_default=3600)
    cache = create_node_cache(config)
    
    # 创建测试上下文
    context = AgentContext(
        user_id="integration_user",
        session_id="integration_session"
    )
    
    # 创建缓存节点
    @cached_node(name="integration_node", cache=cache)
    async def compute_result(context: AgentContext, input_data: dict):
        # 模拟计算延迟
        await asyncio.sleep(0.01)
        return {
            "result": input_data["value"] * 2,
            "computed_at": datetime.now().isoformat()
        }
    
    # 第一次调用（缓存未命中）
    input_data = {"value": 42}
    start_time = time.time()
    result1 = await compute_result(context, input_data)
    first_call_time = time.time() - start_time
    
    # 第二次调用（缓存命中）
    start_time = time.time()
    result2 = await compute_result(context, input_data)
    second_call_time = time.time() - start_time
    
    # 验证结果一致
    assert result1 == result2
    assert result1["result"] == 84
    
    # 验证缓存提速效果
    assert second_call_time < first_call_time
    
    # 验证缓存统计
    assert cache.stats.hit_count == 1
    assert cache.stats.miss_count == 0  # 内部计数可能不同
    assert cache.stats.set_count == 1