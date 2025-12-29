"""
性能优化器测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.ai.knowledge_graph.performance_optimizer import (
    PerformanceOptimizer,
    QueryPerformance,
    PerformanceStats
)

@pytest.mark.unit
class TestQueryPerformance:
    """查询性能指标测试"""
    
    def test_query_performance_creation(self):
        """测试查询性能指标创建"""
        performance = QueryPerformance(
            query_hash="abc123",
            query_type="read",
            execution_time_ms=150.5,
            result_count=10,
            cache_hit=True
        )
        
        assert performance.query_hash == "abc123"
        assert performance.query_type == "read"
        assert performance.execution_time_ms == 150.5
        assert performance.result_count == 10
        assert performance.cache_hit is True
        assert isinstance(performance.timestamp, datetime)
    
    def test_query_performance_to_dict(self):
        """测试查询性能指标序列化"""
        performance = QueryPerformance(
            query_hash="abc123",
            query_type="read", 
            execution_time_ms=150.5,
            result_count=10,
            cache_hit=True
        )
        
        perf_dict = performance.to_dict()
        
        assert perf_dict["query_hash"] == "abc123"
        assert perf_dict["query_type"] == "read"
        assert perf_dict["execution_time_ms"] == 150.5
        assert perf_dict["result_count"] == 10
        assert perf_dict["cache_hit"] is True
        assert "timestamp" in perf_dict

@pytest.mark.unit
class TestPerformanceStats:
    """性能统计测试"""
    
    def test_performance_stats_creation(self):
        """测试性能统计创建"""
        stats = PerformanceStats(
            total_queries=100,
            cache_hit_rate=0.75,
            avg_query_time_ms=120.5,
            slow_queries_count=5,
            peak_qps=50.0,
            current_connections=10
        )
        
        assert stats.total_queries == 100
        assert stats.cache_hit_rate == 0.75
        assert stats.avg_query_time_ms == 120.5
        assert stats.slow_queries_count == 5
        assert stats.peak_qps == 50.0
        assert stats.current_connections == 10
    
    def test_performance_stats_to_dict(self):
        """测试性能统计序列化"""
        stats = PerformanceStats(
            total_queries=100,
            cache_hit_rate=0.75,
            avg_query_time_ms=120.5,
            slow_queries_count=5,
            peak_qps=50.0,
            current_connections=10
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_queries"] == 100
        assert stats_dict["cache_hit_rate"] == 0.75
        assert stats_dict["avg_query_time_ms"] == 120.5

@pytest.mark.unit
class TestPerformanceOptimizer:
    """性能优化器测试"""
    
    @pytest.fixture
    def mock_graph_db(self):
        """Mock图数据库"""
        db = Mock()
        db.execute_read_query = AsyncMock(return_value=[{"result": "test"}])
        db.execute_write_query = AsyncMock(return_value=[{"result": "test"}])
        db.get_connection_stats = Mock(return_value={"active_connections": 5})
        return db
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis客户端"""
        redis_client = Mock()
        redis_client.ping = AsyncMock()
        redis_client.get = AsyncMock()
        redis_client.setex = AsyncMock()
        redis_client.delete = AsyncMock()
        redis_client.keys = AsyncMock(return_value=[])
        redis_client.aclose = AsyncMock()
        return redis_client
    
    @pytest.fixture
    def optimizer(self, mock_graph_db):
        """性能优化器夹具"""
        return PerformanceOptimizer(mock_graph_db)
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer, mock_redis):
        """测试优化器初始化"""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            with patch('src.ai.knowledge_graph.performance_optimizer.settings') as mock_settings:
                mock_settings.CACHE_ENABLED = True
                mock_settings.CACHE_REDIS_URL = "redis://localhost:6379"
                
                await optimizer.initialize()
                
                assert optimizer.redis_client is not None
                mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimizer_close(self, optimizer, mock_redis):
        """测试优化器关闭"""
        optimizer.redis_client = mock_redis
        
        await optimizer.close()
        
        mock_redis.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_cached_query_cache_miss(self, optimizer, mock_graph_db):
        """测试缓存未命中的查询执行"""
        query = "MATCH (n:Person) RETURN n LIMIT 10"
        parameters = {"limit": 10}
        
        result, performance = await optimizer.execute_cached_query(
            query, parameters, cache_enabled=True, query_type="read"
        )
        
        assert isinstance(result, list)
        assert isinstance(performance, QueryPerformance)
        assert performance.query_type == "read"
        assert not performance.cache_hit
        assert performance.execution_time_ms > 0
        
        # 验证调用了数据库查询
        mock_graph_db.execute_read_query.assert_called_once_with(query, parameters)
    
    @pytest.mark.asyncio
    async def test_execute_cached_query_cache_hit(self, optimizer, mock_redis):
        """测试缓存命中的查询执行"""
        optimizer.redis_client = mock_redis
        
        # Mock Redis返回缓存数据
        cached_data = '[{"name": "张三", "age": 30}]'
        mock_redis.get.return_value = cached_data
        
        query = "MATCH (n:Person {name: $name}) RETURN n"
        parameters = {"name": "张三"}
        
        result, performance = await optimizer.execute_cached_query(
            query, parameters, cache_enabled=True, query_type="read"
        )
        
        assert len(result) == 1
        assert result[0]["name"] == "张三"
        assert performance.cache_hit is True
        assert performance.execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_execute_cached_query_write_operation(self, optimizer, mock_graph_db):
        """测试写操作查询（不缓存）"""
        query = "CREATE (p:Person {name: $name}) RETURN p"
        parameters = {"name": "李四"}
        
        result, performance = await optimizer.execute_cached_query(
            query, parameters, cache_enabled=True, query_type="write"
        )
        
        assert isinstance(result, list)
        assert not performance.cache_hit  # 写操作不应该缓存命中
        
        # 验证调用了写查询
        mock_graph_db.execute_write_query.assert_called_once_with(query, parameters)
    
    def test_generate_query_hash(self, optimizer):
        """测试查询哈希生成"""
        query = "MATCH (n:Person) WHERE n.age > $age RETURN n"
        parameters = {"age": 25}
        
        hash1 = optimizer._generate_query_hash(query, parameters)
        hash2 = optimizer._generate_query_hash(query, parameters)
        
        # 相同查询应该生成相同哈希
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5哈希长度
        
        # 不同参数应该生成不同哈希
        hash3 = optimizer._generate_query_hash(query, {"age": 30})
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_get_from_cache_redis_hit(self, optimizer, mock_redis):
        """测试从Redis缓存获取数据"""
        optimizer.redis_client = mock_redis
        
        cached_data = '[{"name": "张三"}]'
        mock_redis.get.return_value = cached_data
        
        query_hash = "test_hash"
        result = await optimizer._get_from_cache(query_hash)
        
        assert result == [{"name": "张三"}]
        mock_redis.get.assert_called_once_with("query:test_hash")
    
    @pytest.mark.asyncio
    async def test_get_from_cache_local_hit(self, optimizer):
        """测试从本地缓存获取数据"""
        # 设置本地缓存
        query_hash = "local_hash"
        cached_data = [{"name": "本地数据"}]
        optimizer.query_cache[query_hash] = {
            "data": cached_data,
            "expires_at": utc_now() + timedelta(seconds=300)
        }
        
        result = await optimizer._get_from_cache(query_hash)
        
        assert result == cached_data
    
    @pytest.mark.asyncio
    async def test_get_from_cache_local_expired(self, optimizer):
        """测试本地缓存过期"""
        # 设置过期的本地缓存
        query_hash = "expired_hash"
        optimizer.query_cache[query_hash] = {
            "data": [{"name": "过期数据"}],
            "expires_at": utc_now() - timedelta(seconds=60)  # 1分钟前过期
        }
        
        result = await optimizer._get_from_cache(query_hash)
        
        assert result is None
        assert query_hash not in optimizer.query_cache  # 应该被清理
    
    @pytest.mark.asyncio
    async def test_set_to_cache_redis(self, optimizer, mock_redis):
        """测试设置Redis缓存"""
        optimizer.redis_client = mock_redis
        optimizer.cache_ttl = 300
        
        query_hash = "set_hash"
        data = [{"name": "新数据"}]
        
        await optimizer._set_to_cache(query_hash, data)
        
        # 验证Redis设置调用
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "query:set_hash"  # key
        assert call_args[0][1] == 300  # ttl
    
    @pytest.mark.asyncio
    async def test_set_to_cache_local(self, optimizer):
        """测试设置本地缓存"""
        query_hash = "local_set_hash"
        data = [{"name": "本地新数据"}]
        
        await optimizer._set_to_cache(query_hash, data)
        
        assert query_hash in optimizer.query_cache
        assert optimizer.query_cache[query_hash]["data"] == data
        assert optimizer.query_cache[query_hash]["expires_at"] > utc_now()
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_pattern(self, optimizer, mock_redis):
        """测试模式匹配缓存失效"""
        optimizer.redis_client = mock_redis
        
        # Mock Redis keys返回
        mock_redis.keys.return_value = ["query:person_123", "query:person_456"]
        
        # 设置本地缓存
        optimizer.query_cache["person_local"] = {"data": [], "expires_at": utc_now()}
        optimizer.query_cache["company_local"] = {"data": [], "expires_at": utc_now()}
        
        await optimizer.invalidate_cache(pattern="person")
        
        # 验证Redis删除调用
        mock_redis.keys.assert_called_once_with("query:*person*")
        mock_redis.delete.assert_called_once_with("query:person_123", "query:person_456")
        
        # 验证本地缓存清理
        assert "person_local" not in optimizer.query_cache
        assert "company_local" in optimizer.query_cache  # 不匹配模式的应该保留
    
    @pytest.mark.asyncio
    async def test_invalidate_cache_all(self, optimizer, mock_redis):
        """测试清理所有缓存"""
        optimizer.redis_client = mock_redis
        mock_redis.keys.return_value = ["query:test1", "query:test2"]
        
        # 设置本地缓存
        optimizer.query_cache["test"] = {"data": [], "expires_at": utc_now()}
        
        await optimizer.invalidate_cache()
        
        # 验证Redis删除所有
        mock_redis.keys.assert_called_once_with("query:*")
        mock_redis.delete.assert_called_once()
        
        # 验证本地缓存清空
        assert len(optimizer.query_cache) == 0
    
    @pytest.mark.asyncio
    async def test_optimize_indexes(self, optimizer, mock_graph_db):
        """测试索引优化"""
        # Mock返回现有索引信息
        mock_graph_db.execute_read_query.return_value = [
            {
                "name": "existing_index",
                "type": "btree",
                "entityType": "NODE",
                "labelsOrTypes": ["Person"],
                "properties": ["name"],
                "state": "ONLINE"
            }
        ]
        
        # 添加一些慢查询记录
        optimizer.performance_metrics = [
            QueryPerformance("hash1", "read", 1500, 10, False),  # 慢查询
            QueryPerformance("hash2", "read", 2000, 5, False),   # 慢查询
        ]
        optimizer.slow_query_threshold = 1000
        
        result = await optimizer.optimize_indexes()
        
        assert "existing_indexes" in result
        assert "recommended_indexes" in result
        assert "analyzed_queries" in result
        assert result["analyzed_queries"] == 2  # 两个慢查询
        assert len(result["existing_indexes"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, optimizer, mock_graph_db):
        """测试获取性能统计"""
        # 设置性能数据
        optimizer.query_count = 100
        optimizer.cache_hits = 75
        optimizer.total_query_time = 5000.0
        optimizer.current_qps = 25.5
        
        # 添加慢查询记录
        optimizer.performance_metrics = [
            QueryPerformance("hash1", "read", 1500, 10, True),   # 慢查询
            QueryPerformance("hash2", "read", 500, 20, False),   # 正常查询
            QueryPerformance("hash3", "read", 2000, 5, False),   # 慢查询
        ]
        optimizer.slow_query_threshold = 1000
        
        stats = await optimizer.get_performance_stats()
        
        assert isinstance(stats, PerformanceStats)
        assert stats.total_queries == 100
        assert stats.cache_hit_rate == 0.75  # 75/100
        assert stats.avg_query_time_ms == 50.0  # 5000/100
        assert stats.slow_queries_count == 2
        assert stats.peak_qps == 25.5
    
    @pytest.mark.asyncio
    async def test_get_slow_queries(self, optimizer):
        """测试获取慢查询"""
        # 添加查询记录
        optimizer.performance_metrics = [
            QueryPerformance("hash1", "read", 1500, 10, False),  # 慢查询
            QueryPerformance("hash2", "read", 500, 20, True),    # 正常查询
            QueryPerformance("hash3", "read", 2000, 5, False),   # 最慢查询
            QueryPerformance("hash4", "read", 1200, 8, False),   # 慢查询
        ]
        optimizer.slow_query_threshold = 1000
        
        slow_queries = await optimizer.get_slow_queries(limit=2)
        
        assert len(slow_queries) == 2
        # 应该按执行时间降序排列
        assert slow_queries[0].execution_time_ms == 2000  # 最慢的排第一
        assert slow_queries[1].execution_time_ms == 1500
    
    @pytest.mark.asyncio
    async def test_analyze_query_patterns(self, optimizer):
        """测试查询模式分析"""
        # 添加各种查询记录
        optimizer.performance_metrics = [
            QueryPerformance("hash1", "read", 50, 10, True),     # <100ms, read, cache hit
            QueryPerformance("hash2", "write", 200, 1, False),   # 100-500ms, write
            QueryPerformance("hash3", "read", 800, 20, False),   # 500-1000ms, read
            QueryPerformance("hash4", "read", 1500, 5, False),   # >1000ms, read
        ]
        
        analysis = await optimizer.analyze_query_patterns()
        
        # 验证查询类型分布
        assert analysis["query_type_distribution"]["read"] == 3
        assert analysis["query_type_distribution"]["write"] == 1
        
        # 验证执行时间分布
        time_dist = analysis["execution_time_distribution"]
        assert time_dist["<100ms"] == 1
        assert time_dist["100-500ms"] == 1
        assert time_dist["500-1000ms"] == 1
        assert time_dist[">1000ms"] == 1
        
        # 验证缓存性能
        cache_perf = analysis["cache_performance"]
        assert cache_perf["total_cacheable_queries"] == 3  # 3个读查询
        assert cache_perf["cache_hits"] == 1
        assert cache_perf["hit_rate"] == 1/3  # 1个缓存命中，3个可缓存查询
    
    def test_get_cache_stats(self, optimizer):
        """测试获取缓存统计"""
        # 设置缓存数据
        optimizer.query_cache["test1"] = {"data": [], "expires_at": utc_now()}
        optimizer.query_cache["test2"] = {"data": [], "expires_at": utc_now()}
        optimizer.query_count = 50
        optimizer.cache_hits = 20
        optimizer.cache_ttl = 300
        optimizer.max_cache_size = 1000
        
        stats = optimizer.get_cache_stats()
        
        assert stats["local_cache_size"] == 2
        assert stats["max_cache_size"] == 1000
        assert stats["cache_hit_rate"] == 0.4  # 20/50
        assert stats["total_queries"] == 50
        assert stats["cache_hits"] == 20
        assert stats["cache_ttl_seconds"] == 300

@pytest.mark.integration
class TestPerformanceOptimizerIntegration:
    """性能优化器集成测试"""
    
    @pytest.mark.neo4j_integration
    @pytest.mark.asyncio
    async def test_real_performance_optimization(self, test_neo4j_config):
        """测试真实性能优化"""
        from src.ai.knowledge_graph.graph_database import Neo4jGraphDatabase
        
        db = Neo4jGraphDatabase(test_neo4j_config)
        optimizer = PerformanceOptimizer(db)
        
        try:
            await db.initialize()
            await optimizer.initialize()
            
            # 执行一些查询来测试缓存
            query = "RETURN 1 as test_value"
            
            # 第一次执行（缓存未命中）
            result1, perf1 = await optimizer.execute_cached_query(
                query, {}, cache_enabled=True, query_type="read"
            )
            
            # 第二次执行（缓存命中）
            result2, perf2 = await optimizer.execute_cached_query(
                query, {}, cache_enabled=True, query_type="read"  
            )
            
            assert result1 == result2
            assert not perf1.cache_hit  # 第一次未命中
            # 注意：如果没有Redis，第二次也可能未命中（仅本地缓存）
            
            # 获取性能统计
            stats = await optimizer.get_performance_stats()
            assert stats.total_queries >= 2
            
        finally:
            await optimizer.close()
            await db.close()

@pytest.mark.performance
class TestOptimizerPerformance:
    """优化器性能测试"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_throughput_queries(self, optimizer, mock_graph_db):
        """测试高吞吐量查询性能"""
        import time
        
        async def execute_query():
            return await optimizer.execute_cached_query(
                "RETURN 1", {}, cache_enabled=True, query_type="read"
            )
        
        # 并发执行大量查询
        start_time = time.time()
        tasks = [execute_query() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 100
        assert all(r[0] for r in results)  # 所有查询都有结果
        
        # 性能应该合理（100个查询在2秒内完成）
        execution_time = end_time - start_time
        assert execution_time < 2.0
        
        # QPS应该合理
        qps = len(results) / execution_time
        assert qps > 50  # 至少50 QPS
    
    @pytest.mark.slow
    def test_cache_memory_usage(self, optimizer):
        """测试缓存内存使用"""
        # 添加大量缓存项
        for i in range(10000):
            optimizer.query_cache[f"query_{i}"] = {
                "data": [{"result": f"data_{i}"}],
                "expires_at": utc_now() + timedelta(seconds=300)
            }
        
        # 验证缓存大小限制
        assert len(optimizer.query_cache) <= optimizer.max_cache_size
        
        # 获取缓存统计
        stats = optimizer.get_cache_stats()
        assert stats["local_cache_size"] <= optimizer.max_cache_size
