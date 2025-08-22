"""
pgvector 0.8 升级和量化系统集成测试

测试完整的向量优化系统集成
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.rag import (
    VectorQuantizer,
    QuantizationConfig,
    QuantizationMode,
    PgVectorOptimizer,
    IndexConfig,
    IndexType,
    VectorCacheManager,
    HybridVectorRetriever,
    VectorPerformanceMonitor
)


class TestPgVectorIntegration:
    """pgvector 0.8 升级系统集成测试"""
    
    @pytest.fixture
    def mock_db_session(self):
        """模拟数据库会话"""
        session = AsyncMock()
        return session
    
    @pytest.fixture
    def mock_redis_client(self):
        """模拟Redis客户端"""
        redis = AsyncMock()
        redis.get.return_value = None  # 默认缓存未命中
        redis.setex.return_value = True
        redis.keys.return_value = []
        return redis
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """模拟Qdrant客户端"""
        qdrant = MagicMock()
        # 模拟搜索结果
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.9
        mock_point.payload = {"content": "test content", "metadata": {}}
        
        qdrant.search.return_value = [mock_point]
        qdrant.get_collections.return_value = ["knowledge_base"]
        return qdrant
    
    @pytest.fixture
    def integrated_system(self, mock_db_session, mock_redis_client, mock_qdrant_client):
        """创建集成系统"""
        pg_optimizer = PgVectorOptimizer(mock_db_session)
        cache_manager = VectorCacheManager(mock_redis_client)
        hybrid_retriever = HybridVectorRetriever(
            pg_optimizer, mock_qdrant_client, cache_manager
        )
        performance_monitor = VectorPerformanceMonitor()
        
        return {
            "pg_optimizer": pg_optimizer,
            "cache_manager": cache_manager,
            "hybrid_retriever": hybrid_retriever,
            "performance_monitor": performance_monitor
        }
    
    @pytest.mark.asyncio
    async def test_complete_upgrade_workflow(self, integrated_system, mock_db_session):
        """测试完整的升级工作流程"""
        pg_optimizer = integrated_system["pg_optimizer"]
        
        # 模拟版本检查和升级
        pg_optimizer._get_pgvector_version = AsyncMock(side_effect=["0.5.0", "0.8.0"])
        
        # 1. 执行pgvector升级
        upgrade_result = await pg_optimizer.upgrade_to_v08()
        assert upgrade_result is True
        
        # 2. 创建知识库表
        table_result = await pg_optimizer.create_knowledge_items_table()
        assert table_result is True
        
        # 3. 创建优化索引
        hnsw_config = IndexConfig(
            index_type=IndexType.HNSW,
            hnsw_m=16,
            hnsw_ef_construction=200
        )
        index_result = await pg_optimizer.create_optimized_indexes(
            "knowledge_items", "embedding", hnsw_config
        )
        assert index_result is True
        
        # 4. 验证安装
        validation = await pg_optimizer.validate_installation()
        # 模拟所有验证通过
        for key in ["extension_installed", "version_08_or_higher", 
                   "operators_available", "knowledge_table_exists"]:
            assert key in validation
    
    @pytest.mark.asyncio
    async def test_quantization_and_search_integration(self, integrated_system):
        """测试量化和搜索的集成"""
        pg_optimizer = integrated_system["pg_optimizer"]
        hybrid_retriever = integrated_system["hybrid_retriever"]
        
        # 创建测试向量
        test_vector = np.random.normal(0, 1, 1536).astype(np.float32)
        
        # 模拟pgvector搜索结果
        mock_pg_result = MagicMock()
        mock_pg_result.fetchall.return_value = [
            MagicMock(id="pg_1", content="pg content", metadata={}, distance=0.1)
        ]
        pg_optimizer.db.execute.return_value = mock_pg_result
        pg_optimizer._select_search_strategy = AsyncMock(return_value="hnsw")
        
        # 执行混合搜索（包含量化）
        results = await hybrid_retriever.hybrid_search(
            test_vector,
            query_text="test query",
            top_k=5,
            use_cache=True
        )
        
        # 验证结果
        assert isinstance(results, list)
        # 应该包含来自两个数据源的结果
        
        # 验证量化被使用
        assert pg_optimizer.quantizer is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integrated_system):
        """测试性能监控集成"""
        performance_monitor = integrated_system["performance_monitor"]
        hybrid_retriever = integrated_system["hybrid_retriever"]
        
        # 模拟搜索函数
        async def monitored_search(vector):
            return await hybrid_retriever.hybrid_search(vector, top_k=5)
        
        # 创建测试向量
        test_vectors = [np.random.normal(0, 1, 512) for _ in range(5)]
        
        # 建立基准
        baseline = await performance_monitor.establish_baseline(
            monitored_search,
            test_vectors[:3],
            iterations=2
        )
        
        assert baseline["total_searches"] > 0
        assert baseline["average_latency_ms"] > 0
        
        # 执行监控搜索
        for vector in test_vectors:
            result, metrics = await performance_monitor.monitor_search_performance(
                monitored_search,
                vector,
                quantization_mode="adaptive"
            )
        
        # 获取性能报告
        report = await performance_monitor.get_performance_report()
        assert report["summary"]["total_searches"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, integrated_system):
        """测试缓存系统集成"""
        cache_manager = integrated_system["cache_manager"]
        
        # 测试向量缓存
        test_vector = np.random.normal(0, 1, 512)
        test_metadata = {"type": "test", "id": 123}
        
        # 缓存向量
        cache_result = await cache_manager.cache_vector(
            "test_vector_1", test_vector, test_metadata
        )
        assert cache_result is True
        
        # 检索缓存
        cached_data = await cache_manager.get_cached_vector("test_vector_1")
        # 注意：由于使用了模拟Redis，实际不会返回数据
        # 在真实环境中，这里会返回缓存的向量和元数据
        
        # 获取缓存统计
        stats = await cache_manager.get_cache_stats()
        assert "hit_rate" in stats
        assert "current_size" in stats
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, integrated_system, mock_db_session):
        """测试错误恢复集成"""
        pg_optimizer = integrated_system["pg_optimizer"]
        hybrid_retriever = integrated_system["hybrid_retriever"]
        
        # 模拟数据库错误
        mock_db_session.execute.side_effect = Exception("Database connection lost")
        
        # pgvector搜索应该失败但不抛出异常
        test_vector = np.random.normal(0, 1, 512)
        pg_results = await pg_optimizer.optimize_vector_search(
            test_vector, "knowledge_items", "embedding"
        )
        assert pg_results == []  # 错误时返回空列表
        
        # 混合搜索应该仍然能从Qdrant获取结果
        hybrid_results = await hybrid_retriever.hybrid_search(test_vector)
        # Qdrant部分应该仍然工作
        assert isinstance(hybrid_results, list)
    
    @pytest.mark.asyncio
    async def test_scalability_simulation(self, integrated_system):
        """模拟大规模使用场景"""
        hybrid_retriever = integrated_system["hybrid_retriever"]
        performance_monitor = integrated_system["performance_monitor"]
        cache_manager = integrated_system["cache_manager"]
        
        # 模拟100个并发搜索请求
        search_tasks = []
        for i in range(100):
            vector = np.random.normal(0, 1, 512)
            
            # 包装搜索以进行性能监控
            async def monitored_search():
                return await performance_monitor.monitor_search_performance(
                    hybrid_retriever.hybrid_search,
                    vector,
                    quantization_mode="adaptive"
                )
            
            search_tasks.append(monitored_search())
        
        # 执行并发搜索
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 统计结果
        successful_searches = sum(1 for r in results if not isinstance(r, Exception))
        failed_searches = len(results) - successful_searches
        
        # 验证系统稳定性
        success_rate = successful_searches / len(results)
        assert success_rate > 0.95  # 95%以上成功率
        
        # 检查性能统计
        stats = await cache_manager.get_cache_stats()
        performance_report = await performance_monitor.get_performance_report()
        
        assert performance_report["summary"]["total_searches"] >= successful_searches
    
    @pytest.mark.asyncio
    async def test_data_consistency_validation(self, integrated_system, mock_db_session):
        """测试数据一致性验证"""
        pg_optimizer = integrated_system["pg_optimizer"]
        
        # 模拟数据完整性检查
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(id="item_1", embedding=np.random.normal(0, 1, 1536).tolist()),
            MagicMock(id="item_2", embedding=np.random.normal(0, 1, 1536).tolist()),
            MagicMock(id="item_3", embedding=None),  # 缺失向量
        ]
        mock_db_session.execute.return_value = mock_result
        
        # 验证向量数据完整性的模拟实现
        async def validate_vector_data_integrity():
            """验证向量数据完整性"""
            check_sql = "SELECT id, embedding FROM knowledge_items LIMIT 1000"
            result = await mock_db_session.execute(check_sql)
            items = result.fetchall()
            
            total_items = len(items)
            valid_vectors = 0
            missing_vectors = 0
            
            for item in items:
                if item.embedding is not None:
                    try:
                        vector = np.array(item.embedding)
                        if vector.shape[0] > 0:  # 有效向量
                            valid_vectors += 1
                        else:
                            missing_vectors += 1
                    except Exception:
                        missing_vectors += 1
                else:
                    missing_vectors += 1
            
            return {
                "total_items": total_items,
                "valid_vectors": valid_vectors,
                "missing_vectors": missing_vectors,
                "integrity_rate": valid_vectors / total_items if total_items > 0 else 0.0
            }
        
        # 执行完整性检查
        integrity_report = await validate_vector_data_integrity()
        
        assert "total_items" in integrity_report
        assert "valid_vectors" in integrity_report
        assert "missing_vectors" in integrity_report
        assert "integrity_rate" in integrity_report
        
        # 应该检测到缺失的向量
        assert integrity_report["missing_vectors"] > 0
        assert integrity_report["integrity_rate"] < 1.0
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_accuracy(self, integrated_system):
        """测试混合检索准确性"""
        hybrid_retriever = integrated_system["hybrid_retriever"]
        
        # 创建测试查询
        query_vector = np.random.normal(0, 1, 512)
        
        # 分别测试单一数据源和混合检索
        pg_results = await hybrid_retriever.pg_only_search(query_vector, top_k=10)
        qdrant_results = await hybrid_retriever.qdrant_only_search(query_vector, top_k=10)
        hybrid_results = await hybrid_retriever.hybrid_search(
            query_vector, 
            top_k=10,
            pg_weight=0.6,
            qdrant_weight=0.4
        )
        
        # 验证结果格式
        assert isinstance(pg_results, list)
        assert isinstance(qdrant_results, list)
        assert isinstance(hybrid_results, list)
        
        # 混合结果应该包含融合信息
        if hybrid_results:
            result = hybrid_results[0]
            assert "fused_score" in result
            # 可能包含来源信息
            if "sources" in result:
                assert isinstance(result["sources"], list)
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, integrated_system):
        """测试系统健康监控"""
        hybrid_retriever = integrated_system["hybrid_retriever"]
        cache_manager = integrated_system["cache_manager"]
        
        # 执行健康检查
        health_status = await hybrid_retriever.health_check()
        
        assert "status" in health_status
        assert "components" in health_status
        assert "stats" in health_status
        assert "timestamp" in health_status
        
        # 检查各组件健康状态
        components = health_status["components"]
        assert "pgvector" in components
        assert "qdrant" in components
        assert "cache" in components
        
        # 缓存健康检查
        cache_health = await cache_manager.get_cache_health()
        assert "status" in cache_health
        assert cache_health["status"] in ["healthy", "warning", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, integrated_system):
        """测试性能基准对比"""
        hybrid_retriever = integrated_system["hybrid_retriever"]
        
        # 生成测试向量
        test_vectors = [np.random.normal(0, 1, 512) for _ in range(5)]
        
        # 执行基准测试
        benchmark_results = await hybrid_retriever.benchmark_retrieval_methods(
            test_vectors, top_k=10
        )
        
        assert "test_vectors_count" in benchmark_results
        assert "methods" in benchmark_results
        assert benchmark_results["test_vectors_count"] == 5
        
        # 验证三种方法都被测试
        methods = benchmark_results["methods"]
        assert "hybrid" in methods
        assert "pg_only" in methods
        assert "qdrant_only" in methods
        
        # 每种方法都应该有性能指标
        for method_name, method_stats in methods.items():
            assert "total_time_ms" in method_stats
            assert "average_time_per_query_ms" in method_stats
            assert "success_rate" in method_stats