"""
pgvector 0.8.0 优化功能测试
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.ai.rag.vector_store import PgVectorStore
from src.ai.rag.quantization import BinaryQuantizer, HalfPrecisionQuantizer, QuantizationManager
from src.core.monitoring.vector_db_metrics import VectorMetricsCollector, VectorQueryMetrics


@pytest.fixture
def mock_connection():
    """模拟数据库连接"""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value="0.8.0")
    return mock_conn


@pytest.fixture
def mock_vector_store():
    """模拟向量存储"""
    store = MagicMock(spec=PgVectorStore)
    store.get_connection = AsyncMock()
    return store


class TestPgVectorStore:
    """测试PgVectorStore类"""
    
    @pytest.mark.asyncio
    async def test_create_collection(self, mock_connection):
        """测试创建向量集合"""
        with patch('apps.api.src.ai.rag.vector_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            vector_store = PgVectorStore("mock://connection")
            
            result = await vector_store.create_collection(
                collection_name="test_collection",
                dimension=384,
                index_type="hnsw",
                distance_metric="l2"
            )
            
            assert result is True
            mock_connection.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_insert_vectors(self, mock_connection):
        """测试插入向量"""
        mock_connection.fetchval.return_value = "test-uuid"
        
        with patch('apps.api.src.ai.rag.vector_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            vector_store = PgVectorStore("mock://connection")
            
            documents = [
                {
                    "content": "test document",
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"type": "test"}
                }
            ]
            
            result = await vector_store.insert_vectors(
                collection_name="test_collection",
                documents=documents
            )
            
            assert len(result) == 1
            mock_connection.fetchval.assert_called()
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, mock_connection):
        """测试相似性搜索"""
        mock_connection.fetch.return_value = [
            {
                "id": "test-uuid",
                "content": "test content",
                "metadata": {"type": "test"},
                "created_at": datetime.now()
            }
        ]
        
        with patch('apps.api.src.ai.rag.vector_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            vector_store = PgVectorStore("mock://connection")
            
            results = await vector_store.similarity_search(
                collection_name="test_collection",
                query_vector=[0.1, 0.2, 0.3],
                limit=5,
                distance_metric="l2"
            )
            
            assert len(results) == 1
            assert results[0]["content"] == "test content"
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, mock_connection):
        """测试获取集合统计信息"""
        mock_connection.fetchrow.return_value = {
            "total_vectors": 1000,
            "table_size": "10 MB",
            "live_tuples": 950,
            "dead_tuples": 50
        }
        mock_connection.fetch.return_value = []
        
        with patch('apps.api.src.ai.rag.vector_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            vector_store = PgVectorStore("mock://connection")
            
            stats = await vector_store.get_collection_stats("test_collection")
            
            assert stats["total_vectors"] == 1000
            assert stats["table_size"] == "10 MB"


class TestQuantization:
    """测试向量量化功能"""
    
    def test_binary_quantizer_train(self):
        """测试二进制量化器训练"""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        quantizer = BinaryQuantizer(bits=8)
        result = quantizer.train(vectors)
        
        assert result is True
        assert quantizer.is_trained is True
        assert quantizer.thresholds is not None
        assert len(quantizer.thresholds) == 3
    
    def test_binary_quantizer_encode_decode(self):
        """测试二进制量化器编码解码"""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        quantizer = BinaryQuantizer(bits=8)
        quantizer.train(vectors)
        
        # 编码
        encoded = quantizer.encode(vectors)
        assert encoded.shape == (2, 3)
        assert encoded.dtype == np.uint8
        
        # 解码
        decoded = quantizer.decode(encoded)
        assert decoded.shape == vectors.shape
        assert decoded.dtype == np.float32
    
    def test_half_precision_quantizer(self):
        """测试半精度量化器"""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)
        
        quantizer = HalfPrecisionQuantizer()
        quantizer.train(vectors)
        
        # 编码为半精度
        encoded = quantizer.encode(vectors)
        assert encoded.dtype == np.float16
        
        # 解码回单精度
        decoded = quantizer.decode(encoded)
        assert decoded.dtype == np.float32
        
        # 检查压缩比
        params = quantizer.get_params()
        assert params["compression_ratio"] == 2.0
    
    @pytest.mark.asyncio
    async def test_quantization_manager_create_config(self, mock_vector_store):
        """测试量化管理器创建配置"""
        mock_vector_store.get_connection.return_value.__aenter__.return_value.fetch.return_value = [
            {"embedding": "[1.0,2.0,3.0]"},
            {"embedding": "[4.0,5.0,6.0]"}
        ]
        mock_vector_store.get_connection.return_value.__aenter__.return_value.execute = AsyncMock()
        
        manager = QuantizationManager(mock_vector_store)
        
        result = await manager.create_quantization_config(
            collection_name="test_collection",
            quantization_type="halfprecision"
        )
        
        assert result is True
        assert "test_collection" in manager.quantizers


class TestVectorMetrics:
    """测试向量监控指标"""
    
    @pytest.mark.asyncio
    async def test_metrics_collector_initialization(self, mock_connection):
        """测试指标收集器初始化"""
        with patch('apps.api.src.core.monitoring.vector_db_metrics.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            collector = VectorMetricsCollector("mock://connection")
            await collector.initialize()
            
            assert collector.pool is not None
            mock_connection.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_record_query_metrics(self, mock_connection):
        """测试记录查询指标"""
        with patch('apps.api.src.core.monitoring.vector_db_metrics.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            collector = VectorMetricsCollector("mock://connection")
            collector.pool = mock_pool.return_value
            
            metrics = VectorQueryMetrics(
                query_id="test-query-id",
                collection_name="test_collection",
                query_type="similarity_search",
                query_vector_dimension=384,
                result_count=10,
                execution_time_ms=15.5,
                index_scan_time_ms=None,
                distance_metric="l2",
                filters_applied=False,
                cache_hit=False,
                timestamp=datetime.now()
            )
            
            await collector.record_query_metrics(metrics)
            mock_connection.execute.assert_called()
    
    @pytest.mark.asyncio
    async def test_collect_index_metrics(self, mock_connection):
        """测试收集索引指标"""
        mock_connection.fetch.return_value = [
            {
                "collection_name": "test_collection",
                "index_name": "test_index",
                "index_type": "hnsw",
                "index_size_bytes": 1024000,
                "tuples_total": 1000,
                "index_scans": 50,
                "tuples_read": 500,
                "tuples_fetched": 100,
                "last_vacuum": datetime.now(),
                "fragmentation_ratio": 0.1
            }
        ]
        
        with patch('apps.api.src.core.monitoring.vector_db_metrics.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            collector = VectorMetricsCollector("mock://connection")
            collector.pool = mock_pool.return_value
            
            metrics = await collector.collect_index_metrics()
            
            assert len(metrics) == 1
            assert metrics[0].collection_name == "test_collection"
            assert metrics[0].index_type == "hnsw"
            assert metrics[0].index_size_bytes == 1024000
    
    @pytest.mark.asyncio
    async def test_get_performance_report(self, mock_connection):
        """测试获取性能报告"""
        mock_connection.fetchrow.return_value = {
            "total_queries": 100,
            "avg_execution_time": 25.5,
            "min_execution_time": 5.0,
            "max_execution_time": 100.0,
            "median_execution_time": 20.0,
            "p95_execution_time": 75.0,
            "cache_hits": 30,
            "query_types_used": 2,
            "avg_result_count": 8.5
        }
        mock_connection.fetch.return_value = []
        
        with patch('apps.api.src.core.monitoring.vector_db_metrics.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            collector = VectorMetricsCollector("mock://connection")
            collector.pool = mock_pool.return_value
            
            report = await collector.get_performance_report(
                collection_name="test_collection",
                time_range_hours=24
            )
            
            assert "report_period" in report
            assert "query_performance" in report
            assert report["query_performance"]["total_queries"] == 100
            assert report["query_performance"]["cache_hit_rate"] == 0.3


class TestPgVectorIntegration:
    """测试pgvector集成功能"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mock_connection):
        """测试端到端工作流"""
        # 模拟数据库响应
        mock_connection.fetchval.side_effect = ["0.8.0", "test-uuid-1", "test-uuid-2"]
        mock_connection.fetch.return_value = [
            {
                "id": "test-uuid-1",
                "content": "test document 1",
                "metadata": {"type": "test"},
                "created_at": datetime.now()
            }
        ]
        
        with patch('apps.api.src.ai.rag.vector_store.asyncpg.create_pool') as mock_pool:
            mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
            
            # 创建向量存储
            vector_store = PgVectorStore("mock://connection")
            
            # 1. 创建集合
            success = await vector_store.create_collection(
                collection_name="integration_test",
                dimension=3,
                index_type="hnsw",
                distance_metric="l2"
            )
            assert success is True
            
            # 2. 插入向量
            documents = [
                {
                    "content": "test document 1",
                    "embedding": [1.0, 2.0, 3.0],
                    "metadata": {"type": "test"}
                },
                {
                    "content": "test document 2", 
                    "embedding": [4.0, 5.0, 6.0],
                    "metadata": {"type": "test"}
                }
            ]
            
            inserted_ids = await vector_store.insert_vectors(
                collection_name="integration_test",
                documents=documents
            )
            assert len(inserted_ids) == 2
            
            # 3. 相似性搜索
            results = await vector_store.similarity_search(
                collection_name="integration_test",
                query_vector=[1.1, 2.1, 3.1],
                limit=5
            )
            assert len(results) == 1
            assert results[0]["content"] == "test document 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])