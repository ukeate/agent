"""
pgvector优化器测试

测试pgvector 0.8升级、索引优化和向量搜索功能
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.rag.pgvector_optimizer import (
    PgVectorOptimizer,
    IndexConfig,
    IndexType
)


class TestPgVectorOptimizer:
    """pgvector优化器测试"""
    
    @pytest.fixture
    def mock_db_session(self):
        """模拟数据库会话"""
        session = AsyncMock(spec=AsyncSession)
        return session
    
    @pytest.fixture
    def optimizer(self, mock_db_session):
        """创建优化器实例"""
        return PgVectorOptimizer(mock_db_session)
    
    @pytest.fixture
    def sample_vector(self):
        """创建测试向量"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1536).astype(np.float32)
    
    @pytest.mark.asyncio
    async def test_get_pgvector_version(self, optimizer, mock_db_session):
        """测试获取pgvector版本"""
        # 模拟数据库返回版本信息
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(extversion="0.8.0")
        mock_db_session.execute.return_value = mock_result
        
        version = await optimizer._get_pgvector_version()
        
        assert version == "0.8.0"
        mock_db_session.execute.assert_called_once()
    
    def test_version_compare(self, optimizer):
        """测试版本比较功能"""
        # 相等版本
        assert optimizer._version_compare("0.8.0", "0.8.0") == 0
        
        # 新版本大于旧版本
        assert optimizer._version_compare("0.8.1", "0.8.0") == 1
        assert optimizer._version_compare("1.0.0", "0.8.0") == 1
        
        # 旧版本小于新版本
        assert optimizer._version_compare("0.7.0", "0.8.0") == -1
        assert optimizer._version_compare("0.8.0", "1.0.0") == -1
        
        # 不同长度版本号
        assert optimizer._version_compare("0.8", "0.8.0") == 0
        assert optimizer._version_compare("0.8.1", "0.8") == 1
    
    @pytest.mark.asyncio
    async def test_upgrade_to_v08_already_upgraded(self, optimizer, mock_db_session):
        """测试已升级到0.8版本的情况"""
        # 模拟当前版本已经是0.8
        optimizer._get_pgvector_version = AsyncMock(return_value="0.8.0")
        
        result = await optimizer.upgrade_to_v08()
        
        assert result is True
        # 不应该执行升级SQL
        mock_db_session.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_upgrade_to_v08_success(self, optimizer, mock_db_session):
        """测试成功升级到0.8版本"""
        # 模拟版本检查：先返回0.5，升级后返回0.8
        version_calls = ["0.5.0", "0.8.0"]
        optimizer._get_pgvector_version = AsyncMock(side_effect=version_calls)
        
        result = await optimizer.upgrade_to_v08()
        
        assert result is True
        assert mock_db_session.execute.call_count >= 1
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upgrade_to_v08_failure(self, optimizer, mock_db_session):
        """测试升级失败的情况"""
        # 模拟数据库执行出错
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = await optimizer.upgrade_to_v08()
        
        assert result is False
        mock_db_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_hnsw_index(self, optimizer, mock_db_session):
        """测试创建HNSW索引"""
        config = IndexConfig(
            index_type=IndexType.HNSW,
            hnsw_m=16,
            hnsw_ef_construction=200
        )
        
        result = await optimizer.create_optimized_indexes(
            "test_table", "embedding", config
        )
        
        assert result is True
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
        
        # 检查SQL包含HNSW索引创建
        execute_calls = mock_db_session.execute.call_args_list
        sql_text = str(execute_calls[0][0][0])
        assert "USING hnsw" in sql_text
        assert "m = 16" in sql_text
        assert "ef_construction = 200" in sql_text
    
    @pytest.mark.asyncio
    async def test_create_ivf_index(self, optimizer, mock_db_session):
        """测试创建IVF索引"""
        config = IndexConfig(
            index_type=IndexType.IVF,
            ivf_lists=1000
        )
        
        result = await optimizer.create_optimized_indexes(
            "test_table", "embedding", config
        )
        
        assert result is True
        
        # 检查SQL包含IVF索引创建
        execute_calls = mock_db_session.execute.call_args_list
        sql_text = str(execute_calls[0][0][0])
        assert "USING ivfflat" in sql_text
        assert "lists = 1000" in sql_text
    
    @pytest.mark.asyncio
    async def test_create_hybrid_index(self, optimizer, mock_db_session):
        """测试创建混合索引"""
        config = IndexConfig(index_type=IndexType.HYBRID)
        
        result = await optimizer.create_optimized_indexes(
            "test_table", "embedding", config
        )
        
        assert result is True
        # 混合索引应该创建两个索引
        assert mock_db_session.execute.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_create_index_failure(self, optimizer, mock_db_session):
        """测试索引创建失败"""
        config = IndexConfig(index_type=IndexType.HNSW)
        mock_db_session.execute.side_effect = Exception("Index creation failed")
        
        result = await optimizer.create_optimized_indexes(
            "test_table", "embedding", config
        )
        
        assert result is False
        mock_db_session.rollback.assert_called()
    
    @pytest.mark.asyncio
    async def test_select_search_strategy(self, optimizer, mock_db_session):
        """测试搜索策略选择"""
        # 模拟大数据量表
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(estimated_rows=2000000)
        mock_db_session.execute.return_value = mock_result
        
        strategy = await optimizer._select_search_strategy("large_table", 10)
        assert strategy == "hnsw"  # 大数据量，小结果集应该选择HNSW
        
        # 模拟小数据量表
        mock_result.fetchone.return_value = MagicMock(estimated_rows=50000)
        strategy = await optimizer._select_search_strategy("small_table", 10)
        assert strategy == "flat"  # 小数据量应该选择暴力搜索
        
        # 模拟中等数据量表
        mock_result.fetchone.return_value = MagicMock(estimated_rows=500000)
        strategy = await optimizer._select_search_strategy("medium_table", 10)
        assert strategy == "ivf"  # 中等数据量应该选择IVF
    
    @pytest.mark.asyncio
    async def test_optimize_vector_search(self, optimizer, mock_db_session, sample_vector):
        """测试优化向量搜索"""
        # 模拟搜索结果
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(id="1", content="test1", metadata={}, distance=0.1),
            MagicMock(id="2", content="test2", metadata={}, distance=0.2),
        ]
        mock_db_session.execute.return_value = mock_result
        
        # 模拟策略选择
        optimizer._select_search_strategy = AsyncMock(return_value="hnsw")
        
        results = await optimizer.optimize_vector_search(
            sample_vector,
            "test_table",
            "embedding",
            top_k=5
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["content"] == "test1"
        assert results[0]["distance"] == 0.1
        
        # 检查性能统计是否更新
        assert optimizer.performance_stats["searches_optimized"] > 0
    
    @pytest.mark.asyncio
    async def test_optimize_vector_search_with_quantization(self, optimizer, mock_db_session, sample_vector):
        """测试带量化的向量搜索"""
        # 模拟量化结果
        optimizer.quantizer.quantize_vector = AsyncMock(
            return_value=(sample_vector * 0.1, {"mode": "int8", "scale": 0.1})
        )
        
        # 模拟搜索结果
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result
        optimizer._select_search_strategy = AsyncMock(return_value="hnsw")
        
        results = await optimizer.optimize_vector_search(
            sample_vector,
            "test_table",
            "embedding",
            quantize=True
        )
        
        assert isinstance(results, list)
        optimizer.quantizer.quantize_vector.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_knowledge_items_table(self, optimizer, mock_db_session):
        """测试创建知识库表"""
        result = await optimizer.create_knowledge_items_table()
        
        assert result is True
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
        
        # 检查SQL包含表创建语句
        execute_calls = mock_db_session.execute.call_args_list
        sql_text = str(execute_calls[0][0][0])
        assert "CREATE TABLE IF NOT EXISTS knowledge_items" in sql_text
        assert "embedding VECTOR(1536)" in sql_text
    
    @pytest.mark.asyncio
    async def test_validate_installation(self, optimizer, mock_db_session):
        """测试pgvector安装验证"""
        # 模拟各种检查的结果
        check_results = [
            MagicMock(fetchone=lambda: MagicMock()),  # 扩展已安装
            MagicMock(fetchone=lambda: MagicMock()),  # 操作符可用
            MagicMock(fetchone=lambda: MagicMock()),  # 表存在
        ]
        mock_db_session.execute.side_effect = check_results
        
        # 模拟版本检查
        optimizer._get_pgvector_version = AsyncMock(return_value="0.8.0")
        
        results = await optimizer.validate_installation()
        
        assert "extension_installed" in results
        assert "version_08_or_higher" in results
        assert "operators_available" in results
        assert "knowledge_table_exists" in results
        
        # 所有检查都应该通过
        assert results["extension_installed"] is True
        assert results["version_08_or_higher"] is True
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, optimizer):
        """测试获取性能指标"""
        # 设置一些性能统计
        optimizer.performance_stats = {
            "searches_optimized": 100,
            "indexes_created": 5,
            "average_search_latency_ms": 25.5
        }
        
        metrics = await optimizer.get_performance_metrics()
        
        assert "stats" in metrics
        assert "quantizer_config" in metrics
        assert "timestamp" in metrics
        assert metrics["stats"]["searches_optimized"] == 100
    
    def test_update_search_stats(self, optimizer):
        """测试搜索统计更新"""
        initial_searches = optimizer.performance_stats["searches_optimized"]
        
        # 更新统计
        optimizer._update_search_stats(50.0)
        
        assert optimizer.performance_stats["searches_optimized"] == initial_searches + 1
        assert optimizer.performance_stats["average_search_latency_ms"] == 50.0
        
        # 再次更新，测试移动平均
        optimizer._update_search_stats(100.0)
        
        assert optimizer.performance_stats["searches_optimized"] == initial_searches + 2
        # 应该是移动平均值，不是简单平均
        assert 50.0 < optimizer.performance_stats["average_search_latency_ms"] < 100.0


class TestIndexConfig:
    """索引配置测试"""
    
    def test_hnsw_config_default(self):
        """测试HNSW配置默认值"""
        config = IndexConfig(index_type=IndexType.HNSW)
        
        assert config.index_type == IndexType.HNSW
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_search == 100
    
    def test_ivf_config_default(self):
        """测试IVF配置默认值"""
        config = IndexConfig(index_type=IndexType.IVF)
        
        assert config.index_type == IndexType.IVF
        assert config.ivf_lists == 1000
        assert config.ivf_probes == 10
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = IndexConfig(
            index_type=IndexType.HNSW,
            hnsw_m=32,
            hnsw_ef_construction=400,
            ivf_lists=2000
        )
        
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construction == 400
        assert config.ivf_lists == 2000


class TestPgVectorOptimizerIntegration:
    """pgvector优化器集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self, mock_db_session):
        """测试完整优化流程"""
        optimizer = PgVectorOptimizer(mock_db_session)
        
        # 模拟成功的操作
        optimizer._get_pgvector_version = AsyncMock(side_effect=["0.5.0", "0.8.0"])
        optimizer._select_search_strategy = AsyncMock(return_value="hnsw")
        
        # 模拟数据库响应
        mock_search_result = MagicMock()
        mock_search_result.fetchall.return_value = [
            MagicMock(id="1", content="test", metadata={}, distance=0.1)
        ]
        mock_db_session.execute.return_value = mock_search_result
        
        # 1. 升级到0.8
        upgrade_result = await optimizer.upgrade_to_v08()
        assert upgrade_result is True
        
        # 2. 创建优化索引
        config = IndexConfig(index_type=IndexType.HNSW)
        index_result = await optimizer.create_optimized_indexes(
            "knowledge_items", "embedding", config
        )
        assert index_result is True
        
        # 3. 执行优化搜索
        test_vector = np.random.normal(0, 1, 1536)
        search_results = await optimizer.optimize_vector_search(
            test_vector, "knowledge_items", "embedding"
        )
        assert len(search_results) == 1
        
        # 4. 获取性能指标
        metrics = await optimizer.get_performance_metrics()
        assert metrics["stats"]["searches_optimized"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_db_session):
        """测试错误处理流程"""
        optimizer = PgVectorOptimizer(mock_db_session)
        
        # 模拟各种错误情况
        mock_db_session.execute.side_effect = Exception("Database connection lost")
        
        # 升级失败
        upgrade_result = await optimizer.upgrade_to_v08()
        assert upgrade_result is False
        
        # 索引创建失败
        config = IndexConfig(index_type=IndexType.HNSW)
        index_result = await optimizer.create_optimized_indexes(
            "test_table", "embedding", config
        )
        assert index_result is False
        
        # 验证回滚被调用
        assert mock_db_session.rollback.call_count >= 2