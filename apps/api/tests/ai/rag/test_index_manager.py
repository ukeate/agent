"""
索引管理器测试
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ai.rag.index_manager import (
    AdvancedIndexManager,
    IndexType,
    DistanceMetric,
    IndexConfig,
    IndexStats
)


@pytest.fixture
def mock_db_session():
    """模拟数据库会话"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def index_manager(mock_db_session):
    """创建索引管理器实例"""
    return AdvancedIndexManager(mock_db_session)


@pytest.mark.asyncio
async def test_create_hnsw_index(index_manager, mock_db_session):
    """测试创建HNSW索引"""
    config = IndexConfig(
        index_type=IndexType.HNSW,
        distance_metric=DistanceMetric.COSINE,
        hnsw_m=16,
        hnsw_ef_construction=200
    )
    
    # 模拟执行成功
    mock_db_session.execute.return_value = MagicMock()
    
    result = await index_manager.create_hnsw_index(
        "test_table", "embedding", config
    )
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called
    
    # 查找包含HNSW的SQL调用
    hnsw_call_found = False
    for call in mock_db_session.execute.call_args_list:
        sql_text = str(call[0][0])
        if "hnsw" in sql_text.lower():
            hnsw_call_found = True
            assert "m = 16" in sql_text
            assert "ef_construction = 200" in sql_text
            break
    assert hnsw_call_found, "HNSW索引创建SQL未找到"


@pytest.mark.asyncio
async def test_create_ivf_index(index_manager, mock_db_session):
    """测试创建IVF索引"""
    config = IndexConfig(
        index_type=IndexType.IVF,
        distance_metric=DistanceMetric.EUCLIDEAN,
        ivf_lists=1000
    )
    
    mock_db_session.execute.return_value = MagicMock()
    
    result = await index_manager.create_ivf_index(
        "test_table", "embedding", config
    )
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called
    
    # 查找包含IVF的SQL调用
    ivf_call_found = False
    for call in mock_db_session.execute.call_args_list:
        sql_text = str(call[0][0])
        if "ivfflat" in sql_text.lower():
            ivf_call_found = True
            assert "lists = 1000" in sql_text
            break
    assert ivf_call_found, "IVF索引创建SQL未找到"


@pytest.mark.asyncio
async def test_create_lsh_index(index_manager, mock_db_session):
    """测试创建LSH索引"""
    config = IndexConfig(
        index_type=IndexType.LSH,
        lsh_hash_tables=10,
        lsh_hash_bits=128
    )
    
    mock_db_session.execute.return_value = MagicMock()
    
    result = await index_manager.create_lsh_index(
        "test_table", "embedding", config
    )
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called
    
    # 验证创建了多个哈希表
    execute_calls = mock_db_session.execute.call_count
    assert execute_calls >= config.lsh_hash_tables * 2  # 每个哈希表需要添加列和创建索引


@pytest.mark.asyncio
async def test_switch_index_type(index_manager, mock_db_session):
    """测试动态切换索引类型"""
    # 模拟查找和删除现有索引
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(indexname="old_index_1"),
        MagicMock(indexname="old_index_2")
    ]
    mock_db_session.execute.return_value = mock_result
    
    # 切换到HNSW
    result = await index_manager.switch_index_type(
        "test_table", "embedding", IndexType.HNSW
    )
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_analyze_data_characteristics(index_manager, mock_db_session):
    """测试数据特征分析"""
    # 模拟数据库返回
    count_result = MagicMock()
    count_result.fetchone.return_value = MagicMock(count=50000)
    
    sample_vectors = [np.random.randn(384).tolist() for _ in range(100)]
    sample_result = MagicMock()
    sample_result.fetchall.return_value = [(v,) for v in sample_vectors]
    
    mock_db_session.execute.side_effect = [count_result, sample_result]
    
    stats = await index_manager._analyze_data_characteristics(
        "test_table", "embedding", 100
    )
    
    assert stats["total_vectors"] == 50000
    assert stats["dimension"] == 384
    assert "density" in stats
    assert "distribution" in stats
    assert "avg_distance" in stats


@pytest.mark.asyncio
async def test_recommend_index_type(index_manager):
    """测试索引类型推荐"""
    # 小数据集
    stats = {"total_vectors": 5000, "dimension": 128, "distribution": "normal"}
    recommended = index_manager._recommend_index_type(stats)
    assert recommended == IndexType.FLAT
    
    # 中等数据集，低维
    stats = {"total_vectors": 50000, "dimension": 64, "distribution": "normal"}
    recommended = index_manager._recommend_index_type(stats)
    assert recommended == IndexType.IVF
    
    # 中等数据集，高维
    stats = {"total_vectors": 50000, "dimension": 512, "distribution": "normal"}
    recommended = index_manager._recommend_index_type(stats)
    assert recommended == IndexType.HNSW
    
    # 超大数据集，超高维
    stats = {"total_vectors": 5000000, "dimension": 1024, "distribution": "skewed"}
    recommended = index_manager._recommend_index_type(stats)
    assert recommended == IndexType.LSH


@pytest.mark.asyncio
async def test_generate_optimal_config(index_manager):
    """测试最优配置生成"""
    # HNSW配置
    stats = {"total_vectors": 500000, "dimension": 384, "distribution": "normal"}
    config = index_manager._generate_optimal_config(IndexType.HNSW, stats)
    
    assert config.index_type == IndexType.HNSW
    assert config.hnsw_m == 32
    assert config.hnsw_ef_construction == 400
    assert config.hnsw_ef_search == 64
    
    # IVF配置
    config = index_manager._generate_optimal_config(IndexType.IVF, stats)
    assert config.index_type == IndexType.IVF
    assert config.ivf_lists > 0
    assert config.ivf_probes > 0
    
    # LSH配置
    config = index_manager._generate_optimal_config(IndexType.LSH, stats)
    assert config.index_type == IndexType.LSH
    assert config.lsh_hash_tables > 0
    assert config.lsh_hash_bits > 0


@pytest.mark.asyncio
async def test_get_ops_class(index_manager):
    """测试距离度量操作符类映射"""
    assert index_manager._get_ops_class(DistanceMetric.COSINE) == "vector_cosine_ops"
    assert index_manager._get_ops_class(DistanceMetric.EUCLIDEAN) == "vector_l2_ops"
    assert index_manager._get_ops_class(DistanceMetric.DOT_PRODUCT) == "vector_ip_ops"
    assert index_manager._get_ops_class(DistanceMetric.MANHATTAN) == "vector_l1_ops"
    assert index_manager._get_ops_class(DistanceMetric.HAMMING) == "bit_hamming_ops"


@pytest.mark.asyncio
async def test_compute_lsh_hash(index_manager):
    """测试LSH哈希计算"""
    vector = np.random.randn(128)
    hash_str = await index_manager.compute_lsh_hash(vector, n_bits=128)
    
    assert len(hash_str) == 128
    assert all(c in '01' for c in hash_str)
    
    # 相同向量应该产生相同哈希
    hash_str2 = await index_manager.compute_lsh_hash(vector, n_bits=128)
    assert hash_str == hash_str2
    
    # 不同向量应该产生不同哈希（大概率）
    vector2 = np.random.randn(128)
    hash_str3 = await index_manager.compute_lsh_hash(vector2, n_bits=128)
    assert hash_str != hash_str3


@pytest.mark.asyncio
async def test_get_index_stats(index_manager):
    """测试获取索引统计信息"""
    # 添加测试统计
    index_manager.index_stats["test_index"] = IndexStats(
        index_type=IndexType.HNSW,
        total_vectors=100000,
        dimension=384,
        build_time_ms=5000.0,
        memory_usage_mb=256.0,
        last_updated=datetime.now(timezone.utc)
    )
    
    # 获取特定索引统计
    stats = await index_manager.get_index_stats("test_index")
    assert stats["index_type"] == "hnsw"
    assert stats["total_vectors"] == 100000
    assert stats["dimension"] == 384
    assert stats["build_time_ms"] == 5000.0
    assert stats["memory_usage_mb"] == 256.0
    
    # 获取所有索引统计
    all_stats = await index_manager.get_index_stats()
    assert "test_index" in all_stats
    assert all_stats["test_index"]["index_type"] == "hnsw"


@pytest.mark.asyncio
async def test_error_handling(index_manager, mock_db_session):
    """测试错误处理"""
    # 模拟数据库错误
    mock_db_session.execute.side_effect = Exception("Database error")
    
    # 创建HNSW索引失败
    config = IndexConfig(index_type=IndexType.HNSW)
    result = await index_manager.create_hnsw_index(
        "test_table", "embedding", config
    )
    assert result is False
    assert mock_db_session.rollback.called
    
    # 重置mock
    mock_db_session.reset_mock()
    mock_db_session.execute.side_effect = Exception("Database error")
    
    # 创建IVF索引失败
    result = await index_manager.create_ivf_index(
        "test_table", "embedding", config
    )
    assert result is False
    assert mock_db_session.rollback.called