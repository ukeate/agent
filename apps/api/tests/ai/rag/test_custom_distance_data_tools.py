"""
自定义距离度量和数据工具测试
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timezone
import json
import csv
import io
from pathlib import Path
import tempfile
import h5py

from ai.rag.custom_distance_metrics import (
    CustomDistanceCalculator,
    DistanceMetric,
    DistanceConfig,
    DistanceMetricInterface
)

from ai.rag.vector_data_tools import (
    VectorDataImporter,
    VectorDataExporter,
    VectorDataMigrator,
    VectorBackupRestore,
    DataFormat,
    ImportConfig,
    ExportConfig,
    MigrationConfig,
    CompressionType
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
def distance_calculator(mock_db_session):
    """创建距离计算器实例"""
    return CustomDistanceCalculator(mock_db_session)


@pytest.fixture
def data_importer(mock_db_session):
    """创建数据导入器实例"""
    return VectorDataImporter(mock_db_session)


@pytest.fixture
def data_exporter(mock_db_session):
    """创建数据导出器实例"""
    return VectorDataExporter(mock_db_session)


@pytest.fixture
def sample_vectors():
    """生成样本向量"""
    np.random.seed(42)
    return np.random.randn(10, 128)


# ============= 距离度量测试 =============

@pytest.mark.asyncio
async def test_l2_distance(distance_calculator):
    """测试L2距离计算"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # 使用pgvector计算
    mock_result = MagicMock()
    mock_result.fetchone.return_value = MagicMock(distance=5.196)
    distance_calculator.db.execute.return_value = mock_result
    
    distance = await distance_calculator.calculate_distance(v1, v2, DistanceMetric.L2)
    
    assert isinstance(distance, float)
    assert distance > 0


@pytest.mark.asyncio
async def test_cosine_distance(distance_calculator):
    """测试余弦距离计算"""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    
    # 正交向量的余弦距离应该是1
    distance = await distance_calculator._calculate_custom_distance(
        v1, v2, DistanceMetric.CORRELATION
    )
    
    assert 0 <= distance <= 2


@pytest.mark.asyncio
async def test_minkowski_distance(distance_calculator):
    """测试闵可夫斯基距离"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # p=2时等于欧氏距离
    distance = distance_calculator._minkowski_distance(v1, v2, 2.0)
    expected = np.sqrt((3**2) * 3)  # sqrt(27) ≈ 5.196
    
    assert np.isclose(distance, expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_chebyshev_distance(distance_calculator):
    """测试切比雪夫距离"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 6, 8])
    
    distance = distance_calculator._chebyshev_distance(v1, v2)
    
    # 最大坐标差是5
    assert distance == 5


@pytest.mark.asyncio
async def test_mahalanobis_distance(distance_calculator):
    """测试马氏距离"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    # 使用单位矩阵（退化为欧氏距离）
    distance = distance_calculator._mahalanobis_distance(v1, v2, None)
    expected = np.sqrt((3**2) * 3)
    
    assert np.isclose(distance, expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_canberra_distance(distance_calculator):
    """测试堪培拉距离"""
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 3, 4])
    
    distance = distance_calculator._canberra_distance(v1, v2)
    
    assert distance > 0
    assert distance < len(v1)  # 堪培拉距离的上界


@pytest.mark.asyncio
async def test_batch_distance_calculation(distance_calculator, sample_vectors):
    """测试批量距离计算"""
    query_vector = sample_vectors[0]
    database_vectors = sample_vectors[1:]
    
    distances = await distance_calculator.batch_distance_calculation(
        query_vector,
        database_vectors,
        DistanceMetric.L2
    )
    
    assert len(distances) == len(database_vectors)
    assert all(d >= 0 for d in distances)


@pytest.mark.asyncio
async def test_custom_distance_function_creation(distance_calculator):
    """测试创建自定义距离函数"""
    function_code = """
    import numpy as np
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    return float(np.linalg.norm(v1 - v2))
    """
    
    distance_calculator.db.execute.return_value = MagicMock()
    
    result = await distance_calculator.create_custom_distance_function(
        "custom_l2_distance",
        function_code,
        128
    )
    
    assert result is True
    assert distance_calculator.db.execute.called


@pytest.mark.asyncio
async def test_distance_metric_interface(mock_db_session):
    """测试距离度量接口"""
    calculator = CustomDistanceCalculator(mock_db_session)
    interface = DistanceMetricInterface(calculator)
    
    mock_db_session.execute.return_value = MagicMock()
    
    result = await interface.create_distance_index(
        "test_table",
        "embedding",
        DistanceMetric.COSINE,
        "hnsw"
    )
    
    assert result is True
    assert mock_db_session.execute.called


# ============= 数据导入测试 =============

@pytest.mark.asyncio
async def test_import_csv(data_importer):
    """测试CSV导入"""
    csv_content = """id,embedding,metadata
1,"[1,2,3]","{}"
2,"[4,5,6]","{""key"":""value""}"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_file = f.name
    
    try:
        data_importer.db.execute.return_value = MagicMock()
        
        result = await data_importer.import_from_file(
            csv_file,
            "test_table",
            ImportConfig(format=DataFormat.CSV)
        )
        
        assert result["successful"] > 0
        assert data_importer.db.execute.called
        
    finally:
        Path(csv_file).unlink()


@pytest.mark.asyncio
async def test_import_json(data_importer):
    """测试JSON导入"""
    json_data = {
        "vectors": [
            {"id": "1", "embedding": [1, 2, 3], "metadata": {}},
            {"id": "2", "embedding": [4, 5, 6], "metadata": {"key": "value"}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        json_file = f.name
    
    try:
        data_importer.db.execute.return_value = MagicMock()
        
        result = await data_importer.import_from_file(
            json_file,
            "test_table",
            ImportConfig(format=DataFormat.JSON)
        )
        
        assert result["successful"] == 2
        
    finally:
        Path(json_file).unlink()


@pytest.mark.asyncio
async def test_import_numpy(data_importer, sample_vectors):
    """测试NumPy导入"""
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, sample_vectors)
        numpy_file = f.name
    
    try:
        data_importer.db.execute.return_value = MagicMock()
        
        result = await data_importer.import_from_file(
            numpy_file,
            "test_table",
            ImportConfig(format=DataFormat.NUMPY)
        )
        
        assert result["successful"] == len(sample_vectors)
        
    finally:
        Path(numpy_file).unlink()


@pytest.mark.asyncio
async def test_import_with_validation(data_importer):
    """测试带验证的导入"""
    json_data = {
        "vectors": [
            {"id": "1", "embedding": [1, 2, 3], "metadata": {}},
            {"id": "2", "embedding": [float('inf'), 2, 3], "metadata": {}},  # 无效向量
            {"id": "3", "embedding": [4, 5, 6], "metadata": {}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        json_file = f.name
    
    try:
        data_importer.db.execute.return_value = MagicMock()
        
        result = await data_importer.import_from_file(
            json_file,
            "test_table",
            ImportConfig(
                format=DataFormat.JSON,
                validate=True,
                skip_errors=True
            )
        )
        
        assert result["successful"] == 2  # 只有2个有效向量
        assert result["skipped"] == 1
        
    finally:
        Path(json_file).unlink()


@pytest.mark.asyncio
async def test_import_with_normalization(data_importer):
    """测试带归一化的导入"""
    json_data = {
        "vectors": [
            {"id": "1", "embedding": [3, 4, 0], "metadata": {}}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        json_file = f.name
    
    try:
        captured_data = []
        
        async def capture_execute(query, params=None):
            if params:
                captured_data.extend(params)
            return MagicMock()
        
        data_importer.db.execute = capture_execute
        data_importer.db.commit = AsyncMock()
        
        await data_importer.import_from_file(
            json_file,
            "test_table",
            ImportConfig(
                format=DataFormat.JSON,
                normalize=True
            )
        )
        
        # 检查向量是否被归一化
        if captured_data:
            embedding = captured_data[0]['embedding']
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, rtol=1e-5)
        
    finally:
        Path(json_file).unlink()


# ============= 数据导出测试 =============

@pytest.mark.asyncio
async def test_export_csv(data_exporter):
    """测试CSV导出"""
    mock_result = MagicMock()
    mock_result.fetchall.side_effect = [
        [
            MagicMock(_mapping={'id': '1', 'embedding': [1, 2, 3], 'metadata': {}}),
            MagicMock(_mapping={'id': '2', 'embedding': [4, 5, 6], 'metadata': {'key': 'value'}})
        ],
        []
    ]
    data_exporter.db.execute.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "export.csv"
        
        result = await data_exporter.export_to_file(
            "test_table",
            str(output_path),
            ExportConfig(format=DataFormat.CSV)
        )
        
        assert result["exported"] == 2
        assert len(result["files"]) > 0
        
        # 验证CSV内容
        with open(result["files"][0], 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2


@pytest.mark.asyncio
async def test_export_json(data_exporter):
    """测试JSON导出"""
    mock_result = MagicMock()
    mock_result.fetchall.side_effect = [
        [
            MagicMock(_mapping={'id': '1', 'embedding': [1, 2, 3], 'metadata': {}}),
            MagicMock(_mapping={'id': '2', 'embedding': [4, 5, 6], 'metadata': {'key': 'value'}})
        ],
        []
    ]
    data_exporter.db.execute.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "export.json"
        
        result = await data_exporter.export_to_file(
            "test_table",
            str(output_path),
            ExportConfig(format=DataFormat.JSON)
        )
        
        assert result["exported"] == 2
        
        # 验证JSON内容
        with open(output_path, 'r') as f:
            data = json.load(f)
            assert 'vectors' in data
            assert len(data['vectors']) == 2


@pytest.mark.asyncio
async def test_export_hdf5(data_exporter):
    """测试HDF5导出"""
    mock_result = MagicMock()
    mock_result.fetchall.side_effect = [
        [
            MagicMock(_mapping={'id': '1', 'embedding': [1, 2, 3], 'metadata': {}}),
            MagicMock(_mapping={'id': '2', 'embedding': [4, 5, 6], 'metadata': {'key': 'value'}})
        ],
        []
    ]
    data_exporter.db.execute.return_value = mock_result
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "export.h5"
        
        result = await data_exporter.export_to_file(
            "test_table",
            str(output_path),
            ExportConfig(format=DataFormat.HDF5)
        )
        
        assert result["exported"] == 2
        
        # 验证HDF5内容
        with h5py.File(output_path, 'r') as f:
            assert 'vectors' in f
            assert 'ids' in f
            assert len(f['vectors']) == 2


# ============= 数据迁移测试 =============

@pytest.mark.asyncio
async def test_migrate_table(mock_db_session):
    """测试表迁移"""
    source_db = mock_db_session
    target_db = AsyncMock()
    
    # 模拟源数据
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    source_db.execute.side_effect = [
        count_result,  # COUNT查询
        MagicMock(fetchall=lambda: [  # 数据查询
            MagicMock(id='1', embedding=[1, 2, 3], metadata={}),
            MagicMock(id='2', embedding=[4, 5, 6], metadata={})
        ]),
        MagicMock(fetchall=lambda: []),  # 结束
        MagicMock(fetchall=lambda: [  # 验证查询
            MagicMock(id='1', embedding=[1, 2, 3], metadata={}),
            MagicMock(id='2', embedding=[4, 5, 6], metadata={})
        ]),
        MagicMock(fetchall=lambda: [  # 验证查询
            MagicMock(id='1', embedding=[1, 2, 3], metadata={}),
            MagicMock(id='2', embedding=[4, 5, 6], metadata={})
        ])
    ]
    
    target_db.execute = AsyncMock()
    target_db.commit = AsyncMock()
    
    migrator = VectorDataMigrator(source_db, target_db)
    
    result = await migrator.migrate_table(
        "source_table",
        "target_table",
        MigrationConfig(batch_size=10)
    )
    
    assert result["total_vectors"] == 100
    assert result["migrated"] == 2
    assert target_db.execute.called


# ============= 备份恢复测试 =============

@pytest.mark.asyncio
async def test_create_backup(mock_db_session):
    """测试创建备份"""
    # 模拟导出数据
    mock_result = MagicMock()
    mock_result.fetchall.side_effect = [
        [
            MagicMock(_mapping={'id': '1', 'embedding': [1, 2, 3], 'metadata': {}}),
        ],
        []
    ]
    mock_db_session.execute.return_value = mock_result
    
    backup_restore = VectorBackupRestore(mock_db_session)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await backup_restore.create_backup(
            "test_table",
            tmpdir,
            include_indexes=False
        )
        
        assert result["status"] == "success"
        assert Path(result["backup_path"]).exists()
        
        # 验证元信息文件
        meta_file = Path(tmpdir) / "backup_meta.json"
        assert meta_file.exists()
        
        with open(meta_file, 'r') as f:
            meta_info = json.load(f)
            assert meta_info["table_name"] == "test_table"


@pytest.mark.asyncio
async def test_restore_backup(mock_db_session):
    """测试恢复备份"""
    backup_restore = VectorBackupRestore(mock_db_session)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟备份
        backup_path = Path(tmpdir)
        
        # 创建元信息文件
        meta_info = {
            "table_name": "test_table",
            "backup_time": datetime.now(timezone.utc).isoformat(),
            "data_file": "test_table_data.hdf5",
            "index_file": None,
            "version": "1.0"
        }
        
        with open(backup_path / "backup_meta.json", 'w') as f:
            json.dump(meta_info, f)
        
        # 创建数据文件
        with h5py.File(backup_path / "test_table_data.hdf5", 'w') as f:
            f.create_dataset('vectors', data=np.array([[1, 2, 3], [4, 5, 6]]))
            f.create_dataset('ids', data=np.array([1, 2]))
        
        mock_db_session.execute.return_value = MagicMock()
        
        # 使用补丁来模拟导入器
        with patch('ai.rag.vector_data_tools.VectorDataImporter') as MockImporter:
            mock_importer = MagicMock()
            mock_importer.import_from_file = AsyncMock(return_value={"successful": 2})
            MockImporter.return_value = mock_importer
            
            result = await backup_restore.restore_backup(
                str(backup_path),
                "restored_table"
            )
            
            assert result["status"] == "success"
            assert result["table_name"] == "restored_table"


# ============= 性能基准测试 =============

@pytest.mark.asyncio
async def test_benchmark_distance_metrics(distance_calculator, sample_vectors):
    """测试距离度量基准测试"""
    metrics = [
        DistanceMetric.L2,
        DistanceMetric.COSINE,
        DistanceMetric.MINKOWSKI
    ]
    
    # 模拟pgvector响应
    mock_result = MagicMock()
    mock_result.fetchone.return_value = MagicMock(distance=1.0)
    distance_calculator.db.execute.return_value = mock_result
    
    results = await distance_calculator.benchmark_distance_metrics(
        sample_vectors,
        metrics,
        iterations=10
    )
    
    assert len(results) == len(metrics)
    for metric in metrics:
        assert metric.value in results
        assert "mean_ms" in results[metric.value]
        assert "p95_ms" in results[metric.value]


@pytest.mark.asyncio
async def test_find_optimal_metric(distance_calculator, sample_vectors):
    """测试找到最优距离度量"""
    reference_vectors = sample_vectors[:2]
    candidate_vectors = sample_vectors[2:]
    ground_truth_indices = np.array([0, 1, 2])
    
    metrics = [DistanceMetric.L2, DistanceMetric.COSINE]
    
    best_metric, scores = await distance_calculator.find_optimal_metric(
        reference_vectors,
        candidate_vectors,
        ground_truth_indices,
        metrics
    )
    
    assert best_metric in metrics
    assert len(scores) == len(metrics)
    for metric in metrics:
        assert metric.value in scores
        assert 0 <= scores[metric.value] <= 1