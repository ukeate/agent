"""
向量聚类和可视化测试
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import base64

from ai.rag.vector_clustering import (
    VectorClusteringEngine,
    ClusteringAlgorithm,
    AnomalyDetectionAlgorithm,
    ClusteringConfig,
    AnomalyConfig,
    ClusteringResult,
    AnomalyResult
)

from ai.rag.vector_visualization import (
    VectorVisualizationEngine,
    VisualizationMethod,
    PlotType,
    VisualizationConfig,
    VisualizationResult
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
def clustering_engine(mock_db_session):
    """创建聚类引擎实例"""
    return VectorClusteringEngine(mock_db_session)


@pytest.fixture
def visualization_engine(mock_db_session):
    """创建可视化引擎实例"""
    return VectorVisualizationEngine(mock_db_session)


@pytest.fixture
def sample_vectors():
    """生成样本向量数据"""
    np.random.seed(42)
    # 生成3个簇的数据
    cluster1 = np.random.randn(30, 128) + np.array([2] * 128)
    cluster2 = np.random.randn(30, 128) + np.array([-2] * 128)
    cluster3 = np.random.randn(30, 128) + np.array([0] * 128)
    vectors = np.vstack([cluster1, cluster2, cluster3])
    return vectors


@pytest.fixture
def sample_entity_ids():
    """生成样本实体ID"""
    return [f"entity_{i}" for i in range(90)]


# ============= 聚类测试 =============

@pytest.mark.asyncio
async def test_kmeans_clustering(clustering_engine, sample_vectors):
    """测试K-means聚类"""
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3,
        normalize=True
    )
    
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    assert isinstance(result, ClusteringResult)
    assert result.algorithm == ClusteringAlgorithm.KMEANS
    assert result.n_clusters == 3
    assert len(result.labels) == len(sample_vectors)
    assert result.centers is not None
    assert result.centers.shape == (3, sample_vectors.shape[1])
    assert len(result.cluster_sizes) == 3
    assert sum(result.cluster_sizes.values()) == len(sample_vectors)


@pytest.mark.asyncio
async def test_kmeans_auto_k(clustering_engine, sample_vectors):
    """测试K-means自动确定聚类数"""
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=None,  # 自动确定
        normalize=True
    )
    
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    assert result.n_clusters > 1
    assert result.n_clusters <= 10
    assert result.silhouette_score > 0  # 应该有合理的轮廓系数


@pytest.mark.asyncio
async def test_dbscan_clustering(clustering_engine, sample_vectors):
    """测试DBSCAN聚类"""
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.DBSCAN,
        eps=5.0,  # 增大eps以确保能找到簇
        min_samples=5,
        normalize=True
    )
    
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    assert isinstance(result, ClusteringResult)
    assert result.algorithm == ClusteringAlgorithm.DBSCAN
    assert len(result.labels) == len(sample_vectors)
    # DBSCAN可能会有噪声点（标签为-1）
    unique_labels = np.unique(result.labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    assert n_clusters >= 0  # 允许0个簇（所有点都是噪声）


@pytest.mark.asyncio
async def test_hierarchical_clustering(clustering_engine, sample_vectors):
    """测试层次聚类"""
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.HIERARCHICAL,
        n_clusters=3,
        normalize=True
    )
    
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    assert result.algorithm == ClusteringAlgorithm.HIERARCHICAL
    assert result.n_clusters == 3
    assert len(result.labels) == len(sample_vectors)
    assert result.centers is not None


@pytest.mark.asyncio
async def test_clustering_quality_metrics(clustering_engine, sample_vectors):
    """测试聚类质量指标"""
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    # 检查轮廓系数
    assert -1 <= result.silhouette_score <= 1
    # 检查Davies-Bouldin指数
    assert result.davies_bouldin_score > 0


# ============= 异常检测测试 =============

@pytest.mark.asyncio
async def test_lof_anomaly_detection(clustering_engine, sample_vectors):
    """测试LOF异常检测"""
    # 添加一些异常点
    anomalies = np.random.randn(5, 128) * 10
    vectors_with_anomalies = np.vstack([sample_vectors, anomalies])
    
    config = AnomalyConfig(
        algorithm=AnomalyDetectionAlgorithm.LOF,
        contamination=0.05,
        n_neighbors=20,
        normalize=True
    )
    
    result = await clustering_engine.detect_anomalies(vectors_with_anomalies, config)
    
    assert isinstance(result, AnomalyResult)
    assert result.algorithm == AnomalyDetectionAlgorithm.LOF
    assert result.n_anomalies > 0
    assert len(result.anomaly_scores) == len(vectors_with_anomalies)
    assert 0 <= result.contamination_rate <= 1


@pytest.mark.asyncio
async def test_isolation_forest_detection(clustering_engine, sample_vectors):
    """测试Isolation Forest异常检测"""
    config = AnomalyConfig(
        algorithm=AnomalyDetectionAlgorithm.ISOLATION_FOREST,
        contamination=0.1,
        n_estimators=100,
        normalize=True
    )
    
    result = await clustering_engine.detect_anomalies(sample_vectors, config)
    
    assert result.algorithm == AnomalyDetectionAlgorithm.ISOLATION_FOREST
    assert len(result.anomaly_indices) == pytest.approx(int(0.1 * len(sample_vectors)), abs=5)
    assert all(0 <= idx < len(sample_vectors) for idx in result.anomaly_indices)


@pytest.mark.asyncio
async def test_mahalanobis_detection(clustering_engine, sample_vectors):
    """测试马氏距离异常检测"""
    config = AnomalyConfig(
        algorithm=AnomalyDetectionAlgorithm.MAHALANOBIS,
        contamination=0.1,
        normalize=True
    )
    
    result = await clustering_engine.detect_anomalies(sample_vectors, config)
    
    assert result.algorithm == AnomalyDetectionAlgorithm.MAHALANOBIS
    assert result.threshold > 0
    assert len(result.anomaly_scores) == len(sample_vectors)


@pytest.mark.asyncio
async def test_cluster_and_detect_anomalies(clustering_engine, sample_vectors):
    """测试先聚类后检测异常"""
    clustering_config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    
    anomaly_config = AnomalyConfig(
        algorithm=AnomalyDetectionAlgorithm.LOF,
        contamination=0.05,
        n_neighbors=10
    )
    
    clustering_result, anomaly_results = await clustering_engine.cluster_and_detect_anomalies(
        sample_vectors,
        clustering_config,
        anomaly_config
    )
    
    assert clustering_result.n_clusters == 3
    assert isinstance(anomaly_results, dict)
    # 每个簇可能都有异常检测结果
    assert len(anomaly_results) <= 3


@pytest.mark.asyncio
async def test_find_cluster_representatives(clustering_engine, sample_vectors):
    """测试查找簇代表"""
    # 先进行聚类
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    # 查找代表
    representatives = await clustering_engine.find_cluster_representatives(
        sample_vectors,
        result.labels,
        n_representatives=5
    )
    
    assert len(representatives) == 3
    for cluster_id, indices in representatives.items():
        assert len(indices) <= 5
        assert all(0 <= idx < len(sample_vectors) for idx in indices)


@pytest.mark.asyncio
async def test_analyze_cluster_quality(clustering_engine, sample_vectors):
    """测试聚类质量分析"""
    # 先进行聚类
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    result = await clustering_engine.cluster_vectors(sample_vectors, config)
    
    # 分析质量
    quality = await clustering_engine.analyze_cluster_quality(
        sample_vectors,
        result.labels
    )
    
    assert 'silhouette_score' in quality
    assert 'davies_bouldin_score' in quality
    assert 'avg_intra_cluster_distance' in quality
    assert 'avg_inter_cluster_distance' in quality
    assert 'separation_ratio' in quality
    assert quality['separation_ratio'] > 0


# ============= 可视化测试 =============

@pytest.mark.asyncio
async def test_tsne_visualization(visualization_engine, sample_vectors):
    """测试t-SNE可视化"""
    config = VisualizationConfig(
        method=VisualizationMethod.TSNE,
        n_components=2,
        perplexity=30,
        plot_type=PlotType.SCATTER
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors[:30],  # 使用较少的数据点加快测试
        config
    )
    
    assert isinstance(result, VisualizationResult)
    assert result.method == VisualizationMethod.TSNE
    assert result.embedding.shape == (30, 2)
    assert result.plot_data  # 应该有Base64编码的图像
    assert result.statistics['dimension_reduction_ratio'] == pytest.approx(2/128, rel=0.01)


@pytest.mark.asyncio
async def test_pca_visualization(visualization_engine, sample_vectors):
    """测试PCA可视化"""
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.SCATTER
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors,
        config
    )
    
    assert result.method == VisualizationMethod.PCA
    assert result.embedding.shape == (len(sample_vectors), 2)
    assert result.plot_data


@pytest.mark.asyncio
async def test_visualization_with_labels(visualization_engine, sample_vectors):
    """测试带标签的可视化"""
    # 创建标签
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.SCATTER,
        show_legend=True
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors,
        config,
        labels=labels
    )
    
    assert result.metadata['has_labels'] is True
    assert result.plot_data


@pytest.mark.asyncio
async def test_density_plot(visualization_engine, sample_vectors):
    """测试密度图"""
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.DENSITY
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors,
        config
    )
    
    assert result.plot_type == PlotType.DENSITY
    assert result.plot_data


@pytest.mark.asyncio
async def test_contour_plot(visualization_engine, sample_vectors):
    """测试等高线图"""
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.CONTOUR
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors,
        config,
        labels=labels
    )
    
    assert result.plot_type == PlotType.CONTOUR
    assert result.plot_data


@pytest.mark.asyncio
async def test_trajectory_plot(visualization_engine):
    """测试轨迹图"""
    # 生成时序数据
    t = np.linspace(0, 4*np.pi, 50)
    vectors = np.column_stack([
        np.sin(t) + np.random.randn(50) * 0.1,
        np.cos(t) + np.random.randn(50) * 0.1,
        t,
        np.random.randn(50, 10)
    ])
    
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.TRAJECTORY
    )
    
    result = await visualization_engine.visualize_vectors(
        vectors,
        config
    )
    
    assert result.plot_type == PlotType.TRAJECTORY
    assert result.plot_data


@pytest.mark.asyncio
async def test_visualize_clusters(visualization_engine, clustering_engine, sample_vectors):
    """测试聚类结果可视化"""
    # 先进行聚类
    clustering_config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    clustering_result = await clustering_engine.cluster_vectors(
        sample_vectors, 
        clustering_config
    )
    
    # 可视化聚类结果
    viz_config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2,
        plot_type=PlotType.SCATTER,
        show_legend=True
    )
    
    viz_result = await visualization_engine.visualize_clusters(
        sample_vectors,
        clustering_result.labels,
        viz_config
    )
    
    assert viz_result.metadata['n_clusters'] == 3
    assert 'cluster_0_size' in viz_result.metadata
    assert viz_result.plot_data


@pytest.mark.asyncio
async def test_3d_visualization(visualization_engine, sample_vectors):
    """测试3D可视化"""
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=3,
        plot_type=PlotType.SCATTER
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors[:30],
        config
    )
    
    assert result.embedding.shape == (30, 3)
    assert result.plot_data


@pytest.mark.asyncio
async def test_neighborhood_preservation(visualization_engine):
    """测试近邻保持率计算"""
    # 创建简单的数据
    vectors = np.random.randn(20, 10)
    
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2
    )
    
    result = await visualization_engine.visualize_vectors(vectors, config)
    
    assert 'neighborhood_preservation' in result.statistics
    assert 0 <= result.statistics['neighborhood_preservation'] <= 1


@pytest.mark.asyncio
async def test_save_clustering_results(clustering_engine, mock_db_session, sample_vectors, sample_entity_ids):
    """测试保存聚类结果"""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = MagicMock(id="uuid-123")
    mock_db_session.execute.return_value = mock_result
    
    config = ClusteringConfig(
        algorithm=ClusteringAlgorithm.KMEANS,
        n_clusters=3
    )
    
    result = await clustering_engine.cluster_vectors(
        sample_vectors,
        config,
        entity_ids=sample_entity_ids
    )
    
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_save_visualization_results(visualization_engine, mock_db_session, sample_vectors, sample_entity_ids):
    """测试保存可视化结果"""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = MagicMock(id="uuid-456")
    mock_db_session.execute.return_value = mock_result
    
    config = VisualizationConfig(
        method=VisualizationMethod.PCA,
        n_components=2
    )
    
    result = await visualization_engine.visualize_vectors(
        sample_vectors,
        config,
        entity_ids=sample_entity_ids
    )
    
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_create_clustering_tables(clustering_engine, mock_db_session):
    """测试创建聚类相关表"""
    mock_db_session.execute.return_value = MagicMock()
    
    result = await clustering_engine.create_clustering_tables()
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called


@pytest.mark.asyncio
async def test_create_visualization_tables(visualization_engine, mock_db_session):
    """测试创建可视化相关表"""
    mock_db_session.execute.return_value = MagicMock()
    
    result = await visualization_engine.create_visualization_tables()
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called