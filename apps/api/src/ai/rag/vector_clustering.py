"""
向量聚类与异常检测

实现K-means、DBSCAN等聚类算法以及LOF、Isolation Forest等异常检测算法
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timezone
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClusteringAlgorithm(str, Enum):
    """聚类算法类型"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    MEAN_SHIFT = "mean_shift"


class AnomalyDetectionAlgorithm(str, Enum):
    """异常检测算法类型"""
    LOF = "lof"                        # Local Outlier Factor
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    MAHALANOBIS = "mahalanobis"
    AUTOENCODER = "autoencoder"


@dataclass
class ClusteringConfig:
    """聚类配置"""
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    n_clusters: Optional[int] = None   # K-means, Hierarchical
    eps: float = 0.5                   # DBSCAN
    min_samples: int = 5               # DBSCAN
    metric: str = "euclidean"          # 距离度量
    max_iter: int = 300                # 最大迭代次数
    random_state: int = 42
    normalize: bool = True             # 是否标准化


@dataclass
class ClusteringResult:
    """聚类结果"""
    algorithm: ClusteringAlgorithm
    n_clusters: int
    labels: np.ndarray
    centers: Optional[np.ndarray]
    silhouette_score: float
    davies_bouldin_score: float
    cluster_sizes: Dict[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    algorithm: AnomalyDetectionAlgorithm = AnomalyDetectionAlgorithm.LOF
    contamination: float = 0.1         # 异常比例
    n_neighbors: int = 20              # LOF
    n_estimators: int = 100            # Isolation Forest
    threshold: float = 0.95            # 阈值
    normalize: bool = True


@dataclass
class AnomalyResult:
    """异常检测结果"""
    algorithm: AnomalyDetectionAlgorithm
    anomaly_indices: List[int]
    anomaly_scores: np.ndarray
    n_anomalies: int
    contamination_rate: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorClusteringEngine:
    """向量聚类引擎"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.scaler = StandardScaler()
        self.clustering_history = []
        self.anomaly_history = []
        
    async def cluster_vectors(
        self,
        vectors: np.ndarray,
        config: ClusteringConfig,
        entity_ids: Optional[List[str]] = None
    ) -> ClusteringResult:
        """执行向量聚类"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 数据预处理
            if config.normalize:
                vectors_normalized = self.scaler.fit_transform(vectors)
            else:
                vectors_normalized = vectors
            
            # 选择聚类算法
            if config.algorithm == ClusteringAlgorithm.KMEANS:
                result = await self._kmeans_clustering(vectors_normalized, config)
            elif config.algorithm == ClusteringAlgorithm.DBSCAN:
                result = await self._dbscan_clustering(vectors_normalized, config)
            elif config.algorithm == ClusteringAlgorithm.HIERARCHICAL:
                result = await self._hierarchical_clustering(vectors_normalized, config)
            else:
                raise ValueError(f"不支持的聚类算法: {config.algorithm}")
            
            # 计算聚类质量指标
            if result.n_clusters > 1:
                result.silhouette_score = silhouette_score(vectors_normalized, result.labels)
                result.davies_bouldin_score = davies_bouldin_score(vectors_normalized, result.labels)
            
            # 统计每个聚类的大小
            unique_labels, counts = np.unique(result.labels, return_counts=True)
            result.cluster_sizes = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            # 保存聚类结果到数据库
            if entity_ids:
                await self._save_clustering_results(entity_ids, result)
            
            # 记录历史
            end_time = asyncio.get_event_loop().time()
            result.metadata['execution_time_ms'] = (end_time - start_time) * 1000
            result.metadata['n_vectors'] = len(vectors)
            self.clustering_history.append(result)
            
            logger.info(f"聚类完成: {result.n_clusters}个簇, Silhouette: {result.silhouette_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"向量聚类失败: {e}")
            raise
    
    async def _kmeans_clustering(
        self,
        vectors: np.ndarray,
        config: ClusteringConfig
    ) -> ClusteringResult:
        """K-means聚类"""
        # 自动确定最佳聚类数（如果未指定）
        if config.n_clusters is None:
            config.n_clusters = await self._find_optimal_k(vectors, max_k=min(10, len(vectors)//2))
        
        kmeans = KMeans(
            n_clusters=config.n_clusters,
            max_iter=config.max_iter,
            random_state=config.random_state,
            n_init=10
        )
        
        labels = kmeans.fit_predict(vectors)
        
        return ClusteringResult(
            algorithm=ClusteringAlgorithm.KMEANS,
            n_clusters=config.n_clusters,
            labels=labels,
            centers=kmeans.cluster_centers_,
            silhouette_score=0.0,
            davies_bouldin_score=0.0,
            cluster_sizes={}
        )
    
    async def _dbscan_clustering(
        self,
        vectors: np.ndarray,
        config: ClusteringConfig
    ) -> ClusteringResult:
        """DBSCAN密度聚类"""
        dbscan = DBSCAN(
            eps=config.eps,
            min_samples=config.min_samples,
            metric=config.metric,
            n_jobs=-1
        )
        
        labels = dbscan.fit_predict(vectors)
        
        # 计算聚类中心（排除噪声点）
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            if label != -1:  # 排除噪声点
                mask = labels == label
                center = vectors[mask].mean(axis=0)
                centers.append(center)
        
        centers = np.array(centers) if centers else None
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        return ClusteringResult(
            algorithm=ClusteringAlgorithm.DBSCAN,
            n_clusters=n_clusters,
            labels=labels,
            centers=centers,
            silhouette_score=0.0,
            davies_bouldin_score=0.0,
            cluster_sizes={},
            metadata={'n_noise_points': np.sum(labels == -1)}
        )
    
    async def _hierarchical_clustering(
        self,
        vectors: np.ndarray,
        config: ClusteringConfig
    ) -> ClusteringResult:
        """层次聚类"""
        if config.n_clusters is None:
            config.n_clusters = await self._find_optimal_k(vectors, max_k=min(10, len(vectors)//2))
        
        clustering = AgglomerativeClustering(
            n_clusters=config.n_clusters,
            metric=config.metric,
            linkage='ward' if config.metric == 'euclidean' else 'average'
        )
        
        labels = clustering.fit_predict(vectors)
        
        # 计算聚类中心
        centers = []
        for i in range(config.n_clusters):
            mask = labels == i
            center = vectors[mask].mean(axis=0)
            centers.append(center)
        
        return ClusteringResult(
            algorithm=ClusteringAlgorithm.HIERARCHICAL,
            n_clusters=config.n_clusters,
            labels=labels,
            centers=np.array(centers),
            silhouette_score=0.0,
            davies_bouldin_score=0.0,
            cluster_sizes={}
        )
    
    async def _find_optimal_k(
        self,
        vectors: np.ndarray,
        max_k: int = 10
    ) -> int:
        """使用肘部法则找到最佳聚类数"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_k + 1, len(vectors))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(vectors, labels))
        
        # 找到Silhouette分数最高的k
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
        else:
            optimal_k = 2
        
        logger.info(f"自动确定最佳聚类数: {optimal_k}")
        return optimal_k
    
    async def detect_anomalies(
        self,
        vectors: np.ndarray,
        config: AnomalyConfig,
        entity_ids: Optional[List[str]] = None
    ) -> AnomalyResult:
        """检测异常向量"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 数据预处理
            if config.normalize:
                vectors_normalized = self.scaler.fit_transform(vectors)
            else:
                vectors_normalized = vectors
            
            # 选择异常检测算法
            if config.algorithm == AnomalyDetectionAlgorithm.LOF:
                result = await self._lof_detection(vectors_normalized, config)
            elif config.algorithm == AnomalyDetectionAlgorithm.ISOLATION_FOREST:
                result = await self._isolation_forest_detection(vectors_normalized, config)
            elif config.algorithm == AnomalyDetectionAlgorithm.MAHALANOBIS:
                result = await self._mahalanobis_detection(vectors_normalized, config)
            else:
                raise ValueError(f"不支持的异常检测算法: {config.algorithm}")
            
            # 保存异常检测结果
            if entity_ids and result.anomaly_indices:
                await self._save_anomaly_results(entity_ids, result)
            
            # 记录历史
            end_time = asyncio.get_event_loop().time()
            result.metadata['execution_time_ms'] = (end_time - start_time) * 1000
            result.metadata['n_vectors'] = len(vectors)
            self.anomaly_history.append(result)
            
            logger.info(f"异常检测完成: 发现{result.n_anomalies}个异常点")
            
            return result
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            raise
    
    async def _lof_detection(
        self,
        vectors: np.ndarray,
        config: AnomalyConfig
    ) -> AnomalyResult:
        """Local Outlier Factor异常检测"""
        lof = LocalOutlierFactor(
            n_neighbors=config.n_neighbors,
            contamination=config.contamination,
            novelty=False
        )
        
        # 预测异常（-1表示异常，1表示正常）
        predictions = lof.fit_predict(vectors)
        anomaly_scores = -lof.negative_outlier_factor_  # 分数越高越异常
        
        # 找出异常点索引
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return AnomalyResult(
            algorithm=AnomalyDetectionAlgorithm.LOF,
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            n_anomalies=len(anomaly_indices),
            contamination_rate=len(anomaly_indices) / len(vectors),
            threshold=np.percentile(anomaly_scores, (1 - config.contamination) * 100)
        )
    
    async def _isolation_forest_detection(
        self,
        vectors: np.ndarray,
        config: AnomalyConfig
    ) -> AnomalyResult:
        """Isolation Forest异常检测"""
        iso_forest = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=42
        )
        
        # 预测异常
        predictions = iso_forest.fit_predict(vectors)
        anomaly_scores = -iso_forest.score_samples(vectors)  # 分数越高越异常
        
        # 找出异常点索引
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return AnomalyResult(
            algorithm=AnomalyDetectionAlgorithm.ISOLATION_FOREST,
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            n_anomalies=len(anomaly_indices),
            contamination_rate=len(anomaly_indices) / len(vectors),
            threshold=np.percentile(anomaly_scores, (1 - config.contamination) * 100)
        )
    
    async def _mahalanobis_detection(
        self,
        vectors: np.ndarray,
        config: AnomalyConfig
    ) -> AnomalyResult:
        """马氏距离异常检测"""
        # 计算均值和协方差
        mean = np.mean(vectors, axis=0)
        cov = np.cov(vectors.T)
        
        # 添加小的正则化项避免奇异矩阵
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.inv(cov)
        
        # 计算马氏距离
        diff = vectors - mean
        mahalanobis_distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        
        # 确定异常阈值
        threshold = np.percentile(mahalanobis_distances, (1 - config.contamination) * 100)
        
        # 找出异常点
        anomaly_indices = np.where(mahalanobis_distances > threshold)[0].tolist()
        
        return AnomalyResult(
            algorithm=AnomalyDetectionAlgorithm.MAHALANOBIS,
            anomaly_indices=anomaly_indices,
            anomaly_scores=mahalanobis_distances,
            n_anomalies=len(anomaly_indices),
            contamination_rate=len(anomaly_indices) / len(vectors),
            threshold=threshold
        )
    
    async def cluster_and_detect_anomalies(
        self,
        vectors: np.ndarray,
        clustering_config: ClusteringConfig,
        anomaly_config: AnomalyConfig,
        entity_ids: Optional[List[str]] = None
    ) -> Tuple[ClusteringResult, Dict[int, AnomalyResult]]:
        """先聚类，然后在每个簇内检测异常"""
        # 执行聚类
        clustering_result = await self.cluster_vectors(vectors, clustering_config, entity_ids)
        
        # 在每个簇内检测异常
        anomaly_results = {}
        
        for cluster_id in np.unique(clustering_result.labels):
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            # 获取簇内向量
            mask = clustering_result.labels == cluster_id
            cluster_vectors = vectors[mask]
            cluster_entity_ids = [entity_ids[i] for i in range(len(mask)) if mask[i]] if entity_ids else None
            
            # 检测异常
            if len(cluster_vectors) > anomaly_config.n_neighbors:
                anomaly_result = await self.detect_anomalies(
                    cluster_vectors,
                    anomaly_config,
                    cluster_entity_ids
                )
                anomaly_results[cluster_id] = anomaly_result
        
        return clustering_result, anomaly_results
    
    async def find_cluster_representatives(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        n_representatives: int = 5
    ) -> Dict[int, List[int]]:
        """找到每个簇的代表性向量"""
        representatives = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # 跳过噪声点
                continue
            
            # 获取簇内向量
            mask = labels == cluster_id
            cluster_vectors = vectors[mask]
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_vectors) <= n_representatives:
                representatives[cluster_id] = cluster_indices.tolist()
            else:
                # 计算簇中心
                center = cluster_vectors.mean(axis=0)
                
                # 找到离中心最近的n个点
                distances = np.linalg.norm(cluster_vectors - center, axis=1)
                top_indices = np.argsort(distances)[:n_representatives]
                representatives[cluster_id] = cluster_indices[top_indices].tolist()
        
        return representatives
    
    async def analyze_cluster_quality(
        self,
        vectors: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """分析聚类质量"""
        quality_metrics = {}
        
        # 计算各种指标
        if len(np.unique(labels)) > 1:
            quality_metrics['silhouette_score'] = silhouette_score(vectors, labels)
            quality_metrics['davies_bouldin_score'] = davies_bouldin_score(vectors, labels)
        
        # 计算簇内和簇间距离
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # 排除噪声点
        
        for i, label_i in enumerate(unique_labels):
            mask_i = labels == label_i
            cluster_i = vectors[mask_i]
            
            # 簇内平均距离
            if len(cluster_i) > 1:
                intra_dist = np.mean(cdist(cluster_i, cluster_i))
                intra_cluster_distances.append(intra_dist)
            
            # 簇间平均距离
            for j, label_j in enumerate(unique_labels[i+1:], i+1):
                mask_j = labels == label_j
                cluster_j = vectors[mask_j]
                inter_dist = np.mean(cdist(cluster_i, cluster_j))
                inter_cluster_distances.append(inter_dist)
        
        quality_metrics['avg_intra_cluster_distance'] = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        quality_metrics['avg_inter_cluster_distance'] = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        quality_metrics['separation_ratio'] = (
            quality_metrics['avg_inter_cluster_distance'] / (quality_metrics['avg_intra_cluster_distance'] + 1e-6)
            if quality_metrics['avg_intra_cluster_distance'] > 0 else 0
        )
        
        # 计算簇大小分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        quality_metrics['cluster_size_std'] = np.std(counts[unique_labels != -1]) if len(counts[unique_labels != -1]) > 0 else 0
        quality_metrics['n_noise_points'] = counts[unique_labels == -1][0] if -1 in unique_labels else 0
        
        return quality_metrics
    
    async def _save_clustering_results(
        self,
        entity_ids: List[str],
        result: ClusteringResult
    ) -> None:
        """保存聚类结果到数据库"""
        try:
            # 创建聚类记录
            insert_sql = """
            INSERT INTO clustering_results 
            (algorithm, n_clusters, silhouette_score, davies_bouldin_score, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            clustering_result = await self.db.execute(
                text(insert_sql),
                (
                    result.algorithm.value,
                    result.n_clusters,
                    float(result.silhouette_score),
                    float(result.davies_bouldin_score),
                    json.dumps(result.metadata),
                    datetime.now(timezone.utc)
                )
            )
            
            clustering_id = clustering_result.fetchone().id
            
            # 保存每个实体的聚类标签
            for entity_id, label in zip(entity_ids, result.labels):
                await self.db.execute(
                    text("""
                    INSERT INTO entity_clusters 
                    (entity_id, clustering_id, cluster_label)
                    VALUES (%s, %s, %s)
                    """),
                    (entity_id, clustering_id, int(label))
                )
            
            await self.db.commit()
            logger.info(f"聚类结果已保存: clustering_id={clustering_id}")
            
        except Exception as e:
            logger.error(f"保存聚类结果失败: {e}")
            await self.db.rollback()
    
    async def _save_anomaly_results(
        self,
        entity_ids: List[str],
        result: AnomalyResult
    ) -> None:
        """保存异常检测结果到数据库"""
        try:
            # 创建异常检测记录
            insert_sql = """
            INSERT INTO anomaly_detection_results 
            (algorithm, n_anomalies, contamination_rate, threshold, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            detection_result = await self.db.execute(
                text(insert_sql),
                (
                    result.algorithm.value,
                    result.n_anomalies,
                    float(result.contamination_rate),
                    float(result.threshold),
                    json.dumps(result.metadata),
                    datetime.now(timezone.utc)
                )
            )
            
            detection_id = detection_result.fetchone().id
            
            # 保存异常实体
            for idx in result.anomaly_indices:
                await self.db.execute(
                    text("""
                    INSERT INTO entity_anomalies 
                    (entity_id, detection_id, anomaly_score)
                    VALUES (%s, %s, %s)
                    """),
                    (entity_ids[idx], detection_id, float(result.anomaly_scores[idx]))
                )
            
            await self.db.commit()
            logger.info(f"异常检测结果已保存: detection_id={detection_id}")
            
        except Exception as e:
            logger.error(f"保存异常检测结果失败: {e}")
            await self.db.rollback()
    
    async def create_clustering_tables(self) -> bool:
        """创建聚类相关表"""
        try:
            # 聚类结果表
            create_clustering_table = """
            CREATE TABLE IF NOT EXISTS clustering_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                algorithm VARCHAR(50) NOT NULL,
                n_clusters INTEGER NOT NULL,
                silhouette_score FLOAT,
                davies_bouldin_score FLOAT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 实体聚类标签表
            CREATE TABLE IF NOT EXISTS entity_clusters (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id VARCHAR(255) NOT NULL,
                clustering_id UUID REFERENCES clustering_results(id),
                cluster_label INTEGER NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 异常检测结果表
            CREATE TABLE IF NOT EXISTS anomaly_detection_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                algorithm VARCHAR(50) NOT NULL,
                n_anomalies INTEGER NOT NULL,
                contamination_rate FLOAT,
                threshold FLOAT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 实体异常表
            CREATE TABLE IF NOT EXISTS entity_anomalies (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id VARCHAR(255) NOT NULL,
                detection_id UUID REFERENCES anomaly_detection_results(id),
                anomaly_score FLOAT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_entity_clusters_entity_id 
            ON entity_clusters(entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_entity_clusters_clustering_id 
            ON entity_clusters(clustering_id);
            
            CREATE INDEX IF NOT EXISTS idx_entity_anomalies_entity_id 
            ON entity_anomalies(entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_entity_anomalies_detection_id 
            ON entity_anomalies(detection_id);
            """
            
            await self.db.execute(text(create_clustering_table))
            await self.db.commit()
            
            logger.info("聚类相关表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建聚类表失败: {e}")
            await self.db.rollback()
            return False