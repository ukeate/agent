"""
向量可视化

实现t-SNE、UMAP等降维可视化方法
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime, timezone
import json
import base64
from io import BytesIO
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# 尝试导入UMAP（可选依赖）
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger = logging.getLogger(__name__)
    logger.warning("UMAP未安装，部分功能将不可用")

logger = logging.getLogger(__name__)


class VisualizationMethod(str, Enum):
    """可视化方法"""
    TSNE = "tsne"
    UMAP = "umap"
    PCA = "pca"
    MDS = "mds"  # Multidimensional Scaling
    ISOMAP = "isomap"


class PlotType(str, Enum):
    """图表类型"""
    SCATTER = "scatter"
    DENSITY = "density"
    CONTOUR = "contour"
    HEATMAP = "heatmap"
    TRAJECTORY = "trajectory"


@dataclass
class VisualizationConfig:
    """可视化配置"""
    method: VisualizationMethod = VisualizationMethod.TSNE
    n_components: int = 2  # 降维到的维度（2D或3D）
    perplexity: int = 30  # t-SNE参数
    n_neighbors: int = 15  # UMAP参数
    min_dist: float = 0.1  # UMAP参数
    random_state: int = 42
    plot_type: PlotType = PlotType.SCATTER
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    color_map: str = "viridis"
    show_legend: bool = True
    show_labels: bool = False
    alpha: float = 0.7


@dataclass
class VisualizationResult:
    """可视化结果"""
    method: VisualizationMethod
    embedding: np.ndarray  # 降维后的坐标
    plot_data: str  # Base64编码的图像
    plot_type: PlotType
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)


class VectorVisualizationEngine:
    """向量可视化引擎"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.visualization_cache = {}
        
    async def visualize_vectors(
        self,
        vectors: np.ndarray,
        config: VisualizationConfig,
        labels: Optional[np.ndarray] = None,
        entity_ids: Optional[List[str]] = None
    ) -> VisualizationResult:
        """可视化高维向量"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 降维
            embedding = await self._reduce_dimensions(vectors, config)
            
            # 生成可视化
            plot_data = await self._create_plot(
                embedding, config, labels, entity_ids
            )
            
            # 计算统计信息
            statistics = await self._compute_statistics(vectors, embedding)
            
            # 创建结果
            result = VisualizationResult(
                method=config.method,
                embedding=embedding,
                plot_data=plot_data,
                plot_type=config.plot_type,
                metadata={
                    'n_vectors': len(vectors),
                    'original_dim': vectors.shape[1],
                    'reduced_dim': embedding.shape[1],
                    'has_labels': labels is not None
                },
                statistics=statistics
            )
            
            # 保存可视化结果
            if entity_ids:
                await self._save_visualization(entity_ids, result)
            
            end_time = asyncio.get_event_loop().time()
            result.metadata['execution_time_ms'] = (end_time - start_time) * 1000
            
            logger.info(f"可视化完成: {config.method.value}, {len(vectors)}个向量")
            
            return result
            
        except Exception as e:
            logger.error(f"向量可视化失败: {e}")
            raise
    
    async def _reduce_dimensions(
        self,
        vectors: np.ndarray,
        config: VisualizationConfig
    ) -> np.ndarray:
        """降维处理"""
        if config.method == VisualizationMethod.TSNE:
            return await self._tsne_reduction(vectors, config)
        elif config.method == VisualizationMethod.UMAP:
            return await self._umap_reduction(vectors, config)
        elif config.method == VisualizationMethod.PCA:
            return await self._pca_reduction(vectors, config)
        else:
            raise ValueError(f"不支持的可视化方法: {config.method}")
    
    async def _tsne_reduction(
        self,
        vectors: np.ndarray,
        config: VisualizationConfig
    ) -> np.ndarray:
        """t-SNE降维"""
        tsne = TSNE(
            n_components=config.n_components,
            perplexity=min(config.perplexity, len(vectors) - 1),
            random_state=config.random_state,
            max_iter=1000,  # 使用max_iter替代n_iter
            learning_rate='auto',
            init='pca'
        )
        
        embedding = tsne.fit_transform(vectors)
        return embedding
    
    async def _umap_reduction(
        self,
        vectors: np.ndarray,
        config: VisualizationConfig
    ) -> np.ndarray:
        """UMAP降维"""
        if not HAS_UMAP:
            logger.warning("UMAP未安装，改用t-SNE")
            return await self._tsne_reduction(vectors, config)
        
        umap = UMAP(
            n_components=config.n_components,
            n_neighbors=min(config.n_neighbors, len(vectors) - 1),
            min_dist=config.min_dist,
            random_state=config.random_state
        )
        
        embedding = umap.fit_transform(vectors)
        return embedding
    
    async def _pca_reduction(
        self,
        vectors: np.ndarray,
        config: VisualizationConfig
    ) -> np.ndarray:
        """PCA降维"""
        pca = PCA(
            n_components=config.n_components,
            random_state=config.random_state
        )
        
        embedding = pca.fit_transform(vectors)
        return embedding
    
    async def _create_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig,
        labels: Optional[np.ndarray] = None,
        entity_ids: Optional[List[str]] = None
    ) -> str:
        """创建可视化图表"""
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        if config.plot_type == PlotType.SCATTER:
            await self._create_scatter_plot(embedding, config, labels, entity_ids)
        elif config.plot_type == PlotType.DENSITY:
            await self._create_density_plot(embedding, config)
        elif config.plot_type == PlotType.CONTOUR:
            await self._create_contour_plot(embedding, config, labels)
        elif config.plot_type == PlotType.HEATMAP:
            await self._create_heatmap_plot(embedding, config)
        elif config.plot_type == PlotType.TRAJECTORY:
            await self._create_trajectory_plot(embedding, config)
        else:
            await self._create_scatter_plot(embedding, config, labels, entity_ids)
        
        # 设置标题和标签
        plt.title(f'{config.method.value.upper()} Visualization', fontsize=14)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        
        # 保存为Base64编码的图像
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        return plot_data
    
    async def _create_scatter_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig,
        labels: Optional[np.ndarray] = None,
        entity_ids: Optional[List[str]] = None
    ):
        """创建散点图"""
        if config.n_components == 2:
            x, y = embedding[:, 0], embedding[:, 1]
            
            if labels is not None:
                # 有标签的散点图
                unique_labels = np.unique(labels)
                colors = plt.cm.get_cmap(config.color_map, len(unique_labels))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    label_name = f'Cluster {label}' if label != -1 else 'Noise'
                    plt.scatter(
                        x[mask], y[mask],
                        c=[colors(i)],
                        label=label_name,
                        alpha=config.alpha,
                        s=50
                    )
                
                if config.show_legend:
                    plt.legend(loc='best')
            else:
                # 无标签的散点图
                plt.scatter(x, y, alpha=config.alpha, s=50, c='blue')
            
            # 添加标签
            if config.show_labels and entity_ids:
                for i, entity_id in enumerate(entity_ids[:20]):  # 只显示前20个标签
                    plt.annotate(
                        entity_id[:10],  # 截断长标签
                        (x[i], y[i]),
                        fontsize=8,
                        alpha=0.7
                    )
        
        elif config.n_components == 3:
            # 3D散点图
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.axes(projection='3d')
            
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.get_cmap(config.color_map, len(unique_labels))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        embedding[mask, 2],
                        c=[colors(i)],
                        label=f'Cluster {label}' if label != -1 else 'Noise',
                        alpha=config.alpha,
                        s=50
                    )
            else:
                ax.scatter(
                    embedding[:, 0],
                    embedding[:, 1],
                    embedding[:, 2],
                    alpha=config.alpha,
                    s=50
                )
            
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
    
    async def _create_density_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig
    ):
        """创建密度图"""
        if config.n_components != 2:
            logger.warning("密度图只支持2D可视化")
            return
        
        x, y = embedding[:, 0], embedding[:, 1]
        
        # 使用hexbin创建密度图
        plt.hexbin(x, y, gridsize=30, cmap=config.color_map, alpha=config.alpha)
        plt.colorbar(label='Count')
    
    async def _create_contour_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig,
        labels: Optional[np.ndarray] = None
    ):
        """创建等高线图"""
        if config.n_components != 2:
            logger.warning("等高线图只支持2D可视化")
            return
        
        x, y = embedding[:, 0], embedding[:, 1]
        
        # 计算2D直方图
        from scipy.stats import gaussian_kde
        
        if labels is not None:
            # 为每个簇创建等高线
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap(config.color_map, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                if label == -1:  # 跳过噪声点
                    continue
                
                mask = labels == label
                if np.sum(mask) < 3:  # 需要至少3个点
                    continue
                
                x_cluster = x[mask]
                y_cluster = y[mask]
                
                # 计算核密度估计
                xy = np.vstack([x_cluster, y_cluster])
                z = gaussian_kde(xy)(xy)
                
                # 绘制等高线
                plt.tricontour(x_cluster, y_cluster, z, levels=5, colors=[colors(i)], alpha=config.alpha)
                
                # 绘制散点
                plt.scatter(x_cluster, y_cluster, c=[colors(i)], s=20, alpha=0.5)
        else:
            # 整体等高线
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            plt.tricontourf(x, y, z, levels=10, cmap=config.color_map, alpha=config.alpha)
            plt.colorbar(label='Density')
    
    async def _create_heatmap_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig
    ):
        """创建热力图"""
        if config.n_components != 2:
            logger.warning("热力图只支持2D可视化")
            return
        
        x, y = embedding[:, 0], embedding[:, 1]
        
        # 创建2D直方图
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        plt.imshow(
            heatmap.T,
            extent=extent,
            origin='lower',
            cmap=config.color_map,
            aspect='auto',
            interpolation='gaussian'
        )
        plt.colorbar(label='Density')
    
    async def _create_trajectory_plot(
        self,
        embedding: np.ndarray,
        config: VisualizationConfig
    ):
        """创建轨迹图（适用于时序数据）"""
        if config.n_components != 2:
            logger.warning("轨迹图只支持2D可视化")
            return
        
        x, y = embedding[:, 0], embedding[:, 1]
        
        # 绘制轨迹线
        plt.plot(x, y, '-', alpha=0.5, linewidth=1, color='blue')
        
        # 绘制起点和终点
        plt.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(x[-1], y[-1], c='red', s=100, marker='s', label='End', zorder=5)
        
        # 绘制中间点
        plt.scatter(x[1:-1], y[1:-1], c=range(1, len(x)-1), cmap=config.color_map, 
                   s=30, alpha=config.alpha, zorder=4)
        
        if config.show_legend:
            plt.legend()
        
        # 添加箭头表示方向
        for i in range(0, len(x)-1, max(1, len(x)//10)):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            plt.arrow(x[i], y[i], dx*0.8, dy*0.8, 
                     head_width=0.05, head_length=0.05, 
                     fc='gray', ec='gray', alpha=0.5)
    
    async def _compute_statistics(
        self,
        original_vectors: np.ndarray,
        embedding: np.ndarray
    ) -> Dict[str, float]:
        """计算降维统计信息"""
        stats = {}
        
        # 计算降维比例
        stats['dimension_reduction_ratio'] = embedding.shape[1] / original_vectors.shape[1]
        
        # 计算嵌入空间的范围
        stats['embedding_range_x'] = float(np.ptp(embedding[:, 0]))
        stats['embedding_range_y'] = float(np.ptp(embedding[:, 1]))
        
        # 计算嵌入的分散程度
        stats['embedding_std_x'] = float(np.std(embedding[:, 0]))
        stats['embedding_std_y'] = float(np.std(embedding[:, 1]))
        
        # 计算最近邻保持率（采样计算）
        if len(original_vectors) <= 1000:
            preservation_rate = await self._compute_neighborhood_preservation(
                original_vectors, embedding, k=10
            )
            stats['neighborhood_preservation'] = preservation_rate
        
        return stats
    
    async def _compute_neighborhood_preservation(
        self,
        original: np.ndarray,
        embedding: np.ndarray,
        k: int = 10
    ) -> float:
        """计算近邻保持率"""
        from sklearn.neighbors import NearestNeighbors
        
        k = min(k, len(original) - 1)
        
        # 原始空间的k近邻
        nn_original = NearestNeighbors(n_neighbors=k+1)
        nn_original.fit(original)
        neighbors_original = nn_original.kneighbors(return_distance=False)[:, 1:]
        
        # 嵌入空间的k近邻
        nn_embedding = NearestNeighbors(n_neighbors=k+1)
        nn_embedding.fit(embedding)
        neighbors_embedding = nn_embedding.kneighbors(return_distance=False)[:, 1:]
        
        # 计算保持率
        preservation = 0
        for i in range(len(original)):
            intersection = len(set(neighbors_original[i]) & set(neighbors_embedding[i]))
            preservation += intersection / k
        
        return preservation / len(original)
    
    async def visualize_clusters(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        config: VisualizationConfig
    ) -> VisualizationResult:
        """可视化聚类结果"""
        # 降维
        embedding = await self._reduce_dimensions(vectors, config)
        
        # 创建增强的聚类可视化
        plt.figure(figsize=config.figure_size, dpi=config.dpi)
        
        if config.n_components == 2:
            x, y = embedding[:, 0], embedding[:, 1]
            
            # 绘制每个簇
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap(config.color_map, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                cluster_points = embedding[mask]
                
                if label == -1:
                    # 噪声点用黑色标记
                    plt.scatter(x[mask], y[mask], c='black', s=20, 
                              alpha=0.3, label='Noise')
                else:
                    # 正常簇
                    plt.scatter(x[mask], y[mask], c=[colors(i)], s=50,
                              alpha=config.alpha, label=f'Cluster {label}')
                    
                    # 绘制簇的凸包
                    if len(cluster_points) >= 3:
                        try:
                            hull = ConvexHull(cluster_points)
                            for simplex in hull.simplices:
                                plt.plot(cluster_points[simplex, 0], 
                                       cluster_points[simplex, 1], 
                                       color=colors(i), alpha=0.2, linewidth=1)
                        except:
                            pass
                    
                    # 标记簇中心
                    center = cluster_points.mean(axis=0)
                    plt.scatter(center[0], center[1], c='red', s=200, 
                              marker='x', linewidths=2)
                    plt.annotate(f'C{label}', (center[0], center[1]), 
                               fontsize=12, fontweight='bold')
            
            if config.show_legend:
                plt.legend(loc='best')
        
        plt.title(f'Cluster Visualization ({len(unique_labels)} clusters)', fontsize=14)
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        
        # 保存图像
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        # 计算聚类统计
        cluster_stats = {}
        for label in unique_labels:
            if label != -1:
                cluster_size = np.sum(labels == label)
                cluster_stats[f'cluster_{label}_size'] = int(cluster_size)
        
        return VisualizationResult(
            method=config.method,
            embedding=embedding,
            plot_data=plot_data,
            plot_type=PlotType.SCATTER,
            metadata={
                'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
                'has_noise': -1 in unique_labels,
                **cluster_stats
            }
        )
    
    async def _save_visualization(
        self,
        entity_ids: List[str],
        result: VisualizationResult
    ) -> None:
        """保存可视化结果"""
        try:
            # 保存可视化记录
            insert_sql = """
            INSERT INTO visualization_results 
            (method, plot_type, plot_data, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            
            viz_result = await self.db.execute(
                text(insert_sql),
                (
                    result.method.value,
                    result.plot_type.value,
                    result.plot_data,
                    json.dumps(result.metadata),
                    datetime.now(timezone.utc)
                )
            )
            
            viz_id = viz_result.fetchone().id
            
            # 保存实体嵌入坐标
            for i, entity_id in enumerate(entity_ids):
                await self.db.execute(
                    text("""
                    INSERT INTO entity_embeddings 
                    (entity_id, visualization_id, x, y, z)
                    VALUES (%s, %s, %s, %s, %s)
                    """),
                    (
                        entity_id,
                        viz_id,
                        float(result.embedding[i, 0]),
                        float(result.embedding[i, 1]),
                        float(result.embedding[i, 2]) if result.embedding.shape[1] > 2 else None
                    )
                )
            
            await self.db.commit()
            logger.info(f"可视化结果已保存: viz_id={viz_id}")
            
        except Exception as e:
            logger.error(f"保存可视化结果失败: {e}")
            await self.db.rollback()
    
    async def create_visualization_tables(self) -> bool:
        """创建可视化相关表"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS visualization_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                method VARCHAR(50) NOT NULL,
                plot_type VARCHAR(50) NOT NULL,
                plot_data TEXT,  -- Base64编码的图像
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 实体嵌入坐标表
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id VARCHAR(255) NOT NULL,
                visualization_id UUID REFERENCES visualization_results(id),
                x FLOAT NOT NULL,
                y FLOAT NOT NULL,
                z FLOAT,  -- 3D可视化时使用
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_entity_embeddings_entity_id 
            ON entity_embeddings(entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_entity_embeddings_viz_id 
            ON entity_embeddings(visualization_id);
            """
            
            await self.db.execute(text(create_table_sql))
            await self.db.commit()
            
            logger.info("可视化相关表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建可视化表失败: {e}")
            await self.db.rollback()
            return False