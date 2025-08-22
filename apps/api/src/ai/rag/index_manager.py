"""
高级向量索引管理器

支持多种索引类型（HNSW、IVF、LSH、Annoy等）和动态切换
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timezone
import json
import hashlib
import psutil
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """索引类型枚举"""
    HNSW = "hnsw"           # 分层导航小世界图
    IVF = "ivf"             # 倒排文件索引
    LSH = "lsh"             # 局部敏感哈希
    FLAT = "flat"           # 暴力搜索
    ANNOY = "annoy"         # Spotify的近似最近邻库


class DistanceMetric(str, Enum):
    """距离度量类型"""
    COSINE = "cosine"                  # 余弦相似度
    EUCLIDEAN = "euclidean"            # 欧氏距离
    MANHATTAN = "manhattan"            # 曼哈顿距离
    DOT_PRODUCT = "dot_product"        # 点积
    HAMMING = "hamming"                # 汉明距离


@dataclass
class IndexConfig:
    """索引配置"""
    index_type: IndexType = IndexType.HNSW
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # HNSW参数
    hnsw_m: int = 16                       # 连接数
    hnsw_ef_construction: int = 200        # 构建时的动态列表大小
    hnsw_ef_search: int = 100              # 搜索时的动态列表大小
    
    # IVF参数
    ivf_lists: int = 1000                  # 聚类数
    ivf_probes: int = 10                   # 探测数
    
    # LSH参数
    lsh_hash_tables: int = 10              # 哈希表数量
    lsh_hash_bits: int = 128               # 哈希位数
    
    # 性能参数
    enable_quantization: bool = False       # 是否启用量化
    use_gpu: bool = False                   # 是否使用GPU加速
    parallel_workers: int = 4               # 并行工作线程数


@dataclass
class IndexStats:
    """索引统计信息"""
    index_type: IndexType
    total_vectors: int
    dimension: int
    build_time_ms: float
    memory_usage_mb: float
    last_updated: datetime


class AdvancedIndexManager:
    """高级向量索引管理器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.index_configs: Dict[str, IndexConfig] = {}
        self.index_stats: Dict[str, IndexStats] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
    async def create_hnsw_index(
        self, 
        table_name: str, 
        vector_column: str,
        config: IndexConfig
    ) -> bool:
        """创建HNSW索引"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 确定操作符类
            ops_class = self._get_ops_class(config.distance_metric)
            
            # 创建HNSW索引
            index_name = f"idx_{table_name}_{vector_column}_hnsw"
            index_sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
            ON {table_name} 
            USING hnsw ({vector_column} {ops_class})
            WITH (
                m = {config.hnsw_m}, 
                ef_construction = {config.hnsw_ef_construction}
            );
            """
            
            await self.db.execute(text(index_sql))
            await self.db.commit()
            
            # 记录统计信息
            end_time = asyncio.get_event_loop().time()
            build_time = (end_time - start_time) * 1000
            
            await self._update_index_stats(
                index_name, IndexType.HNSW, table_name, build_time
            )
            
            logger.info(f"HNSW索引 {index_name} 创建成功，耗时 {build_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"创建HNSW索引失败: {e}")
            await self.db.rollback()
            return False
    
    async def create_ivf_index(
        self, 
        table_name: str, 
        vector_column: str,
        config: IndexConfig
    ) -> bool:
        """创建IVF索引"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 确定操作符类
            ops_class = self._get_ops_class(config.distance_metric)
            
            # 创建IVF索引
            index_name = f"idx_{table_name}_{vector_column}_ivf"
            index_sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
            ON {table_name} 
            USING ivfflat ({vector_column} {ops_class})
            WITH (lists = {config.ivf_lists});
            """
            
            await self.db.execute(text(index_sql))
            await self.db.commit()
            
            # 记录统计信息
            end_time = asyncio.get_event_loop().time()
            build_time = (end_time - start_time) * 1000
            
            await self._update_index_stats(
                index_name, IndexType.IVF, table_name, build_time
            )
            
            logger.info(f"IVF索引 {index_name} 创建成功，耗时 {build_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"创建IVF索引失败: {e}")
            await self.db.rollback()
            return False
    
    async def create_lsh_index(
        self, 
        table_name: str, 
        vector_column: str,
        config: IndexConfig
    ) -> bool:
        """创建LSH索引（通过额外的哈希列实现）"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # LSH需要创建额外的哈希列
            for i in range(config.lsh_hash_tables):
                hash_column = f"{vector_column}_lsh_{i}"
                
                # 添加哈希列（如果不存在）
                add_column_sql = f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND column_name = '{hash_column}'
                    ) THEN
                        ALTER TABLE {table_name} 
                        ADD COLUMN {hash_column} BIT({config.lsh_hash_bits});
                    END IF;
                END $$;
                """
                await self.db.execute(text(add_column_sql))
                
                # 创建哈希索引
                index_name = f"idx_{table_name}_{hash_column}"
                index_sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
                ON {table_name} USING hash ({hash_column});
                """
                await self.db.execute(text(index_sql))
            
            await self.db.commit()
            
            # 记录统计信息
            end_time = asyncio.get_event_loop().time()
            build_time = (end_time - start_time) * 1000
            
            logger.info(f"LSH索引创建成功，包含 {config.lsh_hash_tables} 个哈希表，耗时 {build_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"创建LSH索引失败: {e}")
            await self.db.rollback()
            return False
    
    async def create_annoy_index(
        self, 
        table_name: str, 
        vector_column: str,
        config: IndexConfig
    ) -> bool:
        """创建Annoy索引（通过Python库在应用层实现）"""
        try:
            # Annoy索引在应用层维护，这里只记录配置
            index_name = f"idx_{table_name}_{vector_column}_annoy"
            
            # 存储配置到元数据表
            config_sql = """
            CREATE TABLE IF NOT EXISTS index_metadata (
                index_name VARCHAR(255) PRIMARY KEY,
                table_name VARCHAR(255),
                vector_column VARCHAR(255),
                index_type VARCHAR(50),
                config JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            INSERT INTO index_metadata (index_name, table_name, vector_column, index_type, config)
            VALUES (:index_name, :table_name, :vector_column, :index_type, :config)
            ON CONFLICT (index_name) 
            DO UPDATE SET config = EXCLUDED.config;
            """
            
            await self.db.execute(
                text(config_sql),
                {
                    "index_name": index_name,
                    "table_name": table_name,
                    "vector_column": vector_column,
                    "index_type": IndexType.ANNOY.value,
                    "config": json.dumps({
                        "distance_metric": config.distance_metric.value,
                        "n_trees": 100  # Annoy特定参数
                    })
                }
            )
            await self.db.commit()
            
            logger.info(f"Annoy索引配置 {index_name} 已保存")
            return True
            
        except Exception as e:
            logger.error(f"创建Annoy索引失败: {e}")
            await self.db.rollback()
            return False
    
    async def switch_index_type(
        self,
        table_name: str,
        vector_column: str,
        new_type: IndexType,
        config: Optional[IndexConfig] = None
    ) -> bool:
        """动态切换索引类型"""
        try:
            logger.info(f"切换索引类型到 {new_type}")
            
            # 使用默认配置或提供的配置
            if config is None:
                config = IndexConfig(index_type=new_type)
            else:
                config.index_type = new_type
            
            # 删除旧索引
            await self._drop_existing_indexes(table_name, vector_column)
            
            # 创建新索引
            if new_type == IndexType.HNSW:
                success = await self.create_hnsw_index(table_name, vector_column, config)
            elif new_type == IndexType.IVF:
                success = await self.create_ivf_index(table_name, vector_column, config)
            elif new_type == IndexType.LSH:
                success = await self.create_lsh_index(table_name, vector_column, config)
            elif new_type == IndexType.ANNOY:
                success = await self.create_annoy_index(table_name, vector_column, config)
            else:  # FLAT
                logger.info("切换到FLAT索引（无需创建特定索引）")
                success = True
            
            if success:
                # 保存配置
                self.index_configs[f"{table_name}.{vector_column}"] = config
                logger.info(f"成功切换到 {new_type} 索引")
            
            return success
            
        except Exception as e:
            logger.error(f"切换索引类型失败: {e}")
            return False
    
    async def analyze_and_recommend_index(
        self,
        table_name: str,
        vector_column: str,
        sample_size: int = 10000
    ) -> Tuple[IndexType, IndexConfig]:
        """分析数据特征并推荐最佳索引类型"""
        try:
            # 获取数据统计
            stats = await self._analyze_data_characteristics(
                table_name, vector_column, sample_size
            )
            
            # 基于数据特征推荐索引
            recommended_type = self._recommend_index_type(stats)
            recommended_config = self._generate_optimal_config(recommended_type, stats)
            
            logger.info(f"推荐索引类型: {recommended_type}")
            return recommended_type, recommended_config
            
        except Exception as e:
            logger.error(f"分析推荐失败: {e}")
            return IndexType.HNSW, IndexConfig()
    
    async def _analyze_data_characteristics(
        self,
        table_name: str,
        vector_column: str,
        sample_size: int
    ) -> Dict[str, Any]:
        """分析数据特征"""
        stats = {
            "total_vectors": 0,
            "dimension": 0,
            "density": 0.0,
            "distribution": "unknown",
            "avg_distance": 0.0
        }
        
        # 获取总行数
        count_sql = f"SELECT COUNT(*) as count FROM {table_name}"
        result = await self.db.execute(text(count_sql))
        row = result.fetchone()
        stats["total_vectors"] = row.count if row else 0
        
        # 采样分析
        sample_sql = f"""
        SELECT {vector_column} 
        FROM {table_name}
        TABLESAMPLE SYSTEM (1)
        LIMIT {sample_size}
        """
        
        result = await self.db.execute(text(sample_sql))
        samples = result.fetchall()
        
        if samples:
            # 假设向量以数组形式存储
            vectors = []
            for sample in samples:
                vec = sample[0]
                if vec:
                    vectors.append(np.array(vec))
            
            if vectors:
                vectors = np.array(vectors)
                stats["dimension"] = vectors.shape[1]
                
                # 计算密度（平均距离）
                if len(vectors) > 1:
                    distances = []
                    for i in range(min(100, len(vectors))):
                        for j in range(i+1, min(i+10, len(vectors))):
                            dist = np.linalg.norm(vectors[i] - vectors[j])
                            distances.append(dist)
                    stats["avg_distance"] = np.mean(distances) if distances else 0
                    stats["density"] = 1.0 / (stats["avg_distance"] + 1e-6)
                
                # 分析分布
                variances = np.var(vectors, axis=0)
                if np.std(variances) < 0.1:
                    stats["distribution"] = "uniform"
                elif np.max(variances) / (np.min(variances) + 1e-6) > 10:
                    stats["distribution"] = "skewed"
                else:
                    stats["distribution"] = "normal"
        
        return stats
    
    def _recommend_index_type(self, stats: Dict[str, Any]) -> IndexType:
        """基于数据特征推荐索引类型"""
        total_vectors = stats["total_vectors"]
        dimension = stats["dimension"]
        distribution = stats["distribution"]
        
        # 基于数据规模的推荐
        if total_vectors < 10000:
            # 小数据集用暴力搜索
            return IndexType.FLAT
        elif total_vectors < 100000:
            # 中等数据集
            if dimension < 100:
                return IndexType.IVF  # 低维用IVF
            else:
                return IndexType.HNSW  # 高维用HNSW
        elif total_vectors < 1000000:
            # 大数据集
            if distribution == "uniform":
                return IndexType.IVF  # 均匀分布用IVF
            else:
                return IndexType.HNSW  # 其他分布用HNSW
        else:
            # 超大数据集
            if dimension > 500:
                return IndexType.LSH  # 超高维用LSH
            else:
                return IndexType.HNSW  # 否则用HNSW
    
    def _generate_optimal_config(
        self, 
        index_type: IndexType, 
        stats: Dict[str, Any]
    ) -> IndexConfig:
        """生成最优配置"""
        config = IndexConfig(index_type=index_type)
        
        total_vectors = stats["total_vectors"]
        dimension = stats["dimension"]
        
        if index_type == IndexType.HNSW:
            # 根据数据规模调整HNSW参数
            if total_vectors < 100000:
                config.hnsw_m = 16
                config.hnsw_ef_construction = 200
            elif total_vectors < 1000000:
                config.hnsw_m = 32
                config.hnsw_ef_construction = 400
            else:
                config.hnsw_m = 48
                config.hnsw_ef_construction = 600
            
            config.hnsw_ef_search = config.hnsw_m * 2
            
        elif index_type == IndexType.IVF:
            # 根据数据规模调整IVF参数
            config.ivf_lists = min(int(np.sqrt(total_vectors)), 4096)
            config.ivf_probes = max(10, config.ivf_lists // 100)
            
        elif index_type == IndexType.LSH:
            # 根据维度调整LSH参数
            config.lsh_hash_tables = min(20, max(5, dimension // 50))
            config.lsh_hash_bits = min(256, max(64, dimension * 2))
        
        # 性能优化设置
        if total_vectors > 1000000:
            config.enable_quantization = True
        
        # 并行设置
        config.parallel_workers = min(psutil.cpu_count(), 8)
        
        return config
    
    def _get_ops_class(self, metric: DistanceMetric) -> str:
        """获取距离度量对应的操作符类"""
        mapping = {
            DistanceMetric.COSINE: "vector_cosine_ops",
            DistanceMetric.EUCLIDEAN: "vector_l2_ops",
            DistanceMetric.DOT_PRODUCT: "vector_ip_ops",
            DistanceMetric.MANHATTAN: "vector_l1_ops",
            DistanceMetric.HAMMING: "bit_hamming_ops"
        }
        return mapping.get(metric, "vector_l2_ops")
    
    async def _drop_existing_indexes(
        self, 
        table_name: str, 
        vector_column: str
    ) -> None:
        """删除现有索引"""
        try:
            # 查找所有相关索引
            find_indexes_sql = f"""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = '{table_name}' 
            AND indexdef LIKE '%{vector_column}%'
            """
            
            result = await self.db.execute(text(find_indexes_sql))
            indexes = result.fetchall()
            
            # 删除每个索引
            for index in indexes:
                drop_sql = f"DROP INDEX IF EXISTS {index.indexname}"
                await self.db.execute(text(drop_sql))
            
            await self.db.commit()
            logger.info(f"删除了 {len(indexes)} 个现有索引")
            
        except Exception as e:
            logger.warning(f"删除索引时出错: {e}")
            await self.db.rollback()
    
    async def _update_index_stats(
        self,
        index_name: str,
        index_type: IndexType,
        table_name: str,
        build_time: float
    ) -> None:
        """更新索引统计信息"""
        try:
            # 获取索引大小
            size_sql = f"""
            SELECT pg_relation_size(indexrelid) as size
            FROM pg_stat_user_indexes
            WHERE indexrelname = '{index_name}'
            """
            
            result = await self.db.execute(text(size_sql))
            row = result.fetchone()
            memory_usage_mb = (row.size / 1024 / 1024) if row and row.size else 0
            
            # 获取向量数量和维度
            stats = await self._analyze_data_characteristics(table_name, "", 100)
            
            # 更新统计
            self.index_stats[index_name] = IndexStats(
                index_type=index_type,
                total_vectors=stats["total_vectors"],
                dimension=stats["dimension"],
                build_time_ms=build_time,
                memory_usage_mb=memory_usage_mb,
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.warning(f"更新索引统计失败: {e}")
    
    async def get_index_stats(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """获取索引统计信息"""
        if index_name:
            stats = self.index_stats.get(index_name)
            if stats:
                return {
                    "index_type": stats.index_type.value,
                    "total_vectors": stats.total_vectors,
                    "dimension": stats.dimension,
                    "build_time_ms": stats.build_time_ms,
                    "memory_usage_mb": stats.memory_usage_mb,
                    "last_updated": stats.last_updated.isoformat()
                }
            return {}
        else:
            # 返回所有索引统计
            return {
                name: {
                    "index_type": stats.index_type.value,
                    "total_vectors": stats.total_vectors,
                    "dimension": stats.dimension,
                    "build_time_ms": stats.build_time_ms,
                    "memory_usage_mb": stats.memory_usage_mb,
                    "last_updated": stats.last_updated.isoformat()
                }
                for name, stats in self.index_stats.items()
            }
    
    async def compute_lsh_hash(self, vector: np.ndarray, n_bits: int = 128) -> str:
        """计算向量的LSH哈希值"""
        # 生成随机投影矩阵（实际应用中应该预先生成并保存）
        np.random.seed(42)  # 固定种子以保证一致性
        random_vectors = np.random.randn(n_bits, len(vector))
        
        # 计算投影
        projections = np.dot(random_vectors, vector)
        
        # 转换为二进制哈希
        hash_bits = (projections > 0).astype(int)
        
        # 转换为十六进制字符串
        hash_str = ''.join(str(bit) for bit in hash_bits)
        return hash_str