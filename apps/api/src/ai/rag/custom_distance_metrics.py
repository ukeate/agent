"""
自定义距离度量

实现多种距离计算方法，支持pgvector的内置距离函数和自定义扩展
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
import numba

from src.core.logging import get_logger
logger = get_logger(__name__)

class DistanceMetric(str, Enum):
    """距离度量类型"""
    # pgvector内置
    L2 = "l2"                        # 欧氏距离 <->
    COSINE = "cosine"                # 余弦距离 <=>
    INNER_PRODUCT = "inner_product"  # 内积 <#>
    L1 = "l1"                        # 曼哈顿距离 <+>
    HAMMING = "hamming"              # 汉明距离 <~> (二进制向量)
    JACCARD = "jaccard"              # Jaccard距离 <%> (二进制向量)
    
    # 自定义扩展
    MINKOWSKI = "minkowski"          # 闵可夫斯基距离
    CHEBYSHEV = "chebyshev"          # 切比雪夫距离
    MAHALANOBIS = "mahalanobis"      # 马氏距离
    CORRELATION = "correlation"       # 相关性距离
    CANBERRA = "canberra"            # 堪培拉距离
    BRAYCURTIS = "braycurtis"        # 布雷柯蒂斯距离
    JENSEN_SHANNON = "jensen_shannon" # JS散度
    WASSERSTEIN = "wasserstein"      # Wasserstein距离

@dataclass
class DistanceConfig:
    """距离计算配置"""
    metric: DistanceMetric = DistanceMetric.COSINE
    normalize: bool = True
    use_gpu: bool = False
    batch_size: int = 1000
    # 特定度量的参数
    p: float = 2.0  # Minkowski距离的p值
    covariance_matrix: Optional[np.ndarray] = None  # 马氏距离的协方差矩阵

class CustomDistanceCalculator:
    """自定义距离计算器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.metric_cache = {}
        self.performance_stats = {
            "total_calculations": 0,
            "avg_calculation_time_ms": 0.0
        }
        
    async def calculate_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
        metric: DistanceMetric,
        **kwargs
    ) -> float:
        """计算两个向量之间的距离"""
        try:
            if metric in [DistanceMetric.L2, DistanceMetric.COSINE, 
                         DistanceMetric.INNER_PRODUCT, DistanceMetric.L1]:
                # 使用pgvector内置函数
                return await self._calculate_pgvector_distance(
                    vector1, vector2, metric
                )
            else:
                # 使用自定义实现
                return await self._calculate_custom_distance(
                    vector1, vector2, metric, **kwargs
                )
                
        except Exception as e:
            logger.error(f"距离计算失败: {e}")
            raise
    
    async def _calculate_pgvector_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
        metric: DistanceMetric
    ) -> float:
        """使用pgvector内置函数计算距离"""
        # 选择操作符
        if metric == DistanceMetric.L2:
            operator = "<->"
        elif metric == DistanceMetric.COSINE:
            operator = "<=>"
        elif metric == DistanceMetric.INNER_PRODUCT:
            operator = "<#>"
        elif metric == DistanceMetric.L1:
            operator = "<+>"
        else:
            raise ValueError(f"不支持的pgvector度量: {metric}")
        
        # 执行SQL查询
        query = f"""
        SELECT %s::vector {operator} %s::vector AS distance
        """
        
        result = await self.db.execute(
            text(query),
            (vector1.tolist(), vector2.tolist())
        )
        
        row = result.fetchone()
        return float(row.distance)
    
    async def _calculate_custom_distance(
        self,
        vector1: np.ndarray,
        vector2: np.ndarray,
        metric: DistanceMetric,
        **kwargs
    ) -> float:
        """计算自定义距离度量"""
        if metric == DistanceMetric.MINKOWSKI:
            return self._minkowski_distance(vector1, vector2, kwargs.get('p', 2))
        elif metric == DistanceMetric.CHEBYSHEV:
            return self._chebyshev_distance(vector1, vector2)
        elif metric == DistanceMetric.MAHALANOBIS:
            return self._mahalanobis_distance(
                vector1, vector2, kwargs.get('covariance_matrix')
            )
        elif metric == DistanceMetric.CORRELATION:
            return self._correlation_distance(vector1, vector2)
        elif metric == DistanceMetric.CANBERRA:
            return self._canberra_distance(vector1, vector2)
        elif metric == DistanceMetric.BRAYCURTIS:
            return self._braycurtis_distance(vector1, vector2)
        elif metric == DistanceMetric.JENSEN_SHANNON:
            return self._jensen_shannon_distance(vector1, vector2)
        elif metric == DistanceMetric.WASSERSTEIN:
            return self._wasserstein_distance(vector1, vector2)
        else:
            raise ValueError(f"不支持的自定义度量: {metric}")
    
    @staticmethod
    @numba.jit(nopython=True)
    def _minkowski_distance(v1: np.ndarray, v2: np.ndarray, p: float) -> float:
        """闵可夫斯基距离（使用numba加速）"""
        diff = np.abs(v1 - v2)
        return np.power(np.sum(np.power(diff, p)), 1.0/p)
    
    @staticmethod
    def _chebyshev_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """切比雪夫距离（最大坐标差）"""
        return np.max(np.abs(v1 - v2))
    
    @staticmethod
    def _mahalanobis_distance(
        v1: np.ndarray,
        v2: np.ndarray,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> float:
        """马氏距离"""
        diff = v1 - v2
        
        if covariance_matrix is None:
            # 使用单位矩阵（退化为欧氏距离）
            return np.sqrt(np.dot(diff, diff))
        
        # 计算马氏距离
        inv_cov = np.linalg.inv(covariance_matrix)
        return np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
    
    @staticmethod
    def _correlation_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """相关性距离（1 - 皮尔逊相关系数）"""
        if len(v1) < 2:
            return 1.0
        
        correlation, _ = pearsonr(v1, v2)
        return 1.0 - correlation
    
    @staticmethod
    def _canberra_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """堪培拉距离"""
        numerator = np.abs(v1 - v2)
        denominator = np.abs(v1) + np.abs(v2)
        
        # 避免除零
        mask = denominator != 0
        result = np.sum(numerator[mask] / denominator[mask])
        
        return result
    
    @staticmethod
    def _braycurtis_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """布雷柯蒂斯距离"""
        numerator = np.sum(np.abs(v1 - v2))
        denominator = np.sum(np.abs(v1 + v2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    @staticmethod
    def _jensen_shannon_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Jensen-Shannon散度"""
        # 确保向量是概率分布（归一化）
        p = v1 / np.sum(v1)
        q = v2 / np.sum(v2)
        
        # 计算平均分布
        m = 0.5 * (p + q)
        
        # 计算KL散度
        def kl_divergence(a, b):
            # 避免log(0)
            epsilon = 1e-10
            a = np.clip(a, epsilon, 1)
            b = np.clip(b, epsilon, 1)
            return np.sum(a * np.log(a / b))
        
        # JS散度
        js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        
        # JS距离（平方根）
        return np.sqrt(js_div)
    
    @staticmethod
    def _wasserstein_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Wasserstein距离（Earth Mover's Distance）"""
        # 简化版本：假设一维分布
        # 对于高维需要更复杂的算法
        return distance.wasserstein_distance(v1, v2)
    
    async def batch_distance_calculation(
        self,
        query_vector: np.ndarray,
        database_vectors: np.ndarray,
        metric: DistanceMetric,
        config: Optional[DistanceConfig] = None
    ) -> np.ndarray:
        """批量计算距离"""
        if config is None:
            config = DistanceConfig(metric=metric)
        
        n_vectors = len(database_vectors)
        distances = np.zeros(n_vectors)
        
        # 分批处理
        for i in range(0, n_vectors, config.batch_size):
            batch_end = min(i + config.batch_size, n_vectors)
            batch_vectors = database_vectors[i:batch_end]
            
            if config.use_gpu and self._is_gpu_available():
                batch_distances = await self._gpu_batch_distance(
                    query_vector, batch_vectors, metric
                )
            else:
                batch_distances = await self._cpu_batch_distance(
                    query_vector, batch_vectors, metric, config
                )
            
            distances[i:batch_end] = batch_distances
        
        return distances
    
    async def _cpu_batch_distance(
        self,
        query_vector: np.ndarray,
        batch_vectors: np.ndarray,
        metric: DistanceMetric,
        config: DistanceConfig
    ) -> np.ndarray:
        """CPU批量距离计算"""
        if metric == DistanceMetric.L2:
            # 向量化欧氏距离
            diff = batch_vectors - query_vector
            distances = np.linalg.norm(diff, axis=1)
        elif metric == DistanceMetric.COSINE:
            # 向量化余弦距离
            query_norm = np.linalg.norm(query_vector)
            batch_norms = np.linalg.norm(batch_vectors, axis=1)
            dot_products = np.dot(batch_vectors, query_vector)
            
            # 避免除零
            denominators = query_norm * batch_norms
            mask = denominators != 0
            
            distances = np.ones(len(batch_vectors))
            distances[mask] = 1 - (dot_products[mask] / denominators[mask])
        elif metric == DistanceMetric.L1:
            # 向量化曼哈顿距离
            diff = np.abs(batch_vectors - query_vector)
            distances = np.sum(diff, axis=1)
        elif metric == DistanceMetric.MINKOWSKI:
            # 向量化闵可夫斯基距离
            p = config.p
            diff = np.abs(batch_vectors - query_vector)
            distances = np.power(np.sum(np.power(diff, p), axis=1), 1.0/p)
        else:
            # 其他距离逐个计算
            distances = np.array([
                await self._calculate_custom_distance(
                    query_vector, vec, metric,
                    covariance_matrix=config.covariance_matrix
                )
                for vec in batch_vectors
            ])
        
        return distances
    
    async def _gpu_batch_distance(
        self,
        query_vector: np.ndarray,
        batch_vectors: np.ndarray,
        metric: DistanceMetric
    ) -> np.ndarray:
        """GPU批量距离计算"""
        try:
            import cupy as cp
            
            # 转移到GPU
            query_gpu = cp.asarray(query_vector)
            batch_gpu = cp.asarray(batch_vectors)
            
            if metric == DistanceMetric.L2:
                diff = batch_gpu - query_gpu
                distances_gpu = cp.linalg.norm(diff, axis=1)
            elif metric == DistanceMetric.COSINE:
                query_norm = cp.linalg.norm(query_gpu)
                batch_norms = cp.linalg.norm(batch_gpu, axis=1)
                dot_products = cp.dot(batch_gpu, query_gpu)
                distances_gpu = 1 - (dot_products / (query_norm * batch_norms))
            else:
                # 降级到CPU
                return await self._cpu_batch_distance(
                    query_vector, batch_vectors, metric,
                    DistanceConfig(metric=metric)
                )
            
            # 转回CPU
            return distances_gpu.get()
            
        except ImportError:
            # GPU不可用，降级到CPU
            return await self._cpu_batch_distance(
                query_vector, batch_vectors, metric,
                DistanceConfig(metric=metric)
            )
    
    def _is_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() > 0
        except ImportError:
            return False
    
    async def create_custom_distance_function(
        self,
        function_name: str,
        function_code: str,
        input_dim: int
    ) -> bool:
        """创建自定义SQL距离函数"""
        try:
            # 创建PL/Python函数
            create_function_sql = f"""
            CREATE OR REPLACE FUNCTION {function_name}(
                vector1 float8[],
                vector2 float8[]
            ) RETURNS float8 AS $$
            {function_code}
            $$ LANGUAGE plpython3u IMMUTABLE PARALLEL SAFE;
            """
            
            await self.db.execute(text(create_function_sql))
            await self.db.commit()
            
            logger.info(f"自定义距离函数 {function_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建自定义距离函数失败: {e}")
            await self.db.rollback()
            return False
    
    async def register_distance_operator(
        self,
        operator_symbol: str,
        function_name: str,
        left_type: str = "vector",
        right_type: str = "vector"
    ) -> bool:
        """注册自定义距离操作符"""
        try:
            # 创建操作符
            create_operator_sql = f"""
            CREATE OPERATOR {operator_symbol} (
                LEFTARG = {left_type},
                RIGHTARG = {right_type},
                FUNCTION = {function_name},
                COMMUTATOR = {operator_symbol}
            );
            """
            
            await self.db.execute(text(create_operator_sql))
            await self.db.commit()
            
            logger.info(f"自定义操作符 {operator_symbol} 注册成功")
            return True
            
        except Exception as e:
            logger.error(f"注册自定义操作符失败: {e}")
            await self.db.rollback()
            return False
    
    async def benchmark_distance_metrics(
        self,
        test_vectors: np.ndarray,
        metrics: List[DistanceMetric],
        iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """基准测试不同距离度量的性能"""
        import time
        
        results = {}
        n_vectors = len(test_vectors)
        
        for metric in metrics:
            times = []
            
            for _ in range(iterations):
                # 随机选择两个向量
                idx1, idx2 = np.random.choice(n_vectors, 2, replace=False)
                v1, v2 = test_vectors[idx1], test_vectors[idx2]
                
                start = time.perf_counter()
                await self.calculate_distance(v1, v2, metric)
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # 转换为毫秒
            
            results[metric.value] = {
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "p50_ms": np.percentile(times, 50),
                "p95_ms": np.percentile(times, 95),
                "p99_ms": np.percentile(times, 99)
            }
        
        return results
    
    async def find_optimal_metric(
        self,
        reference_vectors: np.ndarray,
        candidate_vectors: np.ndarray,
        ground_truth_indices: np.ndarray,
        metrics: List[DistanceMetric]
    ) -> Tuple[DistanceMetric, Dict[str, float]]:
        """根据准确率找到最优距离度量"""
        best_metric = None
        best_score = 0
        scores = {}
        
        for metric in metrics:
            # 计算所有距离
            distances = await self.batch_distance_calculation(
                reference_vectors[0],  # 使用第一个作为查询
                candidate_vectors,
                metric
            )
            
            # 获取top-k
            k = len(ground_truth_indices)
            predicted_indices = np.argsort(distances)[:k]
            
            # 计算准确率
            intersection = len(set(predicted_indices) & set(ground_truth_indices))
            accuracy = intersection / k
            
            scores[metric.value] = accuracy
            
            if accuracy > best_score:
                best_score = accuracy
                best_metric = metric
        
        return best_metric, scores

class DistanceMetricInterface:
    """距离度量接口（用于pgvector集成）"""
    
    def __init__(self, calculator: CustomDistanceCalculator):
        self.calculator = calculator
        
    async def create_distance_index(
        self,
        table_name: str,
        column_name: str,
        metric: DistanceMetric,
        index_type: str = "hnsw"
    ) -> bool:
        """创建使用特定距离度量的索引"""
        try:
            # 选择操作符类
            if metric == DistanceMetric.L2:
                ops_class = "vector_l2_ops"
            elif metric == DistanceMetric.COSINE:
                ops_class = "vector_cosine_ops"
            elif metric == DistanceMetric.INNER_PRODUCT:
                ops_class = "vector_ip_ops"
            elif metric == DistanceMetric.L1:
                ops_class = "vector_l1_ops"
            elif metric == DistanceMetric.HAMMING:
                ops_class = "bit_hamming_ops"
            elif metric == DistanceMetric.JACCARD:
                ops_class = "bit_jaccard_ops"
            else:
                # 自定义度量需要先创建操作符类
                ops_class = f"vector_{metric.value}_ops"
                await self._create_custom_ops_class(ops_class, metric)
            
            # 创建索引
            index_name = f"idx_{table_name}_{column_name}_{metric.value}"
            
            if index_type == "hnsw":
                create_index_sql = f"""
                CREATE INDEX {index_name}
                ON {table_name}
                USING hnsw ({column_name} {ops_class})
                WITH (m = 16, ef_construction = 200);
                """
            elif index_type == "ivfflat":
                create_index_sql = f"""
                CREATE INDEX {index_name}
                ON {table_name}
                USING ivfflat ({column_name} {ops_class})
                WITH (lists = 100);
                """
            else:
                raise ValueError(f"不支持的索引类型: {index_type}")
            
            await self.calculator.db.execute(text(create_index_sql))
            await self.calculator.db.commit()
            
            logger.info(f"索引 {index_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            await self.calculator.db.rollback()
            return False
    
    async def _create_custom_ops_class(
        self,
        ops_class_name: str,
        metric: DistanceMetric
    ) -> None:
        """创建自定义操作符类"""
        # 这需要pgvector的扩展支持
        # 实际实现可能需要C语言扩展
        logger.warning(f"自定义操作符类 {ops_class_name} 需要pgvector扩展支持")
    
    async def search_with_custom_metric(
        self,
        table_name: str,
        column_name: str,
        query_vector: np.ndarray,
        metric: DistanceMetric,
        top_k: int = 10,
        filter_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """使用自定义度量进行搜索"""
        try:
            # 构建查询
            if metric in [DistanceMetric.L2, DistanceMetric.COSINE,
                         DistanceMetric.INNER_PRODUCT, DistanceMetric.L1]:
                # 使用pgvector内置操作符
                if metric == DistanceMetric.L2:
                    operator = "<->"
                elif metric == DistanceMetric.COSINE:
                    operator = "<=>"
                elif metric == DistanceMetric.INNER_PRODUCT:
                    operator = "<#>"
                else:  # L1
                    operator = "<+>"
                
                query_sql = f"""
                SELECT *,
                    {column_name} {operator} %s::vector AS distance
                FROM {table_name}
                {f"WHERE {filter_condition}" if filter_condition else ""}
                ORDER BY {column_name} {operator} %s::vector
                LIMIT %s
                """
                
                result = await self.calculator.db.execute(
                    text(query_sql),
                    (query_vector.tolist(), query_vector.tolist(), top_k)
                )
            else:
                # 自定义度量需要全表扫描
                fetch_sql = f"""
                SELECT *, {column_name} AS embedding
                FROM {table_name}
                {f"WHERE {filter_condition}" if filter_condition else ""}
                """
                
                result = await self.calculator.db.execute(text(fetch_sql))
                rows = result.fetchall()
                
                # 计算距离
                distances = []
                for row in rows:
                    dist = await self.calculator.calculate_distance(
                        query_vector,
                        np.array(row.embedding),
                        metric
                    )
                    distances.append({
                        **row._asdict(),
                        "distance": dist
                    })
                
                # 排序并返回top-k
                distances.sort(key=lambda x: x["distance"])
                return distances[:top_k]
            
            # 处理pgvector结果
            results = []
            for row in result.fetchall():
                results.append(row._asdict())
            
            return results
            
        except Exception as e:
            logger.error(f"自定义度量搜索失败: {e}")
            return []
