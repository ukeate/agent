"""
时序向量索引

支持时间序列向量的存储、检索和轨迹分析
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import json
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import statistics

from src.core.logging import get_logger
logger = get_logger(__name__)

class TemporalAggregation(str, Enum):
    """时间聚合方式"""
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    LAST = "last"
    FIRST = "first"
    MEDIAN = "median"

class TrendDirection(str, Enum):
    """趋势方向"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class TemporalVector:
    """时序向量"""
    vector: np.ndarray
    timestamp: datetime
    entity_id: str
    metadata: Dict[str, Any]
    sequence_id: Optional[str] = None

@dataclass
class Trajectory:
    """轨迹"""
    entity_id: str
    vectors: List[TemporalVector]
    start_time: datetime
    end_time: datetime
    total_distance: float
    avg_velocity: float

@dataclass
class TemporalPattern:
    """时序模式"""
    pattern_type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    entities: List[str]
    description: str

class TemporalVectorIndex:
    """时序向量索引"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.cache = {}
        self.stats = {
            "total_vectors": 0,
            "unique_entities": 0,
            "avg_trajectory_length": 0.0,
            "patterns_detected": 0
        }
    
    async def index_temporal_vector(
        self,
        vector: np.ndarray,
        entity_id: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """索引时序向量"""
        try:
            # 插入时序向量
            insert_sql = """
            INSERT INTO temporal_vectors 
            (entity_id, timestamp, vector, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """
            
            result = await self.db.execute(
                text(insert_sql),
                (
                    entity_id,
                    timestamp,
                    vector.tolist(),
                    json.dumps(metadata) if metadata else "{}"
                )
            )
            
            row = result.fetchone()
            await self.db.commit()
            
            # 更新统计
            self.stats["total_vectors"] += 1
            
            # 清除相关缓存
            self._invalidate_cache(entity_id)
            
            return str(row.id)
            
        except Exception as e:
            logger.error(f"索引时序向量失败: {e}")
            await self.db.rollback()
            raise
    
    async def search_temporal_neighbors(
        self,
        query_vector: np.ndarray,
        time_range: Tuple[datetime, datetime],
        top_k: int = 10,
        entity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """搜索时间范围内的最近邻"""
        try:
            # 构建查询
            base_sql = """
            SELECT 
                id,
                entity_id,
                timestamp,
                vector,
                metadata,
                vector <=> %s::vector AS distance
            FROM temporal_vectors
            WHERE timestamp BETWEEN %s AND %s
            """
            
            params = [
                query_vector.tolist(),
                time_range[0],
                time_range[1]
            ]
            
            if entity_filter:
                base_sql += " AND entity_id = ANY(%s)"
                params.append(entity_filter)
            
            base_sql += " ORDER BY vector <=> %s::vector LIMIT %s"
            params.extend([query_vector.tolist(), top_k])
            
            result = await self.db.execute(text(base_sql), params)
            
            results = []
            for row in result.fetchall():
                results.append({
                    "id": str(row.id),
                    "entity_id": row.entity_id,
                    "timestamp": row.timestamp,
                    "vector": np.array(row.vector),
                    "metadata": row.metadata or {},
                    "distance": float(row.distance)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"时序搜索失败: {e}")
            return []
    
    async def compute_trajectory(
        self,
        entity_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_points: int = 2
    ) -> Optional[Trajectory]:
        """计算实体轨迹"""
        try:
            # 获取时序向量
            vectors = await self._get_entity_vectors(entity_id, time_range)
            
            if len(vectors) < min_points:
                return None
            
            # 按时间排序
            vectors.sort(key=lambda v: v.timestamp)
            
            # 计算轨迹距离
            total_distance = 0.0
            for i in range(1, len(vectors)):
                dist = np.linalg.norm(
                    vectors[i].vector - vectors[i-1].vector
                )
                total_distance += dist
            
            # 计算平均速度
            time_diff = (vectors[-1].timestamp - vectors[0].timestamp).total_seconds()
            avg_velocity = total_distance / time_diff if time_diff > 0 else 0
            
            trajectory = Trajectory(
                entity_id=entity_id,
                vectors=vectors,
                start_time=vectors[0].timestamp,
                end_time=vectors[-1].timestamp,
                total_distance=total_distance,
                avg_velocity=avg_velocity
            )
            
            return trajectory
            
        except Exception as e:
            logger.error(f"计算轨迹失败: {e}")
            return None
    
    async def find_similar_trajectories(
        self,
        reference_trajectory: Trajectory,
        similarity_threshold: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[str, float, Trajectory]]:
        """查找相似轨迹"""
        try:
            # 获取所有实体
            entities = await self._get_all_entities()
            
            similar_trajectories = []
            
            for entity_id in entities:
                if entity_id == reference_trajectory.entity_id:
                    continue
                
                # 计算轨迹
                trajectory = await self.compute_trajectory(entity_id)
                if not trajectory:
                    continue
                
                # 计算相似度
                similarity = await self._compute_trajectory_similarity(
                    reference_trajectory, trajectory
                )
                
                if similarity >= similarity_threshold:
                    similar_trajectories.append((entity_id, similarity, trajectory))
            
            # 排序并返回
            similar_trajectories.sort(key=lambda x: x[1], reverse=True)
            return similar_trajectories[:max_results]
            
        except Exception as e:
            logger.error(f"查找相似轨迹失败: {e}")
            return []
    
    async def detect_temporal_patterns(
        self,
        time_range: Tuple[datetime, datetime],
        pattern_types: Optional[List[str]] = None
    ) -> List[TemporalPattern]:
        """检测时序模式"""
        try:
            patterns = []
            
            # 获取时间范围内的所有向量
            vectors_by_entity = await self._get_vectors_by_entity(time_range)
            
            # 检测不同类型的模式
            if not pattern_types or "convergence" in pattern_types:
                convergence_patterns = await self._detect_convergence_patterns(
                    vectors_by_entity, time_range
                )
                patterns.extend(convergence_patterns)
            
            if not pattern_types or "divergence" in pattern_types:
                divergence_patterns = await self._detect_divergence_patterns(
                    vectors_by_entity, time_range
                )
                patterns.extend(divergence_patterns)
            
            if not pattern_types or "periodic" in pattern_types:
                periodic_patterns = await self._detect_periodic_patterns(
                    vectors_by_entity, time_range
                )
                patterns.extend(periodic_patterns)
            
            if not pattern_types or "anomaly" in pattern_types:
                anomaly_patterns = await self._detect_anomaly_patterns(
                    vectors_by_entity, time_range
                )
                patterns.extend(anomaly_patterns)
            
            # 更新统计
            self.stats["patterns_detected"] = len(patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"检测时序模式失败: {e}")
            return []
    
    async def analyze_trend(
        self,
        entity_id: str,
        time_range: Tuple[datetime, datetime],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """分析向量变化趋势"""
        try:
            vectors = await self._get_entity_vectors(entity_id, time_range)
            
            if len(vectors) < 2:
                return {
                    "trend": TrendDirection.STABLE,
                    "confidence": 0.0,
                    "velocity": 0.0,
                    "acceleration": 0.0
                }
            
            # 按时间排序
            vectors.sort(key=lambda v: v.timestamp)
            
            # 计算移动平均
            velocities = []
            for i in range(1, len(vectors)):
                v1, v2 = vectors[i-1], vectors[i]
                time_diff = (v2.timestamp - v1.timestamp).total_seconds()
                if time_diff > 0:
                    velocity = np.linalg.norm(v2.vector - v1.vector) / time_diff
                    velocities.append(velocity)
            
            if not velocities:
                return {
                    "trend": TrendDirection.STABLE,
                    "confidence": 0.0,
                    "velocity": 0.0,
                    "acceleration": 0.0
                }
            
            # 计算趋势
            avg_velocity = np.mean(velocities)
            std_velocity = np.std(velocities)
            
            # 计算加速度
            accelerations = []
            for i in range(1, len(velocities)):
                acc = velocities[i] - velocities[i-1]
                accelerations.append(acc)
            
            avg_acceleration = np.mean(accelerations) if accelerations else 0
            
            # 确定趋势方向
            # 计算向量序列中的总体方向变化
            first_vec = vectors[0].vector
            last_vec = vectors[-1].vector
            total_change = np.linalg.norm(last_vec - first_vec)
            
            if std_velocity / (avg_velocity + 1e-6) > 0.5:
                trend = TrendDirection.VOLATILE
                confidence = 1.0 - (std_velocity / (avg_velocity + 1e-6))
            elif total_change > len(vectors) * 0.5:  # 显著变化
                if avg_acceleration > 0:
                    trend = TrendDirection.INCREASING
                else:
                    trend = TrendDirection.DECREASING
                confidence = min(1.0, total_change / len(vectors) * 0.1)
            elif avg_acceleration > 0.01:
                trend = TrendDirection.INCREASING
                confidence = min(1.0, abs(avg_acceleration) * 10)
            elif avg_acceleration < -0.01:
                trend = TrendDirection.DECREASING
                confidence = min(1.0, abs(avg_acceleration) * 10)
            else:
                trend = TrendDirection.STABLE
                confidence = 1.0 - abs(avg_acceleration) * 100
            
            return {
                "trend": trend,
                "confidence": max(0.0, min(1.0, confidence)),
                "velocity": avg_velocity,
                "acceleration": avg_acceleration,
                "volatility": std_velocity
            }
            
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {
                "trend": TrendDirection.STABLE,
                "confidence": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0
            }
    
    async def aggregate_temporal_vectors(
        self,
        entity_id: str,
        time_range: Tuple[datetime, datetime],
        aggregation: TemporalAggregation = TemporalAggregation.MEAN
    ) -> Optional[np.ndarray]:
        """聚合时序向量"""
        try:
            vectors = await self._get_entity_vectors(entity_id, time_range)
            
            if not vectors:
                return None
            
            vector_arrays = [v.vector for v in vectors]
            
            if aggregation == TemporalAggregation.MEAN:
                result = np.mean(vector_arrays, axis=0)
            elif aggregation == TemporalAggregation.MAX:
                result = np.max(vector_arrays, axis=0)
            elif aggregation == TemporalAggregation.MIN:
                result = np.min(vector_arrays, axis=0)
            elif aggregation == TemporalAggregation.MEDIAN:
                result = np.median(vector_arrays, axis=0)
            elif aggregation == TemporalAggregation.FIRST:
                result = vector_arrays[0]
            elif aggregation == TemporalAggregation.LAST:
                result = vector_arrays[-1]
            else:
                result = np.mean(vector_arrays, axis=0)
            
            return result
            
        except Exception as e:
            logger.error(f"聚合时序向量失败: {e}")
            return None
    
    async def _get_entity_vectors(
        self,
        entity_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[TemporalVector]:
        """获取实体的时序向量"""
        # 检查缓存
        cache_key = f"{entity_id}:{time_range}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            sql = """
            SELECT id, timestamp, vector, metadata
            FROM temporal_vectors
            WHERE entity_id = %s
            """
            params = [entity_id]
            
            if time_range:
                sql += " AND timestamp BETWEEN %s AND %s"
                params.extend([time_range[0], time_range[1]])
            
            sql += " ORDER BY timestamp"
            
            result = await self.db.execute(text(sql), params)
            
            vectors = []
            for row in result.fetchall():
                vectors.append(TemporalVector(
                    vector=np.array(row.vector),
                    timestamp=row.timestamp,
                    entity_id=entity_id,
                    metadata=row.metadata or {}
                ))
            
            # 缓存结果
            self.cache[cache_key] = vectors
            
            return vectors
            
        except Exception as e:
            logger.error(f"获取实体向量失败: {e}")
            return []
    
    async def _get_all_entities(self) -> List[str]:
        """获取所有实体ID"""
        try:
            sql = "SELECT DISTINCT entity_id FROM temporal_vectors"
            result = await self.db.execute(text(sql))
            return [row.entity_id for row in result.fetchall()]
        except Exception as e:
            logger.error(f"获取实体列表失败: {e}")
            return []
    
    async def _compute_trajectory_similarity(
        self,
        traj1: Trajectory,
        traj2: Trajectory
    ) -> float:
        """计算轨迹相似度"""
        # 使用DTW（动态时间规整）或其他轨迹相似度算法
        # 这里使用简化版本：基于向量序列的平均相似度
        
        min_len = min(len(traj1.vectors), len(traj2.vectors))
        if min_len == 0:
            return 0.0
        
        similarities = []
        for i in range(min_len):
            v1 = traj1.vectors[i].vector
            v2 = traj2.vectors[i].vector
            
            # 计算余弦相似度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
            else:
                cos_sim = 0.0
            similarities.append(cos_sim)
        
        return np.mean(similarities)
    
    async def _get_vectors_by_entity(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, List[TemporalVector]]:
        """按实体分组获取向量"""
        try:
            sql = """
            SELECT entity_id, timestamp, vector, metadata
            FROM temporal_vectors
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY entity_id, timestamp
            """
            
            result = await self.db.execute(text(sql), (time_range[0], time_range[1]))
            
            vectors_by_entity = {}
            for row in result.fetchall():
                entity_id = row.entity_id
                if entity_id not in vectors_by_entity:
                    vectors_by_entity[entity_id] = []
                
                vectors_by_entity[entity_id].append(TemporalVector(
                    vector=np.array(row.vector),
                    timestamp=row.timestamp,
                    entity_id=entity_id,
                    metadata=row.metadata or {}
                ))
            
            return vectors_by_entity
            
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return {}
    
    async def _detect_convergence_patterns(
        self,
        vectors_by_entity: Dict[str, List[TemporalVector]],
        time_range: Tuple[datetime, datetime]
    ) -> List[TemporalPattern]:
        """检测收敛模式"""
        patterns = []
        
        # 检测多个实体向量趋向聚集的模式
        entities = list(vectors_by_entity.keys())
        if len(entities) < 2:
            return patterns
        
        # 计算实体间距离的变化
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                e1, e2 = entities[i], entities[j]
                v1_list = vectors_by_entity[e1]
                v2_list = vectors_by_entity[e2]
                
                if len(v1_list) < 2 or len(v2_list) < 2:
                    continue
                
                # 计算初始和最终距离
                initial_dist = np.linalg.norm(
                    v1_list[0].vector - v2_list[0].vector
                )
                final_dist = np.linalg.norm(
                    v1_list[-1].vector - v2_list[-1].vector
                )
                
                # 如果距离显著减小，认为是收敛
                if final_dist < initial_dist * 0.5:
                    patterns.append(TemporalPattern(
                        pattern_type="convergence",
                        confidence=(initial_dist - final_dist) / initial_dist,
                        start_time=time_range[0],
                        end_time=time_range[1],
                        entities=[e1, e2],
                        description=f"实体 {e1} 和 {e2} 向量收敛"
                    ))
        
        return patterns
    
    async def _detect_divergence_patterns(
        self,
        vectors_by_entity: Dict[str, List[TemporalVector]],
        time_range: Tuple[datetime, datetime]
    ) -> List[TemporalPattern]:
        """检测发散模式"""
        patterns = []
        
        # 类似收敛检测，但检测距离增大
        entities = list(vectors_by_entity.keys())
        if len(entities) < 2:
            return patterns
        
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                e1, e2 = entities[i], entities[j]
                v1_list = vectors_by_entity[e1]
                v2_list = vectors_by_entity[e2]
                
                if len(v1_list) < 2 or len(v2_list) < 2:
                    continue
                
                initial_dist = np.linalg.norm(
                    v1_list[0].vector - v2_list[0].vector
                )
                final_dist = np.linalg.norm(
                    v1_list[-1].vector - v2_list[-1].vector
                )
                
                # 如果距离显著增大，认为是发散
                if final_dist > initial_dist * 2:
                    patterns.append(TemporalPattern(
                        pattern_type="divergence",
                        confidence=(final_dist - initial_dist) / (initial_dist + 1e-6),
                        start_time=time_range[0],
                        end_time=time_range[1],
                        entities=[e1, e2],
                        description=f"实体 {e1} 和 {e2} 向量发散"
                    ))
        
        return patterns
    
    async def _detect_periodic_patterns(
        self,
        vectors_by_entity: Dict[str, List[TemporalVector]],
        time_range: Tuple[datetime, datetime]
    ) -> List[TemporalPattern]:
        """检测周期性模式"""
        patterns = []
        
        for entity_id, vectors in vectors_by_entity.items():
            if len(vectors) < 10:  # 需要足够的数据点
                continue
            
            # 计算向量序列的自相关
            vector_sequence = np.array([v.vector for v in vectors])
            
            # 简化的周期性检测
            # 实际应用中可以使用FFT或其他信号处理方法
            distances = []
            for i in range(1, len(vector_sequence)):
                dist = np.linalg.norm(vector_sequence[i] - vector_sequence[i-1])
                distances.append(dist)
            
            if distances:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                
                # 如果变化相对稳定，可能存在周期性
                if std_dist / (mean_dist + 1e-6) < 0.3:
                    patterns.append(TemporalPattern(
                        pattern_type="periodic",
                        confidence=1.0 - (std_dist / (mean_dist + 1e-6)),
                        start_time=vectors[0].timestamp,
                        end_time=vectors[-1].timestamp,
                        entities=[entity_id],
                        description=f"实体 {entity_id} 显示周期性变化"
                    ))
        
        return patterns
    
    async def _detect_anomaly_patterns(
        self,
        vectors_by_entity: Dict[str, List[TemporalVector]],
        time_range: Tuple[datetime, datetime]
    ) -> List[TemporalPattern]:
        """检测异常模式"""
        patterns = []
        
        for entity_id, vectors in vectors_by_entity.items():
            if len(vectors) < 5:
                continue
            
            # 计算向量变化的统计特征
            changes = []
            for i in range(1, len(vectors)):
                change = np.linalg.norm(vectors[i].vector - vectors[i-1].vector)
                changes.append(change)
            
            if changes:
                mean_change = np.mean(changes)
                std_change = np.std(changes)
                
                # 检测异常变化
                for i, change in enumerate(changes):
                    if abs(change - mean_change) > 3 * std_change:  # 3-sigma规则
                        patterns.append(TemporalPattern(
                            pattern_type="anomaly",
                            confidence=min(1.0, abs(change - mean_change) / (std_change + 1e-6) / 3),
                            start_time=vectors[i].timestamp,
                            end_time=vectors[i+1].timestamp,
                            entities=[entity_id],
                            description=f"实体 {entity_id} 在时间点 {vectors[i].timestamp} 出现异常变化"
                        ))
        
        return patterns
    
    def _invalidate_cache(self, entity_id: str) -> None:
        """清除缓存"""
        keys_to_remove = [k for k in self.cache.keys() if entity_id in k]
        for key in keys_to_remove:
            del self.cache[key]
    
    async def create_temporal_table(self) -> bool:
        """创建时序向量表"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS temporal_vectors (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                entity_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                vector VECTOR(1024),
                metadata JSONB DEFAULT '{}',
                sequence_id VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                -- 索引
                CONSTRAINT temporal_vectors_entity_time_unique 
                    UNIQUE (entity_id, timestamp)
            );
            
            -- 创建索引
            CREATE INDEX IF NOT EXISTS idx_temporal_entity 
            ON temporal_vectors(entity_id);
            
            CREATE INDEX IF NOT EXISTS idx_temporal_timestamp 
            ON temporal_vectors(timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_temporal_entity_time 
            ON temporal_vectors(entity_id, timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_temporal_vector_hnsw
            ON temporal_vectors USING hnsw (vector vector_cosine_ops);
            """
            
            await self.db.execute(text(create_table_sql))
            await self.db.commit()
            
            logger.info("时序向量表创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建时序向量表失败: {e}")
            await self.db.rollback()
            return False
