"""
性能优化器
实现查询缓存、索引优化、并发控制和性能监控
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
import asyncio
import hashlib
import time
import redis.asyncio as redis
from .graph_database import Neo4jGraphDatabase
from src.core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

@dataclass
class QueryPerformance:
    """查询性能指标"""
    query_hash: str
    query_type: str
    execution_time_ms: float
    result_count: int
    cache_hit: bool
    timestamp: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_type": self.query_type,
            "execution_time_ms": self.execution_time_ms,
            "result_count": self.result_count,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class PerformanceStats:
    """性能统计"""
    total_queries: int
    cache_hit_rate: float
    avg_query_time_ms: float
    slow_queries_count: int
    peak_qps: float
    current_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hit_rate,
            "avg_query_time_ms": self.avg_query_time_ms,
            "slow_queries_count": self.slow_queries_count,
            "peak_qps": self.peak_qps,
            "current_connections": self.current_connections
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase):
        self.graph_db = graph_db
        self.redis_client: Optional[redis.Redis] = None
        self.query_cache: Dict[str, Any] = {}
        self.performance_metrics: List[QueryPerformance] = []
        self.cache_ttl = settings.NEO4J_CACHE_TTL
        self.slow_query_threshold = 1000  # 1秒
        self.max_cache_size = 10000
        
        # 性能监控
        self.query_count = 0
        self.cache_hits = 0
        self.total_query_time = 0.0
        self.current_qps = 0.0
        self.last_qps_update = utc_now()
        
    async def initialize(self):
        """初始化性能优化器"""
        try:
            # 初始化Redis连接（用于分布式缓存）
            if settings.CACHE_ENABLED:
                self.redis_client = redis.from_url(
                    settings.CACHE_REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis缓存连接成功")
            
            # 启动性能监控任务
            asyncio.create_task(self._performance_monitor())
            
        except Exception as e:
            logger.warning(f"性能优化器初始化失败: {str(e)}")
    
    async def close(self):
        """关闭性能优化器"""
        if self.redis_client:
            await self.redis_client.aclose()
    
    async def execute_cached_query(self,
                                 query: str,
                                 parameters: Dict[str, Any],
                                 cache_enabled: bool = True,
                                 query_type: str = "read") -> Tuple[List[Dict[str, Any]], QueryPerformance]:
        """执行带缓存的查询"""
        start_time = time.time()
        cache_hit = False
        result = []
        
        try:
            # 生成查询哈希
            query_hash = self._generate_query_hash(query, parameters)
            
            # 尝试从缓存获取结果
            if cache_enabled and query_type == "read":
                cached_result = await self._get_from_cache(query_hash)
                if cached_result is not None:
                    result = cached_result
                    cache_hit = True
                    self.cache_hits += 1
            
            # 如果缓存未命中，执行查询
            if not cache_hit:
                if query_type == "read":
                    result = await self.graph_db.execute_read_query(query, parameters)
                else:
                    result = await self.graph_db.execute_write_query(query, parameters)
                
                # 将结果存入缓存
                if cache_enabled and query_type == "read" and result:
                    await self._set_to_cache(query_hash, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.query_count += 1
            self.total_query_time += execution_time
            
            # 更新QPS
            await self._update_qps()
            
            # 创建性能指标
            performance = QueryPerformance(
                query_hash=query_hash,
                query_type=query_type,
                execution_time_ms=execution_time,
                result_count=len(result) if result else 0,
                cache_hit=cache_hit
            )
            
            # 记录性能指标
            self.performance_metrics.append(performance)
            if len(self.performance_metrics) > 1000:  # 限制内存使用
                self.performance_metrics = self.performance_metrics[-500:]
            
            # 记录慢查询
            if execution_time > self.slow_query_threshold:
                logger.warning(f"慢查询检测: {execution_time:.2f}ms, Hash: {query_hash}")
            
            return result, performance
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"查询执行失败: {str(e)}")
            
            performance = QueryPerformance(
                query_hash=self._generate_query_hash(query, parameters),
                query_type=query_type,
                execution_time_ms=execution_time,
                result_count=0,
                cache_hit=False
            )
            
            raise e
    
    def _generate_query_hash(self, query: str, parameters: Dict[str, Any]) -> str:
        """生成查询哈希"""
        query_string = f"{query}:{str(sorted(parameters.items()))}"
        return hashlib.md5(query_string.encode()).hexdigest()
    
    async def _get_from_cache(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """从缓存获取结果"""
        try:
            # 优先使用Redis缓存
            if self.redis_client:
                cached_data = await self.redis_client.get(f"query:{query_hash}")
                if cached_data:
                    import json
                    return json.loads(cached_data)
            
            # 使用本地缓存
            if query_hash in self.query_cache:
                cache_entry = self.query_cache[query_hash]
                if cache_entry["expires_at"] > utc_now():
                    return cache_entry["data"]
                else:
                    # 过期删除
                    del self.query_cache[query_hash]
            
            return None
            
        except Exception as e:
            logger.warning(f"缓存获取失败: {str(e)}")
            return None
    
    async def _set_to_cache(self, query_hash: str, data: List[Dict[str, Any]]):
        """设置缓存"""
        try:
            # 设置Redis缓存
            if self.redis_client:
                import json
                await self.redis_client.setex(
                    f"query:{query_hash}",
                    self.cache_ttl,
                    json.dumps(data, default=str)
                )
            
            # 设置本地缓存
            if len(self.query_cache) < self.max_cache_size:
                self.query_cache[query_hash] = {
                    "data": data,
                    "expires_at": utc_now() + timedelta(seconds=self.cache_ttl)
                }
            
        except Exception as e:
            logger.warning(f"缓存设置失败: {str(e)}")
    
    async def _update_qps(self):
        """更新QPS统计"""
        now = utc_now()
        if (now - self.last_qps_update).seconds >= 1:
            # 每秒更新一次QPS
            time_diff = (now - self.last_qps_update).total_seconds()
            if time_diff > 0:
                self.current_qps = 1.0 / time_diff
                self.last_qps_update = now
    
    async def _performance_monitor(self):
        """性能监控后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                
                # 清理过期缓存
                await self._cleanup_local_cache()
                
                # 记录性能指标
                if self.query_count > 0:
                    cache_hit_rate = self.cache_hits / self.query_count
                    avg_query_time = self.total_query_time / self.query_count
                    
                    logger.info(f"性能统计 - 查询数: {self.query_count}, "
                              f"缓存命中率: {cache_hit_rate:.2%}, "
                              f"平均查询时间: {avg_query_time:.2f}ms")
                
            except Exception as e:
                logger.error(f"性能监控失败: {str(e)}")
    
    async def _cleanup_local_cache(self):
        """清理过期的本地缓存"""
        now = utc_now()
        expired_keys = [
            key for key, value in self.query_cache.items()
            if value["expires_at"] < now
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """使缓存失效"""
        try:
            if pattern:
                # 清理匹配模式的缓存
                if self.redis_client:
                    keys = await self.redis_client.keys(f"query:*{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                
                # 清理本地缓存
                keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.query_cache[key]
            else:
                # 清理所有缓存
                if self.redis_client:
                    keys = await self.redis_client.keys("query:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                
                self.query_cache.clear()
            
            logger.info(f"缓存清理完成，模式: {pattern or '全部'}")
            
        except Exception as e:
            logger.error(f"缓存清理失败: {str(e)}")
    
    async def optimize_indexes(self) -> Dict[str, Any]:
        """优化索引"""
        try:
            optimization_results = {
                "analyzed_queries": 0,
                "recommended_indexes": [],
                "existing_indexes": [],
                "performance_impact": {}
            }
            
            # 获取现有索引
            indexes_query = """
            CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state
            RETURN name, type, entityType, labelsOrTypes, properties, state
            """
            
            existing_indexes = await self.graph_db.execute_read_query(indexes_query)
            optimization_results["existing_indexes"] = existing_indexes
            
            # 分析慢查询模式
            slow_queries = [
                metric for metric in self.performance_metrics[-100:]
                if metric.execution_time_ms > self.slow_query_threshold
            ]
            
            optimization_results["analyzed_queries"] = len(slow_queries)
            
            # 基于查询模式推荐索引
            if slow_queries:
                # 这里可以实现更复杂的索引推荐逻辑
                # 暂时返回一些常用的推荐
                optimization_results["recommended_indexes"] = [
                    {
                        "type": "btree",
                        "properties": ["canonical_form", "type"],
                        "reason": "频繁的实体查找操作"
                    },
                    {
                        "type": "fulltext",
                        "properties": ["text", "canonical_form"],
                        "reason": "文本搜索查询优化"
                    }
                ]
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"索引优化失败: {str(e)}")
            return {"error": str(e)}
    
    async def get_performance_stats(self) -> PerformanceStats:
        """获取性能统计"""
        try:
            cache_hit_rate = self.cache_hits / self.query_count if self.query_count > 0 else 0.0
            avg_query_time = self.total_query_time / self.query_count if self.query_count > 0 else 0.0
            
            # 计算慢查询数量
            slow_queries = [
                m for m in self.performance_metrics
                if m.execution_time_ms > self.slow_query_threshold
            ]
            
            # 获取连接池状态
            connection_stats = self.graph_db.get_connection_stats()
            
            return PerformanceStats(
                total_queries=self.query_count,
                cache_hit_rate=cache_hit_rate,
                avg_query_time_ms=avg_query_time,
                slow_queries_count=len(slow_queries),
                peak_qps=self.current_qps,
                current_connections=connection_stats.get("active_connections", 0)
            )
            
        except Exception as e:
            logger.error(f"获取性能统计失败: {str(e)}")
            return PerformanceStats(
                total_queries=0,
                cache_hit_rate=0.0,
                avg_query_time_ms=0.0,
                slow_queries_count=0,
                peak_qps=0.0,
                current_connections=0
            )
    
    async def get_slow_queries(self, limit: int = 20) -> List[QueryPerformance]:
        """获取慢查询列表"""
        slow_queries = [
            metric for metric in self.performance_metrics
            if metric.execution_time_ms > self.slow_query_threshold
        ]
        
        # 按执行时间降序排序
        slow_queries.sort(key=lambda x: x.execution_time_ms, reverse=True)
        
        return slow_queries[:limit]
    
    async def analyze_query_patterns(self) -> Dict[str, Any]:
        """分析查询模式"""
        try:
            analysis = {
                "query_type_distribution": {},
                "execution_time_distribution": {},
                "cache_performance": {},
                "temporal_patterns": {}
            }
            
            if not self.performance_metrics:
                return analysis
            
            # 查询类型分布
            type_counts = {}
            for metric in self.performance_metrics:
                type_counts[metric.query_type] = type_counts.get(metric.query_type, 0) + 1
            analysis["query_type_distribution"] = type_counts
            
            # 执行时间分布
            time_buckets = {"<100ms": 0, "100-500ms": 0, "500-1000ms": 0, ">1000ms": 0}
            for metric in self.performance_metrics:
                if metric.execution_time_ms < 100:
                    time_buckets["<100ms"] += 1
                elif metric.execution_time_ms < 500:
                    time_buckets["100-500ms"] += 1
                elif metric.execution_time_ms < 1000:
                    time_buckets["500-1000ms"] += 1
                else:
                    time_buckets[">1000ms"] += 1
            analysis["execution_time_distribution"] = time_buckets
            
            # 缓存性能
            cache_hits = sum(1 for m in self.performance_metrics if m.cache_hit)
            total_cacheable = sum(1 for m in self.performance_metrics if m.query_type == "read")
            
            analysis["cache_performance"] = {
                "hit_rate": cache_hits / total_cacheable if total_cacheable > 0 else 0,
                "total_cacheable_queries": total_cacheable,
                "cache_hits": cache_hits
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"查询模式分析失败: {str(e)}")
            return {"error": str(e)}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "local_cache_size": len(self.query_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": self.cache_hits / self.query_count if self.query_count > 0 else 0,
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_ttl_seconds": self.cache_ttl
        }
