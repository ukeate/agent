"""
pgvector性能监控模块
收集和分析向量数据库的性能指标
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import asyncpg
import json
from dataclasses import dataclass, asdict
import time

from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class VectorQueryMetrics:
    """向量查询性能指标"""
    query_id: str
    collection_name: str
    query_type: str  # similarity_search, hybrid_search, etc.
    query_vector_dimension: int
    result_count: int
    execution_time_ms: float
    index_scan_time_ms: Optional[float]
    distance_metric: str
    filters_applied: bool
    cache_hit: bool
    timestamp: datetime


@dataclass
class VectorIndexMetrics:
    """向量索引性能指标"""
    collection_name: str
    index_name: str
    index_type: str  # hnsw, ivfflat
    index_size_bytes: int
    tuples_total: int
    index_scans: int
    tuples_read: int
    tuples_fetched: int
    build_time_ms: Optional[float]
    last_vacuum_time: Optional[datetime]
    fragmentation_ratio: float
    timestamp: datetime


@dataclass
class VectorSystemMetrics:
    """向量系统整体指标"""
    total_collections: int
    total_vectors: int
    total_storage_size_bytes: int
    active_connections: int
    queries_per_second: float
    average_query_time_ms: float
    cache_hit_ratio: float
    error_rate: float
    timestamp: datetime


class VectorMetricsCollector:
    """向量指标收集器"""
    
    def __init__(self, connection_url: str = None):
        self.connection_url = connection_url or settings.PGVECTOR_DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.is_running = False
        
    async def initialize(self):
        """初始化数据库连接和监控表"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=1,
                max_size=3,
                command_timeout=30
            )
            
            async with self.pool.acquire() as conn:
                # 创建指标存储表
                await self._create_metrics_tables(conn)
                
            logger.info("向量性能监控初始化完成")
        except Exception as e:
            logger.error(f"向量性能监控初始化失败: {e}")
            
    async def _create_metrics_tables(self, conn: asyncpg.Connection):
        """创建监控相关的数据表"""
        
        # 查询指标表
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_query_metrics (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(255) NOT NULL,
            collection_name VARCHAR(255) NOT NULL,
            query_type VARCHAR(100) NOT NULL,
            query_vector_dimension INTEGER,
            result_count INTEGER,
            execution_time_ms REAL NOT NULL,
            index_scan_time_ms REAL,
            distance_metric VARCHAR(50),
            filters_applied BOOLEAN DEFAULT FALSE,
            cache_hit BOOLEAN DEFAULT FALSE,
            timestamp TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        # 索引指标表
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_index_metrics (
            id SERIAL PRIMARY KEY,
            collection_name VARCHAR(255) NOT NULL,
            index_name VARCHAR(255) NOT NULL,
            index_type VARCHAR(50) NOT NULL,
            index_size_bytes BIGINT,
            tuples_total BIGINT,
            index_scans BIGINT,
            tuples_read BIGINT,
            tuples_fetched BIGINT,
            build_time_ms REAL,
            last_vacuum_time TIMESTAMP,
            fragmentation_ratio REAL,
            timestamp TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(collection_name, index_name, timestamp)
        )
        """)
        
        # 系统指标表
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS vector_system_metrics (
            id SERIAL PRIMARY KEY,
            total_collections INTEGER,
            total_vectors BIGINT,
            total_storage_size_bytes BIGINT,
            active_connections INTEGER,
            queries_per_second REAL,
            average_query_time_ms REAL,
            cache_hit_ratio REAL,
            error_rate REAL,
            timestamp TIMESTAMP DEFAULT NOW(),
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)
        
        # 创建时间序列索引
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON vector_query_metrics(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_index_metrics_timestamp ON vector_index_metrics(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON vector_system_metrics(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_metrics_collection ON vector_query_metrics(collection_name)")
        
    async def record_query_metrics(self, metrics: VectorQueryMetrics):
        """记录查询性能指标"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                INSERT INTO vector_query_metrics (
                    query_id, collection_name, query_type, query_vector_dimension,
                    result_count, execution_time_ms, index_scan_time_ms, 
                    distance_metric, filters_applied, cache_hit, timestamp
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                metrics.query_id, metrics.collection_name, metrics.query_type,
                metrics.query_vector_dimension, metrics.result_count,
                metrics.execution_time_ms, metrics.index_scan_time_ms,
                metrics.distance_metric, metrics.filters_applied,
                metrics.cache_hit, metrics.timestamp
                )
                
        except Exception as e:
            logger.error(f"记录查询指标失败: {e}")
            
    async def get_performance_report(
        self, 
        collection_name: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.pool:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                since_time = utc_now() - timedelta(hours=time_range_hours)
                
                # 查询性能统计
                query_where = "WHERE timestamp >= $1"
                params = [since_time]
                
                if collection_name:
                    query_where += " AND collection_name = $2"
                    params.append(collection_name)
                
                # 基础查询统计
                query_stats_result = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(execution_time_ms) as avg_execution_time,
                    MIN(execution_time_ms) as min_execution_time,
                    MAX(execution_time_ms) as max_execution_time
                FROM vector_query_metrics 
                {query_where}
                """, *params)
                
                query_stats = dict(query_stats_result) if query_stats_result else {
                    "total_queries": 0,
                    "avg_execution_time": 0,
                    "min_execution_time": 0,
                    "max_execution_time": 0
                }
                
                return {
                    "report_period": {
                        "start_time": since_time.isoformat(),
                        "end_time": utc_now().isoformat(),
                        "collection_name": collection_name
                    },
                    "query_performance": {
                        "total_queries": query_stats["total_queries"] or 0,
                        "average_execution_time_ms": float(query_stats["avg_execution_time"] or 0),
                        "min_execution_time_ms": float(query_stats["min_execution_time"] or 0),
                        "max_execution_time_ms": float(query_stats["max_execution_time"] or 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {"error": str(e)}


# 全局指标收集器实例
metrics_collector = VectorMetricsCollector()


async def get_metrics_collector() -> VectorMetricsCollector:
    """获取指标收集器实例"""
    if not metrics_collector.pool:
        await metrics_collector.initialize()
    return metrics_collector