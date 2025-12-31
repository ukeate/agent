"""
pgvector 0.8.0 优化的向量存储实现
支持HNSW和IVFFlat索引、向量量化、性能监控
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncpg
import numpy as np
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import json
from contextlib import asynccontextmanager
from ...core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class PgVectorStore:
    """优化的pgvector向量存储"""
    
    def __init__(self, connection_url: str = None):
        self.connection_url = connection_url or settings.PGVECTOR_DATABASE_URL
        self.pool: Optional[asyncpg.Pool] = None
        self._performance_metrics = {
            "query_count": 0,
            "total_query_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def initialize_pool(self):
        """初始化数据库连接池"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=2,
                max_size=settings.VECTOR_MAX_CONNECTIONS,
                command_timeout=settings.VECTOR_QUERY_TIMEOUT,
                server_settings={
                    'application_name': 'ai_agent_vector_store',
                    # 这些参数只能在会话级别设置，不能作为服务器设置
                    'hnsw.ef_search': str(settings.HNSW_EF_SEARCH),
                    'ivfflat.probes': str(settings.IVFFLAT_PROBES),
                }
            )
            logger.info("pgvector连接池初始化完成")
            
    async def close_pool(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            
    @asynccontextmanager
    async def get_connection(self):
        """获取数据库连接"""
        if not self.pool:
            await self.initialize_pool()
            
        async with self.pool.acquire() as conn:
            yield conn
            
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        index_type: str = "hnsw",
        distance_metric: str = "l2",
        index_options: Dict[str, Any] = None
    ) -> bool:
        """
        创建向量集合（表）
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
            index_type: 索引类型 (hnsw, ivfflat)
            distance_metric: 距离度量 (l2, cosine, ip, l1)
            index_options: 索引选项
        """
        try:
            # 验证集合名称防止SQL注入
            if not collection_name.replace('_', '').isalnum():
                raise ValueError(f"无效的集合名称: {collection_name}")
            
            async with self.get_connection() as conn:
                # 创建表
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS "{collection_name}" (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    embedding vector({dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
                """
                await conn.execute(create_table_sql)
                
                # 创建优化索引
                await self._create_optimized_index(
                    conn, 
                    collection_name, 
                    "embedding",
                    index_type,
                    distance_metric,
                    index_options or {}
                )
                
                # 创建元数据索引
                await conn.execute(f'CREATE INDEX IF NOT EXISTS "{collection_name}_metadata_idx" ON "{collection_name}" USING GIN (metadata)')
                await conn.execute(f'CREATE INDEX IF NOT EXISTS "{collection_name}_created_at_idx" ON "{collection_name}" (created_at)')
                
                logger.info(f"向量集合 {collection_name} 创建成功，维度: {dimension}, 索引: {index_type}")
                return True
                
        except Exception as e:
            logger.error(f"创建向量集合失败: {e}")
            return False
            
    async def _create_optimized_index(
        self,
        conn: asyncpg.Connection,
        table_name: str,
        column_name: str,
        index_type: str,
        distance_metric: str,
        index_options: Dict[str, Any]
    ):
        """创建优化的向量索引"""
        
        # 确定操作符类
        ops_classes = {
            "l2": "vector_l2_ops",
            "cosine": "vector_cosine_ops", 
            "ip": "vector_ip_ops",
            "l1": "vector_l1_ops"
        }
        
        ops_class = ops_classes.get(distance_metric, "vector_l2_ops")
        index_name = f"{table_name}_{column_name}_{index_type}_idx"
        
        if index_type == "hnsw":
            # HNSW索引优化配置
            m = index_options.get("m", settings.HNSW_M)
            ef_construction = index_options.get("ef_construction", settings.HNSW_EF_CONSTRUCTION)
            
            index_sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} 
            ON {table_name} 
            USING hnsw ({column_name} {ops_class}) 
            WITH (m = {m}, ef_construction = {ef_construction})
            """
            
        elif index_type == "ivfflat":
            # IVFFlat索引优化配置
            lists = index_options.get("lists", settings.IVFFLAT_LISTS)
            
            index_sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
            ON {table_name}
            USING ivfflat ({column_name} {ops_class})
            WITH (lists = {lists})
            """
            
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
            
        await conn.execute(index_sql)
        logger.info(f"优化索引创建完成: {index_name}")
        
    async def insert_vectors(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = None
    ) -> List[str]:
        """
        批量插入向量
        
        Args:
            collection_name: 集合名称
            documents: 文档列表，每个包含 content, embedding, metadata
            batch_size: 批处理大小
            
        Returns:
            插入的文档ID列表
        """
        batch_size = batch_size or settings.VECTOR_BATCH_SIZE
        inserted_ids = []
        
        try:
            async with self.get_connection() as conn:
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    # 准备批量插入数据
                    values = []
                    for doc in batch:
                        embedding_str = self._format_vector_for_postgres(doc["embedding"])
                        values.append((
                            doc["content"],
                            embedding_str,
                            json.dumps(doc.get("metadata", {}))
                        ))
                    
                    # 验证集合名称防止SQL注入
                    if not collection_name.replace('_', '').isalnum():
                        raise ValueError(f"无效的集合名称: {collection_name}")
                    
                    # 执行批量插入
                    insert_sql = f"""
                    INSERT INTO "{collection_name}" (content, embedding, metadata)
                    VALUES ($1, $2, $3::jsonb)
                    RETURNING id
                    """
                    
                    batch_ids = []
                    for value in values:
                        result = await conn.fetchval(insert_sql, *value)
                        batch_ids.append(str(result))
                    
                    inserted_ids.extend(batch_ids)
                    
                    if settings.VECTOR_PERFORMANCE_LOGGING:
                        logger.info(f"批量插入完成: {len(batch)} 条记录")
                        
            logger.info(f"向量插入完成: {len(inserted_ids)} 条记录")
            return inserted_ids
            
        except Exception as e:
            logger.error(f"向量插入失败: {e}")
            raise
            
    async def similarity_search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 10,
        distance_metric: str = "l2",
        filters: Dict[str, Any] = None,
        include_distances: bool = False
    ) -> List[Dict[str, Any]]:
        """
        向量相似性搜索
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            limit: 返回结果数量
            distance_metric: 距离度量
            filters: 元数据过滤器
            include_distances: 是否包含距离值
            
        Returns:
            搜索结果列表
        """
        start_time = utc_now()
        
        try:
            async with self.get_connection() as conn:
                # 格式化查询向量
                query_vector_str = self._format_vector_for_postgres(query_vector)
                
                # 选择距离操作符
                distance_ops = {
                    "l2": "<->",
                    "cosine": "<=>", 
                    "ip": "<#>",
                    "l1": "<+>"
                }
                
                distance_op = distance_ops.get(distance_metric, "<->")
                
                # 构建基础查询
                select_fields = ["id", "content", "metadata", "created_at"]
                if include_distances:
                    select_fields.append(f"embedding {distance_op} '{query_vector_str}' AS distance")
                
                select_clause = ", ".join(select_fields)
                
                # 构建WHERE子句
                where_conditions = []
                params = []
                param_count = 1
                
                if filters:
                    for key, value in filters.items():
                        where_conditions.append(f"metadata->>${param_count} = ${param_count + 1}")
                        params.extend([key, json.dumps(value)])
                        param_count += 2
                
                where_clause = ""
                if where_conditions:
                    where_clause = f"WHERE {' AND '.join(where_conditions)}"
                
                # 构建完整查询
                query_sql = f"""
                SELECT {select_clause}
                FROM {collection_name}
                {where_clause}
                ORDER BY embedding {distance_op} '{query_vector_str}'
                LIMIT {limit}
                """
                
                # 执行查询
                if params:
                    results = await conn.fetch(query_sql, *params)
                else:
                    results = await conn.fetch(query_sql)
                
                # 转换结果
                search_results = []
                for row in results:
                    result = {
                        "id": str(row["id"]),
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"].isoformat()
                    }
                    
                    if include_distances:
                        result["distance"] = float(row["distance"])
                    
                    search_results.append(result)
                
                # 更新性能指标
                query_time = (utc_now() - start_time).total_seconds()
                self._update_performance_metrics(query_time)
                
                if settings.VECTOR_PERFORMANCE_LOGGING and query_time > settings.VECTOR_SLOW_QUERY_THRESHOLD:
                    logger.warning(f"慢查询检测: {query_time:.2f}秒, 查询: {collection_name}")
                
                return search_results
                
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            raise
            
    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        query_text: str = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        混合搜索：向量搜索 + 全文搜索
        
        Args:
            collection_name: 集合名称
            query_vector: 查询向量
            query_text: 查询文本
            limit: 返回结果数量
            vector_weight: 向量搜索权重
            text_weight: 文本搜索权重
            filters: 元数据过滤器
            
        Returns:
            混合搜索结果
        """
        try:
            async with self.get_connection() as conn:
                query_vector_str = self._format_vector_for_postgres(query_vector)
                
                # 构建混合查询
                if query_text:
                    # 使用RRF (Reciprocal Rank Fusion) 算法
                    hybrid_sql = f"""
                    WITH vector_search AS (
                        SELECT id, content, metadata, created_at,
                               embedding <-> '{query_vector_str}' AS vector_distance,
                               ROW_NUMBER() OVER (ORDER BY embedding <-> '{query_vector_str}') AS vector_rank
                        FROM {collection_name}
                        ORDER BY embedding <-> '{query_vector_str}'
                        LIMIT {limit * 2}
                    ),
                    text_search AS (
                        SELECT id, content, metadata, created_at,
                               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $1)) AS text_score,
                               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $1)) DESC) AS text_rank
                        FROM {collection_name}
                        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                        ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $1)) DESC
                        LIMIT {limit * 2}
                    ),
                    combined AS (
                        SELECT COALESCE(v.id, t.id) AS id,
                               COALESCE(v.content, t.content) AS content,
                               COALESCE(v.metadata, t.metadata) AS metadata,
                               COALESCE(v.created_at, t.created_at) AS created_at,
                               v.vector_distance,
                               t.text_score,
                               COALESCE(1.0 / (60 + v.vector_rank), 0) * {vector_weight} +
                               COALESCE(1.0 / (60 + t.text_rank), 0) * {text_weight} AS combined_score
                        FROM vector_search v
                        FULL OUTER JOIN text_search t ON v.id = t.id
                    )
                    SELECT id, content, metadata, created_at, vector_distance, text_score, combined_score
                    FROM combined
                    ORDER BY combined_score DESC
                    LIMIT {limit}
                    """
                    
                    results = await conn.fetch(hybrid_sql, query_text)
                else:
                    # 仅向量搜索
                    return await self.similarity_search(
                        collection_name, query_vector, limit, "l2", filters, True
                    )
                
                # 转换结果
                search_results = []
                for row in results:
                    result = {
                        "id": str(row["id"]),
                        "content": row["content"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"].isoformat(),
                        "vector_distance": float(row["vector_distance"]) if row["vector_distance"] else None,
                        "text_score": float(row["text_score"]) if row["text_score"] else None,
                        "combined_score": float(row["combined_score"])
                    }
                    search_results.append(result)
                
                return search_results
                
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            raise
            
    def _format_vector_for_postgres(self, vector: Union[List[float], np.ndarray]) -> str:
        """格式化向量为PostgreSQL格式"""
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return f"[{','.join(map(str, vector))}]"
        
    def _update_performance_metrics(self, query_time: float):
        """更新性能指标"""
        self._performance_metrics["query_count"] += 1
        self._performance_metrics["total_query_time"] += query_time
        
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            async with self.get_connection() as conn:
                # 基础统计
                basic_stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_vectors,
                    pg_size_pretty(pg_total_relation_size('{collection_name}')) as table_size,
                    pg_stat_get_live_tuples('{collection_name}'::regclass) as live_tuples,
                    pg_stat_get_dead_tuples('{collection_name}'::regclass) as dead_tuples
                FROM {collection_name}
                """)
                
                # 索引统计
                index_stats = await conn.fetch(f"""
                SELECT 
                    indexrelname as indexname,
                    pg_size_pretty(pg_relation_size(indexrelname::regclass)) as index_size,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE tablename = '{collection_name}'
                """)
                
                return {
                    "collection_name": collection_name,
                    "total_vectors": basic_stats["total_vectors"],
                    "table_size": basic_stats["table_size"],
                    "live_tuples": basic_stats["live_tuples"],
                    "dead_tuples": basic_stats["dead_tuples"],
                    "indexes": [dict(row) for row in index_stats],
                    "performance_metrics": self._performance_metrics.copy()
                }
                
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {}
            
    async def optimize_collection(self, collection_name: str) -> bool:
        """优化集合（重建索引、清理死元组等）"""
        try:
            async with self.get_connection() as conn:
                # VACUUM ANALYZE
                await conn.execute(f"VACUUM ANALYZE {collection_name}")
                
                # 重建索引
                indexes = await conn.fetch(f"""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = '{collection_name}'
                AND indexdef LIKE '%vector%'
                """)
                
                for index_row in indexes:
                    index_name = index_row["indexname"]
                    await conn.execute(f"REINDEX INDEX CONCURRENTLY {index_name}")
                    logger.info(f"索引重建完成: {index_name}")
                
                logger.info(f"集合优化完成: {collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"集合优化失败: {e}")
            return False

    async def create_index(
        self,
        table_name: str,
        column_name: str,
        index_type: str,
        distance_metric: str,
        index_options: Dict[str, Any] = None,
    ) -> bool:
        """为已有表创建向量索引"""
        try:
            async with self.get_connection() as conn:
                await self._create_optimized_index(
                    conn,
                    table_name,
                    column_name,
                    index_type,
                    distance_metric,
                    index_options or {},
                )
                logger.info(
                    "向量索引创建完成",
                    extra={
                        "table": table_name,
                        "column": column_name,
                        "index_type": index_type,
                        "distance_metric": distance_metric,
                    },
                )
                return True
        except Exception as e:
            logger.error(f"创建向量索引失败: {e}")
            return False

    async def list_indexes(self) -> list[dict[str, Any]]:
        """列出当前数据库中的向量索引"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT schemaname, tablename, indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname NOT IN ('pg_catalog','information_schema')
                    ORDER BY tablename, indexname
                    """
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"获取索引列表失败: {e}")
            return []

# 全局向量存储实例
vector_store = PgVectorStore()

async def get_vector_store() -> PgVectorStore:
    """获取向量存储实例"""
    if not vector_store.pool:
        await vector_store.initialize_pool()
    return vector_store
