"""
pgvector性能优化器

提供pgvector 0.8升级、索引优化和向量搜索性能优化功能
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from .quantization import VectorQuantizer, QuantizationConfig, QuantizationMode

from src.core.logging import get_logger
logger = get_logger(__name__)

class IndexType(str, Enum):
    """索引类型"""
    HNSW = "hnsw"           # 分层导航小世界图
    IVF = "ivf"             # 倒排文件索引
    FLAT = "flat"           # 暴力搜索
    HYBRID = "hybrid"       # 混合索引

@dataclass
class IndexConfig:
    """索引配置"""
    index_type: IndexType
    hnsw_m: int = 16              # HNSW连接数
    hnsw_ef_construction: int = 200  # HNSW构建参数
    hnsw_ef_search: int = 100     # HNSW搜索参数
    ivf_lists: int = 1000         # IVF聚类数
    ivf_probes: int = 10          # IVF探测数

class PgVectorOptimizer:
    """pgvector性能优化器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.quantizer = VectorQuantizer(QuantizationConfig(QuantizationMode.ADAPTIVE))
        self.performance_stats = {
            "upgrades_performed": 0,
            "indexes_created": 0,
            "searches_optimized": 0,
            "average_search_latency_ms": 0.0
        }
        
    async def upgrade_to_v08(self) -> bool:
        """升级到pgvector 0.8"""
        try:
            # 检查当前版本
            current_version = await self._get_pgvector_version()
            logger.info(f"Current pgvector version: {current_version}")
            
            if current_version and self._version_compare(current_version, "0.8.0") >= 0:
                logger.info("pgvector is already at version 0.8+")
                return True
            
            # 执行升级SQL
            upgrade_sql = """
            -- 升级pgvector扩展
            ALTER EXTENSION vector UPDATE TO '0.8';
            
            -- 启用新的量化功能
            SET shared_preload_libraries = 'vector';
            
            -- 优化向量相关配置
            SET max_parallel_workers_per_gather = 4;
            SET effective_cache_size = '2GB';
            SET random_page_cost = 1.1;
            
            -- 创建新的向量操作符类（如果不存在）
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_opclass 
                    WHERE opcname = 'vector_l2_ops_quantized'
                ) THEN
                    CREATE OPERATOR CLASS vector_l2_ops_quantized
                    DEFAULT FOR TYPE vector USING hnsw AS
                    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
                    FUNCTION 1 vector_l2_distance(vector, vector);
                END IF;
            END
            $$;
            """
            
            await self.db.execute(text(upgrade_sql))
            await self.db.commit()
            
            # 验证升级
            new_version = await self._get_pgvector_version()
            logger.info(f"pgvector upgraded to version: {new_version}")
            
            success = new_version and self._version_compare(new_version, "0.8.0") >= 0
            if success:
                self.performance_stats["upgrades_performed"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"pgvector upgrade failed: {e}")
            await self.db.rollback()
            return False
    
    async def create_optimized_indexes(
        self, 
        table_name: str, 
        vector_column: str,
        config: IndexConfig
    ) -> bool:
        """创建优化的向量索引"""
        try:
            logger.info(f"Creating optimized index for {table_name}.{vector_column} with type {config.index_type}")
            
            if config.index_type == IndexType.HNSW:
                index_sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_{vector_column}_hnsw
                ON {table_name} 
                USING hnsw ({vector_column} vector_l2_ops)
                WITH (m = {config.hnsw_m}, ef_construction = {config.hnsw_ef_construction});
                """
            
            elif config.index_type == IndexType.IVF:
                index_sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_{vector_column}_ivf
                ON {table_name} 
                USING ivfflat ({vector_column} vector_l2_ops)
                WITH (lists = {config.ivf_lists});
                """
            
            elif config.index_type == IndexType.HYBRID:
                # 创建多层索引
                hnsw_sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_{vector_column}_hnsw
                ON {table_name} 
                USING hnsw ({vector_column} vector_l2_ops)
                WITH (m = {config.hnsw_m}, ef_construction = {config.hnsw_ef_construction});
                """
                
                ivf_sql = f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_{vector_column}_ivf_backup
                ON {table_name} 
                USING ivfflat ({vector_column} vector_l2_ops)
                WITH (lists = {config.ivf_lists});
                """
                
                await self.db.execute(text(hnsw_sql))
                await self.db.execute(text(ivf_sql))
                await self.db.commit()
                
                self.performance_stats["indexes_created"] += 2
                logger.info("Hybrid index created successfully")
                return True
            
            else:  # FLAT
                logger.info("FLAT index - no specific index created, will use sequential scan")
                return True
            
            await self.db.execute(text(index_sql))
            await self.db.commit()
            
            # 更新索引统计信息
            await self._update_index_statistics(table_name, vector_column)
            
            self.performance_stats["indexes_created"] += 1
            logger.info(f"Index created successfully for {table_name}.{vector_column}")
            return True
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            await self.db.rollback()
            return False
    
    async def optimize_vector_search(
        self,
        query_vector: np.ndarray,
        table_name: str,
        vector_column: str,
        top_k: int = 10,
        quantize: bool = True
    ) -> List[Dict[str, Any]]:
        """优化的向量搜索"""
        start_time = asyncio.get_running_loop().time()
        
        try:
            # 量化查询向量
            if quantize:
                query_vector, quant_params = await self.quantizer.quantize_vector(query_vector)
                logger.debug(f"Query vector quantized with mode: {quant_params.get('mode', 'unknown')}")
            
            # 动态选择搜索策略
            search_strategy = await self._select_search_strategy(table_name, top_k)
            logger.debug(f"Selected search strategy: {search_strategy}")
            
            if search_strategy == "hnsw":
                # 设置HNSW搜索参数
                await self.db.execute(text("SET hnsw.ef_search = 100"))
                
            search_sql = f"""
            SELECT id, content, metadata, 
                   {vector_column} <-> %s::vector AS distance
            FROM {table_name}
            ORDER BY {vector_column} <-> %s::vector
            LIMIT %s
            """
            
            # 将numpy数组转换为列表
            vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            
            result = await self.db.execute(
                text(search_sql), 
                (vector_list, vector_list, top_k)
            )
            
            results = [
                {
                    "id": row.id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "distance": float(row.distance)
                }
                for row in result.fetchall()
            ]
            
            # 更新性能统计
            end_time = asyncio.get_running_loop().time()
            latency_ms = (end_time - start_time) * 1000
            self._update_search_stats(latency_ms)
            
            logger.debug(f"Vector search completed in {latency_ms:.2f}ms, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _select_search_strategy(
        self, 
        table_name: str, 
        top_k: int
    ) -> str:
        """动态选择搜索策略"""
        try:
            # 获取表统计信息
            stats_sql = f"""
            SELECT 
                pg_relation_size('{table_name}') as table_size,
                (SELECT reltuples::bigint FROM pg_class WHERE relname = '{table_name}') as estimated_rows
            """
            
            result = await self.db.execute(text(stats_sql))
            row = result.fetchone()
            
            if row and row.estimated_rows:
                estimated_rows = row.estimated_rows
                # 根据数据量和查询要求选择策略
                if estimated_rows > 1000000 and top_k <= 20:
                    return "hnsw"  # 大数据量，小结果集用HNSW
                elif estimated_rows < 100000:
                    return "flat"  # 小数据量用暴力搜索
                else:
                    return "ivf"   # 中等数据量用IVF
            
            return "hnsw"  # 默认策略
            
        except Exception as e:
            logger.warning(f"Failed to determine search strategy: {e}, using default")
            return "hnsw"
    
    async def _get_pgvector_version(self) -> Optional[str]:
        """获取pgvector版本"""
        try:
            result = await self.db.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.fetchone()
            return row.extversion if row else None
        except Exception as e:
            logger.warning(f"Failed to get pgvector version: {e}")
            return None
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """比较版本号，返回 -1, 0, 或 1"""
        def normalize(v):
            return [int(x) for x in v.split('.')]
        
        v1_parts = normalize(version1)
        v2_parts = normalize(version2)
        
        # 补齐版本号长度
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for a, b in zip(v1_parts, v2_parts):
            if a < b:
                return -1
            elif a > b:
                return 1
        return 0
    
    async def _update_index_statistics(self, table_name: str, vector_column: str) -> None:
        """更新索引统计信息"""
        try:
            stats_sql = f"ANALYZE {table_name}"
            await self.db.execute(text(stats_sql))
            await self.db.commit()
            logger.debug(f"Updated statistics for {table_name}")
        except Exception as e:
            logger.warning(f"Failed to update index statistics: {e}")
    
    def _update_search_stats(self, latency_ms: float) -> None:
        """更新搜索性能统计"""
        self.performance_stats["searches_optimized"] += 1
        
        # 计算移动平均延迟
        current_avg = self.performance_stats["average_search_latency_ms"]
        n_searches = self.performance_stats["searches_optimized"]
        
        if n_searches == 1:
            self.performance_stats["average_search_latency_ms"] = latency_ms
        else:
            # 使用指数移动平均
            alpha = 0.1  # 平滑因子
            self.performance_stats["average_search_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * current_avg
            )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "stats": self.performance_stats.copy(),
            "quantizer_config": self.quantizer.get_quantization_stats(),
            "timestamp": utc_now().isoformat()
        }
    
    async def create_knowledge_items_table(self) -> bool:
        """创建知识库条目表（如果不存在）"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                embedding VECTOR(1536), -- OpenAI embeddings dimension
                embedding_quantized BYTEA,
                quantization_params_id UUID,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            -- 创建更新时间触发器
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            
            DROP TRIGGER IF EXISTS update_knowledge_items_updated_at ON knowledge_items;
            CREATE TRIGGER update_knowledge_items_updated_at
                BEFORE UPDATE ON knowledge_items
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
            
            await self.db.execute(text(create_table_sql))
            await self.db.commit()
            logger.info("Knowledge items table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create knowledge items table: {e}")
            await self.db.rollback()
            return False
    
    async def validate_installation(self) -> Dict[str, bool]:
        """验证pgvector安装和配置"""
        validation_results = {}
        
        try:
            # 检查扩展是否安装
            result = await self.db.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            validation_results["extension_installed"] = result.fetchone() is not None
            
            # 检查版本
            version = await self._get_pgvector_version()
            validation_results["version_08_or_higher"] = (
                version and self._version_compare(version, "0.8.0") >= 0
            )
            
            # 检查向量操作符
            result = await self.db.execute(
                text("SELECT 1 FROM pg_operator WHERE oprname = '<->'")
            )
            validation_results["operators_available"] = result.fetchone() is not None
            
            # 检查知识库表
            result = await self.db.execute(
                text("SELECT 1 FROM information_schema.tables WHERE table_name = 'knowledge_items'")
            )
            validation_results["knowledge_table_exists"] = result.fetchone() is not None
            
            logger.info(f"pgvector validation completed: {validation_results}")
            
        except Exception as e:
            logger.error(f"pgvector validation failed: {e}")
            validation_results["validation_error"] = str(e)
        
        return validation_results
