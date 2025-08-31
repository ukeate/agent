"""
混合向量检索器（pgvector + Qdrant）

实现pgvector和Qdrant的混合检索，使用RRF算法融合结果
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from .pgvector_optimizer import PgVectorOptimizer
from .vector_cache import VectorCacheManager

logger = logging.getLogger(__name__)


class HybridVectorRetriever:
    """混合向量检索器（pgvector + Qdrant）"""
    
    def __init__(
        self, 
        pg_optimizer: PgVectorOptimizer,
        qdrant_client,
        cache_manager: VectorCacheManager,
        collection_name: str = "knowledge_base"
    ):
        self.pg_optimizer = pg_optimizer
        self.qdrant_client = qdrant_client
        self.cache_manager = cache_manager
        self.collection_name = collection_name
        self.retrieval_stats = {
            "hybrid_searches": 0,
            "pg_only_searches": 0,
            "qdrant_only_searches": 0,
            "cache_hits": 0,
            "average_latency_ms": 0.0
        }
        
    async def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: Optional[str] = None,
        top_k: int = 10,
        pg_weight: float = 0.7,
        qdrant_weight: float = 0.3,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """混合向量搜索"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 检查缓存
            if use_cache and query_text:
                cache_key = f"hybrid_search_{hash(query_text)}_{top_k}"
                cached_result = await self.cache_manager.get_cached_vector(cache_key)
                if cached_result:
                    self.retrieval_stats["cache_hits"] += 1
                    return cached_result[1].get("results", [])
            
            # 并行执行两个搜索
            pg_task = asyncio.create_task(
                self._pg_search(query_vector, top_k * 2)
            )
            
            qdrant_task = asyncio.create_task(
                self._qdrant_search(query_vector, top_k * 2)
            )
            
            pg_results, qdrant_results = await asyncio.gather(
                pg_task, qdrant_task, return_exceptions=True
            )
            
            # 处理异常
            if isinstance(pg_results, Exception):
                logger.error(f"PostgreSQL search failed: {pg_results}")
                pg_results = []
            
            if isinstance(qdrant_results, Exception):
                logger.error(f"Qdrant search failed: {qdrant_results}")
                qdrant_results = []
            
            # 融合搜索结果
            fused_results = await self._fuse_results(
                pg_results, qdrant_results, pg_weight, qdrant_weight
            )
            
            final_results = fused_results[:top_k]
            
            # 缓存结果
            if use_cache and query_text and final_results:
                await self.cache_manager.cache_vector(
                    cache_key,
                    query_vector,
                    {"results": final_results, "query_text": query_text}
                )
            
            # 更新统计
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            self._update_retrieval_stats("hybrid", latency_ms)
            
            logger.info(f"Hybrid search completed in {latency_ms:.2f}ms, returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _pg_search(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """PostgreSQL向量搜索"""
        try:
            results = await self.pg_optimizer.optimize_vector_search(
                query_vector, "knowledge_items", "embedding", top_k
            )
            logger.debug(f"PostgreSQL returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"PostgreSQL search error: {e}")
            return []
    
    async def _qdrant_search(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Qdrant向量搜索"""
        try:
            from qdrant_client.http.models import SearchRequest
            
            search_result = await asyncio.to_thread(
                self.qdrant_client.search,
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            results = [
                {
                    "id": str(point.id),
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "score": point.score
                }
                for point in search_result
            ]
            
            logger.debug(f"Qdrant returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []
    
    async def _fuse_results(
        self,
        pg_results: List[Dict[str, Any]],
        qdrant_results: List[Dict[str, Any]],
        pg_weight: float,
        qdrant_weight: float
    ) -> List[Dict[str, Any]]:
        """融合检索结果使用RRF (Reciprocal Rank Fusion) 算法"""
        result_map = {}
        
        # 处理pgvector结果（距离越小越好）
        for rank, result in enumerate(pg_results):
            result_id = str(result["id"])
            rrf_score = pg_weight / (rank + 1)
            
            if result_id not in result_map:
                result_map[result_id] = {
                    "id": result_id,
                    "content": result["content"],
                    "metadata": result.get("metadata", {}),
                    "pg_distance": result.get("distance"),
                    "pg_rank": rank + 1,
                    "qdrant_score": None,
                    "qdrant_rank": None,
                    "fused_score": rrf_score,
                    "sources": ["pgvector"]
                }
            else:
                result_map[result_id]["fused_score"] += rrf_score
        
        # 处理Qdrant结果（分数越高越好）
        for rank, result in enumerate(qdrant_results):
            result_id = str(result["id"])
            rrf_score = qdrant_weight / (rank + 1)
            
            if result_id not in result_map:
                result_map[result_id] = {
                    "id": result_id,
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "pg_distance": None,
                    "pg_rank": None,
                    "qdrant_score": result.get("score"),
                    "qdrant_rank": rank + 1,
                    "fused_score": rrf_score,
                    "sources": ["qdrant"]
                }
            else:
                result_map[result_id]["qdrant_score"] = result.get("score")
                result_map[result_id]["qdrant_rank"] = rank + 1
                result_map[result_id]["fused_score"] += rrf_score
                result_map[result_id]["sources"].append("qdrant")
        
        # 按融合分数排序
        fused_results = sorted(
            result_map.values(),
            key=lambda x: x["fused_score"],
            reverse=True
        )
        
        logger.debug(f"Fused {len(fused_results)} unique results from {len(pg_results)} PG + {len(qdrant_results)} Qdrant")
        
        return fused_results
    
    async def pg_only_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """仅使用PostgreSQL搜索"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            results = await self._pg_search(query_vector, top_k)
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            self._update_retrieval_stats("pg_only", latency_ms)
            
            return results
            
        except Exception as e:
            logger.error(f"PostgreSQL-only search failed: {e}")
            return []
    
    async def qdrant_only_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """仅使用Qdrant搜索"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            results = await self._qdrant_search(query_vector, top_k)
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            self._update_retrieval_stats("qdrant_only", latency_ms)
            
            return results
            
        except Exception as e:
            logger.error(f"Qdrant-only search failed: {e}")
            return []
    
    def _update_retrieval_stats(self, search_type: str, latency_ms: float) -> None:
        """更新检索统计"""
        if search_type == "hybrid":
            self.retrieval_stats["hybrid_searches"] += 1
        elif search_type == "pg_only":
            self.retrieval_stats["pg_only_searches"] += 1
        elif search_type == "qdrant_only":
            self.retrieval_stats["qdrant_only_searches"] += 1
        
        # 更新平均延迟
        total_searches = (
            self.retrieval_stats["hybrid_searches"] +
            self.retrieval_stats["pg_only_searches"] +
            self.retrieval_stats["qdrant_only_searches"]
        )
        
        if total_searches == 1:
            self.retrieval_stats["average_latency_ms"] = latency_ms
        else:
            # 使用指数移动平均
            alpha = 0.1
            current_avg = self.retrieval_stats["average_latency_ms"]
            self.retrieval_stats["average_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * current_avg
            )
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        cache_stats = await self.cache_manager.get_cache_stats()
        
        total_searches = (
            self.retrieval_stats["hybrid_searches"] +
            self.retrieval_stats["pg_only_searches"] +
            self.retrieval_stats["qdrant_only_searches"]
        )
        
        return {
            "search_stats": self.retrieval_stats.copy(),
            "cache_stats": cache_stats,
            "total_searches": total_searches,
            "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
            "timestamp": utc_now().isoformat()
        }
    
    async def benchmark_retrieval_methods(
        self,
        test_vectors: List[np.ndarray],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """对比不同检索方法的性能"""
        benchmark_results = {
            "test_vectors_count": len(test_vectors),
            "top_k": top_k,
            "methods": {}
        }
        
        methods = [
            ("hybrid", self.hybrid_search),
            ("pg_only", self.pg_only_search),
            ("qdrant_only", self.qdrant_only_search)
        ]
        
        for method_name, method_func in methods:
            start_time = asyncio.get_event_loop().time()
            total_results = 0
            errors = 0
            
            for vector in test_vectors:
                try:
                    results = await method_func(vector, top_k)
                    total_results += len(results)
                except Exception as e:
                    logger.error(f"Benchmark error for {method_name}: {e}")
                    errors += 1
            
            end_time = asyncio.get_event_loop().time()
            total_time_ms = (end_time - start_time) * 1000
            
            benchmark_results["methods"][method_name] = {
                "total_time_ms": total_time_ms,
                "average_time_per_query_ms": total_time_ms / len(test_vectors),
                "total_results": total_results,
                "average_results_per_query": total_results / len(test_vectors),
                "error_count": errors,
                "success_rate": (len(test_vectors) - errors) / len(test_vectors)
            }
        
        logger.info("Retrieval method benchmark completed")
        return benchmark_results
    
    async def health_check(self) -> Dict[str, Any]:
        """检索系统健康检查"""
        health_status = "healthy"
        issues = []
        
        try:
            # 测试pgvector连接
            pg_validation = await self.pg_optimizer.validate_installation()
            if not all(pg_validation.values()):
                health_status = "degraded"
                issues.append("PostgreSQL/pgvector issues detected")
            
            # 测试Qdrant连接
            try:
                collections = await asyncio.to_thread(
                    self.qdrant_client.get_collections
                )
                qdrant_healthy = True
            except Exception as e:
                qdrant_healthy = False
                issues.append(f"Qdrant connection error: {str(e)}")
            
            # 测试缓存系统
            cache_health = await self.cache_manager.get_cache_health()
            if cache_health["status"] != "healthy":
                health_status = "warning"
                issues.extend([f"Cache: {issue}" for issue in cache_health["issues"]])
            
            if not qdrant_healthy and health_status != "degraded":
                health_status = "warning"
            
        except Exception as e:
            health_status = "unhealthy"
            issues.append(f"Health check failed: {str(e)}")
        
        return {
            "status": health_status,
            "issues": issues,
            "components": {
                "pgvector": pg_validation if 'pg_validation' in locals() else {},
                "qdrant": {"healthy": qdrant_healthy if 'qdrant_healthy' in locals() else False},
                "cache": cache_health if 'cache_health' in locals() else {}
            },
            "stats": await self.get_retrieval_stats(),
            "timestamp": utc_now().isoformat()
        }