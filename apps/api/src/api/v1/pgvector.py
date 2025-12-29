"""
pgvector 0.8.0 API端点
提供向量数据库管理、性能监控、量化配置等功能
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.ai.rag.vector_store import get_vector_store, PgVectorStore
from src.ai.rag.hybrid_search import get_hybrid_search_engine, SearchStrategy
from src.ai.rag.data_integrity import VectorDataIntegrityValidator
from src.ai.rag.quantization import get_quantization_manager
from src.core.monitoring.vector_db_metrics import get_metrics_collector, VectorQueryMetrics
from src.ai.rag.embeddings import EmbeddingService
from src.core.config import get_settings
from src.core.database import get_db
from src.core.redis import get_redis
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

router = APIRouter(prefix="/pgvector", tags=["pgvector"])

def _openai_key_configured() -> bool:
    key = (settings.OPENAI_API_KEY or "").strip()
    return bool(key) and not key.startswith("sk-test")

def _parse_time_range_hours(time_range: str) -> int:
    """将前端传入的时间范围转换为小时整数"""
    try:
        if time_range.endswith("h"):
            return int(time_range[:-1])
        if time_range.endswith("d"):
            return int(time_range[:-1]) * 24
        return int(time_range)
    except Exception:
        return 1

def _validate_collection_name(collection_name: str) -> None:
    if not collection_name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="无效的集合名称")

def _normalize_weights(pg_weight: float, qdrant_weight: float) -> tuple[float, float]:
    total = pg_weight + qdrant_weight
    if total <= 0:
        return 0.5, 0.5
    return pg_weight / total, qdrant_weight / total

async def _ensure_pg_collection(
    vector_store: PgVectorStore,
    collection_name: str,
    dimension: int,
) -> None:
    async with vector_store.get_connection() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = $1
            )
            """,
            collection_name,
        )
    if not exists:
        created = await vector_store.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )
        if not created:
            raise HTTPException(status_code=500, detail="创建pgvector集合失败")

def _fuse_results(
    pg_results: List[Dict[str, Any]],
    qdrant_results: List[Dict[str, Any]],
    pg_weight: float,
    qdrant_weight: float,
    limit: int,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    pg_weight, qdrant_weight = _normalize_weights(pg_weight, qdrant_weight)
    fused: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(pg_results):
        result_id = str(result["id"])
        entry = fused.get(result_id)
        if not entry:
            entry = {
                "id": result_id,
                "content": result.get("content", ""),
                "metadata": result.get("metadata") or {},
                "pg_distance": None,
                "qdrant_score": None,
                "fused_score": 0.0,
                "sources": [],
                "pg_rank": None,
                "qdrant_rank": None,
            }
            fused[result_id] = entry
        entry["pg_distance"] = result.get("distance")
        entry["pg_rank"] = rank + 1
        entry["fused_score"] += pg_weight / (rrf_k + rank + 1)
        if "pgvector" not in entry["sources"]:
            entry["sources"].append("pgvector")

    for rank, result in enumerate(qdrant_results):
        result_id = str(result["id"])
        entry = fused.get(result_id)
        if not entry:
            entry = {
                "id": result_id,
                "content": result.get("content", ""),
                "metadata": result.get("metadata") or {},
                "pg_distance": None,
                "qdrant_score": None,
                "fused_score": 0.0,
                "sources": [],
                "pg_rank": None,
                "qdrant_rank": None,
            }
            fused[result_id] = entry
        if not entry.get("content"):
            entry["content"] = result.get("content", "")
        if not entry.get("metadata"):
            entry["metadata"] = result.get("metadata") or {}
        entry["qdrant_score"] = result.get("score")
        entry["qdrant_rank"] = rank + 1
        entry["fused_score"] += qdrant_weight / (rrf_k + rank + 1)
        if "qdrant" not in entry["sources"]:
            entry["sources"].append("qdrant")

    results = list(fused.values())
    for entry in results:
        entry["vector_distance"] = entry["pg_distance"]
        entry["text_score"] = entry["qdrant_score"]
        entry["combined_score"] = entry["fused_score"]

    results.sort(key=lambda x: x["fused_score"], reverse=True)
    return results[:limit]

async def _record_cache_stats(cache_hit: bool) -> None:
    redis = get_redis()
    if not redis:
        return
    key = "pgvector:hybrid_cache:stats"
    field = "hits" if cache_hit else "misses"
    try:
        await redis.hincrby(key, field, 1)
    except Exception as e:
        logger.warning(f"缓存统计更新失败: {e}")

async def _run_pgvector_search(
    vector_store: PgVectorStore,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    filters: Optional[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], float]:
    start = time.perf_counter()
    results = await vector_store.similarity_search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        distance_metric="l2",
        filters=filters,
        include_distances=True,
    )
    return results, (time.perf_counter() - start) * 1000

async def _run_qdrant_search(
    query: str,
    collection_name: str,
    limit: int,
    filters: Optional[Dict[str, Any]],
    use_cache: bool,
) -> tuple[List[Dict[str, Any]], float, bool]:
    engine = get_hybrid_search_engine()
    if not use_cache:
        engine.cache = None
    before_hits = engine.cache.hit_count if engine.cache else 0
    start = time.perf_counter()
    results = await engine.search(
        query=query,
        collection=collection_name,
        limit=limit,
        filters=filters,
        strategy=SearchStrategy.BM25_ONLY,
    )
    elapsed = (time.perf_counter() - start) * 1000
    cache_hit = bool(engine.cache and engine.cache.hit_count > before_hits)
    mapped = [
        {
            "id": r.id,
            "score": r.score,
            "content": r.content,
            "metadata": r.metadata,
        }
        for r in results
    ]
    return mapped, elapsed, cache_hit

async def _execute_hybrid_search(
    *,
    vector_store: PgVectorStore,
    collection_name: str,
    query: str,
    top_k: int,
    pg_weight: float,
    qdrant_weight: float,
    search_mode: str,
    use_cache: bool,
    query_vector: Optional[List[float]],
    filters: Optional[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k 必须大于 0")
    _validate_collection_name(collection_name)

    mode = search_mode
    if mode not in {"hybrid", "pg_only", "qdrant_only"}:
        raise HTTPException(status_code=400, detail="不支持的 search_mode")

    if mode == "pg_only":
        pg_weight, qdrant_weight = 1.0, 0.0
    elif mode == "qdrant_only":
        pg_weight, qdrant_weight = 0.0, 1.0

    need_pg = mode in {"hybrid", "pg_only"}
    if need_pg:
        embedder = EmbeddingService()
        await _ensure_pg_collection(
            vector_store=vector_store,
            collection_name=collection_name,
            dimension=embedder.dimension,
        )
        if query_vector is None:
            if not _openai_key_configured():
                raise HTTPException(status_code=400, detail="未配置OpenAI密钥，无法生成查询向量")
            query_vector = await embedder.embed_text(query)
        if not query_vector:
            raise HTTPException(status_code=500, detail="生成查询向量失败")

    pg_results: List[Dict[str, Any]] = []
    qdrant_results: List[Dict[str, Any]] = []
    pg_time_ms = 0.0
    qdrant_time_ms = 0.0
    cache_hit = False

    pg_limit = top_k * 2 if mode == "hybrid" else top_k
    qdrant_limit = top_k * 2 if mode == "hybrid" else top_k

    if mode == "hybrid":
        pg_task = asyncio.create_task(
            _run_pgvector_search(
                vector_store=vector_store,
                collection_name=collection_name,
                query_vector=query_vector,
                limit=pg_limit,
                filters=filters,
            )
        )
        qdrant_task = asyncio.create_task(
            _run_qdrant_search(
                query=query,
                collection_name=collection_name,
                limit=qdrant_limit,
                filters=filters,
                use_cache=use_cache,
            )
        )
        (pg_results, pg_time_ms), (qdrant_results, qdrant_time_ms, cache_hit) = await asyncio.gather(
            pg_task, qdrant_task
        )
    elif mode == "pg_only":
        pg_results, pg_time_ms = await _run_pgvector_search(
            vector_store=vector_store,
            collection_name=collection_name,
            query_vector=query_vector,
            limit=pg_limit,
            filters=filters,
        )
    else:
        qdrant_results, qdrant_time_ms, cache_hit = await _run_qdrant_search(
            query=query,
            collection_name=collection_name,
            limit=qdrant_limit,
            filters=filters,
            use_cache=use_cache,
        )

    fusion_start = time.perf_counter()
    fused_results = _fuse_results(pg_results, qdrant_results, pg_weight, qdrant_weight, top_k)
    fusion_time_ms = (time.perf_counter() - fusion_start) * 1000

    metrics = {
        "pg_time_ms": pg_time_ms,
        "qdrant_time_ms": qdrant_time_ms,
        "fusion_time_ms": fusion_time_ms,
        "cache_hit": cache_hit,
        "results_count": len(fused_results),
        "semantic_time_ms": pg_time_ms,
        "keyword_time_ms": qdrant_time_ms,
        "total_candidates": len(pg_results) + len(qdrant_results),
        "semantic_candidates": len(pg_results),
        "keyword_candidates": len(qdrant_results),
    }

    return fused_results, metrics, {
        "pg_results": pg_results,
        "qdrant_results": qdrant_results,
        "cache_hit": cache_hit,
        "query_vector": query_vector,
    }

@router.get("/status", summary="获取pgvector系统状态")
async def get_status(vector_store: PgVectorStore = Depends(get_vector_store)):
    """返回pgvector版本与索引/缓存状态"""
    try:
        cache_status = "warning"
        redis = get_redis()
        if redis and settings.CACHE_ENABLED:
            try:
                await redis.ping()
                cache_status = "healthy"
            except Exception:
                cache_status = "error"
        elif settings.CACHE_ENABLED:
            cache_status = "error"

        async with vector_store.get_connection() as conn:
            version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            stats = await conn.fetchrow(
                """
                SELECT sum(pg_total_relation_size(c.oid)) AS total_size
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public' AND c.relkind = 'r'
                """
            )
            index_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM pg_indexes
                WHERE indexdef ILIKE '%hnsw%'
                   OR indexdef ILIKE '%ivfflat%'
                   OR indexdef ILIKE '%vector%'
                """
            )
        upgrade_available = version and version < "0.8.0"
        index_health = "optimal" if index_count and index_count > 0 else "needs_optimization"
        return {
            "pgvector_version": version or "unknown",
            "upgrade_available": bool(upgrade_available),
            "quantization_enabled": settings.VECTOR_QUANTIZATION_ENABLED,
            "cache_status": cache_status,
            "index_health": index_health,
            "last_updated": utc_now().isoformat(),
            "total_size_bytes": stats["total_size"] if stats else 0
        }
    except Exception as e:
        logger.error(f"获取pgvector状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upgrade", summary="升级pgvector版本")
async def upgrade_pgvector(vector_store: PgVectorStore = Depends(get_vector_store)):
    """尝试升级pgvector扩展版本"""
    try:
        async with vector_store.get_connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("ALTER EXTENSION vector UPDATE")
            version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
        return {
            "success": True,
            "message": f"pgvector已升级到 {version or 'unknown'}"
        }
    except Exception as e:
        logger.error(f"升级pgvector失败: {e}")
        raise HTTPException(status_code=500, detail=f"升级失败: {str(e)}")

# 请求和响应模型
class CreateCollectionRequest(ApiBaseModel):
    collection_name: str = Field(..., description="集合名称")
    dimension: int = Field(..., description="向量维度")
    index_type: str = Field(default="hnsw", description="索引类型 (hnsw, ivfflat)")
    distance_metric: str = Field(default="l2", description="距离度量 (l2, cosine, ip, l1)")
    index_options: Dict[str, Any] = Field(default_factory=dict, description="索引选项")

class SimilaritySearchRequest(ApiBaseModel):
    collection_name: str = Field(..., description="集合名称")
    query_vector: List[float] = Field(..., description="查询向量")
    limit: int = Field(default=10, description="返回结果数量")
    distance_metric: str = Field(default="l2", description="距离度量")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤器")
    include_distances: bool = Field(default=False, description="是否包含距离值")

class HybridSearchRequest(ApiBaseModel):
    collection_name: str = Field(default="documents", description="集合名称")
    query_vector: Optional[List[float]] = Field(default=None, description="查询向量")
    query: str = Field(..., description="查询文本")
    top_k: int = Field(default=10, description="返回结果数量")
    pg_weight: float = Field(default=0.7, description="pgvector权重")
    qdrant_weight: float = Field(default=0.3, description="qdrant权重")
    search_mode: str = Field(default="hybrid", description="检索模式: hybrid/pg_only/qdrant_only")
    use_cache: bool = Field(default=False, description="是否使用缓存")
    quantize: bool = Field(default=False, description="是否启用量化")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="元数据过滤器")

class VectorDocument(ApiBaseModel):
    content: str = Field(..., description="文档内容")
    embedding: List[float] = Field(..., description="向量表示")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")

class InsertVectorsRequest(ApiBaseModel):
    collection_name: str = Field(..., description="集合名称")
    documents: List[VectorDocument] = Field(..., description="向量文档列表")
    batch_size: Optional[int] = Field(default=100, description="批处理大小")

class QuantizationConfigRequest(ApiBaseModel):
    collection_name: str = Field(..., description="集合名称")
    quantization_type: str = Field(..., description="量化类型 (binary, halfprecision)")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="量化配置")

class BenchmarkSearchRequest(ApiBaseModel):
    test_queries: List[str] = Field(..., description="测试查询列表")
    top_k: int = Field(default=10, description="返回结果数量")

class IntegrityValidateRequest(ApiBaseModel):
    table_name: str = Field(..., description="表名")
    batch_size: int = Field(default=1000, description="批处理大小")
    vector_column: str = Field(default="embedding", description="向量列名")

class IntegrityRepairRequest(ApiBaseModel):
    integrity_report: Dict[str, Any] = Field(..., description="完整性报告")
    repair_strategy: str = Field(..., description="修复策略")

# 集合管理端点
@router.post("/collections", summary="创建向量集合")
async def create_collection(
    request: CreateCollectionRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """创建向量集合（表）"""
    try:
        success = await vector_store.create_collection(
            collection_name=request.collection_name,
            dimension=request.dimension,
            index_type=request.index_type,
            distance_metric=request.distance_metric,
            index_options=request.index_options
        )
        
        if success:
            return {
                "status": "success",
                "message": f"集合 {request.collection_name} 创建成功",
                "collection_name": request.collection_name,
                "dimension": request.dimension,
                "index_type": request.index_type
            }
        else:
            raise HTTPException(status_code=400, detail="集合创建失败")
            
    except Exception as e:
        logger.error(f"创建集合失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建集合失败: {str(e)}")

@router.post("/vectors", summary="批量插入向量")
async def insert_vectors(
    request: InsertVectorsRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """批量插入向量数据"""
    try:
        # 转换请求数据格式
        documents = []
        for doc in request.documents:
            documents.append({
                "content": doc.content,
                "embedding": doc.embedding,
                "metadata": doc.metadata
            })
        
        # 执行插入
        document_ids = await vector_store.insert_vectors(
            collection_name=request.collection_name,
            documents=documents,
            batch_size=request.batch_size
        )
        
        return {
            "status": "success",
            "message": f"成功插入 {len(document_ids)} 个向量",
            "inserted_count": len(document_ids),
            "document_ids": document_ids
        }
        
    except Exception as e:
        logger.error(f"插入向量失败: {e}")
        raise HTTPException(status_code=500, detail=f"插入向量失败: {str(e)}")

@router.get("/collections/{collection_name}/stats", summary="获取集合统计信息")
async def get_collection_stats(
    collection_name: str,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """获取集合统计信息"""
    try:
        stats = await vector_store.get_collection_stats(collection_name)
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"获取集合统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.post("/search/similarity", summary="向量相似性搜索")
async def similarity_search(
    request: SimilaritySearchRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """向量相似性搜索"""
    query_id = str(uuid.uuid4())
    start_time = utc_now()
    
    try:
        results = await vector_store.similarity_search(
            collection_name=request.collection_name,
            query_vector=request.query_vector,
            limit=request.limit,
            distance_metric=request.distance_metric,
            filters=request.filters,
            include_distances=request.include_distances
        )
        
        execution_time = (utc_now() - start_time).total_seconds() * 1000
        
        return {
            "status": "success",
            "query_id": query_id,
            "execution_time_ms": execution_time,
            "result_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"相似性搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.post("/search/hybrid", summary="混合向量搜索")
async def hybrid_search(
    request: HybridSearchRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """混合向量搜索（向量+文本）"""
    query_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    
    try:
        results, metrics, raw = await _execute_hybrid_search(
            vector_store=vector_store,
            collection_name=request.collection_name,
            query=request.query,
            top_k=request.top_k,
            pg_weight=request.pg_weight,
            qdrant_weight=request.qdrant_weight,
            search_mode=request.search_mode,
            use_cache=request.use_cache,
            query_vector=request.query_vector,
            filters=request.filters,
        )
        metrics["total_time_ms"] = (time.perf_counter() - start_time) * 1000
        await _record_cache_stats(raw["cache_hit"])

        try:
            metrics_collector = await get_metrics_collector()
            await metrics_collector.record_query_metrics(VectorQueryMetrics(
                query_id=query_id,
                collection_name=request.collection_name,
                query_type="hybrid_search",
                query_vector_dimension=len(raw["query_vector"] or []),
                result_count=len(results),
                execution_time_ms=metrics["total_time_ms"],
                index_scan_time_ms=None,
                distance_metric="l2",
                filters_applied=bool(request.filters),
                cache_hit=raw["cache_hit"],
                timestamp=utc_now(),
            ))
        except Exception as e:
            logger.warning(f"记录检索指标失败: {e}")

        return {
            "results": results,
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"混合搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"混合搜索失败: {str(e)}")

@router.post("/search/benchmark", summary="检索方法性能基准")
async def benchmark_search(
    request: BenchmarkSearchRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """执行真实查询的性能基准"""
    if not request.test_queries:
        raise HTTPException(status_code=400, detail="test_queries 不能为空")

    methods = ["hybrid", "pg_only", "qdrant_only"]
    results = []

    for method in methods:
        total_latency = 0.0
        total_results = 0
        success_count = 0
        consistency_scores: List[float] = []

        for query in request.test_queries:
            start = time.perf_counter()
            try:
                search_results, _, raw = await _execute_hybrid_search(
                    vector_store=vector_store,
                    collection_name="documents",
                    query=query,
                    top_k=request.top_k,
                    pg_weight=0.7,
                    qdrant_weight=0.3,
                    search_mode=method,
                    use_cache=False,
                    query_vector=None,
                    filters=None,
                )
                total_results += len(search_results)
                success_count += 1
                if method == "hybrid":
                    pg_ids = {str(r["id"]) for r in raw["pg_results"]}
                    qdrant_ids = {str(r["id"]) for r in raw["qdrant_results"]}
                    union = pg_ids | qdrant_ids
                    if union:
                        consistency_scores.append(len(pg_ids & qdrant_ids) / len(union))
            except Exception as e:
                logger.error(f"基准测试失败({method}): {e}")
            finally:
                total_latency += (time.perf_counter() - start) * 1000

        total_queries = len(request.test_queries)
        avg_latency = total_latency / total_queries if total_queries else 0.0
        avg_results = total_results / success_count if success_count else 0.0
        success_rate = success_count / total_queries if total_queries else 0.0
        accuracy_score = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores else None
        )

        results.append({
            "method": method,
            "avg_latency_ms": avg_latency,
            "results_per_query": avg_results,
            "success_rate": success_rate,
            "accuracy_score": accuracy_score,
        })

    return results

class CreateIndexRequest(ApiBaseModel):
    table_name: str = Field(..., description="表名")
    column_name: str = Field(..., description="向量列名")
    index_type: str = Field(..., description="索引类型 hnsw/ivfflat")
    distance_metric: str = Field(default="l2", description="距离度量 l2/cosine/ip/l1")
    index_options: Dict[str, Any] = Field(default_factory=dict, description="索引参数")

@router.post("/indexes/create", summary="创建向量索引")
async def create_index(
    request: CreateIndexRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """为现有表创建向量索引"""
    try:
        success = await vector_store.create_index(
            table_name=request.table_name,
            column_name=request.column_name,
            index_type=request.index_type,
            distance_metric=request.distance_metric,
            index_options=request.index_options
        )
        if not success:
            raise HTTPException(status_code=400, detail="索引创建失败")
        return {"status": "success", "message": "索引创建成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"索引创建异常: {e}")
        raise HTTPException(status_code=500, detail=f"索引创建失败: {str(e)}")

@router.get("/indexes/list", summary="列出现有向量索引")
async def list_indexes(vector_store: PgVectorStore = Depends(get_vector_store)):
    """返回数据库中所有索引定义"""
    try:
        indexes = await vector_store.list_indexes()
        return {"indexes": indexes}
    except Exception as e:
        logger.error(f"获取索引列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取索引列表失败: {str(e)}")

@router.get("/indexes/{table_name}", summary="获取表索引详情")
async def get_index_info(
    table_name: str,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """获取指定表的索引信息"""
    _validate_collection_name(table_name)
    try:
        async with vector_store.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    i.indexname,
                    i.indexdef,
                    s.idx_scan,
                    s.idx_tup_read,
                    s.idx_tup_fetch
                FROM pg_indexes i
                LEFT JOIN pg_stat_user_indexes s
                    ON i.indexname = s.indexrelname
                WHERE i.tablename = $1
                ORDER BY i.indexname
                """,
                table_name
            )
        return {"table_name": table_name, "indexes": [dict(row) for row in rows]}
    except Exception as e:
        logger.error(f"获取索引详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取索引详情失败: {str(e)}")

@router.post("/quantization/test", summary="测试量化效果")
async def test_quantization(request: dict):
    """基于真实向量样本评估量化效果"""
    try:
        vector_store = await get_vector_store()
        quantization_manager = await get_quantization_manager(vector_store)
        collection = request.get("collection_name") or "documents"
        _validate_collection_name(collection)
        samples = request.get("sample_size", 200)

        async with vector_store.get_connection() as conn:
            rows = await conn.fetch(
                f"""
                SELECT embedding FROM {collection}
                WHERE embedding IS NOT NULL
                ORDER BY RANDOM()
                LIMIT {samples}
                """
            )
        if not rows:
            return []

        import numpy as np
        vectors = []
        for r in rows:
            emb = r["embedding"]
            if isinstance(emb, str):
                vectors.append(np.array([float(v) for v in emb.strip('[]').split(',')]))
            else:
                vectors.append(np.array(emb))
        vectors_np = np.vstack(vectors)

        # 评估不同量化策略
        strategies = ["float32", "int8", "int4", "binary"]
        results = []
        for strategy in strategies:
            if strategy == "float32":
                encoded = vectors_np
            elif strategy == "int8":
                encoded = (vectors_np * 127).astype(np.int8)
            elif strategy == "int4":
                encoded = np.clip(np.round(vectors_np * 7), -8, 7).astype(np.int8)
            else:
                threshold = np.median(vectors_np, axis=0)
                encoded = (vectors_np > threshold).astype(np.uint8)

            # 简单还原误差评估
            if strategy == "float32":
                decoded = encoded
            elif strategy == "int8":
                decoded = encoded.astype(np.float32) / 127
            elif strategy == "int4":
                decoded = encoded.astype(np.float32) / 7
            else:
                decoded = np.where(encoded > 0, threshold + 0.1, threshold - 0.1)

            mse = float(np.mean((vectors_np - decoded) ** 2))
            compression = 1.0 if strategy == "float32" else (32.0 / (8 if strategy == "int8" else 4 if strategy == "int4" else 1))
            results.append({
                "strategy": strategy,
                "mse": mse,
                "compression_ratio": compression,
                "sample_size": vectors_np.shape[0]
            })

        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"量化测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"量化测试失败: {str(e)}")

@router.get("/quantization/config", summary="获取量化配置")
async def get_quantization_config():
    """获取数据库保存的量化配置"""
    try:
        vector_store = await get_vector_store()
        quantization_manager = await get_quantization_manager(vector_store)
        await quantization_manager.ensure_config_table()
        async with vector_store.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT quantizer_type, config_data
                FROM vector_quantization_configs
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
        if not row:
            return {
                "mode": "float32",
                "precision_threshold": 1.0,
                "compression_ratio": 1.0,
                "enable_dynamic": False,
                "current_strategy": "float32",
                "auto_optimization": False
            }
        data = dict(row["config_data"])
        return {
            "mode": data.get("type", "float32"),
            "precision_threshold": data.get("precision_threshold", 1.0),
            "compression_ratio": data.get("compression_ratio", 1.0),
            "enable_dynamic": data.get("enable_dynamic", False),
            "current_strategy": data.get("type", "float32"),
            "auto_optimization": data.get("auto_optimization", False)
        }
    except Exception as e:
        logger.error(f"获取量化配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取量化配置失败: {str(e)}")

@router.post("/quantization/configure", summary="配置向量量化")
async def configure_quantization(
    request: QuantizationConfigRequest,
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """配置向量量化"""
    try:
        quantization_manager = await get_quantization_manager(vector_store)
        
        success = await quantization_manager.create_quantization_config(
            collection_name=request.collection_name,
            quantization_type=request.quantization_type,
            config=request.config
        )
        
        if success:
            return {
                "status": "success",
                "message": f"量化配置创建成功: {request.collection_name}",
                "quantization_type": request.quantization_type,
                "config": request.config
            }
        else:
            raise HTTPException(status_code=400, detail="量化配置创建失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"配置量化失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置量化失败: {str(e)}")

@router.get("/integrity/summary", summary="获取向量数据完整性摘要")
async def get_integrity_summary(
    table_name: str = Query(..., description="表名"),
    db: AsyncSession = Depends(get_db)
):
    """获取向量数据完整性摘要"""
    try:
        if table_name in {"documents", "knowledge_items"}:
            vector_store = await get_vector_store()
            embedder = EmbeddingService()
            await _ensure_pg_collection(vector_store, table_name, embedder.dimension)
        else:
            raise HTTPException(status_code=400, detail="不支持的表名")
        validator = VectorDataIntegrityValidator(db)
        summary = await validator.generate_integrity_summary(table_name)
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        stats = summary.get("statistics", {})
        return {
            "table_name": summary.get("table_name"),
            "total_records": stats.get("total_records", 0),
            "non_null_embeddings": stats.get("non_null_embeddings", 0),
            "null_embeddings": stats.get("null_embeddings", 0),
            "null_rate": stats.get("null_rate", 0),
            "indexes": summary.get("indexes", []),
            "validation_stats": summary.get("validation_stats", {}),
            "timestamp": summary.get("timestamp"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取完整性摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取完整性摘要失败: {str(e)}")

@router.post("/integrity/validate", summary="验证向量数据完整性")
async def validate_integrity(
    request: IntegrityValidateRequest,
    db: AsyncSession = Depends(get_db)
):
    """验证向量数据完整性"""
    try:
        if request.table_name in {"documents", "knowledge_items"}:
            vector_store = await get_vector_store()
            embedder = EmbeddingService()
            await _ensure_pg_collection(vector_store, request.table_name, embedder.dimension)
        else:
            raise HTTPException(status_code=400, detail="不支持的表名")
        validator = VectorDataIntegrityValidator(db)
        report = await validator.validate_vector_data_integrity(
            table_name=request.table_name,
            vector_column=request.vector_column,
            batch_size=request.batch_size
        )
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])
        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"完整性验证失败: {e}")
        raise HTTPException(status_code=500, detail=f"完整性验证失败: {str(e)}")

@router.post("/integrity/repair", summary="修复向量数据")
async def repair_integrity(
    request: IntegrityRepairRequest,
    db: AsyncSession = Depends(get_db)
):
    """修复向量数据"""
    try:
        table_name = request.integrity_report.get("table_name")
        if table_name not in {"documents", "knowledge_items"}:
            raise HTTPException(status_code=400, detail="不支持的表名")
        validator = VectorDataIntegrityValidator(db)
        result = await validator.repair_vector_data(
            integrity_report=request.integrity_report,
            repair_strategy=request.repair_strategy
        )
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据修复失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据修复失败: {str(e)}")

@router.get("/performance/metrics", summary="获取性能指标")
async def get_performance_metrics(time_range: str = Query("1h", description="时间范围")):
    """获取真实的性能指标时间序列数据"""
    try:
        metrics_collector = await get_metrics_collector()
        report = await metrics_collector.get_performance_report(time_range_hours=_parse_time_range_hours(time_range))
        if "error" in report:
            raise HTTPException(status_code=500, detail=report["error"])
        # 将报告转换为时间序列点，前端图表直接可用
        return {
            "report_period": report.get("report_period"),
            "query_performance": report.get("query_performance"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {str(e)}")

@router.get("/performance/targets", summary="获取性能目标")
async def get_performance_targets():
    """基于历史窗口计算性能目标对比"""
    try:
        metrics_collector = await get_metrics_collector()
        if not metrics_collector.pool:
            await metrics_collector.initialize()
        if not metrics_collector.pool:
            raise HTTPException(status_code=503, detail="性能指标数据库未初始化")

        async def _fetch_stats(conn, start_time: datetime, end_time: datetime) -> Dict[str, float]:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_queries,
                    AVG(execution_time_ms) as avg_execution_time,
                    MAX(execution_time_ms) as max_execution_time
                FROM vector_query_metrics
                WHERE timestamp >= $1 AND timestamp < $2
                """,
                start_time,
                end_time,
            )
            return {
                "total_queries": float(row["total_queries"] or 0) if row else 0.0,
                "avg_execution_time": float(row["avg_execution_time"] or 0) if row else 0.0,
                "max_execution_time": float(row["max_execution_time"] or 0) if row else 0.0,
            }

        end_time = utc_now().replace(tzinfo=None)
        window_hours = 24
        current_start = end_time - timedelta(hours=window_hours)
        previous_start = end_time - timedelta(hours=window_hours * 2)

        async with metrics_collector.pool.acquire() as conn:
            current = await _fetch_stats(conn, current_start, end_time)
            previous = await _fetch_stats(conn, previous_start, current_start)

        def _calc_improvement(prev: float, curr: float, higher_is_better: bool) -> float:
            if prev <= 0:
                return 0.0
            return (curr - prev) / prev if higher_is_better else (prev - curr) / prev

        def _latency_status(curr: float, target: float) -> str:
            if target <= 0:
                return "tracking"
            if curr <= target:
                return "achieved"
            if curr <= target * 1.2:
                return "warning"
            return "not_achieved"

        def _count_status(curr: float, target: float) -> str:
            if target <= 0:
                return "tracking"
            if curr >= target:
                return "achieved"
            if curr >= target * 0.8:
                return "warning"
            return "not_achieved"

        avg_target = previous["avg_execution_time"] or current["avg_execution_time"]
        max_target = previous["max_execution_time"] or current["max_execution_time"]
        total_target = previous["total_queries"] or current["total_queries"]

        return [
            {
                "metric": "平均查询延迟(ms)",
                "current": round(current["avg_execution_time"], 2),
                "target": round(avg_target, 2),
                "status": _latency_status(current["avg_execution_time"], avg_target),
                "improvement": _calc_improvement(previous["avg_execution_time"], current["avg_execution_time"], False),
                "unit": "ms"
            },
            {
                "metric": "最大查询延迟(ms)",
                "current": round(current["max_execution_time"], 2),
                "target": round(max_target, 2),
                "status": _latency_status(current["max_execution_time"], max_target),
                "improvement": _calc_improvement(previous["max_execution_time"], current["max_execution_time"], False),
                "unit": "ms"
            },
            {
                "metric": "总查询数",
                "current": int(current["total_queries"]),
                "target": int(total_target),
                "status": _count_status(current["total_queries"], total_target),
                "improvement": _calc_improvement(previous["total_queries"], current["total_queries"], True),
                "unit": "count"
            }
        ]
    except Exception as e:
        logger.error(f"获取性能目标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能目标失败: {str(e)}")

@router.get("/cache/stats", summary="获取检索缓存统计")
async def get_cache_stats():
    """获取混合检索缓存统计"""
    redis = get_redis()
    if not redis:
        return {
            "enabled": False,
            "hit_rate": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "current_size": 0,
            "max_size": settings.CACHE_MAX_ENTRIES,
        }
    try:
        stats_key = "pgvector:hybrid_cache:stats"
        stats = await redis.hgetall(stats_key)
        hits = int(stats.get("hits", 0)) if stats else 0
        misses = int(stats.get("misses", 0)) if stats else 0
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        keys = await redis.keys("hybrid_search:*")
        current_size = len(keys)
        return {
            "enabled": True,
            "hit_rate": hit_rate,
            "cache_hits": hits,
            "cache_misses": misses,
            "current_size": current_size,
            "max_size": settings.CACHE_MAX_ENTRIES,
        }
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")

@router.post("/cache/clear", summary="清空检索缓存")
async def clear_cache():
    """清空混合检索缓存"""
    redis = get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="缓存未初始化")
    try:
        keys = await redis.keys("hybrid_search:*")
        if keys:
            await redis.delete(*keys)
        await redis.delete("pgvector:hybrid_cache:stats")
        return {"success": True}
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")

@router.get("/monitoring/performance-report", summary="获取性能报告")
async def get_performance_report(
    collection_name: Optional[str] = Query(None, description="集合名称"),
    time_range_hours: int = Query(24, description="时间范围（小时）")
):
    """获取性能报告"""
    try:
        metrics_collector = await get_metrics_collector()
        report = await metrics_collector.get_performance_report(
            collection_name=collection_name,
            time_range_hours=time_range_hours
        )
        
        return {
            "status": "success",
            "data": report
        }
        
    except Exception as e:
        logger.error(f"获取性能报告失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能报告失败: {str(e)}")

@router.get("/health", summary="pgvector健康检查")
async def health_check(
    vector_store: PgVectorStore = Depends(get_vector_store)
):
    """检查pgvector服务健康状态"""
    try:
        async with vector_store.get_connection() as conn:
            version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            
            return {
                "status": "healthy",
                "pgvector_version": version or "not_installed",
                "timestamp": utc_now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail=f"服务不健康: {str(e)}")

@router.get("/config", summary="获取pgvector配置")
async def get_pgvector_config():
    """获取当前pgvector配置"""
    return {
        "status": "success",
        "config": {
            "pgvector_enabled": settings.PGVECTOR_ENABLED,
            "pgvector_version": settings.PGVECTOR_VERSION,
            "hnsw": {
                "ef_construction": settings.HNSW_EF_CONSTRUCTION,
                "ef_search": settings.HNSW_EF_SEARCH,
                "m": settings.HNSW_M,
                "iterative_scan": settings.HNSW_ITERATIVE_SCAN
            },
            "ivfflat": {
                "lists": settings.IVFFLAT_LISTS,
                "probes": settings.IVFFLAT_PROBES,
                "iterative_scan": settings.IVFFLAT_ITERATIVE_SCAN
            },
            "quantization": {
                "enabled": settings.VECTOR_QUANTIZATION_ENABLED,
                "binary_quantization": settings.VECTOR_BINARY_QUANTIZATION,
                "halfvec_enabled": settings.VECTOR_HALFVEC_ENABLED
            },
            "monitoring": {
                "enabled": settings.VECTOR_MONITORING_ENABLED,
                "performance_logging": settings.VECTOR_PERFORMANCE_LOGGING
            }
        }
    }
