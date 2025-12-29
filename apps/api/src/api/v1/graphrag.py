"""
GraphRAG API端点

提供GraphRAG系统的HTTP API接口：
- GraphRAG增强查询
- 查询分析和分解
- 推理路径查询
- 知识融合和冲突解决
- 性能监控和调试
"""

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import Field
from src.ai.graphrag.core_engine import get_graphrag_engine
from src.core.utils.timezone_utils import utc_now
from src.ai.graphrag.data_models import (
    GraphRAGRequest, 
    GraphRAGResponse,
    GraphRAGConfig,
    QueryType, 
    RetrievalMode,
    create_graph_rag_request,
    GraphRAGResponse,
    GraphRAGConfig,
    QueryType, 
    RetrievalMode,
    create_graph_rag_request,
    validate_graph_rag_request
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])

_query_logs = deque(maxlen=2000)

def _parse_dt(value: str) -> datetime:
    v = (value or "").strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _day_range(value: str) -> tuple[datetime, datetime]:
    dt = _parse_dt(value)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=dt.tzinfo)
    return start, start + timedelta(days=1)

class GraphRAGQueryRequest(ApiBaseModel):
    """GraphRAG查询请求模型"""
    query: str = Field(..., description="查询文本", min_length=1, max_length=1000)
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="检索模式")
    max_docs: int = Field(default=10, ge=1, le=100, description="最大文档数量")
    include_reasoning: bool = Field(default=True, description="是否包含推理")
    expansion_depth: int = Field(default=2, ge=0, le=5, description="上下文扩展深度")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="置信度阈值")
    query_type: Optional[QueryType] = Field(default=None, description="查询类型(可选)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件")

class GraphRAGQueryResponse(ApiBaseModel):
    """GraphRAG查询响应模型"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[GraphRAGResponse] = Field(None, description="GraphRAG响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    query_id: str = Field(..., description="查询ID")
    performance_metrics: Dict[str, Any] = Field(..., description="性能指标")

class QueryAnalysisRequest(ApiBaseModel):
    """查询分析请求模型"""
    query: str = Field(..., description="查询文本", min_length=1, max_length=1000)
    query_type: Optional[QueryType] = Field(default=None, description="查询类型(可选)")

class ReasoningPathRequest(ApiBaseModel):
    """推理路径请求模型"""
    entity1: str = Field(..., description="源实体", min_length=1)
    entity2: str = Field(..., description="目标实体", min_length=1)
    max_hops: int = Field(default=3, ge=1, le=5, description="最大跳数")
    max_paths: int = Field(default=10, ge=1, le=50, description="最大路径数")

class ConflictResolutionRequest(ApiBaseModel):
    knowledge_sources: List[Dict[str, Any]] = Field(..., description="知识源列表")
    conflicts: Optional[List[Dict[str, Any]]] = Field(default=None, description="可选冲突列表(不传则自动检测)")
    strategy: str = Field(default="highest_confidence", description="冲突解决策略")

@router.post("/query", response_model=GraphRAGQueryResponse)
async def graphrag_query(request: GraphRAGQueryRequest):
    """GraphRAG增强查询
    
    执行GraphRAG增强查询，结合向量检索和图谱推理
    """
    try:
        logger.info(f"收到GraphRAG查询请求: {request.query[:100]}...")
        
        # 转换为内部请求格式
        internal_request = create_graph_rag_request(
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            max_docs=request.max_docs,
            include_reasoning=request.include_reasoning,
            expansion_depth=request.expansion_depth,
            confidence_threshold=request.confidence_threshold,
            query_type=request.query_type,
            filters=request.filters
        )
        
        # 验证请求
        validation_errors = validate_graph_rag_request(internal_request)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"请求验证失败: {'; '.join(validation_errors)}"
            )
        
        # 执行GraphRAG查询
        engine = await get_graphrag_engine()
        result = await engine.enhanced_query(internal_request)

        perf = result.get("performance_metrics") or {}
        _query_logs.append(
            {
                "timestamp": utc_now(),
                "query_id": result.get("query_id"),
                "retrieval_mode": request.retrieval_mode.value,
                "success": True,
                "total_time_ms": float(perf.get("total_time", 0.0)) * 1000,
                "retrieval_time_ms": float(perf.get("retrieval_time", 0.0)) * 1000,
                "reasoning_time_ms": float(perf.get("reasoning_time", 0.0)) * 1000,
                "fusion_time_ms": float(perf.get("fusion_time", 0.0)) * 1000,
                "cache_hit": bool(perf.get("cache_hit", False)),
                "top_score": float((result.get("documents") or [{}])[0].get("final_score", 0.0)),
            }
        )
        
        return GraphRAGQueryResponse(
            success=True,
            data=result,
            error=None,
            query_id=result["query_id"],
            performance_metrics=result["performance_metrics"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GraphRAG查询失败: {e}")
        _query_logs.append({"timestamp": utc_now(), "success": False, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"GraphRAG查询失败: {str(e)}")

@router.post("/query/analyze")
async def analyze_query(request: QueryAnalysisRequest):
    """查询分析和分解
    
    分析查询的类型、复杂度，并分解为子查询
    """
    try:
        logger.info(f"收到查询分析请求: {request.query[:100]}...")
        
        engine = await get_graphrag_engine()
        
        # 执行查询分析
        decomposition = await engine.query_analyzer.analyze_query(
            request.query,
            request.query_type
        )
        
        return {
            "success": True,
            "data": {
                "decomposition": decomposition.to_dict(),
                "analysis": {
                    "detected_query_type": decomposition.decomposition_strategy,
                    "complexity_score": decomposition.complexity_score,
                    "sub_queries_count": len(decomposition.sub_queries),
                    "entities_count": len(decomposition.entity_queries),
                    "relations_count": len(decomposition.relation_queries)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"查询分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询分析失败: {str(e)}")

@router.post("/query/reasoning")
async def reasoning_paths(request: ReasoningPathRequest):
    """推理路径查询
    
    寻找两个实体之间的推理路径
    """
    try:
        logger.info(f"收到推理路径请求: {request.entity1} -> {request.entity2}")
        
        engine = await get_graphrag_engine()
        
        # 创建查询分解对象（用于推理路径生成）
        from src.ai.graphrag.data_models import QueryDecomposition
        
        decomposition = QueryDecomposition(
            original_query=f"Find reasoning paths between {request.entity1} and {request.entity2}",
            sub_queries=[],
            entity_queries=[
                {"entity": request.entity1, "properties": ["all"]},
                {"entity": request.entity2, "properties": ["all"]}
            ],
            relation_queries=[{
                "entity1": request.entity1,
                "entity2": request.entity2,
                "max_hops": request.max_hops,
                "relation_types": ["all"]
            }],
            decomposition_strategy="relational",
            complexity_score=0.5
        )
        
        # 创建简化的图谱上下文
        from src.ai.graphrag.data_models import create_empty_graph_context
        graph_context = create_empty_graph_context()
        
        # 生成推理路径
        reasoning_paths = await engine.reasoning_engine.generate_reasoning_paths(
            decomposition,
            graph_context,
            max_paths=request.max_paths,
            max_depth=request.max_hops
        )
        
        return {
            "success": True,
            "data": {
                "paths": [path.to_dict() for path in reasoning_paths],
                "statistics": {
                    "total_paths": len(reasoning_paths),
                    "avg_hops": sum(p.hops_count for p in reasoning_paths) / max(1, len(reasoning_paths)),
                    "avg_score": sum(p.path_score for p in reasoning_paths) / max(1, len(reasoning_paths))
                }
            }
        }
        
    except Exception as e:
        logger.error(f"推理路径查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理路径查询失败: {str(e)}")

@router.get("/query/{query_id}")
async def get_query_result(query_id: str = Path(..., description="查询ID")):
    """获取查询结果
    
    通过查询ID获取已缓存的查询结果
    """
    try:
        engine = await get_graphrag_engine()
        
        if not engine.cache_manager:
            raise HTTPException(status_code=404, detail="查询结果不存在或已过期")

        result = await engine.cache_manager.get_cached_result_by_query_id(query_id)
        if not result:
            raise HTTPException(status_code=404, detail="查询结果不存在或已过期")

        return {
            "success": True,
            "query_id": query_id,
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取查询结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取查询结果失败: {str(e)}")

@router.post("/fusion/multi-source")
async def multi_source_fusion(request: Dict[str, Any]):
    """多源知识融合
    
    手动融合来自不同源的知识
    """
    try:
        logger.info("收到多源知识融合请求")
        
        # 解析请求数据
        vector_results = request.get('vector_results', [])
        graph_results = request.get('graph_results', {})
        reasoning_results = request.get('reasoning_results', [])
        confidence_threshold = request.get('confidence_threshold', 0.6)
        
        engine = await get_graphrag_engine()
        
        # 创建图谱上下文
        from src.ai.graphrag.data_models import GraphContext
        graph_context = GraphContext(
            entities=graph_results.get('entities', []),
            relations=graph_results.get('relations', []),
            subgraph={},
            reasoning_paths=[],
            expansion_depth=1,
            confidence_score=0.8
        )
        
        # 转换推理结果
        from src.ai.graphrag.data_models import ReasoningPath
        reasoning_paths = []
        for r in reasoning_results:
            path = ReasoningPath(
                path_id=r.get('path_id', ''),
                entities=r.get('entities', []),
                relations=r.get('relations', []),
                path_score=r.get('path_score', 0.0),
                explanation=r.get('explanation', ''),
                evidence=r.get('evidence', []),
                hops_count=r.get('hops_count', 0)
            )
            reasoning_paths.append(path)
        
        # 执行知识融合
        fusion_results = await engine.fusion_engine.fuse_knowledge_sources(
            {'vector': vector_results, 'graph': graph_results},
            graph_context,
            reasoning_paths,
            confidence_threshold
        )
        
        return {
            "success": True,
            "data": {
                "fusion_results": fusion_results,
                "statistics": {
                    "input_sources": len(vector_results) + len(graph_results.get('entities', [])),
                    "output_documents": len(fusion_results.get('final_ranking', [])),
                    "conflicts_detected": len(fusion_results.get('conflicts_detected', [])),
                    "consistency_score": fusion_results.get('consistency_score', 0.0)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"多源知识融合失败: {e}")
        raise HTTPException(status_code=500, detail=f"多源知识融合失败: {str(e)}")

@router.post("/fusion/conflict-resolution") 
async def conflict_resolution(request: ConflictResolutionRequest):
    """冲突解决
    
    解决知识源之间的冲突
    """
    try:
        logger.info("收到冲突解决请求")

        engine = await get_graphrag_engine()
        if not engine._initialized:
            await engine.initialize()
        fusion = engine.fusion_engine
        if not fusion:
            raise HTTPException(status_code=503, detail="融合引擎未初始化")

        from src.ai.graphrag.data_models import KnowledgeSource

        sources = []
        for s in request.knowledge_sources:
            content = str(s.get("content") or "").strip()
            if not content:
                continue
            sources.append(
                KnowledgeSource(
                    source_type=str(s.get("source_type") or "vector"),
                    content=content,
                    confidence=float(s.get("confidence") or 0.0),
                    metadata=s.get("metadata") if isinstance(s.get("metadata"), dict) else {},
                )
            )

        if not sources:
            raise HTTPException(status_code=400, detail="knowledge_sources不能为空")

        conflicts = request.conflicts
        if conflicts is None:
            conflicts = await fusion._detect_conflicts(sources)

        strategy = (request.strategy or "highest_confidence").strip()
        resolver = fusion.conflict_resolution_strategies.get(strategy)
        if not resolver:
            raise HTTPException(status_code=400, detail=f"不支持的strategy: {strategy}")

        resolved_sources = sources
        resolved_conflicts = []
        for conflict in conflicts:
            before = [s.confidence for s in resolved_sources]
            resolved_sources = await resolver(resolved_sources, conflict)
            after = [s.confidence for s in resolved_sources]
            resolved_conflicts.append(
                {
                    "original_conflict": conflict,
                    "resolution_strategy": strategy,
                    "before_confidence": before,
                    "after_confidence": after,
                }
            )
        
        return {
            "success": True,
            "data": {
                "resolved_conflicts": resolved_conflicts,
                "strategy_used": strategy,
                "resolved_sources": [s.to_dict() for s in resolved_sources],
                "resolution_rate": len(resolved_conflicts) / max(1, len(conflicts)),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"冲突解决失败: {e}")
        raise HTTPException(status_code=500, detail=f"冲突解决失败: {str(e)}")

@router.get("/fusion/consistency")
async def consistency_check(query_id: Optional[str] = Query(None, description="查询ID(不传则取最近一次)")):
    """一致性检查
    
    检查指定文档之间的一致性
    """
    try:
        engine = await get_graphrag_engine()
        if not engine._initialized:
            await engine.initialize()
        if not engine.cache_manager:
            raise HTTPException(status_code=400, detail="缓存未启用，无法进行一致性评估")

        cached = None
        if query_id:
            cached = await engine.cache_manager.get_cached_result_by_query_id(query_id)
        else:
            for v in reversed(list(engine.cache_manager.memory_cache.values())):
                if isinstance(v, dict) and v.get("query_id") and isinstance(v.get("documents"), list):
                    cached = v
                    break

        if not cached:
            return {
                "success": True,
                "data": {"query_id": query_id, "consistency_score": None, "documents_checked": 0},
            }

        docs = cached.get("documents") or []
        fusion = engine.fusion_engine
        if not fusion:
            raise HTTPException(status_code=503, detail="融合引擎未初始化")
        consistency_score = await fusion._check_consistency({"documents": docs})

        return {
            "success": True,
            "data": {
                "query_id": cached.get("query_id"),
                "consistency_score": float(consistency_score),
                "documents_checked": len(docs),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"一致性检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"一致性检查失败: {str(e)}")

@router.get("/performance/stats")
async def performance_stats(
    start_time: Optional[str] = Query(None, description="开始时间(ISO8601)"),
    end_time: Optional[str] = Query(None, description="结束时间(ISO8601)"),
):
    """性能统计
    
    获取GraphRAG系统的性能统计信息
    """
    try:
        logs = list(_query_logs)
        if start_time:
            start = _parse_dt(start_time)
            logs = [l for l in logs if isinstance(l.get("timestamp"), datetime) and l["timestamp"] >= start]
        if end_time:
            end = _parse_dt(end_time)
            logs = [l for l in logs if isinstance(l.get("timestamp"), datetime) and l["timestamp"] <= end]

        total = len(logs)
        successful = sum(1 for l in logs if l.get("success"))
        failed = total - successful
        total_time = [l.get("total_time_ms", 0.0) for l in logs if l.get("success")]
        retrieval_time = [l.get("retrieval_time_ms", 0.0) for l in logs if l.get("success")]
        reasoning_time = [l.get("reasoning_time_ms", 0.0) for l in logs if l.get("success")]
        fusion_time = [l.get("fusion_time_ms", 0.0) for l in logs if l.get("success")]
        cache_hits = sum(1 for l in logs if l.get("success") and l.get("cache_hit"))

        by_mode: Dict[str, int] = {}
        for l in logs:
            mode = l.get("retrieval_mode") or "unknown"
            by_mode[mode] = by_mode.get(mode, 0) + 1

        def _avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0
        
        return {
            "success": True,
            "data": {
                "query_statistics": {
                    "total_queries": total,
                    "successful_queries": successful,
                    "failed_queries": failed,
                    "average_response_time": _avg(total_time),
                    "queries_by_type": by_mode,
                },
                "performance_metrics": {
                    "average_retrieval_time": _avg(retrieval_time),
                    "average_reasoning_time": _avg(reasoning_time),
                    "average_fusion_time": _avg(fusion_time),
                    "cache_hit_rate": (cache_hits / total) if total else 0.0,
                },
                "time_period": {
                    "start_time": start_time,
                    "end_time": end_time,
                },
            },
            "timestamp": utc_now().isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")

@router.get("/performance/comparison")
async def rag_comparison(
    baseline_date: str = Query(..., description="基线日期(ISO8601)"),
    comparison_date: str = Query(..., description="对比日期(ISO8601)"),
):
    """RAG对比分析
    
    比较GraphRAG与传统RAG的性能
    """
    try:
        base_start, base_end = _day_range(baseline_date)
        cmp_start, cmp_end = _day_range(comparison_date)

        logs = [l for l in _query_logs if l.get("success") and isinstance(l.get("timestamp"), datetime)]
        base_logs = [l for l in logs if base_start <= l["timestamp"] < base_end]
        cmp_logs = [l for l in logs if cmp_start <= l["timestamp"] < cmp_end]

        if not base_logs or not cmp_logs:
            raise HTTPException(status_code=404, detail="指定日期范围内没有可用于对比的查询数据")

        def _avg_key(items: List[Dict[str, Any]], key: str) -> float:
            vals = [float(i.get(key, 0.0)) for i in items]
            return float(sum(vals) / len(vals)) if vals else 0.0

        baseline_metrics = {
            "avg_response_time_ms": _avg_key(base_logs, "total_time_ms"),
            "avg_retrieval_time_ms": _avg_key(base_logs, "retrieval_time_ms"),
            "avg_reasoning_time_ms": _avg_key(base_logs, "reasoning_time_ms"),
            "avg_fusion_time_ms": _avg_key(base_logs, "fusion_time_ms"),
            "avg_top_score": _avg_key(base_logs, "top_score"),
            "cache_hit_rate": sum(1 for l in base_logs if l.get("cache_hit")) / len(base_logs),
            "total_queries": len(base_logs),
        }
        comparison_metrics = {
            "avg_response_time_ms": _avg_key(cmp_logs, "total_time_ms"),
            "avg_retrieval_time_ms": _avg_key(cmp_logs, "retrieval_time_ms"),
            "avg_reasoning_time_ms": _avg_key(cmp_logs, "reasoning_time_ms"),
            "avg_fusion_time_ms": _avg_key(cmp_logs, "fusion_time_ms"),
            "avg_top_score": _avg_key(cmp_logs, "top_score"),
            "cache_hit_rate": sum(1 for l in cmp_logs if l.get("cache_hit")) / len(cmp_logs),
            "total_queries": len(cmp_logs),
        }

        baseline_rt = baseline_metrics["avg_response_time_ms"] or 1.0
        comparison_rt = comparison_metrics["avg_response_time_ms"]
        response_time_change = ((comparison_rt - baseline_rt) / baseline_rt) * 100
        performance_change = ((baseline_rt - comparison_rt) / baseline_rt) * 100
        accuracy_change = (comparison_metrics["avg_top_score"] - baseline_metrics["avg_top_score"]) * 100

        base_throughput = len(base_logs) / ((base_end - base_start).total_seconds() or 1.0)
        cmp_throughput = len(cmp_logs) / ((cmp_end - cmp_start).total_seconds() or 1.0)
        throughput_change = ((cmp_throughput - base_throughput) / (base_throughput or 1.0)) * 100

        recommendations: List[str] = []
        if response_time_change > 10:
            recommendations.append("响应时间上升明显，建议降低expansion_depth或max_docs")
        if comparison_metrics["cache_hit_rate"] < baseline_metrics["cache_hit_rate"]:
            recommendations.append("缓存命中率下降，建议开启/检查Redis并扩大缓存TTL")
        if not recommendations:
            recommendations.append("当前性能变化不明显，可继续观察并积累更多样本后再对比")
        
        return {
            "success": True,
            "data": {
                "comparison_summary": {
                    "performance_change": performance_change,
                    "response_time_change": response_time_change,
                    "accuracy_change": accuracy_change,
                    "throughput_change": throughput_change,
                },
                "detailed_comparison": {
                    "baseline_metrics": baseline_metrics,
                    "comparison_metrics": comparison_metrics,
                    "differences": {
                        "avg_response_time_ms": comparison_metrics["avg_response_time_ms"]
                        - baseline_metrics["avg_response_time_ms"],
                        "avg_top_score": comparison_metrics["avg_top_score"]
                        - baseline_metrics["avg_top_score"],
                        "cache_hit_rate": comparison_metrics["cache_hit_rate"]
                        - baseline_metrics["cache_hit_rate"],
                    },
                },
                "recommendations": recommendations,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG对比分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"RAG对比分析失败: {str(e)}")

@router.post("/performance/benchmark")
async def benchmark_test(request: Dict[str, Any]):
    """基准测试
    
    执行GraphRAG系统的基准测试
    """
    try:
        logger.info("开始GraphRAG基准测试")
        
        test_queries = request.get('test_queries', [])
        test_config = request.get('config', {})
        
        if not test_queries:
            raise HTTPException(status_code=400, detail="测试查询不能为空")
        
        engine = await get_graphrag_engine()
        results = []
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            try:
                # 创建测试请求
                test_request = create_graph_rag_request(
                    query=query,
                    max_docs=test_config.get('max_docs', 10),
                    include_reasoning=test_config.get('include_reasoning', True),
                    confidence_threshold=test_config.get('confidence_threshold', 0.6)
                )
                
                # 执行查询
                response = await engine.enhanced_query(test_request)
                
                end_time = time.time()
                
                results.append({
                    "query_index": i,
                    "query": query,
                    "response_time": end_time - start_time,
                    "documents_returned": len(response["documents"]),
                    "reasoning_paths": len(response["reasoning_results"]),
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "query_index": i,
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        # 计算统计信息
        successful_results = [r for r in results if r["success"]]
        
        statistics = {
            "total_queries": len(test_queries),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(test_queries),
            "avg_response_time": sum(r["response_time"] for r in successful_results) / max(1, len(successful_results)),
            "avg_documents_per_query": sum(r["documents_returned"] for r in successful_results) / max(1, len(successful_results)),
            "avg_reasoning_paths": sum(r["reasoning_paths"] for r in successful_results) / max(1, len(successful_results))
        }
        
        return {
            "success": True,
            "data": {
                "results": results,
                "statistics": statistics
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"基准测试失败: {str(e)}")

@router.get("/config")
async def get_config():
    """获取配置
    
    获取当前GraphRAG系统的配置信息
    """
    try:
        engine = await get_graphrag_engine()
        
        return {
            "success": True,
            "data": {
                "config": engine.config.to_dict(),
                "status": "initialized" if engine._initialized else "not_initialized"
            }
        }
        
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")

@router.put("/config")
async def update_config(request: Dict[str, Any]):
    """更新配置
    
    更新GraphRAG系统的配置
    """
    try:
        logger.info("收到配置更新请求")
        
        # 验证配置参数
        config_updates = {}
        
        if 'max_expansion_depth' in request:
            config_updates['max_expansion_depth'] = max(0, min(5, request['max_expansion_depth']))
        
        if 'confidence_threshold' in request:
            config_updates['confidence_threshold'] = max(0.0, min(1.0, request['confidence_threshold']))
        
        if 'max_reasoning_paths' in request:
            config_updates['max_reasoning_paths'] = max(1, min(50, request['max_reasoning_paths']))

        engine = await get_graphrag_engine()
        if not engine._initialized:
            await engine.initialize()
        for k, v in config_updates.items():
            if hasattr(engine.config, k):
                setattr(engine.config, k, v)
        
        return {
            "success": True,
            "data": {
                "updated_config": engine.config.to_dict(),
                "message": "配置更新成功"
            }
        }
        
    except Exception as e:
        logger.error(f"配置更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置更新失败: {str(e)}")

@router.post("/debug/explain")
async def explain_result(request: Dict[str, Any]):
    """结果解释
    
    解释GraphRAG查询结果的推理过程
    """
    try:
        logger.info("收到结果解释请求")
        
        query = request.get('query', '')
        reasoning_paths = request.get('reasoning_paths', [])
        
        engine = await get_graphrag_engine()
        
        # 转换推理路径
        from src.ai.graphrag.data_models import ReasoningPath
        paths = []
        for r in reasoning_paths:
            path = ReasoningPath(
                path_id=r.get('path_id', ''),
                entities=r.get('entities', []),
                relations=r.get('relations', []),
                path_score=r.get('path_score', 0.0),
                explanation=r.get('explanation', ''),
                evidence=r.get('evidence', []),
                hops_count=r.get('hops_count', 0)
            )
            paths.append(path)
        
        # 生成解释
        explanation = await engine.reasoning_engine.explain_reasoning_result(
            paths, query
        )
        
        return {
            "success": True,
            "data": {
                "explanation": explanation,
                "reasoning_paths_count": len(paths),
                "query": query
            }
        }
        
    except Exception as e:
        logger.error(f"结果解释失败: {e}")
        raise HTTPException(status_code=500, detail=f"结果解释失败: {str(e)}")

@router.get("/debug/trace/{query_id}")
async def query_trace(query_id: str = Path(..., description="查询ID")):
    """查询追踪
    
    追踪特定查询的执行过程
    """
    try:
        engine = await get_graphrag_engine()
        if not engine._initialized:
            await engine.initialize()
        if not engine.cache_manager:
            raise HTTPException(status_code=404, detail="查询结果不存在或已过期")

        result = await engine.cache_manager.get_cached_result_by_query_id(query_id)
        if not result:
            raise HTTPException(status_code=404, detail="查询结果不存在或已过期")

        perf = result.get("performance_metrics") or {}
        steps = [
            {"step": "retrieval", "duration_ms": float(perf.get("retrieval_time", 0.0)) * 1000, "status": "completed"},
            {"step": "reasoning", "duration_ms": float(perf.get("reasoning_time", 0.0)) * 1000, "status": "completed"},
            {"step": "fusion", "duration_ms": float(perf.get("fusion_time", 0.0)) * 1000, "status": "completed"},
        ]
        trace_data = {
            "query_id": query_id,
            "steps": steps,
            "results": {
                "documents": len(result.get("documents") or []),
                "entities": len((result.get("graph_context") or {}).get("entities") or []),
                "relations": len((result.get("graph_context") or {}).get("relations") or []),
                "reasoning_paths": len(result.get("reasoning_results") or []),
            },
            "total_duration_ms": float(perf.get("total_time", 0.0)) * 1000,
        }
        
        return {
            "success": True,
            "data": trace_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询追踪失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询追踪失败: {str(e)}")

# 导入时间模块
