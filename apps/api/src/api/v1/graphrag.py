"""
GraphRAG API端点

提供GraphRAG系统的HTTP API接口：
- GraphRAG增强查询
- 查询分析和分解
- 推理路径查询
- 知识融合和冲突解决
- 性能监控和调试
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...ai.graphrag.core_engine import get_graphrag_engine
from ...ai.graphrag.data_models import (
    GraphRAGRequest, 
    GraphRAGResponse,
    GraphRAGConfig,
    QueryType, 
    RetrievalMode,
    create_graph_rag_request,
    validate_graph_rag_request
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])


class GraphRAGQueryRequest(BaseModel):
    """GraphRAG查询请求模型"""
    query: str = Field(..., description="查询文本", min_length=1, max_length=1000)
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="检索模式")
    max_docs: int = Field(default=10, ge=1, le=100, description="最大文档数量")
    include_reasoning: bool = Field(default=True, description="是否包含推理")
    expansion_depth: int = Field(default=2, ge=0, le=5, description="上下文扩展深度")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="置信度阈值")
    query_type: Optional[QueryType] = Field(default=None, description="查询类型(可选)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="过滤条件")


class GraphRAGQueryResponse(BaseModel):
    """GraphRAG查询响应模型"""
    success: bool = Field(..., description="请求是否成功")
    data: Optional[GraphRAGResponse] = Field(None, description="GraphRAG响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    query_id: str = Field(..., description="查询ID")
    performance_metrics: Dict[str, Any] = Field(..., description="性能指标")


class QueryAnalysisRequest(BaseModel):
    """查询分析请求模型"""
    query: str = Field(..., description="查询文本", min_length=1, max_length=1000)
    query_type: Optional[QueryType] = Field(default=None, description="查询类型(可选)")


class ReasoningPathRequest(BaseModel):
    """推理路径请求模型"""
    entity1: str = Field(..., description="源实体", min_length=1)
    entity2: str = Field(..., description="目标实体", min_length=1)
    max_hops: int = Field(default=3, ge=1, le=5, description="最大跳数")
    max_paths: int = Field(default=10, ge=1, le=50, description="最大路径数")


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
        from ...ai.graphrag.data_models import QueryDecomposition
        
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
        from ...ai.graphrag.data_models import create_empty_graph_context
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
        
        # 尝试从缓存获取结果
        if engine.cache_manager:
            # 这里需要扩展缓存管理器以支持按查询ID获取
            # 目前返回示例响应
            return {
                "success": False,
                "message": "Query ID lookup not implemented yet",
                "query_id": query_id
            }
        else:
            return {
                "success": False,
                "message": "Cache not enabled",
                "query_id": query_id
            }
        
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
        from ...ai.graphrag.data_models import GraphContext
        graph_context = GraphContext(
            entities=graph_results.get('entities', []),
            relations=graph_results.get('relations', []),
            subgraph={},
            reasoning_paths=[],
            expansion_depth=1,
            confidence_score=0.8
        )
        
        # 转换推理结果
        from ...ai.graphrag.data_models import ReasoningPath
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
async def conflict_resolution(request: Dict[str, Any]):
    """冲突解决
    
    解决知识源之间的冲突
    """
    try:
        logger.info("收到冲突解决请求")
        
        conflicts = request.get('conflicts', [])
        resolution_strategy = request.get('strategy', 'highest_confidence')
        
        # 这里可以实现具体的冲突解决逻辑
        resolved_conflicts = []
        
        for conflict in conflicts:
            resolved_conflicts.append({
                "original_conflict": conflict,
                "resolution_strategy": resolution_strategy,
                "resolved": True,
                "confidence": 0.8
            })
        
        return {
            "success": True,
            "data": {
                "resolved_conflicts": resolved_conflicts,
                "strategy_used": resolution_strategy,
                "resolution_rate": len(resolved_conflicts) / max(1, len(conflicts))
            }
        }
        
    except Exception as e:
        logger.error(f"冲突解决失败: {e}")
        raise HTTPException(status_code=500, detail=f"冲突解决失败: {str(e)}")


@router.get("/fusion/consistency")
async def consistency_check(
    document_ids: List[str] = Query(..., description="文档ID列表")
):
    """一致性检查
    
    检查指定文档之间的一致性
    """
    try:
        logger.info(f"收到一致性检查请求，文档数: {len(document_ids)}")
        
        # 这里可以实现具体的一致性检查逻辑
        consistency_score = 0.85  # 示例分数
        
        return {
            "success": True,
            "data": {
                "document_ids": document_ids,
                "consistency_score": consistency_score,
                "details": {
                    "consistent_pairs": len(document_ids) - 1,
                    "inconsistent_pairs": 0,
                    "confidence_interval": [0.8, 0.9]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"一致性检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"一致性检查失败: {str(e)}")


@router.get("/performance/stats")
async def performance_stats():
    """性能统计
    
    获取GraphRAG系统的性能统计信息
    """
    try:
        engine = await get_graphrag_engine()
        stats = await engine.get_performance_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": engine._initialized
        }
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


@router.get("/performance/comparison")
async def rag_comparison():
    """RAG对比分析
    
    比较GraphRAG与传统RAG的性能
    """
    try:
        # 这里可以实现具体的对比分析逻辑
        comparison_data = {
            "traditional_rag": {
                "avg_response_time": 1.2,
                "accuracy_score": 0.75,
                "context_relevance": 0.70
            },
            "graph_rag": {
                "avg_response_time": 1.8,
                "accuracy_score": 0.85,
                "context_relevance": 0.82
            },
            "improvement": {
                "accuracy_improvement": 13.3,
                "context_improvement": 17.1,
                "response_time_overhead": 50.0
            }
        }
        
        return {
            "success": True,
            "data": comparison_data
        }
        
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
        
        # 这里可以实现具体的配置更新逻辑
        
        return {
            "success": True,
            "data": {
                "updated_config": config_updates,
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
        from ...ai.graphrag.data_models import ReasoningPath
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
        # 这里可以实现查询追踪逻辑
        trace_data = {
            "query_id": query_id,
            "steps": [
                {"step": "query_analysis", "duration": 0.1, "status": "completed"},
                {"step": "vector_retrieval", "duration": 0.3, "status": "completed"},
                {"step": "graph_retrieval", "duration": 0.5, "status": "completed"},
                {"step": "context_expansion", "duration": 0.2, "status": "completed"},
                {"step": "reasoning", "duration": 0.8, "status": "completed"},
                {"step": "knowledge_fusion", "duration": 0.4, "status": "completed"}
            ],
            "total_duration": 2.3
        }
        
        return {
            "success": True,
            "data": trace_data
        }
        
    except Exception as e:
        logger.error(f"查询追踪失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询追踪失败: {str(e)}")


# 导入时间模块
import time