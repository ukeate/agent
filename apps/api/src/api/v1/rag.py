"""
RAG系统 API 路由

包含基础RAG功能和Agentic RAG智能检索功能
"""

from typing import List
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from src.models.schemas.rag import (
    AddDocumentRequest,
    AddDocumentResponse,
    IndexDirectoryRequest,
    IndexFileRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    ResetRequest,
    ResetResponse,
    StatsResponse,
    UpdateIndexRequest,
)
from src.models.schemas.agentic_rag import (
    AgenticQueryRequest,
    AgenticQueryResponse,
    ExplanationRequest,
    ExplanationResponse,
    FeedbackRequest,
    FeedbackResponse,
    AgenticRagStats,
    HealthCheckResponse,
    StreamEvent,
)
from src.services.rag_service import rag_service

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/documents", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """
    添加文档到RAG索引
    """
    try:
        text = request.text or ""
        metadata = request.metadata or {}
        
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文档内容不能为空"
            )
        
        # 生成文档ID
        import uuid
        doc_id = str(uuid.uuid4())
        
        result = await rag_service.add_document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
        )

        if not result["success"]:
            error = result.get("error") or "文档添加失败"
            if error in {"document_text_empty", "document_chunk_empty"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="文档内容为空",
                )
            if (
                "Error code: 401" in error
                or "invalid_api_key" in error
                or "Incorrect API key" in error
            ):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error,
            )

        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    try:
        result = await rag_service.delete_document(document_id)
        if not result["success"]:
            error = result.get("error") or "文档删除失败"
            if "Error code: 401" in error or "invalid_api_key" in error:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/search")
async def search_documents(request: dict):
    """
    搜索文档
    """
    try:
        query = request.get("query", "")
        top_k = request.get("top_k", 3)
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询不能为空"
            )
        
        result = await rag_service.query(
            query=query,
            search_type="semantic",
            limit=top_k,
            score_threshold=0.1,
        )

        if not result["success"]:
            error = result.get("error") or "RAG查询失败"
            if "Error code: 401" in error or "invalid_api_key" in error or "Incorrect API key" in error:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error)

        results: List[dict] = []
        for item in result.get("results") or []:
            results.append(
                {
                    "id": item.get("id", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "metadata": item.get("metadata", {}),
                    "file_path": item.get("file_path", ""),
                }
            )
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
) -> QueryResponse:
    """
    执行 RAG 检索查询
    """
    try:
        import time
        import uuid

        start_time = time.monotonic()
        result = await rag_service.query(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filter_dict=request.filters,
        )
        
        if not result["success"]:
            error = result.get("error") or "RAG查询失败"
            if "Error code: 401" in error or "invalid_api_key" in error or "Incorrect API key" in error:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )
        result["query_id"] = str(uuid.uuid4())
        result["processing_time"] = time.monotonic() - start_time
        result["total_results"] = result.get("count", 0)
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/index/file", response_model=IndexResponse)
async def index_file(
    request: IndexFileRequest,
) -> IndexResponse:
    """
    索引单个文件
    """
    try:
        result = await rag_service.index_file(
            file_path=request.file_path,
            force=request.force,
        )
        
        if not result["success"]:
            error = result.get("error") or "文件索引失败"
            if error.startswith("File not found:"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error)
            if error.startswith("Unsupported file type:"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )
        
        return IndexResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/index/directory", response_model=IndexResponse)
async def index_directory(
    request: IndexDirectoryRequest,
) -> IndexResponse:
    """
    索引目录
    """
    try:
        result = await rag_service.index_directory(
            directory=request.directory,
            recursive=request.recursive,
            force=request.force,
            extensions=request.extensions,
        )
        
        if not result["success"]:
            error = result.get("error") or "目录索引失败"
            if error.startswith("Directory not found:"):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error
            )
        
        return IndexResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Directory indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/index/update", response_model=IndexResponse)
async def update_index(
    request: UpdateIndexRequest,
) -> IndexResponse:
    """
    更新索引
    """
    try:
        result = await rag_service.update_index(
            file_paths=request.file_paths,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return IndexResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/index/stats", response_model=StatsResponse)
async def get_index_stats() -> StatsResponse:
    """
    获取索引统计信息
    """
    try:
        result = await rag_service.get_index_stats()
        
        if not result["success"]:
            return StatsResponse(
                success=False,
                stats={},
                total_disk_size=0,
                health=result.get("health"),
                error=result.get("error")
            )
        
        # 手动构造StatsResponse来确保所有字段都包含
        from src.models.schemas.rag import CollectionStats, HealthStatus
        
        # 构造集合统计
        stats_dict = {}
        for collection_name, collection_data in result["stats"].items():
            stats_dict[collection_name] = CollectionStats(**collection_data)
        
        # 构造健康状态
        health = HealthStatus(**result["health"])
        
        # 构造完整响应
        response = StatsResponse(
            success=result["success"],
            stats=stats_dict,
            total_disk_size=result.get("total_disk_size", 0),
            health=health,
            error=result.get("error")
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/index/reset", response_model=ResetResponse)
async def reset_index(
    request: ResetRequest = ResetRequest(),
) -> ResetResponse:
    """
    重置索引
    """
    try:
        result = await rag_service.reset_index(
            collection=request.collection,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return ResetResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/health")
async def health_check():
    """
    RAG 系统健康检查
    """
    try:
        stats = await rag_service.get_index_stats()
        return {
            "status": "healthy" if stats.get("success") else "degraded",
            "details": stats,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }

# ==================== Agentic RAG 路由 ====================

@router.post("/agentic/query", response_model=AgenticQueryResponse)
async def agentic_query(
    request: AgenticQueryRequest,
) -> AgenticQueryResponse:
    """
    执行 Agentic RAG 智能检索
    
    支持查询理解、自动扩展、多策略检索、结果验证、上下文组合、
    过程解释和失败处理等完整的智能检索流程。
    """
    try:
        # 导入放在这里避免循环导入
        from src.services.agentic_rag_service import agentic_rag_service
        
        result = await agentic_rag_service.intelligent_query(
            query=request.query,
            context_history=request.context_history,
            expansion_strategies=request.expansion_strategies,
            retrieval_strategies=request.retrieval_strategies,
            max_results=request.max_results,
            include_explanation=request.include_explanation,
            session_id=request.session_id,
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Agentic RAG query failed")
            )
        
        return AgenticQueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agentic RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/agentic/query/stream")
async def agentic_query_stream(
    request: AgenticQueryRequest,
) -> StreamingResponse:
    """
    执行 Agentic RAG 智能检索（流式响应）
    
    以Server-Sent Events格式实时返回检索过程中的各个阶段进展，
    包括查询分析、扩展、检索、验证、组合等步骤的实时状态。
    """
    try:
        from src.services.agentic_rag_service import agentic_rag_service
        
        async def generate_stream():
            async for event_data in agentic_rag_service.intelligent_query_stream(
                query=request.query,
                context_history=request.context_history,
                expansion_strategies=request.expansion_strategies,
                retrieval_strategies=request.retrieval_strategies,
                max_results=request.max_results,
                include_explanation=request.include_explanation,
                session_id=request.session_id,
            ):
                # 转换为StreamEvent格式
                event = StreamEvent(**event_data)
                yield f"data: {event.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用nginx缓冲
            }
        )
        
    except Exception as e:
        logger.error(f"Agentic RAG stream query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/agentic/explain", response_model=ExplanationResponse)
async def get_retrieval_explanation(
    query_id: str = None,
    path_id: str = None,
    explanation_level: str = "detailed",
    include_visualization: bool = True,
) -> ExplanationResponse:
    """
    获取检索过程解释
    
    提供指定查询或检索路径的详细解释，包括决策过程、
    置信度分析、改进建议和可视化数据。
    """
    try:
        if not (query_id or path_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="必须提供query_id或path_id")

        from src.services.agentic_rag_service import agentic_rag_service
        
        request_data = ExplanationRequest(
            query_id=query_id,
            path_id=path_id,
            explanation_level=explanation_level,
            include_visualization=include_visualization
        )
        
        result = await agentic_rag_service.get_explanation(request_data)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND if "not found" in result.get("error", "").lower() else status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to get explanation")
            )
        
        return ExplanationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/agentic/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """
    提交用户反馈
    
    接收用户对检索结果的评分、评论和改进建议，
    用于系统学习和优化改进。
    """
    try:
        from src.services.agentic_rag_service import agentic_rag_service
        
        # 在后台处理反馈学习
        background_tasks.add_task(
            agentic_rag_service.process_feedback_learning,
            request
        )
        
        result = await agentic_rag_service.submit_feedback(request)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to submit feedback")
            )
        
        return FeedbackResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit feedback failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/agentic/stats", response_model=AgenticRagStats)
async def get_agentic_rag_stats() -> AgenticRagStats:
    """
    获取 Agentic RAG 统计信息
    
    返回智能检索系统的使用统计、性能指标、
    策略效果分析等综合数据。
    """
    try:
        from src.services.agentic_rag_service import agentic_rag_service
        
        result = await agentic_rag_service.get_statistics()
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to get statistics")
            )
        
        return AgenticRagStats(**result["data"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get Agentic RAG stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/agentic/health", response_model=HealthCheckResponse)
async def agentic_rag_health_check() -> HealthCheckResponse:
    """
    Agentic RAG 系统健康检查
    
    检查智能检索系统各个组件的运行状态，
    包括查询分析器、扩展器、多代理检索器、验证器等。
    """
    try:
        from src.services.agentic_rag_service import agentic_rag_service
        
        result = await agentic_rag_service.health_check()
        
        return HealthCheckResponse(**result)
    except Exception as e:
        logger.error(f"Agentic RAG health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            components={},
            error=str(e)
        )

# ==================== GraphRAG 路由 ====================

@router.post("/graphrag/query")
async def graphrag_query(request: dict):
    """
    GraphRAG增强查询
    
    结合知识图谱和向量检索的混合RAG查询，提供更准确的结果
    """
    try:
        # 导入GraphRAG组件
        from src.ai.graphrag.core_engine import get_graphrag_engine
        from src.ai.graphrag.data_models import create_graph_rag_request, RetrievalMode, validate_graph_rag_request
        
        # 解析请求参数
        query = request.get('query', '')
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="查询不能为空"
            )
        
        # 创建GraphRAG请求
        graphrag_request = create_graph_rag_request(
            query=query,
            retrieval_mode=RetrievalMode(request.get('retrieval_mode', 'hybrid')),
            max_docs=request.get('max_docs', 10),
            include_reasoning=request.get('include_reasoning', True),
            expansion_depth=request.get('expansion_depth', 2),
            confidence_threshold=request.get('confidence_threshold', 0.6),
            filters=request.get('filters')
        )
        
        # 验证请求
        validation_errors = validate_graph_rag_request(graphrag_request)
        if validation_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"请求验证失败: {'; '.join(validation_errors)}"
            )
        
        # 执行GraphRAG查询
        engine = await get_graphrag_engine()
        result = await engine.enhanced_query(graphrag_request)
        
        return {
            "success": True,
            "data": result,
            "query_id": result["query_id"],
            "performance_metrics": result["performance_metrics"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GraphRAG查询失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GraphRAG查询失败: {str(e)}"
        )

@router.get("/graphrag/health")
async def graphrag_health_check():
    """
    GraphRAG系统健康检查
    """
    try:
        from src.ai.graphrag.core_engine import get_graphrag_engine
        
        engine = await get_graphrag_engine()
        stats = await engine.get_performance_stats()
        
        return {
            "status": "healthy" if stats.get("engine_status") == "initialized" else "unhealthy",
            "details": stats
        }
    except Exception as e:
        logger.error(f"GraphRAG健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
