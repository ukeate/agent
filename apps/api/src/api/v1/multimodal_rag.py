"""多模态RAG API端点"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import shutil
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import Field
import aiofiles
from src.ai.rag.multimodal_config import (
    MultimodalConfig,
    ProcessedDocument,
    QAResponse
)
from src.api.base_model import ApiBaseModel
from src.ai.rag.multimodal_qa_chain import MultimodalQAChain
from src.ai.rag.document_processor import MultimodalDocumentProcessor
from src.ai.rag.multimodal_vectorstore import MultimodalVectorStore

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal-rag", tags=["Multimodal RAG"])

class MultimodalQueryRequest(ApiBaseModel):
    """多模态查询请求"""
    query: str = Field(description="查询文本")
    stream: bool = Field(default=False, description="是否流式响应")
    max_tokens: Optional[int] = Field(default=1000, description="最大生成token数")
    temperature: float = Field(default=0.7, description="生成温度")
    include_images: bool = Field(default=True, description="是否包含图像")
    include_tables: bool = Field(default=True, description="是否包含表格")
    top_k: Optional[int] = Field(default=None, description="检索结果数量")

class MultimodalQueryResponse(ApiBaseModel):
    """多模态查询响应"""
    answer: str = Field(description="回答内容")
    sources: List[str] = Field(description="引用来源")
    confidence: float = Field(description="置信度分数")
    processing_time: float = Field(description="处理时间(秒)")
    context_used: Dict[str, int] = Field(description="使用的上下文统计")

class DocumentUploadResponse(ApiBaseModel):
    """文档上传响应"""
    doc_id: str = Field(description="文档ID")
    source_file: str = Field(description="源文件名")
    content_type: str = Field(description="内容类型")
    num_text_chunks: int = Field(description="文本块数量")
    num_images: int = Field(description="图像数量")
    num_tables: int = Field(description="表格数量")
    processing_time: float = Field(description="处理时间(秒)")

class VectorStoreStatus(ApiBaseModel):
    """向量存储状态"""
    text_documents: int = Field(description="文本文档数")
    image_documents: int = Field(description="图像文档数")
    table_documents: int = Field(description="表格文档数")
    total_documents: int = Field(description="总文档数")
    embedding_dimension: int = Field(description="嵌入维度")

# 全局实例
config = MultimodalConfig()
qa_chain: Optional[MultimodalQAChain] = None
document_processor: Optional[MultimodalDocumentProcessor] = None
vector_store: Optional[MultimodalVectorStore] = None

def get_qa_chain() -> MultimodalQAChain:
    """获取QA链实例"""
    global qa_chain
    if qa_chain is None:
        qa_chain = MultimodalQAChain(config, vector_store=get_vector_store())
    return qa_chain

def get_document_processor() -> MultimodalDocumentProcessor:
    """获取文档处理器实例"""
    global document_processor
    if document_processor is None:
        document_processor = MultimodalDocumentProcessor(config)
    return document_processor

def get_vector_store() -> MultimodalVectorStore:
    """获取向量存储实例"""
    global vector_store
    if vector_store is None:
        vector_store = MultimodalVectorStore(config)
    return vector_store

def _is_supported_file_type(filename: str) -> bool:
    """检查是否为支持的文件类型"""
    supported_extensions = {
        '.txt', '.md', '.csv', '.json',  # 文本文件
        '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt',  # 文档
        '.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff',  # 图像
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav'  # 多媒体
    }
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in supported_extensions

async def _ingest_upload_files(
    files: List[UploadFile],
    document_processor: MultimodalDocumentProcessor,
    vector_store: MultimodalVectorStore,
) -> List[str]:
    temp_files = []
    try:
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="文件名不能为空")
            if not _is_supported_file_type(file.filename):
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.filename}")

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file.filename).suffix
            )
            temp_files.append(temp_file.name)

            content = await file.read()
            if len(content) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="文件大小超过50MB限制")
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="文件不能为空")

            async with aiofiles.open(temp_file.name, 'wb') as f:
                await f.write(content)

            processed_doc = await document_processor.process_document(temp_file.name)
            processed_doc = processed_doc.model_copy(update={"source_file": file.filename})
            await vector_store.add_documents(processed_doc)

        return [f.filename for f in files if f.filename]
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")

@router.post("/query", response_model=MultimodalQueryResponse)
async def multimodal_query(
    request: MultimodalQueryRequest,
    qa_chain: MultimodalQAChain = Depends(get_qa_chain)
) -> MultimodalQueryResponse:
    """多模态RAG查询接口
    
    Args:
        request: 查询请求
        qa_chain: QA链依赖注入
        
    Returns:
        查询响应
    """
    try:
        # 执行查询
        response = await qa_chain.arun(
            query=request.query,
            stream=request.stream,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_images=request.include_images,
            include_tables=request.include_tables,
            top_k=request.top_k
        )
        
        return MultimodalQueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            processing_time=response.processing_time,
            context_used=response.context_used
        )
        
    except Exception as e:
        logger.error(f"Error in multimodal query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query-with-files")
async def multimodal_query_with_files(
    query: str = Form(..., description="查询文本"),
    files: List[UploadFile] = File(None, description="上传的文件"),
    stream: bool = Form(False, description="是否流式响应"),
    max_tokens: int = Form(1000, description="最大token数"),
    qa_chain: MultimodalQAChain = Depends(get_qa_chain),
    document_processor: MultimodalDocumentProcessor = Depends(get_document_processor),
    vector_store: MultimodalVectorStore = Depends(get_vector_store)
) -> MultimodalQueryResponse:
    """带文件上传的多模态查询
    
    Args:
        query: 查询文本
        files: 上传的文件列表
        stream: 是否流式响应
        max_tokens: 最大token数
        qa_chain: QA链
        document_processor: 文档处理器
        vector_store: 向量存储
        
    Returns:
        查询响应
    """
    try:
        if files:
            await _ingest_upload_files(files, document_processor, vector_store)
        
        # 执行查询
        response = await qa_chain.arun(
            query=query,
            stream=stream,
            max_tokens=max_tokens,
            context_files=[f.filename for f in files] if files else []
        )
        
        return MultimodalQueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            processing_time=response.processing_time,
            context_used=response.context_used
        )
        
    except Exception as e:
        logger.error(f"Error in query with files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query-with-details")
async def multimodal_query_with_details(
    query: str = Form(..., description="查询文本"),
    files: List[UploadFile] = File(None, description="上传的文件"),
    qa_chain: MultimodalQAChain = Depends(get_qa_chain),
    document_processor: MultimodalDocumentProcessor = Depends(get_document_processor),
    vector_store: MultimodalVectorStore = Depends(get_vector_store),
):
    import time
    start_time = time.time()
    try:
        filenames = []
        if files:
            filenames = await _ingest_upload_files(files, document_processor, vector_store)

        query_context = qa_chain.query_analyzer.analyze_query(query=query, files=filenames)
        retrieval_results = await qa_chain.retriever.retrieve(query_context)
        answer, confidence = qa_chain._generate_answer(query, retrieval_results)

        weights = qa_chain.retriever._calculate_dynamic_weights(query_context)
        retrieval_top_k = query_context.top_k or config.retrieval_top_k
        similarity_threshold = query_context.similarity_threshold or config.similarity_threshold

        def _convert_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for it in items:
                md = it.get("metadata") or {}
                out.append(
                    {
                        "content": it.get("content", ""),
                        "score": float(it.get("score") or 0.0),
                        "source": md.get("source", ""),
                        "metadata": md,
                    }
                )
            return out

        return {
            "query_analysis": {
                "query_type": query_context.query_type.value,
                "requires_image_search": bool(query_context.requires_image_search),
                "requires_table_search": bool(query_context.requires_table_search),
                "filters": query_context.filters,
                "top_k": int(retrieval_top_k),
                "similarity_threshold": float(similarity_threshold),
            },
            "retrieval_strategy": {
                "strategy": "hybrid" if query_context.query_type.value == "mixed" else query_context.query_type.value,
                "weights": {
                    "text": float(weights.text_weight),
                    "image": float(weights.image_weight),
                    "table": float(weights.table_weight),
                },
                "reranking": bool(config.rerank_enabled),
                "top_k": int(retrieval_top_k),
                "similarity_threshold": float(similarity_threshold),
            },
            "retrieval_results": {
                "texts": _convert_items(retrieval_results.texts),
                "images": _convert_items(retrieval_results.images),
                "tables": _convert_items(retrieval_results.tables),
                "sources": retrieval_results.sources,
                "total_results": int(retrieval_results.total_results),
                "retrieval_time_ms": float(retrieval_results.retrieval_time_ms),
            },
            "qa_response": {
                "answer": answer,
                "confidence": float(confidence),
                "processing_time_ms": float((time.time() - start_time) * 1000),
                "context_used": {
                    "text_chunks": len(retrieval_results.texts),
                    "images": len(retrieval_results.images),
                    "tables": len(retrieval_results.tables),
                },
            },
        }
    except Exception as e:
        logger.error(f"Error in query with details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream-query")
async def stream_multimodal_query(
    request: MultimodalQueryRequest,
    qa_chain: MultimodalQAChain = Depends(get_qa_chain)
):
    """流式多模态查询
    
    Args:
        request: 查询请求
        qa_chain: QA链
        
    Returns:
        流式响应
    """
    async def generate():
        """生成流式响应"""
        try:
            import json
            async for chunk in qa_chain.stream_response(
                query=request.query,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                include_images=request.include_images,
                include_tables=request.include_tables
            ):
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            import json
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="要上传的文档"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    document_processor: MultimodalDocumentProcessor = Depends(get_document_processor),
    vector_store: MultimodalVectorStore = Depends(get_vector_store)
) -> DocumentUploadResponse:
    """上传并处理文档
    
    Args:
        file: 上传的文件
        background_tasks: 后台任务
        document_processor: 文档处理器
        vector_store: 向量存储
        
    Returns:
        上传响应
    """
    import time
    start_time = time.time()
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix
    )
    
    try:
        # 保存文件
        content = await file.read()
        async with aiofiles.open(temp_file.name, 'wb') as f:
            await f.write(content)
        
        # 处理文档
        processed_doc = await document_processor.process_document(temp_file.name)
        processed_doc = processed_doc.model_copy(update={"source_file": file.filename})
        
        # 添加到向量存储
        success = await vector_store.add_documents(processed_doc)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to vector store")
        
        # 后台清理临时文件
        background_tasks.add_task(os.unlink, temp_file.name)
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            doc_id=processed_doc.doc_id,
            source_file=file.filename,
            content_type=processed_doc.content_type,
            num_text_chunks=len(processed_doc.texts),
            num_images=len(processed_doc.images),
            num_tables=len(processed_doc.tables),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # 清理临时文件
        try:
            if temp_file.name and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception:
            logger.exception("清理临时文件失败", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(..., description="要上传的文档列表"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    document_processor: MultimodalDocumentProcessor = Depends(get_document_processor),
    vector_store: MultimodalVectorStore = Depends(get_vector_store)
) -> Dict[str, Any]:
    """批量上传文档
    
    Args:
        files: 文件列表
        background_tasks: 后台任务
        document_processor: 文档处理器
        vector_store: 向量存储
        
    Returns:
        批量上传结果
    """
    import time
    start_time = time.time()
    
    results = []
    temp_files = []
    
    try:
        # 保存所有文件
        for file in files:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file.filename).suffix
            )
            temp_files.append(temp_file.name)
            
            content = await file.read()
            async with aiofiles.open(temp_file.name, 'wb') as f:
                await f.write(content)
        
        # 批量处理文档
        processed_docs = await document_processor.process_batch(temp_files)
        
        # 添加到向量存储
        temp_to_name = {tmp: f.filename for tmp, f in zip(temp_files, files) if f.filename}
        for doc in processed_docs:
            doc = doc.model_copy(update={"source_file": temp_to_name.get(doc.source_file, doc.source_file)})
            success = await vector_store.add_documents(doc)
            results.append({
                "filename": doc.source_file,
                "doc_id": doc.doc_id,
                "success": success
            })
        
        # 后台清理临时文件
        for temp_file in temp_files:
            background_tasks.add_task(os.unlink, temp_file)
        
        processing_time = time.time() - start_time
        
        return {
            "total_files": len(files),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "processing_time": processing_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        # 清理临时文件
        for temp_file in temp_files:
            try:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                logger.exception("清理临时文件失败", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=VectorStoreStatus)
async def get_vector_store_status(
    vector_store: MultimodalVectorStore = Depends(get_vector_store)
) -> VectorStoreStatus:
    """获取向量存储状态
    
    Args:
        vector_store: 向量存储
        
    Returns:
        存储状态
    """
    try:
        stats = vector_store.get_statistics()
        return VectorStoreStatus(**stats)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_multimodal_rag_stats(
    vector_store: MultimodalVectorStore = Depends(get_vector_store),
):
    stats = vector_store.get_statistics()
    cache = qa_chain.get_cache_stats() if qa_chain else {"enabled": bool(config.cache_enabled), "size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0}
    return {
        "vector_store_type": config.vector_store_type.value,
        "embedding_model": config.text_embedding_model.value,
        "embedding_dimension": int(config.embedding_dimension),
        "collections": [
            {"name": "multimodal_text", "count": int(stats["text_documents"])},
            {"name": "multimodal_image", "count": int(stats["image_documents"])},
            {"name": "multimodal_table", "count": int(stats["table_documents"])},
        ],
        "text_documents": int(stats["text_documents"]),
        "image_documents": int(stats["image_documents"]),
        "table_documents": int(stats["table_documents"]),
        "total_documents": int(stats["total_documents"]),
        "cache": cache,
        "retrieval": {
            "top_k": int(config.retrieval_top_k),
            "similarity_threshold": float(config.similarity_threshold),
            "rerank_enabled": bool(config.rerank_enabled),
        },
    }

@router.delete("/clear")
async def clear_vector_store(
    vector_store: MultimodalVectorStore = Depends(get_vector_store)
) -> Dict[str, str]:
    """清空向量存储
    
    Args:
        vector_store: 向量存储
        
    Returns:
        操作结果
    """
    try:
        vector_store.clear_all()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache")
async def clear_query_cache(
    qa_chain: MultimodalQAChain = Depends(get_qa_chain)
) -> Dict[str, str]:
    """清空查询缓存
    
    Args:
        qa_chain: QA链
        
    Returns:
        操作结果
    """
    try:
        qa_chain.clear_cache()
        return {"message": "Query cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
from src.core.logging import get_logger
