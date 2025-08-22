"""文档处理API接口"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import os
import logging

from src.ai.document_processing import (
    DocumentProcessor,
    IntelligentChunker,
    DocumentRelationshipAnalyzer,
    AutoTagger,
    DocumentVersionManager
)
from src.ai.document_processing.chunkers import ChunkStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# 全局实例（实际应该使用依赖注入）
document_processor = DocumentProcessor()
chunker = IntelligentChunker()
relationship_analyzer = DocumentRelationshipAnalyzer()
auto_tagger = AutoTagger()
version_manager = DocumentVersionManager()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    enable_ocr: bool = Query(False, description="启用OCR识别"),
    extract_images: bool = Query(True, description="提取图像"),
    auto_tag: bool = Query(True, description="自动生成标签"),
    chunk_strategy: str = Query("semantic", description="分块策略")
):
    """上传并处理文档
    
    Args:
        file: 上传的文件
        enable_ocr: 是否启用OCR
        extract_images: 是否提取图像
        auto_tag: 是否自动生成标签
        chunk_strategy: 分块策略
    
    Returns:
        处理后的文档信息
    """
    try:
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # 处理文档
        processor = DocumentProcessor(
            enable_ocr=enable_ocr,
            extract_images=extract_images
        )
        
        processed_doc = await processor.process_document(
            tmp_path,
            extract_metadata=True,
            generate_embeddings=False
        )
        
        # 智能分块
        if chunk_strategy:
            strategy_map = {
                "semantic": ChunkStrategy.SEMANTIC,
                "fixed": ChunkStrategy.FIXED,
                "adaptive": ChunkStrategy.ADAPTIVE,
                "sliding_window": ChunkStrategy.SLIDING_WINDOW,
                "hierarchical": ChunkStrategy.HIERARCHICAL
            }
            
            chunker.strategy = strategy_map.get(
                chunk_strategy, 
                ChunkStrategy.SEMANTIC
            )
            
            chunks = await chunker.chunk_document(
                processed_doc.content,
                content_type=processed_doc.file_type,
                metadata=processed_doc.metadata
            )
            
            # 添加分块信息
            processed_doc.processing_info["chunks"] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:200],  # 预览
                    "type": chunk.chunk_type,
                    "index": chunk.chunk_index
                }
                for chunk in chunks[:10]  # 限制返回数量
            ]
            processed_doc.processing_info["total_chunks"] = len(chunks)
        
        # 自动标签
        if auto_tag:
            tags = await auto_tagger.generate_tags(processed_doc.to_dict())
            processed_doc.processing_info["auto_tags"] = [
                {
                    "tag": tag.tag,
                    "category": tag.category.value,
                    "confidence": tag.confidence
                }
                for tag in tags
            ]
        
        # 创建版本
        version = await version_manager.create_version(
            doc_id=processed_doc.doc_id,
            content=processed_doc.content,
            change_summary="Initial upload",
            metadata=processed_doc.metadata
        )
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        # 返回结果
        result = processed_doc.to_dict()
        result["version"] = {
            "version_id": version.version_id,
            "version_number": version.version_number
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        # 清理临时文件
        if 'tmp_path' in locals() and tmp_path.exists():
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    concurrent_limit: int = Query(5, description="并发处理限制"),
    continue_on_error: bool = Query(True, description="遇到错误是否继续")
):
    """批量上传并处理文档
    
    Args:
        files: 上传的文件列表
        concurrent_limit: 并发限制
        continue_on_error: 错误处理策略
    
    Returns:
        批量处理结果
    """
    temp_files = []
    results = []
    errors = []
    
    try:
        # 保存所有上传文件
        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file.filename).suffix
            ) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append(Path(tmp_file.name))
        
        # 批量处理
        processed_docs = await document_processor.process_batch(
            temp_files,
            concurrent_limit=concurrent_limit,
            continue_on_error=continue_on_error
        )
        
        # 转换结果
        for doc in processed_docs:
            results.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "file_type": doc.file_type,
                "status": "success"
            })
        
        return JSONResponse(content={
            "total": len(files),
            "success": len(results),
            "failed": len(files) - len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 清理临时文件
        for tmp_file in temp_files:
            if tmp_file.exists():
                os.unlink(tmp_file)


@router.get("/supported-formats")
async def get_supported_formats():
    """获取支持的文档格式
    
    Returns:
        支持的文件格式列表
    """
    return {
        "formats": document_processor.get_supported_formats(),
        "categories": {
            "documents": [".pdf", ".docx", ".doc", ".pptx", ".ppt"],
            "spreadsheets": [".xlsx", ".xls", ".csv"],
            "text": [".txt", ".md", ".markdown", ".rst", ".log"],
            "code": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"],
            "web": [".html", ".htm", ".xml", ".json", ".yaml", ".yml"]
        }
    }


@router.post("/{doc_id}/analyze-relationships")
async def analyze_document_relationships(
    doc_id: str,
    related_doc_ids: List[str] = Query([], description="相关文档ID列表")
):
    """分析文档关系
    
    Args:
        doc_id: 文档ID
        related_doc_ids: 相关文档ID列表
    
    Returns:
        关系分析结果
    """
    try:
        # 这里应该从数据库获取文档
        # 简化版本：创建模拟文档
        documents = [
            {"doc_id": doc_id, "content": "Main document content", "title": "Main Doc"},
            *[{"doc_id": rid, "content": f"Related doc {rid}", "title": f"Doc {rid}"} 
              for rid in related_doc_ids]
        ]
        
        # 分析关系
        result = await relationship_analyzer.analyze_relationships(documents)
        
        return JSONResponse(content={
            "doc_id": doc_id,
            "relationships": [
                {
                    "source": rel.source_doc_id,
                    "target": rel.target_doc_id,
                    "type": rel.relationship_type.value,
                    "confidence": rel.confidence
                }
                for rel in result.get("relationships", [])
            ],
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "documents": cluster.documents,
                    "topic": cluster.topic
                }
                for cluster in result.get("clusters", [])
            ],
            "summary": result.get("summary", {})
        })
        
    except Exception as e:
        logger.error(f"Relationship analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{doc_id}/generate-tags")
async def generate_document_tags(
    doc_id: str,
    content: str = Query(..., description="文档内容"),
    existing_tags: List[str] = Query([], description="已有标签")
):
    """为文档生成标签
    
    Args:
        doc_id: 文档ID
        content: 文档内容
        existing_tags: 已有标签
    
    Returns:
        生成的标签列表
    """
    try:
        document = {
            "doc_id": doc_id,
            "content": content,
            "title": content[:100] if len(content) > 100 else content
        }
        
        tags = await auto_tagger.generate_tags(document, existing_tags)
        
        return JSONResponse(content={
            "doc_id": doc_id,
            "tags": [
                {
                    "tag": tag.tag,
                    "category": tag.category.value,
                    "confidence": tag.confidence,
                    "source": tag.source
                }
                for tag in tags
            ]
        })
        
    except Exception as e:
        logger.error(f"Tag generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/versions")
async def get_document_versions(
    doc_id: str,
    limit: Optional[int] = Query(None, description="限制返回数量")
):
    """获取文档版本历史
    
    Args:
        doc_id: 文档ID
        limit: 限制数量
    
    Returns:
        版本历史列表
    """
    try:
        versions = await version_manager.get_version_history(doc_id, limit)
        
        return JSONResponse(content={
            "doc_id": doc_id,
            "versions": [
                {
                    "version_id": v.version_id,
                    "version_number": v.version_number,
                    "created_at": v.created_at.isoformat(),
                    "change_summary": v.change_summary,
                    "is_current": v.is_current
                }
                for v in versions
            ]
        })
        
    except Exception as e:
        logger.error(f"Version history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{doc_id}/rollback")
async def rollback_document_version(
    doc_id: str,
    target_version_id: str = Query(..., description="目标版本ID")
):
    """回滚文档版本
    
    Args:
        doc_id: 文档ID
        target_version_id: 目标版本ID
    
    Returns:
        新版本信息
    """
    try:
        new_version = await version_manager.rollback_version(
            doc_id, 
            target_version_id
        )
        
        return JSONResponse(content={
            "doc_id": doc_id,
            "new_version": {
                "version_id": new_version.version_id,
                "version_number": new_version.version_number,
                "created_at": new_version.created_at.isoformat(),
                "change_summary": new_version.change_summary
            }
        })
        
    except Exception as e:
        logger.error(f"Version rollback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))