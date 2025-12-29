"""
多模态处理API路由
"""

import re
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from src.ai.multimodal import (
    OpenAIMultimodalClient,
    MultimodalProcessor,
    ProcessingPipeline,
    ContentType,
    ProcessingOptions,
    ModelPriority,
    ModelComplexity,
    StructuredDataExtractor
)
from src.ai.multimodal.config import ModelConfig
from src.services.file_service import FileUploadService
from src.models.schemas.multimodal import (
    ProcessingRequest,
    ProcessingResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    FileUploadResponse,
    ProcessingStatusResponse

)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal", tags=["multimodal"])

# 全局实例
_multimodal_client: Optional[OpenAIMultimodalClient] = None
_processor: Optional[MultimodalProcessor] = None
_pipeline: Optional[ProcessingPipeline] = None
_file_service: Optional[FileUploadService] = None

async def get_multimodal_client() -> OpenAIMultimodalClient:
    """获取多模态客户端"""
    global _multimodal_client
    if not _multimodal_client:
        _multimodal_client = OpenAIMultimodalClient()
        await _multimodal_client.__aenter__()
    return _multimodal_client

async def get_processor() -> MultimodalProcessor:
    """获取处理器"""
    global _processor
    if not _processor:
        client = await get_multimodal_client()
        _processor = MultimodalProcessor(client)
    return _processor

async def get_pipeline() -> ProcessingPipeline:
    """获取处理管道"""
    global _pipeline
    if not _pipeline:
        processor = await get_processor()
        _pipeline = ProcessingPipeline(processor)
        await _pipeline.start()
    return _pipeline

async def get_file_service() -> FileUploadService:
    """获取文件服务"""
    global _file_service
    if not _file_service:
        client = await get_multimodal_client()
        _file_service = FileUploadService(openai_client=client)
    return _file_service

@router.get("/models")
async def list_model_configs():
    """获取多模态模型配置"""
    models = []
    for name, config in ModelConfig.MODEL_CONFIGS.items():
        models.append({"name": name, **config})
    return {"models": models, "total": len(models)}

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_service: FileUploadService = Depends(get_file_service)
):
    """上传文件进行多模态处理"""
    try:
        # 验证文件名安全性
        if not file.filename or not _is_safe_filename(file.filename):
            raise HTTPException(status_code=400, detail="文件名无效或不安全")
        
        # 读取文件内容
        content = await file.read()
        
        # 检查文件大小
        if len(content) > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=413, detail="文件大小超过100MB限制")
        
        # 检查空文件
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="文件不能为空")
        
        # 保存文件
        multimodal_content = await file_service.save_uploaded_file(
            content,
            file.filename,
            file.content_type
        )
        
        return FileUploadResponse(
            content_id=multimodal_content.content_id,
            content_type=multimodal_content.content_type,
            file_size=multimodal_content.file_size,
            mime_type=multimodal_content.mime_type,
            metadata=multimodal_content.metadata
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"文件上传验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败")

@router.post("/process", response_model=ProcessingResponse)
async def process_content(
    request: ProcessingRequest,
    processor: MultimodalProcessor = Depends(get_processor)
):
    """处理多模态内容"""
    try:
        # 构建处理选项
        options = ProcessingOptions(
            priority=ModelPriority(request.priority) if request.priority else ModelPriority.BALANCED,
            complexity=ModelComplexity(request.complexity) if request.complexity else ModelComplexity.MEDIUM,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature or 0.1,
            enable_cache=request.enable_cache if request.enable_cache is not None else True,
            extract_text=request.extract_text if request.extract_text is not None else True,
            extract_objects=request.extract_objects if request.extract_objects is not None else True,
            extract_sentiment=request.extract_sentiment if request.extract_sentiment is not None else False
        )
        
        # 获取文件信息
        file_service = await get_file_service()
        file_info = await file_service.get_file_info(request.content_id)
        
        if not file_info:
            raise HTTPException(status_code=404, detail="文件未找到")
        
        # 创建多模态内容对象
        from src.ai.multimodal.types import MultimodalContent
        content = MultimodalContent(
            content_id=request.content_id,
            content_type=ContentType(request.content_type),
            file_path=file_info["file_path"],
            file_size=file_info["file_size"]
        )
        
        # 处理内容
        result = await processor.process_content(content, options)
        
        # 提取结构化数据
        if request.content_type == "image":
            structured_data = StructuredDataExtractor.extract_from_image(result.extracted_data)
        elif request.content_type == "document":
            structured_data = StructuredDataExtractor.extract_from_document(result.extracted_data)
        elif request.content_type == "video":
            structured_data = StructuredDataExtractor.extract_from_video(result.extracted_data)
        else:
            structured_data = result.extracted_data
        
        return ProcessingResponse(
            content_id=result.content_id,
            status=result.status.value,
            extracted_data=result.extracted_data,
            structured_data=structured_data,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理内容失败: {e}")
        raise HTTPException(status_code=500, detail="处理失败")

@router.post("/process/batch", response_model=BatchProcessingResponse)
async def process_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    pipeline: ProcessingPipeline = Depends(get_pipeline)
):
    """批量处理多模态内容"""
    try:
        # 构建处理选项
        options = ProcessingOptions(
            priority=ModelPriority(request.priority) if request.priority else ModelPriority.BALANCED,
            complexity=ModelComplexity(request.complexity) if request.complexity else ModelComplexity.MEDIUM,
            max_tokens=request.max_tokens or 1000
        )
        
        # 获取文件服务
        file_service = await get_file_service()
        
        # 构建内容列表
        contents = []
        for content_id in request.content_ids:
            file_info = await file_service.get_file_info(content_id)
            if file_info:
                from src.ai.multimodal.types import MultimodalContent
                content = MultimodalContent(
                    content_id=content_id,
                    content_type=ContentType.DOCUMENT,  # 默认类型
                    file_path=file_info["file_path"],
                    file_size=file_info["file_size"]
                )
                contents.append(content)
        
        if not contents:
            raise HTTPException(status_code=404, detail="没有找到有效文件")
        
        # 提交批量处理
        submitted_ids = await pipeline.submit_batch(contents, options)
        
        return BatchProcessingResponse(
            batch_id=f"batch_{submitted_ids[0][:8]}",
            content_ids=submitted_ids,
            status="processing",
            total_items=len(submitted_ids),
            completed_items=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        raise HTTPException(status_code=500, detail="批量处理失败")

@router.get("/status/{content_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    content_id: str,
    pipeline: ProcessingPipeline = Depends(get_pipeline)
):
    """获取处理状态"""
    try:
        result = await pipeline.get_processing_status(content_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="处理任务未找到")
        
        return ProcessingStatusResponse(
            content_id=result.content_id,
            status=result.status.value,
            confidence_score=result.confidence_score,
            processing_time=result.processing_time,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")

@router.get("/queue/status")
async def get_queue_status(
    pipeline: ProcessingPipeline = Depends(get_pipeline)
):
    """获取队列状态"""
    try:
        status = pipeline.get_queue_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取队列状态失败")

@router.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Query(default="分析这张图像"),
    extract_text: bool = Query(default=True),
    extract_objects: bool = Query(default=True),
    priority: str = Query(default="balanced"),
    client: OpenAIMultimodalClient = Depends(get_multimodal_client)
):
    """直接分析图像（不保存文件）"""
    try:
        # 读取图像内容
        image_data = await file.read()
        
        # 检查大小
        if len(image_data) > 20 * 1024 * 1024:  # 20MB
            raise HTTPException(status_code=413, detail="图像大小超过20MB限制")
        
        # 直接处理图像
        result = await client.process_image(
            image_data,
            prompt,
            priority=ModelPriority(priority),
            complexity=ModelComplexity.MEDIUM
        )
        
        # 解析结果
        import json
        try:
            extracted_data = json.loads(result.get("content", "{}"))
        except:
            extracted_data = {"description": result.get("content", "")}
        
        # 提取结构化数据
        structured_data = StructuredDataExtractor.extract_from_image(extracted_data)
        
        return JSONResponse(content={
            "extracted_data": extracted_data,
            "structured_data": structured_data,
            "model_used": result.get("model"),
            "tokens_used": result.get("usage"),
            "cost": result.get("cost"),
            "processing_time": result.get("duration")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像分析失败: {e}")
        raise HTTPException(status_code=500, detail="图像分析失败")

@router.delete("/file/{content_id}")
async def delete_file(
    content_id: str,
    file_service: FileUploadService = Depends(get_file_service)
):
    """删除上传的文件"""
    try:
        success = await file_service.delete_file(content_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="文件未找到")
        
        return {"message": "文件删除成功", "content_id": content_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        raise HTTPException(status_code=500, detail="删除文件失败")

# 辅助函数
def _is_safe_filename(filename: str) -> bool:
    """验证文件名安全性"""
    if not filename or len(filename) > 255:
        return False
    
    # 检查危险字符和路径遍历
    dangerous_patterns = [
        r'\.\.',  # 路径遍历
        r'[<>:"|?*]',  # Windows非法字符
        r'[\x00-\x1f]',  # 控制字符
        r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)',  # Windows保留名
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, filename, re.IGNORECASE):
            return False
    
    # 必须有合法的扩展名
    allowed_extensions = {
        '.jpg', '.jpeg', '.png', '.webp', '.gif',  # 图像
        '.pdf', '.txt', '.docx', '.md', '.csv', '.xlsx',  # 文档
        '.mp4', '.avi', '.mov', '.mkv', '.webm',  # 视频
        '.mp3', '.wav', '.flac', '.ogg', '.m4a'  # 音频
    }
    
    file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    return file_ext in allowed_extensions

# 清理任务
async def cleanup_on_shutdown():
    """关闭时清理资源"""
    global _multimodal_client, _pipeline
    
    if _pipeline:
        await _pipeline.stop()
    
    if _multimodal_client:
        await _multimodal_client.__aexit__(None, None, None)
