"""
文件管理API路由
"""

import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
import structlog

from src.services.file_service import FileUploadService
from src.ai.multimodal.client import OpenAIMultimodalClient

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/files", tags=["files"])

# 全局实例
_file_service: Optional[FileUploadService] = None


async def get_file_service() -> FileUploadService:
    """获取文件服务实例"""
    global _file_service
    if not _file_service:
        openai_client = OpenAIMultimodalClient()
        await openai_client.__aenter__()
        _file_service = FileUploadService(
            upload_path="/tmp/uploads",
            openai_client=openai_client
        )
    return _file_service


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_service: FileUploadService = Depends(get_file_service)
):
    """
    上传文件
    
    支持的文件类型：
    - 图片：JPG, PNG, GIF, BMP, WEBP
    - 文档：PDF, DOC, DOCX, TXT
    - 表格：XLS, XLSX, CSV
    - 演示：PPT, PPTX
    - 音频：MP3, WAV, AAC
    - 视频：MP4, AVI, MOV
    """
    try:
        # 检查文件大小
        file_data = await file.read()
        if len(file_data) > file_service.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"文件太大，最大支持 {file_service.max_file_size // (1024*1024)}MB"
            )
        
        # 保存文件
        content = await file_service.save_uploaded_file(
            file_data=file_data,
            filename=file.filename,
            content_type=file.content_type
        )
        
        return JSONResponse({
            "success": True,
            "message": "文件上传成功",
            "data": {
                "file_id": content.content_id,
                "filename": file.filename,
                "file_size": content.file_size,
                "content_type": content.content_type.value,
                "mime_type": content.mime_type,
                "upload_time": content.metadata.get("upload_timestamp"),
                "openai_file_id": content.metadata.get("openai_file_id")
            }
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail="文件上传失败")


@router.post("/upload/batch")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    file_service: FileUploadService = Depends(get_file_service)
):
    """批量上传文件"""
    results = []
    errors = []
    
    for file in files:
        try:
            file_data = await file.read()
            if len(file_data) > file_service.max_file_size:
                errors.append({
                    "filename": file.filename,
                    "error": f"文件太大，最大支持 {file_service.max_file_size // (1024*1024)}MB"
                })
                continue
            
            content = await file_service.save_uploaded_file(
                file_data=file_data,
                filename=file.filename,
                content_type=file.content_type
            )
            
            results.append({
                "file_id": content.content_id,
                "filename": file.filename,
                "file_size": content.file_size,
                "content_type": content.content_type.value,
                "status": "success"
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "success": len(errors) == 0,
        "uploaded_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors
    })


@router.get("/list")
async def list_files(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    file_service: FileUploadService = Depends(get_file_service)
):
    """获取文件列表"""
    try:
        files = []
        upload_path = file_service.upload_path
        
        # 遍历上传目录
        for root, dirs, filenames in os.walk(upload_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                stat = os.stat(file_path)
                
                # 从文件名中提取文件ID
                if "_" in filename:
                    file_id = filename.split("_")[0]
                    original_name = "_".join(filename.split("_")[1:])
                else:
                    file_id = filename
                    original_name = filename
                
                files.append({
                    "file_id": file_id,
                    "filename": original_name,
                    "file_size": stat.st_size,
                    "created_at": stat.st_ctime,
                    "modified_at": stat.st_mtime,
                    "file_path": file_path
                })
        
        # 排序和分页
        files.sort(key=lambda x: x["created_at"], reverse=True)
        total = len(files)
        paginated_files = files[offset:offset + limit]
        
        return JSONResponse({
            "success": True,
            "data": {
                "files": paginated_files,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        })
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取文件列表失败")


@router.get("/{file_id}")
async def get_file_info(
    file_id: str,
    file_service: FileUploadService = Depends(get_file_service)
):
    """获取文件信息"""
    file_info = await file_service.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件未找到")
    
    return JSONResponse({
        "success": True,
        "data": file_info
    })


@router.get("/{file_id}/download")
async def download_file(
    file_id: str,
    file_service: FileUploadService = Depends(get_file_service)
):
    """下载文件"""
    file_info = await file_service.get_file_info(file_id)
    if not file_info:
        raise HTTPException(status_code=404, detail="文件未找到")
    
    file_path = file_info["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 从文件路径中提取原始文件名
    filename = os.path.basename(file_path)
    if "_" in filename:
        original_filename = "_".join(filename.split("_")[1:])
    else:
        original_filename = filename
    
    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='application/octet-stream'
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    file_service: FileUploadService = Depends(get_file_service)
):
    """删除文件"""
    success = await file_service.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="文件未找到")
    
    return JSONResponse({
        "success": True,
        "message": "文件删除成功"
    })


@router.post("/cleanup")
async def cleanup_old_files(
    days: int = Query(7, ge=1, le=365),
    file_service: FileUploadService = Depends(get_file_service)
):
    """清理旧文件"""
    try:
        deleted_count = await file_service.cleanup_old_files(days)
        return JSONResponse({
            "success": True,
            "message": f"清理完成，删除了 {deleted_count} 个文件",
            "deleted_count": deleted_count
        })
    except Exception as e:
        logger.error(f"清理文件失败: {e}")
        raise HTTPException(status_code=500, detail="清理文件失败")


@router.get("/stats/summary")
async def get_file_stats(
    file_service: FileUploadService = Depends(get_file_service)
):
    """获取文件统计信息"""
    try:
        upload_path = file_service.upload_path
        
        total_files = 0
        total_size = 0
        file_types = {}
        
        for root, dirs, filenames in os.walk(upload_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                stat = os.stat(file_path)
                
                total_files += 1
                total_size += stat.st_size
                
                # 统计文件类型
                ext = os.path.splitext(filename)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return JSONResponse({
            "success": True,
            "data": {
                "total_files": total_files,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "upload_path": str(upload_path)
            }
        })
        
    except Exception as e:
        logger.error(f"获取文件统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取文件统计失败")