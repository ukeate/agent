"""
文件上传和管理服务
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Optional, BinaryIO, Dict, Any
import aiofiles
from src.ai.multimodal.types import MultimodalContent, ContentType
from src.ai.multimodal.validators import ContentValidator
from src.ai.multimodal.client import OpenAIMultimodalClient

from src.core.logging import get_logger
logger = get_logger(__name__)

class FileUploadService:
    """文件上传服务"""
    
    def __init__(
        self, 
        upload_path: str = "/tmp/uploads",
        openai_client: Optional[OpenAIMultimodalClient] = None,
        max_file_size: int = 100 * 1024 * 1024  # 100MB
    ):
        self.upload_path = Path(upload_path)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.openai_client = openai_client
        
    async def save_uploaded_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> MultimodalContent:
        """保存上传的文件"""
        # 生成唯一的文件ID
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # 清理文件名
        safe_filename = ContentValidator.sanitize_filename(filename)
        
        # 确定内容类型
        mime_type = content_type or mimetypes.guess_type(filename)[0]
        
        # 创建日期目录
        date_dir = utc_now().strftime("%Y/%m/%d")
        save_dir = self.upload_path / date_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        file_extension = Path(safe_filename).suffix
        final_filename = f"{file_hash}_{safe_filename}"
        file_path = save_dir / final_filename
        
        # 保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_data)
        
        logger.info(
            "文件保存成功",
            file_id=file_hash,
            filename=safe_filename,
            size=len(file_data),
            path=str(file_path)
        )
        
        # 检测内容类型
        detected_type = ContentValidator.detect_content_type(file_path)
        
        # 验证文件
        is_valid, error_msg, content_type_enum = ContentValidator.validate_file(
            str(file_path),
            detected_type
        )
        
        if not is_valid:
            # 删除无效文件
            os.remove(file_path)
            raise ValueError(f"文件验证失败: {error_msg}")
        
        # 对于支持的文件类型，上传到OpenAI
        openai_file_id = None
        if self.openai_client and content_type_enum in [ContentType.DOCUMENT, ContentType.IMAGE]:
            try:
                openai_file_id = await self._upload_to_openai(file_path, content_type_enum)
            except Exception as e:
                logger.warning(f"上传到OpenAI失败: {e}")
        
        return MultimodalContent(
            content_id=file_hash,
            content_type=content_type_enum,
            file_path=str(file_path),
            file_size=len(file_data),
            mime_type=mime_type,
            metadata={
                "original_filename": filename,
                "upload_timestamp": utc_now().isoformat(),
                "openai_file_id": openai_file_id,
                "file_extension": file_extension
            }
        )
    
    async def _upload_to_openai(
        self, 
        file_path: Path, 
        content_type: ContentType
    ) -> Optional[str]:
        """上传文件到OpenAI并返回文件ID"""
        try:
            # 确定用途
            purpose = "vision" if content_type == ContentType.IMAGE else "assistants"
            
            # 使用多模态客户端上传文件
            result = await self.openai_client.upload_file(
                file_path,
                purpose=purpose
            )
            
            file_id = result.get('id')
            logger.info(f"文件上传到OpenAI成功: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"上传文件到OpenAI失败: {e}")
            return None
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        # 在上传目录中查找文件
        for root, dirs, files in os.walk(self.upload_path):
            for file in files:
                if file_id in file:
                    file_path = Path(root) / file
                    return {
                        "file_id": file_id,
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                        "created_at": datetime.fromtimestamp(
                            file_path.stat().st_ctime
                        ).isoformat()
                    }
        return None
    
    async def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        # 在上传目录中查找并删除文件
        for root, dirs, files in os.walk(self.upload_path):
            for file in files:
                if file_id in file:
                    file_path = Path(root) / file
                    os.remove(file_path)
                    logger.info(f"文件删除成功: {file_id}")
                    return True
        
        logger.warning(f"文件未找到: {file_id}")
        return False
    
    async def cleanup_old_files(self, days: int = 7):
        """清理旧文件"""
        cutoff_time = utc_now().timestamp() - (days * 24 * 3600)
        deleted_count = 0
        
        for root, dirs, files in os.walk(self.upload_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat().st_ctime < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
        
        logger.info(f"清理了 {deleted_count} 个旧文件")
        return deleted_count
