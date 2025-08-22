"""
响应压缩优化
"""

import gzip
import zlib
from typing import Any, Dict, Optional

import structlog
from fastapi import Response

from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class CompressionHandler:
    """压缩处理器"""
    
    def __init__(self):
        self.min_size = 1000  # 最小压缩大小（字节）
        self.compression_level = 6  # 压缩级别 (1-9)
        self.enabled = True
    
    def should_compress(self, content: bytes, content_type: str) -> bool:
        """判断是否应该压缩"""
        if not self.enabled:
            return False
        
        # 检查内容大小
        if len(content) < self.min_size:
            return False
        
        # 检查内容类型
        compressible_types = [
            "text/", "application/json", "application/xml",
            "application/javascript", "application/x-javascript"
        ]
        
        return any(content_type.startswith(t) for t in compressible_types)
    
    def compress_gzip(self, content: bytes) -> bytes:
        """使用gzip压缩"""
        return gzip.compress(content, compresslevel=self.compression_level)
    
    def compress_deflate(self, content: bytes) -> bytes:
        """使用deflate压缩"""
        return zlib.compress(content, self.compression_level)
    
    def compress_response(
        self,
        response: Response,
        accept_encoding: str = ""
    ) -> Response:
        """压缩响应"""
        if not hasattr(response, "body") or not response.body:
            return response
        
        content = response.body
        content_type = response.headers.get("content-type", "")
        
        if not self.should_compress(content, content_type):
            return response
        
        # 检查客户端支持的编码
        if "gzip" in accept_encoding:
            compressed = self.compress_gzip(content)
            response.body = compressed
            response.headers["content-encoding"] = "gzip"
            logger.debug(
                "Response compressed with gzip",
                original_size=len(content),
                compressed_size=len(compressed),
                ratio=f"{(1 - len(compressed)/len(content))*100:.1f}%"
            )
        elif "deflate" in accept_encoding:
            compressed = self.compress_deflate(content)
            response.body = compressed
            response.headers["content-encoding"] = "deflate"
            logger.debug(
                "Response compressed with deflate",
                original_size=len(content),
                compressed_size=len(compressed),
                ratio=f"{(1 - len(compressed)/len(content))*100:.1f}%"
            )
        
        # 更新Content-Length
        if "content-length" in response.headers:
            response.headers["content-length"] = str(len(response.body))
        
        return response
    
    def decompress(self, content: bytes, encoding: str) -> bytes:
        """解压缩内容"""
        if encoding == "gzip":
            return gzip.decompress(content)
        elif encoding == "deflate":
            return zlib.decompress(content)
        else:
            return content
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return {
            "enabled": self.enabled,
            "min_size": self.min_size,
            "compression_level": self.compression_level
        }


# 全局压缩处理器实例
compression_handler = CompressionHandler()