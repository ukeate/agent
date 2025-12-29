"""基础文档解析器接口"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

class ParsedElement:
    """解析的文档元素"""
    
    def __init__(
        self,
        content: str,
        element_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.element_type = element_type
        self.metadata = metadata or {}

class ParsedDocument:
    """解析后的文档结构"""
    
    def __init__(
        self,
        doc_id: str,
        file_path: str,
        file_type: str,
        elements: List[ParsedElement],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_type = file_type
        self.elements = elements
        self.metadata = metadata or {}
        self.parsed_at = utc_now()

class BaseParser(ABC):
    """文档解析器基类"""
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def can_parse(self, file_path: Path) -> bool:
        """检查是否可以解析该文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否可以解析
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    @abstractmethod
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的文档
        """
        raise NotImplementedError
    
    def generate_doc_id(self, file_path: Path) -> str:
        """生成文档ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档ID
        """
        # 基于文件路径和时间戳生成唯一ID
        content = f"{file_path.absolute()}:{utc_now().isoformat()}"
        hash_obj = hashlib.sha256(content.encode())
        return f"doc_{hash_obj.hexdigest()[:16]}"
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """提取文件元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            元数据字典
        """
        stat = file_path.stat()
        return {
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower(),
        }
