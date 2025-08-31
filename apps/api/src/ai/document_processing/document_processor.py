"""主文档处理器"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import hashlib
import json

from .parsers import (
    BaseParser,
    PDFParser,
    WordParser,
    ExcelParser,
    PowerPointParser,
    CodeParser,
    TextParser
)
from .parsers.base_parser import ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)


class ProcessedDocument:
    """处理后的文档完整结构"""
    
    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        file_type: str,
        source: Dict[str, Any],
        processing_info: Dict[str, Any],
        relationships: Optional[Dict[str, Any]] = None,
        versions: Optional[List[Dict[str, Any]]] = None,
        embedding_vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.file_type = file_type
        self.source = source
        self.processing_info = processing_info
        self.relationships = relationships or {}
        self.versions = versions or []
        self.embedding_vector = embedding_vector
        self.metadata = metadata or {}
        self.created_at = created_at or utc_now()
        self.updated_at = updated_at or utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "file_type": self.file_type,
            "source": self.source,
            "processing_info": self.processing_info,
            "relationships": self.relationships,
            "versions": self.versions,
            "embedding_vector": self.embedding_vector,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentProcessor:
    """智能文档处理器
    
    协调多个解析器处理不同格式的文档
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        extract_images: bool = True,
        extract_structure: bool = True,
        max_file_size: int = 50 * 1024 * 1024,  # 50MB
        allowed_extensions: Optional[List[str]] = None
    ):
        """初始化文档处理器
        
        Args:
            enable_ocr: 是否启用OCR
            extract_images: 是否提取图像
            extract_structure: 是否提取文档结构
            max_file_size: 最大文件大小限制（字节）
            allowed_extensions: 允许的文件扩展名列表
        """
        self.enable_ocr = enable_ocr
        self.extract_images = extract_images
        self.extract_structure = extract_structure
        self.max_file_size = max_file_size
        
        # 初始化解析器
        self.parsers: List[BaseParser] = [
            PDFParser(enable_ocr=enable_ocr, extract_images=extract_images),
            WordParser(extract_images=extract_images, extract_styles=extract_structure),
            ExcelParser(extract_formulas=True, extract_charts=True),
            PowerPointParser(extract_images=extract_images, extract_notes=True),
            CodeParser(extract_structure=extract_structure, extract_docstrings=True),
            TextParser(parse_markdown=True, extract_links=True),
        ]
        
        # 构建支持的扩展名集合
        self.supported_extensions = set()
        for parser in self.parsers:
            self.supported_extensions.update(parser.SUPPORTED_EXTENSIONS)
        
        # 如果指定了允许的扩展名，取交集
        if allowed_extensions:
            self.supported_extensions = self.supported_extensions.intersection(
                set(allowed_extensions)
            )
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        extract_metadata: bool = True,
        generate_embeddings: bool = False
    ) -> ProcessedDocument:
        """处理单个文档
        
        Args:
            file_path: 文档文件路径
            extract_metadata: 是否提取元数据
            generate_embeddings: 是否生成向量嵌入
            
        Returns:
            处理后的文档
        """
        file_path = Path(file_path)
        
        # 验证文件
        await self._validate_file(file_path)
        
        # 选择合适的解析器
        parser = self._select_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for file type: {file_path.suffix}")
        
        # 解析文档
        logger.info(f"Processing document: {file_path}")
        parsed_doc = await parser.parse(file_path)
        
        # 转换为ProcessedDocument格式
        processed_doc = await self._convert_to_processed_document(
            parsed_doc,
            file_path,
            extract_metadata=extract_metadata,
            generate_embeddings=generate_embeddings
        )
        
        return processed_doc
    
    async def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        concurrent_limit: int = 5,
        continue_on_error: bool = True
    ) -> List[ProcessedDocument]:
        """批量处理文档
        
        Args:
            file_paths: 文档文件路径列表
            concurrent_limit: 并发处理限制
            continue_on_error: 遇到错误是否继续
            
        Returns:
            处理后的文档列表
        """
        processed_docs = []
        errors = []
        
        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                try:
                    doc = await self.process_document(file_path)
                    return doc, None
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    if not continue_on_error:
                        raise
                    return None, {"file": str(file_path), "error": str(e)}
        
        # 并发处理所有文档
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # 收集结果
        for doc, error in results:
            if doc:
                processed_docs.append(doc)
            if error:
                errors.append(error)
        
        # 记录处理结果
        logger.info(
            f"Batch processing completed: "
            f"{len(processed_docs)} success, {len(errors)} failed"
        )
        
        if errors:
            logger.error(f"Failed documents: {json.dumps(errors, indent=2)}")
        
        return processed_docs
    
    async def _validate_file(self, file_path: Path):
        """验证文件
        
        Args:
            file_path: 文件路径
        """
        # 检查文件存在
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 检查文件大小
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(
                f"File too large: {file_size} bytes "
                f"(max: {self.max_file_size} bytes)"
            )
        
        # 检查文件扩展名
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {sorted(self.supported_extensions)}"
            )
    
    def _select_parser(self, file_path: Path) -> Optional[BaseParser]:
        """选择合适的解析器
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析器实例或None
        """
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    async def _convert_to_processed_document(
        self,
        parsed_doc: ParsedDocument,
        file_path: Path,
        extract_metadata: bool = True,
        generate_embeddings: bool = False
    ) -> ProcessedDocument:
        """转换解析结果为ProcessedDocument格式
        
        Args:
            parsed_doc: 解析后的文档
            file_path: 文件路径
            extract_metadata: 是否提取元数据
            generate_embeddings: 是否生成向量嵌入
            
        Returns:
            处理后的文档
        """
        # 提取标题
        title = self._extract_title(parsed_doc, file_path)
        
        # 合并所有文本内容
        content = self._merge_content(parsed_doc)
        
        # 构建源信息
        source = {
            "type": "upload",
            "original_path": str(file_path.absolute()),
            "file_size": file_path.stat().st_size,
            "mime_type": self._get_mime_type(file_path),
        }
        
        # 构建处理信息
        processing_info = await self._build_processing_info(parsed_doc)
        
        # 提取元数据
        metadata = parsed_doc.metadata.copy() if extract_metadata else {}
        metadata.update({
            "parser_used": parsed_doc.file_type,
            "elements_count": len(parsed_doc.elements),
            "processed_at": utc_now().isoformat(),
        })
        
        # 生成向量嵌入（如果需要）
        embedding_vector = None
        if generate_embeddings:
            # 这里应该调用嵌入服务生成向量
            # embedding_vector = await self._generate_embeddings(content)
            pass
        
        return ProcessedDocument(
            doc_id=parsed_doc.doc_id,
            title=title,
            content=content,
            file_type=parsed_doc.file_type,
            source=source,
            processing_info=processing_info,
            embedding_vector=embedding_vector,
            metadata=metadata
        )
    
    def _extract_title(self, parsed_doc: ParsedDocument, file_path: Path) -> str:
        """提取文档标题
        
        Args:
            parsed_doc: 解析后的文档
            file_path: 文件路径
            
        Returns:
            文档标题
        """
        # 尝试从元数据提取标题
        if "title" in parsed_doc.metadata:
            return parsed_doc.metadata["title"]
        
        # 尝试从第一个标题元素提取
        for element in parsed_doc.elements:
            if element.element_type == "heading":
                return element.content[:200]  # 限制标题长度
        
        # 使用文件名作为标题
        return file_path.stem
    
    def _merge_content(self, parsed_doc: ParsedDocument) -> str:
        """合并所有文本内容
        
        Args:
            parsed_doc: 解析后的文档
            
        Returns:
            合并的文本内容
        """
        text_parts = []
        
        for element in parsed_doc.elements:
            # 跳过base64编码的图像内容
            if element.element_type == "image":
                # 使用图像描述或OCR文本
                if "ocr_text" in element.metadata:
                    text_parts.append(element.metadata["ocr_text"])
                elif "alt_text" in element.metadata:
                    text_parts.append(element.metadata["alt_text"])
            else:
                text_parts.append(element.content)
        
        return "\n\n".join(text_parts)
    
    async def _build_processing_info(
        self,
        parsed_doc: ParsedDocument
    ) -> Dict[str, Any]:
        """构建处理信息
        
        Args:
            parsed_doc: 解析后的文档
            
        Returns:
            处理信息字典
        """
        # 统计各类型元素
        element_types = {}
        for element in parsed_doc.elements:
            element_types[element.element_type] = \
                element_types.get(element.element_type, 0) + 1
        
        # 构建文档块信息（用于后续分块）
        chunks = []
        for idx, element in enumerate(parsed_doc.elements):
            if element.element_type in ["text", "heading", "code"]:
                chunks.append({
                    "chunk_id": f"chunk_{idx}",
                    "content": element.content[:500],  # 限制存储长度
                    "type": element.element_type,
                    "metadata": element.metadata,
                })
        
        # 提取关键实体（简化版）
        key_entities = []
        # 这里应该使用NLP技术提取实体
        
        # 文档结构信息
        structure = {
            "element_types": element_types,
            "total_elements": len(parsed_doc.elements),
            "has_images": "image" in element_types,
            "has_tables": "table" in element_types,
            "has_code": "code" in element_types or "code_block" in element_types,
        }
        
        return {
            "chunks": chunks[:100],  # 限制存储的块数
            "auto_tags": [],  # 将由auto_tagger填充
            "classification": [],  # 将由分类系统填充
            "summary": "",  # 将由摘要系统填充
            "key_entities": key_entities,
            "structure": structure,
        }
    
    def _get_mime_type(self, file_path: Path) -> str:
        """获取文件MIME类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            MIME类型
        """
        import mimetypes
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def _calculate_hash(self, content: str) -> str:
        """计算内容哈希值
        
        Args:
            content: 内容字符串
            
        Returns:
            SHA256哈希值
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表
        
        Returns:
            支持的扩展名列表
        """
        return sorted(list(self.supported_extensions))