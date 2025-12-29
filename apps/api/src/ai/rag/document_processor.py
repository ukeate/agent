"""多模态文档处理管道"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4
import base64
import io
from unstructured.partition.auto import partition
from unstructured.documents.elements import (
    Title, Text, Image, Table, ListItem, NarrativeText
)
from PIL import Image as PILImage
import pytesseract
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .multimodal_config import ProcessedDocument, MultimodalConfig

logger = get_logger(__name__)

# Document processing

# LangChain

class MultimodalDocumentProcessor:
    """多模态文档处理管道"""
    
    SUPPORTED_FORMATS = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".html",
        ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"
    }
    
    def __init__(self, config: MultimodalConfig):
        """初始化文档处理器
        
        Args:
            config: 多模态配置
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    async def process_document(self, file_path: str) -> ProcessedDocument:
        """处理文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            处理后的文档
        """
        file_path = Path(file_path)
        
        # 验证文件
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 生成文档ID
        doc_id = self._generate_doc_id(str(file_path))
        
        # 使用Unstructured解析文档
        try:
            elements = partition(str(file_path))
        except Exception as e:
            logger.error(f"Error parsing document with Unstructured: {e}")
            # 降级处理
            elements = await self._fallback_parsing(file_path)
        
        # 分类内容类型
        texts, tables, images = self._categorize_elements(elements)
        
        # 处理各种内容类型
        processed_texts = await self._process_texts(texts)
        processed_tables = await self._process_tables(tables)
        processed_images = await self._process_images(images, file_path)
        
        # 生成摘要和元数据
        processed_doc = await self._enrich_with_metadata(
            doc_id=doc_id,
            source_file=str(file_path),
            texts=processed_texts,
            tables=processed_tables,
            images=processed_images
        )
        
        return processed_doc
    
    def _generate_doc_id(self, file_path: str) -> str:
        """生成文档ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档ID
        """
        # 基于文件路径和内容的哈希生成ID
        hash_obj = hashlib.md5(file_path.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"
    
    async def _fallback_parsing(self, file_path: Path) -> List[Any]:
        """降级解析方法
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析的元素列表
        """
        elements = []
        
        # 文本文件
        if file_path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                elements.append(Text(content))
        
        # 图像文件
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            elements.append(Image(str(file_path)))
        
        # CSV文件
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            elements.append(Table(df.to_dict()))
        
        return elements
    
    def _categorize_elements(self, elements: List[Any]) -> Tuple[List, List, List]:
        """分类文档元素
        
        Args:
            elements: 文档元素列表
            
        Returns:
            文本、表格、图像元素的元组
        """
        texts = []
        tables = []
        images = []
        
        for element in elements:
            if isinstance(element, (Title, Text, NarrativeText, ListItem)):
                texts.append(element)
            elif isinstance(element, Table):
                tables.append(element)
            elif isinstance(element, Image):
                images.append(element)
            else:
                # 其他类型作为文本处理
                if hasattr(element, "text"):
                    texts.append(element)
        
        return texts, tables, images
    
    async def _process_texts(self, texts: List[Any]) -> List[Dict[str, Any]]:
        """处理文本元素
        
        Args:
            texts: 文本元素列表
            
        Returns:
            处理后的文本块列表
        """
        processed_texts = []
        
        # 合并文本
        combined_text = "\n\n".join([
            str(text) if not hasattr(text, "text") else text.text
            for text in texts
        ])
        
        if not combined_text:
            return processed_texts
        
        # 分块
        chunks = self.text_splitter.split_text(combined_text)
        
        for idx, chunk in enumerate(chunks):
            processed_texts.append({
                "content": chunk,
                "metadata": {
                    "chunk_index": idx,
                    "chunk_size": len(chunk),
                    "type": "text"
                }
            })
        
        return processed_texts
    
    async def _process_tables(self, tables: List[Any]) -> List[Dict[str, Any]]:
        """处理表格元素
        
        Args:
            tables: 表格元素列表
            
        Returns:
            处理后的表格列表
        """
        processed_tables = []
        
        for idx, table in enumerate(tables):
            # 提取表格数据
            if hasattr(table, "metadata"):
                table_data = table.metadata
            else:
                table_data = str(table)
            
            # 尝试解析为结构化数据
            try:
                if isinstance(table_data, dict):
                    headers = list(table_data.keys())
                    rows = list(zip(*table_data.values()))
                elif isinstance(table_data, str):
                    # 简单的表格解析
                    lines = table_data.strip().split("\n")
                    if lines:
                        headers = lines[0].split("\t")
                        rows = [line.split("\t") for line in lines[1:]]
                    else:
                        headers = []
                        rows = []
                else:
                    headers = []
                    rows = []
                
                processed_tables.append({
                    "headers": headers,
                    "rows": rows,
                    "metadata": {
                        "table_index": idx,
                        "num_rows": len(rows),
                        "num_columns": len(headers),
                        "type": "table"
                    }
                })
            except Exception as e:
                logger.warning(f"Error processing table {idx}: {e}")
                processed_tables.append({
                    "data": str(table),
                    "metadata": {
                        "table_index": idx,
                        "type": "table",
                        "error": str(e)
                    }
                })
        
        return processed_tables
    
    async def _process_images(
        self,
        images: List[Any],
        source_path: Path
    ) -> List[Dict[str, Any]]:
        """处理图像元素
        
        Args:
            images: 图像元素列表
            source_path: 源文件路径
            
        Returns:
            处理后的图像列表
        """
        processed_images = []
        
        for idx, image in enumerate(images):
            try:
                image_info = {}
                
                # 获取图像路径
                if hasattr(image, "metadata") and "filename" in image.metadata:
                    image_path = image.metadata["filename"]
                elif hasattr(image, "text"):
                    image_path = image.text
                else:
                    image_path = str(image)
                
                # 处理相对路径
                if not os.path.isabs(image_path):
                    image_path = source_path.parent / image_path
                
                # 读取图像
                if os.path.exists(image_path):
                    with PILImage.open(image_path) as img:
                        # 调整大小
                        if max(img.size) > self.config.max_image_size:
                            img.thumbnail(
                                (self.config.max_image_size, self.config.max_image_size),
                                PILImage.Resampling.LANCZOS
                            )
                        
                        # OCR提取文本
                        try:
                            ocr_text = pytesseract.image_to_string(img)
                            image_info["ocr_text"] = ocr_text
                        except Exception as e:
                            logger.warning(f"OCR failed for image {idx}: {e}")
                            image_info["ocr_text"] = ""
                        
                        # 转换为base64
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        image_info.update({
                            "path": str(image_path),
                            "base64": image_base64,
                            "width": img.width,
                            "height": img.height,
                            "format": img.format,
                            "description": image_info.get("ocr_text", "")[:200],
                            "metadata": {
                                "image_index": idx,
                                "type": "image",
                                "size": (img.width, img.height)
                            }
                        })
                else:
                    logger.warning(f"Image file not found: {image_path}")
                    image_info = {
                        "path": str(image_path),
                        "description": f"Image not found: {image_path}",
                        "metadata": {
                            "image_index": idx,
                            "type": "image",
                            "error": "File not found"
                        }
                    }
                
                processed_images.append(image_info)
                
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                processed_images.append({
                    "description": f"Error processing image: {e}",
                    "metadata": {
                        "image_index": idx,
                        "type": "image",
                        "error": str(e)
                    }
                })
        
        return processed_images
    
    async def _enrich_with_metadata(
        self,
        doc_id: str,
        source_file: str,
        texts: List[Dict],
        tables: List[Dict],
        images: List[Dict]
    ) -> ProcessedDocument:
        """使用元数据丰富文档
        
        Args:
            doc_id: 文档ID
            source_file: 源文件路径
            texts: 文本块列表
            tables: 表格列表
            images: 图像列表
            
        Returns:
            处理后的文档
        """
        # 生成摘要
        summary = await self._generate_summary(texts)
        
        # 提取关键词
        keywords = await self._extract_keywords(texts, summary)
        
        # 确定内容类型
        content_type = self._determine_content_type(source_file)
        
        # 创建处理后的文档
        processed_doc = ProcessedDocument(
            doc_id=doc_id,
            source_file=source_file,
            content_type=content_type,
            texts=texts,
            tables=tables,
            images=images,
            summary=summary,
            keywords=keywords,
            metadata={
                "num_text_chunks": len(texts),
                "num_tables": len(tables),
                "num_images": len(images),
                "file_size": os.path.getsize(source_file) if os.path.exists(source_file) else 0,
                "file_extension": Path(source_file).suffix
            }
        )
        
        return processed_doc
    
    async def _generate_summary(self, texts: List[Dict]) -> str:
        """生成文档摘要
        
        Args:
            texts: 文本块列表
            
        Returns:
            文档摘要
        """
        if not texts:
            return ""
        
        # 取前几个文本块生成摘要
        sample_texts = texts[:3]
        combined = " ".join([t["content"] for t in sample_texts])
        
        # 简单的摘要生成(实际应该使用LLM)
        max_length = 500
        if len(combined) > max_length:
            summary = combined[:max_length] + "..."
        else:
            summary = combined
        
        return summary
    
    async def _extract_keywords(
        self,
        texts: List[Dict],
        summary: str
    ) -> List[str]:
        """提取关键词
        
        Args:
            texts: 文本块列表
            summary: 文档摘要
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取(实际应该使用NLP技术)
        import re
        from collections import Counter
        
        # 合并文本
        all_text = summary + " " + " ".join([t["content"] for t in texts[:5]])
        
        # 提取单词
        words = re.findall(r'\b[a-z]+\b', all_text.lower())
        
        # 过滤停用词
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about",
            "into", "through", "during", "before", "after", "is",
            "are", "was", "were", "been", "be", "have", "has", "had"
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # 获取最常见的关键词
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(10)]
        
        return keywords
    
    def _determine_content_type(self, file_path: str) -> str:
        """确定内容类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            内容类型
        """
        ext = Path(file_path).suffix.lower()
        
        if ext in [".pdf"]:
            return "pdf"
        elif ext in [".docx", ".doc"]:
            return "word"
        elif ext in [".xlsx", ".xls", ".csv"]:
            return "spreadsheet"
        elif ext in [".png", ".jpg", ".jpeg"]:
            return "image"
        elif ext in [".txt", ".md"]:
            return "text"
        elif ext in [".html", ".htm"]:
            return "html"
        else:
            return "unknown"
    
    async def process_batch(
        self,
        file_paths: List[str]
    ) -> List[ProcessedDocument]:
        """批量处理文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            处理后的文档列表
        """
        import asyncio
        
        # 并发处理文档
        tasks = [self.process_document(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_docs = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {file_paths[idx]}: {result}")
            else:
                processed_docs.append(result)
        
        return processed_docs
from src.core.logging import get_logger
