"""Word文档解析器"""

from pathlib import Path
from typing import List, Dict, Any
import base64
import io
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image
from .base_parser import BaseParser, ParsedDocument, ParsedElement

logger = get_logger(__name__)

class WordParser(BaseParser):
    """Word文档解析器
    
    支持.docx格式的文本、表格、图像提取
    """
    
    SUPPORTED_EXTENSIONS = [".docx", ".doc"]
    
    def __init__(self, extract_images: bool = True, extract_styles: bool = True):
        """初始化Word解析器
        
        Args:
            extract_images: 是否提取图像
            extract_styles: 是否提取样式信息
        """
        super().__init__()
        self.extract_images = extract_images
        self.extract_styles = extract_styles
    
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析Word文档
        
        Args:
            file_path: Word文件路径
            
        Returns:
            解析后的文档
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 检查文件格式
        if file_path.suffix.lower() == ".doc":
            logger.warning("Old .doc format detected. Consider converting to .docx for better support")
        
        doc_id = self.generate_doc_id(file_path)
        metadata = self.extract_metadata(file_path)
        elements = []
        
        try:
            # 打开Word文档
            doc = Document(str(file_path))
            
            # 提取文档属性
            core_props = doc.core_properties
            metadata.update({
                "title": core_props.title or "",
                "author": core_props.author or "",
                "subject": core_props.subject or "",
                "keywords": core_props.keywords or "",
                "created": core_props.created.isoformat() if core_props.created else "",
                "modified": core_props.modified.isoformat() if core_props.modified else "",
            })
            
            # 解析文档内容
            for element in doc.element.body:
                if element.tag.endswith('p'):  # 段落
                    para = Paragraph(element, doc)
                    if para.text.strip():
                        elements.append(await self._parse_paragraph(para))
                
                elif element.tag.endswith('tbl'):  # 表格
                    table = Table(element, doc)
                    elements.append(await self._parse_table(table))
            
            # 提取图像
            if self.extract_images:
                images = await self._extract_images(doc)
                elements.extend(images)
            
        except Exception as e:
            logger.error(f"Error parsing Word document: {e}")
            raise
        
        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(file_path),
            file_type="word",
            elements=elements,
            metadata=metadata
        )
    
    async def _parse_paragraph(self, paragraph: Paragraph) -> ParsedElement:
        """解析段落
        
        Args:
            paragraph: Word段落对象
            
        Returns:
            解析的元素
        """
        metadata = {
            "alignment": str(paragraph.alignment) if paragraph.alignment else "left",
        }
        
        # 提取样式信息
        if self.extract_styles and paragraph.style:
            metadata.update({
                "style_name": paragraph.style.name,
                "style_type": str(paragraph.style.type),
            })
            
            # 检查是否为标题
            if paragraph.style.name.startswith('Heading'):
                element_type = "heading"
                try:
                    level = int(paragraph.style.name.replace('Heading ', ''))
                    metadata["heading_level"] = level
                except:
                    metadata["heading_level"] = 1
            else:
                element_type = "text"
        else:
            element_type = "text"
        
        # 提取文本格式信息
        if paragraph.runs:
            formats = []
            for run in paragraph.runs:
                if run.bold:
                    formats.append("bold")
                if run.italic:
                    formats.append("italic")
                if run.underline:
                    formats.append("underline")
            
            if formats:
                metadata["formats"] = list(set(formats))
        
        return ParsedElement(
            content=paragraph.text,
            element_type=element_type,
            metadata=metadata
        )
    
    async def _parse_table(self, table: Table) -> ParsedElement:
        """解析表格
        
        Args:
            table: Word表格对象
            
        Returns:
            解析的元素
        """
        # 提取表格数据
        table_data = []
        headers = []
        
        for row_idx, row in enumerate(table.rows):
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                row_data.append(cell_text)
            
            if row_idx == 0:
                headers = row_data
            else:
                table_data.append(row_data)
        
        # 将表格转换为文本格式
        content_lines = []
        
        # 添加表头
        if headers:
            content_lines.append(" | ".join(headers))
            content_lines.append("-" * (len(" | ".join(headers))))
        
        # 添加数据行
        for row in table_data:
            content_lines.append(" | ".join(row))
        
        content = "\n".join(content_lines)
        
        return ParsedElement(
            content=content,
            element_type="table",
            metadata={
                "headers": headers,
                "rows": table_data,
                "num_rows": len(table.rows),
                "num_columns": len(table.columns),
            }
        )
    
    async def _extract_images(self, doc: Document) -> List[ParsedElement]:
        """提取文档中的图像
        
        Args:
            doc: Word文档对象
            
        Returns:
            图像元素列表
        """
        elements = []
        
        try:
            # 访问文档的关系
            for rel in doc.part.rels.values():
                if "image" in rel.reltype:
                    # 获取图像数据
                    image_part = rel.target_part
                    image_data = image_part.blob
                    
                    # 获取图像信息
                    img = Image.open(io.BytesIO(image_data))
                    
                    # 转换为base64
                    img_base64 = base64.b64encode(image_data).decode()
                    
                    elements.append(ParsedElement(
                        content=img_base64,
                        element_type="image",
                        metadata={
                            "width": img.width,
                            "height": img.height,
                            "format": img.format,
                            "mode": img.mode,
                        }
                    ))
                    
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
        
        return elements
from src.core.logging import get_logger
