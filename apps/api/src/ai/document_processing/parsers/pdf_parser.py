"""PDF文档解析器"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from .base_parser import BaseParser, ParsedDocument, ParsedElement

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class PDFParser(BaseParser):
    """PDF文档解析器
    
    支持文本提取、图像提取、表格识别和OCR
    """
    
    SUPPORTED_EXTENSIONS = [".pdf"]
    
    def __init__(self, enable_ocr: bool = True, extract_images: bool = True):
        """初始化PDF解析器
        
        Args:
            enable_ocr: 是否启用OCR识别
            extract_images: 是否提取图像
        """
        super().__init__()
        self.enable_ocr = enable_ocr
        self.extract_images = extract_images
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available, falling back to basic PDF parsing")
    
    async def parse(self, file_path: Path) -> ParsedDocument:
        """解析PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            解析后的文档
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc_id = self.generate_doc_id(file_path)
        metadata = self.extract_metadata(file_path)
        elements = []
        
        if PYMUPDF_AVAILABLE:
            elements = await self._parse_with_pymupdf(file_path)
        else:
            # 降级使用PyPDF2
            elements = await self._parse_with_pypdf2(file_path)
        
        # 添加PDF特定元数据
        metadata.update({
            "parser": "PyMuPDF" if PYMUPDF_AVAILABLE else "PyPDF2",
            "ocr_enabled": self.enable_ocr,
            "images_extracted": self.extract_images,
        })
        
        return ParsedDocument(
            doc_id=doc_id,
            file_path=str(file_path),
            file_type="pdf",
            elements=elements,
            metadata=metadata
        )
    
    async def _parse_with_pymupdf(self, file_path: Path) -> List[ParsedElement]:
        """使用PyMuPDF解析PDF
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            解析的元素列表
        """
        elements = []
        
        try:
            # 打开PDF文档
            pdf_document = fitz.open(str(file_path))
            
            for page_num, page in enumerate(pdf_document):
                # 提取文本
                text = page.get_text()
                if text.strip():
                    elements.append(ParsedElement(
                        content=text,
                        element_type="text",
                        metadata={
                            "page_number": page_num + 1,
                            "source": "text_extraction"
                        }
                    ))
                
                # 提取表格
                tables = self._extract_tables_from_page(page)
                for idx, table in enumerate(tables):
                    elements.append(ParsedElement(
                        content=table,
                        element_type="table",
                        metadata={
                            "page_number": page_num + 1,
                            "table_index": idx,
                            "source": "table_extraction"
                        }
                    ))
                
                # 提取图像
                if self.extract_images:
                    images = await self._extract_images_from_page(page, page_num)
                    elements.extend(images)
                
                # 如果页面没有文本且启用OCR，尝试OCR识别
                if not text.strip() and self.enable_ocr:
                    ocr_text = await self._ocr_page(page)
                    if ocr_text:
                        elements.append(ParsedElement(
                            content=ocr_text,
                            element_type="text",
                            metadata={
                                "page_number": page_num + 1,
                                "source": "ocr"
                            }
                        ))
            
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Error parsing PDF with PyMuPDF: {e}")
            raise
        
        return elements
    
    async def _parse_with_pypdf2(self, file_path: Path) -> List[ParsedElement]:
        """使用PyPDF2解析PDF（降级方案）
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            解析的元素列表
        """
        import PyPDF2
        
        elements = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    
                    # 提取文本
                    text = page.extract_text()
                    if text.strip():
                        elements.append(ParsedElement(
                            content=text,
                            element_type="text",
                            metadata={
                                "page_number": page_num + 1,
                                "source": "text_extraction"
                            }
                        ))
        
        except Exception as e:
            logger.error(f"Error parsing PDF with PyPDF2: {e}")
            raise
        
        return elements
    
    def _extract_tables_from_page(self, page) -> List[str]:
        """从页面提取表格
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            表格内容列表
        """
        tables = []
        
        try:
            # 尝试识别表格结构
            # 这里使用简单的文本分析，实际可以使用更复杂的表格检测算法
            text = page.get_text("blocks")
            
            for block in text:
                if len(block) >= 5:  # 块包含文本
                    block_text = block[4]
                    # 简单的表格检测：查找包含多个制表符或管道符的行
                    lines = block_text.split('\n')
                    table_lines = []
                    
                    for line in lines:
                        if '\t' in line or '|' in line:
                            table_lines.append(line)
                    
                    if len(table_lines) > 2:  # 至少3行才认为是表格
                        tables.append('\n'.join(table_lines))
        
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    async def _extract_images_from_page(
        self, 
        page, 
        page_num: int
    ) -> List[ParsedElement]:
        """从页面提取图像
        
        Args:
            page: PyMuPDF页面对象
            page_num: 页面编号
            
        Returns:
            图像元素列表
        """
        elements = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # 提取图像数据
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                # 转换为base64
                img_base64 = base64.b64encode(img_data).decode()
                
                # OCR识别图像中的文本
                ocr_text = ""
                if self.enable_ocr and PIL_AVAILABLE and TESSERACT_AVAILABLE:
                    try:
                        img_pil = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img_pil)
                    except Exception as e:
                        logger.warning(f"OCR failed for image: {e}")
                elif self.enable_ocr:
                    logger.warning("OCR requested but PIL or Tesseract not available")
                
                elements.append(ParsedElement(
                    content=img_base64,
                    element_type="image",
                    metadata={
                        "page_number": page_num + 1,
                        "image_index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "ocr_text": ocr_text,
                        "format": "png"
                    }
                ))
                
                pix = None
        
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
        
        return elements
    
    async def _ocr_page(self, page) -> str:
        """对页面进行OCR识别
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            OCR识别的文本
        """
        if not (PIL_AVAILABLE and TESSERACT_AVAILABLE):
            logger.warning("OCR requested but PIL or Tesseract not available")
            return ""
            
        try:
            # 将页面转换为图像
            mat = fitz.Matrix(2, 2)  # 缩放因子
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # 使用PIL和Tesseract进行OCR
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            
            return text
        
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
from src.core.logging import get_logger
