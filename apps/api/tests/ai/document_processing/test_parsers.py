"""解析器模块测试"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.ai.document_processing.parsers.base_parser import BaseParser, ParsedDocument, ParsedElement
from src.ai.document_processing.parsers.text_parser import TextParser
from src.ai.document_processing.parsers.pdf_parser import PDFParser


class TestBaseParser:
    """基础解析器测试"""
    
    def test_can_parse(self):
        """测试文件类型检查"""
        
        class TestParser(BaseParser):
            SUPPORTED_EXTENSIONS = [".txt", ".md"]
            
            async def parse(self, file_path):
                pass
        
        parser = TestParser()
        
        assert parser.can_parse(Path("test.txt")) is True
        assert parser.can_parse(Path("test.md")) is True
        assert parser.can_parse(Path("test.pdf")) is False
        assert parser.can_parse(Path("TEST.TXT")) is True  # 大小写不敏感
    
    def test_generate_doc_id(self):
        """测试文档ID生成"""
        
        class TestParser(BaseParser):
            SUPPORTED_EXTENSIONS = [".txt"]
            
            async def parse(self, file_path):
                pass
        
        parser = TestParser()
        
        doc_id1 = parser.generate_doc_id(Path("/test/file.txt"))
        doc_id2 = parser.generate_doc_id(Path("/test/file.txt"))
        
        # ID应该不同（因为包含时间戳）
        assert doc_id1 != doc_id2
        assert doc_id1.startswith("doc_")
        assert doc_id2.startswith("doc_")
    
    def test_extract_metadata(self):
        """测试元数据提取"""
        
        class TestParser(BaseParser):
            SUPPORTED_EXTENSIONS = [".txt"]
            
            async def parse(self, file_path):
                pass
        
        parser = TestParser()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test content")
            temp_file = Path(f.name)
        
        try:
            metadata = parser.extract_metadata(temp_file)
            
            assert "file_name" in metadata
            assert "file_size" in metadata
            assert "created_at" in metadata
            assert "modified_at" in metadata
            assert "extension" in metadata
            
            assert metadata["file_name"] == temp_file.name
            assert metadata["extension"] == ".txt"
            assert metadata["file_size"] > 0
        
        finally:
            temp_file.unlink()


class TestTextParser:
    """文本解析器测试"""
    
    @pytest.fixture
    def text_parser(self):
        """创建文本解析器实例"""
        return TextParser(parse_markdown=True, extract_links=True)
    
    @pytest.mark.asyncio
    async def test_parse_plain_text(self, text_parser):
        """测试纯文本解析"""
        # 创建临时文本文件
        content = "This is a test document.\nWith multiple lines.\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_file = Path(f.name)
        
        try:
            result = await text_parser.parse(temp_file)
            
            assert isinstance(result, ParsedDocument)
            assert result.file_type == "text"
            assert len(result.elements) > 0
            
            # 检查文本内容
            text_elements = [e for e in result.elements if e.element_type == "text"]
            assert len(text_elements) > 0
            
            combined_text = " ".join(e.content for e in text_elements)
            assert "This is a test document" in combined_text
        
        finally:
            temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_parse_markdown(self, text_parser):
        """测试Markdown解析"""
        markdown_content = """# Main Title
        
## Subtitle

This is a paragraph with **bold** and *italic* text.

### Code Example

```python
def hello():
    print("Hello World!")
```

- List item 1
- List item 2

[Link to example](https://example.com)
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            temp_file = Path(f.name)
        
        try:
            result = await text_parser.parse(temp_file)
            
            assert result.file_type == "markdown"
            
            # 检查不同类型的元素
            element_types = [e.element_type for e in result.elements]
            
            assert "heading" in element_types
            assert "text" in element_types
            assert "code_block" in element_types
            assert "list" in element_types
            
            # 检查链接提取
            link_elements = [e for e in result.elements if e.element_type == "link"]
            if link_elements:  # 如果启用了链接提取
                assert any("https://example.com" in e.content for e in link_elements)
        
        finally:
            temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_parse_json(self, text_parser):
        """测试JSON文件解析"""
        json_content = '''
{
    "name": "Test Document",
    "version": "1.0",
    "data": {
        "items": [1, 2, 3],
        "description": "Sample JSON file"
    }
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            temp_file = Path(f.name)
        
        try:
            result = await text_parser.parse(temp_file)
            
            assert result.file_type == "json"
            assert len(result.elements) > 0
            
            # 应该包含格式化的JSON内容
            content = " ".join(e.content for e in result.elements)
            assert "Test Document" in content
        
        finally:
            temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_parse_csv(self, text_parser):
        """测试CSV文件解析"""
        csv_content = """Name,Age,City
John,30,New York
Jane,25,San Francisco
Bob,35,Chicago
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = Path(f.name)
        
        try:
            result = await text_parser.parse(temp_file)
            
            assert result.file_type == "csv"
            
            # 应该包含表格元素
            table_elements = [e for e in result.elements if e.element_type == "table"]
            assert len(table_elements) > 0
            
            # 检查表格内容
            table_content = table_elements[0].content
            assert "Name" in table_content
            assert "John" in table_content
        
        finally:
            temp_file.unlink()


@pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="Requires optional dependencies")
class TestPDFParser:
    """PDF解析器测试（需要PyMuPDF）"""
    
    @pytest.fixture
    def pdf_parser(self):
        """创建PDF解析器实例"""
        return PDFParser(enable_ocr=False, extract_images=False)
    
    def test_initialization(self, pdf_parser):
        """测试初始化"""
        assert pdf_parser.enable_ocr is False
        assert pdf_parser.extract_images is False
        assert ".pdf" in pdf_parser.SUPPORTED_EXTENSIONS
    
    @pytest.mark.asyncio
    async def test_parse_pdf_not_found(self, pdf_parser):
        """测试不存在的PDF文件"""
        non_existent_file = Path("/non/existent/file.pdf")
        
        with pytest.raises(FileNotFoundError):
            await pdf_parser.parse(non_existent_file)
    
    def test_can_parse_pdf(self, pdf_parser):
        """测试PDF文件类型检查"""
        assert pdf_parser.can_parse(Path("test.pdf")) is True
        assert pdf_parser.can_parse(Path("test.txt")) is False
        assert pdf_parser.can_parse(Path("TEST.PDF")) is True


class TestParserIntegration:
    """解析器集成测试"""
    
    @pytest.mark.asyncio
    async def test_multiple_file_types(self):
        """测试多种文件类型的解析"""
        test_files = []
        
        # 创建不同类型的测试文件
        file_contents = {
            ".txt": "Plain text content",
            ".md": "# Markdown Title\n\nMarkdown content",
            ".json": '{"key": "value"}',
            ".csv": "A,B,C\n1,2,3"
        }
        
        try:
            for ext, content in file_contents.items():
                with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                    f.write(content)
                    test_files.append(Path(f.name))
            
            # 创建解析器
            text_parser = TextParser()
            
            # 解析所有文件
            results = []
            for file_path in test_files:
                if text_parser.can_parse(file_path):
                    result = await text_parser.parse(file_path)
                    results.append(result)
            
            # 验证所有文件都被解析了
            assert len(results) == len(test_files)
            
            # 验证每个结果
            for result in results:
                assert isinstance(result, ParsedDocument)
                assert len(result.elements) > 0
        
        finally:
            # 清理临时文件
            for file_path in test_files:
                file_path.unlink()


@pytest.mark.parametrize("file_extension,expected_type", [
    (".txt", "text"),
    (".md", "markdown"), 
    (".json", "json"),
    (".csv", "csv"),
    (".log", "text"),
])
@pytest.mark.asyncio
async def test_file_type_detection(file_extension, expected_type):
    """参数化测试文件类型检测"""
    content = "Sample content for testing"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as f:
        f.write(content)
        temp_file = Path(f.name)
    
    try:
        parser = TextParser()
        result = await parser.parse(temp_file)
        
        assert result.file_type == expected_type
    
    finally:
        temp_file.unlink()