"""文档处理器测试"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.ai.document_processing.document_processor import DocumentProcessor, ProcessedDocument
from src.ai.document_processing.parsers.base_parser import ParsedDocument, ParsedElement


@pytest.fixture
def document_processor():
    """创建文档处理器实例"""
    return DocumentProcessor(
        enable_ocr=False,  # 测试时禁用OCR
        extract_images=False,
        max_file_size=10 * 1024 * 1024  # 10MB
    )


@pytest.fixture
def sample_parsed_doc():
    """示例解析文档"""
    elements = [
        ParsedElement(
            content="This is a test document",
            element_type="text",
            metadata={"page": 1}
        ),
        ParsedElement(
            content="# Main Title",
            element_type="heading",
            metadata={"level": 1}
        ),
        ParsedElement(
            content="Some code example",
            element_type="code",
            metadata={"language": "python"}
        )
    ]
    
    return ParsedDocument(
        doc_id="test_doc_123",
        file_path="/test/sample.txt",
        file_type="text",
        elements=elements,
        metadata={"title": "Test Document", "author": "Test User"}
    )


@pytest.mark.asyncio
async def test_process_document_success(document_processor):
    """测试成功处理文档"""
    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document content.")
        temp_file_path = Path(f.name)
    
    try:
        # Mock解析器
        mock_parser = Mock()
        mock_parsed_doc = ParsedDocument(
            doc_id="test_doc",
            file_path=str(temp_file_path),
            file_type="text",
            elements=[
                ParsedElement("This is a test document content.", "text", {})
            ],
            metadata={"title": "Test"}
        )
        mock_parser.parse = AsyncMock(return_value=mock_parsed_doc)
        mock_parser.can_parse = Mock(return_value=True)
        
        # 替换解析器
        document_processor.parsers = [mock_parser]
        
        # 处理文档
        result = await document_processor.process_document(temp_file_path)
        
        # 验证结果
        assert isinstance(result, ProcessedDocument)
        assert result.title == "Test"
        assert result.file_type == "text"
        assert "This is a test document content." in result.content
        
    finally:
        # 清理临时文件
        temp_file_path.unlink()


@pytest.mark.asyncio
async def test_process_document_file_not_found(document_processor):
    """测试文件不存在的情况"""
    non_existent_file = Path("/non/existent/file.txt")
    
    with pytest.raises(FileNotFoundError):
        await document_processor.process_document(non_existent_file)


@pytest.mark.asyncio
async def test_process_document_file_too_large(document_processor):
    """测试文件过大的情况"""
    # 创建临时大文件
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        # 写入超过限制的内容
        large_content = "x" * (document_processor.max_file_size + 1)
        f.write(large_content.encode())
        temp_file_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="File too large"):
            await document_processor.process_document(temp_file_path)
    finally:
        temp_file_path.unlink()


@pytest.mark.asyncio
async def test_process_document_unsupported_format(document_processor):
    """测试不支持的文件格式"""
    # 创建不支持的文件类型
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write("test content".encode())
        temp_file_path = Path(f.name)
    
    try:
        with pytest.raises(ValueError, match="Unsupported file type"):
            await document_processor.process_document(temp_file_path)
    finally:
        temp_file_path.unlink()


@pytest.mark.asyncio
async def test_process_batch_success(document_processor):
    """测试批量处理成功"""
    # 创建多个临时文件
    temp_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(f"Test document {i}")
            temp_files.append(Path(f.name))
    
    try:
        # Mock解析器
        mock_parser = Mock()
        mock_parser.can_parse = Mock(return_value=True)
        
        async def mock_parse(file_path):
            return ParsedDocument(
                doc_id=f"doc_{file_path.name}",
                file_path=str(file_path),
                file_type="text",
                elements=[ParsedElement(f"Content of {file_path.name}", "text", {})],
                metadata={}
            )
        
        mock_parser.parse = mock_parse
        document_processor.parsers = [mock_parser]
        
        # 批量处理
        results = await document_processor.process_batch(
            temp_files, 
            concurrent_limit=2,
            continue_on_error=True
        )
        
        # 验证结果
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ProcessedDocument)
    
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            temp_file.unlink()


@pytest.mark.asyncio
async def test_process_batch_with_errors(document_processor):
    """测试批量处理时的错误处理"""
    # 创建一个存在的文件和一个不存在的文件
    valid_files = []
    invalid_files = []
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Valid content")
        valid_files.append(Path(f.name))
    
    invalid_files.append(Path("/non/existent/file.txt"))
    
    all_files = valid_files + invalid_files
    
    try:
        # Mock解析器用于有效文件
        mock_parser = Mock()
        mock_parser.can_parse = Mock(return_value=True)
        mock_parser.parse = AsyncMock(return_value=ParsedDocument(
            doc_id="valid_doc",
            file_path=str(valid_files[0]),
            file_type="text",
            elements=[ParsedElement("Valid content", "text", {})],
            metadata={}
        ))
        
        document_processor.parsers = [mock_parser]
        
        # 批量处理（继续处理错误）
        results = await document_processor.process_batch(
            all_files,
            continue_on_error=True
        )
        
        # 应该只有一个成功的结果
        assert len(results) == 1
        assert results[0].content == "Valid content"
    
    finally:
        # 清理有效文件
        for temp_file in valid_files:
            temp_file.unlink()


def test_extract_title(document_processor, sample_parsed_doc):
    """测试标题提取"""
    # 测试从元数据提取
    title = document_processor._extract_title(sample_parsed_doc, Path("/test/file.txt"))
    assert title == "Test Document"
    
    # 测试从标题元素提取
    sample_parsed_doc.metadata.clear()
    title = document_processor._extract_title(sample_parsed_doc, Path("/test/file.txt"))
    assert title == "# Main Title"
    
    # 测试使用文件名
    sample_parsed_doc.elements = [
        ParsedElement("No heading content", "text", {})
    ]
    title = document_processor._extract_title(sample_parsed_doc, Path("/test/file.txt"))
    assert title == "file"


def test_merge_content(document_processor, sample_parsed_doc):
    """测试内容合并"""
    content = document_processor._merge_content(sample_parsed_doc)
    
    expected_parts = [
        "This is a test document",
        "# Main Title", 
        "Some code example"
    ]
    
    for part in expected_parts:
        assert part in content


def test_get_supported_formats(document_processor):
    """测试获取支持的格式列表"""
    formats = document_processor.get_supported_formats()
    
    assert isinstance(formats, list)
    assert len(formats) > 0
    assert all(fmt.startswith('.') for fmt in formats)


@pytest.mark.asyncio 
async def test_build_processing_info(document_processor, sample_parsed_doc):
    """测试构建处理信息"""
    processing_info = await document_processor._build_processing_info(sample_parsed_doc)
    
    # 验证结构
    assert "chunks" in processing_info
    assert "auto_tags" in processing_info
    assert "classification" in processing_info
    assert "summary" in processing_info
    assert "key_entities" in processing_info
    assert "structure" in processing_info
    
    # 验证结构信息
    structure = processing_info["structure"]
    assert "element_types" in structure
    assert "total_elements" in structure
    assert structure["total_elements"] == 3
    assert structure["has_code"] is True


def test_calculate_hash():
    """测试哈希计算功能"""
    processor = DocumentProcessor()
    
    content1 = "test content"
    content2 = "test content"
    content3 = "different content"
    
    hash1 = processor._calculate_hash(content1)
    hash2 = processor._calculate_hash(content2)  
    hash3 = processor._calculate_hash(content3)
    
    assert hash1 == hash2  # 相同内容应该有相同哈希
    assert hash1 != hash3  # 不同内容应该有不同哈希


def test_get_mime_type(document_processor):
    """测试MIME类型检测"""
    # 测试常见文件类型
    assert document_processor._get_mime_type(Path("test.txt")) == "text/plain"
    assert document_processor._get_mime_type(Path("test.pdf")) == "application/pdf"
    assert document_processor._get_mime_type(Path("test.json")) == "application/json"
    
    # 测试未知类型
    assert document_processor._get_mime_type(Path("test.unknown")) == "application/octet-stream"


@pytest.mark.parametrize("enable_ocr,extract_images", [
    (True, True),
    (True, False), 
    (False, True),
    (False, False)
])
def test_document_processor_initialization(enable_ocr, extract_images):
    """测试不同参数的初始化"""
    processor = DocumentProcessor(
        enable_ocr=enable_ocr,
        extract_images=extract_images
    )
    
    assert processor.enable_ocr == enable_ocr
    assert processor.extract_images == extract_images
    assert len(processor.parsers) > 0
    assert len(processor.supported_extensions) > 0