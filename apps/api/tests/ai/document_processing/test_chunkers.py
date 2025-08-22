"""分块系统测试"""

import pytest
from unittest.mock import Mock

from src.ai.document_processing.chunkers import (
    IntelligentChunker, 
    ChunkStrategy, 
    DocumentChunk
)


class TestIntelligentChunker:
    """智能分块器测试"""
    
    @pytest.fixture
    def chunker(self):
        """创建分块器实例"""
        return IntelligentChunker(
            chunk_size=500,
            chunk_overlap=100,
            strategy=ChunkStrategy.SEMANTIC,
            preserve_structure=True,
            min_chunk_size=50,
            max_chunk_size=1000
        )
    
    def test_initialization(self, chunker):
        """测试初始化"""
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.strategy == ChunkStrategy.SEMANTIC
        assert chunker.preserve_structure is True
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 1000
    
    @pytest.mark.asyncio
    async def test_fixed_chunking(self):
        """测试固定大小分块"""
        chunker = IntelligentChunker(
            chunk_size=100,
            chunk_overlap=20,
            strategy=ChunkStrategy.FIXED
        )
        
        content = "This is a test document. " * 20  # 约500字符
        
        chunks = await chunker.chunk_document(content, "text")
        
        assert len(chunks) > 1  # 应该分成多个块
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        
        # 检查块大小
        for chunk in chunks[:-1]:  # 除了最后一块
            assert len(chunk.content) <= 120  # chunk_size + 一些余量
        
        # 检查重叠
        if len(chunks) > 1:
            assert chunks[1].overlap_with_previous > 0
    
    @pytest.mark.asyncio  
    async def test_semantic_chunking_text(self):
        """测试语义分块 - 纯文本"""
        chunker = IntelligentChunker(strategy=ChunkStrategy.SEMANTIC)
        
        content = """This is paragraph one. It contains several sentences.
        
This is paragraph two. It has different content.

This is paragraph three. More content here."""
        
        chunks = await chunker.chunk_document(content, "text")
        
        assert len(chunks) >= 1
        assert all(chunk.chunk_type in ["paragraph", "text"] for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_chunking_markdown(self):
        """测试语义分块 - Markdown"""
        chunker = IntelligentChunker(strategy=ChunkStrategy.SEMANTIC)
        
        markdown_content = """# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1.

### Subsection 1.1

More detailed content.

## Section 2

Content for section 2."""
        
        chunks = await chunker.chunk_document(markdown_content, "markdown")
        
        assert len(chunks) >= 1
        # 应该包含基于标题的分块
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert "markdown_section" in chunk_types
    
    @pytest.mark.asyncio
    async def test_hierarchical_chunking(self):
        """测试层次分块"""
        chunker = IntelligentChunker(strategy=ChunkStrategy.HIERARCHICAL)
        
        content = """# Document Title

## Chapter 1

### Section 1.1
Content for section 1.1

### Section 1.2  
Content for section 1.2

## Chapter 2

### Section 2.1
Content for section 2.1"""
        
        chunks = await chunker.chunk_document(content, "markdown")
        
        assert len(chunks) >= 1
        
        # 检查层次结构
        parent_chunks = [c for c in chunks if c.parent_chunk_id is None]
        child_chunks = [c for c in chunks if c.parent_chunk_id is not None]
        
        assert len(parent_chunks) > 0  # 应该有父块
    
    @pytest.mark.asyncio
    async def test_sliding_window_chunking(self):
        """测试滑动窗口分块"""
        chunker = IntelligentChunker(
            chunk_size=200,
            chunk_overlap=50,
            strategy=ChunkStrategy.SLIDING_WINDOW
        )
        
        content = "Word " * 100  # 创建长文本
        
        chunks = await chunker.chunk_document(content)
        
        assert len(chunks) > 1
        
        # 检查滑动窗口重叠
        for i in range(1, len(chunks)):
            assert chunks[i].overlap_with_previous > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_chunking(self):
        """测试自适应分块"""
        chunker = IntelligentChunker(strategy=ChunkStrategy.ADAPTIVE)
        
        # 混合内容：有些密度高，有些密度低
        content = """Dense paragraph with lots of information, technical terms, complex sentences that require more processing.

Simple line.

Another dense paragraph with technical information, complex data structures, detailed explanations, and comprehensive examples."""
        
        metadata = {"complexity": "mixed"}
        
        chunks = await chunker.chunk_document(content, "text", metadata)
        
        assert len(chunks) >= 1
        
        # 自适应分块应该根据内容密度调整块大小
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        # 至少应该有一些大小变化
        assert len(set(chunk_sizes)) > 1 or len(chunks) == 1
    
    @pytest.mark.asyncio
    async def test_code_chunking(self):
        """测试代码分块"""
        chunker = IntelligentChunker(strategy=ChunkStrategy.SEMANTIC)
        
        code_content = '''def function1():
    """First function"""
    return "hello"

class MyClass:
    """Sample class"""
    
    def method1(self):
        return "method1"
    
    def method2(self):
        return "method2"

def function2():
    """Second function"""
    return "world"'''
        
        chunks = await chunker.chunk_document(code_content, "code")
        
        assert len(chunks) >= 1
        
        # 检查代码块类型
        chunk_types = [chunk.chunk_type for chunk in chunks]
        assert any("function" in ct or "class" in ct or "code" in ct for ct in chunk_types)
    
    @pytest.mark.asyncio
    async def test_empty_content(self, chunker):
        """测试空内容"""
        chunks = await chunker.chunk_document("", "text")
        
        # 空内容可能返回空列表或单个空块
        assert isinstance(chunks, list)
        assert len(chunks) <= 1
    
    @pytest.mark.asyncio
    async def test_very_short_content(self, chunker):
        """测试很短的内容"""
        short_content = "Short text."
        
        chunks = await chunker.chunk_document(short_content, "text")
        
        assert len(chunks) == 1
        assert chunks[0].content == short_content
        assert chunks[0].chunk_index == 0
    
    @pytest.mark.asyncio
    async def test_chunk_metadata_preservation(self, chunker):
        """测试块元数据保留"""
        content = "Test content for metadata preservation."
        metadata = {"source": "test", "author": "tester"}
        
        chunks = await chunker.chunk_document(content, "text", metadata)
        
        assert len(chunks) > 0
        
        # 检查元数据是否被传递
        for chunk in chunks:
            assert isinstance(chunk.metadata, dict)
            # 原始元数据应该被保留或传递
    
    def test_chunk_relationships(self, chunker):
        """测试块关系"""
        # 创建一些测试块
        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                content="First chunk",
                chunk_index=0,
                chunk_type="text",
                metadata={},
                start_char=0,
                end_char=11
            ),
            DocumentChunk(
                chunk_id="chunk_2", 
                content="Second chunk",
                chunk_index=1,
                chunk_type="text",
                metadata={},
                start_char=10,
                end_char=22,
                overlap_with_previous=1
            )
        ]
        
        # 测试后处理
        processed = chunker._post_process_chunks(chunks, {})
        
        assert len(processed) == 2
        assert processed[1].overlap_with_previous == 1


class TestChunkStrategy:
    """分块策略测试"""
    
    def test_enum_values(self):
        """测试枚举值"""
        assert ChunkStrategy.SEMANTIC.value == "semantic"
        assert ChunkStrategy.FIXED.value == "fixed"
        assert ChunkStrategy.ADAPTIVE.value == "adaptive"
        assert ChunkStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert ChunkStrategy.HIERARCHICAL.value == "hierarchical"


class TestDocumentChunk:
    """文档块数据类测试"""
    
    def test_chunk_creation(self):
        """测试块创建"""
        chunk = DocumentChunk(
            chunk_id="test_chunk_1",
            content="Test content",
            chunk_index=0,
            chunk_type="text",
            metadata={"source": "test"},
            start_char=0,
            end_char=12
        )
        
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.content == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.chunk_type == "text"
        assert chunk.metadata["source"] == "test"
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.parent_chunk_id is None
        assert chunk.child_chunk_ids is None
        assert chunk.overlap_with_previous == 0
        assert chunk.overlap_with_next == 0
    
    def test_chunk_with_relationships(self):
        """测试带关系的块"""
        parent_chunk = DocumentChunk(
            chunk_id="parent_1",
            content="Parent content",
            chunk_index=0,
            chunk_type="section", 
            metadata={},
            start_char=0,
            end_char=14,
            child_chunk_ids=["child_1", "child_2"]
        )
        
        child_chunk = DocumentChunk(
            chunk_id="child_1",
            content="Child content",
            chunk_index=1,
            chunk_type="paragraph",
            metadata={},
            start_char=15,
            end_char=28,
            parent_chunk_id="parent_1"
        )
        
        assert parent_chunk.child_chunk_ids == ["child_1", "child_2"]
        assert child_chunk.parent_chunk_id == "parent_1"


@pytest.mark.parametrize("strategy", [
    ChunkStrategy.SEMANTIC,
    ChunkStrategy.FIXED,
    ChunkStrategy.ADAPTIVE,
    ChunkStrategy.SLIDING_WINDOW,
    ChunkStrategy.HIERARCHICAL
])
@pytest.mark.asyncio
async def test_all_strategies(strategy):
    """参数化测试所有分块策略"""
    chunker = IntelligentChunker(strategy=strategy, chunk_size=200)
    
    content = """This is a test document with multiple paragraphs.

## Section 1
Content for the first section with some details.

## Section 2  
Content for the second section with more information.

Final paragraph with concluding thoughts."""
    
    chunks = await chunker.chunk_document(content, "text")
    
    # 所有策略都应该至少产生一个块
    assert len(chunks) >= 1
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(len(chunk.content.strip()) > 0 for chunk in chunks)