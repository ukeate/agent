"""
测试嵌入服务
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ai.rag.embeddings import EmbeddingService, TextChunker


class TestEmbeddingService:
    """测试嵌入服务"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI客户端"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def embedding_service(self, mock_openai_client):
        """创建测试用的嵌入服务"""
        service = EmbeddingService()
        service.client = mock_openai_client
        return service

    @pytest.mark.asyncio
    async def test_embed_text_success(self, embedding_service, mock_openai_client):
        """测试成功生成嵌入向量"""
        text = "这是一个测试文本"
        
        # Mock缓存未命中
        with patch('ai.rag.embeddings.get_redis', return_value=AsyncMock(get=AsyncMock(return_value=None))):
            result = await embedding_service.embed_text(text)
        
        # 验证结果
        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)
        
        # 验证API调用
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=text,
        )

    @pytest.mark.asyncio
    async def test_embed_text_with_cache(self, embedding_service):
        """测试缓存命中的情况"""
        text = "测试文本"
        cached_embedding = [0.5] * 1536
        
        # Mock缓存命中
        mock_redis = AsyncMock()
        mock_redis.get.return_value = '{"embedding": [0.5]}'
        
        with patch('ai.rag.embeddings.get_redis', return_value=mock_redis):
            with patch('json.loads', return_value=cached_embedding):
                result = await embedding_service.embed_text(text)
        
        # 验证返回缓存结果
        assert result == cached_embedding

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_service, mock_openai_client):
        """测试批量嵌入"""
        texts = ["文本1", "文本2", "文本3"]
        
        # Mock批量响应
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Mock无缓存
        with patch('ai.rag.embeddings.get_redis', return_value=AsyncMock(get=AsyncMock(return_value=None))):
            results = await embedding_service.embed_batch(texts)
        
        # 验证结果
        assert len(results) == 3
        assert all(len(emb) == 1536 for emb in results)
        
        # 验证API调用
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=texts,
        )

    def test_cosine_similarity(self, embedding_service):
        """测试余弦相似度计算"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        # 垂直向量相似度为0
        similarity1 = embedding_service.cosine_similarity(vec1, vec2)
        assert abs(similarity1 - 0.0) < 1e-10
        
        # 相同向量相似度为1
        similarity2 = embedding_service.cosine_similarity(vec1, vec3)
        assert abs(similarity2 - 1.0) < 1e-10
        
        # 零向量处理
        similarity3 = embedding_service.cosine_similarity([0, 0, 0], vec1)
        assert similarity3 == 0.0


class TestTextChunker:
    """测试文本分块器"""

    @pytest.fixture
    def chunker(self):
        """创建分块器"""
        return TextChunker(chunk_size=100, overlap=20)

    def test_chunk_text(self, chunker):
        """测试文本分块"""
        text = "第一段内容。\n\n第二段内容，这段比较长，包含更多的信息和细节。\n\n第三段内容。"
        
        chunks = chunker.chunk_text(text)
        
        # 验证分块结果
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("start" in chunk for chunk in chunks)
        assert all("end" in chunk for chunk in chunks)
        
        # 验证内容不为空
        assert all(chunk["content"].strip() for chunk in chunks)

    def test_chunk_code_python(self, chunker):
        """测试Python代码分块"""
        code = '''def function1():
    """第一个函数"""
    return "hello"

class TestClass:
    """测试类"""
    
    def method1(self):
        return "world"

async def async_function():
    """异步函数"""
    await asyncio.sleep(1)
'''
        
        chunks = chunker.chunk_code(code, "python")
        
        # 验证分块结果
        assert len(chunks) > 0
        assert any("function1" in chunk["content"] for chunk in chunks)
        assert any("TestClass" in chunk["content"] for chunk in chunks)
        assert any("async_function" in chunk["content"] for chunk in chunks)
        
        # 验证包含行号信息
        assert all("start_line" in chunk for chunk in chunks)
        assert all("end_line" in chunk for chunk in chunks)

    def test_chunk_code_javascript(self, chunker):
        """测试JavaScript代码分块（回退到文本分块）"""
        code = '''function hello() {
    return "hello";
}

const world = () => {
    return "world";
};'''
        
        chunks = chunker.chunk_code(code, "javascript")
        
        # JavaScript应该回退到文本分块
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)

    def test_empty_text(self, chunker):
        """测试空文本处理"""
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0

    def test_single_paragraph(self, chunker):
        """测试单段落文本"""
        text = "这是一个短段落。"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == len(text)