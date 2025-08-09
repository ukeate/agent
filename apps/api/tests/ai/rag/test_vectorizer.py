"""
测试向量化器
"""

import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ai.rag.vectorizer import FileVectorizer


class TestFileVectorizer:
    """测试文件向量化器"""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant客户端"""
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)  # 无结果
        return mock_client

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock嵌入服务"""
        mock_service = AsyncMock()
        mock_service.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
        return mock_service

    @pytest.fixture
    def mock_chunker(self):
        """Mock文本分块器"""
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.return_value = [
            {"content": "测试内容1", "start": 0, "end": 10},
            {"content": "测试内容2", "start": 10, "end": 20}
        ]
        mock_chunker.chunk_code.return_value = [
            {"content": "def test():\n    pass", "start_line": 0, "end_line": 1, "type": "function"},
            {"content": "class Test:\n    pass", "start_line": 2, "end_line": 3, "type": "class"}
        ]
        return mock_chunker

    @pytest.fixture
    def vectorizer(self, mock_qdrant_client, mock_embedding_service, mock_chunker):
        """创建测试用的向量化器"""
        vectorizer = FileVectorizer()
        vectorizer.client = mock_qdrant_client
        vectorizer.embedding_service = mock_embedding_service
        vectorizer.chunker = mock_chunker
        return vectorizer

    def test_get_file_type(self, vectorizer):
        """测试文件类型识别"""
        assert vectorizer._get_file_type("/path/to/file.py") == "python"
        assert vectorizer._get_file_type("/path/to/file.js") == "javascript"
        assert vectorizer._get_file_type("/path/to/file.md") == "markdown"
        assert vectorizer._get_file_type("/path/to/file.txt") == "text"
        assert vectorizer._get_file_type("/path/to/file.unknown") == "text"

    @pytest.mark.asyncio
    async def test_vectorize_file_success(self, vectorizer, mock_qdrant_client):
        """测试成功向量化文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def hello():\n    print('hello')\n")
            temp_file = f.name
        
        try:
            result = await vectorizer.vectorize_file(temp_file)
            
            # 验证结果
            assert result["status"] == "indexed"
            assert result["file"] == temp_file
            assert result["chunks"] == 2
            assert result["collection"] == "code"
            
            # 验证Qdrant调用
            mock_qdrant_client.upsert.assert_called_once()
        
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_vectorize_file_not_found(self, vectorizer):
        """测试文件不存在的情况"""
        with pytest.raises(FileNotFoundError):
            await vectorizer.vectorize_file("/non/existent/file.py")

    @pytest.mark.asyncio
    async def test_vectorize_file_unsupported_type(self, vectorizer):
        """测试不支持的文件类型"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bin', delete=False) as f:
            f.write("binary data")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                await vectorizer.vectorize_file(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_vectorize_file_already_indexed(self, vectorizer, mock_qdrant_client):
        """测试文件已索引的情况"""
        # Mock文件已存在
        mock_qdrant_client.scroll.return_value = ([MagicMock()], None)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('hello')")
            temp_file = f.name
        
        try:
            result = await vectorizer.vectorize_file(temp_file)
            
            # 验证跳过结果
            assert result["status"] == "skipped"
            assert result["file"] == temp_file
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_vectorize_directory(self, vectorizer):
        """测试目录向量化"""
        # 创建临时目录结构
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            with open(os.path.join(temp_dir, "test1.py"), "w") as f:
                f.write("print('test1')")
            with open(os.path.join(temp_dir, "test2.md"), "w") as f:
                f.write("# Test Document")
            with open(os.path.join(temp_dir, "ignore.bin"), "w") as f:
                f.write("binary")
            
            results = await vectorizer.vectorize_directory(temp_dir, recursive=False)
            
            # 验证结果
            assert len(results) == 2  # 只处理支持的文件类型
            assert any(r["file"].endswith("test1.py") for r in results)
            assert any(r["file"].endswith("test2.md") for r in results)

    @pytest.mark.asyncio
    async def test_update_index(self, vectorizer, mock_qdrant_client):
        """测试索引更新"""
        # Mock文件不存在（需要删除）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('test')")
            temp_file = f.name
        
        # 删除文件模拟文件不存在的情况
        os.unlink(temp_file)
        
        results = await vectorizer.update_index([temp_file])
        
        # 验证结果
        assert len(results) == 1
        assert results[0]["status"] == "removed"
        assert results[0]["file"] == temp_file

    @pytest.mark.asyncio
    async def test_remove_from_index(self, vectorizer, mock_qdrant_client):
        """测试从索引中移除文件"""
        file_path = "/path/to/test.py"
        
        await vectorizer.remove_from_index(file_path)
        
        # 验证两个集合都被调用删除
        assert mock_qdrant_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_get_index_stats(self, vectorizer, mock_qdrant_client):
        """测试获取索引统计"""
        # Mock集合信息
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 50
        mock_info.segments_count = 1
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info
        
        stats = await vectorizer.get_index_stats()
        
        # 验证统计结果
        assert "documents" in stats
        assert "code" in stats
        assert stats["documents"]["vectors_count"] == 100
        assert stats["documents"]["points_count"] == 50
        assert stats["documents"]["status"] == "green"

    def test_file_hash(self, vectorizer):
        """测试文件哈希计算"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("测试内容")
            temp_file = f.name
        
        try:
            hash1 = vectorizer._get_file_hash(temp_file)
            hash2 = vectorizer._get_file_hash(temp_file)
            
            # 相同文件应该有相同哈希
            assert hash1 == hash2
            assert isinstance(hash1, str)
            assert len(hash1) == 32  # MD5哈希长度
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)