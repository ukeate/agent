"""
测试检索功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ai.rag.retriever import SemanticRetriever, HybridRetriever, QueryIntentClassifier


class TestSemanticRetriever:
    """测试语义检索器"""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant客户端"""
        mock_client = MagicMock()
        
        # Mock搜索结果
        mock_hit = MagicMock()
        mock_hit.id = "test_id_1"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "content": "测试内容",
            "file_path": "/test/file.py",
            "file_type": "python",
            "chunk_index": 0,
        }
        mock_client.search.return_value = [mock_hit]
        
        return mock_client

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock嵌入服务"""
        mock_service = AsyncMock()
        mock_service.embed_text.return_value = [0.1] * 1536
        return mock_service

    @pytest.fixture
    def retriever(self, mock_qdrant_client, mock_embedding_service):
        """创建测试用的语义检索器"""
        retriever = SemanticRetriever()
        retriever.client = mock_qdrant_client
        retriever.embedding_service = mock_embedding_service
        return retriever

    @pytest.mark.asyncio
    async def test_search_success(self, retriever, mock_qdrant_client, mock_embedding_service):
        """测试成功搜索"""
        query = "查找测试内容"
        
        results = await retriever.search(
            query=query,
            collection="documents",
            limit=10,
            score_threshold=0.7
        )
        
        # 验证嵌入服务调用
        mock_embedding_service.embed_text.assert_called_once_with(query)
        
        # 验证Qdrant搜索调用
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["collection_name"] == "documents"
        assert call_args.kwargs["limit"] == 10
        assert call_args.kwargs["score_threshold"] == 0.7
        
        # 验证结果
        assert len(results) == 1
        assert results[0]["id"] == "test_id_1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "测试内容"
        assert results[0]["file_path"] == "/test/file.py"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, retriever, mock_qdrant_client):
        """测试带过滤条件的搜索"""
        query = "查找测试内容"
        filters = {"file_type": "python", "tags": ["test", "unit"]}
        
        await retriever.search(
            query=query,
            filter_dict=filters
        )
        
        # 验证调用包含过滤器
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_multi_collection_search(self, retriever, mock_qdrant_client):
        """测试多集合搜索"""
        query = "查找测试内容"
        collections = ["documents", "code"]
        
        # Mock不同集合的结果
        mock_hits = [
            MagicMock(id="doc_1", score=0.9, payload={"content": "文档内容"}),
            MagicMock(id="code_1", score=0.8, payload={"content": "代码内容"}),
        ]
        mock_qdrant_client.search.side_effect = [[mock_hits[0]], [mock_hits[1]]]
        
        results = await retriever.multi_collection_search(
            query=query,
            collections=collections,
            limit=5
        )
        
        # 验证多次搜索调用
        assert mock_qdrant_client.search.call_count == 2
        
        # 验证结果合并和排序
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]  # 按分数降序
        assert results[0]["collection"] in collections
        assert results[1]["collection"] in collections


class TestHybridRetriever:
    """测试混合检索器"""

    @pytest.fixture
    def mock_semantic_retriever(self):
        """Mock语义检索器"""
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [
            {
                "id": "semantic_1",
                "score": 0.9,
                "content": "语义匹配内容",
                "file_path": "/test/semantic.py",
                "file_type": "python",
                "chunk_index": 0,
                "metadata": {},
            }
        ]
        return mock_retriever

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant客户端"""
        mock_client = MagicMock()
        
        # Mock关键词搜索结果
        mock_point = MagicMock()
        mock_point.id = "keyword_1"
        mock_point.payload = {
            "content": "关键词匹配内容关键词匹配内容",
            "file_path": "/test/keyword.py",
            "file_type": "python",
            "chunk_index": 0,
        }
        mock_client.scroll.return_value = ([mock_point], None)
        
        return mock_client

    @pytest.fixture
    def hybrid_retriever(self, mock_semantic_retriever, mock_qdrant_client):
        """创建测试用的混合检索器"""
        retriever = HybridRetriever()
        retriever.semantic_retriever = mock_semantic_retriever
        retriever.client = mock_qdrant_client
        return retriever

    def test_extract_keywords(self, hybrid_retriever):
        """测试关键词提取"""
        text = "This is a test document with keywords and important terms."
        keywords = hybrid_retriever._extract_keywords(text)
        
        # 验证关键词提取
        assert isinstance(keywords, list)
        assert "test" in keywords
        assert "document" in keywords
        assert "keywords" in keywords
        assert "important" in keywords
        assert "terms" in keywords
        
        # 验证停用词被过滤
        assert "this" not in keywords
        assert "is" not in keywords
        assert "a" not in keywords
        assert "with" not in keywords
        assert "and" not in keywords

    def test_calculate_bm25_score(self, hybrid_retriever):
        """测试BM25分数计算"""
        query_keywords = ["test", "python"]
        doc_content = "This is a test document about python programming and testing."
        
        score = hybrid_retriever._calculate_bm25_score(query_keywords, doc_content)
        
        # 验证分数
        assert isinstance(score, float)
        assert score > 0  # 应该有正分数

    @pytest.mark.asyncio
    async def test_hybrid_search(self, hybrid_retriever, mock_semantic_retriever, mock_qdrant_client):
        """测试混合搜索"""
        query = "test python code"
        
        results = await hybrid_retriever.hybrid_search(
            query=query,
            collection="code",
            limit=10,
            semantic_weight=0.7,
            keyword_weight=0.3
        )
        
        # 验证语义搜索调用
        mock_semantic_retriever.search.assert_called_once()
        
        # 验证关键词搜索调用
        mock_qdrant_client.scroll.assert_called_once()
        
        # 验证结果融合
        assert isinstance(results, list)
        assert len(results) > 0
        
        # 验证结果包含融合后的分数
        for result in results:
            assert "final_score" in result
            assert "semantic_score" in result
            assert "keyword_score" in result

    @pytest.mark.asyncio
    async def test_rerank_results(self, hybrid_retriever):
        """测试结果重新排序"""
        results = [
            {"file_path": "/test/file1.py", "score": 0.9, "content": "content1"},
            {"file_path": "/test/file1.py", "score": 0.8, "content": "content2"},
            {"file_path": "/test/file2.py", "score": 0.7, "content": "content3"},
            {"file_path": "/test/file3.py", "score": 0.6, "content": "content4"},
        ]
        
        reranked = await hybrid_retriever.rerank_results(
            query="test",
            results=results
        )
        
        # 验证重排序结果
        assert len(reranked) == len(results)
        
        # 验证多样性（不同文件的结果应该分布更均匀）
        file_paths = [r["file_path"] for r in reranked[:3]]
        unique_files = set(file_paths)
        assert len(unique_files) >= 2  # 前3个结果中至少有2个不同文件


class TestQueryIntentClassifier:
    """测试查询意图分类器"""

    @pytest.fixture
    def classifier(self):
        """创建分类器"""
        return QueryIntentClassifier()

    def test_classify_code_intent(self, classifier):
        """测试代码相关查询分类"""
        queries = [
            "How to implement a function in Python?",
            "Fix this bug in my code",
            "What's wrong with this algorithm?",
            "Debug this error message"
        ]
        
        for query in queries:
            intent = classifier.classify(query)
            assert intent["type"] == "code"

    def test_classify_documentation_intent(self, classifier):
        """测试文档相关查询分类"""
        queries = [
            "What is machine learning?",
            "Explain how neural networks work",
            "Show me an example of REST API",
            "How to guide for beginners"
        ]
        
        for query in queries:
            intent = classifier.classify(query)
            assert intent["type"] == "documentation"

    def test_classify_language_detection(self, classifier):
        """测试编程语言检测"""
        test_cases = [
            ("Python list comprehension", "python"),
            ("JavaScript async await", "javascript"),
            ("TypeScript interfaces", "typescript"),
            ("Java Spring Boot", "java"),
            ("C++ templates", "cpp"),
            ("Go goroutines", "go"),
            ("Rust ownership", "rust"),
        ]
        
        for query, expected_lang in test_cases:
            intent = classifier.classify(query)
            assert intent["language"] == expected_lang

    def test_classify_general_intent(self, classifier):
        """测试通用查询分类"""
        queries = [
            "Hello world",
            "What's the weather today?",
            "Random question about life",
        ]
        
        for query in queries:
            intent = classifier.classify(query)
            assert intent["type"] == "general"
            assert intent["language"] is None

    def test_empty_query(self, classifier):
        """测试空查询处理"""
        intent = classifier.classify("")
        
        assert intent["type"] == "general"
        assert intent["language"] is None
        assert intent["framework"] is None
        assert intent["keywords"] == []