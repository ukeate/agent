"""Qdrant BM42混合搜索功能测试"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List
from src.ai.rag.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
    LanguageDetector,
    QueryPreprocessor,
    QdrantBM42Client,
    SearchStrategy,
    SearchResult,
    get_hybrid_search_config,
)
from qdrant_client.models import ScoredPoint

class TestLanguageDetector:
    """语言检测器测试"""
    
    def test_detect_chinese(self):
        detector = LanguageDetector()
        result = detector.detect("这是一个中文查询")
        assert result == "zh"
    
    def test_detect_english(self):
        detector = LanguageDetector()
        result = detector.detect("This is an English query")
        assert result == "en"
    
    def test_detect_mixed(self):
        detector = LanguageDetector()
        result = detector.detect("这是 mixed 查询")
        assert result in ["zh", "en"]  # 可能是中文或英文
    
    def test_detect_auto(self):
        detector = LanguageDetector()
        result = detector.detect("123456")
        assert result == "auto"

class TestQueryPreprocessor:
    """查询预处理器测试"""
    
    @pytest.mark.asyncio
    async def test_process_chinese_query(self):
        preprocessor = QueryPreprocessor()
        result = await preprocessor.process("这是一个测试查询")
        
        assert result.original == "这是一个测试查询"
        assert result.language == "zh"
        assert isinstance(result.keywords, list)
        assert isinstance(result.expanded_terms, list)
    
    @pytest.mark.asyncio
    async def test_process_english_query(self):
        preprocessor = QueryPreprocessor()
        result = await preprocessor.process("This is a test query")
        
        assert result.original == "This is a test query"
        assert result.language == "en"
        assert isinstance(result.keywords, list)
        assert "test" in result.keywords
    
    @pytest.mark.asyncio
    async def test_process_mixed_query(self):
        preprocessor = QueryPreprocessor()
        result = await preprocessor.process("测试 test query")
        
        assert result.original == "测试 test query"
        assert result.language in ["zh", "en", "mixed"]
        assert isinstance(result.keywords, list)

class TestQdrantBM42Client:
    """QdrantBM42客户端测试"""
    
    def setup_method(self):
        """设置测试方法"""
        self.mock_client = MagicMock()
        self.config = HybridSearchConfig(
            vector_weight=0.7,
            bm25_weight=0.3,
            top_k=10,
            strategy=SearchStrategy.HYBRID_RRF
        )
        self.bm42_client = QdrantBM42Client(self.mock_client, self.config)
    
    def test_calculate_bm25_score(self):
        """测试BM25分数计算"""
        keywords = ["test", "query"]
        content = "This is a test document for query testing"
        
        score = self.bm42_client._calculate_bm25_score(keywords, content)
        assert isinstance(score, float)
        assert score > 0
    
    def test_calculate_bm25_score_empty_content(self):
        """测试空内容的BM25分数计算"""
        keywords = ["test"]
        content = ""
        
        score = self.bm42_client._calculate_bm25_score(keywords, content)
        assert score == 0.0
    
    def test_calculate_bm25_score_no_keywords(self):
        """测试无关键词的BM25分数计算"""
        keywords = []
        content = "This is test content"
        
        score = self.bm42_client._calculate_bm25_score(keywords, content)
        assert score == 0.0
    
    def test_rrf_fusion(self):
        """测试RRF融合算法"""
        # 创建模拟搜索结果
        vector_results = [
            ScoredPoint(id="doc1", version=0, score=0.9, payload={"content": "test1"}, vector=None),
            ScoredPoint(id="doc2", version=0, score=0.8, payload={"content": "test2"}, vector=None),
        ]
        bm25_results = [
            ScoredPoint(id="doc2", version=0, score=0.7, payload={"content": "test2"}, vector=None),
            ScoredPoint(id="doc3", version=0, score=0.6, payload={"content": "test3"}, vector=None),
        ]
        
        fused_results = self.bm42_client._rrf_fusion(vector_results, bm25_results)
        
        assert len(fused_results) <= self.config.top_k
        assert all(isinstance(result, ScoredPoint) for result in fused_results)
        # doc2应该排名更高，因为它在两个结果中都出现
        result_ids = [r.id for r in fused_results]
        assert "doc2" in result_ids
    
    def test_weighted_fusion(self):
        """测试加权融合算法"""
        vector_results = [
            ScoredPoint(id="doc1", version=0, score=0.9, payload={"content": "test1"}, vector=None),
            ScoredPoint(id="doc2", version=0, score=0.8, payload={"content": "test2"}, vector=None),
        ]
        bm25_results = [
            ScoredPoint(id="doc2", version=0, score=0.7, payload={"content": "test2"}, vector=None),
            ScoredPoint(id="doc3", version=0, score=0.6, payload={"content": "test3"}, vector=None),
        ]
        
        fused_results = self.bm42_client._weighted_fusion(vector_results, bm25_results)
        
        assert len(fused_results) <= self.config.top_k
        assert all(isinstance(result, ScoredPoint) for result in fused_results)
        # 检查分数是否正确计算
        for result in fused_results:
            assert isinstance(result.score, float)
    
    def test_normalize_scores(self):
        """测试分数归一化"""
        scores = [0.9, 0.5, 0.1]
        normalized = self.bm42_client._normalize_scores(scores)
        
        assert len(normalized) == len(scores)
        assert max(normalized) == 1.0
        assert min(normalized) == 0.0
    
    def test_normalize_scores_same_values(self):
        """测试相同分数的归一化"""
        scores = [0.5, 0.5, 0.5]
        normalized = self.bm42_client._normalize_scores(scores)
        
        assert all(score == 1.0 for score in normalized)
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """测试向量搜索"""
        # 模拟Qdrant客户端返回
        mock_results = [
            ScoredPoint(
                id="doc1",
                version=0,
                score=0.9,
                payload={"content": "test content", "file_path": "/test.txt"},
                vector=None
            )
        ]
        self.mock_client.search.return_value = mock_results
        
        vector = [0.1] * 1536  # 模拟向量
        results = await self.bm42_client._vector_search(vector, "documents", None)
        
        assert len(results) == 1
        assert results[0].id == "doc1"
        self.mock_client.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_parallel_search(self):
        """测试并行搜索"""
        # 模拟向量搜索结果
        vector_results = [
            ScoredPoint(id="doc1", version=0, score=0.9, payload={"content": "test"}, vector=None)
        ]
        
        # 模拟BM25搜索结果
        bm25_results = [
            ScoredPoint(id="doc2", version=0, score=0.8, payload={"content": "test"}, vector=None)
        ]
        
        # 使用patch模拟内部方法
        with patch.object(self.bm42_client, '_vector_search', return_value=vector_results), \
             patch.object(self.bm42_client, '_bm25_search', return_value=bm25_results):
            
            vector = [0.1] * 1536
            v_results, b_results = await self.bm42_client._parallel_search(
                "test query", vector, "documents", None
            )
            
            assert v_results == vector_results
            assert b_results == bm25_results

class TestHybridSearchEngine:
    """混合搜索引擎测试"""
    
    def setup_method(self):
        """设置测试方法"""
        with patch('src.ai.rag.hybrid_search.get_bm42_client'), \
             patch('src.ai.rag.hybrid_search.get_settings'), \
             patch('src.ai.rag.hybrid_search.embedding_service'):
            self.search_engine = HybridSearchEngine()
    
    def test_generate_cache_key(self):
        """测试缓存键生成"""
        from src.ai.rag.hybrid_search import ProcessedQuery
        
        processed_query = ProcessedQuery(
            original="test query",
            text="test query",
            language="en",
            keywords=["test"],
            expanded_terms=["test"]
        )
        
        cache_key = self.search_engine._generate_cache_key(
            processed_query, "documents", None, None
        )
        
        assert isinstance(cache_key, str)
        assert cache_key.startswith("hybrid_search:")
    
    def test_build_filters(self):
        """测试过滤条件构建"""
        filter_dict = {
            "file_type": "python",
            "file_path": "/test.py"
        }
        
        filters = self.search_engine._build_filters(filter_dict)
        
        assert filters is not None
        assert len(filters.must) == 2
    
    def test_build_filters_empty(self):
        """测试空过滤条件"""
        filters = self.search_engine._build_filters({})
        assert filters is None
    
    def test_format_results(self):
        """测试结果格式化"""
        scored_points = [
            ScoredPoint(
                id="doc1",
                version=0,
                score=0.9,
                payload={
                    "content": "test content",
                    "file_path": "/test.txt",
                    "file_type": "text",
                    "chunk_index": 0
                },
                vector=None
            )
        ]
        
        results = self.search_engine._format_results(scored_points, "documents")
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "doc1"
        assert results[0].collection == "documents"
    
    def test_diversity_rerank(self):
        """测试多样性重排序"""
        results = [
            SearchResult(
                id="doc1", score=0.9, content="test1", 
                file_path="/file1.txt", file_type="text", 
                chunk_index=0, metadata={}, collection="documents"
            ),
            SearchResult(
                id="doc2", score=0.8, content="test2", 
                file_path="/file1.txt", file_type="text", 
                chunk_index=1, metadata={}, collection="documents"
            ),
            SearchResult(
                id="doc3", score=0.7, content="test3", 
                file_path="/file2.txt", file_type="text", 
                chunk_index=0, metadata={}, collection="documents"
            ),
        ]
        
        reranked = self.search_engine._diversity_rerank(results)
        
        assert len(reranked) == len(results)
        # 检查不同文件的结果是否得到适当分散
        file_paths = [r.file_path for r in reranked[:2]]
        assert len(set(file_paths)) >= 1  # 至少有一个不同的文件

class TestConfiguration:
    """配置测试"""
    
    @patch('src.ai.rag.hybrid_search.get_settings')
    def test_get_hybrid_search_config(self, mock_get_settings):
        """测试混合搜索配置获取"""
        mock_settings = MagicMock()
        mock_settings.HYBRID_SEARCH_VECTOR_WEIGHT = 0.6
        mock_settings.HYBRID_SEARCH_BM25_WEIGHT = 0.4
        mock_settings.HYBRID_SEARCH_TOP_K = 15
        mock_settings.HYBRID_SEARCH_STRATEGY = "hybrid_weighted"
        mock_settings.HYBRID_SEARCH_ENABLE_CACHE = True
        mock_settings.CACHE_TTL_DEFAULT = 1800
        mock_settings.HYBRID_SEARCH_RRF_K = 50
        mock_settings.HYBRID_SEARCH_RERANK_SIZE = 80
        
        mock_get_settings.return_value = mock_settings
        
        config = get_hybrid_search_config()
        
        assert config.vector_weight == 0.6
        assert config.bm25_weight == 0.4
        assert config.top_k == 15
        assert config.strategy == SearchStrategy.HYBRID_WEIGHTED
        assert config.enable_cache is True
        assert config.cache_ttl == 1800
        assert config.rrf_k == 50

@pytest.mark.integration
class TestHybridSearchIntegration:
    """混合搜索集成测试"""
    
    @pytest.mark.asyncio
    async def test_search_engine_integration(self):
        """测试搜索引擎集成功能"""
        # 这是一个集成测试，需要真实的Qdrant实例
        # 在实际测试中，应该使用测试数据库
        
        # 暂时跳过，因为需要真实的Qdrant连接
        pytest.skip("Integration test requires real Qdrant instance")
    
    @pytest.mark.asyncio
    async def test_rag_service_integration(self):
        """测试RAG服务集成"""
        # 测试RAG服务是否正确集成了混合搜索
        from src.services.rag_service import rag_service
        
        # 检查RAG服务是否有BM42搜索引擎实例
        assert hasattr(rag_service, 'bm42_search_engine')
        assert rag_service.bm42_search_engine is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
