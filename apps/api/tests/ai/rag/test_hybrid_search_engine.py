"""
混合搜索引擎测试
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from ai.rag.hybrid_search_engine import (
    HybridSearchEngine,
    SearchMode,
    FusionStrategy,
    SearchConfig,
    SearchResult
)


@pytest.fixture
def mock_db_session():
    """模拟数据库会话"""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def search_engine(mock_db_session):
    """创建搜索引擎实例"""
    return HybridSearchEngine(mock_db_session)


@pytest.fixture
def sample_query_vector():
    """示例查询向量"""
    return np.random.randn(384)


@pytest.mark.asyncio
async def test_semantic_search(search_engine, mock_db_session, sample_query_vector):
    """测试语义搜索"""
    # 模拟数据库返回
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(
            id="1",
            content="This is a test document about machine learning",
            metadata={"category": "AI"},
            distance=0.2
        ),
        MagicMock(
            id="2",
            content="Another document about deep learning",
            metadata={"category": "AI"},
            distance=0.3
        )
    ]
    mock_db_session.execute.return_value = mock_result
    
    config = SearchConfig(search_mode=SearchMode.SEMANTIC, top_k=2)
    results = await search_engine.hybrid_search(
        query="machine learning",
        query_vector=sample_query_vector,
        config=config
    )
    
    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].semantic_score == 0.8  # 1 - 0.2
    assert results[0].final_score == 0.8
    assert results[1].id == "2"
    assert results[1].semantic_score == 0.7  # 1 - 0.3


@pytest.mark.asyncio
async def test_keyword_search(search_engine, mock_db_session):
    """测试关键词搜索"""
    # 模拟数据库返回
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        MagicMock(
            id="1",
            content="This document contains machine learning keywords",
            metadata={"category": "AI"},
            rank=0.9
        ),
        MagicMock(
            id="2",
            content="Another document with learning content",
            metadata={"category": "Education"},
            rank=0.6
        )
    ]
    mock_db_session.execute.return_value = mock_result
    
    config = SearchConfig(search_mode=SearchMode.KEYWORD, top_k=2)
    results = await search_engine.hybrid_search(
        query="machine learning",
        config=config
    )
    
    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].keyword_score == 1.0  # 0.9/0.9 (normalized)
    assert results[1].keyword_score == pytest.approx(0.667, rel=0.01)  # 0.6/0.9


@pytest.mark.asyncio
async def test_hybrid_search_rrf(search_engine, mock_db_session, sample_query_vector):
    """测试混合搜索（RRF融合）"""
    # 模拟语义搜索结果
    semantic_results = [
        SearchResult(
            id="1",
            content="Document 1",
            metadata={},
            semantic_score=0.9,
            keyword_score=0.0,
            final_score=0.9,
            distance=0.1
        ),
        SearchResult(
            id="2",
            content="Document 2",
            metadata={},
            semantic_score=0.8,
            keyword_score=0.0,
            final_score=0.8,
            distance=0.2
        )
    ]
    
    # 模拟关键词搜索结果
    keyword_results = [
        SearchResult(
            id="2",
            content="Document 2",
            metadata={},
            semantic_score=0.0,
            keyword_score=0.95,
            final_score=0.95,
            distance=1.0,
            highlights=["test highlight"]
        ),
        SearchResult(
            id="3",
            content="Document 3",
            metadata={},
            semantic_score=0.0,
            keyword_score=0.7,
            final_score=0.7,
            distance=1.0
        )
    ]
    
    config = SearchConfig(
        search_mode=SearchMode.HYBRID,
        fusion_strategy=FusionStrategy.RRF,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # 执行RRF融合
    fused_results = await search_engine._rrf_fusion(
        semantic_results, keyword_results, config
    )
    
    assert len(fused_results) == 3
    # 文档2应该排名最高，因为它在两个搜索中都出现
    assert fused_results[0].id == "2"
    assert fused_results[0].highlights == ["test highlight"]


@pytest.mark.asyncio
async def test_hybrid_search_linear(search_engine, mock_db_session, sample_query_vector):
    """测试混合搜索（线性融合）"""
    semantic_results = [
        SearchResult(
            id="1",
            content="Document 1",
            metadata={},
            semantic_score=0.9,
            keyword_score=0.0,
            final_score=0.9,
            distance=0.1
        )
    ]
    
    keyword_results = [
        SearchResult(
            id="1",
            content="Document 1",
            metadata={},
            semantic_score=0.0,
            keyword_score=0.8,
            final_score=0.8,
            distance=1.0
        )
    ]
    
    config = SearchConfig(
        search_mode=SearchMode.HYBRID,
        fusion_strategy=FusionStrategy.LINEAR,
        semantic_weight=0.6,
        keyword_weight=0.4,
        min_relevance_score=0.5
    )
    
    fused_results = await search_engine._linear_fusion(
        semantic_results, keyword_results, config
    )
    
    assert len(fused_results) == 1
    assert fused_results[0].id == "1"
    # 线性融合: 0.6 * 0.9 + 0.4 * 0.8 = 0.54 + 0.32 = 0.86
    assert fused_results[0].final_score == pytest.approx(0.86, rel=0.01)


@pytest.mark.asyncio
async def test_query_expansion(search_engine):
    """测试查询扩展"""
    original_query = "ml and ai for nlp"
    expanded_query = await search_engine._expand_query(original_query)
    
    assert "machine learning" in expanded_query
    assert "artificial intelligence" in expanded_query
    assert "natural language processing" in expanded_query


@pytest.mark.asyncio
async def test_apply_synonyms(search_engine):
    """测试同义词处理"""
    original_query = "search for similar vectors fast"
    processed_query = await search_engine._apply_synonyms(original_query)
    
    assert "find" in processed_query
    assert "retrieve" in processed_query
    assert "related" in processed_query
    assert "quick" in processed_query


@pytest.mark.asyncio
async def test_extract_highlights(search_engine):
    """测试高亮提取"""
    content = "This is a long document about machine learning and artificial intelligence. It covers various aspects of deep learning and neural networks."
    query = "machine learning"
    
    highlights = await search_engine._extract_highlights(content, query, max_highlights=2)
    
    assert len(highlights) <= 2
    assert any("<mark>machine</mark>" in h for h in highlights)
    assert any("<mark>learning</mark>" in h for h in highlights)


@pytest.mark.asyncio
async def test_create_full_text_index(search_engine, mock_db_session):
    """测试创建全文索引"""
    mock_db_session.execute.return_value = MagicMock()
    
    result = await search_engine.create_full_text_index(
        "knowledge_items", "content"
    )
    
    assert result is True
    assert mock_db_session.execute.called
    assert mock_db_session.commit.called
    
    # 验证SQL包含正确的操作
    calls = mock_db_session.execute.call_args_list
    sql_texts = [str(call[0][0]) for call in calls]
    assert any("tsvector" in sql for sql in sql_texts)
    assert any("GIN" in sql for sql in sql_texts)


@pytest.mark.asyncio
async def test_cross_encoder_reranking(search_engine):
    """测试交叉编码器重排序"""
    query = "machine learning algorithms"
    results = [
        SearchResult(
            id="1",
            content="This document is about machine learning and algorithms",
            metadata={},
            semantic_score=0.7,
            keyword_score=0.8,
            final_score=0.75,
            distance=0.3
        ),
        SearchResult(
            id="2",
            content="Deep learning neural networks",
            metadata={},
            semantic_score=0.8,
            keyword_score=0.6,
            final_score=0.7,
            distance=0.2
        )
    ]
    
    config = SearchConfig(fusion_strategy=FusionStrategy.CROSS_ENCODER)
    reranked = await search_engine._rerank_with_cross_encoder(
        query, results, config
    )
    
    # 文档1应该排名更高，因为它包含更多查询词
    assert reranked[0].id == "1"


@pytest.mark.asyncio
async def test_search_stats(search_engine, mock_db_session, sample_query_vector):
    """测试搜索统计"""
    # 模拟搜索结果
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_db_session.execute.return_value = mock_result
    
    # 执行几次搜索
    config1 = SearchConfig(search_mode=SearchMode.SEMANTIC)
    await search_engine.hybrid_search("test", sample_query_vector, config1)
    
    config2 = SearchConfig(search_mode=SearchMode.KEYWORD)
    await search_engine.hybrid_search("test", None, config2)
    
    config3 = SearchConfig(search_mode=SearchMode.HYBRID)
    await search_engine.hybrid_search("test", sample_query_vector, config3)
    
    stats = await search_engine.get_search_stats()
    
    assert stats["total_searches"] == 3
    assert stats["semantic_searches"] == 1
    assert stats["keyword_searches"] == 1
    assert stats["hybrid_searches"] == 1
    assert stats["avg_latency_ms"] > 0


@pytest.mark.asyncio
async def test_error_handling(search_engine, mock_db_session):
    """测试错误处理"""
    # 模拟数据库错误
    mock_db_session.execute.side_effect = Exception("Database error")
    
    config = SearchConfig(search_mode=SearchMode.KEYWORD)
    results = await search_engine.hybrid_search("test query", config=config)
    
    assert results == []  # 应该返回空列表而不是抛出异常


@pytest.mark.asyncio
async def test_minimum_relevance_filtering(search_engine):
    """测试最小相关性过滤"""
    semantic_results = [
        SearchResult(
            id="1",
            content="Highly relevant",
            metadata={},
            semantic_score=0.9,
            keyword_score=0.0,
            final_score=0.9,
            distance=0.1
        ),
        SearchResult(
            id="2",
            content="Less relevant",
            metadata={},
            semantic_score=0.3,
            keyword_score=0.0,
            final_score=0.3,
            distance=0.7
        )
    ]
    
    keyword_results = []
    
    config = SearchConfig(
        fusion_strategy=FusionStrategy.LINEAR,
        semantic_weight=1.0,
        keyword_weight=0.0,
        min_relevance_score=0.5
    )
    
    fused_results = await search_engine._linear_fusion(
        semantic_results, keyword_results, config
    )
    
    # 只有分数>=0.5的结果应该被返回
    assert len(fused_results) == 1
    assert fused_results[0].id == "1"