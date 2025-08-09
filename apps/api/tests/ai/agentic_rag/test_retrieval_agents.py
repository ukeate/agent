"""
多策略检索代理协作系统集成测试

测试各个检索代理的协作能力、结果融合机制和动态策略选择算法
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.ai.agentic_rag.retrieval_agents import (
    MultiAgentRetriever, SemanticRetrievalAgent, KeywordRetrievalAgent, 
    StructuredRetrievalAgent, RetrievalResult, RetrievalStrategy
)
from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent


# 全局fixtures
@pytest.fixture
def sample_query_analysis():
    """创建示例查询分析结果"""
    return QueryAnalysis(
        query_text="机器学习算法实现",
        intent_type=QueryIntent.CODE,
        confidence=0.8,
        complexity_score=0.6,
        entities=["机器学习", "算法"],
        keywords=["机器学习", "算法", "实现"],
        domain="技术",
        sentiment="neutral",
        language="zh"
    )

@pytest.fixture
def mock_semantic_results():
    """Mock语义检索结果"""
    return [
        {
            "id": "semantic_1",
            "score": 0.85,
            "content": "机器学习算法的Python实现示例",
            "file_path": "/docs/ml_python.md",
            "file_type": "markdown",
            "chunk_index": 0,
            "metadata": {"collection": "documents"}
        },
        {
            "id": "semantic_2", 
            "score": 0.75,
            "content": "深度学习神经网络算法详解",
            "file_path": "/docs/deep_learning.md",
            "file_type": "markdown",
            "chunk_index": 1,
            "metadata": {"collection": "documents"}
        }
    ]

@pytest.fixture
def mock_keyword_results():
    """Mock关键词检索结果"""
    return [
        {
            "id": "keyword_1",
            "score": 0.8,
            "content": "机器学习算法库的实现和使用",
            "file_path": "/code/ml_lib.py",
            "file_type": "python",
            "bm25_score": 2.5,
            "keyword_matches": 8
        },
        {
            "id": "keyword_2",
            "score": 0.6,
            "content": "算法实现的最佳实践指南",
            "file_path": "/docs/best_practices.md",
            "file_type": "markdown",
            "bm25_score": 1.8,
            "keyword_matches": 5
        }
    ]

@pytest.fixture
def mock_structured_results():
    """Mock结构化检索结果"""
    return [
        {
            "id": "struct_机器学习",
            "score": 0.8,
            "content": "结构化数据匹配: 机器学习",
            "source_type": "database",
            "entity": "机器学习",
            "metadata": {
                "query_type": "structured",
                "match_type": "entity_match"
            }
        }
    ]


class TestRetrievalAgents:
    """检索代理单元测试"""


class TestSemanticRetrievalAgent:
    """语义检索代理测试"""
    
    @pytest.fixture
    def semantic_agent(self):
        """创建语义检索代理实例"""
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever') as mock_retriever:
            agent = SemanticRetrievalAgent()
            agent.semantic_retriever = mock_retriever.return_value
            return agent
    
    def test_agent_initialization(self, semantic_agent):
        """测试代理初始化"""
        assert semantic_agent.name == "SemanticExpert"
        assert semantic_agent.strategy == RetrievalStrategy.SEMANTIC
        assert semantic_agent.performance_stats["total_queries"] == 0
        assert semantic_agent.semantic_retriever is not None
    
    def test_is_suitable_for_query_high_suitability(self, semantic_agent):
        """测试高适用性查询"""
        query_analysis = QueryAnalysis(
            query_text="什么是深度学习",
            intent_type=QueryIntent.FACTUAL,  # 适合语义检索
            confidence=0.8,
            complexity_score=0.7,  # 高复杂度
            entities=["深度学习"],  # 有实体
            keywords=["深度学习", "是什么"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        suitability = semantic_agent.is_suitable_for_query(query_analysis)
        
        assert suitability >= 0.8  # 高适用性
    
    def test_is_suitable_for_query_low_suitability(self, semantic_agent):
        """测试低适用性查询"""
        query_analysis = QueryAnalysis(
            query_text="简单测试",
            intent_type=QueryIntent.CREATIVE,  # 创作类查询适用性较低
            confidence=0.5,
            complexity_score=0.2,  # 低复杂度
            entities=[],  # 无实体
            keywords=["测试"],
            domain=None,
            sentiment="neutral",
            language="zh"
        )
        
        suitability = semantic_agent.is_suitable_for_query(query_analysis)
        
        assert suitability >= 0.6  # 基础适用性
        assert suitability <= 0.9
    
    @pytest.mark.asyncio
    async def test_retrieve_success(self, semantic_agent, sample_query_analysis, mock_semantic_results):
        """测试成功检索"""
        # Mock语义检索器
        semantic_agent.semantic_retriever.search = AsyncMock(return_value=mock_semantic_results)
        
        result = await semantic_agent.retrieve(sample_query_analysis, limit=5)
        
        assert isinstance(result, RetrievalResult)
        assert result.agent_type == RetrievalStrategy.SEMANTIC
        assert result.query == sample_query_analysis.query_text
        assert len(result.results) == 2
        assert result.score > 0.0
        assert result.confidence > 0.0
        assert result.processing_time > 0.0
        assert "语义向量搜索" in result.explanation
        
        # 验证调用参数
        semantic_agent.semantic_retriever.search.assert_called_once_with(
            query=sample_query_analysis.query_text,
            collection="code",  # CODE意图应使用code集合
            limit=5,
            score_threshold=0.3,
            filter_dict=None
        )
    
    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, semantic_agent, sample_query_analysis):
        """测试空结果检索"""
        semantic_agent.semantic_retriever.search = AsyncMock(return_value=[])
        
        result = await semantic_agent.retrieve(sample_query_analysis)
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert len(result.results) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_exception_handling(self, semantic_agent, sample_query_analysis):
        """测试检索异常处理"""
        semantic_agent.semantic_retriever.search = AsyncMock(side_effect=Exception("Database Error"))
        
        result = await semantic_agent.retrieve(sample_query_analysis)
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert len(result.results) == 0
        assert "语义检索失败" in result.explanation
        assert "Database Error" in result.explanation
        
        # 验证性能统计更新
        assert semantic_agent.performance_stats["total_queries"] == 1


class TestKeywordRetrievalAgent:
    """关键词检索代理测试"""
    
    @pytest.fixture
    def keyword_agent(self):
        """创建关键词检索代理实例"""
        return KeywordRetrievalAgent()
    
    def test_agent_initialization(self, keyword_agent):
        """测试代理初始化"""
        assert keyword_agent.name == "KeywordExpert"
        assert keyword_agent.strategy == RetrievalStrategy.KEYWORD
        assert keyword_agent.k1 == 1.2
        assert keyword_agent.b == 0.75
    
    def test_is_suitable_for_query_procedural(self, keyword_agent):
        """测试程序性查询适用性"""
        query_analysis = QueryAnalysis(
            query_text="如何实现排序算法",
            intent_type=QueryIntent.PROCEDURAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=[],
            keywords=["如何", "实现", "排序", "算法"],  # 多个关键词
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        suitability = keyword_agent.is_suitable_for_query(query_analysis)
        
        assert suitability >= 0.8  # 程序性查询 + 多关键词
    
    def test_is_suitable_for_query_short_query(self, keyword_agent):
        """测试短查询适用性"""
        query_analysis = QueryAnalysis(
            query_text="Python函数",  # 短查询
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.3,
            entities=[],
            keywords=["Python", "函数"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        suitability = keyword_agent.is_suitable_for_query(query_analysis)
        
        assert suitability >= 0.8  # 短查询更适合关键词匹配
    
    def test_calculate_bm25_score(self, keyword_agent):
        """测试BM25分数计算"""
        query_terms = ["machine", "learning"]
        document_terms = ["machine", "learning", "algorithm", "implementation"]
        document_length = 4
        avg_doc_length = 5.0
        term_frequencies = {"machine": 1, "learning": 1}
        total_docs = 100
        
        score = keyword_agent._calculate_bm25_score(
            query_terms, document_terms, document_length, 
            avg_doc_length, term_frequencies, total_docs
        )
        
        assert score > 0.0
        assert isinstance(score, float)
    
    @pytest.mark.asyncio
    async def test_retrieve_with_keyword_matching(self, keyword_agent, sample_query_analysis):
        """测试关键词匹配检索"""
        # Mock语义检索器（关键词代理使用它作为候选源）
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever') as mock_retriever_class:
            mock_retriever = mock_retriever_class.return_value
            mock_retriever.search = AsyncMock(return_value=[
                {
                    "id": "test_1",
                    "score": 0.7,
                    "content": "机器学习算法实现示例代码",
                    "file_path": "/test.py",
                    "file_type": "python",
                    "chunk_index": 0,
                    "metadata": {}
                },
                {
                    "id": "test_2", 
                    "score": 0.6,
                    "content": "数据科学工具使用指南",
                    "file_path": "/guide.md",
                    "file_type": "markdown",
                    "chunk_index": 0,
                    "metadata": {}
                }
            ])
            
            result = await keyword_agent.retrieve(sample_query_analysis, limit=5)
            
            assert isinstance(result, RetrievalResult)
            assert result.agent_type == RetrievalStrategy.KEYWORD
            assert result.processing_time > 0.0
            
            # 应该有BM25评分的结果
            for item in result.results:
                assert "bm25_score" in item
                assert "keyword_matches" in item


class TestStructuredRetrievalAgent:
    """结构化检索代理测试"""
    
    @pytest.fixture
    def structured_agent(self):
        """创建结构化检索代理实例"""
        return StructuredRetrievalAgent()
    
    def test_agent_initialization(self, structured_agent):
        """测试代理初始化"""
        assert structured_agent.name == "StructuredExpert"
        assert structured_agent.strategy == RetrievalStrategy.STRUCTURED
    
    def test_is_suitable_for_query_with_entities(self, structured_agent):
        """测试有实体的查询适用性"""
        query_analysis = QueryAnalysis(
            query_text="查询用户张三的订单信息",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["张三", "订单信息"],  # 多个实体
            keywords=["查询", "用户", "订单"],
            domain="业务",
            sentiment="neutral",
            language="zh"
        )
        
        suitability = structured_agent.is_suitable_for_query(query_analysis)
        
        assert suitability >= 0.7  # 多实体 + 事实查询 + 业务领域
    
    def test_is_suitable_for_query_no_entities(self, structured_agent):
        """测试无实体查询适用性"""
        query_analysis = QueryAnalysis(
            query_text="创造性写作技巧",
            intent_type=QueryIntent.CREATIVE,
            confidence=0.8,
            complexity_score=0.4,
            entities=[],  # 无实体
            keywords=["创造性", "写作", "技巧"],
            domain="文学",
            sentiment="neutral",
            language="zh"
        )
        
        suitability = structured_agent.is_suitable_for_query(query_analysis)
        
        assert suitability <= 0.4  # 低适用性
    
    @pytest.mark.asyncio
    async def test_retrieve_with_entities(self, structured_agent):
        """测试有实体的结构化检索"""
        query_analysis = QueryAnalysis(
            query_text="用户管理系统",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["用户", "管理系统"],
            keywords=["用户", "管理", "系统"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        with patch('src.ai.agentic_rag.retrieval_agents.get_db_session') as mock_session:
            # Mock数据库会话
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            result = await structured_agent.retrieve(query_analysis, limit=5)
            
            assert isinstance(result, RetrievalResult)
            assert result.agent_type == RetrievalStrategy.STRUCTURED
            assert len(result.results) <= len(query_analysis.entities)
            assert result.processing_time > 0.0
            
            # 验证结果中包含实体信息
            for item in result.results:
                assert "entity" in item
                assert item["entity"] in query_analysis.entities
    
    @pytest.mark.asyncio
    async def test_retrieve_database_error_fallback(self, structured_agent):
        """测试数据库错误时的后备方案"""
        query_analysis = QueryAnalysis(
            query_text="系统错误",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["系统"],
            keywords=["系统", "错误"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        with patch('src.ai.agentic_rag.retrieval_agents.get_db_session') as mock_session:
            # Mock数据库会话抛出异常
            mock_session.side_effect = Exception("Database connection failed")
            
            result = await structured_agent.retrieve(query_analysis, limit=5)
            
            assert isinstance(result, RetrievalResult)
            assert len(result.results) <= len(query_analysis.entities)
            
            # 应该有后备结果
            for item in result.results:
                assert item["source_type"] == "analysis"
                assert item["metadata"]["match_type"] == "entity_analysis"


class TestMultiAgentRetriever:
    """多代理检索协调器测试"""
    
    @pytest.fixture
    def multi_agent_retriever(self):
        """创建多代理检索器实例"""
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever'):
            return MultiAgentRetriever()
    
    @pytest.fixture
    def sample_retrieval_results(self):
        """创建示例检索结果"""
        return [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="机器学习算法",
                results=[
                    {"id": "s1", "score": 0.9, "content": "语义结果1"},
                    {"id": "s2", "score": 0.8, "content": "语义结果2"}
                ],
                score=0.85,
                confidence=0.9,
                processing_time=0.1,
                explanation="语义检索结果"
            ),
            RetrievalResult(
                agent_type=RetrievalStrategy.KEYWORD,
                query="机器学习算法",
                results=[
                    {"id": "k1", "score": 0.7, "content": "关键词结果1"},
                    {"id": "s1", "score": 0.6, "content": "语义结果1"}  # 重复项
                ],
                score=0.65,
                confidence=0.7,
                processing_time=0.2,
                explanation="关键词检索结果"
            )
        ]
    
    def test_multi_agent_initialization(self, multi_agent_retriever):
        """测试多代理系统初始化"""
        assert len(multi_agent_retriever.agents) == 3
        assert RetrievalStrategy.SEMANTIC in multi_agent_retriever.agents
        assert RetrievalStrategy.KEYWORD in multi_agent_retriever.agents
        assert RetrievalStrategy.STRUCTURED in multi_agent_retriever.agents
        
        # 验证策略权重
        assert RetrievalStrategy.SEMANTIC in multi_agent_retriever.strategy_weights
        assert multi_agent_retriever.strategy_weights[RetrievalStrategy.SEMANTIC] == 1.0
    
    def test_select_strategies_automatic(self, multi_agent_retriever, sample_query_analysis):
        """测试自动策略选择"""
        strategies = multi_agent_retriever.select_strategies(sample_query_analysis)
        
        assert len(strategies) > 0
        assert all(isinstance(item, tuple) for item in strategies)
        assert all(len(item) == 2 for item in strategies)  # (strategy, score)
        assert all(item[1] > 0.3 for item in strategies)  # 分数阈值过滤
        
        # 验证排序（分数从高到低）
        scores = [score for _, score in strategies]
        assert scores == sorted(scores, reverse=True)
    
    def test_select_strategies_code_query(self, multi_agent_retriever):
        """测试代码查询的策略选择"""
        code_query = QueryAnalysis(
            query_text="Python函数实现",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.5,
            entities=["Python"],
            keywords=["Python", "函数", "实现"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        strategies = multi_agent_retriever.select_strategies(code_query)
        
        # 语义检索应该有较高适用性
        strategy_dict = dict(strategies)
        assert RetrievalStrategy.SEMANTIC in strategy_dict
        assert strategy_dict[RetrievalStrategy.SEMANTIC] > 0.5
    
    @pytest.mark.asyncio
    async def test_retrieve_parallel_execution(self, multi_agent_retriever, sample_query_analysis):
        """测试并行检索执行"""
        # Mock各个代理的检索方法
        for agent in multi_agent_retriever.agents.values():
            agent.retrieve = AsyncMock(return_value=RetrievalResult(
                agent_type=agent.strategy,
                query=sample_query_analysis.query_text,
                results=[{"id": f"{agent.strategy}_test", "score": 0.8, "content": "test"}],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            ))
        
        results = await multi_agent_retriever.retrieve(
            sample_query_analysis,
            enable_parallel=True
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        
        # 验证所有代理都被调用
        for agent in multi_agent_retriever.agents.values():
            if any(strategy == agent.strategy for strategy, _ in 
                   multi_agent_retriever.select_strategies(sample_query_analysis)):
                agent.retrieve.assert_called()
    
    @pytest.mark.asyncio
    async def test_retrieve_serial_execution(self, multi_agent_retriever, sample_query_analysis):
        """测试串行检索执行"""
        # Mock各个代理的检索方法
        for agent in multi_agent_retriever.agents.values():
            agent.retrieve = AsyncMock(return_value=RetrievalResult(
                agent_type=agent.strategy,
                query=sample_query_analysis.query_text,
                results=[{"id": f"{agent.strategy}_test", "score": 0.8, "content": "test"}],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            ))
        
        results = await multi_agent_retriever.retrieve(
            sample_query_analysis,
            enable_parallel=False
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_retrieve_with_specific_strategies(self, multi_agent_retriever, sample_query_analysis):
        """测试指定策略检索"""
        # Mock语义代理
        semantic_agent = multi_agent_retriever.agents[RetrievalStrategy.SEMANTIC]
        semantic_agent.retrieve = AsyncMock(return_value=RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query=sample_query_analysis.query_text,
            results=[{"id": "semantic_test", "score": 0.9, "content": "semantic result"}],
            score=0.9,
            confidence=0.9,
            processing_time=0.1
        ))
        
        results = await multi_agent_retriever.retrieve(
            sample_query_analysis,
            strategies=[RetrievalStrategy.SEMANTIC]
        )
        
        assert len(results) == 1
        assert results[0].agent_type == RetrievalStrategy.SEMANTIC
        semantic_agent.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_agent_exception_handling(self, multi_agent_retriever, sample_query_analysis):
        """测试代理异常处理"""
        # Mock一个代理抛出异常
        semantic_agent = multi_agent_retriever.agents[RetrievalStrategy.SEMANTIC]
        semantic_agent.retrieve = AsyncMock(side_effect=Exception("Agent Error"))
        
        # Mock其他代理正常工作
        keyword_agent = multi_agent_retriever.agents[RetrievalStrategy.KEYWORD]
        keyword_agent.retrieve = AsyncMock(return_value=RetrievalResult(
            agent_type=RetrievalStrategy.KEYWORD,
            query=sample_query_analysis.query_text,
            results=[{"id": "keyword_test", "score": 0.7, "content": "keyword result"}],
            score=0.7,
            confidence=0.7,
            processing_time=0.1
        ))
        
        results = await multi_agent_retriever.retrieve(sample_query_analysis)
        
        # 应该只返回正常工作的代理结果
        assert len(results) >= 1
        assert all(r.agent_type != RetrievalStrategy.SEMANTIC for r in results)
    
    def test_fuse_results_weighted_score(self, multi_agent_retriever, sample_retrieval_results):
        """测试加权分数融合"""
        fused = multi_agent_retriever.fuse_results(
            sample_retrieval_results,
            fusion_method="weighted_score",
            max_results=5
        )
        
        assert isinstance(fused, list)
        assert len(fused) <= 5
        assert len(fused) == 3  # 去重后应该有3个唯一结果
        
        # 验证融合分数存在
        for item in fused:
            assert "fused_score" in item
            assert "agent_type" in item
            assert "agent_confidence" in item
        
        # 验证按融合分数降序排列
        scores = [item["fused_score"] for item in fused]
        assert scores == sorted(scores, reverse=True)
    
    def test_fuse_results_rank_fusion(self, multi_agent_retriever, sample_retrieval_results):
        """测试排名融合（RRF）"""
        fused = multi_agent_retriever.fuse_results(
            sample_retrieval_results,
            fusion_method="rank_fusion",
            max_results=10
        )
        
        assert isinstance(fused, list)
        assert len(fused) <= 10
        
        # 验证RRF分数
        for item in fused:
            assert "fused_score" in item
            assert item["fused_score"] > 0
    
    def test_fuse_results_confidence_weighted(self, multi_agent_retriever, sample_retrieval_results):
        """测试置信度加权融合"""
        fused = multi_agent_retriever.fuse_results(
            sample_retrieval_results,
            fusion_method="confidence_weighted",
            max_results=10
        )
        
        assert isinstance(fused, list)
        
        # 验证归一化权重
        for item in fused:
            assert "normalized_weight" in item
            assert 0 <= item["normalized_weight"] <= 1
    
    def test_fuse_results_empty_input(self, multi_agent_retriever):
        """测试空输入融合"""
        fused = multi_agent_retriever.fuse_results([])
        
        assert fused == []
    
    def test_deduplicate_results(self, multi_agent_retriever):
        """测试结果去重"""
        items = [
            {"id": "1", "content": "相同内容", "file_path": "/test.txt"},
            {"id": "1", "content": "相同内容", "file_path": "/test.txt"},  # 完全重复（相同ID）
            {"id": "3", "content": "不同内容", "file_path": "/other.txt"}
        ]
        
        unique_items = multi_agent_retriever._deduplicate_results(items)
        
        assert len(unique_items) == 2  # 去重后只有2个
        assert unique_items[0]["id"] == "1"  # 保留第一个
        assert unique_items[1]["id"] == "3"
    
    def test_get_retrieval_explanation(self, multi_agent_retriever, sample_query_analysis, sample_retrieval_results):
        """测试检索过程解释生成"""
        fused_results = [{"id": "test1"}, {"id": "test2"}]
        
        explanation = multi_agent_retriever.get_retrieval_explanation(
            sample_query_analysis,
            sample_retrieval_results,
            fused_results
        )
        
        assert isinstance(explanation, str)
        assert "查询分析" in explanation
        assert sample_query_analysis.intent_type.value in explanation
        assert "使用策略" in explanation
        assert "最终融合" in explanation
        assert str(len(fused_results)) in explanation
        
        # 应该包含各代理的信息
        for result in sample_retrieval_results:
            assert result.agent_type.value in explanation
    
    def test_get_performance_summary(self, multi_agent_retriever):
        """测试性能摘要获取"""
        # 更新一些性能统计
        for agent in multi_agent_retriever.agents.values():
            agent.performance_stats["total_queries"] = 5
            agent.performance_stats["avg_response_time"] = 0.15
            agent.performance_stats["success_rate"] = 0.8
        
        summary = multi_agent_retriever.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 3  # 三个代理
        
        for strategy_name, agent_info in summary.items():
            assert "name" in agent_info
            assert "stats" in agent_info
            assert agent_info["stats"]["total_queries"] == 5
            assert agent_info["stats"]["success_rate"] == 0.8


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_multi_agent_workflow(self):
        """测试完整的多代理检索工作流"""
        # 创建查询分析
        query_analysis = QueryAnalysis(
            query_text="Python机器学习库的使用方法",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.7,
            entities=["Python", "机器学习"],
            keywords=["Python", "机器学习", "库", "使用", "方法"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        # Mock所有依赖
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever') as mock_semantic_cls:
            with patch('src.ai.agentic_rag.retrieval_agents.get_db_session') as mock_db:
                # Mock语义检索结果
                mock_semantic = mock_semantic_cls.return_value
                mock_semantic.search = AsyncMock(return_value=[
                    {
                        "id": "python_ml_1",
                        "score": 0.9,
                        "content": "Python机器学习库scikit-learn使用教程",
                        "file_path": "/docs/sklearn_tutorial.md",
                        "file_type": "markdown",
                        "chunk_index": 0,
                        "metadata": {"collection": "code"}
                    },
                    {
                        "id": "python_ml_2",
                        "score": 0.85,
                        "content": "pandas数据处理与机器学习结合使用",
                        "file_path": "/docs/pandas_ml.md", 
                        "file_type": "markdown",
                        "chunk_index": 1,
                        "metadata": {"collection": "code"}
                    }
                ])
                
                # Mock数据库会话
                mock_db.return_value.__aenter__ = AsyncMock()
                mock_db.return_value.__aexit__ = AsyncMock()
                
                # 创建多代理检索器
                retriever = MultiAgentRetriever()
                
                # 执行检索
                results = await retriever.retrieve(
                    query_analysis,
                    limit=10,
                    enable_parallel=True
                )
                
                # 验证结果
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(isinstance(r, RetrievalResult) for r in results)
                
                # 验证包含不同类型的代理结果
                agent_types = {r.agent_type for r in results}
                assert RetrievalStrategy.SEMANTIC in agent_types
                
                # 结果融合
                fused_results = retriever.fuse_results(results, max_results=5)
                
                assert len(fused_results) <= 5
                assert all("fused_score" in item for item in fused_results)
                
                # 生成解释
                explanation = retriever.get_retrieval_explanation(
                    query_analysis, results, fused_results
                )
                
                assert isinstance(explanation, str)
                assert len(explanation) > 0
                
                # 性能统计
                performance = retriever.get_performance_summary()
                assert isinstance(performance, dict)
                assert len(performance) >= 1  # 至少有一个代理执行了检索
    
    @pytest.mark.asyncio
    async def test_strategy_selection_adaptation(self):
        """测试策略选择的自适应性"""
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever'):
            retriever = MultiAgentRetriever()
            
            # 测试不同类型查询的策略选择
            test_cases = [
                {
                    "query": QueryAnalysis(
                        query_text="什么是深度学习",
                        intent_type=QueryIntent.FACTUAL,
                        confidence=0.8,
                        complexity_score=0.3,
                        entities=["深度学习"],
                        keywords=["什么是", "深度学习"],
                        domain="技术",
                        sentiment="neutral",
                        language="zh"
                    ),
                    "expected_strategies": [RetrievalStrategy.SEMANTIC]
                },
                {
                    "query": QueryAnalysis(
                        query_text="如何实现快速排序算法",
                        intent_type=QueryIntent.PROCEDURAL,
                        confidence=0.8,
                        complexity_score=0.6,
                        entities=["快速排序"],
                        keywords=["如何", "实现", "快速", "排序", "算法"],
                        domain="技术",
                        sentiment="neutral",
                        language="zh"
                    ),
                    "expected_strategies": [RetrievalStrategy.KEYWORD, RetrievalStrategy.SEMANTIC]
                }
            ]
            
            for case in test_cases:
                strategies = retriever.select_strategies(case["query"])
                strategy_types = [s for s, _ in strategies]
                
                # 验证包含期望的策略
                for expected in case["expected_strategies"]:
                    assert expected in strategy_types, f"Missing {expected} for query: {case['query'].query_text}"
    
    @pytest.mark.asyncio 
    async def test_error_resilience_and_fallback(self):
        """测试错误恢复能力和后备机制"""
        query_analysis = QueryAnalysis(
            query_text="测试查询",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["测试"],
            keywords=["测试", "查询"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        with patch('src.ai.agentic_rag.retrieval_agents.SemanticRetriever') as mock_semantic_cls:
            with patch('src.ai.agentic_rag.retrieval_agents.get_db_session') as mock_db:
                # Mock语义检索失败
                mock_semantic = mock_semantic_cls.return_value
                mock_semantic.search = AsyncMock(side_effect=Exception("Network Error"))
                
                # Mock数据库连接失败
                mock_db.side_effect = Exception("Database Error")
                
                retriever = MultiAgentRetriever()
                
                # 即使部分代理失败，系统应该仍然能返回结果
                results = await retriever.retrieve(query_analysis)
                
                # 至少关键词代理应该能工作（它有自己的后备机制）
                assert isinstance(results, list)
                
                # 验证错误被正确处理而不是抛出异常
                if results:
                    for result in results:
                        assert isinstance(result, RetrievalResult)
                        # 失败的代理应该返回空结果但不抛出异常
                        if result.score == 0.0:
                            assert "失败" in result.explanation or "Error" in result.explanation


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])