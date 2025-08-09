"""
查询扩展器单元测试
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.ai.agentic_rag.query_expander import (
    QueryExpander, ExpandedQuery, ExpansionStrategy
)
from src.ai.agentic_rag.query_analyzer import (
    QueryAnalysis, QueryIntent
)


class TestQueryExpander:
    """查询扩展器测试"""
    
    @pytest.fixture
    def expander(self):
        """创建查询扩展器实例"""
        expander = QueryExpander()
        # 创建一个mock客户端
        mock_client = MagicMock()
        expander.client = mock_client
        return expander
    
    @pytest.fixture
    def sample_query_analysis(self):
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
    def mock_openai_response(self):
        """Mock OpenAI API响应"""
        def create_mock_response(content):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = content
            return mock_response
        return create_mock_response
    
    def test_build_synonym_dict(self, expander):
        """测试同义词词典构建"""
        synonym_dict = expander._synonym_dict
        
        assert isinstance(synonym_dict, dict)
        assert len(synonym_dict) > 0
        assert "机器学习" in synonym_dict
        assert "ML" in synonym_dict["机器学习"]
        assert "实现" in synonym_dict
        assert "开发" in synonym_dict["实现"]
    
    def test_build_domain_terms(self, expander):
        """测试领域术语映射构建"""
        domain_terms = expander._domain_terms
        
        assert isinstance(domain_terms, dict)
        assert "技术" in domain_terms
        assert "业务" in domain_terms
        assert "学术" in domain_terms
        assert "编程" in domain_terms["技术"]
        assert "需求" in domain_terms["业务"]
    
    def test_select_strategies_factual(self, expander):
        """测试事实性查询的策略选择"""
        query_analysis = QueryAnalysis(
            query_text="什么是机器学习",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.3,
            entities=[],
            keywords=[],
            domain=None,
            sentiment="neutral",
            language="zh"
        )
        
        strategies = expander._select_strategies(query_analysis)
        
        assert ExpansionStrategy.SYNONYM in strategies
        assert ExpansionStrategy.SEMANTIC in strategies
        assert ExpansionStrategy.MULTILINGUAL in strategies
    
    def test_select_strategies_procedural(self, expander):
        """测试程序性查询的策略选择"""
        query_analysis = QueryAnalysis(
            query_text="如何实现机器学习算法",
            intent_type=QueryIntent.PROCEDURAL,
            confidence=0.8,
            complexity_score=0.7,
            entities=["机器学习"],
            keywords=["实现", "算法"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        strategies = expander._select_strategies(query_analysis)
        
        assert ExpansionStrategy.SYNONYM in strategies
        assert ExpansionStrategy.DECOMPOSITION in strategies
        assert ExpansionStrategy.CONTEXTUAL in strategies
    
    def test_select_strategies_complex_query(self, expander):
        """测试复杂查询的策略选择"""
        query_analysis = QueryAnalysis(
            query_text="详细解释深度学习神经网络架构设计原理并提供实现示例",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.8,  # 高复杂度
            entities=["深度学习", "神经网络"],
            keywords=["解释", "架构", "设计", "实现"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        strategies = expander._select_strategies(query_analysis)
        
        assert ExpansionStrategy.DECOMPOSITION in strategies  # 高复杂度应包含分解策略
    
    @pytest.mark.asyncio
    async def test_expand_synonyms(self, expander, sample_query_analysis):
        """测试同义词扩展"""
        result = await expander._expand_synonyms(sample_query_analysis)
        
        assert isinstance(result, ExpandedQuery)
        assert result.strategy == ExpansionStrategy.SYNONYM
        assert result.original_query == sample_query_analysis.query_text
        assert isinstance(result.expanded_queries, list)
        assert result.confidence >= 0.0
        assert result.explanation is not None
        
        # 应该包含一些扩展查询
        if result.expanded_queries:
            # 检查是否有同义词替换
            original_terms = ["机器学习", "算法", "实现"]
            synonyms_found = any(
                any(term not in expanded for term in original_terms)
                for expanded in result.expanded_queries
            )
            assert synonyms_found or len(result.expanded_queries) == 0  # 如果没有找到同义词则为空
    
    @pytest.mark.asyncio
    async def test_expand_semantic_with_openai(self, expander, sample_query_analysis, mock_openai_response):
        """测试使用OpenAI的语义扩展"""
        mock_content = '{"expanded_queries": ["ML模型开发", "人工智能算法构建", "机器学习代码编写"], "confidence": 0.85}'
        expander.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_content)
        )
        
        result = await expander._expand_semantic(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.SEMANTIC
        assert len(result.expanded_queries) == 3
        assert result.confidence == 0.85
        assert "ML模型开发" in result.expanded_queries
        
        # 验证API调用
        expander.client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_expand_semantic_fallback(self, expander, sample_query_analysis):
        """测试语义扩展失败时的后备方案"""
        # Mock OpenAI调用失败
        expander.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await expander._expand_semantic(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.SEMANTIC
        assert isinstance(result.expanded_queries, list)
        assert result.confidence < 0.5  # 后备方案置信度应较低
        assert "后备方案" in result.explanation
    
    @pytest.mark.asyncio
    async def test_rewrite_contextual_no_history(self, expander, sample_query_analysis):
        """测试无历史上下文的上下文改写"""
        result = await expander._rewrite_contextual(sample_query_analysis, None)
        
        assert result.strategy == ExpansionStrategy.CONTEXTUAL
        assert len(result.expanded_queries) == 0
        assert result.confidence == 0.0
        assert "缺少上下文历史" in result.explanation
    
    @pytest.mark.asyncio
    async def test_rewrite_contextual_with_history(self, expander, sample_query_analysis, mock_openai_response):
        """测试有历史上下文的上下文改写"""
        context_history = ["我们讨论了神经网络", "特别是深度学习", "现在想了解实现方法"]
        
        mock_content = '{"rewritten_queries": ["基于神经网络和深度学习的机器学习算法实现", "深度学习机器学习算法的具体实现方法"], "confidence": 0.8}'
        expander.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_content)
        )
        
        result = await expander._rewrite_contextual(sample_query_analysis, context_history)
        
        assert result.strategy == ExpansionStrategy.CONTEXTUAL
        assert len(result.expanded_queries) == 2
        assert result.confidence == 0.8
        assert "基于对话上下文" in result.explanation
    
    @pytest.mark.asyncio
    async def test_rewrite_contextual_fallback(self, expander, sample_query_analysis):
        """测试上下文改写的后备方案"""
        context_history = ["神经网络基础", "深度学习原理"]
        
        # Mock OpenAI调用失败
        expander.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await expander._rewrite_contextual(sample_query_analysis, context_history)
        
        assert result.strategy == ExpansionStrategy.CONTEXTUAL
        assert isinstance(result.expanded_queries, list)
        assert result.confidence < 0.5
        assert "后备方案" in result.explanation
    
    @pytest.mark.asyncio
    async def test_decompose_query(self, expander, sample_query_analysis, mock_openai_response):
        """测试查询分解"""
        mock_content = '{"sub_questions": ["什么是机器学习算法", "常用的机器学习算法有哪些", "如何选择合适的算法", "算法实现的技术栈"], "confidence": 0.9}'
        expander.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_content)
        )
        
        result = await expander._decompose_query(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.DECOMPOSITION
        assert len(result.expanded_queries) == 4
        assert len(result.sub_questions) == 4
        assert result.confidence == 0.9
        assert "分解为多个具体的子问题" in result.explanation
    
    @pytest.mark.asyncio
    async def test_decompose_query_fallback(self, expander, sample_query_analysis):
        """测试查询分解的后备方案"""
        # Mock OpenAI调用失败
        expander.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await expander._decompose_query(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.DECOMPOSITION
        assert isinstance(result.expanded_queries, list)
        assert isinstance(result.sub_questions, list)
        assert result.confidence < 0.5
        assert "后备方案" in result.explanation
        
        # 应该包含基于关键词和实体的子问题
        all_content = " ".join(result.expanded_queries)
        assert any(keyword in all_content for keyword in ["什么是", "关于", "详细信息"])
    
    @pytest.mark.asyncio
    async def test_expand_multilingual_zh_to_en(self, expander, sample_query_analysis, mock_openai_response):
        """测试中文到英文的多语言扩展"""
        mock_content = '{"translations": ["machine learning algorithm implementation", "ML algorithm development", "AI algorithm coding"], "confidence": 0.85}'
        expander.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_content)
        )
        
        result = await expander._expand_multilingual(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.MULTILINGUAL
        assert len(result.expanded_queries) == 3
        assert result.confidence == 0.85
        assert result.language_variants is not None
        assert "英文" in result.language_variants
        assert "machine learning" in result.expanded_queries[0].lower()
    
    @pytest.mark.asyncio
    async def test_expand_multilingual_en_to_zh(self, expander, mock_openai_response):
        """测试英文到中文的多语言扩展"""
        english_query_analysis = QueryAnalysis(
            query_text="How to implement machine learning algorithms",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.6,
            entities=["machine learning", "algorithms"],
            keywords=["implement", "algorithms"],
            domain="技术",
            sentiment="neutral",
            language="en"
        )
        
        mock_content = '{"translations": ["如何实现机器学习算法", "机器学习算法的实现方法", "机器学习算法开发"], "confidence": 0.85}'
        expander.client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_content)
        )
        
        result = await expander._expand_multilingual(english_query_analysis)
        
        assert result.strategy == ExpansionStrategy.MULTILINGUAL
        assert len(result.expanded_queries) == 3
        assert "如何实现" in result.expanded_queries[0]
    
    @pytest.mark.asyncio
    async def test_expand_multilingual_fallback(self, expander, sample_query_analysis):
        """测试多语言扩展的后备方案"""
        # Mock OpenAI调用失败
        expander.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        result = await expander._expand_multilingual(sample_query_analysis)
        
        assert result.strategy == ExpansionStrategy.MULTILINGUAL
        assert result.confidence < 0.5
        assert "后备方案" in result.explanation
    
    @pytest.mark.asyncio
    async def test_expand_query_multiple_strategies(self, expander, sample_query_analysis, mock_openai_response):
        """测试使用多种策略扩展查询"""
        # Mock所有OpenAI调用
        mock_responses = [
            '{"expanded_queries": ["ML算法实现", "人工智能算法开发"], "confidence": 0.8}',
            '{"sub_questions": ["什么是机器学习", "如何选择算法"], "confidence": 0.9}',
            '{"translations": ["machine learning implementation"], "confidence": 0.7}'
        ]
        
        expander.client.chat.completions.create = AsyncMock(
            side_effect=[mock_openai_response(response) for response in mock_responses]
        )
        
        strategies = [ExpansionStrategy.SEMANTIC, ExpansionStrategy.DECOMPOSITION, ExpansionStrategy.MULTILINGUAL]
        results = await expander.expand_query(sample_query_analysis, strategies=strategies)
        
        assert len(results) >= 1  # 至少有同义词扩展（不需要OpenAI）
        
        # 检查策略类型
        strategy_types = {result.strategy for result in results}
        assert ExpansionStrategy.SYNONYM in strategy_types  # 同义词扩展总是存在
    
    @pytest.mark.asyncio
    async def test_expand_query_auto_strategy_selection(self, expander, sample_query_analysis):
        """测试自动策略选择"""
        results = await expander.expand_query(sample_query_analysis)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # 应该至少包含同义词扩展
        strategies = {result.strategy for result in results}
        assert ExpansionStrategy.SYNONYM in strategies
    
    def test_get_best_expansions_empty_list(self, expander):
        """测试空扩展结果列表"""
        best_expansions = expander.get_best_expansions([])
        
        assert isinstance(best_expansions, list)
        assert len(best_expansions) == 0
    
    def test_get_best_expansions_single_result(self, expander):
        """测试单个扩展结果"""
        expanded_query = ExpandedQuery(
            original_query="测试查询",
            expanded_queries=["扩展查询1", "扩展查询2"],
            strategy=ExpansionStrategy.SYNONYM,
            confidence=0.8
        )
        
        best_expansions = expander.get_best_expansions([expanded_query])
        
        assert len(best_expansions) == 2
        assert "扩展查询1" in best_expansions
        assert "扩展查询2" in best_expansions
    
    def test_get_best_expansions_multiple_results_with_duplicates(self, expander):
        """测试多个扩展结果包含重复项"""
        results = [
            ExpandedQuery(
                original_query="测试查询",
                expanded_queries=["查询A", "查询B"],
                strategy=ExpansionStrategy.SYNONYM,
                confidence=0.8
            ),
            ExpandedQuery(
                original_query="测试查询",
                expanded_queries=["查询A", "查询C"],  # 查询A重复
                strategy=ExpansionStrategy.SEMANTIC,
                confidence=0.9  # 更高置信度
            )
        ]
        
        best_expansions = expander.get_best_expansions(results)
        
        # 应该去重，且保留高置信度版本
        assert len(best_expansions) == 3
        assert "查询A" in best_expansions
        assert "查询B" in best_expansions
        assert "查询C" in best_expansions
        
        # 查询A应该是来自置信度更高的结果
        # 这个测试主要验证去重逻辑
    
    def test_get_best_expansions_max_results_limit(self, expander):
        """测试最大结果数量限制"""
        expanded_query = ExpandedQuery(
            original_query="测试查询",
            expanded_queries=[f"查询{i}" for i in range(15)],  # 15个扩展查询
            strategy=ExpansionStrategy.SYNONYM,
            confidence=0.8
        )
        
        best_expansions = expander.get_best_expansions([expanded_query], max_results=5)
        
        assert len(best_expansions) <= 5