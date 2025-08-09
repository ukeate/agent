"""
查询分析器单元测试
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.ai.agentic_rag.query_analyzer import (
    QueryAnalyzer, QueryAnalysis, QueryIntent, QueryContext
)


class TestQueryAnalyzer:
    """查询分析器测试"""
    
    @pytest.fixture
    def analyzer(self):
        """创建查询分析器实例"""
        with patch('src.ai.agentic_rag.query_analyzer.get_openai_client') as mock_get_client:
            # Mock OpenAI client instance
            mock_client = AsyncMock()
            mock_get_client.return_value = mock_client
            analyzer = QueryAnalyzer()
            analyzer.client = mock_client  # 直接设置client
            return analyzer
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API响应"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"intent": "factual", "confidence": 0.85, "reasoning": "查询具体信息"}'
        return mock_response
    
    def test_preprocess_query(self, analyzer):
        """测试查询预处理"""
        # 测试空白字符处理
        assert analyzer._preprocess_query("  hello   world  ") == "hello world"
        
        # 测试标点符号统一
        assert analyzer._preprocess_query("你好？这是什么！") == "你好?这是什么!"
        
        # 测试空字符串
        assert analyzer._preprocess_query("") == ""
    
    def test_rule_based_intent_classification(self, analyzer):
        """测试基于规则的意图分类"""
        # 测试代码相关查询
        intent, confidence = analyzer._rule_based_intent_classification("这个Python函数有bug")
        assert intent == QueryIntent.CODE
        assert 0.0 <= confidence <= 1.0
        
        # 测试程序性查询
        intent, confidence = analyzer._rule_based_intent_classification("如何实现这个功能")
        assert intent == QueryIntent.PROCEDURAL
        assert 0.0 <= confidence <= 1.0
        
        # 测试创造性查询
        intent, confidence = analyzer._rule_based_intent_classification("帮我设计一个方案")
        assert intent == QueryIntent.CREATIVE
        assert 0.0 <= confidence <= 1.0
        
        # 测试事实性查询
        intent, confidence = analyzer._rule_based_intent_classification("什么是机器学习")
        assert intent == QueryIntent.FACTUAL
        assert 0.0 <= confidence <= 1.0
        
        # 测试无明确关键词
        intent, confidence = analyzer._rule_based_intent_classification("随便说说")
        assert intent == QueryIntent.EXPLORATORY
        assert confidence == 0.4
    
    def test_assess_complexity(self, analyzer):
        """测试复杂度评估"""
        # 简单查询
        simple_complexity = analyzer._assess_complexity("你好")
        assert 0.0 <= simple_complexity <= 1.0
        
        # 复杂查询
        complex_query = "请详细解释机器学习中的神经网络架构设计原理，并提供Python实现示例，包括前向传播和反向传播算法"
        complex_complexity = analyzer._assess_complexity(complex_query)
        assert 0.0 <= complex_complexity <= 1.0
        
        # 包含技术术语
        tech_query = "使用FastAPI和PostgreSQL实现RESTful API接口"
        tech_complexity = analyzer._assess_complexity(tech_query)
        assert 0.0 <= tech_complexity <= 1.0
        
        # 复杂查询应该比简单查询有更高的复杂度
        assert complex_complexity > simple_complexity
    
    def test_extract_entities(self, analyzer):
        """测试实体提取"""
        query = "使用Python和FastAPI开发API接口，文件路径是src/main.py，访问https://example.com"
        entities = analyzer._extract_entities(query)
        
        # 验证实体提取基本功能工作
        assert len(entities) > 0
        
        # URL应该被提取
        assert "https://example.com" in entities
        
        # 文件扩展名应该被提取（从文件路径模式）
        assert "py" in entities
        
        # 测试另一个简单的英文单词实体提取
        english_query = "use Python FastAPI and REST API development"
        english_entities = analyzer._extract_entities(english_query)
        # 在纯英文环境下应该能匹配到技术术语
        assert len(english_entities) > 0
    
    def test_extract_keywords(self, analyzer):
        """测试关键词提取"""
        query = "机器学习算法实现和优化方法研究"
        keywords = analyzer._extract_keywords(query)
        
        # 应该包含主要关键词，排除停用词
        assert any(kw in keywords for kw in ["机器", "学习", "算法", "实现", "优化", "方法", "研究"])
        assert "的" not in keywords  # 停用词应被过滤
        
        # 检查去重
        query_with_duplicates = "algorithm function algorithm function"
        keywords = analyzer._extract_keywords(query_with_duplicates)
        assert len(keywords) == len(set(keywords))  # 无重复
    
    def test_identify_domain(self, analyzer):
        """测试领域识别"""
        # 技术领域
        tech_query = "Python代码开发和数据库设计"
        domain = asyncio.run(analyzer._identify_domain(tech_query))
        assert domain == "技术"
        
        # 商业领域
        business_query = "市场营销策略和销售管理"
        domain = asyncio.run(analyzer._identify_domain(business_query))
        assert domain == "商业"
        
        # 无明确领域
        neutral_query = "今天天气很好"
        domain = asyncio.run(analyzer._identify_domain(neutral_query))
        assert domain is None
    
    def test_analyze_sentiment(self, analyzer):
        """测试情感分析"""
        # 积极情感
        positive_query = "这个功能很好用，我很满意"
        sentiment = analyzer._analyze_sentiment(positive_query)
        assert sentiment == "positive"
        
        # 消极情感
        negative_query = "这个系统有问题，太差了"
        sentiment = analyzer._analyze_sentiment(negative_query)
        assert sentiment == "negative"
        
        # 中性情感
        neutral_query = "请说明一下这个功能"
        sentiment = analyzer._analyze_sentiment(neutral_query)
        assert sentiment == "neutral"
    
    def test_detect_language(self, analyzer):
        """测试语言检测"""
        # 中文
        chinese_query = "这是一个中文查询"
        language = analyzer._detect_language(chinese_query)
        assert language == "zh"
        
        # 英文
        english_query = "This is an English query"
        language = analyzer._detect_language(english_query)
        assert language == "en"
        
        # 混合（中文占多数）
        mixed_query = "这是一个中文mixed查询"  # 中文字符更多
        language = analyzer._detect_language(mixed_query)
        assert language == "zh"
    
    @pytest.mark.asyncio
    async def test_classify_intent_with_openai(self, analyzer, mock_openai_response):
        """测试使用OpenAI的意图分类"""
        analyzer.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        
        query = "什么是机器学习"
        intent, confidence = await analyzer._classify_intent(query)
        
        assert intent == QueryIntent.FACTUAL
        assert confidence == 0.85
        
        # 验证API调用
        analyzer.client.chat.completions.create.assert_called_once()
        call_args = analyzer.client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"
        assert call_args[1]["temperature"] == 0.1
    
    @pytest.mark.asyncio
    async def test_classify_intent_fallback(self, analyzer):
        """测试意图分类失败时的后备方案"""
        # Mock OpenAI调用失败
        analyzer.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        
        query = "如何实现这个功能"
        intent, confidence = await analyzer._classify_intent(query)
        
        # 应该使用规则方法作为后备
        assert intent == QueryIntent.PROCEDURAL
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_query_complete(self, analyzer, mock_openai_response):
        """测试完整的查询分析流程"""
        analyzer.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        
        query = "如何使用Python实现机器学习算法"
        context_history = ["之前我们讨论了数据预处理"]
        
        result = await analyzer.analyze_query(query, context_history)
        
        # 验证返回结果类型
        assert isinstance(result, QueryAnalysis)
        assert result.query_text == query
        assert isinstance(result.intent_type, QueryIntent)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.complexity_score <= 1.0
        assert isinstance(result.entities, list)
        assert isinstance(result.keywords, list)
        assert result.language in ["zh", "en"]
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_analyze_query_with_context(self, analyzer, mock_openai_response):
        """测试带上下文的查询分析"""
        analyzer.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        
        query = "继续上一个话题"
        context_history = ["我们讨论机器学习", "特别是神经网络"]
        
        result = await analyzer.analyze_query(query, context_history)
        
        # 验证上下文被传递给LLM
        call_args = analyzer.client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "对话历史" in user_message
        assert "机器学习" in user_message


class TestQueryContext:
    """查询上下文管理器测试"""
    
    @pytest.fixture
    def context_manager(self):
        """创建上下文管理器"""
        return QueryContext(max_history=3)
    
    @pytest.fixture
    def sample_analysis(self):
        """创建示例查询分析结果"""
        return QueryAnalysis(
            query_text="测试查询",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=["测试"],
            keywords=["查询"],
            domain="技术",
            sentiment="neutral"
        )
    
    def test_add_query(self, context_manager, sample_analysis):
        """测试添加查询到历史"""
        context_manager.add_query(sample_analysis)
        assert len(context_manager.query_history) == 1
        assert context_manager.query_history[0] == sample_analysis
    
    def test_max_history_limit(self, context_manager, sample_analysis):
        """测试历史记录数量限制"""
        # 添加超过限制的查询
        for i in range(5):
            analysis = QueryAnalysis(
                query_text=f"查询{i}",
                intent_type=QueryIntent.FACTUAL,
                confidence=0.8,
                complexity_score=0.5,
                entities=[],
                keywords=[],
                domain=None,
                sentiment="neutral"
            )
            context_manager.add_query(analysis)
        
        # 应该只保留最近的3条记录
        assert len(context_manager.query_history) == 3
        assert context_manager.query_history[-1].query_text == "查询4"
        assert context_manager.query_history[0].query_text == "查询2"
    
    def test_get_context_for_query(self, context_manager):
        """测试获取查询上下文"""
        # 空历史
        context = context_manager.get_context_for_query("当前查询")
        assert context == []
        
        # 添加一些历史查询
        for i in range(3):
            analysis = QueryAnalysis(
                query_text=f"历史查询{i}",
                intent_type=QueryIntent.FACTUAL,
                confidence=0.8,
                complexity_score=0.5,
                entities=[],
                keywords=[],
                domain=None,
                sentiment="neutral"
            )
            context_manager.add_query(analysis)
        
        context = context_manager.get_context_for_query("当前查询")
        assert len(context) == 3
        assert context == ["历史查询0", "历史查询1", "历史查询2"]
    
    def test_get_session_summary_empty(self, context_manager):
        """测试空会话摘要"""
        summary = context_manager.get_session_summary()
        assert summary["total_queries"] == 0
        assert summary["session_duration"] == 0
    
    def test_get_session_summary_with_data(self, context_manager):
        """测试有数据的会话摘要"""
        # 添加不同类型的查询
        analyses = [
            QueryAnalysis("查询1", QueryIntent.FACTUAL, 0.8, 0.3, [], [], None, "neutral", "zh"),
            QueryAnalysis("查询2", QueryIntent.CODE, 0.9, 0.7, [], [], None, "neutral", "en"),
            QueryAnalysis("查询3", QueryIntent.FACTUAL, 0.7, 0.5, [], [], None, "neutral", "zh"),
        ]
        
        for analysis in analyses:
            context_manager.add_query(analysis)
        
        summary = context_manager.get_session_summary()
        
        assert summary["total_queries"] == 3
        assert summary["session_duration"] > 0
        assert summary["intent_distribution"]["factual"] == 2
        assert summary["intent_distribution"]["code"] == 1
        assert summary["average_complexity"] == 0.5
        assert set(summary["languages"]) == {"zh", "en"}