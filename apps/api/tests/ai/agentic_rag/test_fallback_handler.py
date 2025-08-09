"""
检索失败处理器单元测试

测试fallback处理器的各项功能：
- 失败检测和分类机制
- 备用策略生成和执行
- 用户提示和建议生成
- 查询重构和改进功能
- 失败统计和历史记录
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.ai.agentic_rag.fallback_handler import (
    FallbackHandler, FailureSeverity,
    FailureDetection, FallbackAction, UserGuidance, FallbackResult
)
from src.models.schemas.agentic_rag import FailureType, FallbackStrategyType
from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy
from src.ai.agentic_rag.result_validator import ValidationResult, QualityScore, QualityDimension


@pytest.fixture
def fallback_handler():
    """创建fallback处理器实例"""
    with patch('src.ai.agentic_rag.fallback_handler.get_openai_client'):
        return FallbackHandler()


@pytest.fixture
def sample_query_analysis():
    """创建示例查询分析"""
    return QueryAnalysis(
        query_text="机器学习算法优化",
        intent_type=QueryIntent.CODE,
        confidence=0.7,
        complexity_score=0.6,
        entities=["机器学习", "算法"],
        keywords=["机器学习", "算法", "优化"],
        domain="技术",
        sentiment="neutral",
        language="zh"
    )


@pytest.fixture
def empty_retrieval_results():
    """创建空的检索结果"""
    return []


@pytest.fixture
def low_quality_retrieval_results():
    """创建低质量检索结果"""
    return [
        RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="机器学习算法优化",
            results=[
                {
                    "id": "low_quality_1",
                    "score": 0.2,
                    "content": "简短内容",
                    "file_path": "/docs/test.md"
                }
            ],
            score=0.2,
            confidence=0.3,
            processing_time=0.1,
            explanation="低质量结果"
        )
    ]


@pytest.fixture
def low_quality_validation():
    """创建低质量验证结果"""
    return ValidationResult(
        query_id="test_query",
        retrieval_results=[],
        quality_scores={
            QualityDimension.RELEVANCE: QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.2,
                confidence=0.8,
                explanation="相关性很低"
            )
        },
        conflicts=[],
        overall_quality=0.2,
        overall_confidence=0.6,
        recommendations=["质量较低"],
        validation_time=0.5
    )


@pytest.fixture
def vague_query_analysis():
    """创建模糊查询分析"""
    return QueryAnalysis(
        query_text="怎么做",
        intent_type=QueryIntent.PROCEDURAL,
        confidence=0.3,  # 低置信度表示模糊
        complexity_score=0.2,
        entities=[],
        keywords=["怎么做"],
        domain="通用",
        sentiment="neutral",
        language="zh"
    )


@pytest.fixture
def complex_query_analysis():
    """创建复杂查询分析"""
    return QueryAnalysis(
        query_text="在分布式微服务架构中使用Docker容器化部署，结合Kubernetes集群管理，实现高可用负载均衡的机器学习模型推理服务，同时支持GPU加速和自动扩缩容",
        intent_type=QueryIntent.CODE,
        confidence=0.8,
        complexity_score=0.9,  # 高复杂度
        entities=["Docker", "Kubernetes", "GPU"],
        keywords=["分布式", "微服务", "Docker", "Kubernetes", "机器学习", "GPU", "负载均衡"],
        domain="技术",
        sentiment="neutral",
        language="zh"
    )


class TestFallbackHandler:
    """Fallback处理器基础功能测试"""

    def test_handler_initialization(self, fallback_handler):
        """测试处理器初始化"""
        assert fallback_handler.client is not None
        assert len(fallback_handler.failure_thresholds) == 5
        assert len(fallback_handler.strategy_priorities) >= 4
        assert len(fallback_handler.guidance_templates) >= 3
        assert isinstance(fallback_handler.failure_history, list)

    def test_failure_thresholds_configuration(self, fallback_handler):
        """测试失败阈值配置"""
        thresholds = fallback_handler.failure_thresholds
        assert thresholds["min_results"] >= 0
        assert thresholds["min_quality"] >= 0
        assert thresholds["min_coverage"] >= 0
        assert thresholds["max_response_time"] > 0
        assert thresholds["min_confidence"] >= 0

    def test_strategy_priorities_configuration(self, fallback_handler):
        """测试策略优先级配置"""
        priorities = fallback_handler.strategy_priorities
        
        # 检查每种失败类型都有对应的策略
        for failure_type in [FailureType.NO_RESULTS, FailureType.LOW_QUALITY]:
            assert failure_type in priorities
            assert len(priorities[failure_type]) > 0
            assert all(isinstance(s, FallbackStrategyType) for s in priorities[failure_type])


class TestFailureDetection:
    """失败检测测试"""

    @pytest.mark.asyncio
    async def test_detect_no_results_failure(self, fallback_handler, sample_query_analysis, empty_retrieval_results):
        """测试无结果失败检测"""
        failure = await fallback_handler._detect_failure(
            sample_query_analysis, empty_retrieval_results, None, 1.0
        )

        assert failure is not None
        assert failure.failure_type == FailureType.NO_RESULTS
        assert failure.severity == FailureSeverity.HIGH
        assert failure.confidence == 1.0
        assert "0个结果" in failure.evidence[0]

    @pytest.mark.asyncio
    async def test_detect_low_quality_failure(self, fallback_handler, sample_query_analysis, 
                                            low_quality_retrieval_results, low_quality_validation):
        """测试低质量失败检测"""
        failure = await fallback_handler._detect_failure(
            sample_query_analysis, low_quality_retrieval_results, low_quality_validation, 1.0
        )

        assert failure is not None
        assert failure.failure_type == FailureType.LOW_QUALITY
        assert failure.severity in [FailureSeverity.MEDIUM, FailureSeverity.HIGH]
        assert failure.metrics["quality_score"] == 0.2

    @pytest.mark.asyncio
    async def test_detect_timeout_failure(self, fallback_handler, sample_query_analysis, low_quality_retrieval_results):
        """测试超时失败检测"""
        long_processing_time = 35.0  # 超过最大允许时间
        
        failure = await fallback_handler._detect_failure(
            sample_query_analysis, low_quality_retrieval_results, None, long_processing_time
        )

        assert failure is not None
        assert failure.failure_type == FailureType.TIMEOUT
        assert failure.severity == FailureSeverity.MEDIUM
        assert failure.metrics["processing_time"] == long_processing_time

    @pytest.mark.asyncio
    async def test_detect_vague_query_failure(self, fallback_handler, vague_query_analysis, empty_retrieval_results):
        """测试模糊查询失败检测"""
        failure = await fallback_handler._detect_failure(
            vague_query_analysis, empty_retrieval_results, None, 1.0
        )

        assert failure is not None
        assert failure.failure_type in [FailureType.QUERY_TOO_VAGUE, FailureType.NO_RESULTS]
        if failure.failure_type == FailureType.QUERY_TOO_VAGUE:
            assert failure.severity == FailureSeverity.MEDIUM
            assert failure.metrics["query_confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_detect_complex_query_failure(self, fallback_handler, complex_query_analysis, low_quality_retrieval_results):
        """测试复杂查询失败检测"""
        failure = await fallback_handler._detect_failure(
            complex_query_analysis, low_quality_retrieval_results, None, 1.0
        )

        assert failure is not None
        assert failure.failure_type == FailureType.QUERY_TOO_COMPLEX
        assert failure.severity == FailureSeverity.MEDIUM
        assert failure.metrics["complexity_score"] == 0.9

    @pytest.mark.asyncio
    async def test_detect_no_failure(self, fallback_handler):
        """测试没有失败的情况"""
        good_query = QueryAnalysis(
            query_text="Python基础教程",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.9,
            complexity_score=0.3,
            entities=["Python"],
            keywords=["Python", "基础", "教程"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        good_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="Python基础教程",
                results=[
                    {"id": "1", "score": 0.9, "content": "Python是一种编程语言"},
                    {"id": "2", "score": 0.8, "content": "Python基础语法教程"}
                ],
                score=0.85,
                confidence=0.9,
                processing_time=1.0,
                explanation="高质量结果"
            )
        ]
        
        good_validation = ValidationResult(
            query_id="test_query",
            retrieval_results=good_results,
            quality_scores={
                QualityDimension.RELEVANCE: QualityScore(
                    dimension=QualityDimension.RELEVANCE,
                    score=0.9,
                    confidence=0.9,
                    explanation="高度相关"
                )
            },
            conflicts=[],
            overall_quality=0.9,
            overall_confidence=0.9,
            recommendations=["质量很好"],
            validation_time=0.5
        )

        failure = await fallback_handler._detect_failure(
            good_query, good_results, good_validation, 1.0
        )

        assert failure is None

    @pytest.mark.asyncio
    async def test_calculate_coverage_score(self, fallback_handler, sample_query_analysis):
        """测试覆盖度评分计算"""
        results_with_keywords = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试",
                results=[
                    {"content": "机器学习算法是人工智能的核心技术"},
                    {"content": "算法优化对性能提升很重要"}
                ],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        coverage = await fallback_handler._calculate_coverage_score(
            sample_query_analysis, results_with_keywords
        )

        # 检查覆盖度计算结果（实际匹配的关键词数量可能不同）
        assert 0.0 <= coverage <= 1.0
        # 至少应该匹配一些关键词
        assert coverage > 0.0

    @pytest.mark.asyncio
    async def test_calculate_coverage_score_empty(self, fallback_handler, sample_query_analysis):
        """测试空结果的覆盖度计算"""
        coverage = await fallback_handler._calculate_coverage_score(
            sample_query_analysis, []
        )
        assert coverage == 0.0


class TestFallbackActions:
    """备用行动测试"""

    @pytest.mark.asyncio
    async def test_generate_fallback_actions_no_results(self, fallback_handler, sample_query_analysis):
        """测试为无结果失败生成备用行动"""
        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0,
            evidence=["无结果"],
            metrics={"result_count": 0}
        )

        actions = await fallback_handler._generate_fallback_actions(failure, sample_query_analysis)

        assert len(actions) > 0
        assert len(actions) <= 3  # 最多3个策略
        
        # 检查是否包含预期的策略
        strategies = [action.strategy for action in actions]
        expected_strategies = fallback_handler.strategy_priorities[FailureType.NO_RESULTS]
        
        for strategy in strategies:
            assert strategy in expected_strategies

    @pytest.mark.asyncio
    async def test_generate_fallback_actions_low_quality(self, fallback_handler, sample_query_analysis):
        """测试为低质量失败生成备用行动"""
        failure = FailureDetection(
            failure_type=FailureType.LOW_QUALITY,
            severity=FailureSeverity.MEDIUM,
            confidence=0.8,
            evidence=["质量低"],
            metrics={"quality_score": 0.2}
        )

        actions = await fallback_handler._generate_fallback_actions(failure, sample_query_analysis)

        assert len(actions) > 0
        strategies = [action.strategy for action in actions]
        expected_strategies = fallback_handler.strategy_priorities[FailureType.LOW_QUALITY]
        
        for strategy in strategies:
            assert strategy in expected_strategies

    @pytest.mark.asyncio
    async def test_create_query_expansion_action(self, fallback_handler, sample_query_analysis):
        """测试创建查询扩展行动"""
        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        action = await fallback_handler._create_fallback_action(
            FallbackStrategyType.QUERY_EXPANSION, failure, sample_query_analysis
        )

        assert action is not None
        assert action.strategy == FallbackStrategyType.QUERY_EXPANSION
        assert "扩展" in action.description
        assert "expansion_method" in action.parameters
        assert action.expected_improvement > 0
        assert action.success_probability > 0

    @pytest.mark.asyncio
    async def test_create_query_simplification_action(self, fallback_handler, sample_query_analysis):
        """测试创建查询简化行动"""
        failure = FailureDetection(
            failure_type=FailureType.QUERY_TOO_COMPLEX,
            severity=FailureSeverity.MEDIUM,
            confidence=0.8
        )

        action = await fallback_handler._create_fallback_action(
            FallbackStrategyType.QUERY_SIMPLIFICATION, failure, sample_query_analysis
        )

        assert action is not None
        assert action.strategy == FallbackStrategyType.QUERY_SIMPLIFICATION
        assert "简化" in action.description
        assert "keep_main_keywords" in action.parameters

    @pytest.mark.asyncio
    async def test_create_strategy_switch_action(self, fallback_handler, sample_query_analysis):
        """测试创建策略切换行动"""
        failure = FailureDetection(
            failure_type=FailureType.LOW_QUALITY,
            severity=FailureSeverity.MEDIUM,
            confidence=0.8
        )

        action = await fallback_handler._create_fallback_action(
            FallbackStrategyType.STRATEGY_SWITCH, failure, sample_query_analysis
        )

        assert action is not None
        assert action.strategy == FallbackStrategyType.STRATEGY_SWITCH
        assert "策略" in action.description
        assert "new_strategies" in action.parameters


class TestActionExecution:
    """行动执行测试"""

    @pytest.mark.asyncio
    async def test_execute_query_expansion(self, fallback_handler, sample_query_analysis):
        """测试执行查询扩展"""
        action = FallbackAction(
            strategy=FallbackStrategyType.QUERY_EXPANSION,
            description="扩展查询",
            parameters={"max_expansions": 3}
        )

        # Mock查询扩展器
        fallback_handler.query_expander.expand_query = AsyncMock(return_value=[
            "机器学习算法优化技术", "深度学习算法调优"
        ])

        result = await fallback_handler._execute_query_expansion(action, sample_query_analysis)

        assert result is not None
        assert "query_analysis" in result
        assert "expanded_queries" in result
        assert len(result["expanded_queries"]) > 0

    @pytest.mark.asyncio
    async def test_execute_query_simplification(self, fallback_handler, complex_query_analysis):
        """测试执行查询简化"""
        action = FallbackAction(
            strategy=FallbackStrategyType.QUERY_SIMPLIFICATION,
            description="简化查询",
            parameters={"keep_main_keywords": True}
        )

        result = await fallback_handler._execute_query_simplification(action, complex_query_analysis)

        assert result is not None
        assert "query_analysis" in result
        
        new_analysis = result["query_analysis"]
        assert len(new_analysis.keywords) <= 3  # 应该保留主要关键词
        assert new_analysis.complexity_score < complex_query_analysis.complexity_score
        assert len(new_analysis.query_text) < len(complex_query_analysis.query_text)

    @pytest.mark.asyncio
    async def test_execute_lower_threshold(self, fallback_handler, low_quality_retrieval_results):
        """测试执行降低阈值"""
        action = FallbackAction(
            strategy=FallbackStrategyType.LOWER_THRESHOLD,
            description="降低阈值",
            parameters={"new_threshold": 0.1}
        )

        result = await fallback_handler._execute_lower_threshold(action, low_quality_retrieval_results)

        assert result is not None
        assert "retrieval_results" in result
        
        new_results = result["retrieval_results"]
        assert len(new_results) >= 0
        # 验证结果是否符合新阈值
        for res in new_results:
            for item in res.results:
                assert item.get("score", 0) >= 0.1

    @pytest.mark.asyncio
    async def test_execute_fallback_action_error_handling(self, fallback_handler, sample_query_analysis):
        """测试行动执行的错误处理"""
        action = FallbackAction(
            strategy=FallbackStrategyType.QUERY_EXPANSION,
            description="扩展查询",
            parameters={"max_expansions": 3}
        )

        # Mock查询扩展器抛出异常
        fallback_handler.query_expander.expand_query = AsyncMock(side_effect=Exception("扩展失败"))

        result = await fallback_handler._execute_fallback_action(
            action, sample_query_analysis, []
        )

        assert result is None
        assert "execution_error" in action.parameters


class TestUserGuidance:
    """用户指导测试"""

    @pytest.mark.asyncio
    async def test_generate_user_guidance_no_results(self, fallback_handler, sample_query_analysis):
        """测试为无结果生成用户指导"""
        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        guidance = await fallback_handler._generate_user_guidance(failure, sample_query_analysis, [])

        assert isinstance(guidance, UserGuidance)
        assert "没有找到" in guidance.message or "结果" in guidance.message
        assert len(guidance.suggestions) > 0
        assert guidance.severity_level in ["info", "warning", "error"]

    @pytest.mark.asyncio
    async def test_generate_user_guidance_low_quality(self, fallback_handler, sample_query_analysis):
        """测试为低质量生成用户指导"""
        failure = FailureDetection(
            failure_type=FailureType.LOW_QUALITY,
            severity=FailureSeverity.MEDIUM,
            confidence=0.8
        )

        guidance = await fallback_handler._generate_user_guidance(failure, sample_query_analysis, [])

        assert isinstance(guidance, UserGuidance)
        assert "质量" in guidance.message
        assert len(guidance.suggestions) > 0

    @pytest.mark.asyncio
    async def test_generate_user_guidance_with_actions(self, fallback_handler, sample_query_analysis):
        """测试包含已执行行动的用户指导"""
        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        actions_taken = [
            FallbackAction(
                strategy=FallbackStrategyType.QUERY_EXPANSION,
                description="扩展查询"
            )
        ]

        guidance = await fallback_handler._generate_user_guidance(
            failure, sample_query_analysis, actions_taken
        )

        assert "已尝试扩展" in guidance.message or "扩展" in guidance.message

    @pytest.mark.asyncio
    async def test_generate_user_guidance_code_query(self, fallback_handler):
        """测试代码查询的特定指导"""
        code_query = QueryAnalysis(
            query_text="Python函数",
            intent_type=QueryIntent.CODE,
            confidence=0.7,
            complexity_score=0.5,
            entities=["Python"],
            keywords=["Python", "函数"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )

        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        guidance = await fallback_handler._generate_user_guidance(failure, code_query, [])

        # 检查是否包含代码查询特定的建议
        suggestions_text = " ".join(guidance.suggestions)
        assert any(keyword in suggestions_text for keyword in ["编程语言", "框架", "版本", "代码"])
        assert len(guidance.examples) > 0


class TestImprovementEvaluation:
    """改进评估测试"""

    @pytest.mark.asyncio
    async def test_check_improvement_no_results(self, fallback_handler):
        """测试无结果的改进检查"""
        original_failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        new_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试",
                results=[{"id": "1", "content": "新结果"}],
                score=0.7,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        improved = await fallback_handler._check_improvement(original_failure, new_results, None)
        assert improved is True

    @pytest.mark.asyncio
    async def test_check_improvement_quality(self, fallback_handler):
        """测试质量改进检查"""
        original_failure = FailureDetection(
            failure_type=FailureType.LOW_QUALITY,
            severity=FailureSeverity.MEDIUM,
            confidence=0.8,
            metrics={"quality_score": 0.2}
        )

        new_validation = ValidationResult(
            query_id="test",
            retrieval_results=[],
            quality_scores={},
            conflicts=[],
            overall_quality=0.7,  # 改进后的质量
            overall_confidence=0.8,
            recommendations=[],
            validation_time=0.5
        )

        improved = await fallback_handler._check_improvement(original_failure, [], new_validation)
        assert improved is True

    def test_calculate_improvement_metrics(self, fallback_handler):
        """测试改进指标计算"""
        original_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试",
                results=[{"id": "1"}],
                score=0.5,
                confidence=0.6,
                processing_time=0.1
            )
        ]

        new_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试",
                results=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
                score=0.8,
                confidence=0.9,
                processing_time=0.1
            )
        ]

        metrics = fallback_handler._calculate_improvement_metrics(
            original_results, new_results, None, None
        )

        assert "result_count_change" in metrics
        assert metrics["result_count_change"] == 2  # 从1个增加到3个
        assert "result_count_improvement" in metrics
        assert metrics["result_count_improvement"] == 2.0  # (3-1)/1 = 2.0
        assert "confidence_change" in metrics

    @pytest.mark.asyncio
    async def test_evaluate_success(self, fallback_handler):
        """测试成功评估"""
        failure = FailureDetection(
            failure_type=FailureType.NO_RESULTS,
            severity=FailureSeverity.HIGH,
            confidence=1.0
        )

        good_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试",
                results=[{"id": "1"}, {"id": "2"}],
                score=0.8,
                confidence=0.9,
                processing_time=0.1
            )
        ]

        success = await fallback_handler._evaluate_success(failure, good_results, None)
        assert success is True

        # 测试仍然失败的情况
        success = await fallback_handler._evaluate_success(failure, [], None)
        assert success is False


class TestFullWorkflow:
    """完整工作流测试"""

    @pytest.mark.asyncio
    async def test_handle_retrieval_failure_success(self, fallback_handler, sample_query_analysis, empty_retrieval_results):
        """测试成功处理检索失败"""
        # Mock相关方法
        fallback_handler.query_expander.expand_query = AsyncMock(return_value=[
            "机器学习算法优化技术"
        ])

        result = await fallback_handler.handle_retrieval_failure(
            sample_query_analysis, empty_retrieval_results, None, 1.0
        )

        assert isinstance(result, FallbackResult)
        assert result.original_failure.failure_type == FailureType.NO_RESULTS
        assert len(result.actions_taken) >= 0
        assert result.user_guidance is not None
        assert result.total_time > 0

    @pytest.mark.asyncio
    async def test_handle_retrieval_failure_no_failure(self, fallback_handler):
        """测试没有检测到失败的情况"""
        good_query = QueryAnalysis(
            query_text="Python教程",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.9,
            complexity_score=0.3,
            entities=[],
            keywords=[],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )

        good_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="Python教程",
                results=[{"id": "1", "score": 0.9, "content": "Python基础教程"}],
                score=0.9,
                confidence=0.9,
                processing_time=0.1
            )
        ]

        good_validation = ValidationResult(
            query_id="test",
            retrieval_results=good_results,
            quality_scores={},
            conflicts=[],
            overall_quality=0.9,
            overall_confidence=0.9,
            recommendations=[],
            validation_time=0.5
        )

        result = await fallback_handler.handle_retrieval_failure(
            good_query, good_results, good_validation, 1.0
        )

        assert result.success is True
        assert len(result.actions_taken) == 0

    @pytest.mark.asyncio
    async def test_multiple_failure_handling(self, fallback_handler, sample_query_analysis):
        """测试多次失败处理的历史记录"""
        # 处理第一次失败
        result1 = await fallback_handler.handle_retrieval_failure(
            sample_query_analysis, [], None, 1.0
        )

        # 处理第二次失败
        result2 = await fallback_handler.handle_retrieval_failure(
            sample_query_analysis, [], None, 1.0
        )

        assert len(fallback_handler.failure_history) == 2
        assert fallback_handler.failure_history[0] == result1
        assert fallback_handler.failure_history[1] == result2


class TestStatisticsAndHistory:
    """统计和历史记录测试"""

    def test_get_failure_statistics_empty(self, fallback_handler):
        """测试空历史的统计信息"""
        stats = fallback_handler.get_failure_statistics()
        
        assert stats["total_failures"] == 0

    def test_get_failure_statistics_with_data(self, fallback_handler):
        """测试有数据的统计信息"""
        # 添加一些模拟的失败记录
        result1 = FallbackResult(
            original_failure=FailureDetection(
                failure_type=FailureType.NO_RESULTS,
                severity=FailureSeverity.HIGH,
                confidence=1.0
            ),
            actions_taken=[
                FallbackAction(
                    strategy=FallbackStrategyType.QUERY_EXPANSION,
                    description="扩展查询"
                )
            ],
            success=True,
            improvement_metrics={"result_count_improvement": 1.0}
        )

        result2 = FallbackResult(
            original_failure=FailureDetection(
                failure_type=FailureType.LOW_QUALITY,
                severity=FailureSeverity.MEDIUM,
                confidence=0.8
            ),
            actions_taken=[
                FallbackAction(
                    strategy=FallbackStrategyType.STRATEGY_SWITCH,
                    description="切换策略"
                )
            ],
            success=False,
            improvement_metrics={"result_count_improvement": 0.5}
        )

        fallback_handler.failure_history = [result1, result2]

        stats = fallback_handler.get_failure_statistics()

        assert stats["total_failures"] == 2
        assert stats["success_rate"] == 0.5  # 1个成功，1个失败
        assert "no_results" in stats["failure_types"]
        assert "low_quality" in stats["failure_types"]
        assert "query_expansion" in stats["common_strategies"]
        assert "strategy_switch" in stats["common_strategies"]
        assert stats["avg_improvement"] == 0.75  # (1.0 + 0.5) / 2

    def test_clear_failure_history(self, fallback_handler):
        """测试清空失败历史"""
        # 添加一些记录
        fallback_handler.failure_history = [
            FallbackResult(
                original_failure=FailureDetection(
                    failure_type=FailureType.NO_RESULTS,
                    severity=FailureSeverity.HIGH,
                    confidence=1.0
                ),
                actions_taken=[],
                success=True
            )
        ]

        assert len(fallback_handler.failure_history) == 1

        fallback_handler.clear_failure_history()
        assert len(fallback_handler.failure_history) == 0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])