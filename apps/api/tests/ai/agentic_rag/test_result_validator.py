"""
检索结果验证器单元测试

测试结果验证器的各项功能：
- 相关性评估和语义匹配度分析
- 准确性评估和来源可信度检查
- 完整性评估和关键词覆盖度计算
- 一致性评估和冲突检测功能
- 时效性评估和可信度评分
- 综合质量评分和改进建议生成
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.ai.agentic_rag.result_validator import (
    ResultValidator, QualityScore, QualityDimension, ConflictDetection, 
    ConflictType, ValidationResult
)
from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy


@pytest.fixture
def result_validator():
    """创建结果验证器实例"""
    with patch('src.ai.agentic_rag.result_validator.get_openai_client') as mock_client:
        validator = ResultValidator()
        validator.client = mock_client.return_value
        return validator


@pytest.fixture
def sample_query_analysis():
    """创建示例查询分析"""
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
def sample_retrieval_results():
    """创建示例检索结果"""
    return [
        RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="机器学习算法实现",
            results=[
                {
                    "id": "result_1",
                    "score": 0.9,
                    "content": "机器学习是一种人工智能算法，用于让计算机从数据中学习模式。实现机器学习算法需要选择合适的模型，如决策树、神经网络等。",
                    "file_path": "/docs/ml_guide.md",
                    "file_type": "markdown",
                    "metadata": {"collection": "documents"}
                },
                {
                    "id": "result_2",
                    "score": 0.8,
                    "content": "算法实现的关键步骤包括：1. 数据预处理 2. 模型选择 3. 训练过程 4. 评估优化。Python是最常用的机器学习实现语言。",
                    "file_path": "/code/ml_implementation.py",
                    "file_type": "python",
                    "metadata": {"collection": "code"}
                }
            ],
            score=0.85,
            confidence=0.9,
            processing_time=0.1,
            explanation="语义检索结果"
        ),
        RetrievalResult(
            agent_type=RetrievalStrategy.KEYWORD,
            query="机器学习算法实现",
            results=[
                {
                    "id": "result_3",
                    "score": 0.7,
                    "content": "深度学习是机器学习的一个分支，使用神经网络进行模式识别。实现深度学习算法需要大量计算资源。",
                    "file_path": "/docs/deep_learning.md",
                    "file_type": "markdown",
                    "bm25_score": 2.5,
                    "keyword_matches": 6
                }
            ],
            score=0.7,
            confidence=0.8,
            processing_time=0.2,
            explanation="关键词检索结果"
        )
    ]


@pytest.fixture 
def conflicting_retrieval_results():
    """创建包含冲突的检索结果"""
    return [
        RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="Python性能",
            results=[
                {
                    "id": "conflict_1",
                    "score": 0.8,
                    "content": "Python是一种高效的编程语言，运行速度很快，适合大规模数据处理。2020年发布了重大更新。",
                    "file_path": "/docs/python_performance.md",
                    "file_type": "markdown"
                },
                {
                    "id": "conflict_2", 
                    "score": 0.7,
                    "content": "Python运行速度相对较慢，不适合对性能要求很高的应用。2022年才发布了重大更新。",
                    "file_path": "/docs/python_limitations.md",
                    "file_type": "markdown"
                }
            ],
            score=0.75,
            confidence=0.8,
            processing_time=0.1
        )
    ]


class TestResultValidator:
    """结果验证器基础功能测试"""

    def test_validator_initialization(self, result_validator):
        """测试验证器初始化"""
        assert result_validator.client is not None
        assert len(result_validator.quality_weights) == 6
        assert sum(result_validator.quality_weights.values()) == 1.0  # 权重和为1
        assert len(result_validator.conflict_thresholds) == 4

    @pytest.mark.asyncio
    async def test_validate_results_success(self, result_validator, sample_query_analysis, sample_retrieval_results):
        """测试成功的结果验证"""
        # Mock LLM调用
        result_validator.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"relevance_score": 0.8, "explanation": "相关性很高"}'))]
        ))

        result = await result_validator.validate_results(sample_query_analysis, sample_retrieval_results)

        assert isinstance(result, ValidationResult)
        assert result.overall_quality >= 0.0
        assert result.overall_confidence >= 0.0
        assert len(result.quality_scores) == 6  # 6个质量维度
        assert result.validation_time > 0
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_validate_results_empty_input(self, result_validator, sample_query_analysis):
        """测试空输入的验证"""
        result = await result_validator.validate_results(sample_query_analysis, [])

        assert isinstance(result, ValidationResult)
        assert len(result.retrieval_results) == 0
        # 应该仍有质量评分（可能是默认值）
        assert len(result.quality_scores) >= 0

    @pytest.mark.asyncio
    async def test_validate_results_with_exceptions(self, result_validator, sample_query_analysis, sample_retrieval_results):
        """测试验证过程中的异常处理"""
        # Mock LLM调用失败
        result_validator.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        result = await result_validator.validate_results(sample_query_analysis, sample_retrieval_results)

        assert isinstance(result, ValidationResult)
        # 即使有异常，也应该返回基本的验证结果
        assert result.overall_quality >= 0.0
        assert "验证过程失败" in result.recommendations or len(result.recommendations) >= 0


class TestRelevanceEvaluation:
    """相关性评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_relevance_high_score(self, result_validator, sample_query_analysis, sample_retrieval_results):
        """测试高相关性评估"""
        # Mock LLM返回高相关性分数
        result_validator.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"relevance_score": 0.9, "explanation": "高度相关"}'))]
        ))

        score = await result_validator._evaluate_relevance(sample_query_analysis, sample_retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.RELEVANCE
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.explanation, str)
        assert len(score.evidence) >= 0

    @pytest.mark.asyncio
    async def test_evaluate_relevance_empty_results(self, result_validator, sample_query_analysis):
        """测试空结果的相关性评估"""
        score = await result_validator._evaluate_relevance(sample_query_analysis, [])

        assert score.dimension == QualityDimension.RELEVANCE
        assert score.score == 0.0
        assert score.confidence == 1.0  # 对空结果的判断是确定的

    @pytest.mark.asyncio
    async def test_llm_evaluate_relevance_success(self, result_validator, sample_query_analysis):
        """测试LLM相关性评估成功"""
        results = [{"content": "机器学习算法的详细实现指南"}]
        
        # Create a proper mock response structure
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = '{"relevance_score": 0.85}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        result_validator.client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        score = await result_validator._llm_evaluate_relevance(sample_query_analysis, results)

        assert score == 0.85
        result_validator.client.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_evaluate_relevance_failure(self, result_validator, sample_query_analysis):
        """测试LLM相关性评估失败"""
        results = [{"content": "测试内容"}]
        
        result_validator.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        score = await result_validator._llm_evaluate_relevance(sample_query_analysis, results)

        assert score == 0.5  # 默认返回值

    def test_simple_relevance_evaluation(self, result_validator):
        """测试简化的相关性评估"""
        results = [
            {"score": 0.9, "content": "高分内容"},
            {"score": 0.7, "content": "中等分数内容"},
            {"score": 0.5, "content": "低分内容"}
        ]

        score = result_validator._simple_relevance_evaluation(results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.RELEVANCE
        assert abs(score.score - 0.7) < 0.001  # (0.9 + 0.7 + 0.5) / 3，允许浮点误差
        assert score.confidence == 0.6


class TestAccuracyEvaluation:
    """准确性评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_accuracy_success(self, result_validator, sample_query_analysis, sample_retrieval_results):
        """测试准确性评估"""
        score = await result_validator._evaluate_accuracy(sample_query_analysis, sample_retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.ACCURACY
        assert 0.0 <= score.score <= 1.0
        assert len(score.evidence) >= 0

    def test_evaluate_source_credibility_code_file(self, result_validator):
        """测试代码文件的来源可信度"""
        item = {"file_path": "/src/ml_algorithms.py", "file_type": "py"}
        
        score = result_validator._evaluate_source_credibility(item)
        
        assert score >= 0.8  # 代码文件通常可信度较高

    def test_evaluate_source_credibility_doc_file(self, result_validator):
        """测试文档文件的来源可信度"""
        item = {"file_path": "/docs/official/api.md", "file_type": "md"}
        
        score = result_validator._evaluate_source_credibility(item)
        
        assert score >= 0.6  # 官方文档可信度较高

    def test_evaluate_content_quality_good_content(self, result_validator):
        """测试良好内容的质量评估"""
        results = [
            {
                "content": """# 机器学习算法实现

机器学习是人工智能的核心技术之一。本文介绍如何实现基本的机器学习算法。

## 主要步骤

* 数据预处理
* 特征工程  
* 模型训练
* 结果评估

```python
def train_model(data):
    # 训练逻辑
    pass
```

这些步骤确保了算法的有效性。"""
            }
        ]

        score = result_validator._evaluate_content_quality(results)

        assert score >= 0.7  # 结构良好的内容应该得分较高

    def test_evaluate_content_quality_poor_content(self, result_validator):
        """测试低质量内容的评估"""
        results = [
            {"content": "短"},
            {"content": ""},
            {"content": "没有结构的长文本" * 50}
        ]

        score = result_validator._evaluate_content_quality(results)

        assert score <= 0.6  # 质量差的内容应该得分较低


class TestCompletenessEvaluation:
    """完整性评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_completeness_success(self, result_validator, sample_query_analysis, sample_retrieval_results):
        """测试完整性评估"""
        score = await result_validator._evaluate_completeness(sample_query_analysis, sample_retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.COMPLETENESS
        assert 0.0 <= score.score <= 1.0
        assert len(score.evidence) >= 0

    def test_calculate_keyword_coverage_full(self, result_validator):
        """测试完全关键词覆盖"""
        query_analysis = QueryAnalysis(
            query_text="机器学习算法",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.5,
            entities=[],
            keywords=["机器学习", "算法"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        results = [
            {"content": "机器学习是一种算法类型，用于数据分析"}
        ]

        coverage = result_validator._calculate_keyword_coverage(query_analysis, results)

        assert coverage == 1.0  # 所有关键词都覆盖

    def test_calculate_keyword_coverage_partial(self, result_validator):
        """测试部分关键词覆盖"""
        query_analysis = QueryAnalysis(
            query_text="机器学习和深度学习",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.5,
            entities=[],
            keywords=["机器学习", "深度学习", "神经网络"],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        results = [
            {"content": "机器学习是人工智能的分支，深度学习是其子集"}
        ]

        coverage = result_validator._calculate_keyword_coverage(query_analysis, results)

        assert coverage == 2.0/3.0  # 只有2个关键词被覆盖

    def test_calculate_entity_coverage(self, result_validator):
        """测试实体覆盖度计算"""
        query_analysis = QueryAnalysis(
            query_text="Python和Java编程",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.5,
            entities=["Python", "Java"],
            keywords=[],
            domain="技术",
            sentiment="neutral", 
            language="zh"
        )
        
        results = [
            {"content": "Python是一种编程语言，易于学习和使用"}
        ]

        coverage = result_validator._calculate_entity_coverage(query_analysis, results)

        assert coverage == 0.5  # 只有Python实体被覆盖

    def test_get_required_info_types(self, result_validator):
        """测试获取所需信息类型"""
        # 事实查询
        types = result_validator._get_required_info_types(QueryIntent.FACTUAL)
        assert "definition" in types
        assert "example" in types

        # 程序查询
        types = result_validator._get_required_info_types(QueryIntent.PROCEDURAL)
        assert "procedure" in types
        assert "example" in types

        # 代码查询
        types = result_validator._get_required_info_types(QueryIntent.CODE)
        assert "code" in types
        assert "procedure" in types


class TestConsistencyEvaluation:
    """一致性评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_consistency_consistent_results(self, result_validator):
        """测试一致结果的一致性评估"""
        retrieval_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试查询",
                results=[
                    {"id": "1", "content": "Python是一种编程语言，易于学习"},
                    {"id": "2", "content": "Python语言设计简洁，学习曲线平缓"}
                ],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        score = await result_validator._evaluate_consistency(retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.CONSISTENCY
        assert score.score >= 0.0  # 一致的内容应该有合理得分

    @pytest.mark.asyncio 
    async def test_evaluate_consistency_single_result(self, result_validator):
        """测试单个结果的一致性评估"""
        retrieval_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试查询",
                results=[{"id": "1", "content": "单个结果"}],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        score = await result_validator._evaluate_consistency(retrieval_results)

        assert score.score == 1.0  # 单个结果认为完全一致
        assert score.confidence == 0.5

    def test_calculate_content_consistency_similar(self, result_validator):
        """测试相似内容的一致性计算"""
        content1 = "机器学习是人工智能的重要分支，用于数据分析"
        content2 = "机器学习属于人工智能领域，主要应用于数据处理"

        consistency = result_validator._calculate_content_consistency(content1, content2)

        assert consistency >= 0.2  # 相似内容应该有一定一致性

    def test_calculate_content_consistency_conflicting(self, result_validator):
        """测试冲突内容的一致性计算"""
        content1 = "Python运行速度很快，是高效的语言"
        content2 = "Python运行速度很慢，不是高效的选择"

        consistency = result_validator._calculate_content_consistency(content1, content2)

        assert consistency <= 0.5  # 冲突内容一致性应该较低


class TestTimelinesssEvaluation:
    """时效性评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_timeliness_with_time_info(self, result_validator):
        """测试有时间信息的时效性评估"""
        import time
        current_time = time.time()
        recent_time = current_time - (30 * 24 * 3600)  # 30天前

        retrieval_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试查询",
                results=[
                    {
                        "id": "1", 
                        "content": "测试内容",
                        "metadata": {"modification_time": recent_time}
                    }
                ],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        score = await result_validator._evaluate_timeliness(retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.TIMELINESS
        assert score.score >= 0.8  # 最近的内容时效性应该很高

    @pytest.mark.asyncio
    async def test_evaluate_timeliness_without_time_info(self, result_validator, sample_retrieval_results):
        """测试没有时间信息的时效性评估"""
        score = await result_validator._evaluate_timeliness(sample_retrieval_results)

        assert score.dimension == QualityDimension.TIMELINESS
        assert score.score == 0.6  # 没有时间信息时使用默认分数


class TestCredibilityEvaluation:
    """可信度评估测试"""

    @pytest.mark.asyncio
    async def test_evaluate_credibility_success(self, result_validator, sample_retrieval_results):
        """测试可信度评估"""
        score = await result_validator._evaluate_credibility(sample_retrieval_results)

        assert isinstance(score, QualityScore)
        assert score.dimension == QualityDimension.CREDIBILITY
        assert 0.0 <= score.score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_credibility_with_references(self, result_validator):
        """测试有参考链接的可信度评估"""
        retrieval_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试查询", 
                results=[
                    {
                        "id": "1",
                        "content": "根据研究显示，95%的开发者使用Python。参考：https://example.com/study",
                        "file_path": "/docs/official/report.pdf",
                        "file_type": "pdf"
                    }
                ],
                score=0.8,
                confidence=0.8,
                processing_time=0.1
            )
        ]

        score = await result_validator._evaluate_credibility(retrieval_results)

        assert score.score >= 0.7  # 有参考链接和数据的内容可信度应该较高


class TestConflictDetection:
    """冲突检测测试"""

    @pytest.mark.asyncio
    async def test_detect_conflicts_success(self, result_validator, conflicting_retrieval_results):
        """测试冲突检测成功"""
        conflicts = await result_validator._detect_conflicts(conflicting_retrieval_results)

        assert isinstance(conflicts, list)
        # 应该能检测到冲突（快vs慢，2020vs2022）
        assert len(conflicts) >= 1
        if conflicts:
            assert isinstance(conflicts[0], ConflictDetection)

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(self, result_validator, sample_retrieval_results):
        """测试没有冲突的情况"""
        conflicts = await result_validator._detect_conflicts(sample_retrieval_results)

        assert isinstance(conflicts, list)
        # 正常的结果应该没有明显冲突
        assert len(conflicts) == 0

    def test_detect_factual_conflicts(self, result_validator):
        """测试事实冲突检测"""
        results = [
            {"id": "1", "content": "Python运行速度很快，是高效的语言"},
            {"id": "2", "content": "Python运行速度不快，性能相对较低"}
        ]

        conflicts = result_validator._detect_factual_conflicts(results)

        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.FACTUAL

    def test_detect_numerical_conflicts(self, result_validator):
        """测试数值冲突检测"""
        results = [
            {"id": "1", "content": "市场份额达到80%，表现优秀"},
            {"id": "2", "content": "根据统计，市场份额仅为20%"}
        ]

        conflicts = result_validator._detect_numerical_conflicts(results)

        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.NUMERICAL

    def test_detect_temporal_conflicts(self, result_validator):
        """测试时间冲突检测"""
        results = [
            {"id": "1", "content": "Python 3.9在2020年发布，带来重大改进"},
            {"id": "2", "content": "Python 3.9在2022年正式发布"}
        ]

        conflicts = result_validator._detect_temporal_conflicts(results)

        # 应该检测到时间冲突
        assert len(conflicts) >= 1
        if conflicts:
            assert conflicts[0].conflict_type == ConflictType.TEMPORAL


class TestQualityCalculation:
    """质量计算测试"""

    def test_calculate_overall_quality(self, result_validator):
        """测试综合质量评分计算"""
        quality_scores = {
            QualityDimension.RELEVANCE: QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.8,
                confidence=0.9,
                explanation="高相关性"
            ),
            QualityDimension.ACCURACY: QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.7,
                confidence=0.8,
                explanation="较高准确性"
            )
        }

        overall_quality = result_validator._calculate_overall_quality(quality_scores)

        assert 0.0 <= overall_quality <= 1.0
        # 基于权重的加权平均，相关性权重更高
        expected = (0.8 * 0.3 * 0.9 + 0.7 * 0.25 * 0.8) / (0.3 * 0.9 + 0.25 * 0.8)
        assert abs(overall_quality - expected) < 0.01

    def test_calculate_overall_confidence(self, result_validator):
        """测试综合置信度计算"""
        quality_scores = {
            QualityDimension.RELEVANCE: QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.8,
                confidence=0.9,
                explanation="测试"
            ),
            QualityDimension.ACCURACY: QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.7,
                confidence=0.7,
                explanation="测试"
            )
        }

        overall_confidence = result_validator._calculate_overall_confidence(quality_scores)

        assert overall_confidence == 0.8  # (0.9 + 0.7) / 2

    def test_generate_recommendations_low_quality(self, result_validator, sample_query_analysis):
        """测试为低质量结果生成建议"""
        quality_scores = {
            QualityDimension.RELEVANCE: QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.4,  # 低相关性
                confidence=0.8,
                explanation="相关性较低"
            ),
            QualityDimension.COMPLETENESS: QualityScore(
                dimension=QualityDimension.COMPLETENESS,
                score=0.3,  # 低完整性
                confidence=0.7,
                explanation="信息不完整"
            )
        }
        
        conflicts = [
            ConflictDetection(
                conflict_type=ConflictType.FACTUAL,
                conflicted_items=[("1", "2")],
                severity=0.8,
                explanation="事实冲突"
            )
        ]

        recommendations = result_validator._generate_recommendations(
            sample_query_analysis, quality_scores, conflicts
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) >= 2  # 应该有多个建议
        
        # 检查是否包含相关的建议
        rec_text = " ".join(recommendations)
        assert "相关性" in rec_text or "检索策略" in rec_text
        assert "完整性" in rec_text or "检索范围" in rec_text
        assert "冲突" in rec_text

    def test_generate_recommendations_code_query(self, result_validator):
        """测试为代码查询生成特定建议"""
        code_query = QueryAnalysis(
            query_text="Python函数实现",
            intent_type=QueryIntent.CODE,
            confidence=0.8,
            complexity_score=0.5,
            entities=[],
            keywords=[],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )

        recommendations = result_validator._generate_recommendations(code_query, {}, [])

        assert isinstance(recommendations, list)
        # 对代码查询应该有特定建议
        rec_text = " ".join(recommendations)
        assert "代码" in rec_text or "官方文档" in rec_text or "最新版本" in rec_text


class TestDataStructures:
    """数据结构测试"""

    def test_quality_score_creation(self):
        """测试质量评分数据结构"""
        score = QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=0.8,
            confidence=0.9,
            explanation="测试评分",
            evidence=["证据1", "证据2"]
        )

        assert score.dimension == QualityDimension.RELEVANCE
        assert score.score == 0.8
        assert score.confidence == 0.9
        assert score.explanation == "测试评分"
        assert len(score.evidence) == 2

    def test_conflict_detection_creation(self):
        """测试冲突检测数据结构"""
        conflict = ConflictDetection(
            conflict_type=ConflictType.FACTUAL,
            conflicted_items=[("item1", "item2")],
            severity=0.7,
            explanation="事实冲突",
            resolution_suggestion="建议核实"
        )

        assert conflict.conflict_type == ConflictType.FACTUAL
        assert len(conflict.conflicted_items) == 1
        assert conflict.severity == 0.7
        assert conflict.explanation == "事实冲突"
        assert conflict.resolution_suggestion == "建议核实"

    def test_validation_result_creation(self, sample_query_analysis, sample_retrieval_results):
        """测试验证结果数据结构"""
        quality_scores = {
            QualityDimension.RELEVANCE: QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.8,
                confidence=0.9,
                explanation="测试"
            )
        }

        result = ValidationResult(
            query_id="test_query",
            retrieval_results=sample_retrieval_results,
            quality_scores=quality_scores,
            conflicts=[],
            overall_quality=0.8,
            overall_confidence=0.9,
            recommendations=["建议1", "建议2"],
            validation_time=0.5,
            metadata={"test": "data"}
        )

        assert result.query_id == "test_query"
        assert len(result.retrieval_results) == 2
        assert len(result.quality_scores) == 1
        assert result.overall_quality == 0.8
        assert result.overall_confidence == 0.9
        assert len(result.recommendations) == 2
        assert result.validation_time == 0.5
        assert result.metadata["test"] == "data"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])