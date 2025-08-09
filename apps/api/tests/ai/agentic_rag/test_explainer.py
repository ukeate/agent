"""
检索解释器单元测试

测试检索解释器的各项功能：
- 检索路径记录和管理
- 决策过程记录和分析
- 置信度分析和级别判断
- 解释内容生成和格式化
- 可视化数据生成
- 改进建议和替代方法生成
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.ai.agentic_rag.explainer import (
    RetrievalExplainer, DecisionRecord, RetrievalPath, ConfidenceAnalysis,
    ExplanationOutput, ExplanationLevel, DecisionPoint, ConfidenceLevel
)
from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy
from src.ai.agentic_rag.result_validator import ValidationResult, QualityScore, QualityDimension
from src.ai.agentic_rag.context_composer import ComposedContext, KnowledgeFragment, FragmentType


@pytest.fixture
def explainer():
    """创建解释器实例"""
    with patch('src.ai.agentic_rag.explainer.get_openai_client') as mock_client:
        explainer = RetrievalExplainer()
        explainer.client = mock_client.return_value
        return explainer


@pytest.fixture
def sample_query_analysis():
    """创建示例查询分析"""
    return QueryAnalysis(
        query_text="Python机器学习库使用方法",
        intent_type=QueryIntent.CODE,
        confidence=0.8,
        complexity_score=0.6,
        entities=["Python", "机器学习"],
        keywords=["Python", "机器学习", "库", "使用", "方法"],
        domain="技术",
        sentiment="neutral",
        language="zh"
    )


@pytest.fixture
def sample_decision_record():
    """创建示例决策记录"""
    return DecisionRecord(
        decision_point=DecisionPoint.STRATEGY_SELECTION,
        timestamp=time.time(),
        input_data={"query_intent": "CODE", "complexity": 0.6},
        decision_made={"strategies": ["semantic", "keyword"], "primary": "semantic"},
        reasoning="基于代码查询意图，选择语义检索作为主要策略，关键词检索作为辅助",
        confidence=0.85,
        alternatives_considered=[
            {"strategy": "structured", "score": 0.3, "reason": "实体较少"}
        ],
        execution_time=0.05,
        success=True
    )


@pytest.fixture
def sample_retrieval_results():
    """创建示例检索结果"""
    return [
        RetrievalResult(
            agent_type=RetrievalStrategy.SEMANTIC,
            query="Python机器学习库使用方法",
            results=[
                {
                    "id": "result_1",
                    "score": 0.9,
                    "content": "scikit-learn是Python中最流行的机器学习库，提供了分类、回归、聚类等算法",
                    "file_path": "/docs/sklearn_guide.md",
                    "file_type": "markdown"
                },
                {
                    "id": "result_2",
                    "score": 0.85,
                    "content": "import sklearn\\nfrom sklearn.ensemble import RandomForestClassifier\\n# 使用随机森林分类器",
                    "file_path": "/code/ml_example.py",
                    "file_type": "python"
                }
            ],
            score=0.875,
            confidence=0.9,
            processing_time=0.15,
            explanation="语义向量搜索找到高相关性结果"
        ),
        RetrievalResult(
            agent_type=RetrievalStrategy.KEYWORD,
            query="Python机器学习库使用方法",
            results=[
                {
                    "id": "result_3",
                    "score": 0.7,
                    "content": "pandas和numpy是机器学习的基础库，用于数据处理和数值计算",
                    "file_path": "/docs/ml_basics.md",
                    "file_type": "markdown",
                    "bm25_score": 2.3,
                    "keyword_matches": 4
                }
            ],
            score=0.7,
            confidence=0.8,
            processing_time=0.12,
            explanation="关键词匹配找到相关文档"
        )
    ]


@pytest.fixture
def sample_validation_result():
    """创建示例验证结果"""
    quality_scores = {
        QualityDimension.RELEVANCE: QualityScore(
            dimension=QualityDimension.RELEVANCE,
            score=0.85,
            confidence=0.9,
            explanation="结果与查询高度相关",
            evidence=["包含Python和机器学习关键词", "提供了具体使用示例"]
        ),
        QualityDimension.ACCURACY: QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=0.8,
            confidence=0.85,
            explanation="信息准确性良好",
            evidence=["来源可信", "内容结构清晰"]
        ),
        QualityDimension.COMPLETENESS: QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.75,
            confidence=0.8,
            explanation="信息相对完整",
            evidence=["涵盖基本概念", "缺少高级用法"]
        )
    }
    
    return ValidationResult(
        query_id="test_query",
        retrieval_results=[],  # 简化
        quality_scores=quality_scores,
        conflicts=[],
        overall_quality=0.8,
        overall_confidence=0.85,
        recommendations=["建议查看官方文档获取更多高级用法"],
        validation_time=0.25
    )


@pytest.fixture
def sample_composed_context():
    """创建示例组合上下文"""
    fragments = [
        KnowledgeFragment(
            id="frag_1",
            content="scikit-learn库安装：pip install scikit-learn",
            source="/docs/install.md",
            fragment_type=FragmentType.CODE,
            relevance_score=0.9,
            quality_score=0.8,
            information_density=0.7,
            tokens=15,
            metadata={"file_path": "/docs/install.md"}
        ),
        KnowledgeFragment(
            id="frag_2",
            content="机器学习是让计算机从数据中学习规律的方法",
            source="/docs/concepts.md",
            fragment_type=FragmentType.DEFINITION,
            relevance_score=0.8,
            quality_score=0.9,
            information_density=0.6,
            tokens=20,
            metadata={"file_path": "/docs/concepts.md"}
        )
    ]
    
    return ComposedContext(
        query_id="test_query",
        selected_fragments=fragments,
        total_tokens=35,
        information_density=0.7,
        diversity_score=0.8,
        coherence_score=0.75,
        relationships=[],
        composition_strategy="balanced",
        optimization_metrics={"selection_time": 0.1},
        metadata={"total_fragments": 10}
    )


class TestRetrievalExplainer:
    """检索解释器基础功能测试"""
    
    def test_explainer_initialization(self, explainer):
        """测试解释器初始化"""
        assert explainer.client is not None
        assert len(explainer.explanation_templates) == 3  # SIMPLE, DETAILED, TECHNICAL
        assert len(explainer.confidence_thresholds) == 5
        assert isinstance(explainer.path_records, dict)
    
    def test_start_path_recording(self, explainer, sample_query_analysis):
        """测试开始路径记录"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        
        assert path_id.startswith("path_")
        assert path_id in explainer.path_records
        
        path = explainer.path_records[path_id]
        assert path.path_id == path_id
        assert path.query_analysis == sample_query_analysis
        assert len(path.decisions) == 0
    
    def test_record_decision(self, explainer, sample_query_analysis, sample_decision_record):
        """测试记录决策"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        explainer.record_decision(path_id, sample_decision_record)
        
        path = explainer.path_records[path_id]
        assert len(path.decisions) == 1
        assert path.decisions[0] == sample_decision_record
    
    def test_record_decision_invalid_path(self, explainer, sample_decision_record):
        """测试记录决策到无效路径"""
        explainer.record_decision("invalid_path_id", sample_decision_record)
        
        # 不应该抛出异常，但也不应该记录任何内容
        assert "invalid_path_id" not in explainer.path_records
    
    def test_finish_path_recording(self, explainer, sample_query_analysis, sample_decision_record):
        """测试完成路径记录"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        explainer.record_decision(path_id, sample_decision_record)
        explainer.finish_path_recording(path_id, 2.5, 5)
        
        path = explainer.path_records[path_id]
        assert path.total_time == 2.5
        assert path.final_results_count == 5
        assert path.success_rate == 1.0  # 1个成功决策
        assert path.path_visualization is not None
    
    def test_finish_path_recording_with_failures(self, explainer, sample_query_analysis):
        """测试包含失败决策的路径记录"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        
        # 添加成功和失败的决策
        success_decision = DecisionRecord(
            decision_point=DecisionPoint.STRATEGY_SELECTION,
            timestamp=time.time(),
            input_data={},
            decision_made={},
            reasoning="成功决策",
            confidence=0.8,
            success=True
        )
        
        failure_decision = DecisionRecord(
            decision_point=DecisionPoint.RETRIEVAL_EXECUTION,
            timestamp=time.time(),
            input_data={},
            decision_made={},
            reasoning="失败决策",
            confidence=0.5,
            success=False,
            error_message="检索失败"
        )
        
        explainer.record_decision(path_id, success_decision)
        explainer.record_decision(path_id, failure_decision)
        explainer.finish_path_recording(path_id, 1.0, 2)
        
        path = explainer.path_records[path_id]
        assert path.success_rate == 0.5  # 1个成功，1个失败
    
    def test_get_path_record(self, explainer, sample_query_analysis):
        """测试获取路径记录"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        
        retrieved_path = explainer.get_path_record(path_id)
        assert retrieved_path is not None
        assert retrieved_path.path_id == path_id
        
        # 测试获取不存在的路径
        invalid_path = explainer.get_path_record("invalid_id")
        assert invalid_path is None
    
    def test_list_path_records(self, explainer, sample_query_analysis):
        """测试列出路径记录"""
        initial_count = len(explainer.list_path_records())
        
        path_id1 = explainer.start_path_recording(sample_query_analysis)
        path_id2 = explainer.start_path_recording(sample_query_analysis)
        
        records = explainer.list_path_records()
        assert len(records) == initial_count + 2
        assert path_id1 in records
        assert path_id2 in records
    
    def test_clear_path_records(self, explainer, sample_query_analysis):
        """测试清空路径记录"""
        explainer.start_path_recording(sample_query_analysis)
        explainer.start_path_recording(sample_query_analysis)
        
        assert len(explainer.path_records) >= 2
        
        explainer.clear_path_records()
        assert len(explainer.path_records) == 0


class TestConfidenceAnalysis:
    """置信度分析测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_confidence_high_confidence(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result):
        """测试高置信度分析"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        confidence_analysis = await explainer._analyze_confidence(
            path, sample_retrieval_results, sample_validation_result
        )
        
        assert isinstance(confidence_analysis, ConfidenceAnalysis)
        assert 0.0 <= confidence_analysis.overall_confidence <= 1.0
        assert confidence_analysis.confidence_level in list(ConfidenceLevel)
        assert len(confidence_analysis.confidence_breakdown) >= 3
        assert isinstance(confidence_analysis.uncertainty_sources, list)
        assert len(confidence_analysis.confidence_explanation) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_confidence_low_confidence(self, explainer):
        """测试低置信度分析"""
        # 创建低置信度的查询分析
        low_confidence_query = QueryAnalysis(
            query_text="模糊查询",
            intent_type=QueryIntent.CREATIVE,
            confidence=0.3,  # 低置信度
            complexity_score=0.8,
            entities=[],
            keywords=["模糊"],
            domain=None,
            sentiment="neutral",
            language="zh"
        )
        
        path_id = explainer.start_path_recording(low_confidence_query)
        path = explainer.path_records[path_id]
        
        # 创建低置信度的检索结果
        low_confidence_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="模糊查询",
                results=[],  # 空结果
                score=0.2,
                confidence=0.3,
                processing_time=0.1
            )
        ]
        
        confidence_analysis = await explainer._analyze_confidence(
            path, low_confidence_results, None
        )
        
        assert confidence_analysis.overall_confidence < 0.6
        assert confidence_analysis.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        assert len(confidence_analysis.uncertainty_sources) > 0
    
    def test_get_confidence_level(self, explainer):
        """测试置信度级别判断"""
        assert explainer._get_confidence_level(0.9) == ConfidenceLevel.VERY_HIGH
        assert explainer._get_confidence_level(0.7) == ConfidenceLevel.HIGH
        assert explainer._get_confidence_level(0.5) == ConfidenceLevel.MEDIUM
        assert explainer._get_confidence_level(0.3) == ConfidenceLevel.LOW
        assert explainer._get_confidence_level(0.1) == ConfidenceLevel.VERY_LOW
    
    @pytest.mark.asyncio
    async def test_generate_confidence_explanation(self, explainer):
        """测试置信度解释生成"""
        breakdown = {
            "query_analysis": 0.8,
            "retrieval": 0.9,
            "validation": 0.7,
            "decisions": 0.8
        }
        uncertainty_sources = ["查询意图有些模糊"]
        reliability_factors = {
            "result_count": 0.8,
            "strategy_diversity": 0.6,
            "execution_success": 0.9,
            "time_efficiency": 0.7
        }
        
        explanation = await explainer._generate_confidence_explanation(
            0.8, breakdown, uncertainty_sources, reliability_factors
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 50  # 应该是详细的解释
        assert "0.80" in explanation  # 包含具体数值
        assert "较高" in explanation or "高" in explanation  # 包含级别描述


class TestExplanationGeneration:
    """解释内容生成测试"""
    
    @pytest.mark.asyncio
    async def test_explain_retrieval_process_success(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result):
        """测试成功的检索过程解释"""
        # 设置路径记录
        path_id = explainer.start_path_recording(sample_query_analysis)
        
        # 添加一些决策记录
        decision = DecisionRecord(
            decision_point=DecisionPoint.STRATEGY_SELECTION,
            timestamp=time.time(),
            input_data={"intent": "CODE"},
            decision_made={"primary_strategy": "semantic"},
            reasoning="基于代码查询选择语义检索",
            confidence=0.85
        )
        explainer.record_decision(path_id, decision)
        explainer.finish_path_recording(path_id, 1.5, 3)
        
        # Mock LLM调用
        explainer.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="这是改进建议"))]
        ))
        
        explanation = await explainer.explain_retrieval_process(
            path_id, sample_retrieval_results, sample_validation_result
        )
        
        assert isinstance(explanation, ExplanationOutput)
        assert explanation.query == sample_query_analysis.query_text
        assert explanation.explanation_level == ExplanationLevel.DETAILED
        assert len(explanation.summary) > 0
        assert len(explanation.detailed_explanation) > 0
        assert len(explanation.decision_rationale) >= 1
        assert explanation.flow_diagram is not None
        assert explanation.metrics_chart is not None
        assert len(explanation.timeline) > 0
        assert explanation.generation_time > 0
    
    @pytest.mark.asyncio
    async def test_explain_retrieval_process_invalid_path(self, explainer, sample_retrieval_results):
        """测试无效路径ID的异常处理"""
        with pytest.raises(ValueError, match="Path record not found"):
            await explainer.explain_retrieval_process(
                "invalid_path_id", sample_retrieval_results
            )
    
    def test_generate_summary_simple_level(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result):
        """测试简单级别摘要生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        summary = explainer._generate_summary(
            path, sample_retrieval_results, sample_validation_result, ExplanationLevel.SIMPLE
        )
        
        assert isinstance(summary, str)
        assert sample_query_analysis.query_text in summary
        assert "2种检索策略" in summary  # semantic + keyword
        assert "优秀" in summary or "良好" in summary  # 基于质量评分
    
    def test_generate_summary_detailed_level(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result):
        """测试详细级别摘要生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        path.total_time = 1.2
        
        summary = explainer._generate_summary(
            path, sample_retrieval_results, sample_validation_result, ExplanationLevel.DETAILED
        )
        
        assert isinstance(summary, str)
        assert "code类型查询" in summary.lower()
        assert ("semantic" in summary or "语义" in summary) and ("keyword" in summary or "关键词" in summary)  # 策略列表
        assert "1.20秒" in summary  # 执行时间
        assert "0.80" in summary  # 质量分数
    
    @pytest.mark.asyncio
    async def test_generate_detailed_explanation_complete(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result, sample_composed_context):
        """测试完整的详细解释生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        # 添加策略选择决策
        strategy_decision = DecisionRecord(
            decision_point=DecisionPoint.STRATEGY_SELECTION,
            timestamp=time.time(),
            input_data={},
            decision_made={},
            reasoning="选择多策略并行检索以提高覆盖度",
            confidence=0.8
        )
        explainer.record_decision(path_id, strategy_decision)
        
        explanation = await explainer._generate_detailed_explanation(
            path, sample_retrieval_results, sample_validation_result, sample_composed_context, ExplanationLevel.DETAILED
        )
        
        assert isinstance(explanation, str)
        assert "## 查询分析阶段" in explanation
        assert "## 策略选择阶段" in explanation
        assert "## 检索执行阶段" in explanation
        assert "## 结果验证阶段" in explanation
        assert "## 上下文组合阶段" in explanation
        
        # 检查是否包含具体信息
        assert sample_query_analysis.query_text in explanation
        assert "semantic" in explanation.lower()
        assert ("relevance" in explanation.lower() or "相关性" in explanation)
    
    def test_extract_decision_rationale_multiple_levels(self, explainer, sample_query_analysis):
        """测试不同级别的决策理由提取"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        # 添加多个决策
        decisions = [
            DecisionRecord(
                decision_point=DecisionPoint.STRATEGY_SELECTION,
                timestamp=time.time(),
                input_data={"query_type": "CODE"},
                decision_made={"strategy": "semantic", "algorithm": "vector_search"},
                reasoning="代码查询适合语义匹配",
                confidence=0.8
            ),
            DecisionRecord(
                decision_point=DecisionPoint.RESULT_FUSION,
                timestamp=time.time(),
                input_data={"result_count": 5},
                decision_made={"method": "weighted_score"},
                reasoning="采用加权分数融合提高准确性",
                confidence=0.85,
                execution_time=0.05
            )
        ]
        
        for decision in decisions:
            explainer.record_decision(path_id, decision)
        
        # 测试简单级别
        simple_rationale = explainer._extract_decision_rationale(path, ExplanationLevel.SIMPLE)
        assert len(simple_rationale) == 2
        assert any("semantic" in r for r in simple_rationale)
        
        # 测试详细级别
        detailed_rationale = explainer._extract_decision_rationale(path, ExplanationLevel.DETAILED)
        assert len(detailed_rationale) == 2
        assert any("strategy_selection" in r for r in detailed_rationale)
        assert any("0.80" in r for r in detailed_rationale)
        
        # 测试技术级别
        technical_rationale = explainer._extract_decision_rationale(path, ExplanationLevel.TECHNICAL)
        assert len(technical_rationale) == 2
        assert any("vector_search" in r for r in technical_rationale)
        assert any("0.050s" in r for r in technical_rationale)


class TestVisualizationGeneration:
    """可视化数据生成测试"""
    
    def test_generate_flow_diagram(self, explainer, sample_query_analysis, sample_retrieval_results):
        """测试流程图生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        flow_diagram = explainer._generate_flow_diagram(path, sample_retrieval_results)
        
        assert isinstance(flow_diagram, dict)
        assert "nodes" in flow_diagram
        assert "edges" in flow_diagram
        assert "layout" in flow_diagram
        
        nodes = flow_diagram["nodes"]
        assert len(nodes) >= 5  # query, analysis, strategies, fusion, output
        
        # 检查节点类型
        node_types = {node["type"] for node in nodes}
        assert "start" in node_types
        assert "process" in node_types
        assert "agent" in node_types
        assert "end" in node_types
        
        # 检查边连接
        edges = flow_diagram["edges"]
        assert len(edges) >= len(nodes) - 1  # 至少应该连通
    
    def test_generate_metrics_chart(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result):
        """测试指标图表生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        metrics_chart = explainer._generate_metrics_chart(
            path, sample_retrieval_results, sample_validation_result
        )
        
        assert isinstance(metrics_chart, dict)
        assert "performance" in metrics_chart
        assert "quality" in metrics_chart
        assert "strategy_contribution" in metrics_chart
        
        # 检查性能图表
        performance = metrics_chart["performance"]
        assert performance["type"] == "bar"
        assert "labels" in performance["data"]
        assert "values" in performance["data"]
        assert len(performance["data"]["labels"]) == len(performance["data"]["values"])
        
        # 检查质量图表
        quality = metrics_chart["quality"]
        assert quality["type"] == "radar"
        assert len(quality["data"]["labels"]) >= 3  # 至少3个质量维度
        
        # 检查策略贡献图表
        contribution = metrics_chart["strategy_contribution"]
        assert contribution["type"] == "pie"
        assert "semantic" in contribution["data"]["labels"]
        assert "keyword" in contribution["data"]["labels"]
    
    def test_generate_timeline(self, explainer, sample_query_analysis, sample_decision_record):
        """测试时间线生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        explainer.record_decision(path_id, sample_decision_record)
        explainer.finish_path_recording(path_id, 2.0, 3)
        
        timeline = explainer._generate_timeline(path)
        
        assert isinstance(timeline, list)
        assert len(timeline) >= 3  # start, decision, end
        
        # 检查时间线排序
        timestamps = [event["timestamp"] for event in timeline]
        assert timestamps == sorted(timestamps)
        
        # 检查事件类型
        event_types = {event["type"] for event in timeline}
        assert "start" in event_types
        assert "decision" in event_types
        assert "end" in event_types
        
        # 检查具体内容
        start_event = next(e for e in timeline if e["type"] == "start")
        assert sample_query_analysis.query_text in start_event["description"]
    
    def test_generate_path_visualization(self, explainer, sample_query_analysis, sample_decision_record):
        """测试路径可视化生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        # 添加决策记录
        explainer.record_decision(path_id, sample_decision_record)
        
        # 添加失败决策
        failure_decision = DecisionRecord(
            decision_point=DecisionPoint.RETRIEVAL_EXECUTION,
            timestamp=time.time(),
            input_data={},
            decision_made={},
            reasoning="检索执行",
            confidence=0.6,
            success=False,
            error_message="网络超时"
        )
        explainer.record_decision(path_id, failure_decision)
        explainer.finish_path_recording(path_id, 1.5, 2)
        
        visualization = explainer._generate_path_visualization(path)
        
        assert isinstance(visualization, dict)
        assert visualization["path_id"] == path_id
        assert visualization["total_steps"] == 2
        assert visualization["success_rate"] == 0.5  # 1成功，1失败
        assert visualization["execution_time"] == 1.5
        
        steps = visualization["steps"]
        assert len(steps) == 2
        
        # 检查成功步骤
        success_step = next(s for s in steps if s["success"])
        assert success_step["confidence"] == 0.85
        
        # 检查失败步骤
        failure_step = next(s for s in steps if not s["success"])
        assert "error" in failure_step
        assert failure_step["error"] == "网络超时"


class TestImprovementSuggestions:
    """改进建议测试"""
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_low_confidence(self, explainer):
        """测试低置信度的改进建议生成"""
        query_analysis = QueryAnalysis(
            query_text="模糊查询",
            intent_type=QueryIntent.CREATIVE,
            confidence=0.4,
            complexity_score=0.3,
            entities=[],
            keywords=["模糊"],
            domain=None,
            sentiment="neutral",
            language="zh"
        )
        
        path_id = explainer.start_path_recording(query_analysis)
        path = explainer.path_records[path_id]
        
        # 创建低质量结果
        poor_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="模糊查询",
                results=[],
                score=0.3,
                confidence=0.4,
                processing_time=0.1
            )
        ]
        
        # 创建低置信度分析
        confidence_analysis = ConfidenceAnalysis(
            overall_confidence=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_breakdown={
                "query_analysis": 0.4,
                "retrieval": 0.3,
                "validation": 0.5
            },
            uncertainty_sources=["查询意图不明确", "检索结果稀少"]
        )
        
        suggestions = await explainer._generate_improvement_suggestions(
            path, poor_results, None, confidence_analysis
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 2
        
        # 应该包含针对性建议
        suggestions_text = " ".join(suggestions)
        assert "置信度" in suggestions_text
        assert "查询" in suggestions_text or "关键词" in suggestions_text
    
    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_no_results(self, explainer, sample_query_analysis):
        """测试无结果时的改进建议生成"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        confidence_analysis = ConfidenceAnalysis(
            overall_confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH
        )
        
        suggestions = await explainer._generate_improvement_suggestions(
            path, [], None, confidence_analysis  # 空结果
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 3
        
        suggestions_text = " ".join(suggestions)
        assert "未检索到结果" in suggestions_text
        assert "简化查询" in suggestions_text or "通用" in suggestions_text
        assert "拼写" in suggestions_text or "英文" in suggestions_text
    
    def test_suggest_alternative_approaches_by_intent(self, explainer):
        """测试基于查询意图的替代方法建议"""
        # 测试代码查询
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
        
        path_id = explainer.start_path_recording(code_query)
        path = explainer.path_records[path_id]
        
        alternatives = explainer._suggest_alternative_approaches(path, [])
        
        assert isinstance(alternatives, list)
        alternatives_text = " ".join(alternatives)
        assert "API" in alternatives_text or "函数" in alternatives_text
        assert "官方文档" in alternatives_text
        
        # 测试事实查询
        factual_query = QueryAnalysis(
            query_text="什么是机器学习",
            intent_type=QueryIntent.FACTUAL,
            confidence=0.8,
            complexity_score=0.3,
            entities=[],
            keywords=[],
            domain="技术",
            sentiment="neutral",
            language="zh"
        )
        
        path_id2 = explainer.start_path_recording(factual_query)
        path2 = explainer.path_records[path_id2]
        
        alternatives2 = explainer._suggest_alternative_approaches(path2, [])
        alternatives_text2 = " ".join(alternatives2)
        assert "定义" in alternatives_text2 or "概念" in alternatives_text2
        assert "权威" in alternatives_text2 or "学术" in alternatives_text2
    
    def test_suggest_alternative_approaches_unused_strategies(self, explainer, sample_query_analysis):
        """测试基于未使用策略的替代建议"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        path = explainer.path_records[path_id]
        
        # 只使用语义检索
        partial_results = [
            RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query="测试查询",
                results=[],
                score=0.7,
                confidence=0.8,
                processing_time=0.1
            )
        ]
        
        alternatives = explainer._suggest_alternative_approaches(path, partial_results)
        
        alternatives_text = " ".join(alternatives)
        # 应该建议未使用的策略或其他方法
        assert len(alternatives) > 0  # 至少应该有一些建议


class TestDebugAndUtilities:
    """调试和工具功能测试"""
    
    @pytest.mark.asyncio
    async def test_generate_debug_report_success(self, explainer, sample_query_analysis, sample_decision_record):
        """测试生成调试报告成功"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        explainer.record_decision(path_id, sample_decision_record)
        explainer.finish_path_recording(path_id, 1.2, 3)
        
        report = await explainer.generate_debug_report(path_id)
        
        assert isinstance(report, dict)
        assert report["path_id"] == path_id
        assert report["query"] == sample_query_analysis.query_text
        
        # 检查分析部分
        analysis = report["analysis"]
        assert analysis["intent_type"] == sample_query_analysis.intent_type.value
        assert analysis["confidence"] == sample_query_analysis.confidence
        assert analysis["entities"] == sample_query_analysis.entities
        
        # 检查执行部分
        execution = report["execution"]
        assert execution["total_time"] == 1.2
        assert execution["final_results_count"] == 3
        assert execution["decision_count"] == 1
        
        # 检查决策部分
        decisions = report["decisions"]
        assert len(decisions) == 1
        assert decisions[0]["decision_point"] == sample_decision_record.decision_point.value
    
    @pytest.mark.asyncio
    async def test_generate_debug_report_not_found(self, explainer):
        """测试生成不存在路径的调试报告"""
        report = await explainer.generate_debug_report("invalid_path_id")
        
        assert isinstance(report, dict)
        assert "error" in report
        assert report["error"] == "Path record not found"


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_explanation_workflow(self, explainer, sample_query_analysis, sample_retrieval_results, sample_validation_result, sample_composed_context):
        """测试完整的解释工作流"""
        # 1. 开始路径记录
        path_id = explainer.start_path_recording(sample_query_analysis)
        
        # 2. 记录多个决策点
        decisions = [
            DecisionRecord(
                decision_point=DecisionPoint.QUERY_ANALYSIS,
                timestamp=time.time(),
                input_data={"query": sample_query_analysis.query_text},
                decision_made={"intent": "CODE", "complexity": 0.6},
                reasoning="分析查询为代码类型，复杂度中等",
                confidence=0.85
            ),
            DecisionRecord(
                decision_point=DecisionPoint.STRATEGY_SELECTION,
                timestamp=time.time() + 0.1,
                input_data={"intent": "CODE", "entities": sample_query_analysis.entities},
                decision_made={"strategies": ["semantic", "keyword"], "weights": [0.7, 0.3]},
                reasoning="选择语义主导的多策略检索",
                confidence=0.8
            ),
            DecisionRecord(
                decision_point=DecisionPoint.RESULT_FUSION,
                timestamp=time.time() + 0.5,
                input_data={"result_count": 3},
                decision_made={"fusion_method": "weighted_score", "threshold": 0.3},
                reasoning="使用加权分数融合，过滤低分结果",
                confidence=0.9
            )
        ]
        
        for decision in decisions:
            explainer.record_decision(path_id, decision)
        
        # 3. 完成路径记录
        explainer.finish_path_recording(path_id, 2.1, 3)
        
        # 4. Mock LLM调用
        explainer.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="建议优化查询词语以提高精确度"))]
        ))
        
        # 5. 生成完整解释
        explanation = await explainer.explain_retrieval_process(
            path_id, 
            sample_retrieval_results, 
            sample_validation_result,
            sample_composed_context,
            ExplanationLevel.DETAILED
        )
        
        # 6. 验证解释结果
        assert isinstance(explanation, ExplanationOutput)
        assert explanation.explanation_id.startswith("explanation_")
        assert explanation.query == sample_query_analysis.query_text
        assert explanation.explanation_level == ExplanationLevel.DETAILED
        
        # 验证置信度分析
        assert 0.0 <= explanation.confidence_analysis.overall_confidence <= 1.0
        assert len(explanation.confidence_analysis.confidence_breakdown) >= 3
        
        # 验证解释内容
        assert len(explanation.summary) > 50
        assert "查询分析阶段" in explanation.detailed_explanation
        assert len(explanation.decision_rationale) == 3  # 3个决策
        
        # 验证可视化数据
        assert len(explanation.flow_diagram["nodes"]) >= 5
        assert len(explanation.flow_diagram["edges"]) >= 4
        assert "performance" in explanation.metrics_chart
        assert len(explanation.timeline) >= 5  # start + 3 decisions + end
        
        # 验证建议
        assert isinstance(explanation.improvement_suggestions, list)
        assert isinstance(explanation.alternative_approaches, list)
        
        # 7. 生成调试报告
        debug_report = await explainer.generate_debug_report(path_id)
        assert debug_report["path_id"] == path_id
        assert debug_report["execution"]["decision_count"] == 3
    
    @pytest.mark.asyncio
    async def test_multiple_explanation_levels(self, explainer, sample_query_analysis, sample_retrieval_results):
        """测试不同解释级别的效果"""
        path_id = explainer.start_path_recording(sample_query_analysis)
        explainer.finish_path_recording(path_id, 1.0, 2)
        
        # Mock LLM调用
        explainer.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="模拟建议"))]
        ))
        
        # 测试不同级别
        levels = [ExplanationLevel.SIMPLE, ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]
        explanations = {}
        
        for level in levels:
            explanation = await explainer.explain_retrieval_process(
                path_id, sample_retrieval_results, explanation_level=level
            )
            explanations[level] = explanation
        
        # 验证不同级别的复杂度
        simple_summary = explanations[ExplanationLevel.SIMPLE].summary
        detailed_summary = explanations[ExplanationLevel.DETAILED].summary
        technical_summary = explanations[ExplanationLevel.TECHNICAL].summary
        
        # 简单级别应该更短更通俗
        assert len(simple_summary) < len(detailed_summary)
        assert "2种检索策略" in simple_summary
        
        # 详细级别应该包含更多技术信息
        assert ("语义" in detailed_summary or "semantic" in detailed_summary) and ("关键词" in detailed_summary or "keyword" in detailed_summary)
        assert "耗时" in detailed_summary
        
        # 技术级别应该包含算法细节
        assert "复杂度" in technical_summary or "并行" in technical_summary


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])