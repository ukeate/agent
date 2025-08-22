"""CoT推理解释器单元测试"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.ai.explainer.cot_reasoning_explainer import (
    CoTReasoningExplainer,
    ReasoningStep,
    ReasoningChain
)
from src.ai.explainer.decision_tracker import DecisionTracker
from src.models.schemas.explanation import (
    ExplanationType,
    ExplanationLevel,
    EvidenceType
)


class TestReasoningStep:
    """测试推理步骤"""
    
    def test_create_reasoning_step(self):
        """测试创建推理步骤"""
        step = ReasoningStep(
            step_id="step_1",
            step_type="问题分解",
            description="分解问题为子问题",
            input_data={"problem": "复杂决策"},
            reasoning_process="将复杂问题分解为多个子问题",
            output_data={"sub_problems": ["子问题1", "子问题2"]},
            confidence=0.8
        )
        
        assert step.step_id == "step_1"
        assert step.step_type == "问题分解"
        assert step.confidence == 0.8
        assert "sub_problems" in step.output_data
        assert step.timestamp is not None


class TestReasoningChain:
    """测试推理链"""
    
    def test_create_reasoning_chain(self):
        """测试创建推理链"""
        chain = ReasoningChain("chain_001", "测试问题")
        
        assert chain.chain_id == "chain_001"
        assert chain.problem_statement == "测试问题"
        assert len(chain.steps) == 0
        assert chain.overall_confidence == 0.0
    
    def test_add_step_to_chain(self):
        """测试向推理链添加步骤"""
        chain = ReasoningChain("chain_001", "测试问题")
        
        step1 = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="分析步骤",
            input_data={},
            reasoning_process="分析过程",
            output_data={},
            confidence=0.8
        )
        
        step2 = ReasoningStep(
            step_id="step_2", 
            step_type="推导",
            description="推导步骤",
            input_data={},
            reasoning_process="推导过程",
            output_data={},
            confidence=0.6
        )
        
        chain.add_step(step1)
        chain.add_step(step2)
        
        assert len(chain.steps) == 2
        assert chain.overall_confidence == 0.7  # (0.8 + 0.6) / 2
    
    def test_get_reasoning_path(self):
        """测试获取推理路径"""
        chain = ReasoningChain("chain_001", "测试问题")
        
        step = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="分析步骤",
            input_data={},
            reasoning_process="分析过程",
            output_data={},
            confidence=0.8
        )
        
        chain.add_step(step)
        path = chain.get_reasoning_path()
        
        assert len(path) == 1
        assert path[0]["step_id"] == "step_1"
        assert path[0]["step_type"] == "分析"
        assert path[0]["confidence"] == 0.8


class TestCoTReasoningExplainer:
    """测试CoT推理解释器"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这是一个生成的推理步骤。"
        mock_client.chat_completion.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def explainer(self, mock_openai_client):
        """创建测试用的CoT解释器"""
        return CoTReasoningExplainer(mock_openai_client)
    
    @pytest.fixture
    def sample_decision_tracker(self):
        """创建样本决策跟踪器"""
        tracker = DecisionTracker("test_decision_001", "贷款审批决策")
        
        # 添加一些决策数据
        tracker.create_node("start", "开始审批流程", {"application_id": "APP001"})
        tracker.add_confidence_factor(
            factor_name="credit_score",
            factor_value=750,
            weight=0.8,
            impact=0.7,
            source="credit_bureau"
        )
        tracker.add_confidence_factor(
            factor_name="income_level",
            factor_value=60000,
            weight=0.7,
            impact=0.6,
            source="financial_statement"
        )
        tracker.finalize_decision("approved", 0.85, "申请通过")
        
        return tracker
    
    def test_create_cot_explainer(self, explainer):
        """测试创建CoT解释器"""
        assert explainer is not None
        assert explainer.openai_client is not None
        assert "analytical" in explainer.reasoning_modes
        assert "deductive" in explainer.reasoning_modes
        assert "inductive" in explainer.reasoning_modes
        assert "abductive" in explainer.reasoning_modes
    
    def test_reasoning_modes_configuration(self, explainer):
        """测试推理模式配置"""
        analytical_mode = explainer.reasoning_modes["analytical"]
        assert analytical_mode["description"] == "分析性推理"
        assert "steps" in analytical_mode
        assert len(analytical_mode["steps"]) > 0
        
        deductive_mode = explainer.reasoning_modes["deductive"]
        assert deductive_mode["description"] == "演绎推理"
        assert "前提确立" in deductive_mode["steps"]
    
    def test_generate_cot_explanation_basic(self, explainer, sample_decision_tracker):
        """测试基本CoT解释生成"""
        reasoning_chain, explanation = explainer.generate_cot_explanation(
            sample_decision_tracker,
            reasoning_mode="analytical",
            explanation_level=ExplanationLevel.DETAILED
        )
        
        assert reasoning_chain is not None
        assert explanation is not None
        assert reasoning_chain.chain_id is not None
        assert explanation.explanation_type == ExplanationType.REASONING
        assert explanation.decision_id == "test_decision_001"
        assert len(reasoning_chain.steps) > 0
    
    def test_generate_cot_explanation_different_modes(self, explainer, sample_decision_tracker):
        """测试不同推理模式的CoT解释生成"""
        modes = ["analytical", "deductive", "inductive", "abductive"]
        
        for mode in modes:
            reasoning_chain, explanation = explainer.generate_cot_explanation(
                sample_decision_tracker,
                reasoning_mode=mode,
                explanation_level=ExplanationLevel.SUMMARY
            )
            
            assert reasoning_chain is not None
            assert explanation is not None
            assert explanation.metadata is not None
            assert "reasoning_type" in explanation.metadata
    
    def test_create_reasoning_chain(self, explainer, sample_decision_tracker):
        """测试创建推理链"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        assert reasoning_chain is not None
        assert "test_decision_001" in reasoning_chain.chain_id
        assert "analytical" in reasoning_chain.chain_id
        assert reasoning_chain.problem_statement is not None
        assert "贷款审批决策" in reasoning_chain.problem_statement
    
    def test_build_problem_statement(self, explainer, sample_decision_tracker):
        """测试构建问题陈述"""
        problem_statement = explainer._build_problem_statement(sample_decision_tracker)
        
        assert problem_statement is not None
        assert "贷款审批决策" in problem_statement
        assert "2个置信度因子" in problem_statement
        assert "决策问题" in problem_statement
    
    def test_execute_reasoning_steps(self, explainer, sample_decision_tracker):
        """测试执行推理步骤"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        explainer._execute_reasoning_steps(
            reasoning_chain, sample_decision_tracker, "analytical"
        )
        
        assert len(reasoning_chain.steps) > 0
        # analytical模式有4个步骤
        assert len(reasoning_chain.steps) == 4
        
        # 检查步骤类型
        step_types = [step.step_type for step in reasoning_chain.steps]
        assert "问题分解" in step_types
        assert "证据收集" in step_types
        assert "因果分析" in step_types
        assert "结论推导" in step_types
    
    def test_execute_single_reasoning_step(self, explainer, sample_decision_tracker):
        """测试执行单个推理步骤"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        step = explainer._execute_single_reasoning_step(
            "问题分解",
            reasoning_chain,
            sample_decision_tracker,
            "analytical",
            1
        )
        
        assert step is not None
        assert step.step_id == "step_1_问题分解"
        assert step.step_type == "问题分解"
        assert step.confidence > 0
        assert step.reasoning_process is not None
        assert step.input_data is not None
        assert step.output_data is not None
    
    def test_prepare_step_input(self, explainer, sample_decision_tracker):
        """测试准备步骤输入数据"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        input_data = explainer._prepare_step_input(
            "问题分解", sample_decision_tracker, reasoning_chain
        )
        
        assert "decision_context" in input_data
        assert "confidence_factors" in input_data
        assert "decision_path" in input_data
        assert "previous_steps" in input_data
        assert input_data["decision_context"] == "贷款审批决策"
        assert len(input_data["confidence_factors"]) == 2
    
    def test_calculate_step_confidence(self, explainer):
        """测试计算步骤置信度"""
        input_data = {
            "confidence_factors": [{"factor": "test1"}, {"factor": "test2"}]
        }
        output_data = {
            "step_conclusion": "结论",
            "key_insights": ["洞察1"],
            "identified_factors": ["因子1"],
            "uncertainty_notes": ["不确定性1"]
        }
        reasoning_process = "这是一个测试推理过程" * 10  # 确保长度
        
        confidence = explainer._calculate_step_confidence(
            "测试步骤", input_data, output_data, reasoning_process
        )
        
        assert 0.1 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_generate_final_conclusion(self, explainer, sample_decision_tracker):
        """测试生成最终结论"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        # 添加一些测试步骤
        step = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="测试步骤",
            input_data={},
            reasoning_process="测试推理",
            output_data={"step_conclusion": "测试结论"},
            confidence=0.8
        )
        reasoning_chain.add_step(step)
        
        explainer._generate_final_conclusion(reasoning_chain, sample_decision_tracker)
        
        assert reasoning_chain.final_conclusion is not None
        assert len(reasoning_chain.final_conclusion) > 0
    
    def test_convert_to_explanation(self, explainer, sample_decision_tracker):
        """测试转换为解释对象"""
        reasoning_chain = explainer._create_reasoning_chain(
            sample_decision_tracker, "analytical"
        )
        
        # 添加测试步骤
        step = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="测试步骤",
            input_data={},
            reasoning_process="测试推理过程",
            output_data={"step_conclusion": "测试结论"},
            confidence=0.8
        )
        reasoning_chain.add_step(step)
        reasoning_chain.final_conclusion = "最终结论"
        
        explanation = explainer._convert_to_explanation(
            reasoning_chain, sample_decision_tracker, ExplanationLevel.DETAILED
        )
        
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.REASONING
        assert explanation.decision_id == "test_decision_001"
        assert len(explanation.components) > 0
        assert explanation.confidence_metrics is not None
        assert explanation.visualization_data is not None
        
        # 检查推理组件
        reasoning_component = explanation.components[0]
        assert reasoning_component.evidence_type == EvidenceType.REASONING_STEP
        assert reasoning_component.evidence_source == "cot_reasoning"
    
    def test_create_reasoning_components(self, explainer):
        """测试创建推理组件"""
        reasoning_chain = ReasoningChain("test_chain", "测试问题")
        
        step1 = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="分析步骤",
            input_data={},
            reasoning_process="详细的分析推理过程",
            output_data={"step_conclusion": "分析结论"},
            confidence=0.8
        )
        
        step2 = ReasoningStep(
            step_id="step_2",
            step_type="推导",
            description="推导步骤",
            input_data={},
            reasoning_process="详细的推导推理过程",
            output_data={"step_conclusion": "推导结论"},
            confidence=0.7
        )
        
        reasoning_chain.add_step(step1)
        reasoning_chain.add_step(step2)
        
        components = explainer._create_reasoning_components(reasoning_chain)
        
        assert len(components) == 2
        assert components[0].factor_name == "推理步骤1: 分析"
        assert components[1].factor_name == "推理步骤2: 推导"
        assert components[0].weight == 0.5  # 1/2
        assert components[1].weight == 0.5  # 1/2
        assert components[0].impact_score == 0.8
        assert components[1].impact_score == 0.7
    
    def test_calculate_reasoning_confidence(self, explainer):
        """测试计算推理置信度指标"""
        reasoning_chain = ReasoningChain("test_chain", "测试问题")
        reasoning_chain.overall_confidence = 0.75
        
        confidence_metrics = explainer._calculate_reasoning_confidence(reasoning_chain)
        
        assert confidence_metrics is not None
        assert confidence_metrics.overall_confidence == 0.75
        assert confidence_metrics.uncertainty_score == 0.25  # 1 - 0.75
        assert confidence_metrics.confidence_interval_lower is not None
        assert confidence_metrics.confidence_interval_upper is not None
        assert confidence_metrics.confidence_interval_lower >= 0.0
        assert confidence_metrics.confidence_interval_upper <= 1.0
    
    def test_generate_reasoning_counterfactuals(self, explainer):
        """测试生成推理反事实场景"""
        reasoning_chain = ReasoningChain("test_chain", "测试问题")
        
        step1 = ReasoningStep(
            step_id="step_1",
            step_type="高置信度分析",
            description="高置信度步骤",
            input_data={},
            reasoning_process="高置信度推理",
            output_data={},
            confidence=0.9
        )
        
        step2 = ReasoningStep(
            step_id="step_2",
            step_type="低置信度推导",
            description="低置信度步骤", 
            input_data={},
            reasoning_process="低置信度推理",
            output_data={},
            confidence=0.4
        )
        
        reasoning_chain.add_step(step1)
        reasoning_chain.add_step(step2)
        
        counterfactuals = explainer._generate_reasoning_counterfactuals(reasoning_chain)
        
        assert len(counterfactuals) == 2  # 取置信度最高的2个
        assert counterfactuals[0].scenario_name == "如果高置信度分析推理不同"
        assert counterfactuals[1].scenario_name == "如果低置信度推导推理不同"
        assert counterfactuals[0].probability == 0.1  # 1 - 0.9
        assert counterfactuals[1].probability == 0.6  # 1 - 0.4
    
    def test_generate_reasoning_visualization(self, explainer):
        """测试生成推理可视化数据"""
        reasoning_chain = ReasoningChain("test_chain", "测试问题")
        
        step = ReasoningStep(
            step_id="step_1",
            step_type="分析",
            description="分析步骤",
            input_data={},
            reasoning_process="推理过程",
            output_data={},
            confidence=0.8
        )
        reasoning_chain.add_step(step)
        
        viz_data = explainer._generate_reasoning_visualization(reasoning_chain)
        
        assert "reasoning_flow" in viz_data
        assert "confidence_progression" in viz_data
        
        flow_chart = viz_data["reasoning_flow"]
        assert flow_chart["chart_type"] == "flow"
        assert len(flow_chart["nodes"]) == 1
        assert flow_chart["nodes"][0]["id"] == "step_1"
        assert flow_chart["nodes"][0]["confidence"] == 0.8
        
        confidence_chart = viz_data["confidence_progression"]
        assert confidence_chart["chart_type"] == "line"
        assert len(confidence_chart["data"]) == 1
        assert confidence_chart["data"][0]["step"] == 1
        assert confidence_chart["data"][0]["confidence"] == 0.8
    
    def test_openai_api_failure_fallback(self, sample_decision_tracker):
        """测试OpenAI API失败时的降级处理"""
        # 创建一个会失败的mock客户端
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API Error")
        
        explainer = CoTReasoningExplainer(mock_client)
        
        reasoning_chain, explanation = explainer.generate_cot_explanation(
            sample_decision_tracker,
            reasoning_mode="analytical",
            explanation_level=ExplanationLevel.SUMMARY
        )
        
        assert reasoning_chain is not None
        assert explanation is not None
        assert "推理生成失败" in reasoning_chain.problem_statement
        assert explanation.metadata is not None
        assert explanation.metadata.get("fallback_mode") is True
    
    def test_create_fallback_explanation(self, explainer, sample_decision_tracker):
        """测试创建降级解释"""
        error_msg = "测试错误"
        
        fallback_explanation = explainer._create_fallback_explanation(
            sample_decision_tracker, error_msg
        )
        
        assert fallback_explanation is not None
        assert fallback_explanation.explanation_type == ExplanationType.REASONING
        assert fallback_explanation.decision_id == "test_decision_001"
        assert error_msg in fallback_explanation.summary_explanation
        assert fallback_explanation.metadata["error"] == error_msg
        assert fallback_explanation.metadata["fallback_mode"] is True
        assert fallback_explanation.confidence_metrics.overall_confidence < 0.5
    
    def test_extract_evidence(self, explainer, sample_decision_tracker):
        """测试提取证据数据"""
        evidence = explainer._extract_evidence(sample_decision_tracker)
        
        assert len(evidence) == 2
        assert evidence[0]["factor_name"] == "credit_score"
        assert evidence[0]["factor_value"] == 750
        assert evidence[0]["source"] == "credit_bureau"
        assert evidence[1]["factor_name"] == "income_level"
        assert evidence[1]["factor_value"] == 60000
    
    def test_get_analytical_framework(self, explainer):
        """测试获取分析框架"""
        framework = explainer._get_analytical_framework()
        
        assert "approach" in framework
        assert "criteria" in framework
        assert "method" in framework
        assert framework["approach"] == "systematic_analysis"
        assert "relevance" in framework["criteria"]
    
    def test_get_validation_criteria(self, explainer):
        """测试获取验证标准"""
        criteria = explainer._get_validation_criteria()
        
        assert len(criteria) > 0
        assert "逻辑一致性" in criteria
        assert "证据支撑度" in criteria
        assert "结论合理性" in criteria
    
    def test_format_input_data(self, explainer):
        """测试格式化输入数据"""
        input_data = {
            "confidence_factors": [{"factor1": "value1"}, {"factor2": "value2"}],
            "decision_path": [{"step1": "data"}],
            "simple_value": "test_value"
        }
        
        formatted = explainer._format_input_data(input_data)
        
        assert "confidence_factors: 2个因子" in formatted
        assert "decision_path: 1项" in formatted
        assert "simple_value: test_value" in formatted
    
    def test_extract_conclusion_from_reasoning(self, explainer):
        """测试从推理中提取结论"""
        reasoning_text = """
        首先我们分析了问题
        接下来进行了推导
        因此，我们得出结论：贷款应该被批准
        其他一些分析
        """
        
        conclusion = explainer._extract_conclusion_from_reasoning(reasoning_text)
        
        assert "贷款应该被批准" in conclusion
    
    def test_extract_insights_from_reasoning(self, explainer):
        """测试从推理中提取洞察"""
        reasoning_text = """
        分析过程中发现了重要的模式
        这个发现很关键
        另一个洞察是数据质量很高
        """
        
        insights = explainer._extract_insights_from_reasoning(reasoning_text)
        
        assert len(insights) <= 3
        assert any("重要" in insight for insight in insights)
        assert any("关键" in insight for insight in insights)
    
    def test_assess_evidence_quality(self, explainer):
        """测试评估证据质量"""
        input_data = {
            "confidence_factors": [
                {"weight": 0.8},
                {"weight": 0.6},
                {"weight": 0.9}
            ]
        }
        
        quality = explainer._assess_evidence_quality(input_data)
        
        assert "overall" in quality
        assert "completeness" in quality
        assert "reliability" in quality
        assert 0.0 <= quality["overall"] <= 1.0
        assert quality["reliability"] == 0.7666666666666667  # (0.8+0.6+0.9)/3
    
    def test_assess_conclusion_confidence(self, explainer):
        """测试评估结论置信度"""
        high_confidence_text = "我确定这个结论是正确的，明确支持这个决策"
        low_confidence_text = "这可能是正确的，或许需要更多数据，不确定"
        
        high_confidence = explainer._assess_conclusion_confidence(high_confidence_text)
        low_confidence = explainer._assess_conclusion_confidence(low_confidence_text)
        
        assert high_confidence > low_confidence
        assert 0.3 <= high_confidence <= 1.0
        assert 0.3 <= low_confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])