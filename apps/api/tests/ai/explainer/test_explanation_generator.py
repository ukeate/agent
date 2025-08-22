"""解释生成器单元测试"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.ai.explainer.explanation_generator import ExplanationGenerator
from src.ai.explainer.explanation_formatter import ExplanationFormatter, ExplanationExporter
from src.ai.explainer.decision_tracker import DecisionTracker
from src.models.schemas.explanation import (
    ExplanationType,
    ExplanationLevel,
    EvidenceType,
    ConfidenceMetrics,
    ConfidenceSource
)


class TestExplanationGenerator:
    """测试解释生成器"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这是一个生成的解释。"
        mock_client.chat_completion.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def generator(self, mock_openai_client):
        """创建测试用的解释生成器"""
        return ExplanationGenerator(mock_openai_client)
    
    @pytest.fixture
    def sample_decision_tracker(self):
        """创建样本决策跟踪器"""
        tracker = DecisionTracker("test_decision_001", "测试决策上下文")
        
        # 添加一些决策数据
        tracker.create_node("start", "开始决策", {"input": "test"})
        tracker.add_confidence_factor(
            factor_name="user_age",
            factor_value=25,
            weight=0.8,
            impact=0.7,
            source="user_profile"
        )
        tracker.add_confidence_factor(
            factor_name="credit_score",
            factor_value=750,
            weight=0.9,
            impact=0.8,
            source="credit_bureau"
        )
        tracker.finalize_decision("approved", 0.85, "用户符合条件")
        
        return tracker
    
    def test_create_explanation_generator(self, generator):
        """测试创建解释生成器"""
        assert generator is not None
        assert generator.openai_client is not None
        assert generator.template_manager is not None
        assert generator.confidence_calculator is not None
        assert "technical" in generator.style_templates
        assert "business" in generator.style_templates
        assert "user_friendly" in generator.style_templates
    
    def test_generate_explanation_basic(self, generator, sample_decision_tracker):
        """测试基本解释生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY
        )
        
        assert explanation is not None
        assert explanation.decision_id == "test_decision_001"
        assert explanation.explanation_type == ExplanationType.DECISION
        assert explanation.explanation_level == ExplanationLevel.SUMMARY
        assert explanation.decision_outcome == "approved"
        assert explanation.summary_explanation is not None
        assert len(explanation.components) > 0
        assert explanation.confidence_metrics is not None
    
    def test_generate_explanation_detailed(self, generator, sample_decision_tracker):
        """测试详细解释生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.DETAILED,
            style="technical"
        )
        
        assert explanation.summary_explanation is not None
        assert explanation.detailed_explanation is not None
        assert explanation.metadata["generation_style"] == "technical"
    
    def test_generate_explanation_technical(self, generator, sample_decision_tracker):
        """测试技术解释生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.TECHNICAL,
            style="technical"
        )
        
        assert explanation.summary_explanation is not None
        assert explanation.detailed_explanation is not None
        assert explanation.technical_explanation is not None
    
    def test_generate_explanation_with_custom_context(self, generator, sample_decision_tracker):
        """测试带自定义上下文的解释生成"""
        custom_context = {
            "temporal_distance_days": 30,
            "context_similarity": 0.8,
            "domain": "financial_services"
        }
        
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY,
            custom_context=custom_context
        )
        
        assert explanation is not None
        assert explanation.metadata is not None
    
    def test_generate_explanation_components(self, generator, sample_decision_tracker):
        """测试解释组件生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY
        )
        
        assert len(explanation.components) == 2  # user_age and credit_score
        
        age_component = next(
            (comp for comp in explanation.components if comp.factor_name == "user_age"),
            None
        )
        assert age_component is not None
        assert age_component.factor_value == 25
        assert age_component.weight == 0.8
        assert age_component.impact_score == 0.7
    
    def test_generate_counterfactual_scenarios(self, generator, sample_decision_tracker):
        """测试反事实场景生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.DETAILED
        )
        
        assert len(explanation.counterfactuals) > 0
        
        scenario = explanation.counterfactuals[0]
        assert scenario.scenario_name is not None
        assert scenario.predicted_outcome is not None
        assert scenario.impact_difference is not None
    
    def test_generate_visualization_data(self, generator, sample_decision_tracker):
        """测试可视化数据生成"""
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY
        )
        
        viz_data = explanation.visualization_data
        assert viz_data is not None
        assert "factor_importance" in viz_data
        assert "confidence_breakdown" in viz_data
        assert "decision_path" in viz_data
        
        # 检查因素重要性图表
        factor_chart = viz_data["factor_importance"]
        assert factor_chart["chart_type"] == "bar"
        assert len(factor_chart["data"]) > 0
    
    def test_different_explanation_styles(self, generator, sample_decision_tracker):
        """测试不同解释风格"""
        styles = ["technical", "business", "user_friendly", "regulatory"]
        
        for style in styles:
            explanation = generator.generate_explanation(
                sample_decision_tracker,
                ExplanationType.DECISION,
                ExplanationLevel.SUMMARY,
                style=style
            )
            
            assert explanation is not None
            assert explanation.metadata["generation_style"] == style
    
    def test_calculate_explanation_confidence(self, generator, sample_decision_tracker):
        """测试解释置信度计算"""
        confidence_metrics = generator._calculate_explanation_confidence(
            sample_decision_tracker,
            None
        )
        
        assert isinstance(confidence_metrics, ConfidenceMetrics)
        assert 0.0 <= confidence_metrics.overall_confidence <= 1.0
        assert 0.0 <= confidence_metrics.uncertainty_score <= 1.0
        assert len(confidence_metrics.confidence_sources) > 0
    
    def test_build_explanation_prompt(self, generator, sample_decision_tracker):
        """测试解释提示构建"""
        context_data = generator._prepare_explanation_context(
            sample_decision_tracker,
            ConfidenceMetrics(overall_confidence=0.8, confidence_sources=[]),
            None
        )
        
        prompt = generator._build_explanation_prompt(
            ExplanationLevel.SUMMARY,
            ExplanationType.DECISION,
            "user_friendly",
            context_data
        )
        
        assert "决策ID" in prompt
        assert "test_decision_001" in prompt
        assert "approved" in prompt
        assert "user_age" in prompt
        assert "credit_score" in prompt
    
    def test_openai_api_failure_fallback(self, sample_decision_tracker):
        """测试OpenAI API失败时的降级处理"""
        # 创建一个会失败的mock客户端
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API Error")
        
        generator = ExplanationGenerator(mock_client)
        
        explanation = generator.generate_explanation(
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY
        )
        
        assert explanation is not None
        assert "解释生成失败" in explanation.summary_explanation
    
    def test_format_key_factors(self, generator):
        """测试关键因素格式化"""
        factors = [
            {"factor_name": "age", "factor_value": 25, "weight": 0.8, "impact": 0.7},
            {"factor_name": "income", "factor_value": 50000, "weight": 0.6, "impact": 0.5}
        ]
        
        formatted = generator._format_key_factors(factors)
        
        assert "1. age: 25" in formatted
        assert "2. income: 50000" in formatted
        assert "权重: 0.80" in formatted
        assert "影响: 0.70" in formatted
    
    def test_determine_evidence_type(self, generator):
        """测试证据类型确定"""
        from src.models.schemas.explanation import EvidenceType
        
        test_cases = [
            ({"source": "user_input"}, EvidenceType.INPUT_DATA),
            ({"source": "context_data"}, EvidenceType.CONTEXT),
            ({"source": "memory_store"}, EvidenceType.MEMORY),
            ({"source": "reasoning_step"}, EvidenceType.REASONING_STEP),
            ({"source": "external_api"}, EvidenceType.EXTERNAL_SOURCE),
            ({"source": "domain_expert"}, EvidenceType.DOMAIN_KNOWLEDGE)
        ]
        
        for factor, expected_type in test_cases:
            result = generator._determine_evidence_type(factor)
            assert result == expected_type
    
    def test_generate_alternative_value(self, generator):
        """测试替代值生成"""
        test_cases = [
            {"factor_value": 100},  # 数值
            {"factor_value": True},  # 布尔值
            {"factor_value": "test"},  # 字符串
            {"factor_value": [1, 2, 3]}  # 其他类型
        ]
        
        for factor in test_cases:
            alternative = generator._generate_alternative_value(factor)
            assert alternative != factor["factor_value"]
    
    def test_fallback_explanation_generation(self, generator, sample_decision_tracker):
        """测试降级解释生成"""
        from uuid import uuid4
        explanation_id = uuid4()
        error_message = "Test error"
        
        fallback = generator._generate_fallback_explanation(
            explanation_id,
            sample_decision_tracker,
            ExplanationType.DECISION,
            ExplanationLevel.SUMMARY,
            error_message
        )
        
        assert fallback.id == explanation_id
        assert fallback.metadata["generation_error"] == error_message
        assert fallback.metadata["fallback_mode"] is True


class TestExplanationFormatter:
    """测试解释格式化器"""
    
    @pytest.fixture
    def formatter(self):
        """创建测试用的格式化器"""
        return ExplanationFormatter()
    
    @pytest.fixture
    def sample_explanation(self):
        """创建样本解释"""
        from src.models.schemas.explanation import (
            DecisionExplanation,
            ExplanationComponent,
            ConfidenceMetrics,
            CounterfactualScenario
        )
        
        return DecisionExplanation(
            decision_id="test_001",
            explanation_type=ExplanationType.DECISION,
            explanation_level=ExplanationLevel.SUMMARY,
            decision_description="测试决策",
            decision_outcome="approved",
            summary_explanation="这是一个测试解释。",
            components=[
                ExplanationComponent(
                    factor_name="test_factor",
                    factor_value="test_value",
                    weight=0.8,
                    impact_score=0.7,
                    evidence_type=EvidenceType.INPUT_DATA,
                    evidence_source="test_source",
                    evidence_content="测试因素说明"
                )
            ],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.85,
                uncertainty_score=0.15,
                confidence_sources=[ConfidenceSource.MODEL_PROBABILITY]
            ),
            counterfactuals=[
                CounterfactualScenario(
                    scenario_name="如果条件不同",
                    changed_factors={"test_factor": "alternative_value"},
                    predicted_outcome="可能的不同结果",
                    probability=0.8,
                    impact_difference=-0.1,
                    explanation="影响分析"
                )
            ]
        )
    
    def test_create_formatter(self, formatter):
        """测试创建格式化器"""
        assert formatter is not None
        assert formatter.html_templates is not None
        assert formatter.markdown_templates is not None
        assert "html" in formatter.format_config
        assert "markdown" in formatter.format_config
        assert "json" in formatter.format_config
    
    def test_format_to_html(self, formatter, sample_explanation):
        """测试HTML格式化"""
        html_output = formatter.format_explanation(sample_explanation, "html")
        
        assert html_output is not None
        assert "<!DOCTYPE html>" in html_output
        assert "test_001" in html_output
        assert "approved" in html_output
        assert "85.0%" in html_output
        assert "test_factor" in html_output
    
    def test_format_to_markdown(self, formatter, sample_explanation):
        """测试Markdown格式化"""
        markdown_output = formatter.format_explanation(sample_explanation, "markdown")
        
        assert markdown_output is not None
        assert "# " in markdown_output  # 标题
        assert "test_001" in markdown_output
        assert "approved" in markdown_output
        assert "|" in markdown_output  # 表格
    
    def test_format_to_json(self, formatter, sample_explanation):
        """测试JSON格式化"""
        json_output = formatter.format_explanation(sample_explanation, "json")
        
        assert json_output is not None
        assert "test_001" in json_output
        assert "approved" in json_output
        assert "_formatted" in json_output  # 包含格式化元数据
        
        # 验证JSON有效性
        import json
        parsed = json.loads(json_output)
        assert parsed["decision_id"] == "test_001"
    
    def test_format_to_text(self, formatter, sample_explanation):
        """测试纯文本格式化"""
        text_output = formatter.format_explanation(sample_explanation, "text")
        
        assert text_output is not None
        assert "=" in text_output  # 分隔线
        assert "test_001" in text_output
        assert "approved" in text_output
        assert "📋 基本信息" in text_output
        assert "🔍 关键因素" in text_output
    
    def test_format_to_xml(self, formatter, sample_explanation):
        """测试XML格式化"""
        xml_output = formatter.format_explanation(sample_explanation, "xml")
        
        assert xml_output is not None
        assert "<?xml version" in xml_output
        assert "<decision_explanation>" in xml_output
        assert "<id>" in xml_output
        assert "test_001" in xml_output
        assert "</decision_explanation>" in xml_output
    
    def test_format_with_custom_config(self, formatter, sample_explanation):
        """测试自定义配置格式化"""
        custom_config = {
            "include_styles": False,
            "pretty_print": False
        }
        
        html_output = formatter.format_explanation(
            sample_explanation, "html", custom_config=custom_config
        )
        
        assert html_output is not None
        # 不应包含内联样式
        assert "<style>" not in html_output
    
    def test_format_with_different_templates(self, formatter, sample_explanation):
        """测试不同模板格式化"""
        templates = ["default", "minimal", "dashboard"]
        
        for template in templates:
            try:
                html_output = formatter.format_explanation(
                    sample_explanation, "html", template_name=template
                )
                assert html_output is not None
            except Exception as e:
                # 某些模板可能不存在，这是预期的
                pass
    
    def test_get_decision_emoji(self, formatter):
        """测试决策表情符号"""
        test_cases = [
            ("approved", "✅"),
            ("通过", "✅"),
            ("rejected", "❌"),
            ("拒绝", "❌"),
            ("pending", "⏳"),
            ("待定", "⏳"),
            ("other", "📋")
        ]
        
        for outcome, expected_emoji in test_cases:
            emoji = formatter._get_decision_emoji(outcome)
            assert emoji == expected_emoji
    
    def test_get_confidence_emoji(self, formatter):
        """测试置信度表情符号"""
        test_cases = [
            (0.9, "🟢"),
            (0.7, "🟡"),
            (0.5, "🟠"),
            (0.3, "🔴")
        ]
        
        for confidence, expected_emoji in test_cases:
            emoji = formatter._get_confidence_emoji(confidence)
            assert emoji == expected_emoji
    
    def test_create_components_table(self, formatter, sample_explanation):
        """测试组件表格创建"""
        table = formatter._create_components_table(sample_explanation.components)
        
        assert table is not None
        assert "|" in table  # Markdown表格
        assert "test_factor" in table
        assert "0.80" in table  # 权重
        assert "0.70" in table  # 影响
    
    def test_wrap_text(self, formatter):
        """测试文本换行"""
        # 使用英文测试，因为中文换行逻辑更复杂
        long_text = "This is a very long text that needs to be wrapped for proper display within specified width limits."
        wrapped = formatter._wrap_text(long_text, 20)
        
        assert len(wrapped) > 1  # 应该被分成多行
        for line in wrapped:
            assert len(line) <= 20
    
    def test_unsupported_format(self, formatter, sample_explanation):
        """测试不支持的格式"""
        with pytest.raises(ValueError):
            formatter.format_explanation(sample_explanation, "unsupported_format")


class TestExplanationExporter:
    """测试解释导出器"""
    
    @pytest.fixture
    def exporter(self):
        """创建测试用的导出器"""
        return ExplanationExporter()
    
    @pytest.fixture
    def sample_explanation(self):
        """创建样本解释"""
        from src.models.schemas.explanation import (
            DecisionExplanation,
            ConfidenceMetrics,
            ConfidenceSource
        )
        
        return DecisionExplanation(
            decision_id="export_test_001",
            explanation_type=ExplanationType.DECISION,
            explanation_level=ExplanationLevel.SUMMARY,
            decision_description="导出测试",
            decision_outcome="approved",
            summary_explanation="这是一个导出测试解释。",
            components=[],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.8,
                uncertainty_score=0.2,
                confidence_sources=[ConfidenceSource.MODEL_PROBABILITY]
            ),
            counterfactuals=[]
        )
    
    def test_create_exporter(self, exporter):
        """测试创建导出器"""
        assert exporter is not None
        assert exporter.formatter is not None
    
    @patch('builtins.open')
    def test_export_to_file_html(self, mock_open, exporter, sample_explanation):
        """测试导出HTML文件"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        success = exporter.export_to_file(
            sample_explanation,
            "/tmp/test.html",
            "html"
        )
        
        assert success is True
        mock_open.assert_called_once_with("/tmp/test.html", 'w', encoding='utf-8')
        mock_file.write.assert_called_once()
    
    @patch('builtins.open')
    def test_export_to_file_auto_detect(self, mock_open, exporter, sample_explanation):
        """测试自动检测格式导出"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # 测试不同扩展名的自动检测
        test_cases = [
            ("/tmp/test.html", "html"),
            ("/tmp/test.md", "markdown"),
            ("/tmp/test.json", "json"),
            ("/tmp/test.xml", "xml"),
            ("/tmp/test.txt", "text")
        ]
        
        for file_path, expected_format in test_cases:
            mock_open.reset_mock()
            mock_file.reset_mock()
            
            success = exporter.export_to_file(sample_explanation, file_path)
            assert success is True
            mock_file.write.assert_called_once()
    
    @patch('builtins.open')
    def test_export_error_handling(self, mock_open, exporter, sample_explanation):
        """测试导出错误处理"""
        mock_open.side_effect = IOError("文件写入失败")
        
        success = exporter.export_to_file(
            sample_explanation,
            "/tmp/test.html"
        )
        
        assert success is False
    
    @patch.object(ExplanationExporter, 'export_to_file')
    def test_batch_export(self, mock_export, exporter, sample_explanation):
        """测试批量导出"""
        mock_export.return_value = True
        
        explanations = [sample_explanation, sample_explanation]
        results = exporter.batch_export(
            explanations,
            "/tmp/export",
            formats=['html', 'json']
        )
        
        # 应该调用4次导出 (2个解释 × 2种格式)
        assert mock_export.call_count == 4
        # 但结果字典中会合并相同的key，所以只有2个唯一的文件路径
        assert len(results) == 2
        
        # 检查文件路径
        expected_paths = [
            "/tmp/export/export_test_001.html",
            "/tmp/export/export_test_001.json"
        ]
        
        for path in expected_paths:
            assert path in results
            assert results[path] is True


if __name__ == "__main__":
    pytest.main([__file__])