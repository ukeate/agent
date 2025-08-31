#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型评估报告生成器测试模块
测试HTML报告生成、数据可视化、性能分析等功能
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from ai.model_evaluation.report_generator import (
    EvaluationReportGenerator,
    ReportConfig,
    EvaluationResult,
    BenchmarkResult,
    ModelInfo,
    PerformanceStats,
    ComparisonReport
)


class TestEvaluationReportGenerator:
    """测试评估报告生成器"""

    @pytest.fixture
    def sample_model_info(self) -> ModelInfo:
        """示例模型信息"""
        return ModelInfo(
            name="test-model",
            version="1.0.0",
            architecture="transformer",
            parameters=125000000,
            size_mb=500.0,
            framework="transformers"
        )

    @pytest.fixture
    def sample_benchmark_result(self) -> BenchmarkResult:
        """示例基准测试结果"""
        return BenchmarkResult(
            benchmark_name="GLUE",
            dataset_name="cola",
            metrics={
                "accuracy": 0.85,
                "f1": 0.83,
                "precision": 0.86,
                "recall": 0.81
            },
            execution_time=120.5,
            samples_processed=1000,
            error_rate=0.02
        )

    @pytest.fixture
    def sample_performance_stats(self) -> PerformanceStats:
        """示例性能统计"""
        return PerformanceStats(
            avg_inference_time=0.15,
            max_inference_time=0.45,
            min_inference_time=0.08,
            throughput=850.0,
            memory_usage_mb=1200.0,
            gpu_utilization=0.75,
            cpu_utilization=0.45
        )

    @pytest.fixture
    def sample_evaluation_result(self, sample_model_info, sample_benchmark_result, sample_performance_stats) -> EvaluationResult:
        """示例评估结果"""
        return EvaluationResult(
            evaluation_id="eval_123",
            model_info=sample_model_info,
            benchmark_results=[sample_benchmark_result],
            performance_stats=sample_performance_stats,
            overall_score=0.84,
            timestamp=utc_now(),
            duration=timedelta(minutes=5),
            status="completed"
        )

    @pytest.fixture
    def report_config(self) -> ReportConfig:
        """报告配置"""
        return ReportConfig(
            title="模型评估报告",
            include_charts=True,
            include_detailed_metrics=True,
            include_performance_analysis=True,
            include_comparison=True,
            output_format="html",
            template_name="default"
        )

    @pytest.fixture
    def report_generator(self) -> EvaluationReportGenerator:
        """报告生成器实例"""
        return EvaluationReportGenerator()

    def test_generate_evaluation_report(self, report_generator, sample_evaluation_result, report_config):
        """测试生成评估报告"""
        results = [sample_evaluation_result]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.html"
            report_config.output_path = output_path
            
            # 生成报告
            report_path = report_generator.generate_evaluation_report(results, report_config)
            
            # 验证报告文件生成
            assert report_path.exists()
            assert report_path.suffix == '.html'
            
            # 验证报告内容
            content = report_path.read_text(encoding='utf-8')
            assert "模型评估报告" in content
            assert "test-model" in content
            assert "GLUE" in content
            assert "0.85" in content  # accuracy

    def test_generate_comparison_report(self, report_generator, sample_evaluation_result, report_config):
        """测试生成对比报告"""
        # 创建两个不同的评估结果
        result1 = sample_evaluation_result
        result2 = EvaluationResult(
            evaluation_id="eval_124",
            model_info=ModelInfo(
                name="test-model-2",
                version="2.0.0",
                architecture="transformer",
                parameters=350000000,
                size_mb=1400.0,
                framework="transformers"
            ),
            benchmark_results=[BenchmarkResult(
                benchmark_name="GLUE",
                dataset_name="cola",
                metrics={
                    "accuracy": 0.88,
                    "f1": 0.86,
                    "precision": 0.89,
                    "recall": 0.84
                },
                execution_time=180.2,
                samples_processed=1000,
                error_rate=0.015
            )],
            performance_stats=PerformanceStats(
                avg_inference_time=0.22,
                max_inference_time=0.55,
                min_inference_time=0.12,
                throughput=650.0,
                memory_usage_mb=2100.0,
                gpu_utilization=0.85,
                cpu_utilization=0.55
            ),
            overall_score=0.87,
            timestamp=utc_now(),
            duration=timedelta(minutes=8),
            status="completed"
        )
        
        results = [result1, result2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparison_report.html"
            report_config.output_path = output_path
            
            # 生成对比报告
            report_path = report_generator.generate_comparison_report(results, report_config)
            
            # 验证报告文件生成
            assert report_path.exists()
            content = report_path.read_text(encoding='utf-8')
            
            # 验证对比内容
            assert "test-model" in content
            assert "test-model-2" in content
            assert "模型对比" in content or "Model Comparison" in content

    def test_generate_performance_chart(self, report_generator, sample_evaluation_result):
        """测试生成性能图表"""
        results = [sample_evaluation_result]
        
        # 生成性能图表数据
        chart_data = report_generator._generate_performance_chart(results)
        
        assert isinstance(chart_data, dict)
        assert "labels" in chart_data
        assert "datasets" in chart_data
        assert len(chart_data["datasets"]) > 0

    def test_generate_metrics_summary(self, report_generator, sample_evaluation_result):
        """测试生成指标摘要"""
        results = [sample_evaluation_result]
        
        summary = report_generator._generate_metrics_summary(results)
        
        assert isinstance(summary, dict)
        assert "total_evaluations" in summary
        assert "avg_score" in summary
        assert "best_model" in summary
        assert summary["total_evaluations"] == 1
        assert summary["avg_score"] == 0.84

    def test_export_to_json(self, report_generator, sample_evaluation_result):
        """测试导出为JSON格式"""
        results = [sample_evaluation_result]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.json"
            
            # 导出JSON报告
            json_path = report_generator.export_to_json(results, output_path)
            
            # 验证文件生成和内容
            assert json_path.exists()
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "evaluations" in data
            assert "summary" in data
            assert len(data["evaluations"]) == 1
            assert data["evaluations"][0]["model_info"]["name"] == "test-model"

    def test_export_to_csv(self, report_generator, sample_evaluation_result):
        """测试导出为CSV格式"""
        results = [sample_evaluation_result]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.csv"
            
            # 导出CSV报告
            csv_path = report_generator.export_to_csv(results, output_path)
            
            # 验证文件生成和内容
            assert csv_path.exists()
            content = csv_path.read_text(encoding='utf-8')
            
            # 验证CSV头和数据
            assert "model_name" in content
            assert "overall_score" in content
            assert "test-model" in content
            assert "0.84" in content

    @patch('ai.model_evaluation.report_generator.jinja2.Environment')
    def test_template_rendering(self, mock_jinja_env, report_generator, sample_evaluation_result, report_config):
        """测试模板渲染"""
        mock_template = Mock()
        mock_template.render.return_value = "<html>Test Report</html>"
        mock_jinja_env.return_value.get_template.return_value = mock_template
        
        results = [sample_evaluation_result]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.html"
            report_config.output_path = output_path
            
            # 生成报告
            report_path = report_generator.generate_evaluation_report(results, report_config)
            
            # 验证模板被调用
            mock_jinja_env.return_value.get_template.assert_called_once()
            mock_template.render.assert_called_once()

    def test_error_handling_invalid_template(self, report_generator, sample_evaluation_result):
        """测试无效模板处理"""
        config = ReportConfig(
            title="Test Report",
            template_name="non_existent_template",
            output_path=Path("/tmp/test_report.html")
        )
        
        results = [sample_evaluation_result]
        
        # 应该抛出异常或使用默认模板
        with pytest.raises((FileNotFoundError, Exception)):
            report_generator.generate_evaluation_report(results, config)

    def test_empty_results_handling(self, report_generator, report_config):
        """测试空结果处理"""
        results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_report.html"
            report_config.output_path = output_path
            
            # 生成报告
            report_path = report_generator.generate_evaluation_report(results, report_config)
            
            # 验证报告仍然生成但包含空状态信息
            assert report_path.exists()
            content = report_path.read_text(encoding='utf-8')
            assert "没有评估结果" in content or "No evaluation results" in content

    def test_large_dataset_performance(self, report_generator, sample_model_info, sample_performance_stats):
        """测试大数据集性能"""
        # 创建大量评估结果
        results = []
        for i in range(100):
            result = EvaluationResult(
                evaluation_id=f"eval_{i}",
                model_info=sample_model_info,
                benchmark_results=[],
                performance_stats=sample_performance_stats,
                overall_score=0.8 + (i % 10) * 0.01,
                timestamp=utc_now(),
                duration=timedelta(minutes=i % 10 + 1),
                status="completed"
            )
            results.append(result)
        
        # 测试报告生成性能
        start_time = utc_now()
        summary = report_generator._generate_metrics_summary(results)
        end_time = utc_now()
        
        # 验证生成时间合理（应该在几秒内完成）
        assert (end_time - start_time).total_seconds() < 10
        assert summary["total_evaluations"] == 100


class TestComparisonReport:
    """测试对比报告功能"""

    @pytest.fixture
    def comparison_report(self) -> ComparisonReport:
        """对比报告实例"""
        return ComparisonReport()

    def test_compare_models_performance(self, comparison_report):
        """测试模型性能对比"""
        model_stats = {
            "model_a": PerformanceStats(
                avg_inference_time=0.15,
                throughput=800.0,
                memory_usage_mb=1200.0,
                gpu_utilization=0.75,
                cpu_utilization=0.45,
                max_inference_time=0.45,
                min_inference_time=0.08
            ),
            "model_b": PerformanceStats(
                avg_inference_time=0.22,
                throughput=650.0,
                memory_usage_mb=1800.0,
                gpu_utilization=0.85,
                cpu_utilization=0.55,
                max_inference_time=0.55,
                min_inference_time=0.12
            )
        }
        
        comparison = comparison_report.compare_performance(model_stats)
        
        assert "fastest_model" in comparison
        assert "most_efficient_memory" in comparison
        assert "highest_throughput" in comparison
        assert comparison["fastest_model"] == "model_a"
        assert comparison["most_efficient_memory"] == "model_a"

    def test_generate_improvement_suggestions(self, comparison_report):
        """测试生成改进建议"""
        poor_stats = PerformanceStats(
            avg_inference_time=1.5,  # 慢
            throughput=100.0,        # 低吞吐量
            memory_usage_mb=5000.0,  # 高内存使用
            gpu_utilization=0.95,    # 高GPU利用率
            cpu_utilization=0.85,    # 高CPU利用率
            max_inference_time=3.0,
            min_inference_time=0.5
        )
        
        suggestions = comparison_report.generate_improvement_suggestions(poor_stats)
        
        assert len(suggestions) > 0
        assert any("推理时间" in s or "inference" in s for s in suggestions)
        assert any("内存" in s or "memory" in s for s in suggestions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])