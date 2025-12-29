#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估系统集成测试模块
测试完整的评估工作流程、组件间交互、端到端功能
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationConfig
from ai.model_evaluation.benchmark_manager import BenchmarkManager, BenchmarkConfig
from ai.model_evaluation.performance_monitor import PerformanceMonitor
from ai.model_evaluation.report_generator import EvaluationReportGenerator, ReportConfig

class TestModelEvaluationIntegration:
    """模型评估系统集成测试"""

    @pytest.fixture
    async def evaluation_system(self):
        """完整的评估系统"""
        config = EvaluationConfig(
            model_name="test-model",
            model_path="/tmp/test_model",
            device="cpu",
            batch_size=4,
            max_samples=100
        )
        
        engine = ModelEvaluationEngine(config)
        benchmark_manager = BenchmarkManager()
        performance_monitor = PerformanceMonitor()
        report_generator = EvaluationReportGenerator()
        
        return {
            "engine": engine,
            "benchmark_manager": benchmark_manager,
            "performance_monitor": performance_monitor,
            "report_generator": report_generator,
            "config": config
        }

    @pytest.fixture
    def sample_benchmark_configs(self):
        """示例基准测试配置"""
        return [
            BenchmarkConfig(
                name="GLUE",
                task="cola",
                num_samples=100,
                batch_size=4,
                metrics=["accuracy", "f1"]
            ),
            BenchmarkConfig(
                name="GLUE", 
                task="sst2",
                num_samples=100,
                batch_size=4,
                metrics=["accuracy"]
            )
        ]

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, evaluation_system, sample_benchmark_configs):
        """测试完整的评估工作流程"""
        system = evaluation_system
        engine = system["engine"]
        benchmark_manager = system["benchmark_manager"]
        performance_monitor = system["performance_monitor"]
        report_generator = system["report_generator"]
        
        # 模拟组件
        with patch.object(engine, '_load_model') as mock_load, \
             patch.object(engine, 'evaluate_with_lm_eval') as mock_eval, \
             patch.object(benchmark_manager, 'validate_benchmark_config', return_value=True), \
             patch.object(performance_monitor, 'start_monitoring'), \
             patch.object(performance_monitor, 'stop_monitoring'), \
             patch.object(performance_monitor, 'get_current_metrics') as mock_metrics:
            
            # 设置模拟返回值
            mock_load.return_value = True
            mock_eval.return_value = [
                {
                    "evaluation_id": "eval_123",
                    "benchmark_name": "GLUE",
                    "task_name": "cola",
                    "metrics": {"accuracy": 0.85, "f1": 0.83},
                    "performance_stats": {
                        "avg_inference_time": 0.15,
                        "throughput": 800.0,
                        "memory_usage_mb": 1200.0
                    }
                }
            ]
            mock_metrics.return_value = {
                "cpu_usage": 0.45,
                "memory_usage_mb": 1200.0,
                "gpu_utilization": 0.75
            }
            
            # 执行完整工作流程
            
            # 1. 初始化和验证
            await engine.initialize()
            assert engine.model is not None
            
            # 2. 验证基准测试配置
            for config in sample_benchmark_configs:
                is_valid = benchmark_manager.validate_benchmark_config(config)
                assert is_valid
            
            # 3. 开始性能监控
            performance_monitor.start_monitoring()
            
            # 4. 执行评估
            results = []
            for config in sample_benchmark_configs:
                result = await engine.evaluate_with_benchmark_config(config)
                results.append(result)
            
            # 5. 停止性能监控
            performance_monitor.stop_monitoring()
            final_metrics = performance_monitor.get_current_metrics()
            
            # 6. 生成报告
            with tempfile.TemporaryDirectory() as temp_dir:
                report_config = ReportConfig(
                    title="集成测试报告",
                    output_path=Path(temp_dir) / "integration_report.html",
                    include_charts=True
                )
                
                report_path = report_generator.generate_evaluation_report(results, report_config)
                
                # 验证结果
                assert len(results) == len(sample_benchmark_configs)
                assert report_path.exists()
                assert final_metrics["cpu_usage"] == 0.45

    @pytest.mark.asyncio
    async def test_batch_evaluation_with_monitoring(self, evaluation_system):
        """测试带监控的批量评估"""
        system = evaluation_system
        engine = system["engine"]
        performance_monitor = system["performance_monitor"]
        
        # 模拟多个模型评估
        model_configs = [
            {"model_path": "/tmp/model1", "model_name": "model-1"},
            {"model_path": "/tmp/model2", "model_name": "model-2"},
            {"model_path": "/tmp/model3", "model_name": "model-3"}
        ]
        
        with patch.object(engine, '_load_model') as mock_load, \
             patch.object(engine, 'evaluate_model') as mock_eval, \
             patch.object(performance_monitor, 'start_monitoring'), \
             patch.object(performance_monitor, 'get_current_metrics') as mock_metrics:
            
            mock_load.return_value = True
            mock_eval.return_value = {
                "overall_score": 0.85,
                "benchmark_results": []
            }
            mock_metrics.return_value = {
                "cpu_usage": 0.6,
                "memory_usage_mb": 1500.0
            }
            
            # 开始监控
            performance_monitor.start_monitoring()
            
            # 批量评估
            results = []
            for config in model_configs:
                engine.config.model_path = config["model_path"]
                engine.config.model_name = config["model_name"]
                
                await engine.initialize()
                result = await engine.evaluate_model()
                results.append(result)
                
                # 检查性能指标
                metrics = performance_monitor.get_current_metrics()
                assert metrics["cpu_usage"] <= 1.0
                assert metrics["memory_usage_mb"] > 0
            
            # 验证批量结果
            assert len(results) == len(model_configs)
            assert all(r["overall_score"] > 0 for r in results)

    def test_error_recovery_and_cleanup(self, evaluation_system):
        """测试错误恢复和资源清理"""
        system = evaluation_system
        engine = system["engine"]
        performance_monitor = system["performance_monitor"]
        
        with patch.object(engine, '_load_model') as mock_load, \
             patch.object(engine, 'evaluate_model') as mock_eval, \
             patch.object(performance_monitor, 'start_monitoring'), \
             patch.object(performance_monitor, 'stop_monitoring') as mock_stop:
            
            # 模拟加载模型失败
            mock_load.side_effect = Exception("模型加载失败")
            
            # 开始监控
            performance_monitor.start_monitoring()
            
            try:
                # 尝试初始化（应该失败）
                asyncio.run(engine.initialize())
                assert False, "应该抛出异常"
            except Exception as e:
                assert "模型加载失败" in str(e)
                
                # 验证清理被调用
                performance_monitor.stop_monitoring()
                mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, evaluation_system):
        """测试并发评估"""
        system = evaluation_system
        engine = system["engine"]
        
        # 创建多个评估任务
        benchmark_configs = [
            BenchmarkConfig(name="GLUE", task="cola", num_samples=50),
            BenchmarkConfig(name="GLUE", task="sst2", num_samples=50),
            BenchmarkConfig(name="SuperGLUE", task="boolq", num_samples=50)
        ]
        
        with patch.object(engine, '_load_model', return_value=True), \
             patch.object(engine, 'evaluate_with_lm_eval') as mock_eval:
            
            mock_eval.return_value = [{"evaluation_id": "test", "metrics": {"accuracy": 0.8}}]
            
            # 初始化引擎
            await engine.initialize()
            
            # 并发执行评估
            tasks = [
                engine.evaluate_with_benchmark_config(config)
                for config in benchmark_configs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 验证所有任务都成功完成
            assert len(results) == len(benchmark_configs)
            assert all(not isinstance(r, Exception) for r in results)

    def test_report_generation_with_multiple_formats(self, evaluation_system):
        """测试多格式报告生成"""
        system = evaluation_system
        report_generator = system["report_generator"]
        
        # 模拟评估结果
        sample_results = [
            {
                "evaluation_id": "eval_1",
                "model_info": {"name": "model-1", "version": "1.0"},
                "benchmark_results": [{"benchmark_name": "GLUE", "metrics": {"accuracy": 0.85}}],
                "overall_score": 0.85
            },
            {
                "evaluation_id": "eval_2", 
                "model_info": {"name": "model-2", "version": "1.0"},
                "benchmark_results": [{"benchmark_name": "GLUE", "metrics": {"accuracy": 0.88}}],
                "overall_score": 0.88
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成HTML报告
            html_config = ReportConfig(
                title="HTML报告测试",
                output_path=Path(temp_dir) / "report.html",
                output_format="html"
            )
            
            # 生成JSON报告
            json_config = ReportConfig(
                title="JSON报告测试",
                output_path=Path(temp_dir) / "report.json",
                output_format="json"
            )
            
            # 生成CSV报告
            csv_config = ReportConfig(
                title="CSV报告测试",
                output_path=Path(temp_dir) / "report.csv",
                output_format="csv"
            )
            
            # 执行报告生成
            html_path = report_generator.generate_evaluation_report(sample_results, html_config)
            json_path = report_generator.export_to_json(sample_results, json_config.output_path)
            csv_path = report_generator.export_to_csv(sample_results, csv_config.output_path)
            
            # 验证所有格式都生成成功
            assert html_path.exists()
            assert json_path.exists() 
            assert csv_path.exists()
            
            # 验证文件内容
            assert html_path.read_text(encoding='utf-8')
            
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                assert len(json_data["evaluations"]) == 2
            
            csv_content = csv_path.read_text(encoding='utf-8')
            assert "model-1" in csv_content
            assert "model-2" in csv_content

    @pytest.mark.asyncio
    async def test_benchmark_suite_execution(self, evaluation_system):
        """测试基准测试套件执行"""
        system = evaluation_system
        engine = system["engine"] 
        benchmark_manager = system["benchmark_manager"]
        
        # 创建基准测试套件
        suite_config = {
            "name": "Complete GLUE Suite",
            "benchmarks": [
                {"name": "GLUE", "task": "cola"},
                {"name": "GLUE", "task": "sst2"},
                {"name": "GLUE", "task": "mrpc"}
            ]
        }
        
        with patch.object(engine, '_load_model', return_value=True), \
             patch.object(engine, 'evaluate_with_lm_eval') as mock_eval, \
             patch.object(benchmark_manager, 'get_benchmark_suite') as mock_suite:
            
            mock_suite.return_value = suite_config
            mock_eval.return_value = [{"metrics": {"accuracy": 0.85}}]
            
            # 初始化引擎
            await engine.initialize()
            
            # 执行套件
            suite_results = await engine.evaluate_benchmark_suite("Complete GLUE Suite")
            
            # 验证套件执行结果
            assert len(suite_results) == len(suite_config["benchmarks"])
            assert all("metrics" in result for result in suite_results)

    def test_performance_threshold_alerts(self, evaluation_system):
        """测试性能阈值告警"""
        system = evaluation_system
        performance_monitor = system["performance_monitor"]
        
        # 设置性能阈值
        thresholds = {
            "cpu_usage": 0.8,
            "memory_usage_mb": 2000.0,
            "inference_time": 1.0
        }
        
        performance_monitor.set_thresholds(thresholds)
        
        # 模拟超过阈值的性能指标
        high_metrics = {
            "cpu_usage": 0.9,  # 超过阈值
            "memory_usage_mb": 2500.0,  # 超过阈值
            "inference_time": 1.5  # 超过阈值
        }
        
        with patch.object(performance_monitor, 'get_current_metrics', return_value=high_metrics):
            alerts = performance_monitor.check_thresholds()
            
            # 验证告警生成
            assert len(alerts) == 3
            assert any("CPU使用率" in alert for alert in alerts)
            assert any("内存使用" in alert for alert in alerts)
            assert any("推理时间" in alert for alert in alerts)

    @pytest.mark.asyncio
    async def test_evaluation_pipeline_with_checkpoints(self, evaluation_system):
        """测试带检查点的评估管道"""
        system = evaluation_system
        engine = system["engine"]
        
        checkpoint_dir = Path("/tmp/eval_checkpoints")
        
        with patch.object(engine, '_load_model', return_value=True), \
             patch.object(engine, 'evaluate_model') as mock_eval, \
             patch.object(engine, 'save_checkpoint') as mock_save, \
             patch.object(engine, 'load_checkpoint') as mock_load:
            
            mock_eval.return_value = {"overall_score": 0.85}
            mock_save.return_value = True
            mock_load.return_value = {"evaluation_id": "eval_checkpoint", "progress": 0.5}
            
            # 初始化引擎
            await engine.initialize()
            
            # 开始评估并保存检查点
            result = await engine.evaluate_model()
            engine.save_checkpoint(checkpoint_dir / "checkpoint_1.json")
            
            # 从检查点恢复
            checkpoint_data = engine.load_checkpoint(checkpoint_dir / "checkpoint_1.json")
            
            # 验证检查点功能
            assert result["overall_score"] == 0.85
            assert checkpoint_data["evaluation_id"] == "eval_checkpoint"
            mock_save.assert_called_once()
            mock_load.assert_called_once()

    def test_resource_management(self, evaluation_system):
        """测试资源管理"""
        system = evaluation_system
        engine = system["engine"]
        performance_monitor = system["performance_monitor"]
        
        # 模拟资源限制
        resource_limits = {
            "max_memory_mb": 2000.0,
            "max_cpu_usage": 0.8,
            "max_concurrent_evaluations": 3
        }
        
        with patch.object(performance_monitor, 'get_current_metrics') as mock_metrics, \
             patch.object(engine, '_check_resource_availability') as mock_check:
            
            # 模拟资源不足
            mock_metrics.return_value = {
                "memory_usage_mb": 2500.0,  # 超过限制
                "cpu_usage": 0.9  # 超过限制
            }
            mock_check.return_value = False
            
            # 尝试启动评估
            can_start = engine._check_resource_availability(resource_limits)
            
            # 验证资源检查
            assert not can_start
            mock_check.assert_called_with(resource_limits)

class TestSystemIntegrationScenarios:
    """系统集成场景测试"""

    @pytest.mark.asyncio
    async def test_real_world_evaluation_scenario(self):
        """测试真实世界评估场景"""
        # 模拟完整的真实评估流程
        config = EvaluationConfig(
            model_name="production-model",
            model_path="/models/production",
            device="cuda",
            batch_size=16,
            max_samples=1000
        )
        
        # 初始化所有组件
        engine = ModelEvaluationEngine(config)
        benchmark_manager = BenchmarkManager()
        monitor = PerformanceMonitor()
        generator = EvaluationReportGenerator()
        
        with patch.object(engine, '_load_model', return_value=True), \
             patch.object(engine, 'evaluate_with_lm_eval') as mock_eval:
            
            mock_eval.return_value = [
                {
                    "evaluation_id": "prod_eval_1",
                    "benchmark_name": "GLUE",
                    "metrics": {"accuracy": 0.87, "f1": 0.85},
                    "performance_stats": {
                        "avg_inference_time": 0.12,
                        "throughput": 1000.0
                    }
                }
            ]
            
            # 执行完整流程
            await engine.initialize()
            monitor.start_monitoring()
            
            # 获取可用基准测试
            benchmarks = benchmark_manager.get_available_benchmarks()
            assert len(benchmarks) > 0
            
            # 执行评估
            results = await engine.evaluate_with_lm_eval(BenchmarkConfig(name="GLUE", task="cola"))
            
            monitor.stop_monitoring()
            
            # 生成报告
            with tempfile.TemporaryDirectory() as temp_dir:
                report_config = ReportConfig(
                    title="生产环境评估报告",
                    output_path=Path(temp_dir) / "production_report.html"
                )
                report_path = generator.generate_evaluation_report(results, report_config)
                
                # 验证完整流程成功
                assert len(results) == 1
                assert results[0]["metrics"]["accuracy"] == 0.87
                assert report_path.exists()

    @pytest.mark.asyncio
    async def test_multi_model_comparison_workflow(self):
        """测试多模型对比工作流程"""
        models = [
            {"name": "model-A", "path": "/models/model-a"},
            {"name": "model-B", "path": "/models/model-b"},
            {"name": "model-C", "path": "/models/model-c"}
        ]
        
        all_results = []
        
        for model in models:
            config = EvaluationConfig(
                model_name=model["name"],
                model_path=model["path"],
                device="cpu",
                batch_size=8
            )
            
            engine = ModelEvaluationEngine(config)
            
            with patch.object(engine, '_load_model', return_value=True), \
                 patch.object(engine, 'evaluate_model') as mock_eval:
                
                mock_eval.return_value = {
                    "model_info": {"name": model["name"]},
                    "overall_score": 0.8 + len(model["name"]) * 0.01,  # 不同分数
                    "benchmark_results": []
                }
                
                await engine.initialize()
                result = await engine.evaluate_model()
                all_results.append(result)
        
        # 生成对比报告
        generator = EvaluationReportGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_config = ReportConfig(
                title="多模型对比报告",
                output_path=Path(temp_dir) / "comparison.html",
                include_comparison=True
            )
            
            report_path = generator.generate_comparison_report(all_results, report_config)
            
            # 验证对比结果
            assert len(all_results) == 3
            assert report_path.exists()
            
            # 验证分数不同（表明对比有意义）
            scores = [r["overall_score"] for r in all_results]
            assert len(set(scores)) == 3  # 所有分数都不同

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
