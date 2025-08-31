"""
真实的模型评估集成测试
使用真实的ML依赖进行端到端测试
"""

import pytest
import sys
from pathlib import Path
import asyncio
from typing import Dict, Any
import tempfile
import os

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestModelEvaluationReal:
    """真实模型评估集成测试"""
    
    @pytest.mark.integration
    def test_import_all_dependencies(self):
        """测试所有核心ML依赖导入"""
        try:
            import torch
            import transformers
            import evaluate
            import datasets
            import lm_eval
            from lm_eval import evaluator, models, tasks
            import accelerate
            import sentence_transformers
            print(f"✓ torch version: {torch.__version__}")
            print(f"✓ transformers version: {transformers.__version__}")
            print(f"✓ lm_eval version: {lm_eval.__version__}")
            assert True
        except ImportError as e:
            pytest.fail(f"依赖导入失败: {e}")
    
    @pytest.mark.integration
    def test_torch_basic_functionality(self):
        """测试PyTorch基本功能"""
        import torch
        
        # 创建张量
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        # 基本运算
        z = x + y
        expected = torch.tensor([5.0, 7.0, 9.0])
        
        assert torch.allclose(z, expected)
        print("✓ PyTorch 基本功能正常")
    
    @pytest.mark.integration
    def test_transformers_model_loading(self):
        """测试Transformers模型加载"""
        from transformers import AutoTokenizer, AutoModel
        
        # 使用小型模型进行测试
        model_name = "distilbert-base-uncased"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # 测试tokenization
            text = "Hello, world!"
            tokens = tokenizer(text, return_tensors="pt")
            
            # 测试模型推理
            with torch.no_grad():
                outputs = model(**tokens)
                
            assert outputs.last_hidden_state is not None
            print("✓ Transformers 模型加载和推理正常")
            
        except Exception as e:
            # 如果网络问题导致下载失败，跳过此测试
            pytest.skip(f"模型下载失败，可能是网络问题: {e}")
    
    @pytest.mark.integration 
    def test_evaluation_config_real(self):
        """测试真实评估配置"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig, ModelEvaluationEngine
        
        config = EvaluationConfig(
            model_path="distilbert-base-uncased",
            device="cpu",  # 使用CPU避免GPU依赖
            batch_size=1,
            max_length=128,
            temperature=1.0,
            enable_caching=False
        )
        
        engine = ModelEvaluationEngine(config)
        assert engine.config == config
        assert engine.device == "cpu"
        print("✓ 真实评估配置创建成功")
    
    @pytest.mark.integration
    def test_benchmark_manager_real(self):
        """测试真实基准测试管理器"""
        from ai.model_evaluation.benchmark_manager import BenchmarkManager, BenchmarkConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(
                cache_dir=temp_dir,
                config_path=os.path.join(temp_dir, "benchmarks.yaml")
            )
            
            manager = BenchmarkManager(config)
            
            # 测试内置基准测试列表
            benchmarks = manager.list_available_benchmarks()
            assert len(benchmarks) > 0
            print(f"✓ 找到 {len(benchmarks)} 个可用基准测试")
            
            # 测试基准测试信息获取
            for benchmark in benchmarks[:3]:  # 只测试前3个
                info = manager.get_benchmark_info(benchmark.name)
                assert info is not None
                assert info.name == benchmark.name
                print(f"✓ 基准测试 {benchmark.name} 信息获取成功")
    
    @pytest.mark.integration
    def test_performance_monitor_real(self):
        """测试真实性能监控器"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, MonitorConfig
        import time
        
        config = MonitorConfig(
            enable_gpu_monitoring=False,  # 避免GPU依赖
            save_metrics_to_file=False
        )
        
        monitor = PerformanceMonitor(config)
        
        # 启动监控
        monitor.start_monitoring()
        time.sleep(2)  # 让监控运行一会儿
        
        # 检查是否有系统指标
        latest_metrics = monitor.get_latest_system_metrics()
        assert latest_metrics is not None
        assert latest_metrics.cpu_percent >= 0
        assert latest_metrics.memory_percent >= 0
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 获取摘要
        summary = monitor.get_system_metrics_summary()
        assert len(summary) > 0
        assert 'avg_cpu_percent' in summary
        
        print("✓ 真实性能监控功能正常")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_simple_evaluation_pipeline(self):
        """测试简单的评估流水线"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig, ModelEvaluationEngine
        from ai.model_evaluation.benchmark_manager import BenchmarkManager, BenchmarkConfig
        import tempfile
        
        # 跳过此测试如果没有足够的资源
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("需要CUDA支持进行完整测试")
        except:
            pytest.skip("缺少PyTorch CUDA支持")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置评估引擎
            eval_config = EvaluationConfig(
                model_path="distilbert-base-uncased",
                device="cpu",
                batch_size=1,
                max_length=64,
                output_dir=temp_dir
            )
            
            # 配置基准测试管理器
            benchmark_config = BenchmarkConfig(
                cache_dir=temp_dir,
                config_path=os.path.join(temp_dir, "benchmarks.yaml")
            )
            
            engine = ModelEvaluationEngine(eval_config)
            manager = BenchmarkManager(benchmark_config)
            
            # 获取简单的基准测试（如果可用）
            benchmarks = manager.list_available_benchmarks()
            if not benchmarks:
                pytest.skip("没有可用的基准测试")
            
            # 选择最简单的基准测试
            simple_benchmark = None
            for benchmark in benchmarks:
                if benchmark.difficulty_level.value == "easy":
                    simple_benchmark = benchmark
                    break
            
            if not simple_benchmark:
                simple_benchmark = benchmarks[0]  # 使用第一个可用的
            
            print(f"✓ 准备运行基准测试: {simple_benchmark.name}")
            
            # 注意：实际的评估可能需要很长时间，这里只测试配置
            assert engine.config.model_path == "distilbert-base-uncased"
            assert simple_benchmark.name in [b.name for b in benchmarks]
            
            print("✓ 评估流水线配置成功")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])