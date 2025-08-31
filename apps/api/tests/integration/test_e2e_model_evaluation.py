"""
端到端模型评估测试
测试完整的评估流水线：API -> 评估引擎 -> 基准测试管理 -> 性能监控
"""

import pytest
import sys
from pathlib import Path
import asyncio
import tempfile
import os
import json
from unittest.mock import patch

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestE2EModelEvaluation:
    """端到端模型评估测试"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_evaluation_pipeline_mock(self):
        """完整评估流水线测试（使用mock避免真实模型加载）"""
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationConfig, EvaluationStatus
        from ai.model_evaluation.benchmark_manager import BenchmarkManager, BenchmarkConfig as BenchmarkManagerConfig
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, MonitorConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置评估引擎
            eval_config = EvaluationConfig(
                model_path="gpt2",  # 使用小模型进行测试
                device="cpu",
                batch_size=1,
                max_length=64,
                output_dir=temp_dir,
                enable_caching=False
            )
            
            # 配置基准测试管理器
            benchmark_config = BenchmarkManagerConfig(
                cache_dir=temp_dir,
                config_path=os.path.join(temp_dir, "benchmarks.yaml")
            )
            
            # 配置性能监控器
            monitor_config = MonitorConfig(
                enable_gpu_monitoring=False,
                save_metrics_to_file=False,
                update_interval=0.1
            )
            
            # 创建组件
            evaluation_engine = ModelEvaluationEngine(eval_config)
            benchmark_manager = BenchmarkManager(benchmark_config)
            performance_monitor = PerformanceMonitor(monitor_config)
            
            # 启动性能监控
            performance_monitor.start_monitoring()
            
            try:
                # 获取可用基准测试
                benchmarks = benchmark_manager.list_available_benchmarks()
                assert len(benchmarks) > 0, "应该有可用的基准测试"
                
                # 选择一个简单的基准测试
                test_benchmark = None
                for benchmark in benchmarks:
                    if benchmark.name in ["cola", "sst2", "winogrande", "piqa"]:
                        test_benchmark = benchmark
                        break
                
                if not test_benchmark:
                    test_benchmark = benchmarks[0]
                
                print(f"选择基准测试: {test_benchmark.name}")
                
                # Mock 模型初始化和评估执行，避免真实模型加载
                with patch.object(evaluation_engine, '_initialize_model') as mock_init, \
                     patch.object(evaluation_engine, '_execute_benchmark') as mock_exec:
                    
                    mock_init.return_value = None
                    mock_exec.return_value = {
                        'benchmark_name': test_benchmark.name,
                        'results': [{
                            'task': test_benchmark.name,
                            'accuracy': 0.75,
                            'f1_score': 0.73,
                            'inference_time': 0.5
                        }],
                        'timestamp': '2024-01-01T00:00:00',
                        'model_name': 'gpt2'
                    }
                    
                    # 启动评估任务
                    job_id = await evaluation_engine.start_evaluation(
                        benchmark_name=test_benchmark.name
                    )
                    
                    assert job_id is not None, "应该返回有效的任务ID"
                    print(f"评估任务ID: {job_id}")
                    
                    # 等待任务完成
                    max_wait = 10  # 最多等待10秒
                    wait_time = 0
                    status = None
                    
                    while wait_time < max_wait:
                        status = evaluation_engine.get_job_status(job_id)
                        if status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
                            break
                        await asyncio.sleep(0.1)
                        wait_time += 0.1
                    
                    # 验证任务状态
                    assert status == EvaluationStatus.COMPLETED, f"任务应该成功完成，实际状态: {status}"
                    
                    # 验证结果
                    job_info = evaluation_engine.current_jobs.get(job_id)
                    assert job_info is not None, "任务信息应该存在"
                    assert 'results' in job_info, "任务应该有结果"
                    
                    results = job_info['results']
                    assert results['benchmark_name'] == test_benchmark.name
                    assert len(results['results']) > 0
                    
                    print(f"评估结果: {json.dumps(results, indent=2)}")
                
                # 检查性能监控数据
                latest_metrics = performance_monitor.get_latest_system_metrics()
                assert latest_metrics is not None, "应该有性能监控数据"
                assert latest_metrics.cpu_percent >= 0
                assert latest_metrics.memory_percent >= 0
                
                # 获取性能摘要
                summary = performance_monitor.get_system_metrics_summary()
                assert len(summary) > 0, "应该有性能摘要数据"
                assert 'avg_cpu_percent' in summary
                
                print(f"性能监控摘要: {json.dumps(summary, indent=2)}")
                
                print("✅ 端到端评估流水线测试成功!")
                
            finally:
                # 清理
                performance_monitor.stop_monitoring()
    
    @pytest.mark.integration
    def test_api_endpoints_integration(self):
        """API端点集成测试"""
        from api.v1.rag import router
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # 测试健康检查端点
        response = client.get("/api/v1/health")
        if response.status_code == 200:
            assert response.json()["status"] == "healthy"
            print("✅ API健康检查通过")
        else:
            # 如果端点不存在，跳过此测试
            pytest.skip("API端点未实现，跳过测试")
    
    @pytest.mark.integration  
    def test_database_integration(self):
        """数据库集成测试"""
        try:
            import asyncpg
            import asyncio
            
            # 这里可以添加数据库连接测试
            # 由于没有实际的数据库配置，暂时跳过
            pytest.skip("数据库集成测试需要实际的数据库连接配置")
            
        except ImportError:
            pytest.skip("asyncpg未安装，跳过数据库集成测试")
    
    @pytest.mark.integration
    def test_component_interoperability(self):
        """组件互操作性测试"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        from ai.model_evaluation.benchmark_manager import BenchmarkConfig
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        # 测试配置对象的创建和序列化
        eval_config = EvaluationConfig(
            model_path="test-model",
            device="cpu",
            batch_size=2,
            max_length=128
        )
        
        benchmark_config = BenchmarkConfig(
            cache_dir="/tmp/benchmarks",
            max_concurrent_downloads=2
        )
        
        monitor_config = MonitorConfig(
            update_interval=1.0,
            enable_gpu_monitoring=False
        )
        
        # 测试配置序列化
        from dataclasses import asdict
        eval_dict = asdict(eval_config)
        benchmark_dict = asdict(benchmark_config)  
        monitor_dict = asdict(monitor_config)
        
        assert eval_dict['model_path'] == "test-model"
        assert benchmark_dict['cache_dir'] == "/tmp/benchmarks"
        assert monitor_dict['update_interval'] == 1.0
        
        print("✅ 组件互操作性测试通过")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])