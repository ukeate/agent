"""
模型压缩API端点测试

测试所有模型压缩相关的API端点
"""

import pytest
import json
import tempfile
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# 需要创建应用实例用于测试
from main import app
from src.ai.model_compression.models import CompressionMethod, QuantizationMethod, PrecisionType


class SimpleTestModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def sample_model_file():
    """创建示例模型文件"""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        torch.save(SimpleTestModel(), tmp_file.name)
        yield tmp_file.name


class TestCompressionAPI:
    """压缩API测试类"""
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/api/v1/model_compression/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "supported_methods" in data
    
    def test_list_supported_methods(self, client):
        """测试列出支持的方法"""
        response = client.get("/api/v1/model_compression/methods")
        assert response.status_code == 200
        
        data = response.json()
        assert "quantization" in data
        assert "distillation" in data
        assert "pruning" in data
        assert "mixed" in data
    
    def test_create_quantization_job(self, client, sample_model_file):
        """测试创建量化任务"""
        job_data = {
            "job_name": "test_quantization",
            "model_path": sample_model_file,
            "compression_method": "quantization",
            "quantization_config": {
                "method": "ptq",
                "precision": "int8",
                "calibration_dataset_size": 100
            }
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=job_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "success"
    
    def test_create_pruning_job(self, client, sample_model_file):
        """测试创建剪枝任务"""
        job_data = {
            "job_name": "test_pruning",
            "model_path": sample_model_file,
            "compression_method": "pruning",
            "pruning_config": {
                "pruning_type": "unstructured",
                "sparsity_ratio": 0.5,
                "importance_metric": "magnitude"
            }
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=job_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
    
    def test_create_distillation_job(self, client, sample_model_file):
        """测试创建蒸馏任务"""
        # 创建学生模型文件
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as student_file:
            torch.save(SimpleTestModel(), student_file.name)
            student_path = student_file.name
        
        job_data = {
            "job_name": "test_distillation",
            "model_path": sample_model_file,
            "compression_method": "distillation",
            "distillation_config": {
                "teacher_model": sample_model_file,
                "student_model": student_path,
                "temperature": 3.0,
                "alpha": 0.5,
                "num_epochs": 2
            }
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=job_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
    
    def test_get_job_status(self, client, sample_model_file):
        """测试获取任务状态"""
        # 首先创建一个任务
        job_data = {
            "job_name": "status_test",
            "model_path": sample_model_file,
            "compression_method": "quantization"
        }
        
        create_response = client.post("/api/v1/model_compression/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # 获取任务状态
        response = client.get(f"/api/v1/model_compression/jobs/{job_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["job_id"] == job_id
    
    def test_get_job_result(self, client, sample_model_file):
        """测试获取任务结果"""
        # 创建任务
        job_data = {
            "job_name": "result_test",
            "model_path": sample_model_file,
            "compression_method": "quantization"
        }
        
        create_response = client.post("/api/v1/model_compression/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # 获取任务结果
        response = client.get(f"/api/v1/model_compression/jobs/{job_id}/result")
        
        # 任务可能还没完成，所以可能返回202或200
        assert response.status_code in [200, 202]
        
        if response.status_code == 200:
            data = response.json()
            assert "result" in data
    
    def test_cancel_job(self, client, sample_model_file):
        """测试取消任务"""
        # 创建任务
        job_data = {
            "job_name": "cancel_test",
            "model_path": sample_model_file,
            "compression_method": "quantization"
        }
        
        create_response = client.post("/api/v1/model_compression/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # 取消任务
        response = client.post(f"/api/v1/model_compression/jobs/{job_id}/cancel")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_retry_failed_job(self, client, sample_model_file):
        """测试重试失败的任务"""
        # 创建任务
        job_data = {
            "job_name": "retry_test",
            "model_path": sample_model_file,
            "compression_method": "quantization"
        }
        
        create_response = client.post("/api/v1/model_compression/jobs", json=job_data)
        job_id = create_response.json()["job_id"]
        
        # 尝试重试任务（可能还没失败）
        response = client.post(f"/api/v1/model_compression/jobs/{job_id}/retry")
        
        # 可能返回200（成功重试）或400（任务状态不允许重试）
        assert response.status_code in [200, 400]
    
    def test_list_all_jobs(self, client):
        """测试列出所有任务"""
        response = client.get("/api/v1/model_compression/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
    
    def test_get_pipeline_statistics(self, client):
        """测试获取流水线统计信息"""
        response = client.get("/api/v1/model_compression/pipeline/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_jobs" in data
        assert "active_jobs" in data
        assert "completed_jobs" in data
        assert "failed_jobs" in data
    
    def test_benchmark_performance(self, client, sample_model_file):
        """测试性能基准测试"""
        benchmark_data = {
            "model_path": sample_model_file,
            "hardware_config": {
                "device": "cpu",
                "batch_size": 1,
                "sequence_length": 10
            },
            "num_runs": 5
        }
        
        response = client.post("/api/v1/model_compression/benchmark", json=benchmark_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "benchmark_id" in data
    
    def test_get_benchmark_result(self, client, sample_model_file):
        """测试获取基准测试结果"""
        # 首先运行基准测试
        benchmark_data = {
            "model_path": sample_model_file,
            "hardware_config": {"device": "cpu"}
        }
        
        benchmark_response = client.post("/api/v1/model_compression/benchmark", json=benchmark_data)
        benchmark_id = benchmark_response.json()["benchmark_id"]
        
        # 获取结果
        response = client.get(f"/api/v1/model_compression/benchmark/{benchmark_id}")
        
        # 可能还在运行或已完成
        assert response.status_code in [200, 202]
    
    def test_recommend_strategy(self, client, sample_model_file):
        """测试策略推荐"""
        recommendation_data = {
            "model_path": sample_model_file,
            "target_constraints": {
                "max_size_reduction": 0.5,
                "max_accuracy_loss": 0.05,
                "target_device": "cpu"
            },
            "optimization_goals": ["size", "speed"]
        }
        
        response = client.post("/api/v1/model_compression/recommend", json=recommendation_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
    
    def test_get_default_config(self, client):
        """测试获取默认配置"""
        response = client.get("/api/v1/model_compression/config/quantization")
        assert response.status_code == 200
        
        data = response.json()
        assert "method" in data
        assert "precision" in data
    
    def test_model_analysis(self, client, sample_model_file):
        """测试模型分析"""
        response = client.post(
            f"/api/v1/model_compression/analyze?model_path={sample_model_file}"
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "model_info" in data
        assert "parameters" in data["model_info"]
        assert "size_mb" in data["model_info"]
    
    def test_compare_models(self, client, sample_model_file):
        """测试模型对比"""
        # 创建另一个模型文件用于对比
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(SimpleTestModel(), tmp_file.name)
            model2_path = tmp_file.name
        
        comparison_data = {
            "model_paths": [sample_model_file, model2_path],
            "comparison_metrics": ["size", "parameters", "flops"]
        }
        
        response = client.post("/api/v1/model_compression/compare", json=comparison_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "comparison_results" in data
    
    def test_invalid_job_creation(self, client):
        """测试无效任务创建"""
        invalid_job_data = {
            "job_name": "invalid_test",
            "model_path": "/nonexistent/model.pth",  # 不存在的路径
            "compression_method": "quantization"
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=invalid_job_data)
        # API可能接受任务但在执行时失败，或者立即拒绝
        # 这取决于具体实现
        assert response.status_code in [200, 400, 422]
    
    def test_invalid_job_id(self, client):
        """测试无效任务ID"""
        response = client.get("/api/v1/model_compression/jobs/invalid_job_id/status")
        assert response.status_code == 404
    
    def test_missing_required_fields(self, client):
        """测试缺少必需字段"""
        incomplete_job_data = {
            "job_name": "incomplete_test"
            # 缺少model_path和compression_method
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=incomplete_job_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_compression_method(self, client, sample_model_file):
        """测试无效压缩方法"""
        invalid_job_data = {
            "job_name": "invalid_method_test",
            "model_path": sample_model_file,
            "compression_method": "invalid_method"
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=invalid_job_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.ai.model_compression.compression_pipeline.CompressionPipeline')
    def test_pipeline_error_handling(self, mock_pipeline, client, sample_model_file):
        """测试流水线错误处理"""
        # 模拟流水线错误
        mock_instance = Mock()
        mock_instance.submit_job.side_effect = Exception("Pipeline error")
        mock_pipeline.return_value = mock_instance
        
        job_data = {
            "job_name": "error_test",
            "model_path": sample_model_file,
            "compression_method": "quantization"
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=job_data)
        # 应该处理错误并返回适当的状态码
        assert response.status_code in [500, 400]


class TestCompressionAPIEdgeCases:
    """压缩API边界情况测试"""
    
    def test_empty_job_list(self, client):
        """测试空任务列表"""
        # 清理可能存在的任务（如果API支持）
        response = client.get("/api/v1/model_compression/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
    
    def test_concurrent_job_creation(self, client, sample_model_file):
        """测试并发任务创建"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_job(job_name):
            job_data = {
                "job_name": job_name,
                "model_path": sample_model_file,
                "compression_method": "quantization"
            }
            
            try:
                response = client.post("/api/v1/model_compression/jobs", json=job_data)
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))
        
        # 创建多个并发请求
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_job, args=[f"concurrent_job_{i}"])
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        success_count = 0
        while not results.empty():
            status, code = results.get()
            if status == "success" and code == 200:
                success_count += 1
        
        # 至少应该有一些成功的请求
        assert success_count > 0
    
    def test_large_model_handling(self, client):
        """测试大模型处理"""
        # 创建一个较大的模型
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(1000, 1000) for _ in range(5)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(LargeModel(), tmp_file.name)
            large_model_path = tmp_file.name
        
        job_data = {
            "job_name": "large_model_test",
            "model_path": large_model_path,
            "compression_method": "quantization"
        }
        
        response = client.post("/api/v1/model_compression/jobs", json=job_data)
        # 大模型也应该能够处理
        assert response.status_code == 200
    
    def test_api_rate_limiting(self, client, sample_model_file):
        """测试API速率限制（如果实现了的话）"""
        # 快速发送大量请求
        responses = []
        for i in range(10):
            job_data = {
                "job_name": f"rate_limit_test_{i}",
                "model_path": sample_model_file,
                "compression_method": "quantization"
            }
            
            response = client.post("/api/v1/model_compression/jobs", json=job_data)
            responses.append(response.status_code)
        
        # 大部分请求应该成功，但如果有速率限制，可能会有429状态码
        success_count = sum(1 for code in responses if code == 200)
        assert success_count >= 5  # 至少一半应该成功


@pytest.mark.parametrize("compression_method,config_key", [
    ("quantization", "quantization_config"),
    ("pruning", "pruning_config"),
    ("distillation", "distillation_config"),
])
def test_different_compression_job_types(compression_method, config_key, client, sample_model_file):
    """参数化测试不同类型的压缩任务"""
    base_job_data = {
        "job_name": f"test_{compression_method}",
        "model_path": sample_model_file,
        "compression_method": compression_method
    }
    
    # 添加适当的配置
    if compression_method == "quantization":
        base_job_data[config_key] = {
            "method": "ptq",
            "precision": "int8"
        }
    elif compression_method == "pruning":
        base_job_data[config_key] = {
            "pruning_type": "unstructured",
            "sparsity_ratio": 0.5
        }
    elif compression_method == "distillation":
        base_job_data[config_key] = {
            "teacher_model": sample_model_file,
            "student_model": sample_model_file,
            "temperature": 3.0,
            "alpha": 0.5
        }
    
    response = client.post("/api/v1/model_compression/jobs", json=base_job_data)
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])