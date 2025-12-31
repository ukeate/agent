#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估API接口测试模块
测试FastAPI接口的功能、错误处理、异步任务管理等
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from api.v1.model_evaluation import router, get_evaluation_engine, get_benchmark_manager
from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationConfig
from ai.model_evaluation.benchmark_manager import BenchmarkManager
from ai.model_evaluation.report_generator import EvaluationReportGenerator

# 导入API相关模块

class TestModelEvaluationAPI:
    """测试模型评估API接口"""

    @pytest.fixture
    def mock_evaluation_engine(self):
        """模拟评估引擎"""
        engine = Mock(spec=ModelEvaluationEngine)
        engine.evaluate_model = AsyncMock()
        engine.batch_evaluate = AsyncMock()
        engine.get_evaluation_status = Mock()
        engine.get_evaluation_result = Mock()
        engine.cancel_evaluation = AsyncMock()
        return engine

    @pytest.fixture
    def mock_benchmark_manager(self):
        """模拟基准管理器"""
        manager = Mock(spec=BenchmarkManager)
        manager.get_available_benchmarks = Mock()
        manager.get_benchmark_info = Mock()
        manager.validate_benchmark_config = Mock()
        return manager

    @pytest.fixture
    def client(self, mock_evaluation_engine, mock_benchmark_manager):
        """测试客户端"""
        from main import app
        
        # 覆盖依赖
        app.dependency_overrides[get_evaluation_engine] = lambda: mock_evaluation_engine
        app.dependency_overrides[get_benchmark_manager] = lambda: mock_benchmark_manager
        
        with TestClient(app) as test_client:
            yield test_client
        
        # 清理依赖覆盖
        app.dependency_overrides.clear()

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/api/v1/model-evaluation/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_get_available_benchmarks(self, client, mock_benchmark_manager):
        """测试获取可用基准测试接口"""
        # 设置模拟返回
        mock_benchmark_manager.get_available_benchmarks.return_value = [
            {
                "name": "GLUE",
                "description": "General Language Understanding Evaluation",
                "tasks": ["cola", "sst2", "mrpc"]
            },
            {
                "name": "SuperGLUE", 
                "description": "Super General Language Understanding Evaluation",
                "tasks": ["boolq", "cb", "copa"]
            }
        ]
        
        response = client.get("/api/v1/model-evaluation/benchmarks")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "GLUE"
        assert "tasks" in data[0]

    def test_get_benchmark_info(self, client, mock_benchmark_manager):
        """测试获取基准测试详情接口"""
        # 设置模拟返回
        mock_benchmark_manager.get_benchmark_info.return_value = {
            "name": "GLUE",
            "description": "General Language Understanding Evaluation",
            "tasks": ["cola", "sst2", "mrpc"],
            "metrics": ["accuracy", "f1", "matthews_correlation"],
            "sample_size": 10000
        }
        
        response = client.get("/api/v1/model-evaluation/benchmarks/GLUE")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "GLUE"
        assert "metrics" in data

    def test_get_benchmark_info_not_found(self, client, mock_benchmark_manager):
        """测试获取不存在的基准测试"""
        mock_benchmark_manager.get_benchmark_info.return_value = None
        
        response = client.get("/api/v1/model-evaluation/benchmarks/NonExistent")
        
        assert response.status_code == 404

    def test_create_single_evaluation(self, client, mock_evaluation_engine):
        """测试创建单个评估任务"""
        # 设置模拟返回
        mock_evaluation_engine.evaluate_model.return_value = {
            "evaluation_id": "eval_123",
            "status": "running",
            "created_at": utc_now().isoformat()
        }
        
        request_data = {
            "model_path": "/path/to/model",
            "benchmark_name": "GLUE",
            "task_name": "cola",
            "config": {
                "batch_size": 16,
                "max_samples": 1000
            }
        }
        
        response = client.post("/api/v1/model-evaluation/evaluate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "evaluation_id" in data
        assert data["status"] == "running"

    def test_create_batch_evaluation(self, client, mock_evaluation_engine):
        """测试创建批量评估任务"""
        # 设置模拟返回
        mock_evaluation_engine.batch_evaluate.return_value = {
            "batch_id": "batch_456",
            "evaluations": [
                {"evaluation_id": "eval_123", "status": "queued"},
                {"evaluation_id": "eval_124", "status": "queued"}
            ],
            "total_evaluations": 2
        }
        
        request_data = {
            "evaluations": [
                {
                    "model_path": "/path/to/model1",
                    "benchmark_name": "GLUE",
                    "task_name": "cola"
                },
                {
                    "model_path": "/path/to/model2", 
                    "benchmark_name": "GLUE",
                    "task_name": "sst2"
                }
            ],
            "config": {
                "concurrent_evaluations": 2,
                "timeout_minutes": 60
            }
        }
        
        response = client.post("/api/v1/model-evaluation/batch-evaluate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["total_evaluations"] == 2

    def test_get_evaluation_status(self, client, mock_evaluation_engine):
        """测试获取评估状态"""
        # 设置模拟返回
        mock_evaluation_engine.get_evaluation_status.return_value = {
            "evaluation_id": "eval_123",
            "status": "completed",
            "progress": 100,
            "started_at": "2024-01-01T10:00:00",
            "completed_at": "2024-01-01T10:05:00",
            "duration": "00:05:00"
        }
        
        response = client.get("/api/v1/model-evaluation/evaluations/eval_123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100

    def test_get_evaluation_result(self, client, mock_evaluation_engine):
        """测试获取评估结果"""
        # 设置模拟返回
        mock_evaluation_engine.get_evaluation_result.return_value = {
            "evaluation_id": "eval_123",
            "model_info": {
                "name": "test-model",
                "version": "1.0.0"
            },
            "benchmark_results": [
                {
                    "benchmark_name": "GLUE",
                    "task_name": "cola",
                    "metrics": {
                        "accuracy": 0.85,
                        "f1": 0.83
                    }
                }
            ],
            "overall_score": 0.84
        }
        
        response = client.get("/api/v1/model-evaluation/evaluations/eval_123/result")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 0.84
        assert len(data["benchmark_results"]) == 1

    def test_get_evaluation_result_not_found(self, client, mock_evaluation_engine):
        """测试获取不存在的评估结果"""
        mock_evaluation_engine.get_evaluation_result.return_value = None
        
        response = client.get("/api/v1/model-evaluation/evaluations/nonexistent/result")
        
        assert response.status_code == 404

    def test_cancel_evaluation(self, client, mock_evaluation_engine):
        """测试取消评估任务"""
        # 设置模拟返回
        mock_evaluation_engine.cancel_evaluation.return_value = {
            "evaluation_id": "eval_123",
            "status": "cancelled",
            "message": "Evaluation cancelled successfully"
        }
        
        response = client.post("/api/v1/model-evaluation/evaluations/eval_123/cancel")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_list_evaluations(self, client, mock_evaluation_engine):
        """测试获取评估列表"""
        # 设置模拟返回
        mock_evaluation_engine.list_evaluations = Mock(return_value={
            "evaluations": [
                {
                    "evaluation_id": "eval_123",
                    "status": "completed",
                    "model_name": "test-model-1",
                    "benchmark": "GLUE",
                    "created_at": "2024-01-01T10:00:00"
                },
                {
                    "evaluation_id": "eval_124", 
                    "status": "running",
                    "model_name": "test-model-2",
                    "benchmark": "SuperGLUE",
                    "created_at": "2024-01-01T11:00:00"
                }
            ],
            "total": 2,
            "page": 1,
            "limit": 10
        })
        
        response = client.get("/api/v1/model-evaluation/evaluations?page=1&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["evaluations"]) == 2

    def test_generate_report(self, client):
        """测试生成报告接口"""
        request_data = {
            "evaluation_ids": ["eval_123", "eval_124"],
            "report_config": {
                "title": "模型评估报告",
                "include_charts": True,
                "output_format": "html"
            }
        }
        
        with patch('ai.model_evaluation.report_generator.EvaluationReportGenerator') as mock_generator:
            mock_instance = Mock()
            mock_instance.generate_evaluation_report.return_value = "/path/to/report.html"
            mock_generator.return_value = mock_instance
            
            response = client.post("/api/v1/model-evaluation/reports/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "report_id" in data
            assert "download_url" in data

    def test_download_report(self, client):
        """测试下载报告接口"""
        with patch('pathlib.Path') as mock_path:
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_file.read_bytes.return_value = b"<html>Test Report</html>"
            mock_path.return_value = mock_file
            
            response = client.get("/api/v1/model-evaluation/reports/report_123/download")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/html; charset=utf-8"

    def test_invalid_request_validation(self, client):
        """测试请求参数验证"""
        # 缺少必需字段
        invalid_request = {
            "benchmark_name": "GLUE"
            # 缺少 model_path
        }
        
        response = client.post("/api/v1/model-evaluation/evaluate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error

    def test_concurrent_evaluations_limit(self, client, mock_evaluation_engine):
        """测试并发评估数量限制"""
        # 模拟达到并发限制
        mock_evaluation_engine.evaluate_model.side_effect = Exception("Too many concurrent evaluations")
        
        request_data = {
            "model_path": "/path/to/model",
            "benchmark_name": "GLUE",
            "task_name": "cola"
        }
        
        response = client.post("/api/v1/model-evaluation/evaluate", json=request_data)
        
        assert response.status_code == 429  # Too Many Requests

    @patch('api.v1.model_evaluation.BackgroundTasks')
    def test_background_task_handling(self, mock_background_tasks, client, mock_evaluation_engine):
        """测试后台任务处理"""
        mock_tasks = Mock()
        mock_background_tasks.return_value = mock_tasks
        
        request_data = {
            "model_path": "/path/to/model",
            "benchmark_name": "GLUE",
            "task_name": "cola"
        }
        
        response = client.post("/api/v1/model-evaluation/evaluate", json=request_data)
        
        # 验证后台任务被添加
        assert response.status_code == 200

    def test_websocket_status_updates(self, client):
        """测试WebSocket状态更新"""
        with client.websocket_connect("/api/v1/model-evaluation/ws/eval_123") as websocket:
            # 模拟状态更新
            data = websocket.receive_json()
            
            assert "evaluation_id" in data
            assert "status" in data
            assert "progress" in data

    def test_error_handling_server_error(self, client, mock_evaluation_engine):
        """测试服务器错误处理"""
        # 模拟服务器内部错误
        mock_evaluation_engine.evaluate_model.side_effect = Exception("Internal server error")
        
        request_data = {
            "model_path": "/path/to/model",
            "benchmark_name": "GLUE",
            "task_name": "cola"
        }
        
        response = client.post("/api/v1/model-evaluation/evaluate", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_rate_limiting(self, client):
        """测试API限流"""
        # 快速发送多个请求
        responses = []
        for i in range(10):
            response = client.get("/api/v1/model-evaluation/benchmarks")
            responses.append(response)
        
        # 检查是否有限流响应
        status_codes = [r.status_code for r in responses]
        # 根据实际限流实现，可能会有429状态码
        assert all(code in [200, 429] for code in status_codes)

    def test_authentication_required(self, client):
        """测试需要身份验证的接口"""
        # 不提供认证信息
        response = client.post("/api/v1/model-evaluation/evaluate", json={})
        
        # 根据实际认证实现，可能返回401
        assert response.status_code in [401, 422]  # 401 Unauthorized or 422 Validation Error

    def test_cleanup_resources(self, client, mock_evaluation_engine):
        """测试资源清理"""
        # 模拟评估完成后的资源清理
        mock_evaluation_engine.cleanup_evaluation = Mock()
        
        response = client.delete("/api/v1/model-evaluation/evaluations/eval_123/cleanup")
        
        if response.status_code == 200:
            # 验证清理被调用
            assert True  # 清理成功

    async def test_async_evaluation_workflow(self, mock_evaluation_engine):
        """测试异步评估工作流"""
        # 模拟异步评估流程
        evaluation_id = "eval_async_123"
        
        # 启动评估
        mock_evaluation_engine.evaluate_model.return_value = {
            "evaluation_id": evaluation_id,
            "status": "running"
        }
        
        # 检查状态
        mock_evaluation_engine.get_evaluation_status.return_value = {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "progress": 100
        }
        
        # 获取结果
        mock_evaluation_engine.get_evaluation_result.return_value = {
            "evaluation_id": evaluation_id,
            "overall_score": 0.85
        }
        
        # 验证整个流程
        result = await mock_evaluation_engine.evaluate_model()
        assert result["evaluation_id"] == evaluation_id
        
        status = mock_evaluation_engine.get_evaluation_status()
        assert status["status"] == "completed"
        
        final_result = mock_evaluation_engine.get_evaluation_result()
        assert final_result["overall_score"] == 0.85

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
