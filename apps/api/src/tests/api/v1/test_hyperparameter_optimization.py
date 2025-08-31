"""
超参数优化API单元测试
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from main import app
from ai.hyperparameter_optimization.models import (
    ExperimentState, TrialState, OptimizationAlgorithm
)


class TestHyperparameterOptimizationAPI:
    """超参数优化API测试类"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def experiment_data(self):
        """实验数据"""
        return {
            "name": "Test Experiment",
            "description": "API test experiment",
            "algorithm": "tpe",
            "parameter_ranges": {
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1},
                "batch_size": {"type": "int", "low": 16, "high": 128}
            },
            "optimization_config": {
                "n_trials": 100,
                "timeout": 3600,
                "direction": "minimize"
            }
        }
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_create_experiment(self, mock_manager, client, experiment_data):
        """测试创建实验API"""
        # 模拟ExperimentManager
        mock_experiment = Mock()
        mock_experiment.id = 1
        mock_experiment.name = experiment_data["name"]
        mock_experiment.state = ExperimentState.CREATED
        mock_experiment.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.create_experiment.return_value = mock_experiment
        mock_manager.return_value = mock_manager_instance
        
        # 发送POST请求
        response = client.post("/api/v1/hyperparameter-optimization/experiments", 
                             json=experiment_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == 1
        assert result["name"] == experiment_data["name"]
        assert result["state"] == "created"
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_experiment(self, mock_manager, client):
        """测试获取实验API"""
        experiment_id = 1
        
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.name = "Test Experiment"
        mock_experiment.state = ExperimentState.CREATED
        mock_experiment.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_experiment.return_value = mock_experiment
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == experiment_id
        assert result["name"] == "Test Experiment"
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_experiment_not_found(self, mock_manager, client):
        """测试获取不存在的实验"""
        experiment_id = 999
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_experiment.return_value = None
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_list_experiments(self, mock_manager, client):
        """测试列出实验API"""
        mock_experiments = [
            Mock(id=1, name="Experiment 1", state=ExperimentState.CREATED),
            Mock(id=2, name="Experiment 2", state=ExperimentState.RUNNING)
        ]
        
        for exp in mock_experiments:
            exp.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.list_experiments.return_value = mock_experiments
        mock_manager.return_value = mock_manager_instance
        
        response = client.get("/api/v1/hyperparameter-optimization/experiments")
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_start_experiment(self, mock_manager, client):
        """测试启动实验API"""
        experiment_id = 1
        
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.state = ExperimentState.RUNNING
        mock_experiment.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.update_experiment_state.return_value = mock_experiment
        mock_manager.return_value = mock_manager_instance
        
        response = client.post(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/start")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["state"] == "running"
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_stop_experiment(self, mock_manager, client):
        """测试停止实验API"""
        experiment_id = 1
        
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.state = ExperimentState.STOPPED
        mock_experiment.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.update_experiment_state.return_value = mock_experiment
        mock_manager.return_value = mock_manager_instance
        
        response = client.post(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/stop")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["state"] == "stopped"
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_delete_experiment(self, mock_manager, client):
        """测试删除实验API"""
        experiment_id = 1
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.delete_experiment.return_value = True
        mock_manager.return_value = mock_manager_instance
        
        response = client.delete(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["message"] == "实验删除成功"
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_create_trial(self, mock_manager, client):
        """测试创建试验API"""
        experiment_id = 1
        trial_data = {
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "state": "running"
        }
        
        mock_trial = Mock()
        mock_trial.id = 1
        mock_trial.experiment_id = experiment_id
        mock_trial.parameters = trial_data["parameters"]
        mock_trial.state = TrialState.RUNNING
        mock_trial.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.create_trial.return_value = mock_trial
        mock_manager.return_value = mock_manager_instance
        
        response = client.post(
            f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/trials",
            json=trial_data
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == 1
        assert result["experiment_id"] == experiment_id
        assert result["parameters"] == trial_data["parameters"]
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_update_trial_result(self, mock_manager, client):
        """测试更新试验结果API"""
        trial_id = 1
        result_data = {
            "value": 0.95,
            "state": "complete",
            "metrics": {"accuracy": 0.95, "loss": 0.05}
        }
        
        mock_trial = Mock()
        mock_trial.id = trial_id
        mock_trial.value = result_data["value"]
        mock_trial.state = TrialState.COMPLETE
        mock_trial.metrics = result_data["metrics"]
        mock_trial.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.update_trial_result.return_value = mock_trial
        mock_manager.return_value = mock_manager_instance
        
        response = client.put(
            f"/api/v1/hyperparameter-optimization/trials/{trial_id}/result",
            json=result_data
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["value"] == result_data["value"]
        assert result["state"] == "complete"
        assert result["metrics"] == result_data["metrics"]
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_experiment_trials(self, mock_manager, client):
        """测试获取实验试验列表API"""
        experiment_id = 1
        
        mock_trials = [
            Mock(id=1, experiment_id=experiment_id, value=0.9, state=TrialState.COMPLETE),
            Mock(id=2, experiment_id=experiment_id, value=0.85, state=TrialState.RUNNING)
        ]
        
        for trial in mock_trials:
            trial.created_at = utc_now()
            trial.parameters = {"learning_rate": 0.01}
            trial.metrics = {}
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_experiment_trials.return_value = mock_trials
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/trials")
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_best_trial(self, mock_manager, client):
        """测试获取最佳试验API"""
        experiment_id = 1
        
        mock_trial = Mock()
        mock_trial.id = 5
        mock_trial.experiment_id = experiment_id
        mock_trial.value = 0.95
        mock_trial.state = TrialState.COMPLETE
        mock_trial.parameters = {"learning_rate": 0.01, "batch_size": 32}
        mock_trial.created_at = utc_now()
        mock_trial.metrics = {"accuracy": 0.95}
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_best_trial.return_value = mock_trial
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/best-trial")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == 5
        assert result["value"] == 0.95
        assert result["parameters"]["learning_rate"] == 0.01
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_experiment_statistics(self, mock_manager, client):
        """测试获取实验统计信息API"""
        experiment_id = 1
        
        mock_stats = {
            "total_trials": 10,
            "completed_trials": 7,
            "running_trials": 2,
            "failed_trials": 1,
            "best_value": 0.95,
            "average_value": 0.87
        }
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_experiment_statistics.return_value = mock_stats
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/statistics")
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["total_trials"] == 10
        assert result["completed_trials"] == 7
        assert result["best_value"] == 0.95
    
    @patch('api.v1.hyperparameter_optimization.HyperparameterOptimizer')
    def test_optimize_endpoint(self, mock_optimizer, client):
        """测试优化端点API"""
        optimization_request = {
            "parameter_ranges": {
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.1}
            },
            "config": {
                "n_trials": 10,
                "algorithm": "tpe",
                "direction": "minimize"
            },
            "objective_function": "lambda params: params['learning_rate'] ** 2"
        }
        
        mock_result = {
            "best_value": 0.001,
            "best_params": {"learning_rate": 0.001},
            "n_trials": 10,
            "optimization_history": []
        }
        
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.optimize.return_value = mock_result
        mock_optimizer.return_value = mock_optimizer_instance
        
        response = client.post("/api/v1/hyperparameter-optimization/optimize", 
                             json=optimization_request)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["best_value"] == 0.001
        assert result["n_trials"] == 10
    
    def test_invalid_experiment_data(self, client):
        """测试无效实验数据"""
        invalid_data = {
            "name": "",  # 空名称
            "algorithm": "invalid_algorithm"  # 无效算法
        }
        
        response = client.post("/api/v1/hyperparameter-optimization/experiments", 
                             json=invalid_data)
        
        assert response.status_code == 422  # 验证错误
    
    def test_missing_required_fields(self, client):
        """测试缺少必需字段"""
        incomplete_data = {
            "name": "Test"
            # 缺少其他必需字段
        }
        
        response = client.post("/api/v1/hyperparameter-optimization/experiments", 
                             json=incomplete_data)
        
        assert response.status_code == 422  # 验证错误
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_search_experiments(self, mock_manager, client):
        """测试搜索实验API"""
        query = "test"
        
        mock_experiments = [
            Mock(id=1, name="Test Experiment 1", state=ExperimentState.CREATED),
            Mock(id=2, name="Another Test", state=ExperimentState.RUNNING)
        ]
        
        for exp in mock_experiments:
            exp.created_at = utc_now()
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.search_experiments.return_value = mock_experiments
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/search?q={query}")
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 2
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_get_trial_history(self, mock_manager, client):
        """测试获取试验历史API"""
        experiment_id = 1
        
        mock_history = [
            {"trial_id": 1, "value": 0.9, "timestamp": utc_now().isoformat()},
            {"trial_id": 2, "value": 0.85, "timestamp": utc_now().isoformat()}
        ]
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.get_trial_history.return_value = mock_history
        mock_manager.return_value = mock_manager_instance
        
        response = client.get(f"/api/v1/hyperparameter-optimization/experiments/{experiment_id}/history")
        
        assert response.status_code == 200
        
        result = response.json()
        assert len(result) == 2
        assert result[0]["trial_id"] == 1
    
    @patch('api.v1.hyperparameter_optimization.ExperimentManager')
    def test_prune_trial(self, mock_manager, client):
        """测试剪枝试验API"""
        trial_id = 1
        prune_data = {"reason": "Early stopping"}
        
        mock_trial = Mock()
        mock_trial.id = trial_id
        mock_trial.state = TrialState.PRUNED
        mock_trial.created_at = utc_now()
        mock_trial.parameters = {}
        mock_trial.metrics = {}
        
        mock_manager_instance = AsyncMock()
        mock_manager_instance.prune_trial.return_value = mock_trial
        mock_manager.return_value = mock_manager_instance
        
        response = client.post(f"/api/v1/hyperparameter-optimization/trials/{trial_id}/prune", 
                             json=prune_data)
        
        assert response.status_code == 200
        
        result = response.json()
        assert result["state"] == "pruned"
    
    def test_concurrent_requests(self, client):
        """测试并发请求处理"""
        import concurrent.futures
        import threading
        
        # 模拟多个并发请求
        def make_request():
            return client.get("/api/v1/hyperparameter-optimization/experiments")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            
            responses = []
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                responses.append(response.status_code)
        
        # 所有请求都应该成功（即使返回空列表）
        assert all(status in [200, 404] for status in responses)