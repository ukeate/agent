"""
实验管理器单元测试
"""
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, Any

from ai.hyperparameter_optimization.experiment_manager import ExperimentManager
from ai.hyperparameter_optimization.models import (
    Experiment, Trial, TrialState, ExperimentState, OptimizationAlgorithm
)


class TestExperimentManager:
    """实验管理器测试类"""
    
    @pytest_asyncio.fixture
    async def manager(self):
        """实验管理器实例"""
        with patch('ai.hyperparameter_optimization.experiment_manager.get_db_session') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__aenter__.return_value = mock_session
            
            manager = ExperimentManager()
            manager.db_session = mock_session
            yield manager
    
    @pytest.fixture
    def experiment_data(self):
        """实验数据"""
        return {
            "name": "Test Experiment",
            "description": "Test experiment for unit testing",
            "algorithm": OptimizationAlgorithm.TPE,
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
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, manager, experiment_data):
        """测试创建实验"""
        mock_experiment = Mock()
        mock_experiment.id = 1
        mock_experiment.name = experiment_data["name"]
        mock_experiment.state = ExperimentState.CREATED
        
        manager.db_session.add = Mock()
        manager.db_session.commit = Mock()
        manager.db_session.refresh = Mock()
        
        with patch('ai.hyperparameter_optimization.models.Experiment', return_value=mock_experiment):
            result = await manager.create_experiment(experiment_data)
            
            assert result.id == 1
            assert result.name == experiment_data["name"]
            assert result.state == ExperimentState.CREATED
            
            manager.db_session.add.assert_called_once()
            manager.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_experiment(self, manager):
        """测试获取实验"""
        experiment_id = 1
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.name = "Test Experiment"
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=mock_experiment)
        manager.db_session.query.return_value = mock_query
        
        result = await manager.get_experiment(experiment_id)
        
        assert result.id == experiment_id
        assert result.name == "Test Experiment"
    
    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self, manager):
        """测试获取不存在的实验"""
        experiment_id = 999
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=None)
        manager.db_session.query.return_value = mock_query
        
        result = await manager.get_experiment(experiment_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, manager):
        """测试列出实验"""
        mock_experiments = [
            Mock(id=1, name="Experiment 1"),
            Mock(id=2, name="Experiment 2")
        ]
        
        mock_query = Mock()
        mock_query.offset.return_value.limit.return_value.all = Mock(
            return_value=mock_experiments
        )
        manager.db_session.query.return_value = mock_query
        
        result = await manager.list_experiments(skip=0, limit=10)
        
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2
    
    @pytest.mark.asyncio
    async def test_update_experiment_state(self, manager):
        """测试更新实验状态"""
        experiment_id = 1
        new_state = ExperimentState.RUNNING
        
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.state = ExperimentState.CREATED
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=mock_experiment)
        manager.db_session.query.return_value = mock_query
        manager.db_session.commit = Mock()
        
        result = await manager.update_experiment_state(experiment_id, new_state)
        
        assert result.state == new_state
        manager.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_experiment(self, manager):
        """测试删除实验"""
        experiment_id = 1
        
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=mock_experiment)
        manager.db_session.query.return_value = mock_query
        manager.db_session.delete = Mock()
        manager.db_session.commit = Mock()
        
        result = await manager.delete_experiment(experiment_id)
        
        assert result is True
        manager.db_session.delete.assert_called_once_with(mock_experiment)
        manager.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_experiment_not_found(self, manager):
        """测试删除不存在的实验"""
        experiment_id = 999
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=None)
        manager.db_session.query.return_value = mock_query
        
        result = await manager.delete_experiment(experiment_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_trial(self, manager):
        """测试创建试验"""
        experiment_id = 1
        trial_data = {
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "state": TrialState.RUNNING
        }
        
        mock_trial = Mock()
        mock_trial.id = 1
        mock_trial.experiment_id = experiment_id
        mock_trial.parameters = trial_data["parameters"]
        
        manager.db_session.add = Mock()
        manager.db_session.commit = Mock()
        manager.db_session.refresh = Mock()
        
        with patch('ai.hyperparameter_optimization.models.Trial', return_value=mock_trial):
            result = await manager.create_trial(experiment_id, trial_data)
            
            assert result.id == 1
            assert result.experiment_id == experiment_id
            assert result.parameters == trial_data["parameters"]
    
    @pytest.mark.asyncio
    async def test_update_trial_result(self, manager):
        """测试更新试验结果"""
        trial_id = 1
        value = 0.95
        state = TrialState.COMPLETE
        metrics = {"accuracy": 0.95, "loss": 0.05}
        
        mock_trial = Mock()
        mock_trial.id = trial_id
        mock_trial.value = None
        mock_trial.state = TrialState.RUNNING
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=mock_trial)
        manager.db_session.query.return_value = mock_query
        manager.db_session.commit = Mock()
        
        result = await manager.update_trial_result(trial_id, value, state, metrics)
        
        assert result.value == value
        assert result.state == state
        assert result.metrics == metrics
        manager.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_experiment_trials(self, manager):
        """测试获取实验的试验列表"""
        experiment_id = 1
        mock_trials = [
            Mock(id=1, experiment_id=experiment_id, value=0.9),
            Mock(id=2, experiment_id=experiment_id, value=0.8)
        ]
        
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all = Mock(
            return_value=mock_trials
        )
        manager.db_session.query.return_value = mock_query
        
        result = await manager.get_experiment_trials(experiment_id)
        
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2
    
    @pytest.mark.asyncio
    async def test_get_best_trial(self, manager):
        """测试获取最佳试验"""
        experiment_id = 1
        
        # 模拟实验
        mock_experiment = Mock()
        mock_experiment.optimization_config = {"direction": "minimize"}
        
        mock_query_exp = Mock()
        mock_query_exp.filter.return_value.first = Mock(return_value=mock_experiment)
        
        # 模拟试验
        mock_trials = [
            Mock(id=1, value=0.9, state=TrialState.COMPLETE),
            Mock(id=2, value=0.8, state=TrialState.COMPLETE),  # 最佳
            Mock(id=3, value=0.95, state=TrialState.COMPLETE)
        ]
        
        mock_query_trial = Mock()
        mock_query_trial.filter.return_value.order_by.return_value.all = Mock(return_value=mock_trials)
        
        manager.db_session.query.side_effect = [mock_query_trial, mock_query_exp]
        
        result = await manager.get_best_trial(experiment_id)
        
        assert result.id == 2
        assert result.value == 0.8
    
    @pytest.mark.asyncio
    async def test_get_best_trial_maximize(self, manager):
        """测试获取最佳试验（最大化）"""
        experiment_id = 1
        
        # 模拟实验（最大化）
        mock_experiment = Mock()
        mock_experiment.optimization_config = {"direction": "maximize"}
        
        mock_query_exp = Mock()
        mock_query_exp.filter.return_value.first = Mock(return_value=mock_experiment)
        
        # 模拟试验
        mock_trials = [
            Mock(id=1, value=0.9, state=TrialState.COMPLETE),
            Mock(id=2, value=0.8, state=TrialState.COMPLETE),
            Mock(id=3, value=0.95, state=TrialState.COMPLETE)  # 最佳
        ]
        
        mock_query_trial = Mock()
        mock_query_trial.filter.return_value.order_by.return_value.all = Mock(return_value=mock_trials)
        
        manager.db_session.query.side_effect = [mock_query_trial, mock_query_exp]
        
        result = await manager.get_best_trial(experiment_id)
        
        assert result.id == 3
        assert result.value == 0.95
    
    @pytest.mark.asyncio
    async def test_get_experiment_statistics(self, manager):
        """测试获取实验统计信息"""
        experiment_id = 1
        
        mock_trials = [
            Mock(state=TrialState.COMPLETE, value=0.9),
            Mock(state=TrialState.COMPLETE, value=0.8),
            Mock(state=TrialState.RUNNING, value=None),
            Mock(state=TrialState.FAILED, value=None)
        ]
        
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all = Mock(return_value=mock_trials)
        manager.db_session.query.return_value = mock_query
        
        stats = await manager.get_experiment_statistics(experiment_id)
        
        assert stats["total_trials"] == 4
        assert stats["completed_trials"] == 2
        assert stats["running_trials"] == 1
        assert stats["failed_trials"] == 1
        assert stats["best_value"] == 0.8  # 假设是最小化
        assert abs(stats["average_value"] - 0.85) < 1e-10
    
    @pytest.mark.asyncio
    async def test_prune_trial(self, manager):
        """测试试验剪枝"""
        trial_id = 1
        reason = "Early stopping due to poor performance"
        
        mock_trial = Mock()
        mock_trial.id = trial_id
        mock_trial.state = TrialState.RUNNING
        
        mock_query = Mock()
        mock_query.filter.return_value.first = Mock(return_value=mock_trial)
        manager.db_session.query.return_value = mock_query
        manager.db_session.commit = Mock()
        
        result = await manager.prune_trial(trial_id, reason)
        
        assert result.state == TrialState.PRUNED
        manager.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_trial_history(self, manager):
        """测试获取试验历史"""
        experiment_id = 1
        
        mock_trials = [
            Mock(
                id=1, 
                value=0.9, 
                created_at=datetime(2024, 1, 1, 10, 0),
                parameters={"lr": 0.01}
            ),
            Mock(
                id=2, 
                value=0.8, 
                created_at=datetime(2024, 1, 1, 11, 0),
                parameters={"lr": 0.001}
            )
        ]
        
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all = Mock(
            return_value=mock_trials
        )
        manager.db_session.query.return_value = mock_query
        
        history = await manager.get_trial_history(experiment_id)
        
        assert len(history) == 2
        assert history[0]["trial_id"] == 1
        assert history[0]["value"] == 0.9
        assert history[1]["trial_id"] == 2
        assert history[1]["value"] == 0.8
    
    @pytest.mark.asyncio
    async def test_search_experiments(self, manager):
        """测试搜索实验"""
        query = "test"
        
        mock_exp1 = Mock()
        mock_exp1.id = 1
        mock_exp1.name = "Test Experiment 1"
        
        mock_exp2 = Mock()
        mock_exp2.id = 2
        mock_exp2.name = "Another Test"
        
        mock_experiments = [mock_exp1, mock_exp2]
        
        mock_query_obj = Mock()
        mock_query_obj.filter.return_value.all = Mock(return_value=mock_experiments)
        manager.db_session.query.return_value = mock_query_obj
        
        result = await manager.search_experiments(query)
        
        assert len(result) == 2
        assert result[0].name == "Test Experiment 1"
        assert result[1].name == "Another Test"
    
    @pytest.mark.asyncio
    async def test_concurrent_trial_creation(self, manager):
        """测试并发试验创建"""
        experiment_id = 1
        
        # 模拟并发创建多个试验
        trial_data_list = [
            {"parameters": {"lr": 0.01}, "state": TrialState.RUNNING},
            {"parameters": {"lr": 0.001}, "state": TrialState.RUNNING},
            {"parameters": {"lr": 0.1}, "state": TrialState.RUNNING}
        ]
        
        mock_trials = []
        for i, data in enumerate(trial_data_list):
            mock_trial = Mock()
            mock_trial.id = i + 1
            mock_trial.experiment_id = experiment_id
            mock_trial.parameters = data["parameters"]
            mock_trials.append(mock_trial)
        
        manager.db_session.add = Mock()
        manager.db_session.commit = Mock()
        manager.db_session.refresh = Mock()
        
        results = []
        with patch('ai.hyperparameter_optimization.models.Trial', side_effect=mock_trials):
            for trial_data in trial_data_list:
                result = await manager.create_trial(experiment_id, trial_data)
                results.append(result)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.id == i + 1
            assert result.experiment_id == experiment_id