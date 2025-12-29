"""
实验管理器单元测试
"""

import pytest
import pytest_asyncio
from contextlib import asynccontextmanager
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from fastapi import HTTPException
from src.core.utils.timezone_utils import utc_now
from ai.hyperparameter_optimization.experiment_manager import ExperimentManager
from ai.hyperparameter_optimization.models import (
    ExperimentRequest,
    HyperparameterRangeSchema,
    TrialState,
)

class TestExperimentManager:
    """实验管理器测试类"""

    @pytest_asyncio.fixture
    async def mock_session(self):
        session = AsyncMock()
        session.add = Mock()
        session.add_all = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest_asyncio.fixture
    async def manager(self, mock_session):
        @asynccontextmanager
        async def session_factory():
            yield mock_session

        return ExperimentManager(session_factory=session_factory)

    @pytest.fixture
    def experiment_request(self):
        return ExperimentRequest(
            name="Test Experiment",
            description="Test experiment for unit testing",
            algorithm="tpe",
            objective="minimize",
            n_trials=100,
            timeout=3600,
            early_stopping=True,
            patience=20,
            min_improvement=0.001,
            max_concurrent_trials=5,
            parameters=[
                HyperparameterRangeSchema(name="learning_rate", type="float", low=0.001, high=0.1),
                HyperparameterRangeSchema(name="batch_size", type="int", low=16, high=128),
            ],
        )

    @pytest.mark.asyncio
    async def test_create_experiment(self, manager, mock_session, experiment_request):
        """测试创建实验"""
        mock_experiment = Mock()
        mock_experiment.id = "exp-1"
        mock_experiment.name = experiment_request.name
        mock_experiment.status = "created"
        mock_experiment.algorithm = experiment_request.algorithm
        mock_experiment.objective = experiment_request.objective
        mock_experiment.created_at = utc_now()

        mock_study = Mock()

        with patch('ai.hyperparameter_optimization.experiment_manager.ExperimentModel', return_value=mock_experiment), \
             patch('ai.hyperparameter_optimization.experiment_manager.StudyMetadataModel', return_value=mock_study):
            result = await manager.create_experiment(experiment_request)

        assert result.id == str(mock_experiment.id)
        assert result.name == experiment_request.name
        assert result.status == "created"
        assert mock_session.add.call_count == 2
        assert mock_session.commit.await_count == 2
        mock_session.refresh.assert_awaited_once_with(mock_experiment)

    @pytest.mark.asyncio
    async def test_get_experiment(self, manager, mock_session):
        """测试获取实验"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.name = "Test Experiment"
        mock_experiment.description = "desc"
        mock_experiment.status = "created"
        mock_experiment.algorithm = "tpe"
        mock_experiment.objective = "minimize"
        mock_experiment.config = "{}"
        mock_experiment.parameters = "[]"
        mock_experiment.best_value = None
        mock_experiment.best_params = None
        mock_experiment.total_trials = 0
        mock_experiment.successful_trials = 0
        mock_experiment.pruned_trials = 0
        mock_experiment.failed_trials = 0
        mock_experiment.created_at = utc_now()
        mock_experiment.started_at = None
        mock_experiment.completed_at = None

        mock_trials = [Mock(id=1), Mock(id=2)]

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment

        result_trials = Mock()
        result_trials.scalars.return_value.all.return_value = mock_trials

        mock_session.execute = AsyncMock(side_effect=[result_exp, result_trials])

        result = await manager.get_experiment(experiment_id)

        assert result.id == str(mock_experiment.id)
        assert result.trials_count == 2

    @pytest.mark.asyncio
    async def test_get_experiment_not_found(self, manager, mock_session):
        """测试获取不存在的实验"""
        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_exp)

        with pytest.raises(HTTPException) as exc:
            await manager.get_experiment("11111111-1111-1111-1111-111111111111")

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_experiments(self, manager, mock_session):
        """测试列出实验"""
        mock_experiments = [
            Mock(id="1", name="Experiment 1", status="created", algorithm="tpe", objective="minimize", created_at=utc_now()),
            Mock(id="2", name="Experiment 2", status="created", algorithm="tpe", objective="minimize", created_at=utc_now()),
        ]
        result_list = Mock()
        result_list.scalars.return_value.all.return_value = mock_experiments
        mock_session.execute = AsyncMock(return_value=result_list)

        result = await manager.list_experiments(skip=0, limit=10)

        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"

    @pytest.mark.asyncio
    async def test_update_experiment_state(self, manager, mock_session):
        """测试更新实验状态"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.status = "created"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment
        mock_session.execute = AsyncMock(return_value=result_exp)

        result = await manager.update_experiment_state(experiment_id, "running")

        assert result.status == "running"
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once_with(mock_experiment)

    @pytest.mark.asyncio
    async def test_delete_experiment(self, manager, mock_session):
        """测试删除实验"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.status = "completed"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment
        result_delete = Mock()
        result_delete.rowcount = 1
        result_delete_meta = Mock()

        mock_session.execute = AsyncMock(side_effect=[result_exp, result_delete, result_delete_meta])

        result = await manager.delete_experiment(experiment_id)

        assert result["status"] == "deleted"
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_experiment_not_found(self, manager, mock_session):
        """测试删除不存在的实验"""
        experiment_id = "11111111-1111-1111-1111-111111111111"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = None
        result_delete = Mock()
        result_delete.rowcount = 0

        mock_session.execute = AsyncMock(side_effect=[result_exp, result_delete])

        with pytest.raises(HTTPException) as exc:
            await manager.delete_experiment(experiment_id)

        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_trial(self, manager, mock_session):
        """测试创建试验"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        trial_data = {
            "parameters": {"learning_rate": 0.01, "batch_size": 32},
            "state": TrialState.RUNNING,
        }

        result_max = Mock()
        result_max.scalar_one.return_value = 0
        mock_session.execute = AsyncMock(return_value=result_max)

        mock_trial = Mock()
        mock_trial.id = "trial-1"
        mock_trial.experiment_id = experiment_id
        mock_trial.parameters = trial_data["parameters"]

        with patch('ai.hyperparameter_optimization.experiment_manager.TrialModel', return_value=mock_trial):
            result = await manager.create_trial(experiment_id, trial_data)

        assert result.id == "trial-1"
        assert result.parameters == trial_data["parameters"]
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once_with(mock_trial)

    @pytest.mark.asyncio
    async def test_update_trial_result(self, manager, mock_session):
        """测试更新试验结果"""
        trial_id = "22222222-2222-2222-2222-222222222222"
        value = 0.95
        state = TrialState.COMPLETE
        metrics = {"accuracy": 0.95, "loss": 0.05}

        mock_trial = Mock()
        mock_trial.id = trial_id
        mock_trial.value = None
        mock_trial.state = TrialState.RUNNING.value
        mock_trial.start_time = utc_now()
        mock_trial.end_time = None
        mock_trial.duration = None

        result_trial = Mock()
        result_trial.scalar_one_or_none.return_value = mock_trial
        mock_session.execute = AsyncMock(return_value=result_trial)

        result = await manager.update_trial_result(trial_id, value, state, metrics)

        assert result.value == value
        assert result.state == TrialState.COMPLETE.value
        assert result.user_attrs == metrics
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once_with(mock_trial)

    @pytest.mark.asyncio
    async def test_get_experiment_trials(self, manager, mock_session):
        """测试获取实验的试验列表"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_trials = [
            Mock(id=1, experiment_id=experiment_id, value=0.9),
            Mock(id=2, experiment_id=experiment_id, value=0.8),
        ]

        result_trials = Mock()
        result_trials.scalars.return_value.all.return_value = mock_trials
        mock_session.execute = AsyncMock(return_value=result_trials)

        result = await manager.get_experiment_trials(experiment_id)

        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 2

    @pytest.mark.asyncio
    async def test_get_best_trial(self, manager, mock_session):
        """测试获取最佳试验"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.objective = "minimize"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment

        mock_trials = [
            Mock(id=1, value=0.9),
            Mock(id=2, value=0.8),
            Mock(id=3, value=0.95),
        ]
        result_trials = Mock()
        result_trials.scalars.return_value.all.return_value = mock_trials

        mock_session.execute = AsyncMock(side_effect=[result_exp, result_trials])

        result = await manager.get_best_trial(experiment_id)

        assert result.id == 2
        assert result.value == 0.8

    @pytest.mark.asyncio
    async def test_get_best_trial_maximize(self, manager, mock_session):
        """测试获取最佳试验（最大化）"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.objective = "maximize"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment

        mock_trials = [
            Mock(id=1, value=0.9),
            Mock(id=2, value=0.8),
            Mock(id=3, value=0.95),
        ]
        result_trials = Mock()
        result_trials.scalars.return_value.all.return_value = mock_trials

        mock_session.execute = AsyncMock(side_effect=[result_exp, result_trials])

        result = await manager.get_best_trial(experiment_id)

        assert result.id == 3
        assert result.value == 0.95

    @pytest.mark.asyncio
    async def test_get_experiment_statistics(self, manager, mock_session):
        """测试获取实验统计信息"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        mock_trials = [
            Mock(state=TrialState.COMPLETE.value, value=0.9),
            Mock(state=TrialState.COMPLETE.value, value=0.8),
            Mock(state=TrialState.RUNNING.value, value=None),
            Mock(state=TrialState.FAILED.value, value=None),
        ]

        mock_experiment = Mock()
        mock_experiment.id = experiment_id
        mock_experiment.objective = "minimize"

        result_exp = Mock()
        result_exp.scalar_one_or_none.return_value = mock_experiment
        mock_session.execute = AsyncMock(return_value=result_exp)

        with patch.object(manager, 'get_experiment_trials', AsyncMock(return_value=mock_trials)):
            stats = await manager.get_experiment_statistics(experiment_id)

        assert stats["total_trials"] == 4
        assert stats["completed_trials"] == 2
        assert stats["running_trials"] == 1
        assert stats["failed_trials"] == 1
        assert stats["best_value"] == 0.8
        assert abs(stats["average_value"] - 0.85) < 1e-10

    @pytest.mark.asyncio
    async def test_prune_trial(self, manager):
        """测试试验剪枝"""
        mock_trial = Mock()
        mock_trial.state = TrialState.PRUNED.value

        with patch.object(manager, 'update_trial_result', AsyncMock(return_value=mock_trial)):
            result = await manager.prune_trial("trial-1", "reason")

        assert result.state == TrialState.PRUNED.value

    @pytest.mark.asyncio
    async def test_get_trial_history(self, manager):
        """测试获取试验历史"""
        mock_trials = [
            Mock(id=1, value=0.9, start_time=utc_now(), end_time=None, parameters={"lr": 0.01}),
            Mock(id=2, value=0.8, start_time=utc_now(), end_time=None, parameters={"lr": 0.001}),
        ]

        with patch.object(manager, 'get_experiment_trials', AsyncMock(return_value=mock_trials)):
            history = await manager.get_trial_history("exp-1")

        assert len(history) == 2
        assert history[0]["trial_id"] == "1"
        assert history[1]["trial_id"] == "2"

    @pytest.mark.asyncio
    async def test_search_experiments(self, manager, mock_session):
        """测试搜索实验"""
        mock_experiments = [
            Mock(id=1, name="Test Experiment 1"),
            Mock(id=2, name="Another Test"),
        ]
        result_search = Mock()
        result_search.scalars.return_value.all.return_value = mock_experiments
        mock_session.execute = AsyncMock(return_value=result_search)

        result = await manager.search_experiments("test")

        assert len(result) == 2
        assert result[0].name == "Test Experiment 1"
        assert result[1].name == "Another Test"

    @pytest.mark.asyncio
    async def test_concurrent_trial_creation(self, manager, mock_session):
        """测试并发试验创建"""
        experiment_id = "11111111-1111-1111-1111-111111111111"
        trial_data_list = [
            {"trial_number": 1, "parameters": {"lr": 0.01}, "state": TrialState.RUNNING},
            {"trial_number": 2, "parameters": {"lr": 0.001}, "state": TrialState.RUNNING},
            {"trial_number": 3, "parameters": {"lr": 0.1}, "state": TrialState.RUNNING},
        ]

        mock_trials = []
        for i, data in enumerate(trial_data_list):
            mock_trial = Mock()
            mock_trial.id = f"trial-{i+1}"
            mock_trial.experiment_id = experiment_id
            mock_trial.parameters = data["parameters"]
            mock_trials.append(mock_trial)

        with patch('ai.hyperparameter_optimization.experiment_manager.TrialModel', side_effect=mock_trials):
            results = []
            for trial_data in trial_data_list:
                result = await manager.create_trial(experiment_id, trial_data)
                results.append(result)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.id == f"trial-{i+1}"
            assert result.experiment_id == experiment_id
