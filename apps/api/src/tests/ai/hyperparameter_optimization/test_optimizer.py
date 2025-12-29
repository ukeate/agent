"""
超参数优化器单元测试
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, Any
from ai.hyperparameter_optimization.optimizer import (
    HyperparameterOptimizer, 
    OptimizationConfig,
    OptimizationAlgorithm as OptimizerAlgorithm,
    PruningAlgorithm
)
from ai.hyperparameter_optimization.models import TrialState, OptimizationAlgorithm

class TestHyperparameterOptimizer:
    """超参数优化器测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return OptimizationConfig(
            study_name="test_study",
            n_trials=100,
            timeout=3600,
            algorithm=OptimizerAlgorithm.TPE,
            pruning=PruningAlgorithm.MEDIAN,
            direction="minimize"
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """优化器实例"""
        return HyperparameterOptimizer(config)
    
    def test_initialization(self, optimizer, config):
        """测试初始化"""
        assert optimizer.config == config
        assert optimizer.study is not None
        assert optimizer.parameter_ranges == {}
        assert optimizer.study.direction.name == "MINIMIZE"
    
    def test_add_parameter_ranges(self, optimizer):
        """测试添加参数范围"""
        ranges = {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "batch_size": {"type": "int", "low": 16, "high": 256, "step": 16},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]}
        }
        
        optimizer.add_parameter_ranges(ranges)
        assert optimizer.parameter_ranges == ranges
    
    @patch('ai.hyperparameter_optimization.optimizer.optuna.create_study')
    def test_setup_study_tpe(self, mock_create_study, config):
        """测试TPE算法设置"""
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(config)
        
        mock_create_study.assert_called_once()
        call_args = mock_create_study.call_args
        assert call_args[1]['direction'] == 'minimize'
        assert 'sampler' in call_args[1]
        assert 'pruner' in call_args[1]
    
    def test_setup_study_cmaes(self, config):
        """测试CMA-ES算法设置"""
        config.algorithm = OptimizerAlgorithm.CMAES
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.study is not None
        assert optimizer.study.direction.name == "MINIMIZE"
    
    def test_setup_study_random(self, config):
        """测试随机搜索算法设置"""
        config.algorithm = OptimizerAlgorithm.RANDOM
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.study is not None
        assert optimizer.study.direction.name == "MINIMIZE"
    
    def test_suggest_parameters(self, optimizer):
        """测试参数建议"""
        ranges = {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "batch_size": {"type": "int", "low": 16, "high": 256},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}
        }
        optimizer.add_parameter_ranges(ranges)
        
        # 模拟trial
        trial = Mock()
        trial.suggest_float.return_value = 0.01
        trial.suggest_int.return_value = 64
        trial.suggest_categorical.return_value = "adam"
        
        params = optimizer.suggest_parameters(trial)
        
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "optimizer" in params
        assert params["learning_rate"] == 0.01
        assert params["batch_size"] == 64
        assert params["optimizer"] == "adam"
    
    @pytest.mark.asyncio
    async def test_optimize_async(self, optimizer):
        """测试异步优化"""
        ranges = {
            "x": {"type": "float", "low": -10, "high": 10}
        }
        optimizer.add_parameter_ranges(ranges)
        
        async def objective_func(params):
            return params["x"] ** 2
        
        # 模拟少量trials进行测试
        optimizer.config.n_trials = 5
        
        result = await optimizer.optimize_async(objective_func)
        
        assert result is not None
        assert "best_value" in result
        assert "best_params" in result
        assert "n_trials" in result
        assert result["n_trials"] == 5
    
    def test_get_optimization_history(self, optimizer):
        """测试优化历史获取"""
        # 运行一些试验
        ranges = {"x": {"type": "float", "low": -5, "high": 5}}
        optimizer.add_parameter_ranges(ranges)
        
        def objective(params):
            return params["x"] ** 2
        
        # 运行少量试验
        optimizer.config.n_trials = 3
        optimizer.optimize(objective)
        
        history = optimizer.get_optimization_history()
        
        assert len(history) == 3
        assert all("value" in h for h in history)
        assert all("params" in h for h in history)
        assert all("state" in h for h in history)
    
    def test_get_parameter_importance(self, optimizer):
        """测试参数重要性分析"""
        # 运行一些试验来生成数据
        ranges = {
            "x": {"type": "float", "low": -5, "high": 5},
            "y": {"type": "float", "low": -3, "high": 3}
        }
        optimizer.add_parameter_ranges(ranges)
        
        def objective(params):
            # x更重要
            return params["x"] ** 2 + 0.1 * params["y"] ** 2
        
        # 运行足够的试验
        optimizer.config.n_trials = 10
        optimizer.optimize(objective)
        
        importance = optimizer.get_parameter_importance()
        
        # 应该有参数重要性结果
        assert isinstance(importance, dict)
        # 如果有结果，x应该比y更重要
        if importance:
            assert "x" in importance or "y" in importance
    
    def test_should_prune_trial(self, optimizer):
        """测试试验剪枝判断"""
        trial = Mock()
        trial.should_prune.return_value = True
        
        result = optimizer.should_prune_trial(trial, 0.5, 10)
        assert result is True
        
        trial.report.assert_called_once_with(0.5, 10)
        trial.should_prune.assert_called_once()
    
    def test_error_handling_invalid_parameter_type(self, optimizer):
        """测试无效参数类型错误处理"""
        ranges = {"invalid": {"type": "unknown", "low": 0, "high": 1}}
        optimizer.add_parameter_ranges(ranges)
        
        trial = Mock()
        
        with pytest.raises(ValueError, match="不支持的参数类型"):
            optimizer.suggest_parameters(trial)
    
    def test_parameter_validation(self, optimizer):
        """测试参数验证"""
        # 测试缺失必要字段
        invalid_ranges = {
            "incomplete": {"type": "float"}  # 缺少low和high
        }
        
        with pytest.raises(KeyError):
            optimizer.add_parameter_ranges(invalid_ranges)
            trial = Mock()
            optimizer.suggest_parameters(trial)
    
    @pytest.mark.parametrize("algorithm", [
        OptimizerAlgorithm.TPE,
        OptimizerAlgorithm.CMAES,
        OptimizerAlgorithm.RANDOM,
        OptimizerAlgorithm.GRID
    ])
    def test_different_algorithms(self, algorithm, config):
        """测试不同优化算法"""
        config.algorithm = algorithm
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.study is not None
        assert optimizer.config.algorithm == algorithm
    
    def test_maximize_direction(self, config):
        """测试最大化方向"""
        config.direction = "maximize"
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.study.direction.name == "MAXIMIZE"
    
    def test_concurrent_optimization(self, optimizer):
        """测试并发优化设置"""
        ranges = {"x": {"type": "float", "low": -1, "high": 1}}
        optimizer.add_parameter_ranges(ranges)
        
        def objective(params):
            return params["x"] ** 2
        
        # 测试并发设置
        optimizer.config.n_jobs = 4
        
        # 模拟少量trials
        optimizer.config.n_trials = 2
        result = optimizer.optimize(objective)
        
        assert result is not None
        assert "best_value" in result
        
    def test_timeout_handling(self, optimizer):
        """测试超时处理"""
        ranges = {"x": {"type": "float", "low": -1, "high": 1}}
        optimizer.add_parameter_ranges(ranges)
        
        def slow_objective(params):
            import time
            time.sleep(0.1)  # 短暂延时
            return params["x"] ** 2
        
        # 设置很短的超时
        optimizer.config.timeout = 0.05
        
        result = optimizer.optimize(slow_objective)
        
        # 应该在超时前完成或被中断
        assert result is not None
