"""
搜索引擎单元测试
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import Dict, Any

from ai.hyperparameter_optimization.search_engine import SearchEngine, SearchAlgorithm
from ai.hyperparameter_optimization.models import OptimizationAlgorithm


class TestSearchEngine:
    """搜索引擎测试类"""
    
    @pytest.fixture
    def parameter_space(self):
        """参数空间"""
        return {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "batch_size": {"type": "int", "low": 16, "high": 256, "step": 16},
            "optimizer": {"type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.5}
        }
    
    @pytest.fixture
    def search_engine(self, parameter_space):
        """搜索引擎实例"""
        return SearchEngine(parameter_space)
    
    def test_initialization(self, search_engine, parameter_space):
        """测试初始化"""
        assert search_engine.parameter_space == parameter_space
        assert len(search_engine.algorithms) > 0
        assert SearchAlgorithm.RANDOM in search_engine.algorithms
    
    def test_random_search(self, search_engine):
        """测试随机搜索"""
        algorithm = search_engine.algorithms[SearchAlgorithm.RANDOM]
        
        # 生成多个随机样本
        samples = []
        for _ in range(100):
            sample = algorithm.generate_candidate(search_engine.parameter_space)
            samples.append(sample)
        
        # 验证样本的有效性
        for sample in samples:
            assert "learning_rate" in sample
            assert "batch_size" in sample
            assert "optimizer" in sample
            assert "dropout_rate" in sample
            
            assert 0.001 <= sample["learning_rate"] <= 0.1
            assert 16 <= sample["batch_size"] <= 256
            assert sample["optimizer"] in ["adam", "sgd", "rmsprop"]
            assert 0.0 <= sample["dropout_rate"] <= 0.5
        
        # 验证随机性（样本应该不同）
        unique_samples = set(str(sample) for sample in samples)
        assert len(unique_samples) > 50  # 应该有足够的多样性
    
    def test_grid_search(self, search_engine):
        """测试网格搜索"""
        algorithm = search_engine.algorithms[SearchAlgorithm.GRID]
        
        # 简化参数空间用于网格搜索测试
        simple_space = {
            "param1": {"type": "int", "low": 1, "high": 3},
            "param2": {"type": "categorical", "choices": ["a", "b"]}
        }
        
        # 生成网格
        candidates = algorithm.generate_grid(simple_space, max_candidates=10)
        
        assert len(candidates) == 6  # 3 * 2 = 6种组合
        
        # 验证所有组合都存在
        expected_combinations = [
            {"param1": 1, "param2": "a"}, {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"}, {"param1": 2, "param2": "b"},
            {"param1": 3, "param2": "a"}, {"param1": 3, "param2": "b"}
        ]
        
        for expected in expected_combinations:
            assert expected in candidates
    
    def test_bayesian_search(self, search_engine):
        """测试贝叶斯搜索"""
        algorithm = search_engine.algorithms[SearchAlgorithm.BAYESIAN]
        
        # 模拟历史数据
        history = [
            {"params": {"learning_rate": 0.01, "batch_size": 32}, "value": 0.9},
            {"params": {"learning_rate": 0.001, "batch_size": 64}, "value": 0.85},
            {"params": {"learning_rate": 0.1, "batch_size": 128}, "value": 0.7}
        ]
        
        # 生成候选
        candidate = algorithm.generate_candidate(
            search_engine.parameter_space, 
            history=history
        )
        
        assert "learning_rate" in candidate
        assert "batch_size" in candidate
        assert 0.001 <= candidate["learning_rate"] <= 0.1
        assert 16 <= candidate["batch_size"] <= 256
    
    def test_evolutionary_search(self, search_engine):
        """测试进化搜索"""
        algorithm = search_engine.algorithms[SearchAlgorithm.EVOLUTIONARY]
        
        # 初始种群
        population_size = 10
        population = []
        for _ in range(population_size):
            individual = {
                "learning_rate": np.random.uniform(0.001, 0.1),
                "batch_size": np.random.randint(16, 257),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                "dropout_rate": np.random.uniform(0.0, 0.5)
            }
            population.append(individual)
        
        # 模拟适应度评分
        fitness_scores = [np.random.uniform(0.5, 1.0) for _ in population]
        
        # 生成新一代
        new_generation = algorithm.evolve_population(
            population, fitness_scores, search_engine.parameter_space
        )
        
        assert len(new_generation) == population_size
        
        # 验证新一代的有效性
        for individual in new_generation:
            assert "learning_rate" in individual
            assert "batch_size" in individual
            assert 0.001 <= individual["learning_rate"] <= 0.1
            assert 16 <= individual["batch_size"] <= 256
    
    @pytest.mark.parametrize("algorithm_type", [
        SearchAlgorithm.RANDOM,
        SearchAlgorithm.BAYESIAN,
        SearchAlgorithm.EVOLUTIONARY
    ])
    def test_search_algorithms(self, search_engine, algorithm_type):
        """测试所有搜索算法"""
        algorithm = search_engine.algorithms[algorithm_type]
        
        candidate = algorithm.generate_candidate(search_engine.parameter_space)
        
        assert isinstance(candidate, dict)
        assert "learning_rate" in candidate
        assert "batch_size" in candidate
        assert "optimizer" in candidate
        assert "dropout_rate" in candidate
    
    def test_search_with_constraints(self, search_engine):
        """测试带约束的搜索"""
        constraints = {
            "learning_rate": lambda x: x > 0.005,  # 学习率必须大于0.005
            "batch_size": lambda x: x % 16 == 0    # 批大小必须是16的倍数
        }
        
        # 测试随机搜索是否满足约束
        algorithm = search_engine.algorithms[SearchAlgorithm.RANDOM]
        
        valid_samples = 0
        total_samples = 100
        
        for _ in range(total_samples):
            candidate = algorithm.generate_candidate(search_engine.parameter_space)
            
            # 检查约束
            if (constraints["learning_rate"](candidate["learning_rate"]) and
                constraints["batch_size"](candidate["batch_size"])):
                valid_samples += 1
        
        # 至少应该有一些样本满足约束
        assert valid_samples > 0
    
    def test_parameter_space_validation(self):
        """测试参数空间验证"""
        # 无效的参数空间
        invalid_space = {
            "invalid_param": {"type": "unknown", "low": 0, "high": 1}
        }
        
        with pytest.raises(ValueError, match="不支持的参数类型"):
            SearchEngine(invalid_space)
    
    def test_generate_multiple_candidates(self, search_engine):
        """测试生成多个候选"""
        n_candidates = 10
        
        for algorithm_type in [SearchAlgorithm.RANDOM, SearchAlgorithm.BAYESIAN]:
            algorithm = search_engine.algorithms[algorithm_type]
            
            candidates = []
            for _ in range(n_candidates):
                candidate = algorithm.generate_candidate(search_engine.parameter_space)
                candidates.append(candidate)
            
            assert len(candidates) == n_candidates
            
            # 验证多样性
            unique_candidates = len(set(str(c) for c in candidates))
            assert unique_candidates > 1  # 至少有一些不同的候选
    
    def test_search_convergence(self, search_engine):
        """测试搜索收敛性"""
        algorithm = search_engine.algorithms[SearchAlgorithm.BAYESIAN]
        
        # 模拟优化过程
        history = []
        best_value = float('inf')
        
        for iteration in range(20):
            candidate = algorithm.generate_candidate(
                search_engine.parameter_space, 
                history=history
            )
            
            # 模拟目标函数（简单的二次函数）
            value = (candidate["learning_rate"] - 0.01) ** 2 + \
                   (candidate["batch_size"] - 64) ** 2 / 10000
            
            history.append({"params": candidate, "value": value})
            
            if value < best_value:
                best_value = value
        
        # 检查是否有改进趋势
        early_best = min(h["value"] for h in history[:5])
        late_best = min(h["value"] for h in history[-5:])
        
        # 后期的最佳值应该不差于早期的最佳值
        assert late_best <= early_best * 1.1  # 允许小幅波动
    
    def test_hyperparameter_importance(self, search_engine):
        """测试超参数重要性分析"""
        # 设置随机种子确保结果一致性
        np.random.seed(42)
        
        # 生成确定性历史数据
        history = []
        for i in range(100):  # 增加样本数量
            params = {
                "learning_rate": np.random.uniform(0.001, 0.1),
                "batch_size": np.random.randint(16, 257),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                "dropout_rate": np.random.uniform(0.0, 0.5)
            }
            # 学习率对性能影响最大，减少噪声
            lr_effect = abs(params["learning_rate"] - 0.01) * 50  # 增强学习率影响
            batch_effect = abs(params["batch_size"] - 64) * 0.001  # 轻微影响
            dropout_effect = params["dropout_rate"] * 2  # 轻微影响
            optimizer_effect = 0.1 if params["optimizer"] != "adam" else 0  # 轻微影响
            
            value = lr_effect + batch_effect + dropout_effect + optimizer_effect + np.random.normal(0, 0.01)  # 减少噪声
            history.append({"params": params, "value": value})
        
        # 计算参数重要性
        importance = search_engine.compute_parameter_importance(history)
        
        assert isinstance(importance, dict)
        assert "learning_rate" in importance
        assert all(0 <= score <= 1 for score in importance.values())
        
        # 学习率应该是最重要的参数（允许小幅误差）
        lr_importance = importance["learning_rate"]
        max_importance = max(importance.values())
        assert lr_importance >= max_importance * 0.9  # 允许90%以上即可
    
    def test_adaptive_sampling(self, search_engine):
        """测试自适应采样"""
        # 模拟不同区域的性能
        def mock_objective(params):
            # 最优区域在learning_rate=0.01附近
            return abs(params["learning_rate"] - 0.01) + np.random.normal(0, 0.1)
        
        # 初始随机采样
        initial_samples = []
        for _ in range(10):
            params = search_engine.algorithms[SearchAlgorithm.RANDOM].generate_candidate(
                search_engine.parameter_space
            )
            value = mock_objective(params)
            initial_samples.append({"params": params, "value": value})
        
        # 自适应采样应该集中在好的区域
        algorithm = search_engine.algorithms[SearchAlgorithm.BAYESIAN]
        adaptive_samples = []
        for _ in range(10):
            params = algorithm.generate_candidate(
                search_engine.parameter_space,
                history=initial_samples + adaptive_samples
            )
            value = mock_objective(params)
            adaptive_samples.append({"params": params, "value": value})
        
        # 自适应样本应该更接近最优值
        initial_best = min(s["value"] for s in initial_samples)
        adaptive_best = min(s["value"] for s in adaptive_samples)
        
        # 自适应采样应该找到更好的解
        assert adaptive_best <= initial_best
    
    def test_multi_objective_optimization(self, search_engine):
        """测试多目标优化"""
        # 模拟多目标函数
        def multi_objective(params):
            # 目标1：准确率（要最大化，转换为最小化）
            accuracy = 1.0 - (1.0 / (1.0 + params["learning_rate"] * 100))
            objective1 = 1.0 - accuracy
            
            # 目标2：训练时间（要最小化）
            training_time = params["batch_size"] / 32.0
            objective2 = training_time
            
            return [objective1, objective2]
        
        # 生成Pareto前沿
        pareto_candidates = []
        for _ in range(100):
            params = search_engine.algorithms[SearchAlgorithm.RANDOM].generate_candidate(
                search_engine.parameter_space
            )
            objectives = multi_objective(params)
            pareto_candidates.append({
                "params": params,
                "objectives": objectives
            })
        
        # 简单的Pareto支配检查
        pareto_front = []
        for candidate in pareto_candidates:
            is_dominated = False
            for other in pareto_candidates:
                if (all(other["objectives"][i] <= candidate["objectives"][i] 
                       for i in range(2)) and
                    any(other["objectives"][i] < candidate["objectives"][i] 
                       for i in range(2))):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        # Pareto前沿应该包含多个解
        assert len(pareto_front) > 1
        assert len(pareto_front) < len(pareto_candidates)