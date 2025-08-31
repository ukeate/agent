"""
集成的超参数搜索引擎

提供预设的任务配置和算法对比功能，简化不同场景下的超参数优化。
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import logging
import numpy as np
from enum import Enum

from .optimizer import (
    HyperparameterOptimizer,
    OptimizationConfig,
    HyperparameterRange,
    OptimizationAlgorithm,
    PruningAlgorithm
)


class SearchAlgorithm(Enum):
    """搜索算法枚举"""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


class SearchAlgorithmImpl:
    """搜索算法实现基类"""
    
    def generate_candidate(self, parameter_space: Dict, history: List = None) -> Dict:
        """生成候选参数"""
        raise NotImplementedError
    
    def generate_grid(self, parameter_space: Dict, max_candidates: int = 100) -> List[Dict]:
        """生成网格搜索参数"""
        raise NotImplementedError
    
    def evolve_population(self, population: List, fitness_scores: List, parameter_space: Dict) -> List:
        """进化种群"""
        raise NotImplementedError


class RandomSearchAlgorithm(SearchAlgorithmImpl):
    """随机搜索算法"""
    
    def generate_candidate(self, parameter_space: Dict, history: List = None) -> Dict:
        candidate = {}
        for param_name, config in parameter_space.items():
            if config["type"] == "float":
                if config.get("log", False):
                    value = np.exp(np.random.uniform(np.log(config["low"]), np.log(config["high"])))
                else:
                    value = np.random.uniform(config["low"], config["high"])
                candidate[param_name] = value
            elif config["type"] == "int":
                step = config.get("step", 1)
                low = config["low"]
                high = config["high"]
                value = np.random.choice(range(low, high + 1, step))
                candidate[param_name] = int(value)
            elif config["type"] == "categorical":
                candidate[param_name] = np.random.choice(config["choices"])
        return candidate


class GridSearchAlgorithm(SearchAlgorithmImpl):
    """网格搜索算法"""
    
    def generate_grid(self, parameter_space: Dict, max_candidates: int = 100) -> List[Dict]:
        import itertools
        
        param_values = {}
        for param_name, config in parameter_space.items():
            if config["type"] == "int":
                step = config.get("step", 1)
                values = list(range(config["low"], config["high"] + 1, step))
                param_values[param_name] = values[:10]  # 限制每个参数的值数量
            elif config["type"] == "categorical":
                param_values[param_name] = config["choices"]
            else:
                # 对于浮点数，取几个代表性的值
                param_values[param_name] = [config["low"], (config["low"] + config["high"]) / 2, config["high"]]
        
        # 生成所有组合
        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]
        candidates = []
        
        for combination in itertools.product(*values):
            candidate = dict(zip(keys, combination))
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break
        
        return candidates


class BayesianSearchAlgorithm(SearchAlgorithmImpl):
    """贝叶斯搜索算法"""
    
    def generate_candidate(self, parameter_space: Dict, history: List = None) -> Dict:
        # 简化实现：如果有历史，基于最佳结果附近搜索
        if history and len(history) > 0:
            best_trial = min(history, key=lambda x: x["value"])
            best_params = best_trial["params"]
            
            candidate = {}
            for param_name, config in parameter_space.items():
                if param_name in best_params:
                    if config["type"] == "float":
                        # 在最佳值附近搜索
                        center = best_params[param_name]
                        range_width = (config["high"] - config["low"]) * 0.1
                        value = np.random.normal(center, range_width)
                        value = np.clip(value, config["low"], config["high"])
                        candidate[param_name] = value
                    elif config["type"] == "int":
                        center = best_params[param_name]
                        step = config.get("step", 1)
                        noise = np.random.randint(-2, 3) * step
                        value = center + noise
                        value = np.clip(value, config["low"], config["high"])
                        candidate[param_name] = int(value)
                    elif config["type"] == "categorical":
                        # 80%概率选择最佳值，20%概率随机
                        if np.random.random() < 0.8:
                            candidate[param_name] = best_params[param_name]
                        else:
                            candidate[param_name] = np.random.choice(config["choices"])
                else:
                    # 新参数，随机生成
                    random_algo = RandomSearchAlgorithm()
                    candidate.update(random_algo.generate_candidate({param_name: config}))
            return candidate
        else:
            # 没有历史，使用随机搜索
            random_algo = RandomSearchAlgorithm()
            return random_algo.generate_candidate(parameter_space)


class EvolutionarySearchAlgorithm(SearchAlgorithmImpl):
    """进化搜索算法"""
    
    def generate_candidate(self, parameter_space: Dict, history: List = None) -> Dict[str, Any]:
        """生成候选参数（初始个体）"""
        candidate = {}
        for param_name, config in parameter_space.items():
            if config["type"] == "float":
                candidate[param_name] = np.random.uniform(config["low"], config["high"])
            elif config["type"] == "int":
                candidate[param_name] = np.random.randint(config["low"], config["high"] + 1)
            elif config["type"] == "categorical":
                candidate[param_name] = np.random.choice(config["choices"])
        return candidate
    
    def evolve_population(self, population: List, fitness_scores: List, parameter_space: Dict) -> List:
        # 选择最佳个体
        sorted_indices = np.argsort(fitness_scores)
        elite_size = len(population) // 4
        elite = [population[i] for i in sorted_indices[:elite_size]]
        
        # 生成新一代
        new_generation = elite.copy()
        
        while len(new_generation) < len(population):
            # 选择父母
            parent1 = elite[np.random.randint(len(elite))]
            parent2 = elite[np.random.randint(len(elite))]
            
            # 交叉
            child = {}
            for param_name in parameter_space:
                if np.random.random() < 0.5:
                    child[param_name] = parent1.get(param_name)
                else:
                    child[param_name] = parent2.get(param_name)
            
            # 变异
            if np.random.random() < 0.1:
                param_to_mutate = np.random.choice(list(parameter_space.keys()))
                config = parameter_space[param_to_mutate]
                
                if config["type"] == "float":
                    child[param_to_mutate] = np.random.uniform(config["low"], config["high"])
                elif config["type"] == "int":
                    child[param_to_mutate] = np.random.randint(config["low"], config["high"] + 1)
                elif config["type"] == "categorical":
                    child[param_to_mutate] = np.random.choice(config["choices"])
            
            new_generation.append(child)
        
        return new_generation[:len(population)]


class SearchEngine:
    """搜索引擎"""
    
    def __init__(self, parameter_space: Dict):
        self.parameter_space = self._validate_parameter_space(parameter_space)
        self.algorithms = {
            SearchAlgorithm.RANDOM: RandomSearchAlgorithm(),
            SearchAlgorithm.GRID: GridSearchAlgorithm(),
            SearchAlgorithm.BAYESIAN: BayesianSearchAlgorithm(),
            SearchAlgorithm.EVOLUTIONARY: EvolutionarySearchAlgorithm()
        }
    
    def _validate_parameter_space(self, parameter_space: Dict) -> Dict:
        """验证参数空间"""
        for param_name, config in parameter_space.items():
            if config["type"] not in ["float", "int", "categorical"]:
                raise ValueError(f"不支持的参数类型: {config['type']}")
        return parameter_space
    
    def compute_parameter_importance(self, history: List) -> Dict[str, float]:
        """计算参数重要性"""
        if not history:
            return {}
        
        # 简化实现：基于参数值与目标值的相关性
        param_names = list(history[0]["params"].keys())
        importance = {}
        
        for param_name in param_names:
            values = [h["params"][param_name] for h in history]
            targets = [h["value"] for h in history]
            
            # 对于数值参数，计算相关性
            if isinstance(values[0], (int, float)):
                correlation = abs(np.corrcoef(values, targets)[0, 1])
                importance[param_name] = correlation if not np.isnan(correlation) else 0
            else:
                # 对于分类参数，计算方差比
                unique_values = list(set(values))
                group_variances = []
                for val in unique_values:
                    group_targets = [t for v, t in zip(values, targets) if v == val]
                    if len(group_targets) > 1:
                        group_variances.append(np.var(group_targets))
                
                if group_variances:
                    importance[param_name] = np.mean(group_variances) / (np.var(targets) + 1e-10)
                else:
                    importance[param_name] = 0
        
        # 归一化
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance


# 保留原有的HyperparameterSearchEngine类
class ResourceManager:
    """资源管理器（简化实现）"""
    def __init__(self):
        pass


class HyperparameterSearchEngine:
    """集成的超参数搜索引擎"""
    
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.presets = self._load_presets()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """加载预设配置"""
        
        return {
            "llm_fine_tuning": {
                "parameters": [
                    HyperparameterRange("learning_rate", "float", 1e-5, 1e-3, log=True),
                    HyperparameterRange("batch_size", "categorical", choices=[4, 8, 16, 32]),
                    HyperparameterRange("lora_r", "int", 8, 64, step=8),
                    HyperparameterRange("lora_alpha", "int", 16, 128, step=16),
                    HyperparameterRange("lora_dropout", "float", 0.05, 0.3),
                    HyperparameterRange("warmup_steps", "int", 50, 500, step=50),
                    HyperparameterRange("gradient_accumulation_steps", "categorical", choices=[1, 2, 4, 8])
                ],
                "config": OptimizationConfig(
                    study_name="llm_fine_tuning",
                    algorithm=OptimizationAlgorithm.TPE,
                    pruning=PruningAlgorithm.HYPERBAND,
                    direction="maximize",
                    n_trials=100,
                    early_stopping=True,
                    patience=15
                )
            },
            
            "model_compression": {
                "parameters": [
                    HyperparameterRange("quantization_bits", "categorical", choices=[4, 8, 16]),
                    HyperparameterRange("calibration_samples", "int", 100, 1000, step=100),
                    HyperparameterRange("compression_ratio", "float", 0.1, 0.9),
                    HyperparameterRange("distillation_temperature", "float", 3.0, 10.0),
                    HyperparameterRange("distillation_alpha", "float", 0.1, 0.9)
                ],
                "config": OptimizationConfig(
                    study_name="model_compression",
                    algorithm=OptimizationAlgorithm.TPE,
                    pruning=PruningAlgorithm.MEDIAN,
                    direction="maximize",
                    n_trials=50,
                    early_stopping=True,
                    patience=10
                )
            },
            
            "neural_architecture_search": {
                "parameters": [
                    HyperparameterRange("num_layers", "int", 2, 20),
                    HyperparameterRange("hidden_size", "categorical", choices=[64, 128, 256, 512, 1024]),
                    HyperparameterRange("activation", "categorical", choices=["relu", "gelu", "swish", "leaky_relu"]),
                    HyperparameterRange("dropout_rate", "float", 0.0, 0.5),
                    HyperparameterRange("use_batch_norm", "boolean"),
                    HyperparameterRange("optimizer_type", "categorical", choices=["adam", "adamw", "sgd", "rmsprop"])
                ],
                "config": OptimizationConfig(
                    study_name="neural_architecture_search",
                    algorithm=OptimizationAlgorithm.TPE,
                    pruning=PruningAlgorithm.SUCCESSIVE_HALVING,
                    direction="maximize",
                    n_trials=200,
                    early_stopping=True,
                    patience=25
                )
            },
            
            "classical_ml": {
                "parameters": [
                    HyperparameterRange("model_type", "categorical", choices=["svm", "random_forest", "xgboost", "lightgbm"]),
                    HyperparameterRange("svm_c", "float", 1e-3, 1e3, log=True),
                    HyperparameterRange("svm_gamma", "float", 1e-4, 1e1, log=True),
                    HyperparameterRange("rf_n_estimators", "int", 10, 1000),
                    HyperparameterRange("rf_max_depth", "int", 3, 50),
                    HyperparameterRange("xgb_learning_rate", "float", 0.01, 0.3, log=True),
                    HyperparameterRange("xgb_n_estimators", "int", 50, 1000),
                    HyperparameterRange("xgb_max_depth", "int", 3, 15)
                ],
                "config": OptimizationConfig(
                    study_name="classical_ml",
                    algorithm=OptimizationAlgorithm.TPE,
                    pruning=PruningAlgorithm.MEDIAN,
                    direction="maximize",
                    n_trials=150,
                    early_stopping=True,
                    patience=20
                )
            },
            
            "reinforcement_learning": {
                "parameters": [
                    HyperparameterRange("learning_rate", "float", 1e-5, 1e-2, log=True),
                    HyperparameterRange("batch_size", "categorical", choices=[32, 64, 128, 256]),
                    HyperparameterRange("buffer_size", "int", 10000, 1000000, log=True),
                    HyperparameterRange("discount_factor", "float", 0.9, 0.999),
                    HyperparameterRange("exploration_rate", "float", 0.01, 1.0),
                    HyperparameterRange("target_update_freq", "int", 100, 10000),
                    HyperparameterRange("network_architecture", "categorical", choices=["mlp", "cnn", "lstm"])
                ],
                "config": OptimizationConfig(
                    study_name="reinforcement_learning",
                    algorithm=OptimizationAlgorithm.TPE,
                    pruning=PruningAlgorithm.HYPERBAND,
                    direction="maximize",
                    n_trials=300,
                    early_stopping=True,
                    patience=30
                )
            }
        }
    
    def optimize_for_task(
        self, 
        task_name: str, 
        objective_function: Callable[[Dict[str, Any]], float],
        custom_config: Optional[OptimizationConfig] = None,
        custom_parameters: Optional[List[HyperparameterRange]] = None
    ) -> Dict[str, Any]:
        """为特定任务执行优化"""
        
        if task_name in self.presets:
            # 使用预设配置
            preset = self.presets[task_name]
            config = custom_config or preset["config"]
            parameters = custom_parameters or preset["parameters"]
        else:
            if not custom_config or not custom_parameters:
                raise ValueError(f"Unknown task {task_name}. Please provide custom config and parameters.")
            config = custom_config
            parameters = custom_parameters
        
        # 创建优化任务
        study_name = f"{task_name}_{int(utc_now().timestamp())}"
        config.study_name = study_name
        
        # 检查资源
        if not self.resource_manager.can_start_trial():
            raise RuntimeError("Insufficient resources to start optimization")
        
        # 注册资源使用
        self.resource_manager.register_trial(study_name, {"gpu": 1.0, "cpu": 4.0})
        
        try:
            # 创建优化器
            optimizer = HyperparameterOptimizer(config)
            for param_range in parameters:
                optimizer.add_parameter_range(param_range)
            
            # 启动优化
            result = optimizer.optimize(objective_function)
            
            # 创建可视化
            visualizations = optimizer.create_visualizations(
                save_path=f"./optimization_results/{study_name}"
            )
            
            result["visualizations"] = list(visualizations.keys())
            
            self.logger.info(f"Optimization completed for task: {task_name}")
            
            return result
        
        finally:
            # 注销资源
            self.resource_manager.unregister_trial(study_name)
    
    def compare_algorithms(
        self, 
        task_name: str,
        objective_function: Callable[[Dict[str, Any]], float],
        algorithms: List[OptimizationAlgorithm] = None
    ) -> Dict[str, Dict[str, Any]]:
        """比较不同算法效果"""
        
        if algorithms is None:
            algorithms = [
                OptimizationAlgorithm.TPE,
                OptimizationAlgorithm.CMAES,
                OptimizationAlgorithm.RANDOM
            ]
        
        results = {}
        
        for algorithm in algorithms:
            self.logger.info(f"Testing algorithm: {algorithm.value}")
            
            # 创建配置
            config = OptimizationConfig(
                study_name=f"{task_name}_{algorithm.value}",
                algorithm=algorithm,
                pruning=PruningAlgorithm.HYPERBAND,
                direction="maximize",
                n_trials=50
            )
            
            # 获取参数
            if task_name in self.presets:
                parameters = self.presets[task_name]["parameters"]
            else:
                raise ValueError(f"Unknown task {task_name}. Cannot compare algorithms without preset parameters.")
            
            # 执行优化
            result = self.optimize_for_task(
                f"{task_name}_{algorithm.value}",
                objective_function,
                config,
                parameters
            )
            
            results[algorithm.value] = result
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(results)
        
        return {
            "results": results,
            "comparison": comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成算法对比报告"""
        
        report = {
            "best_algorithm": None,
            "best_value": None,
            "algorithm_ranking": [],
            "performance_summary": {}
        }
        
        # 统计各算法表现
        for algorithm, result in results.items():
            best_value = result.get("best_value")
            stats = result.get("stats", {})
            
            if best_value is not None:
                if report["best_value"] is None or best_value > report["best_value"]:
                    report["best_algorithm"] = algorithm
                    report["best_value"] = best_value
                
                report["algorithm_ranking"].append({
                    "algorithm": algorithm,
                    "best_value": best_value,
                    "total_trials": stats.get("total_trials", 0),
                    "successful_trials": stats.get("successful_trials", 0),
                    "success_rate": stats.get("successful_trials", 0) / max(stats.get("total_trials", 1), 1)
                })
        
        # 排序
        report["algorithm_ranking"].sort(key=lambda x: x["best_value"], reverse=True)
        
        # 生成性能摘要
        if report["algorithm_ranking"]:
            best = report["algorithm_ranking"][0]
            worst = report["algorithm_ranking"][-1]
            
            report["performance_summary"] = {
                "best_algorithm": best["algorithm"],
                "worst_algorithm": worst["algorithm"],
                "improvement": best["best_value"] - worst["best_value"] if worst["best_value"] else 0,
                "avg_success_rate": sum(alg["success_rate"] for alg in report["algorithm_ranking"]) / len(report["algorithm_ranking"]),
                "total_algorithms_tested": len(report["algorithm_ranking"])
            }
        
        return report
    
    def get_preset_tasks(self) -> List[str]:
        """获取预设任务列表"""
        return list(self.presets.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """获取任务信息"""
        
        if task_name not in self.presets:
            raise ValueError(f"Unknown task: {task_name}")
        
        preset = self.presets[task_name]
        
        return {
            "task_name": task_name,
            "algorithm": preset["config"].algorithm.value,
            "pruning": preset["config"].pruning.value,
            "n_trials": preset["config"].n_trials,
            "direction": preset["config"].direction,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "low": p.low,
                    "high": p.high,
                    "choices": p.choices,
                    "log": p.log
                }
                for p in preset["parameters"]
            ]
        }
    
    def create_custom_task(
        self,
        task_name: str,
        parameters: List[Dict[str, Any]],
        algorithm: str = "tpe",
        pruning: str = "hyperband",
        direction: str = "maximize",
        n_trials: int = 100,
        early_stopping: bool = True,
        patience: int = 20
    ) -> str:
        """创建自定义任务配置"""
        
        # 创建参数范围
        param_ranges = []
        for p in parameters:
            param_range = HyperparameterRange(**p)
            param_ranges.append(param_range)
        
        # 创建配置
        config = OptimizationConfig(
            study_name=task_name,
            algorithm=OptimizationAlgorithm(algorithm),
            pruning=PruningAlgorithm(pruning),
            direction=direction,
            n_trials=n_trials,
            early_stopping=early_stopping,
            patience=patience
        )
        
        # 添加到预设
        self.presets[task_name] = {
            "parameters": param_ranges,
            "config": config
        }
        
        self.logger.info(f"Created custom task: {task_name}")
        
        return task_name
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        return self.resource_manager.get_resource_stats()