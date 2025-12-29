"""
性能基准测试和参数优化模块
提供自动化的超参数调优、性能基准测试和模型优化建议
"""

import time
import numpy as np
from src.core.tensorflow_config import tensorflow_lazy
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
import json
from pathlib import Path
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..qlearning.base import QLearningConfig, AlgorithmType
from ..qlearning.dqn import DQNAgent
from ..qlearning.double_dqn import DoubleDQNAgent
from ..qlearning.dueling_dqn import DuelingDQNAgent
from .gpu_accelerator import GPUAccelerator, GPUConfig
from .optimized_replay_buffer import OptimizedReplayBuffer, BufferConfig
from .integration_test import TestEnvironment
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class OptimizationTarget(Enum):
    """优化目标类型"""
    CONVERGENCE_SPEED = "convergence_speed"
    FINAL_PERFORMANCE = "final_performance"
    SAMPLE_EFFICIENCY = "sample_efficiency"
    TRAINING_SPEED = "training_speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    STABILITY = "stability"

class BenchmarkMetric(Enum):
    """基准测试指标"""
    EPISODE_REWARD = "episode_reward"
    CONVERGENCE_TIME = "convergence_time"
    TRAINING_FPS = "training_fps"
    INFERENCE_FPS = "inference_fps"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    SAMPLE_EFFICIENCY = "sample_efficiency"

@dataclass
class HyperparameterSpace:
    """超参数搜索空间"""
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)
    batch_size: List[int] = None
    buffer_size: List[int] = None
    epsilon_decay: Tuple[float, float] = (0.99, 0.999)
    discount_factor: Tuple[float, float] = (0.9, 0.999)
    target_update_frequency: List[int] = None
    network_architecture: List[List[int]] = None
    
    def __post_init__(self):
        if self.batch_size is None:
            self.batch_size = [16, 32, 64, 128]
        if self.buffer_size is None:
            self.buffer_size = [10000, 50000, 100000]
        if self.target_update_frequency is None:
            self.target_update_frequency = [100, 500, 1000]
        if self.network_architecture is None:
            self.network_architecture = [
                [64, 64],
                [128, 128],
                [128, 128, 64],
                [256, 128, 64]
            ]

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    episodes_per_run: int = 1000
    num_seeds: int = 5
    timeout_minutes: int = 30
    convergence_threshold: float = 0.8
    stability_window: int = 100
    parallel_runs: int = 2
    save_detailed_logs: bool = True
    output_dir: str = "./benchmark_results"

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 hyperparameter_space: HyperparameterSpace,
                 optimization_target: OptimizationTarget,
                 benchmark_config: BenchmarkConfig):
        self.hyperparameter_space = hyperparameter_space
        self.optimization_target = optimization_target
        self.benchmark_config = benchmark_config
        
        # 创建输出目录
        Path(benchmark_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化优化器
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        logger.info("初始化超参数优化器", target=optimization_target.value)
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """执行超参数优化"""
        logger.info("开始超参数优化", n_trials=n_trials)
        
        # 创建Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=f"qlearning_optimization_{self.optimization_target.value}"
        )
        
        # 运行优化
        self.study.optimize(
            self._objective_function,
            n_trials=n_trials,
            timeout=self.benchmark_config.timeout_minutes * 60
        )
        
        # 获取最佳参数
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info("优化完成", best_value=round(best_value, 4))
        logger.info("最佳参数", params=self.best_params)
        
        # 保存优化结果
        self._save_optimization_results()
        
        return {
            "best_params": self.best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
            "optimization_history": self.optimization_history
        }
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """优化目标函数"""
        # 采样超参数
        params = self._sample_hyperparameters(trial)
        
        # 运行基准测试
        benchmark_results = self._run_benchmark_with_params(params)
        
        # 计算目标值
        objective_value = self._calculate_objective_value(benchmark_results)
        
        # 记录历史
        self.optimization_history.append({
            "trial_number": trial.number,
            "params": params,
            "objective_value": objective_value,
            "benchmark_results": benchmark_results
        })
        
        return objective_value
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """采样超参数"""
        params = {}
        
        # 学习率
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            *self.hyperparameter_space.learning_rate,
            log=True
        )
        
        # 批次大小
        params["batch_size"] = trial.suggest_categorical(
            "batch_size",
            self.hyperparameter_space.batch_size
        )
        
        # 缓冲区大小
        params["buffer_size"] = trial.suggest_categorical(
            "buffer_size",
            self.hyperparameter_space.buffer_size
        )
        
        # Epsilon衰减
        params["epsilon_decay"] = trial.suggest_float(
            "epsilon_decay",
            *self.hyperparameter_space.epsilon_decay
        )
        
        # 折扣因子
        params["discount_factor"] = trial.suggest_float(
            "discount_factor",
            *self.hyperparameter_space.discount_factor
        )
        
        # 目标网络更新频率
        params["target_update_frequency"] = trial.suggest_categorical(
            "target_update_frequency",
            self.hyperparameter_space.target_update_frequency
        )
        
        # 网络架构
        architecture_idx = trial.suggest_int(
            "architecture_idx",
            0,
            len(self.hyperparameter_space.network_architecture) - 1
        )
        params["network_architecture"] = self.hyperparameter_space.network_architecture[architecture_idx]
        
        return params
    
    def _run_benchmark_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定参数运行基准测试"""
        # 创建配置
        config = QLearningConfig(
            algorithm_type=AlgorithmType.DQN,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            buffer_size=params["buffer_size"],
            epsilon_decay=params["epsilon_decay"],
            discount_factor=params["discount_factor"],
            target_update_frequency=params["target_update_frequency"],
            network_architecture={
                "hidden_layers": params["network_architecture"],
                "activation": "relu",
                "optimizer": "adam"
            }
        )
        
        # 运行多次实验
        all_results = []
        
        for seed in range(self.benchmark_config.num_seeds):
            np.random.seed(seed)
            tensorflow_lazy.tf.random.set_seed(seed)
            
            result = self._single_benchmark_run(config, seed)
            all_results.append(result)
        
        # 聚合结果
        return self._aggregate_benchmark_results(all_results)
    
    def _single_benchmark_run(self, config: QLearningConfig, seed: int) -> Dict[str, Any]:
        """单次基准测试运行"""
        # 创建环境和智能体
        env = TestEnvironment()
        agent = DQNAgent(
            agent_id=f"benchmark_agent_{seed}",
            state_size=env.state_size,
            action_size=env.action_size,
            config=config,
            action_space=env.action_space
        )
        
        # 训练智能体
        episode_rewards = []
        episode_losses = []
        convergence_episode = None
        
        start_time = time.time()
        
        for episode in range(self.benchmark_config.episodes_per_run):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            
            for step in range(200):  # 最大步数
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # 更新智能体
                from ..qlearning.base import Experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                loss = agent.update_q_value(experience)
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            if loss_count > 0:
                episode_losses.append(episode_loss / loss_count)
            
            # 检查收敛
            if episode >= 100 and convergence_episode is None:
                recent_avg = np.mean(episode_rewards[-100:])
                if recent_avg >= self.benchmark_config.convergence_threshold:
                    convergence_episode = episode
        
        training_time = time.time() - start_time
        
        return {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "convergence_episode": convergence_episode,
            "training_time": training_time,
            "final_performance": np.mean(episode_rewards[-100:]),
            "seed": seed
        }
    
    def _aggregate_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合基准测试结果"""
        aggregated = {
            "mean_final_performance": np.mean([r["final_performance"] for r in results]),
            "std_final_performance": np.std([r["final_performance"] for r in results]),
            "mean_training_time": np.mean([r["training_time"] for r in results]),
            "convergence_rate": np.mean([1.0 if r["convergence_episode"] is not None else 0.0 for r in results]),
            "mean_convergence_episode": np.mean([r["convergence_episode"] for r in results if r["convergence_episode"] is not None]),
            "stability": self._calculate_stability(results),
            "sample_efficiency": self._calculate_sample_efficiency(results)
        }
        
        return aggregated
    
    def _calculate_stability(self, results: List[Dict[str, Any]]) -> float:
        """计算训练稳定性"""
        final_performances = [r["final_performance"] for r in results]
        if len(final_performances) <= 1:
            return 0.0
        
        cv = np.std(final_performances) / np.mean(final_performances)  # 变异系数
        return max(0, 1 - cv)  # 稳定性评分
    
    def _calculate_sample_efficiency(self, results: List[Dict[str, Any]]) -> float:
        """计算样本效率"""
        convergence_episodes = [r["convergence_episode"] for r in results if r["convergence_episode"] is not None]
        if not convergence_episodes:
            return 0.0
        
        # 样本效率 = 1 / 平均收敛所需episode数
        return 1.0 / np.mean(convergence_episodes)
    
    def _calculate_objective_value(self, benchmark_results: Dict[str, Any]) -> float:
        """计算目标值"""
        if self.optimization_target == OptimizationTarget.CONVERGENCE_SPEED:
            return benchmark_results.get("sample_efficiency", 0)
        
        elif self.optimization_target == OptimizationTarget.FINAL_PERFORMANCE:
            return benchmark_results.get("mean_final_performance", 0)
        
        elif self.optimization_target == OptimizationTarget.SAMPLE_EFFICIENCY:
            return benchmark_results.get("sample_efficiency", 0)
        
        elif self.optimization_target == OptimizationTarget.TRAINING_SPEED:
            training_time = benchmark_results.get("mean_training_time", float('inf'))
            return 1.0 / training_time if training_time > 0 else 0
        
        elif self.optimization_target == OptimizationTarget.STABILITY:
            return benchmark_results.get("stability", 0)
        
        else:
            # 综合评分
            performance = benchmark_results.get("mean_final_performance", 0)
            stability = benchmark_results.get("stability", 0)
            efficiency = benchmark_results.get("sample_efficiency", 0)
            
            return 0.4 * performance + 0.3 * stability + 0.3 * efficiency
    
    def _save_optimization_results(self):
        """保存优化结果"""
        results = {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "optimization_target": self.optimization_target.value,
            "n_trials": len(self.study.trials),
            "optimization_history": self.optimization_history
        }
        
        # 保存JSON结果
        result_file = Path(self.benchmark_config.output_dir) / "optimization_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 生成可视化
        self._create_optimization_plots()
        
        logger.info("优化结果已保存", path=str(result_file))
    
    def _create_optimization_plots(self):
        """创建优化过程可视化"""
        if not self.optimization_history:
            return
        
        # 优化进度图
        trial_numbers = [h["trial_number"] for h in self.optimization_history]
        objective_values = [h["objective_value"] for h in self.optimization_history]
        
        plt.figure(figsize=(12, 8))
        
        # 子图1: 优化进度
        plt.subplot(2, 2, 1)
        plt.plot(trial_numbers, objective_values, 'b-', alpha=0.7)
        plt.plot(trial_numbers, np.cummax(objective_values), 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # 子图2: 参数重要性（如果可用）
        plt.subplot(2, 2, 2)
        if hasattr(self.study, 'best_params'):
            param_importance = optuna.importance.get_param_importances(self.study)
            params = list(param_importance.keys())
            importance = list(param_importance.values())
            
            plt.barh(params, importance)
            plt.xlabel('Importance')
            plt.title('Parameter Importance')
        
        # 子图3: 参数分布
        plt.subplot(2, 2, 3)
        learning_rates = [h["params"]["learning_rate"] for h in self.optimization_history]
        batch_sizes = [h["params"]["batch_size"] for h in self.optimization_history]
        
        plt.scatter(learning_rates, objective_values, c=batch_sizes, cmap='viridis', alpha=0.6)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Objective Value')
        plt.colorbar(label='Batch Size')
        plt.title('Learning Rate vs Performance')
        
        # 子图4: 收敛时间分布
        plt.subplot(2, 2, 4)
        convergence_times = []
        for h in self.optimization_history:
            if "benchmark_results" in h and "mean_convergence_episode" in h["benchmark_results"]:
                convergence_times.append(h["benchmark_results"]["mean_convergence_episode"])
        
        if convergence_times:
            plt.hist(convergence_times, bins=20, alpha=0.7)
            plt.xlabel('Convergence Episode')
            plt.ylabel('Frequency')
            plt.title('Convergence Time Distribution')
        
        plt.tight_layout()
        
        plot_file = Path(self.benchmark_config.output_dir) / "optimization_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("优化可视化已保存", path=str(plot_file))

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_algorithm_comparison(self) -> Dict[str, Any]:
        """运行算法对比基准测试"""
        logger.info("开始算法对比基准测试")
        
        algorithms = [
            ("DQN", DQNAgent),
            ("DoubleDQN", DoubleDQNAgent),
            ("DuelingDQN", DuelingDQNAgent)
        ]
        
        algorithm_results = {}
        
        for name, agent_class in algorithms:
            logger.info("测试算法", name=name)
            
            config = QLearningConfig(
                algorithm_type=AlgorithmType.DQN,
                learning_rate=0.001,
                batch_size=64,
                buffer_size=50000,
                epsilon_decay=0.995
            )
            
            # 运行多次实验
            results = []
            for seed in range(self.config.num_seeds):
                result = self._run_single_algorithm_test(agent_class, config, seed)
                results.append(result)
            
            # 聚合结果
            algorithm_results[name] = self._aggregate_algorithm_results(results)
        
        self.results["algorithm_comparison"] = algorithm_results
        return algorithm_results
    
    def run_scalability_benchmark(self) -> Dict[str, Any]:
        """运行可扩展性基准测试"""
        logger.info("开始可扩展性基准测试")
        
        # 测试不同的状态空间大小
        state_sizes = [4, 8, 16, 32, 64]
        scalability_results = {}
        
        for state_size in state_sizes:
            logger.info("测试状态空间大小", state_size=state_size)
            
            env = TestEnvironment(state_size=state_size, action_size=4)
            config = QLearningConfig(batch_size=32, buffer_size=10000)
            
            agent = DQNAgent(
                agent_id=f"scalability_test_{state_size}",
                state_size=state_size,
                action_size=4,
                config=config,
                action_space=env.action_space
            )
            
            # 测试训练速度
            start_time = time.time()
            
            for episode in range(100):  # 较少的episode用于快速测试
                state = env.reset()
                
                for step in range(50):
                    action = agent.get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    from ..qlearning.base import Experience
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )
                    
                    agent.update_q_value(experience)
                    state = next_state
                    
                    if done:
                        break
            
            training_time = time.time() - start_time
            
            # 测试推理速度
            test_state = env.reset()
            inference_times = []
            
            for _ in range(1000):
                start_time = time.time()
                action = agent.get_action(test_state, exploration=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            scalability_results[state_size] = {
                "training_time": training_time,
                "inference_fps": 1.0 / np.mean(inference_times),
                "memory_usage": self._estimate_memory_usage(agent)
            }
        
        self.results["scalability"] = scalability_results
        return scalability_results
    
    def run_gpu_benchmark(self) -> Dict[str, Any]:
        """运行GPU性能基准测试"""
        logger.info("开始GPU性能基准测试")
        
        try:
            gpu_config = GPUConfig(
                enable_mixed_precision=True,
                enable_xla=True,
                batch_size_multiplier=4
            )
            
            accelerator = GPUAccelerator(gpu_config)
            
            # 创建测试模型
            def create_test_model():
                return tensorflow_lazy.tf.keras.Sequential([
                    tensorflow_lazy.tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
                    tensorflow_lazy.tf.keras.layers.Dense(64, activation='relu'),
                    tensorflow_lazy.tf.keras.layers.Dense(4)
                ])
            
            model = accelerator.create_distributed_model(create_test_model)
            
            # 性能基准测试
            test_data = tensorflow_lazy.tf.random.normal((1000, 8))
            benchmark_results = accelerator.benchmark_performance(model, test_data)
            
            # 设备信息
            device_info = accelerator.get_device_info()
            
            gpu_results = {
                "benchmark_results": benchmark_results,
                "device_info": device_info,
                "gpu_available": len(device_info["gpu_devices"]) > 0
            }
            
            self.results["gpu_benchmark"] = gpu_results
            return gpu_results
            
        except Exception as e:
            logger.exception("GPU基准测试失败")
            return {"error": str(e)}
    
    def _run_single_algorithm_test(self, agent_class, config: QLearningConfig, seed: int) -> Dict[str, Any]:
        """运行单个算法测试"""
        np.random.seed(seed)
        tensorflow_lazy.tf.random.set_seed(seed)
        
        env = TestEnvironment()
        agent = agent_class(
            agent_id=f"benchmark_{seed}",
            state_size=env.state_size,
            action_size=env.action_size,
            config=config,
            action_space=env.action_space
        )
        
        episode_rewards = []
        convergence_episode = None
        
        start_time = time.time()
        
        for episode in range(self.config.episodes_per_run):
            state = env.reset()
            episode_reward = 0
            
            for step in range(200):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                from ..qlearning.base import Experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                agent.update_q_value(experience)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 检查收敛
            if episode >= 100 and convergence_episode is None:
                recent_avg = np.mean(episode_rewards[-100:])
                if recent_avg >= self.config.convergence_threshold:
                    convergence_episode = episode
        
        training_time = time.time() - start_time
        
        return {
            "episode_rewards": episode_rewards,
            "convergence_episode": convergence_episode,
            "training_time": training_time,
            "final_performance": np.mean(episode_rewards[-100:]),
            "seed": seed
        }
    
    def _aggregate_algorithm_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合算法结果"""
        return {
            "mean_final_performance": np.mean([r["final_performance"] for r in results]),
            "std_final_performance": np.std([r["final_performance"] for r in results]),
            "mean_training_time": np.mean([r["training_time"] for r in results]),
            "convergence_rate": np.mean([1.0 if r["convergence_episode"] is not None else 0.0 for r in results]),
            "mean_convergence_episode": np.mean([r["convergence_episode"] for r in results if r["convergence_episode"] is not None])
        }
    
    def _estimate_memory_usage(self, agent) -> float:
        """估算内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def generate_benchmark_report(self):
        """生成基准测试报告"""
        logger.info("性能基准测试报告")
        logger.info("报告分隔线", line="=" * 60)
        
        # 算法对比
        if "algorithm_comparison" in self.results:
            logger.info("算法性能对比")
            for algo, results in self.results["algorithm_comparison"].items():
                logger.info(
                    "算法结果",
                    algorithm=algo,
                    mean_final_performance=round(results["mean_final_performance"], 3),
                    std_final_performance=round(results["std_final_performance"], 3),
                    convergence_rate=round(results["convergence_rate"] * 100, 1),
                    mean_training_time=round(results["mean_training_time"], 2),
                )
        
        # 可扩展性
        if "scalability" in self.results:
            logger.info("可扩展性测试")
            for state_size, results in self.results["scalability"].items():
                logger.info(
                    "可扩展性结果",
                    state_size=state_size,
                    inference_fps=round(results["inference_fps"], 2),
                    training_time=round(results["training_time"], 2),
                )
        
        # GPU性能
        if "gpu_benchmark" in self.results:
            gpu_results = self.results["gpu_benchmark"]
            if "error" not in gpu_results:
                logger.info("GPU性能")
                benchmark = gpu_results["benchmark_results"]
                logger.info("吞吐量", samples_per_sec=round(benchmark["throughput"], 2))
                logger.info("内存使用", mb=round(benchmark["memory_usage"], 2))
        
        # 保存报告
        report_file = Path(self.config.output_dir) / "benchmark_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("基准测试报告已保存", path=str(report_file))

def run_hyperparameter_optimization(
    target: str = "final_performance",
    n_trials: int = 50
) -> Dict[str, Any]:
    """运行超参数优化"""
    
    hyperparameter_space = HyperparameterSpace()
    optimization_target = OptimizationTarget(target)
    benchmark_config = BenchmarkConfig(episodes_per_run=500, num_seeds=3)
    
    optimizer = HyperparameterOptimizer(
        hyperparameter_space=hyperparameter_space,
        optimization_target=optimization_target,
        benchmark_config=benchmark_config
    )
    
    results = optimizer.optimize(n_trials=n_trials)
    return results

def run_performance_benchmark() -> Dict[str, Any]:
    """运行性能基准测试"""
    
    config = BenchmarkConfig(
        episodes_per_run=1000,
        num_seeds=5,
        save_detailed_logs=True
    )
    
    benchmark = PerformanceBenchmark(config)
    
    # 运行所有基准测试
    benchmark.run_algorithm_comparison()
    benchmark.run_scalability_benchmark()
    benchmark.run_gpu_benchmark()
    
    # 生成报告
    benchmark.generate_benchmark_report()
    
    return benchmark.results

if __name__ == "__main__":
    # 运行超参数优化
    setup_logging()
    logger.info("开始超参数优化")
    optimization_results = run_hyperparameter_optimization(n_trials=20)
    
    # 运行性能基准测试
    logger.info("开始性能基准测试")
    benchmark_results = run_performance_benchmark()
    
    logger.info("所有测试完成")
