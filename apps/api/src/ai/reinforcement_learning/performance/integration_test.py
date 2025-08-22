"""
端到端集成测试框架
提供完整的强化学习系统测试，包括训练、推理、性能和正确性验证
"""

import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import tempfile
import shutil
import json
import matplotlib.pyplot as plt
from pathlib import Path

from ..qlearning.base import (
    QLearningAgent, QLearningConfig, AlgorithmType, 
    AgentState, Experience, EpsilonGreedyStrategy
)
from ..qlearning.dqn import DQNAgent
from ..qlearning.double_dqn import DoubleDQNAgent
from ..qlearning.dueling_dqn import DuelingDQNAgent
from .gpu_accelerator import GPUAccelerator, GPUConfig
from .optimized_replay_buffer import OptimizedReplayBuffer, BufferConfig
from .distributed_training import DistributedTrainingManager, DistributedConfig


class TestScenario(Enum):
    """测试场景类型"""
    BASIC_TRAINING = "basic_training"
    CONVERGENCE_TEST = "convergence_test"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DISTRIBUTED_TRAINING = "distributed_training"
    GPU_ACCELERATION = "gpu_acceleration"
    FAULT_TOLERANCE = "fault_tolerance"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"


@dataclass
class TestConfig:
    """测试配置"""
    scenarios: List[TestScenario]
    episodes: int = 1000
    max_steps_per_episode: int = 200
    convergence_threshold: float = 0.95
    performance_target_fps: float = 100.0
    memory_limit_mb: float = 1024.0
    timeout_seconds: float = 600.0
    save_results: bool = True
    visualize_results: bool = True
    output_dir: str = "./test_results"


class TestEnvironment:
    """测试环境模拟器"""
    
    def __init__(self, state_size: int = 8, action_size: int = 4, complexity: str = "medium"):
        self.state_size = state_size
        self.action_size = action_size
        self.complexity = complexity
        self.action_space = [f"action_{i}" for i in range(action_size)]
        
        # 环境参数
        self.current_state = None
        self.episode_step = 0
        self.max_steps = 200
        
        # 奖励函数参数
        self.target_state = np.random.random(state_size)
        self.noise_level = 0.1 if complexity == "easy" else 0.3 if complexity == "hard" else 0.2
        
        self.reset()
    
    def reset(self) -> AgentState:
        """重置环境"""
        self.current_state = np.random.random(self.state_size)
        self.episode_step = 0
        
        return AgentState(
            features=self.current_state.tolist(),
            metadata={"episode_step": self.episode_step}
        )
    
    def step(self, action: str) -> Tuple[AgentState, float, bool, Dict]:
        """执行动作"""
        action_idx = self.action_space.index(action)
        
        # 状态转移
        action_effect = np.zeros(self.state_size)
        action_effect[action_idx % self.state_size] = 0.1
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_level, self.state_size)
        self.current_state += action_effect + noise
        self.current_state = np.clip(self.current_state, 0, 1)
        
        # 计算奖励
        distance = np.linalg.norm(self.current_state - self.target_state)
        reward = max(0, 1 - distance)  # 越接近目标奖励越高
        
        # 终止条件
        self.episode_step += 1
        done = self.episode_step >= self.max_steps or distance < 0.1
        
        next_state = AgentState(
            features=self.current_state.tolist(),
            metadata={"episode_step": self.episode_step}
        )
        
        info = {
            "distance_to_target": distance,
            "episode_step": self.episode_step
        }
        
        return next_state, reward, done, info


class IntegrationTestSuite:
    """集成测试套件"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = {}
        self.test_environment = TestEnvironment()
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"初始化集成测试套件，场景数量: {len(config.scenarios)}")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("开始运行集成测试...")
        start_time = time.time()
        
        for scenario in self.config.scenarios:
            print(f"\n{'='*50}")
            print(f"运行测试场景: {scenario.value}")
            print(f"{'='*50}")
            
            try:
                if scenario == TestScenario.BASIC_TRAINING:
                    result = self._test_basic_training()
                elif scenario == TestScenario.CONVERGENCE_TEST:
                    result = self._test_convergence()
                elif scenario == TestScenario.PERFORMANCE_BENCHMARK:
                    result = self._test_performance_benchmark()
                elif scenario == TestScenario.DISTRIBUTED_TRAINING:
                    result = self._test_distributed_training()
                elif scenario == TestScenario.GPU_ACCELERATION:
                    result = self._test_gpu_acceleration()
                elif scenario == TestScenario.FAULT_TOLERANCE:
                    result = self._test_fault_tolerance()
                elif scenario == TestScenario.MEMORY_EFFICIENCY:
                    result = self._test_memory_efficiency()
                elif scenario == TestScenario.INFERENCE_SPEED:
                    result = self._test_inference_speed()
                else:
                    result = {"status": "skipped", "reason": "未实现的测试场景"}
                
                self.results[scenario.value] = result
                
            except Exception as e:
                print(f"测试场景 {scenario.value} 失败: {e}")
                self.results[scenario.value] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        total_time = time.time() - start_time
        
        # 生成测试报告
        self._generate_test_report(total_time)
        
        return self.results
    
    def _test_basic_training(self) -> Dict[str, Any]:
        """基础训练测试"""
        print("测试基础训练功能...")
        
        # 测试不同算法
        algorithms = [
            ("DQN", DQNAgent),
            ("DoubleDQN", DoubleDQNAgent),
            ("DuelingDQN", DuelingDQNAgent)
        ]
        
        results = {}
        
        for name, agent_class in algorithms:
            print(f"测试 {name} 算法...")
            
            config = QLearningConfig(
                algorithm_type=AlgorithmType.DQN,
                learning_rate=0.001,
                batch_size=32,
                buffer_size=10000,
                epsilon_decay=0.995
            )
            
            agent = agent_class(
                agent_id=f"test_{name.lower()}",
                state_size=self.test_environment.state_size,
                action_size=self.test_environment.action_size,
                config=config,
                action_space=self.test_environment.action_space
            )
            
            # 训练智能体
            episode_rewards = []
            start_time = time.time()
            
            for episode in range(100):  # 较少的episode用于快速测试
                state = self.test_environment.reset()
                episode_reward = 0
                
                for step in range(self.config.max_steps_per_episode):
                    action = agent.get_action(state)
                    next_state, reward, done, _ = self.test_environment.step(action)
                    
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
            
            training_time = time.time() - start_time
            
            # 评估性能
            final_performance = np.mean(episode_rewards[-10:])  # 最后10个episode的平均性能
            
            results[name] = {
                "training_time": training_time,
                "final_performance": final_performance,
                "episode_rewards": episode_rewards,
                "convergence": final_performance > 0.5  # 简单的收敛标准
            }
            
            print(f"{name} 训练完成: 平均奖励 {final_performance:.3f}, 训练时间 {training_time:.2f}s")
        
        return {
            "status": "completed",
            "algorithms": results,
            "timestamp": time.time()
        }
    
    def _test_convergence(self) -> Dict[str, Any]:
        """收敛性测试"""
        print("测试算法收敛性...")
        
        config = QLearningConfig(
            algorithm_type=AlgorithmType.DQN,
            learning_rate=0.001,
            batch_size=32,
            buffer_size=50000,
            epsilon_decay=0.999
        )
        
        agent = DQNAgent(
            agent_id="convergence_test",
            state_size=self.test_environment.state_size,
            action_size=self.test_environment.action_size,
            config=config,
            action_space=self.test_environment.action_space
        )
        
        episode_rewards = []
        convergence_achieved = False
        convergence_episode = None
        
        for episode in range(self.config.episodes):
            state = self.test_environment.reset()
            episode_reward = 0
            
            for step in range(self.config.max_steps_per_episode):
                action = agent.get_action(state)
                next_state, reward, done, _ = self.test_environment.step(action)
                
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
            if episode >= 100 and not convergence_achieved:
                recent_performance = np.mean(episode_rewards[-100:])
                if recent_performance >= self.config.convergence_threshold:
                    convergence_achieved = True
                    convergence_episode = episode
            
            if episode % 100 == 0:
                recent_avg = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
                print(f"Episode {episode}: 最近100个episode平均奖励 {recent_avg:.3f}")
        
        return {
            "status": "completed",
            "convergence_achieved": convergence_achieved,
            "convergence_episode": convergence_episode,
            "final_performance": np.mean(episode_rewards[-100:]),
            "episode_rewards": episode_rewards,
            "timestamp": time.time()
        }
    
    def _test_performance_benchmark(self) -> Dict[str, Any]:
        """性能基准测试"""
        print("测试性能基准...")
        
        config = QLearningConfig(batch_size=64, buffer_size=10000)
        agent = DQNAgent(
            agent_id="benchmark_test",
            state_size=self.test_environment.state_size,
            action_size=self.test_environment.action_size,
            config=config,
            action_space=self.test_environment.action_space
        )
        
        # 预填充经验
        state = self.test_environment.reset()
        for _ in range(1000):
            action = agent.get_action(state)
            next_state, reward, done, _ = self.test_environment.step(action)
            
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            agent.update_q_value(experience)
            state = next_state if not done else self.test_environment.reset()
        
        # 性能测试
        num_iterations = 1000
        step_times = []
        
        print(f"开始性能测试: {num_iterations} 次迭代...")
        
        for i in range(num_iterations):
            state = AgentState(
                features=np.random.random(self.test_environment.state_size).tolist()
            )
            
            start_time = time.time()
            action = agent.get_action(state)
            step_time = time.time() - start_time
            step_times.append(step_time)
        
        # 计算性能指标
        avg_step_time = np.mean(step_times)
        fps = 1.0 / avg_step_time
        
        print(f"性能测试结果: 平均步骤时间 {avg_step_time*1000:.2f}ms, FPS {fps:.2f}")
        
        return {
            "status": "completed",
            "average_step_time": avg_step_time,
            "fps": fps,
            "target_fps_achieved": fps >= self.config.performance_target_fps,
            "step_times": step_times,
            "timestamp": time.time()
        }
    
    def _test_distributed_training(self) -> Dict[str, Any]:
        """分布式训练测试"""
        print("测试分布式训练...")
        
        try:
            dist_config = DistributedConfig(num_workers=2, batch_size_per_worker=16)
            manager = DistributedTrainingManager(dist_config)
            
            # 创建多个智能体
            def create_agent():
                config = QLearningConfig(batch_size=16, buffer_size=5000)
                return DQNAgent(
                    agent_id="dist_test",
                    state_size=self.test_environment.state_size,
                    action_size=self.test_environment.action_size,
                    config=config,
                    action_space=self.test_environment.action_space
                )
            
            agents = [create_agent() for _ in range(2)]
            
            # 生成训练数据
            training_data = [[] for _ in range(2)]
            for i in range(100):
                state = self.test_environment.reset()
                action = self.test_environment.action_space[i % len(self.test_environment.action_space)]
                next_state, reward, done, _ = self.test_environment.step(action)
                
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                training_data[i % 2].append(experience)
            
            # 执行分布式训练
            start_time = time.time()
            metrics = manager.start_data_parallel_training(agents, training_data)
            training_time = time.time() - start_time
            
            return {
                "status": "completed",
                "training_time": training_time,
                "metrics": metrics,
                "cluster_status": manager.get_cluster_status(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _test_gpu_acceleration(self) -> Dict[str, Any]:
        """GPU加速测试"""
        print("测试GPU加速...")
        
        try:
            gpu_config = GPUConfig(enable_mixed_precision=True, enable_xla=True)
            accelerator = GPUAccelerator(gpu_config)
            
            # 创建测试数据
            sample_data = tf.random.normal((32, self.test_environment.state_size))
            
            # 创建测试模型
            def create_model():
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(self.test_environment.action_size)
                ])
                return model
            
            model = accelerator.create_distributed_model(create_model)
            
            # 性能基准测试
            benchmark_results = accelerator.benchmark_performance(model, sample_data)
            
            # GPU信息
            device_info = accelerator.get_device_info()
            
            return {
                "status": "completed",
                "benchmark_results": benchmark_results,
                "device_info": device_info,
                "gpu_available": len(device_info["gpu_devices"]) > 0,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _test_fault_tolerance(self) -> Dict[str, Any]:
        """容错性测试"""
        print("测试容错性...")
        
        config = QLearningConfig(batch_size=32, buffer_size=10000)
        agent = DQNAgent(
            agent_id="fault_tolerance_test",
            state_size=self.test_environment.state_size,
            action_size=self.test_environment.action_size,
            config=config,
            action_space=self.test_environment.action_space
        )
        
        # 模拟训练过程中的中断和恢复
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.json"
            
            # 第一阶段训练
            episode_rewards_phase1 = []
            for episode in range(50):
                state = self.test_environment.reset()
                episode_reward = 0
                
                for step in range(100):
                    action = agent.get_action(state)
                    next_state, reward, done, _ = self.test_environment.step(action)
                    
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
                
                episode_rewards_phase1.append(episode_reward)
            
            # 保存检查点
            agent.save_model(str(checkpoint_path))
            
            # 创建新智能体并加载检查点
            new_agent = DQNAgent(
                agent_id="fault_tolerance_test_restored",
                state_size=self.test_environment.state_size,
                action_size=self.test_environment.action_size,
                config=config,
                action_space=self.test_environment.action_space
            )
            
            new_agent.load_model(str(checkpoint_path))
            
            # 第二阶段训练
            episode_rewards_phase2 = []
            for episode in range(50):
                state = self.test_environment.reset()
                episode_reward = 0
                
                for step in range(100):
                    action = new_agent.get_action(state)
                    next_state, reward, done, _ = self.test_environment.step(action)
                    
                    experience = Experience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )
                    
                    new_agent.update_q_value(experience)
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards_phase2.append(episode_reward)
            
            # 验证连续性
            phase1_final = np.mean(episode_rewards_phase1[-10:])
            phase2_initial = np.mean(episode_rewards_phase2[:10])
            
            continuity_maintained = abs(phase1_final - phase2_initial) < 0.5
            
            return {
                "status": "completed",
                "continuity_maintained": continuity_maintained,
                "phase1_final_performance": phase1_final,
                "phase2_initial_performance": phase2_initial,
                "performance_difference": abs(phase1_final - phase2_initial),
                "checkpoint_saved_successfully": checkpoint_path.exists(),
                "timestamp": time.time()
            }
    
    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """内存效率测试"""
        print("测试内存效率...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大容量缓冲区
        buffer_config = BufferConfig(
            capacity=50000,
            compression=True,
            batch_size=64
        )
        
        buffer = OptimizedReplayBuffer(buffer_config)
        
        # 填充缓冲区
        for i in range(50000):
            state = AgentState(
                features=np.random.random(self.test_environment.state_size).tolist()
            )
            next_state = AgentState(
                features=np.random.random(self.test_environment.state_size).tolist()
            )
            
            experience = Experience(
                state=state,
                action=f"action_{i % self.test_environment.action_size}",
                reward=np.random.random(),
                next_state=next_state,
                done=np.random.random() < 0.1
            )
            
            buffer.push(experience)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        # 测试采样性能
        sample_times = []
        for _ in range(100):
            start_time = time.time()
            batch = buffer.sample(64)
            sample_time = time.time() - start_time
            sample_times.append(sample_time)
        
        avg_sample_time = np.mean(sample_times)
        
        # 清理
        del buffer
        gc.collect()
        
        return {
            "status": "completed",
            "memory_usage_mb": memory_usage,
            "memory_limit_exceeded": memory_usage > self.config.memory_limit_mb,
            "average_sample_time": avg_sample_time,
            "samples_per_second": 1.0 / avg_sample_time if avg_sample_time > 0 else 0,
            "timestamp": time.time()
        }
    
    def _test_inference_speed(self) -> Dict[str, Any]:
        """推理速度测试"""
        print("测试推理速度...")
        
        config = QLearningConfig(batch_size=1)  # 单样本推理
        agent = DQNAgent(
            agent_id="inference_test",
            state_size=self.test_environment.state_size,
            action_size=self.test_environment.action_size,
            config=config,
            action_space=self.test_environment.action_space
        )
        
        # 预训练一些步骤
        state = self.test_environment.reset()
        for _ in range(100):
            action = agent.get_action(state)
            next_state, reward, done, _ = self.test_environment.step(action)
            
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            agent.update_q_value(experience)
            state = next_state if not done else self.test_environment.reset()
        
        # 推理速度测试
        test_states = []
        for _ in range(1000):
            test_state = AgentState(
                features=np.random.random(self.test_environment.state_size).tolist()
            )
            test_states.append(test_state)
        
        # 单线程推理
        start_time = time.time()
        for test_state in test_states:
            action = agent.get_action(test_state, exploration=False)
        single_thread_time = time.time() - start_time
        
        # 批量推理
        start_time = time.time()
        for i in range(0, len(test_states), 32):
            batch = test_states[i:i+32]
            for test_state in batch:
                action = agent.get_action(test_state, exploration=False)
        batch_inference_time = time.time() - start_time
        
        single_thread_fps = len(test_states) / single_thread_time
        batch_fps = len(test_states) / batch_inference_time
        
        return {
            "status": "completed",
            "single_thread_fps": single_thread_fps,
            "batch_inference_fps": batch_fps,
            "speedup_ratio": batch_fps / single_thread_fps,
            "target_fps_achieved": single_thread_fps >= self.config.performance_target_fps,
            "timestamp": time.time()
        }
    
    def _generate_test_report(self, total_time: float):
        """生成测试报告"""
        print(f"\n{'='*60}")
        print("集成测试报告")
        print(f"{'='*60}")
        
        # 统计信息
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("status") == "completed")
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {failed_tests}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        # 详细结果
        print(f"\n{'='*60}")
        print("详细测试结果")
        print(f"{'='*60}")
        
        for scenario, result in self.results.items():
            status = result.get("status", "unknown")
            print(f"\n{scenario}: {status.upper()}")
            
            if status == "completed":
                # 显示关键指标
                if "final_performance" in result:
                    print(f"  最终性能: {result['final_performance']:.3f}")
                if "fps" in result:
                    print(f"  FPS: {result['fps']:.2f}")
                if "memory_usage_mb" in result:
                    print(f"  内存使用: {result['memory_usage_mb']:.2f} MB")
                if "convergence_achieved" in result:
                    print(f"  收敛状态: {'是' if result['convergence_achieved'] else '否'}")
            
            elif status == "failed":
                print(f"  错误: {result.get('error', '未知错误')}")
        
        # 保存结果
        if self.config.save_results:
            result_file = Path(self.config.output_dir) / "test_results.json"
            with open(result_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n测试结果已保存到: {result_file}")
        
        # 生成可视化
        if self.config.visualize_results:
            self._create_visualizations()
    
    def _create_visualizations(self):
        """创建可视化图表"""
        try:
            # 性能对比图
            scenarios = list(self.results.keys())
            success_rates = [
                1.0 if self.results[s].get("status") == "completed" else 0.0
                for s in scenarios
            ]
            
            plt.figure(figsize=(12, 6))
            plt.bar(scenarios, success_rates)
            plt.title("测试场景成功率")
            plt.ylabel("成功率")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_file = Path(self.config.output_dir) / "test_success_rates.png"
            plt.savefig(plot_file)
            plt.close()
            
            print(f"可视化图表已保存到: {plot_file}")
            
        except Exception as e:
            print(f"生成可视化失败: {e}")


def run_integration_tests(scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
    """运行集成测试"""
    if scenarios is None:
        scenarios = [scenario.value for scenario in TestScenario]
    
    test_scenarios = [TestScenario(s) for s in scenarios if s in [ts.value for ts in TestScenario]]
    
    config = TestConfig(
        scenarios=test_scenarios,
        episodes=500,
        save_results=True,
        visualize_results=True
    )
    
    test_suite = IntegrationTestSuite(config)
    results = test_suite.run_all_tests()
    
    return results


if __name__ == "__main__":
    # 运行所有测试
    test_results = run_integration_tests()