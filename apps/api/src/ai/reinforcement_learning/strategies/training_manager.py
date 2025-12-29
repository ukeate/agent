"""
训练管理器实现

负责训练过程的调度、监控和优化
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque
from enum import Enum
import json
import os
from ..environment.simulator import StepResult

from src.core.logging import get_logger
logger = get_logger(__name__)

class TrainingPhase(Enum):
    """训练阶段"""
    WARMUP = "warmup"
    TRAINING = "training"
    EVALUATION = "evaluation"
    FINE_TUNING = "fine_tuning"
    COMPLETED = "completed"

class LearningRateScheduleType(Enum):
    """学习率调度类型"""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_DECAY = "step_decay"
    ADAPTIVE = "adaptive"

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    evaluation_frequency: int = 100
    save_frequency: int = 500
    
    # 学习率调度
    initial_learning_rate: float = 0.001
    learning_rate_schedule: LearningRateScheduleType = LearningRateScheduleType.EXPONENTIAL_DECAY
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 1000
    min_learning_rate: float = 1e-6
    
    # 早停参数
    early_stopping: bool = True
    patience: int = 200
    min_improvement: float = 0.001
    
    # 性能目标
    target_reward: Optional[float] = None
    target_success_rate: Optional[float] = None
    
    # 检查点和日志
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_best_only: bool = True
    
    # 训练阶段配置
    warmup_episodes: int = 100
    evaluation_episodes: int = 50

@dataclass
class TrainingMetrics:
    """训练指标"""
    # 基本指标
    episode: int = 0
    total_steps: int = 0
    training_time: float = 0.0
    current_phase: TrainingPhase = TrainingPhase.WARMUP
    
    # 性能指标
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rates: List[float] = field(default_factory=list)
    
    # 统计指标
    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = float('-inf')
    min_reward: float = float('inf')
    
    # 学习指标
    learning_rates: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    
    # 探索指标
    exploration_rates: List[float] = field(default_factory=list)
    
    # 最佳性能
    best_reward: float = float('-inf')
    best_episode: int = 0
    
    # 早停指标
    no_improvement_count: int = 0
    early_stopped: bool = False

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.initial_lr = config.initial_learning_rate
        self.current_lr = config.initial_learning_rate
        self.step_count = 0
        
        # 自适应调度参数
        self.performance_history = deque(maxlen=100)
        self.lr_adjustment_patience = 50
        self.lr_adjustment_factor = 0.5
        self.no_improvement_count = 0
    
    def get_learning_rate(self, episode: int, performance: Optional[float] = None) -> float:
        """获取当前学习率"""
        self.step_count += 1
        
        if self.config.learning_rate_schedule == LearningRateScheduleType.CONSTANT:
            self.current_lr = self.initial_lr
            
        elif self.config.learning_rate_schedule == LearningRateScheduleType.LINEAR_DECAY:
            decay_progress = min(episode / self.config.lr_decay_steps, 1.0)
            self.current_lr = self.initial_lr * (1.0 - decay_progress)
            
        elif self.config.learning_rate_schedule == LearningRateScheduleType.EXPONENTIAL_DECAY:
            self.current_lr = self.initial_lr * (self.config.lr_decay_rate ** (episode // self.config.lr_decay_steps))
            
        elif self.config.learning_rate_schedule == LearningRateScheduleType.COSINE_ANNEALING:
            progress = min(episode / self.config.max_episodes, 1.0)
            self.current_lr = self.config.min_learning_rate + \
                             (self.initial_lr - self.config.min_learning_rate) * \
                             (1 + np.cos(np.pi * progress)) / 2
                             
        elif self.config.learning_rate_schedule == LearningRateScheduleType.STEP_DECAY:
            decay_factor = self.config.lr_decay_rate ** (episode // self.config.lr_decay_steps)
            self.current_lr = self.initial_lr * decay_factor
            
        elif self.config.learning_rate_schedule == LearningRateScheduleType.ADAPTIVE:
            self.current_lr = self._adaptive_learning_rate(performance)
        
        # 确保不低于最小学习率
        self.current_lr = max(self.current_lr, self.config.min_learning_rate)
        
        return self.current_lr
    
    def _adaptive_learning_rate(self, performance: Optional[float]) -> float:
        """自适应学习率调整"""
        if performance is None:
            return self.current_lr
        
        self.performance_history.append(performance)
        
        if len(self.performance_history) < self.lr_adjustment_patience:
            return self.current_lr
        
        # 检查是否有改进
        recent_performance = np.mean(list(self.performance_history)[-10:])
        earlier_performance = np.mean(list(self.performance_history)[-self.lr_adjustment_patience:-10])
        
        if recent_performance <= earlier_performance + self.config.min_improvement:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        # 如果长时间无改进，降低学习率
        if self.no_improvement_count >= self.lr_adjustment_patience:
            self.current_lr *= self.lr_adjustment_factor
            self.no_improvement_count = 0
        
        return self.current_lr

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 100, min_improvement: float = 0.001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.should_stop = False
    
    def check(self, score: float) -> bool:
        """检查是否应该早停"""
        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        if self.no_improvement_count >= self.patience:
            self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """重置早停状态"""
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.should_stop = False

class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.success_indicators = deque(maxlen=window_size)
        
    def add_episode(self, reward: float, length: int, success: bool = False):
        """添加回合数据"""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.success_indicators.append(success)
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.rewards) == 0:
            return {}
        
        rewards_array = np.array(self.rewards)
        lengths_array = np.array(self.lengths)
        
        stats = {
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'min_reward': np.min(rewards_array),
            'max_reward': np.max(rewards_array),
            'median_reward': np.median(rewards_array),
            'mean_length': np.mean(lengths_array),
            'std_length': np.std(lengths_array),
            'success_rate': np.mean(self.success_indicators) if self.success_indicators else 0.0
        }
        
        return stats

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, config: TrainingConfig, agent, environment):
        self.config = config
        self.agent = agent
        self.environment = environment
        
        # 初始化组件
        self.metrics = TrainingMetrics()
        self.lr_scheduler = LearningRateScheduler(config)
        self.early_stopping = EarlyStopping(config.patience, config.min_improvement) if config.early_stopping else None
        self.performance_tracker = PerformanceTracker()
        
        # 创建目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 训练状态
        self.training_start_time = None
        self.episode_start_time = None
        
        # 回调函数
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加训练回调函数"""
        self.callbacks.append(callback)
    
    def train(self) -> TrainingMetrics:
        """开始训练"""
        self.training_start_time = time.time()
        self.metrics.current_phase = TrainingPhase.WARMUP
        
        try:
            for episode in range(self.config.max_episodes):
                self.metrics.episode = episode
                
                # 更新训练阶段
                self._update_training_phase(episode)
                
                # 执行一个回合
                episode_reward, episode_length, episode_info = self._run_episode(episode)
                
                # 更新指标
                self._update_metrics(episode_reward, episode_length, episode_info)
                
                # 更新学习率
                current_lr = self.lr_scheduler.get_learning_rate(episode, self.metrics.mean_reward)
                self.metrics.learning_rates.append(current_lr)
                
                # 更新智能体学习率（如果支持）
                if hasattr(self.agent, 'set_learning_rate'):
                    self.agent.set_learning_rate(current_lr)
                
                # 执行回调
                self._execute_callbacks(episode)
                
                # 评估和保存
                if episode % self.config.evaluation_frequency == 0:
                    self._evaluate(episode)
                
                if episode % self.config.save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # 检查早停
                if self.early_stopping and self.early_stopping.check(self.metrics.mean_reward):
                    self.metrics.early_stopped = True
                    break
                
                # 检查目标达成
                if self._check_training_complete():
                    break
            
            self.metrics.current_phase = TrainingPhase.COMPLETED
            
        except Exception as e:
            logger.error("训练过程中出现错误", error=str(e), exc_info=True)
            raise
        finally:
            # 保存最终结果
            self._save_final_results()
        
        return self.metrics
    
    def _run_episode(self, episode: int) -> Tuple[float, int, Dict[str, Any]]:
        """运行一个回合"""
        self.episode_start_time = time.time()
        
        state = self.environment.reset()
        total_reward = 0.0
        step_count = 0
        episode_info = {'success': False}
        
        for step in range(self.config.max_steps_per_episode):
            # 选择动作
            action = self.agent.act(state, episode)
            
            # 执行动作
            step_result = self.environment.step(action)
            next_state, reward, done, info = self._unpack_step_result(step_result)
            
            # 训练智能体
            if hasattr(self.agent, 'learn'):
                self.agent.learn(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            total_reward += reward
            step_count += 1
            self.metrics.total_steps += 1
            
            # 更新回合信息
            if info:
                episode_info.update(info)
            
            if done:
                break
        
        # 回合结束处理
        if hasattr(self.agent, 'on_episode_end'):
            self.agent.on_episode_end(total_reward)
        
        episode_info['episode_time'] = time.time() - self.episode_start_time
        
        return total_reward, step_count, episode_info
    
    def _update_metrics(self, reward: float, length: int, episode_info: Dict[str, Any]):
        """更新训练指标"""
        self.metrics.episode_rewards.append(reward)
        self.metrics.episode_lengths.append(length)
        
        # 更新统计指标
        if len(self.metrics.episode_rewards) > 0:
            rewards_array = np.array(self.metrics.episode_rewards[-100:])  # 最近100个回合
            self.metrics.mean_reward = np.mean(rewards_array)
            self.metrics.std_reward = np.std(rewards_array)
            self.metrics.max_reward = max(self.metrics.max_reward, reward)
            self.metrics.min_reward = min(self.metrics.min_reward, reward)
        
        # 更新最佳性能
        if reward > self.metrics.best_reward:
            self.metrics.best_reward = reward
            self.metrics.best_episode = self.metrics.episode
        
        # 更新性能追踪器
        success = episode_info.get('success', False)
        self.performance_tracker.add_episode(reward, length, success)
        
        # 更新总训练时间
        if self.training_start_time:
            self.metrics.training_time = time.time() - self.training_start_time
    
    def _update_training_phase(self, episode: int):
        """更新训练阶段"""
        if episode < self.config.warmup_episodes:
            self.metrics.current_phase = TrainingPhase.WARMUP
        elif episode < self.config.max_episodes - 100:  # 最后100回合用于微调
            self.metrics.current_phase = TrainingPhase.TRAINING
        else:
            self.metrics.current_phase = TrainingPhase.FINE_TUNING
    
    def _evaluate(self, episode: int):
        """评估当前策略"""
        self.metrics.current_phase = TrainingPhase.EVALUATION
        
        eval_rewards = []
        eval_successes = []
        
        for _ in range(self.config.evaluation_episodes):
            state = self.environment.reset()
            total_reward = 0.0
            success = False
            
            for _ in range(self.config.max_steps_per_episode):
                # 评估时使用确定性策略
                action = self.agent.act(state, episode, evaluation=True)
                step_result = self.environment.step(action)
                next_state, reward, done, info = self._unpack_step_result(step_result)
                
                state = next_state
                total_reward += reward
                
                if info and info.get('success', False):
                    success = True
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_successes.append(success)
        
        # 计算评估统计
        eval_mean_reward = np.mean(eval_rewards)
        eval_success_rate = np.mean(eval_successes)
        
        self.metrics.success_rates.append(eval_success_rate)
        
        logger.info(
            "评估结果",
            episode=episode,
            eval_reward=f"{eval_mean_reward:.3f}",
            success_rate=f"{eval_success_rate:.3f}",
        )
    
    def _execute_callbacks(self, episode: int):
        """执行回调函数"""
        callback_data = {
            'episode': episode,
            'metrics': self.metrics,
            'agent': self.agent,
            'performance': self.performance_tracker.get_statistics()
        }
        
        for callback in self.callbacks:
            try:
                callback(callback_data)
            except Exception as e:
                logger.error("回调执行错误", error=str(e), exc_info=True)
    
    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{episode}.json")
        
        checkpoint_data = {
            'episode': episode,
            'metrics': {
                'episode_rewards': self.metrics.episode_rewards[-100:],  # 保存最近100个
                'mean_reward': self.metrics.mean_reward,
                'best_reward': self.metrics.best_reward,
                'best_episode': self.metrics.best_episode,
                'training_time': self.metrics.training_time
            },
            'config': {
                'max_episodes': self.config.max_episodes,
                'initial_learning_rate': self.config.initial_learning_rate,
                'learning_rate_schedule': self.config.learning_rate_schedule.value
            }
        }
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error("保存检查点失败", error=str(e), exc_info=True)
        
        # 保存智能体模型（如果支持）
        if hasattr(self.agent, 'save_model'):
            model_path = os.path.join(self.config.checkpoint_dir, f"agent_model_{episode}")
            self.agent.save_model(model_path)
    
    def _save_final_results(self):
        """保存最终结果"""
        results_path = os.path.join(self.config.log_dir, "training_results.json")
        
        results = {
            'training_completed': True,
            'total_episodes': self.metrics.episode,
            'total_steps': self.metrics.total_steps,
            'training_time': self.metrics.training_time,
            'best_reward': self.metrics.best_reward,
            'best_episode': self.metrics.best_episode,
            'final_mean_reward': self.metrics.mean_reward,
            'early_stopped': self.metrics.early_stopped,
            'performance_statistics': self.performance_tracker.get_statistics()
        }
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.error("保存最终结果失败", error=str(e), exc_info=True)

    def _unpack_step_result(self, step_result):
        if isinstance(step_result, StepResult):
            return step_result.next_state, step_result.reward, step_result.done, step_result.info
        return step_result
    
    def _check_training_complete(self) -> bool:
        """检查训练是否完成"""
        # 检查目标奖励
        if self.config.target_reward is not None:
            if self.metrics.mean_reward >= self.config.target_reward:
                logger.info("达到目标奖励", target_reward=self.config.target_reward)
                return True
        
        # 检查目标成功率
        if self.config.target_success_rate is not None and len(self.metrics.success_rates) > 0:
            recent_success_rate = np.mean(self.metrics.success_rates[-10:])
            if recent_success_rate >= self.config.target_success_rate:
                logger.info("达到目标成功率", target_success_rate=self.config.target_success_rate)
                return True
        
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'episodes_completed': self.metrics.episode,
            'total_steps': self.metrics.total_steps,
            'training_time': self.metrics.training_time,
            'current_phase': self.metrics.current_phase.value,
            'best_performance': {
                'reward': self.metrics.best_reward,
                'episode': self.metrics.best_episode
            },
            'current_performance': {
                'mean_reward': self.metrics.mean_reward,
                'std_reward': self.metrics.std_reward,
                'success_rate': self.metrics.success_rates[-1] if self.metrics.success_rates else 0.0
            },
            'early_stopped': self.metrics.early_stopped,
            'performance_trends': self.performance_tracker.get_statistics()
        }
