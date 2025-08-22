"""
探索策略实现

实现各种探索策略来平衡探索和利用
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum


class ExplorationMode(Enum):
    """探索模式"""
    EPSILON_GREEDY = "epsilon_greedy"
    DECAYING_EPSILON = "decaying_epsilon"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    BOLTZMANN = "boltzmann"
    NOISE_BASED = "noise_based"
    CURIOSITY_DRIVEN = "curiosity_driven"


@dataclass
class ExplorationConfig:
    """探索策略配置"""
    mode: ExplorationMode
    initial_exploration: float = 1.0
    final_exploration: float = 0.01
    decay_steps: int = 10000
    decay_type: str = "exponential"  # "exponential", "linear", "polynomial"
    
    # UCB参数
    ucb_c: float = 2.0
    
    # Thompson Sampling参数
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    
    # Boltzmann参数
    temperature: float = 1.0
    temperature_decay: float = 0.99
    min_temperature: float = 0.1
    
    # 噪声参数
    noise_type: str = "gaussian"  # "gaussian", "uniform", "ornstein_uhlenbeck"
    noise_scale: float = 0.1
    
    # 好奇心驱动参数
    curiosity_weight: float = 0.1
    novelty_threshold: float = 0.1


class ExplorationStrategy(ABC):
    """探索策略抽象基类"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        self.config = config
        self.action_size = action_size
        self.step_count = 0
        self.exploration_history = []
        
    @abstractmethod
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        选择动作
        
        Args:
            q_values: Q值数组
            step: 当前步数
            
        Returns:
            int: 选择的动作
        """
        pass
    
    @abstractmethod
    def update(self, action: int, reward: float, **kwargs):
        """
        更新策略参数
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            **kwargs: 其他参数
        """
        pass
    
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        """获取当前探索率"""
        if step is None:
            step = self.step_count
        return self.config.initial_exploration
    
    def reset(self):
        """重置策略状态"""
        self.step_count = 0
        self.exploration_history.clear()


class EpsilonGreedyStrategy(ExplorationStrategy):
    """Epsilon-greedy探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.epsilon = config.initial_exploration
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        Epsilon-greedy动作选择
        """
        self.step_count += 1
        
        if np.random.random() < self.epsilon:
            # 随机探索
            action = np.random.randint(self.action_size)
            exploration_type = "explore"
        else:
            # 贪婪选择
            action = np.argmax(q_values)
            exploration_type = "exploit"
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'epsilon': self.epsilon,
            'action': action,
            'q_values': q_values.copy(),
            'type': exploration_type
        })
        
        return action
    
    def update(self, action: int, reward: float, **kwargs):
        """更新epsilon值（固定策略无需更新）"""
        pass
    
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        return self.epsilon


class DecayingEpsilonGreedyStrategy(ExplorationStrategy):
    """衰减Epsilon-greedy策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.epsilon = config.initial_exploration
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        衰减epsilon-greedy动作选择
        """
        self.step_count += 1
        
        # 更新epsilon
        self._update_epsilon()
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_size)
            exploration_type = "explore"
        else:
            action = np.argmax(q_values)
            exploration_type = "exploit"
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'epsilon': self.epsilon,
            'action': action,
            'q_values': q_values.copy(),
            'type': exploration_type
        })
        
        return action
    
    def _update_epsilon(self):
        """更新epsilon值"""
        progress = min(self.step_count / self.config.decay_steps, 1.0)
        
        if self.config.decay_type == "exponential":
            self.epsilon = self.config.final_exploration + \
                          (self.config.initial_exploration - self.config.final_exploration) * \
                          math.exp(-5 * progress)
        elif self.config.decay_type == "linear":
            self.epsilon = self.config.initial_exploration - \
                          (self.config.initial_exploration - self.config.final_exploration) * progress
        elif self.config.decay_type == "polynomial":
            self.epsilon = self.config.final_exploration + \
                          (self.config.initial_exploration - self.config.final_exploration) * \
                          (1 - progress) ** 2
        else:
            # 默认指数衰减
            self.epsilon = self.config.final_exploration + \
                          (self.config.initial_exploration - self.config.final_exploration) * \
                          math.exp(-5 * progress)
        
        self.epsilon = max(self.epsilon, self.config.final_exploration)
    
    def update(self, action: int, reward: float, **kwargs):
        """更新策略（epsilon在select_action中更新）"""
        pass
    
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        return self.epsilon


class UCBStrategy(ExplorationStrategy):
    """Upper Confidence Bound探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.action_counts = np.zeros(action_size)
        self.action_values = np.zeros(action_size)
        self.c = config.ucb_c
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        UCB动作选择
        """
        self.step_count += 1
        
        # 如果某些动作从未被选择，优先选择它们
        if np.any(self.action_counts == 0):
            unvisited = np.where(self.action_counts == 0)[0]
            action = np.random.choice(unvisited)
            exploration_type = "unvisited"
        else:
            # 计算UCB值
            total_counts = np.sum(self.action_counts)
            confidence_bounds = self.c * np.sqrt(np.log(total_counts) / self.action_counts)
            ucb_values = q_values + confidence_bounds
            
            action = np.argmax(ucb_values)
            exploration_type = "ucb"
        
        # 记录选择
        self.action_counts[action] += 1
        
        # 记录探索历史
        confidence_bounds = self.c * np.sqrt(np.log(np.sum(self.action_counts)) / np.maximum(self.action_counts, 1))
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'q_values': q_values.copy(),
            'action_counts': self.action_counts.copy(),
            'confidence_bounds': confidence_bounds,
            'type': exploration_type
        })
        
        return action
    
    def update(self, action: int, reward: float, **kwargs):
        """更新动作值估计"""
        if self.action_counts[action] > 0:
            # 增量更新
            n = self.action_counts[action]
            self.action_values[action] += (reward - self.action_values[action]) / n
    
    def reset(self):
        """重置UCB统计"""
        super().reset()
        self.action_counts = np.zeros(self.action_size)
        self.action_values = np.zeros(self.action_size)


class ThompsonSamplingStrategy(ExplorationStrategy):
    """Thompson Sampling探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        # 每个动作的Beta分布参数
        self.alpha = np.full(action_size, config.prior_alpha)
        self.beta = np.full(action_size, config.prior_beta)
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        Thompson Sampling动作选择
        """
        self.step_count += 1
        
        # 从每个动作的Beta分布中采样
        sampled_rewards = np.random.beta(self.alpha, self.beta)
        
        # 结合Q值和采样奖励
        combined_values = q_values + sampled_rewards
        action = np.argmax(combined_values)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'q_values': q_values.copy(),
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy(),
            'sampled_rewards': sampled_rewards,
            'combined_values': combined_values,
            'type': 'thompson_sampling'
        })
        
        return action
    
    def update(self, action: int, reward: float, **kwargs):
        """更新Beta分布参数"""
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
    
    def reset(self):
        """重置Thompson Sampling参数"""
        super().reset()
        self.alpha = np.full(self.action_size, self.config.prior_alpha)
        self.beta = np.full(self.action_size, self.config.prior_beta)


class BoltzmannExplorationStrategy(ExplorationStrategy):
    """Boltzmann (Softmax)探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.temperature = config.temperature
        self.min_temperature = config.min_temperature
        self.temperature_decay = config.temperature_decay
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        Boltzmann动作选择
        """
        self.step_count += 1
        
        # 计算softmax概率
        scaled_q_values = q_values / max(self.temperature, 1e-8)
        # 数值稳定性：减去最大值
        exp_values = np.exp(scaled_q_values - np.max(scaled_q_values))
        probabilities = exp_values / np.sum(exp_values)
        
        # 根据概率选择动作
        action = np.random.choice(self.action_size, p=probabilities)
        
        # 衰减温度
        self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'q_values': q_values.copy(),
            'temperature': self.temperature,
            'probabilities': probabilities,
            'type': 'boltzmann'
        })
        
        return action
    
    def update(self, action: int, reward: float, **kwargs):
        """更新策略（温度在select_action中衰减）"""
        pass
    
    def get_exploration_rate(self, step: Optional[int] = None) -> float:
        """返回当前温度作为探索率的代理"""
        return self.temperature


class NoiseBasedExplorationStrategy(ExplorationStrategy):
    """基于噪声的探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.noise_scale = config.noise_scale
        self.noise_type = config.noise_type
        
        # Ornstein-Uhlenbeck噪声参数
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_state = np.zeros(action_size)
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None) -> int:
        """
        基于噪声的动作选择
        """
        self.step_count += 1
        
        # 添加噪声到Q值
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_scale, size=q_values.shape)
        elif self.noise_type == "uniform":
            noise = np.random.uniform(-self.noise_scale, self.noise_scale, size=q_values.shape)
        elif self.noise_type == "ornstein_uhlenbeck":
            noise = self._ou_noise()
        else:
            noise = np.random.normal(0, self.noise_scale, size=q_values.shape)
        
        noisy_q_values = q_values + noise
        action = np.argmax(noisy_q_values)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'q_values': q_values.copy(),
            'noise': noise,
            'noisy_q_values': noisy_q_values,
            'noise_scale': self.noise_scale,
            'type': f'noise_{self.noise_type}'
        })
        
        return action
    
    def _ou_noise(self):
        """生成Ornstein-Uhlenbeck噪声"""
        dx = self.ou_theta * (self.ou_mu - self.ou_state) + \
             self.ou_sigma * np.random.normal(size=self.ou_state.shape)
        self.ou_state += dx
        return self.ou_state * self.noise_scale
    
    def update(self, action: int, reward: float, **kwargs):
        """更新噪声参数"""
        # 可以根据性能调整噪声尺度
        pass
    
    def reset(self):
        """重置噪声状态"""
        super().reset()
        self.ou_state = np.zeros(self.action_size)


class CuriosityDrivenExploration(ExplorationStrategy):
    """好奇心驱动的探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int):
        super().__init__(config, action_size)
        self.curiosity_weight = config.curiosity_weight
        self.novelty_threshold = config.novelty_threshold
        
        # 状态-动作访问统计
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.state_novelty_scores = {}
        self.recent_states = deque(maxlen=1000)
    
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None, 
                     state: Optional[np.ndarray] = None) -> int:
        """
        好奇心驱动的动作选择
        """
        self.step_count += 1
        
        if state is None:
            # 如果没有提供状态，使用标准epsilon-greedy
            if np.random.random() < 0.1:
                action = np.random.randint(self.action_size)
                exploration_type = "random"
            else:
                action = np.argmax(q_values)
                exploration_type = "greedy"
        else:
            # 计算好奇心奖励
            curiosity_rewards = self._compute_curiosity_rewards(state)
            
            # 结合Q值和好奇心奖励
            combined_values = q_values + self.curiosity_weight * curiosity_rewards
            action = np.argmax(combined_values)
            exploration_type = "curiosity_driven"
            
            # 更新状态访问统计
            state_key = self._state_to_key(state)
            self.state_action_counts[state_key][action] += 1
            self.recent_states.append(state_key)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'q_values': q_values.copy(),
            'curiosity_rewards': curiosity_rewards if state is not None else None,
            'type': exploration_type
        })
        
        return action
    
    def _compute_curiosity_rewards(self, state: np.ndarray) -> np.ndarray:
        """计算每个动作的好奇心奖励"""
        state_key = self._state_to_key(state)
        curiosity_rewards = np.zeros(self.action_size)
        
        for action in range(self.action_size):
            # 基于访问频率的好奇心
            visit_count = self.state_action_counts[state_key][action]
            curiosity_rewards[action] = 1.0 / (1.0 + visit_count)
            
            # 基于状态新颖性的好奇心
            novelty_score = self._compute_state_novelty(state)
            curiosity_rewards[action] += novelty_score
        
        return curiosity_rewards
    
    def _compute_state_novelty(self, state: np.ndarray) -> float:
        """计算状态新颖性分数"""
        state_key = self._state_to_key(state)
        
        if state_key in self.state_novelty_scores:
            return self.state_novelty_scores[state_key]
        
        # 计算与历史状态的相似性
        if len(self.recent_states) == 0:
            novelty_score = 1.0
        else:
            # 简化的新颖性计算：基于状态键的唯一性
            unique_states = set(self.recent_states)
            if state_key in unique_states:
                frequency = list(self.recent_states).count(state_key)
                novelty_score = 1.0 / (1.0 + frequency)
            else:
                novelty_score = 1.0
        
        self.state_novelty_scores[state_key] = novelty_score
        return novelty_score
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """将状态转换为可哈希的键"""
        # 量化状态以减少维度
        quantized = np.round(state, 2)
        return str(quantized.tolist())
    
    def update(self, action: int, reward: float, **kwargs):
        """更新好奇心参数"""
        # 可以根据奖励调整好奇心权重
        pass
    
    def reset(self):
        """重置好奇心状态"""
        super().reset()
        self.state_action_counts.clear()
        self.state_novelty_scores.clear()
        self.recent_states.clear()


class AdaptiveExplorationStrategy(ExplorationStrategy):
    """自适应探索策略"""
    
    def __init__(self, config: ExplorationConfig, action_size: int, 
                 strategies: List[ExplorationStrategy]):
        super().__init__(config, action_size)
        self.strategies = strategies
        self.strategy_weights = np.ones(len(strategies)) / len(strategies)
        self.strategy_rewards = [deque(maxlen=100) for _ in strategies]
        self.current_strategy_idx = 0
        
    def select_action(self, q_values: np.ndarray, step: Optional[int] = None, **kwargs) -> int:
        """
        自适应策略选择
        """
        self.step_count += 1
        
        # 根据权重选择策略
        self.current_strategy_idx = np.random.choice(len(self.strategies), p=self.strategy_weights)
        current_strategy = self.strategies[self.current_strategy_idx]
        
        # 使用选中的策略选择动作
        action = current_strategy.select_action(q_values, step, **kwargs)
        
        # 记录探索历史
        self.exploration_history.append({
            'step': self.step_count,
            'action': action,
            'strategy_idx': self.current_strategy_idx,
            'strategy_type': type(current_strategy).__name__,
            'strategy_weights': self.strategy_weights.copy(),
            'type': 'adaptive'
        })
        
        return action
    
    def update(self, action: int, reward: float, **kwargs):
        """更新策略权重"""
        # 记录当前策略的奖励
        self.strategy_rewards[self.current_strategy_idx].append(reward)
        
        # 更新所有子策略
        for strategy in self.strategies:
            strategy.update(action, reward, **kwargs)
        
        # 定期更新策略权重
        if self.step_count % 100 == 0:
            self._update_strategy_weights()
    
    def _update_strategy_weights(self):
        """基于性能更新策略权重"""
        strategy_scores = []
        
        for rewards in self.strategy_rewards:
            if len(rewards) > 0:
                score = np.mean(rewards)
            else:
                score = 0.0
            strategy_scores.append(score)
        
        # 软最大转换为权重
        if max(strategy_scores) > min(strategy_scores):
            exp_scores = np.exp(np.array(strategy_scores) - max(strategy_scores))
            self.strategy_weights = exp_scores / np.sum(exp_scores)
        else:
            # 如果性能相同，使用均匀权重
            self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
    
    def reset(self):
        """重置自适应策略"""
        super().reset()
        for strategy in self.strategies:
            strategy.reset()
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        for rewards in self.strategy_rewards:
            rewards.clear()