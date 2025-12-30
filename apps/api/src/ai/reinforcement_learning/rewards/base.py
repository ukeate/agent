"""
奖励函数基础框架

定义奖励函数的抽象基类和核心数据结构
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from enum import Enum

class RewardType(Enum):
    """奖励类型枚举"""
    LINEAR = "linear"
    STEP = "step" 
    THRESHOLD = "threshold"
    GAUSSIAN = "gaussian"
    WEIGHTED = "weighted"
    PRODUCT = "product"
    PIECEWISE = "piecewise"
    ADAPTIVE = "adaptive"
    PROGRESS_BASED = "progress_based"
    CURRICULUM = "curriculum"

@dataclass
class RewardConfig:
    """奖励函数配置"""
    reward_type: RewardType
    parameters: Dict[str, Any] = field(default_factory=dict)
    normalization: bool = True
    clipping: Optional[Tuple[float, float]] = None
    scaling_factor: float = 1.0
    
    # 多维度奖励配置
    dimensions: List[str] = field(default_factory=list)
    dimension_weights: Dict[str, float] = field(default_factory=dict)
    
    # 自适应参数
    adaptation_rate: float = 0.01
    adaptation_window: int = 1000
    min_value: float = -10.0
    max_value: float = 10.0

@dataclass
class RewardMetrics:
    """奖励指标统计"""
    total_reward: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    
    # 统计指标
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = float('inf')
    max_reward: float = float('-inf')
    
    # 维度分解
    dimension_rewards: Dict[str, float] = field(default_factory=dict)
    dimension_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 适应性指标
    adaptation_history: List[float] = field(default_factory=list)
    reward_density: float = 0.0  # 非零奖励比例
    reward_sparsity: float = 1.0  # 奖励稀疏度

class RewardFunction(ABC):
    """奖励函数抽象基类"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.metrics = RewardMetrics()
        self.step_count = 0
        self.episode_count = 0
        
        # 奖励历史用于自适应调整
        self.reward_history = []
        self.recent_rewards = []
        
    @abstractmethod
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        计算奖励值
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            done: 是否结束
            info: 额外信息
            
        Returns:
            float: 计算的奖励值
        """
        ...
    
    def compute_multi_dimensional_reward(self, 
                                       state: np.ndarray, 
                                       action: int, 
                                       next_state: np.ndarray, 
                                       done: bool,
                                       info: Dict[str, Any] = None) -> Dict[str, float]:
        """
        计算多维度奖励
        
        Returns:
            Dict[str, float]: 各维度奖励值
        """
        # 基础实现：单一奖励值映射到所有维度
        reward = self.compute_reward(state, action, next_state, done, info)
        
        if not self.config.dimensions:
            return {"default": reward}
            
        # 按权重分配到各维度
        dimension_rewards = {}
        total_weight = sum(self.config.dimension_weights.get(dim, 1.0) 
                          for dim in self.config.dimensions)
        
        for dim in self.config.dimensions:
            weight = self.config.dimension_weights.get(dim, 1.0)
            dimension_rewards[dim] = reward * (weight / total_weight)
            
        return dimension_rewards
    
    def update_metrics(self, reward: float, dimension_rewards: Dict[str, float] = None):
        """更新奖励指标"""
        self.step_count += 1
        self.metrics.total_reward += reward
        self.metrics.step_rewards.append(reward)
        
        # 更新统计指标
        self.metrics.min_reward = min(self.metrics.min_reward, reward)
        self.metrics.max_reward = max(self.metrics.max_reward, reward)
        
        # 计算均值和标准差
        if len(self.metrics.step_rewards) > 0:
            rewards_array = np.array(self.metrics.step_rewards)
            self.metrics.mean_reward = np.mean(rewards_array)
            self.metrics.std_reward = np.std(rewards_array)
            
        # 更新维度奖励
        if dimension_rewards:
            for dim, dim_reward in dimension_rewards.items():
                if dim not in self.metrics.dimension_rewards:
                    self.metrics.dimension_rewards[dim] = 0.0
                    self.metrics.dimension_stats[dim] = {
                        'mean': 0.0, 'std': 0.0, 'min': float('inf'), 'max': float('-inf')
                    }
                    
                self.metrics.dimension_rewards[dim] += dim_reward
                
                # 更新维度统计
                stats = self.metrics.dimension_stats[dim]
                stats['min'] = min(stats['min'], dim_reward)
                stats['max'] = max(stats['max'], dim_reward)
                
        # 计算奖励密度和稀疏度
        non_zero_rewards = len([r for r in self.metrics.step_rewards if abs(r) > 1e-6])
        total_rewards = len(self.metrics.step_rewards)
        self.metrics.reward_density = non_zero_rewards / max(total_rewards, 1)
        self.metrics.reward_sparsity = 1.0 - self.metrics.reward_density
        
        # 维护历史记录用于自适应
        self.reward_history.append(reward)
        self.recent_rewards.append(reward)
        
        # 保持最近的奖励窗口
        if len(self.recent_rewards) > self.config.adaptation_window:
            self.recent_rewards.pop(0)
    
    def normalize_reward(self, reward: float) -> float:
        """奖励归一化"""
        if not self.config.normalization:
            return reward
            
        # 基于历史统计的Z-score归一化
        if len(self.recent_rewards) > 10:
            mean_reward = np.mean(self.recent_rewards)
            std_reward = np.std(self.recent_rewards) + 1e-8
            normalized = (reward - mean_reward) / std_reward
        else:
            normalized = reward
            
        # 应用缩放因子
        normalized *= self.config.scaling_factor
        
        # 应用裁剪
        if self.config.clipping:
            min_val, max_val = self.config.clipping
            normalized = np.clip(normalized, min_val, max_val)
            
        return normalized
    
    def on_episode_start(self):
        """回合开始时调用"""
        self.episode_count += 1
        
    def on_episode_end(self, episode_reward: float):
        """回合结束时调用"""
        self.metrics.episode_rewards.append(episode_reward)
        
        # 更新维度统计的均值和标准差
        for dim in self.metrics.dimension_stats:
            dim_rewards = []
            # 这里需要更复杂的逻辑来收集每个维度的奖励历史
            # 简化实现
            if len(self.metrics.episode_rewards) > 0:
                stats = self.metrics.dimension_stats[dim]
                dim_reward_history = [self.metrics.dimension_rewards[dim]]
                stats['mean'] = np.mean(dim_reward_history)
                stats['std'] = np.std(dim_reward_history) if len(dim_reward_history) > 1 else 0.0
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = RewardMetrics()
        self.step_count = 0
        self.episode_count = 0
        self.reward_history.clear()
        self.recent_rewards.clear()
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """获取奖励摘要"""
        return {
            'total_reward': self.metrics.total_reward,
            'mean_reward': self.metrics.mean_reward,
            'std_reward': self.metrics.std_reward,
            'min_reward': self.metrics.min_reward,
            'max_reward': self.metrics.max_reward,
            'reward_density': self.metrics.reward_density,
            'reward_sparsity': self.metrics.reward_sparsity,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'dimension_rewards': self.metrics.dimension_rewards,
            'dimension_stats': self.metrics.dimension_stats
        }
