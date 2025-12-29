"""
多臂老虎机算法抽象基类

定义多臂老虎机算法的统一接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class MultiArmedBandit(ABC):
    """多臂老虎机算法抽象基类"""
    
    def __init__(self, n_arms: int, random_state: Optional[int] = None):
        """
        初始化多臂老虎机
        
        Args:
            n_arms: 臂的数量
            random_state: 随机种子
        """
        self.n_arms = n_arms
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
        # 统计信息
        self.n_pulls = np.zeros(n_arms)  # 每个臂的拉取次数
        self.rewards = np.zeros(n_arms)  # 每个臂的累积奖励
        self.total_pulls = 0  # 总拉取次数
        self.total_reward = 0.0  # 总奖励
        
        # 性能指标
        self.regret_history = []  # 遗憾历史
        self.reward_history = []  # 奖励历史
        self.arm_history = []  # 选择的臂历史
        
    @abstractmethod
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        选择一个臂
        
        Args:
            context: 上下文信息（用于上下文感知的老虎机）
            
        Returns:
            选择的臂索引
        """
        raise NotImplementedError
    
    def update(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        更新算法状态
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息
        """
        self.n_pulls[arm] += 1
        self.rewards[arm] += reward
        self.total_pulls += 1
        self.total_reward += reward
        
        # 记录历史
        self.reward_history.append(reward)
        self.arm_history.append(arm)
        
        # 更新算法特定的状态
        self._update_algorithm_state(arm, reward, context)
        
    @abstractmethod
    def _update_algorithm_state(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        更新算法特定的状态
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息
        """
        raise NotImplementedError
    
    def get_arm_stats(self) -> Dict[str, Any]:
        """
        获取每个臂的统计信息
        
        Returns:
            包含统计信息的字典
        """
        estimated_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.n_pulls[i] > 0:
                estimated_rewards[i] = self.rewards[i] / self.n_pulls[i]
        
        return {
            "n_pulls": self.n_pulls.copy(),
            "total_rewards": self.rewards.copy(),
            "estimated_rewards": estimated_rewards,
            "confidence_intervals": self._get_confidence_intervals()
        }
    
    @abstractmethod
    def _get_confidence_intervals(self) -> np.ndarray:
        """
        计算每个臂的置信区间
        
        Returns:
            每个臂的置信区间上界
        """
        raise NotImplementedError
    
    def get_best_arm(self) -> int:
        """
        获取当前最佳臂
        
        Returns:
            最佳臂的索引
        """
        stats = self.get_arm_stats()
        return int(np.argmax(stats["estimated_rewards"]))
    
    def calculate_regret(self, optimal_reward: float) -> float:
        """
        计算累积遗憾
        
        Args:
            optimal_reward: 最优奖励
            
        Returns:
            累积遗憾值
        """
        if not self.reward_history:
            return 0.0
            
        cumulative_regret = 0.0
        for reward in self.reward_history:
            cumulative_regret += optimal_reward - reward
            
        return cumulative_regret
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        if self.total_pulls == 0:
            return {
                "total_pulls": 0,
                "average_reward": 0.0,
                "best_arm": -1,
                "exploration_rate": 0.0
            }
        
        # 计算探索率（选择非最佳臂的比例）
        best_arm = self.get_best_arm()
        exploration_count = sum(1 for arm in self.arm_history if arm != best_arm)
        exploration_rate = exploration_count / len(self.arm_history) if self.arm_history else 0.0
        
        return {
            "total_pulls": self.total_pulls,
            "average_reward": self.total_reward / self.total_pulls,
            "best_arm": best_arm,
            "exploration_rate": exploration_rate,
            "arm_selection_distribution": self.n_pulls / self.total_pulls if self.total_pulls > 0 else np.zeros(self.n_arms)
        }
    
    def reset(self):
        """重置算法状态"""
        self.n_pulls = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.total_pulls = 0
        self.total_reward = 0.0
        self.regret_history = []
        self.reward_history = []
        self.arm_history = []
        self._reset_algorithm_state()
    
    @abstractmethod
    def _reset_algorithm_state(self):
        """重置算法特定的状态"""
        raise NotImplementedError
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(n_arms={self.n_arms}, total_pulls={self.total_pulls})"
