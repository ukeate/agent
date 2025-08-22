"""
Upper Confidence Bound (UCB) 多臂老虎机算法实现

UCB算法通过计算每个臂的置信上界来平衡探索与利用，
选择具有最高置信上界的臂。
"""

import math
from typing import Dict, Any, Optional
import numpy as np

from .base import MultiArmedBandit


class UCBBandit(MultiArmedBandit):
    """Upper Confidence Bound (UCB) 多臂老虎机算法"""
    
    def __init__(self, n_arms: int, c: float = 2.0, random_state: Optional[int] = None):
        """
        初始化UCB算法
        
        Args:
            n_arms: 臂的数量
            c: 置信参数，控制探索程度（通常取值为2.0）
            random_state: 随机种子
        """
        super().__init__(n_arms, random_state)
        self.c = c
        
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        选择具有最高UCB值的臂
        
        Args:
            context: 上下文信息（UCB算法不使用上下文）
            
        Returns:
            选择的臂索引
        """
        # 如果还有未探索的臂，优先选择
        unplayed_arms = np.where(self.n_pulls == 0)[0]
        if len(unplayed_arms) > 0:
            return int(self.rng.choice(unplayed_arms))
        
        # 计算每个臂的UCB值
        ucb_values = self._calculate_ucb_values()
        
        # 选择UCB值最高的臂
        return int(np.argmax(ucb_values))
    
    def _calculate_ucb_values(self) -> np.ndarray:
        """
        计算每个臂的UCB值
        
        Returns:
            每个臂的UCB值数组
        """
        # 避免除零错误
        safe_pulls = np.maximum(self.n_pulls, 1e-10)
        
        # 计算平均奖励
        average_rewards = self.rewards / safe_pulls
        
        # 计算置信区间
        if self.total_pulls <= 0:
            confidence_bounds = np.full(self.n_arms, float('inf'))
        else:
            log_t = math.log(max(self.total_pulls, 1))
            confidence_bounds = self.c * np.sqrt(log_t / safe_pulls)
        
        # UCB值 = 平均奖励 + 置信区间
        ucb_values = average_rewards + confidence_bounds
        
        return ucb_values
    
    def _update_algorithm_state(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        UCB算法不需要额外的状态更新
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息
        """
        # UCB算法的状态更新已在基类的update方法中完成
        pass
    
    def _get_confidence_intervals(self) -> np.ndarray:
        """
        计算每个臂的置信区间上界
        
        Returns:
            每个臂的置信区间上界
        """
        if self.total_pulls <= 0:
            return np.full(self.n_arms, float('inf'))
        
        safe_pulls = np.maximum(self.n_pulls, 1e-10)
        log_t = math.log(max(self.total_pulls, 1))
        confidence_bounds = self.c * np.sqrt(log_t / safe_pulls)
        
        return confidence_bounds
    
    def _reset_algorithm_state(self):
        """
        重置UCB算法特定状态
        
        UCB算法没有额外的状态需要重置
        """
        pass
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """
        获取算法参数
        
        Returns:
            算法参数字典
        """
        return {
            "algorithm": "UCB",
            "c": self.c,
            "n_arms": self.n_arms,
            "random_state": self.random_state
        }
    
    def get_ucb_values(self) -> np.ndarray:
        """
        获取当前所有臂的UCB值
        
        Returns:
            所有臂的UCB值
        """
        if self.total_pulls == 0:
            return np.full(self.n_arms, float('inf'))
        return self._calculate_ucb_values()
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的算法统计信息
        
        Returns:
            详细统计信息字典
        """
        base_stats = self.get_arm_stats()
        ucb_values = self.get_ucb_values()
        
        return {
            **base_stats,
            "ucb_values": ucb_values,
            "algorithm_params": self.get_algorithm_params(),
            "performance_metrics": self.get_performance_metrics()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"UCBBandit(n_arms={self.n_arms}, c={self.c}, total_pulls={self.total_pulls})"