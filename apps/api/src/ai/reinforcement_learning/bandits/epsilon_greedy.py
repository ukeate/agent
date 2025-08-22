"""
Epsilon-Greedy 多臂老虎机算法实现

Epsilon-Greedy算法以epsilon概率进行随机探索，
以(1-epsilon)概率选择当前最佳的臂。
"""

from typing import Dict, Any, Optional, Union
import numpy as np

from .base import MultiArmedBandit


class EpsilonGreedyBandit(MultiArmedBandit):
    """Epsilon-Greedy 多臂老虎机算法"""
    
    def __init__(
        self,
        n_arms: int,
        epsilon: Union[float, str] = 0.1,
        decay_rate: float = 0.995,
        min_epsilon: float = 0.01,
        random_state: Optional[int] = None
    ):
        """
        初始化Epsilon-Greedy算法
        
        Args:
            n_arms: 臂的数量
            epsilon: 探索率，可以是固定值或"decay"表示衰减
            decay_rate: epsilon衰减率（当epsilon="decay"时使用）
            min_epsilon: 最小epsilon值
            random_state: 随机种子
        """
        super().__init__(n_arms, random_state)
        self.initial_epsilon = epsilon
        self.epsilon = 0.1 if epsilon == "decay" else epsilon
        self.use_decay = epsilon == "decay"
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        使用epsilon-greedy策略选择臂
        
        Args:
            context: 上下文信息（Epsilon-Greedy基础版本不使用上下文）
            
        Returns:
            选择的臂索引
        """
        # 更新epsilon（如果使用衰减）
        if self.use_decay:
            self._update_epsilon()
        
        # 以epsilon概率进行随机探索
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        else:
            # 选择当前最佳臂（贪婪选择）
            return self._select_greedy_arm()
    
    def _select_greedy_arm(self) -> int:
        """
        选择当前估计奖励最高的臂
        
        Returns:
            最佳臂的索引
        """
        # 如果有未探索的臂，优先选择
        unplayed_arms = np.where(self.n_pulls == 0)[0]
        if len(unplayed_arms) > 0:
            return int(self.rng.choice(unplayed_arms))
        
        # 计算平均奖励
        average_rewards = self.rewards / np.maximum(self.n_pulls, 1e-10)
        
        # 选择平均奖励最高的臂，如有相同值则随机选择
        max_reward = np.max(average_rewards)
        best_arms = np.where(average_rewards == max_reward)[0]
        
        return int(self.rng.choice(best_arms))
    
    def _update_epsilon(self):
        """更新epsilon值（衰减模式）"""
        if self.use_decay:
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.decay_rate
            )
    
    def _update_algorithm_state(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        Epsilon-Greedy算法不需要额外的状态更新
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息
        """
        # Epsilon-Greedy算法的状态更新已在基类的update方法中完成
        pass
    
    def _get_confidence_intervals(self) -> np.ndarray:
        """
        计算每个臂的置信区间（基于标准误差）
        
        Returns:
            每个臂的置信区间上界
        """
        if self.total_pulls <= 0:
            return np.zeros(self.n_arms)
        
        # 计算每个臂的标准误差
        average_rewards = np.zeros(self.n_arms)
        standard_errors = np.zeros(self.n_arms)
        
        for i in range(self.n_arms):
            if self.n_pulls[i] > 0:
                average_rewards[i] = self.rewards[i] / self.n_pulls[i]
                
                # 估计方差（假设奖励在[0,1]范围内）
                variance = average_rewards[i] * (1 - average_rewards[i])
                standard_errors[i] = np.sqrt(variance / self.n_pulls[i])
        
        # 使用1.96倍标准误差作为95%置信区间
        confidence_bounds = 1.96 * standard_errors
        
        return confidence_bounds
    
    def set_epsilon(self, epsilon: float):
        """
        设置新的epsilon值
        
        Args:
            epsilon: 新的探索率
        """
        self.epsilon = max(0.0, min(1.0, epsilon))
    
    def get_current_epsilon(self) -> float:
        """
        获取当前epsilon值
        
        Returns:
            当前epsilon值
        """
        return self.epsilon
    
    def _reset_algorithm_state(self):
        """重置Epsilon-Greedy算法特定状态"""
        if self.use_decay:
            self.epsilon = 0.1
        else:
            self.epsilon = self.initial_epsilon
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """
        获取算法参数
        
        Returns:
            算法参数字典
        """
        return {
            "algorithm": "Epsilon-Greedy",
            "initial_epsilon": self.initial_epsilon,
            "current_epsilon": self.epsilon,
            "use_decay": self.use_decay,
            "decay_rate": self.decay_rate,
            "min_epsilon": self.min_epsilon,
            "n_arms": self.n_arms,
            "random_state": self.random_state
        }
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        获取探索相关统计信息
        
        Returns:
            探索统计信息字典
        """
        total_actions = len(self.arm_history)
        if total_actions == 0:
            return {
                "exploration_rate": 0.0,
                "exploitation_rate": 0.0,
                "random_actions": 0,
                "greedy_actions": 0
            }
        
        # 估算探索动作数量（简单估计）
        estimated_random_actions = int(total_actions * self.epsilon)
        estimated_greedy_actions = total_actions - estimated_random_actions
        
        return {
            "current_epsilon": self.epsilon,
            "estimated_exploration_rate": self.epsilon,
            "estimated_exploitation_rate": 1 - self.epsilon,
            "estimated_random_actions": estimated_random_actions,
            "estimated_greedy_actions": estimated_greedy_actions,
            "total_actions": total_actions
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的算法统计信息
        
        Returns:
            详细统计信息字典
        """
        base_stats = self.get_arm_stats()
        exploration_stats = self.get_exploration_stats()
        
        return {
            **base_stats,
            **exploration_stats,
            "algorithm_params": self.get_algorithm_params(),
            "performance_metrics": self.get_performance_metrics()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"EpsilonGreedyBandit(n_arms={self.n_arms}, "
                f"epsilon={self.epsilon:.3f}, total_pulls={self.total_pulls})")