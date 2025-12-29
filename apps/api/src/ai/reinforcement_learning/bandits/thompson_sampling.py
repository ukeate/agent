"""
Thompson Sampling 多臂老虎机算法实现

Thompson Sampling是一种贝叶斯方法，通过维护每个臂奖励分布的后验概率，
从后验分布中采样来决定选择哪个臂。
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import MultiArmedBandit

class ThompsonSamplingBandit(MultiArmedBandit):
    """Thompson Sampling 多臂老虎机算法"""
    
    def __init__(
        self, 
        n_arms: int, 
        alpha_init: float = 1.0, 
        beta_init: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        初始化Thompson Sampling算法
        
        Args:
            n_arms: 臂的数量
            alpha_init: Beta分布的初始alpha参数（成功数+1）
            beta_init: Beta分布的初始beta参数（失败数+1）
            random_state: 随机种子
        """
        super().__init__(n_arms, random_state)
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        
        # 每个臂的Beta分布参数
        self.alpha = np.full(n_arms, alpha_init)
        self.beta = np.full(n_arms, beta_init)
        
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        从每个臂的后验分布中采样，选择采样值最高的臂
        
        Args:
            context: 上下文信息（Thompson Sampling基础版本不使用上下文）
            
        Returns:
            选择的臂索引
        """
        # 从每个臂的Beta分布中采样
        sampled_values = np.array([
            self.rng.beta(self.alpha[i], self.beta[i]) 
            for i in range(self.n_arms)
        ])
        
        # 选择采样值最高的臂
        return int(np.argmax(sampled_values))
    
    def _update_algorithm_state(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        更新选中臂的Beta分布参数
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息
        """
        # 将连续奖励转换为二项结果
        # 这里假设奖励是0-1之间的值，可以直接作为成功概率
        # 更复杂的情况可能需要不同的转换方法
        if reward > 0.5:
            # 成功：更新alpha参数
            self.alpha[arm] += 1
        else:
            # 失败：更新beta参数  
            self.beta[arm] += 1
    
    def update_with_binary_reward(self, arm: int, success: bool):
        """
        使用二项奖励更新算法状态
        
        Args:
            arm: 被选择的臂
            success: 是否成功（True为成功，False为失败）
        """
        if success:
            self.alpha[arm] += 1
            reward = 1.0
        else:
            self.beta[arm] += 1
            reward = 0.0
            
        # 更新基类的统计信息
        self.n_pulls[arm] += 1
        self.rewards[arm] += reward
        self.total_pulls += 1
        self.total_reward += reward
        
        # 记录历史
        self.reward_history.append(reward)
        self.arm_history.append(arm)
    
    def update_with_continuous_reward(self, arm: int, reward: float, n_trials: int = 1):
        """
        使用连续奖励更新算法状态
        
        Args:
            arm: 被选择的臂
            reward: 连续奖励值
            n_trials: 试验次数
        """
        # 将连续奖励转换为成功次数
        # 假设奖励在[0,1]范围内，直接作为成功率
        reward_clamped = np.clip(reward, 0, 1)
        success_count = reward_clamped * n_trials
        failure_count = n_trials - success_count
        
        # 更新Beta分布参数
        self.alpha[arm] += success_count
        self.beta[arm] += failure_count
        
        # 更新基类的统计信息
        self.n_pulls[arm] += n_trials
        self.rewards[arm] += reward * n_trials
        self.total_pulls += n_trials
        self.total_reward += reward * n_trials
        
        # 记录历史
        self.reward_history.append(reward)
        self.arm_history.append(arm)
    
    def _get_confidence_intervals(self) -> np.ndarray:
        """
        计算每个臂的置信区间（基于Beta分布的方差）
        
        Returns:
            每个臂的置信区间上界
        """
        # Beta分布的均值和方差
        means = self.alpha / (self.alpha + self.beta)
        variances = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        
        # 使用2倍标准差作为置信区间
        confidence_bounds = 2 * np.sqrt(variances)
        
        return confidence_bounds
    
    def get_posterior_stats(self) -> Dict[str, np.ndarray]:
        """
        获取每个臂的后验分布统计信息
        
        Returns:
            包含后验统计信息的字典
        """
        # Beta分布的均值和方差
        means = self.alpha / (self.alpha + self.beta)
        variances = (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        
        return {
            "posterior_means": means,
            "posterior_variances": variances,
            "alpha_params": self.alpha.copy(),
            "beta_params": self.beta.copy()
        }
    
    def sample_from_posterior(self, n_samples: int = 1000) -> np.ndarray:
        """
        从每个臂的后验分布中采样
        
        Args:
            n_samples: 采样数量
            
        Returns:
            形状为(n_samples, n_arms)的采样数组
        """
        samples = np.zeros((n_samples, self.n_arms))
        
        for i in range(self.n_arms):
            samples[:, i] = self.rng.beta(self.alpha[i], self.beta[i], size=n_samples)
            
        return samples
    
    def get_arm_probabilities(self, n_samples: int = 1000) -> np.ndarray:
        """
        计算每个臂是最优臂的概率
        
        Args:
            n_samples: 用于估计的采样数量
            
        Returns:
            每个臂是最优臂的概率
        """
        samples = self.sample_from_posterior(n_samples)
        
        # 计算每次采样中最优臂
        best_arms = np.argmax(samples, axis=1)
        
        # 计算每个臂被选为最优的概率
        probabilities = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            probabilities[arm] = np.sum(best_arms == arm) / n_samples
            
        return probabilities
    
    def _reset_algorithm_state(self):
        """重置Thompson Sampling算法特定状态"""
        self.alpha = np.full(self.n_arms, self.alpha_init)
        self.beta = np.full(self.n_arms, self.beta_init)
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """
        获取算法参数
        
        Returns:
            算法参数字典
        """
        return {
            "algorithm": "Thompson Sampling",
            "alpha_init": self.alpha_init,
            "beta_init": self.beta_init,
            "n_arms": self.n_arms,
            "random_state": self.random_state
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的算法统计信息
        
        Returns:
            详细统计信息字典
        """
        base_stats = self.get_arm_stats()
        posterior_stats = self.get_posterior_stats()
        arm_probabilities = self.get_arm_probabilities()
        
        return {
            **base_stats,
            **posterior_stats,
            "arm_selection_probabilities": arm_probabilities,
            "algorithm_params": self.get_algorithm_params(),
            "performance_metrics": self.get_performance_metrics()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ThompsonSamplingBandit(n_arms={self.n_arms}, "
                f"alpha_init={self.alpha_init}, beta_init={self.beta_init}, "
                f"total_pulls={self.total_pulls})")
