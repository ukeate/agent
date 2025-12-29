"""
Linear Contextual Bandit 线性上下文多臂老虎机算法实现

线性上下文老虎机考虑上下文特征，假设奖励是上下文特征的线性函数，
使用在线学习方法更新每个臂的参数。
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base import MultiArmedBandit

class LinearContextualBandit(MultiArmedBandit):
    """线性上下文多臂老虎机算法"""
    
    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        初始化线性上下文老虎机
        
        Args:
            n_arms: 臂的数量
            n_features: 上下文特征维度
            alpha: 置信参数，控制探索程度
            lambda_reg: 正则化参数
            random_state: 随机种子
        """
        super().__init__(n_arms, random_state)
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # 每个臂的线性模型参数
        # A[i]是第i个臂的协方差矩阵的逆
        self.A = np.array([
            lambda_reg * np.identity(n_features) 
            for _ in range(n_arms)
        ])
        
        # b[i]是第i个臂的奖励向量
        self.b = np.zeros((n_arms, n_features))
        
        # theta[i]是第i个臂的参数向量
        self.theta = np.zeros((n_arms, n_features))
        
    def select_arm(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        使用线性UCB策略选择臂
        
        Args:
            context: 包含特征向量的上下文信息
            
        Returns:
            选择的臂索引
            
        Raises:
            ValueError: 如果context为None或不包含features
        """
        if context is None or 'features' not in context:
            raise ValueError("Linear Contextual Bandit需要提供包含'features'的context")
        
        features = np.array(context['features'])
        if features.shape[0] != self.n_features:
            raise ValueError(f"特征维度不匹配：期望{self.n_features}，得到{features.shape[0]}")
        
        # 计算每个臂的UCB值
        ucb_values = self._calculate_linear_ucb_values(features)
        
        # 选择UCB值最高的臂
        return int(np.argmax(ucb_values))
    
    def _calculate_linear_ucb_values(self, features: np.ndarray) -> np.ndarray:
        """
        计算线性UCB值
        
        Args:
            features: 上下文特征向量
            
        Returns:
            每个臂的UCB值
        """
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # 更新theta参数
            try:
                self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]
            
            # 计算预期奖励
            expected_reward = features @ self.theta[arm]
            
            # 计算置信区间
            try:
                A_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[arm])
            
            confidence_bound = self.alpha * np.sqrt(features @ A_inv @ features)
            
            # UCB值 = 预期奖励 + 置信区间
            ucb_values[arm] = expected_reward + confidence_bound
        
        return ucb_values
    
    def _update_algorithm_state(self, arm: int, reward: float, context: Optional[Dict[str, Any]] = None):
        """
        更新线性模型参数
        
        Args:
            arm: 被选择的臂
            reward: 获得的奖励
            context: 上下文信息（必须包含特征）
        """
        if context is None or 'features' not in context:
            raise ValueError("更新时必须提供包含'features'的context")
        
        features = np.array(context['features'])
        
        # 更新A矩阵：A = A + x*x^T
        self.A[arm] += np.outer(features, features)
        
        # 更新b向量：b = b + r*x
        self.b[arm] += reward * features
    
    def _get_confidence_intervals(self) -> np.ndarray:
        """
        获取置信区间（返回平均置信界限）
        
        Returns:
            每个臂的平均置信界限
        """
        # 对于上下文老虎机，置信区间依赖于具体的上下文
        # 这里返回基于单位向量的平均置信界限
        avg_confidence = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            try:
                A_inv = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(self.A[arm])
            
            # 使用单位向量计算平均置信界限
            unit_vectors = np.identity(self.n_features)
            confidence_bounds = []
            
            for unit_vec in unit_vectors:
                cb = self.alpha * np.sqrt(unit_vec @ A_inv @ unit_vec)
                confidence_bounds.append(cb)
            
            avg_confidence[arm] = np.mean(confidence_bounds)
        
        return avg_confidence
    
    def predict_reward(self, arm: int, features: np.ndarray) -> float:
        """
        预测给定上下文下某个臂的奖励
        
        Args:
            arm: 臂索引
            features: 上下文特征向量
            
        Returns:
            预测的奖励值
        """
        if arm >= self.n_arms or arm < 0:
            raise ValueError(f"臂索引{arm}超出范围[0, {self.n_arms-1}]")
        
        # 更新theta参数
        try:
            self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
        except np.linalg.LinAlgError:
            self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]
        
        return features @ self.theta[arm]
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        获取模型参数
        
        Returns:
            包含模型参数的字典
        """
        # 更新所有theta参数
        for arm in range(self.n_arms):
            try:
                self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
            except np.linalg.LinAlgError:
                self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]
        
        return {
            "theta": self.theta.copy(),
            "A_matrices": self.A.copy(),
            "b_vectors": self.b.copy()
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性（基于theta参数的L2范数）
        
        Returns:
            每个臂的特征重要性向量
        """
        importance = np.zeros((self.n_arms, self.n_features))
        
        for arm in range(self.n_arms):
            try:
                self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
            except np.linalg.LinAlgError:
                self.theta[arm] = np.linalg.pinv(self.A[arm]) @ self.b[arm]
            
            # 使用绝对值作为重要性度量
            importance[arm] = np.abs(self.theta[arm])
        
        return importance
    
    def _reset_algorithm_state(self):
        """重置线性上下文老虎机特定状态"""
        # 重置A矩阵
        self.A = np.array([
            self.lambda_reg * np.identity(self.n_features) 
            for _ in range(self.n_arms)
        ])
        
        # 重置b向量和theta参数
        self.b = np.zeros((self.n_arms, self.n_features))
        self.theta = np.zeros((self.n_arms, self.n_features))
    
    def get_algorithm_params(self) -> Dict[str, Any]:
        """
        获取算法参数
        
        Returns:
            算法参数字典
        """
        return {
            "algorithm": "Linear Contextual Bandit",
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "random_state": self.random_state
        }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的算法统计信息
        
        Returns:
            详细统计信息字典
        """
        base_stats = self.get_arm_stats()
        model_params = self.get_model_parameters()
        feature_importance = self.get_feature_importance()
        
        return {
            **base_stats,
            "model_parameters": model_params,
            "feature_importance": feature_importance,
            "algorithm_params": self.get_algorithm_params(),
            "performance_metrics": self.get_performance_metrics()
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"LinearContextualBandit(n_arms={self.n_arms}, "
                f"n_features={self.n_features}, alpha={self.alpha}, "
                f"total_pulls={self.total_pulls})")
