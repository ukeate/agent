"""
基础奖励函数实现

实现常用的基础奖励函数类型
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional
from .base import RewardFunction, RewardConfig, RewardType

class LinearReward(RewardFunction):
    """线性奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.slope = config.parameters.get('slope', 1.0)
        self.intercept = config.parameters.get('intercept', 0.0)
        self.target_feature = config.parameters.get('target_feature', 0)  # 目标状态特征索引
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        线性奖励: R = slope * feature + intercept
        """
        if isinstance(self.target_feature, int):
            feature_value = next_state[self.target_feature] if len(next_state) > self.target_feature else 0.0
        elif isinstance(self.target_feature, str) and info:
            feature_value = info.get(self.target_feature, 0.0)
        else:
            feature_value = np.sum(next_state)  # 默认使用状态和
            
        reward = self.slope * feature_value + self.intercept
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class StepReward(RewardFunction):
    """阶跃奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.positive_reward = config.parameters.get('positive_reward', 1.0)
        self.negative_reward = config.parameters.get('negative_reward', -1.0)
        self.neutral_reward = config.parameters.get('neutral_reward', 0.0)
        self.success_condition = config.parameters.get('success_condition', 'goal_reached')
        self.failure_condition = config.parameters.get('failure_condition', 'collision')
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        阶跃奖励: 根据条件返回固定奖励值
        """
        if info is None:
            info = {}
        
        # 检查成功条件
        if info.get(self.success_condition, False) or (done and info.get('success', False)):
            reward = self.positive_reward
        # 检查失败条件  
        elif info.get(self.failure_condition, False) or (done and info.get('failure', False)):
            reward = self.negative_reward
        else:
            reward = self.neutral_reward
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class ThresholdReward(RewardFunction):
    """阈值奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.thresholds = config.parameters.get('thresholds', [0.0])
        self.rewards = config.parameters.get('rewards', [0.0, 1.0])
        self.target_feature = config.parameters.get('target_feature', 0)
        self.comparison = config.parameters.get('comparison', 'greater')  # 'greater' or 'less'
        
        # 确保奖励数量比阈值多1
        if len(self.rewards) != len(self.thresholds) + 1:
            raise ValueError("rewards数量应该比thresholds多1")
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        阈值奖励: 根据特征值与阈值的比较返回不同奖励
        """
        if isinstance(self.target_feature, int):
            feature_value = next_state[self.target_feature] if len(next_state) > self.target_feature else 0.0
        elif isinstance(self.target_feature, str) and info:
            feature_value = info.get(self.target_feature, 0.0)
        else:
            feature_value = np.sum(next_state)
        
        # 根据阈值确定奖励
        reward_index = 0
        for i, threshold in enumerate(self.thresholds):
            if self.comparison == 'greater':
                if feature_value > threshold:
                    reward_index = i + 1
                else:
                    break
            else:  # 'less'
                if feature_value < threshold:
                    reward_index = i + 1
                else:
                    break
        
        reward = self.rewards[reward_index]
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class GaussianReward(RewardFunction):
    """高斯奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.target_value = config.parameters.get('target_value', 0.0)
        self.sigma = config.parameters.get('sigma', 1.0)
        self.amplitude = config.parameters.get('amplitude', 1.0)
        self.target_feature = config.parameters.get('target_feature', 0)
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        高斯奖励: R = amplitude * exp(-0.5 * ((feature - target) / sigma)^2)
        """
        if isinstance(self.target_feature, int):
            feature_value = next_state[self.target_feature] if len(next_state) > self.target_feature else 0.0
        elif isinstance(self.target_feature, str) and info:
            feature_value = info.get(self.target_feature, 0.0)
        else:
            feature_value = np.sum(next_state)
            
        # 计算高斯奖励
        deviation = feature_value - self.target_value
        reward = self.amplitude * math.exp(-0.5 * (deviation / self.sigma) ** 2)
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class DistanceBasedReward(RewardFunction):
    """基于距离的奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.target_position = np.array(config.parameters.get('target_position', [0.0, 0.0]))
        self.distance_type = config.parameters.get('distance_type', 'euclidean')  # 'euclidean', 'manhattan'
        self.reward_type = config.parameters.get('reward_type', 'negative_distance')  # 'negative_distance', 'inverse_distance'
        self.max_distance = config.parameters.get('max_distance', 10.0)
        self.position_indices = config.parameters.get('position_indices', [0, 1])  # 状态中位置的索引
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        基于距离的奖励
        """
        # 提取当前位置
        if info and 'position' in info:
            current_position = np.array(info['position'])
        else:
            current_position = next_state[self.position_indices]
        
        # 计算距离
        if self.distance_type == 'euclidean':
            distance = np.linalg.norm(current_position - self.target_position)
        elif self.distance_type == 'manhattan':
            distance = np.sum(np.abs(current_position - self.target_position))
        else:
            distance = np.linalg.norm(current_position - self.target_position)
        
        # 计算奖励
        if self.reward_type == 'negative_distance':
            reward = -distance
        elif self.reward_type == 'inverse_distance':
            reward = 1.0 / (1.0 + distance)
        elif self.reward_type == 'progress':
            # 基于进度的奖励（需要上一步的距离）
            if hasattr(self, 'previous_distance'):
                progress = self.previous_distance - distance
                reward = progress
            else:
                reward = 0.0
            self.previous_distance = distance
        else:
            reward = -distance
        
        # 归一化到合理范围
        if self.reward_type == 'negative_distance':
            reward = reward / self.max_distance  # 归一化到[-1, 0]
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class SparseReward(RewardFunction):
    """稀疏奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.goal_reward = config.parameters.get('goal_reward', 1.0)
        self.step_penalty = config.parameters.get('step_penalty', -0.01)
        self.success_condition = config.parameters.get('success_condition', 'goal_reached')
        self.max_steps = config.parameters.get('max_steps', 1000)
        self.current_step = 0
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        稀疏奖励: 只在特定条件下给予奖励，其他时候给予小的负奖励
        """
        self.current_step += 1
        
        if info is None:
            info = {}
        
        # 检查是否达成目标
        if info.get(self.success_condition, False) or (done and info.get('success', False)):
            reward = self.goal_reward
        # 超时惩罚
        elif self.current_step >= self.max_steps or (done and not info.get('success', False)):
            reward = self.step_penalty * 10  # 更大的惩罚
        else:
            # 每步小惩罚，鼓励快速完成
            reward = self.step_penalty
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def on_episode_start(self):
        """回合开始时重置步数"""
        super().on_episode_start()
        self.current_step = 0
