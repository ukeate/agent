"""
复合奖励函数实现

实现多个奖励函数的组合方式
"""

import numpy as np
from typing import Dict, List, Any, Union
from .base import RewardFunction, RewardConfig, RewardType


class WeightedReward(RewardFunction):
    """加权奖励函数"""
    
    def __init__(self, config: RewardConfig, reward_functions: List[RewardFunction], weights: List[float]):
        super().__init__(config)
        self.reward_functions = reward_functions
        self.weights = weights
        
        if len(reward_functions) != len(weights):
            raise ValueError("reward_functions和weights长度必须相同")
        
        # 归一化权重
        total_weight = sum(abs(w) for w in weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in weights]
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        加权奖励: R = Σ(weight_i * reward_i)
        """
        total_reward = 0.0
        individual_rewards = []
        
        for reward_func, weight in zip(self.reward_functions, self.weights):
            individual_reward = reward_func.compute_reward(state, action, next_state, done, info)
            individual_rewards.append(individual_reward)
            total_reward += weight * individual_reward
        
        # 存储各组件奖励用于分析
        if info is not None:
            info['component_rewards'] = individual_rewards
            info['component_weights'] = self.weights
        
        # 应用归一化
        reward = self.normalize_reward(total_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def compute_multi_dimensional_reward(self, 
                                       state: np.ndarray, 
                                       action: int, 
                                       next_state: np.ndarray, 
                                       done: bool,
                                       info: Dict[str, Any] = None) -> Dict[str, float]:
        """多维度加权奖励"""
        dimension_rewards = {}
        
        for i, (reward_func, weight) in enumerate(zip(self.reward_functions, self.weights)):
            if hasattr(reward_func, 'compute_multi_dimensional_reward'):
                func_dim_rewards = reward_func.compute_multi_dimensional_reward(state, action, next_state, done, info)
            else:
                func_reward = reward_func.compute_reward(state, action, next_state, done, info)
                func_dim_rewards = {f"function_{i}": func_reward}
            
            for dim, dim_reward in func_dim_rewards.items():
                if dim not in dimension_rewards:
                    dimension_rewards[dim] = 0.0
                dimension_rewards[dim] += weight * dim_reward
        
        return dimension_rewards


class ProductReward(RewardFunction):
    """乘积奖励函数"""
    
    def __init__(self, config: RewardConfig, reward_functions: List[RewardFunction]):
        super().__init__(config)
        self.reward_functions = reward_functions
        self.epsilon = config.parameters.get('epsilon', 1e-8)  # 避免数值问题
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        乘积奖励: R = Π(reward_i + epsilon)
        """
        product_reward = 1.0
        individual_rewards = []
        
        for reward_func in self.reward_functions:
            individual_reward = reward_func.compute_reward(state, action, next_state, done, info)
            individual_rewards.append(individual_reward)
            # 添加epsilon避免乘积为0
            product_reward *= (individual_reward + self.epsilon)
        
        # 存储各组件奖励用于分析
        if info is not None:
            info['component_rewards'] = individual_rewards
        
        # 应用归一化
        reward = self.normalize_reward(product_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward


class PiecewiseReward(RewardFunction):
    """分段奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.conditions = config.parameters.get('conditions', [])
        self.reward_functions = config.parameters.get('reward_functions', [])
        self.default_reward_function = config.parameters.get('default_reward_function')
        
        if len(self.conditions) != len(self.reward_functions):
            raise ValueError("conditions和reward_functions长度必须相同")
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        分段奖励: 根据条件选择不同的奖励函数
        """
        selected_function = self.default_reward_function
        selected_condition = "default"
        
        # 检查条件并选择相应的奖励函数
        for condition, reward_func in zip(self.conditions, self.reward_functions):
            if self._evaluate_condition(condition, state, action, next_state, done, info):
                selected_function = reward_func
                selected_condition = condition.get('name', 'unknown')
                break
        
        if selected_function is None:
            reward = 0.0
        else:
            reward = selected_function.compute_reward(state, action, next_state, done, info)
        
        # 记录使用的条件
        if info is not None:
            info['selected_condition'] = selected_condition
        
        # 应用归一化
        reward = self.normalize_reward(reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                           state: np.ndarray, action: int, next_state: np.ndarray, 
                           done: bool, info: Dict[str, Any] = None) -> bool:
        """评估条件是否满足"""
        condition_type = condition.get('type', 'always')
        
        if condition_type == 'always':
            return True
        elif condition_type == 'state_threshold':
            feature_idx = condition.get('feature_idx', 0)
            threshold = condition.get('threshold', 0.0)
            comparison = condition.get('comparison', 'greater')
            
            if len(next_state) > feature_idx:
                feature_value = next_state[feature_idx]
                if comparison == 'greater':
                    return feature_value > threshold
                elif comparison == 'less':
                    return feature_value < threshold
                elif comparison == 'equal':
                    return abs(feature_value - threshold) < 1e-6
        elif condition_type == 'action_match':
            target_action = condition.get('action', 0)
            return action == target_action
        elif condition_type == 'done_state':
            return done == condition.get('done', True)
        elif condition_type == 'info_key':
            key = condition.get('key', '')
            expected_value = condition.get('value', True)
            if info and key in info:
                return info[key] == expected_value
        
        return False


class MultiObjectiveReward(RewardFunction):
    """多目标奖励函数"""
    
    def __init__(self, config: RewardConfig, objectives: Dict[str, RewardFunction], 
                 scalarization_method: str = 'weighted_sum'):
        super().__init__(config)
        self.objectives = objectives
        self.scalarization_method = scalarization_method
        self.objective_weights = config.parameters.get('objective_weights', {})
        
        # 默认权重
        if not self.objective_weights:
            weight = 1.0 / len(objectives)
            self.objective_weights = {name: weight for name in objectives.keys()}
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        多目标奖励
        """
        objective_rewards = {}
        
        # 计算各目标的奖励
        for name, objective_func in self.objectives.items():
            objective_rewards[name] = objective_func.compute_reward(state, action, next_state, done, info)
        
        # 标量化
        if self.scalarization_method == 'weighted_sum':
            total_reward = 0.0
            for name, reward in objective_rewards.items():
                weight = self.objective_weights.get(name, 0.0)
                total_reward += weight * reward
        elif self.scalarization_method == 'min_max':
            # 使用最小-最大标量化
            rewards_array = np.array(list(objective_rewards.values()))
            total_reward = np.min(rewards_array) + 0.1 * np.max(rewards_array)
        elif self.scalarization_method == 'chebyshev':
            # 切比雪夫标量化
            rewards_array = np.array(list(objective_rewards.values()))
            weights_array = np.array([self.objective_weights.get(name, 1.0) for name in objective_rewards.keys()])
            total_reward = -np.max(weights_array * np.abs(rewards_array))
        else:
            # 默认加权和
            total_reward = sum(objective_rewards.values()) / len(objective_rewards)
        
        # 存储目标奖励用于分析
        if info is not None:
            info['objective_rewards'] = objective_rewards
            info['scalarization_method'] = self.scalarization_method
        
        # 应用归一化
        reward = self.normalize_reward(total_reward)
        
        # 更新指标
        dimension_rewards = objective_rewards
        self.update_metrics(reward, dimension_rewards)
        
        return reward
    
    def compute_multi_dimensional_reward(self, 
                                       state: np.ndarray, 
                                       action: int, 
                                       next_state: np.ndarray, 
                                       done: bool,
                                       info: Dict[str, Any] = None) -> Dict[str, float]:
        """返回各目标的奖励"""
        dimension_rewards = {}
        
        for name, objective_func in self.objectives.items():
            dimension_rewards[name] = objective_func.compute_reward(state, action, next_state, done, info)
        
        return dimension_rewards


class HierarchicalReward(RewardFunction):
    """层次化奖励函数"""
    
    def __init__(self, config: RewardConfig, hierarchy_levels: List[List[RewardFunction]]):
        super().__init__(config)
        self.hierarchy_levels = hierarchy_levels
        self.level_weights = config.parameters.get('level_weights', [1.0] * len(hierarchy_levels))
        self.combination_method = config.parameters.get('combination_method', 'weighted_sum')
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        层次化奖励: 在不同层次计算奖励然后合并
        """
        level_rewards = []
        
        for level_idx, level_functions in enumerate(self.hierarchy_levels):
            level_reward = 0.0
            
            if self.combination_method == 'weighted_sum':
                # 每个层次内部用平均值
                for func in level_functions:
                    level_reward += func.compute_reward(state, action, next_state, done, info)
                level_reward /= len(level_functions) if level_functions else 1.0
            elif self.combination_method == 'max':
                # 每个层次内部取最大值
                level_rewards_temp = []
                for func in level_functions:
                    level_rewards_temp.append(func.compute_reward(state, action, next_state, done, info))
                level_reward = max(level_rewards_temp) if level_rewards_temp else 0.0
            
            level_rewards.append(level_reward)
        
        # 跨层次组合
        total_reward = 0.0
        for level_reward, weight in zip(level_rewards, self.level_weights):
            total_reward += weight * level_reward
        
        # 存储层次奖励用于分析
        if info is not None:
            info['level_rewards'] = level_rewards
            info['level_weights'] = self.level_weights
        
        # 应用归一化
        reward = self.normalize_reward(total_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward