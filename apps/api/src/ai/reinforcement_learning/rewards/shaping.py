"""
奖励塑形技术实现

实现各种奖励塑形方法来改善学习效率
"""

import numpy as np
import ast
import math
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import deque
from .base import RewardFunction, RewardConfig

class PotentialBasedShaping(RewardFunction):
    """基于势能的奖励塑形"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction, 
                 potential_function: Callable[[np.ndarray], float]):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.potential_function = potential_function
        self.gamma = config.parameters.get('gamma', 0.99)  # 折扣因子
        self.shaping_weight = config.parameters.get('shaping_weight', 1.0)
        
        # 存储上一个状态的势能值
        self.previous_potential = None
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        基于势能的奖励塑形: R' = R + γ*Φ(s') - Φ(s)
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 计算当前状态和下一状态的势能
        current_potential = self.potential_function(state)
        next_potential = self.potential_function(next_state) if not done else 0.0
        
        # 计算势能差
        if self.previous_potential is not None:
            potential_diff = self.gamma * current_potential - self.previous_potential
        else:
            potential_diff = 0.0
        
        # 应用奖励塑形
        shaped_reward = base_reward + self.shaping_weight * potential_diff
        
        # 更新上一个势能值
        self.previous_potential = current_potential if not done else None
        
        # 记录塑形信息
        if info is not None:
            info['base_reward'] = base_reward
            info['potential_diff'] = potential_diff
            info['current_potential'] = current_potential
            info['next_potential'] = next_potential
        
        # 应用归一化
        reward = self.normalize_reward(shaped_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def on_episode_start(self):
        """回合开始时重置势能"""
        super().on_episode_start()
        self.previous_potential = None

class DifferenceReward(RewardFunction):
    """差分奖励函数"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.difference_metric = config.parameters.get('difference_metric', 'improvement')
        self.baseline_window = config.parameters.get('baseline_window', 100)
        self.smoothing_factor = config.parameters.get('smoothing_factor', 0.1)
        
        # 基线性能追踪
        self.baseline_rewards = deque(maxlen=self.baseline_window)
        self.current_baseline = 0.0
        self.improvement_history = []
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        差分奖励: 强调相对于基线的改进
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 更新基线
        self.baseline_rewards.append(base_reward)
        if len(self.baseline_rewards) > 10:
            new_baseline = np.mean(self.baseline_rewards)
            self.current_baseline = (1 - self.smoothing_factor) * self.current_baseline + \
                                   self.smoothing_factor * new_baseline
        
        # 计算差分奖励
        if self.difference_metric == 'improvement':
            difference_reward = base_reward - self.current_baseline
        elif self.difference_metric == 'relative_improvement':
            if abs(self.current_baseline) > 1e-6:
                difference_reward = (base_reward - self.current_baseline) / abs(self.current_baseline)
            else:
                difference_reward = base_reward
        elif self.difference_metric == 'percentile_rank':
            # 基于百分位数的奖励
            if len(self.baseline_rewards) > 0:
                sorted_rewards = sorted(self.baseline_rewards)
                rank = np.searchsorted(sorted_rewards, base_reward)
                percentile = rank / len(sorted_rewards)
                difference_reward = percentile - 0.5  # 中心化到[-0.5, 0.5]
            else:
                difference_reward = 0.0
        else:
            difference_reward = base_reward - self.current_baseline
        
        # 记录改进历史
        self.improvement_history.append(difference_reward)
        
        # 记录差分信息
        if info is not None:
            info['base_reward'] = base_reward
            info['baseline'] = self.current_baseline
            info['difference_reward'] = difference_reward
            info['improvement'] = base_reward - self.current_baseline
        
        # 应用归一化
        reward = self.normalize_reward(difference_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward

class CuriosityReward(RewardFunction):
    """基于好奇心的内在奖励"""
    
    def __init__(self, config: RewardConfig, base_reward_function: Optional[RewardFunction] = None):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.curiosity_weight = config.parameters.get('curiosity_weight', 0.1)
        self.novelty_threshold = config.parameters.get('novelty_threshold', 0.1)
        self.state_memory_size = config.parameters.get('state_memory_size', 1000)
        
        # 状态访问记录
        self.state_visit_counts = {}
        self.state_history = deque(maxlen=self.state_memory_size)
        self.novelty_scores = []
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        好奇心驱动奖励: 结合外在奖励和内在好奇心奖励
        """
        # 计算外在奖励
        extrinsic_reward = 0.0
        if self.base_reward_function is not None:
            extrinsic_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 计算内在好奇心奖励
        intrinsic_reward = self._compute_intrinsic_reward(next_state)
        
        # 组合奖励
        total_reward = extrinsic_reward + self.curiosity_weight * intrinsic_reward
        
        # 记录好奇心信息
        if info is not None:
            info['extrinsic_reward'] = extrinsic_reward
            info['intrinsic_reward'] = intrinsic_reward
            info['curiosity_weight'] = self.curiosity_weight
        
        # 应用归一化
        reward = self.normalize_reward(total_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def _compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """计算内在好奇心奖励"""
        # 将状态转换为可哈希的形式
        state_key = self._state_to_key(state)
        
        # 更新访问计数
        if state_key not in self.state_visit_counts:
            self.state_visit_counts[state_key] = 0
        self.state_visit_counts[state_key] += 1
        
        # 添加到历史
        self.state_history.append(state_key)
        
        # 计算新颖性分数（基于访问频率的倒数）
        visit_count = self.state_visit_counts[state_key]
        novelty_score = 1.0 / math.sqrt(visit_count)
        
        # 计算与最近状态的相似性
        similarity_bonus = self._compute_similarity_bonus(state)
        
        # 组合内在奖励
        intrinsic_reward = novelty_score + similarity_bonus
        
        # 记录新颖性分数
        self.novelty_scores.append(novelty_score)
        
        return intrinsic_reward
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """将状态转换为可哈希的键"""
        # 简单的状态量化
        quantized = np.round(state, 2)  # 保留2位小数
        return str(quantized.tolist())
    
    def _compute_similarity_bonus(self, current_state: np.ndarray) -> float:
        """计算与历史状态的相似性奖励"""
        if len(self.state_history) < 2:
            return 0.0
        
        # 计算与最近几个状态的平均距离
        recent_states = list(self.state_history)[-5:]  # 最近5个状态
        distances = []
        
        for state_key in recent_states:
            try:
                # 尝试恢复状态向量（简化实现）
                state_list = ast.literal_eval(state_key)
                historical_state = np.array(state_list)
                distance = np.linalg.norm(current_state - historical_state)
                distances.append(distance)
            except:
                continue
        
        if distances:
            avg_distance = np.mean(distances)
            # 距离越大，相似性奖励越小
            similarity_bonus = max(0.0, self.novelty_threshold - avg_distance) / self.novelty_threshold
        else:
            similarity_bonus = 0.0
        
        return similarity_bonus * 0.1  # 缩放因子

class DenseRewardShaping(RewardFunction):
    """密集奖励塑形"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction, 
                 intermediate_goals: List[Callable[[np.ndarray, int, np.ndarray], float]]):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.intermediate_goals = intermediate_goals
        self.goal_weights = config.parameters.get('goal_weights', [1.0] * len(intermediate_goals))
        self.shaping_decay = config.parameters.get('shaping_decay', 1.0)  # 塑形强度衰减
        
        if len(self.goal_weights) != len(intermediate_goals):
            self.goal_weights = [1.0] * len(intermediate_goals)
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        密集奖励塑形: 添加中间目标的奖励信号
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 计算中间目标奖励
        intermediate_rewards = []
        total_intermediate_reward = 0.0
        
        for goal_func, weight in zip(self.intermediate_goals, self.goal_weights):
            goal_reward = goal_func(state, action, next_state)
            intermediate_rewards.append(goal_reward)
            total_intermediate_reward += weight * goal_reward
        
        # 应用衰减
        decayed_intermediate_reward = total_intermediate_reward * self.shaping_decay
        
        # 组合奖励
        shaped_reward = base_reward + decayed_intermediate_reward
        
        # 记录塑形信息
        if info is not None:
            info['base_reward'] = base_reward
            info['intermediate_rewards'] = intermediate_rewards
            info['total_intermediate_reward'] = total_intermediate_reward
            info['shaping_decay'] = self.shaping_decay
        
        # 应用归一化
        reward = self.normalize_reward(shaped_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def update_shaping_decay(self, new_decay: float):
        """更新塑形衰减因子"""
        self.shaping_decay = max(0.0, min(1.0, new_decay))

class HindsightExperienceReward(RewardFunction):
    """后见之明经验奖励"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.goal_substitution_prob = config.parameters.get('goal_substitution_prob', 0.8)
        self.max_goals = config.parameters.get('max_goals', 4)
        
        # 存储轨迹信息
        self.trajectory = []
        self.achieved_goals = []
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        HER风格的奖励计算
        """
        # 存储轨迹
        self.trajectory.append({
            'state': state.copy(),
            'action': action,
            'next_state': next_state.copy(),
            'done': done,
            'info': info.copy() if info else {}
        })
        
        # 计算原始奖励
        original_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 如果回合结束，生成后见之明经验
        if done:
            her_rewards = self._generate_hindsight_rewards()
            if her_rewards:
                # 使用后见之明奖励的平均值
                hindsight_reward = np.mean(her_rewards)
                combined_reward = (original_reward + hindsight_reward) / 2
            else:
                combined_reward = original_reward
            
            # 清空轨迹
            self.trajectory.clear()
        else:
            combined_reward = original_reward
        
        # 应用归一化
        reward = self.normalize_reward(combined_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def _generate_hindsight_rewards(self) -> List[float]:
        """生成后见之明经验的奖励"""
        if len(self.trajectory) < 2:
            return []
        
        her_rewards = []
        
        # 选择一些状态作为虚拟目标
        num_goals = min(self.max_goals, len(self.trajectory))
        goal_indices = np.random.choice(len(self.trajectory), num_goals, replace=False)
        
        for goal_idx in goal_indices:
            goal_state = self.trajectory[goal_idx]['next_state']
            
            # 计算达到这个虚拟目标的奖励
            for step in self.trajectory:
                # 简化的目标达成检查
                distance_to_goal = np.linalg.norm(step['next_state'] - goal_state)
                if distance_to_goal < 0.1:  # 阈值可配置
                    her_rewards.append(1.0)  # 成功达到虚拟目标
                else:
                    her_rewards.append(-0.01)  # 小的负奖励
        
        return her_rewards

class RewardClipping(RewardFunction):
    """奖励裁剪和缩放"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.clip_min = config.parameters.get('clip_min', -1.0)
        self.clip_max = config.parameters.get('clip_max', 1.0)
        self.scaling_factor = config.parameters.get('scaling_factor', 1.0)
        self.clip_type = config.parameters.get('clip_type', 'hard')  # 'hard', 'soft', 'tanh'
        
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        奖励裁剪和缩放
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 应用缩放
        scaled_reward = base_reward * self.scaling_factor
        
        # 应用裁剪
        if self.clip_type == 'hard':
            clipped_reward = np.clip(scaled_reward, self.clip_min, self.clip_max)
        elif self.clip_type == 'soft':
            # 软裁剪：使用sigmoid函数
            if scaled_reward > self.clip_max:
                excess = scaled_reward - self.clip_max
                clipped_reward = self.clip_max + 0.1 * (1 / (1 + np.exp(-excess)))
            elif scaled_reward < self.clip_min:
                deficit = self.clip_min - scaled_reward
                clipped_reward = self.clip_min - 0.1 * (1 / (1 + np.exp(-deficit)))
            else:
                clipped_reward = scaled_reward
        elif self.clip_type == 'tanh':
            # 使用tanh进行软裁剪
            normalized = scaled_reward / max(abs(self.clip_min), abs(self.clip_max))
            clipped_reward = max(abs(self.clip_min), abs(self.clip_max)) * np.tanh(normalized)
        else:
            clipped_reward = np.clip(scaled_reward, self.clip_min, self.clip_max)
        
        # 记录裁剪信息
        if info is not None:
            info['base_reward'] = base_reward
            info['scaled_reward'] = scaled_reward
            info['clipped_reward'] = clipped_reward
            info['was_clipped'] = abs(clipped_reward - scaled_reward) > 1e-6
        
        # 应用归一化
        reward = self.normalize_reward(clipped_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
