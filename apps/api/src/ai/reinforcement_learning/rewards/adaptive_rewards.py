"""
自适应奖励函数实现

实现能够根据学习进度和表现动态调整的奖励函数
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Callable
from collections import deque
from .base import RewardFunction, RewardConfig

class AdaptiveReward(RewardFunction):
    """自适应奖励函数"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.adaptation_rate = config.parameters.get('adaptation_rate', 0.01)
        self.adaptation_window = config.parameters.get('adaptation_window', 1000)
        self.min_scale = config.parameters.get('min_scale', 0.1)
        self.max_scale = config.parameters.get('max_scale', 10.0)
        
        # 自适应参数
        self.reward_scale = 1.0
        self.performance_history = deque(maxlen=self.adaptation_window)
        self.adaptation_history = []
        
        # 性能指标
        self.target_performance = config.parameters.get('target_performance', 0.0)
        self.performance_metric = config.parameters.get('performance_metric', 'mean_reward')
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        自适应奖励计算
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 应用自适应缩放
        adapted_reward = base_reward * self.reward_scale
        
        # 更新性能历史
        self.performance_history.append(base_reward)
        
        # 定期调整奖励缩放
        if len(self.performance_history) >= self.adaptation_window and self.step_count % 100 == 0:
            self._adapt_reward_scale()
        
        # 应用归一化
        reward = self.normalize_reward(adapted_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def _adapt_reward_scale(self):
        """调整奖励缩放因子"""
        if len(self.performance_history) == 0:
            return
        
        # 计算当前性能
        if self.performance_metric == 'mean_reward':
            current_performance = np.mean(self.performance_history)
        elif self.performance_metric == 'std_reward':
            current_performance = np.std(self.performance_history)
        elif self.performance_metric == 'max_reward':
            current_performance = np.max(self.performance_history)
        elif self.performance_metric == 'min_reward':
            current_performance = np.min(self.performance_history)
        else:
            current_performance = np.mean(self.performance_history)
        
        # 计算性能差距
        performance_gap = self.target_performance - current_performance
        
        # 调整奖励缩放
        if abs(performance_gap) > 1e-6:
            # 如果性能低于目标，增加奖励缩放；如果高于目标，减少奖励缩放
            scale_adjustment = self.adaptation_rate * performance_gap
            self.reward_scale *= (1.0 + scale_adjustment)
            
            # 限制缩放范围
            self.reward_scale = np.clip(self.reward_scale, self.min_scale, self.max_scale)
        
        # 记录适应历史
        self.adaptation_history.append({
            'step': self.step_count,
            'performance': current_performance,
            'performance_gap': performance_gap,
            'reward_scale': self.reward_scale
        })
        
        # 更新指标
        self.metrics.adaptation_history.append(self.reward_scale)

class ProgressBasedReward(RewardFunction):
    """基于进度的奖励函数"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.progress_stages = config.parameters.get('progress_stages', [0.25, 0.5, 0.75, 1.0])
        self.stage_multipliers = config.parameters.get('stage_multipliers', [2.0, 1.5, 1.0, 0.5])
        self.max_episodes = config.parameters.get('max_episodes', 10000)
        
        if len(self.progress_stages) != len(self.stage_multipliers):
            raise ValueError("progress_stages和stage_multipliers长度必须相同")
        
        # 当前进度状态
        self.current_stage = 0
        self.progress_ratio = 0.0
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        基于进度的奖励调整
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 更新进度
        self._update_progress()
        
        # 应用进度倍数
        stage_multiplier = self.stage_multipliers[self.current_stage]
        progress_reward = base_reward * stage_multiplier
        
        # 记录进度信息
        if info is not None:
            info['progress_stage'] = self.current_stage
            info['progress_ratio'] = self.progress_ratio
            info['stage_multiplier'] = stage_multiplier
        
        # 应用归一化
        reward = self.normalize_reward(progress_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def _update_progress(self):
        """更新学习进度"""
        self.progress_ratio = min(self.episode_count / self.max_episodes, 1.0)
        
        # 确定当前阶段
        for i, stage_threshold in enumerate(self.progress_stages):
            if self.progress_ratio <= stage_threshold:
                self.current_stage = i
                break
        else:
            self.current_stage = len(self.progress_stages) - 1

class CurriculumReward(RewardFunction):
    """课程学习奖励函数"""
    
    def __init__(self, config: RewardConfig, curriculum_stages: List[Dict[str, Any]]):
        super().__init__(config)
        self.curriculum_stages = curriculum_stages
        self.current_stage_idx = 0
        self.stage_success_threshold = config.parameters.get('stage_success_threshold', 0.8)
        self.stage_episode_window = config.parameters.get('stage_episode_window', 100)
        self.auto_advance = config.parameters.get('auto_advance', True)
        
        # 当前阶段的成功率追踪
        self.stage_success_history = deque(maxlen=self.stage_episode_window)
        self.stage_episodes = 0
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        课程学习奖励计算
        """
        if self.current_stage_idx >= len(self.curriculum_stages):
            self.current_stage_idx = len(self.curriculum_stages) - 1
        
        current_stage = self.curriculum_stages[self.current_stage_idx]
        
        # 获取当前阶段的奖励函数
        reward_function = current_stage.get('reward_function')
        if reward_function is None:
            reward = 0.0
        else:
            reward = reward_function.compute_reward(state, action, next_state, done, info)
        
        # 应用阶段特定的修正
        stage_modifier = current_stage.get('reward_modifier', 1.0)
        stage_reward = reward * stage_modifier
        
        # 记录阶段信息
        if info is not None:
            info['curriculum_stage'] = self.current_stage_idx
            info['stage_name'] = current_stage.get('name', f'stage_{self.current_stage_idx}')
            info['stage_modifier'] = stage_modifier
        
        # 应用归一化
        final_reward = self.normalize_reward(stage_reward)
        
        # 更新指标
        self.update_metrics(final_reward)
        
        return final_reward
    
    def on_episode_end(self, episode_reward: float):
        """回合结束时更新课程进度"""
        super().on_episode_end(episode_reward)
        
        self.stage_episodes += 1
        
        # 判断当前回合是否成功
        current_stage = self.curriculum_stages[self.current_stage_idx]
        success_threshold = current_stage.get('success_threshold', self.stage_success_threshold)
        
        episode_success = episode_reward >= success_threshold
        self.stage_success_history.append(episode_success)
        
        # 检查是否需要进入下一阶段
        if self.auto_advance and self._should_advance_stage():
            self._advance_to_next_stage()
    
    def _should_advance_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        if len(self.stage_success_history) < self.stage_episode_window:
            return False
        
        success_rate = sum(self.stage_success_history) / len(self.stage_success_history)
        return success_rate >= self.stage_success_threshold
    
    def _advance_to_next_stage(self):
        """进入下一阶段"""
        if self.current_stage_idx < len(self.curriculum_stages) - 1:
            self.current_stage_idx += 1
            self.stage_episodes = 0
            self.stage_success_history.clear()
            
            # 记录阶段切换
            stage_info = {
                'step': self.step_count,
                'episode': self.episode_count,
                'new_stage': self.current_stage_idx,
                'stage_name': self.curriculum_stages[self.current_stage_idx].get('name', f'stage_{self.current_stage_idx}')
            }
            self.metrics.adaptation_history.append(stage_info)
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        if self.current_stage_idx >= len(self.curriculum_stages):
            return {}
        
        current_stage = self.curriculum_stages[self.current_stage_idx]
        success_rate = sum(self.stage_success_history) / len(self.stage_success_history) if self.stage_success_history else 0.0
        
        return {
            'stage_index': self.current_stage_idx,
            'stage_name': current_stage.get('name', f'stage_{self.current_stage_idx}'),
            'stage_episodes': self.stage_episodes,
            'success_rate': success_rate,
            'success_threshold': self.stage_success_threshold,
            'ready_to_advance': self._should_advance_stage()
        }

class DynamicDifficultyReward(RewardFunction):
    """动态难度调整奖励函数"""
    
    def __init__(self, config: RewardConfig, base_reward_function: RewardFunction):
        super().__init__(config)
        self.base_reward_function = base_reward_function
        self.target_success_rate = config.parameters.get('target_success_rate', 0.6)
        self.difficulty_adjustment_rate = config.parameters.get('difficulty_adjustment_rate', 0.02)
        self.min_difficulty = config.parameters.get('min_difficulty', 0.1)
        self.max_difficulty = config.parameters.get('max_difficulty', 2.0)
        self.evaluation_window = config.parameters.get('evaluation_window', 200)
        
        # 动态难度参数
        self.current_difficulty = 1.0
        self.success_history = deque(maxlen=self.evaluation_window)
        self.difficulty_history = []
    
    def compute_reward(self, 
                      state: np.ndarray, 
                      action: int, 
                      next_state: np.ndarray, 
                      done: bool,
                      info: Dict[str, Any] = None) -> float:
        """
        动态难度调整奖励计算
        """
        # 计算基础奖励
        base_reward = self.base_reward_function.compute_reward(state, action, next_state, done, info)
        
        # 应用难度调整
        # 难度越高，奖励要求越严格
        difficulty_adjusted_reward = base_reward / self.current_difficulty
        
        # 定期调整难度
        if len(self.success_history) >= self.evaluation_window and self.step_count % 50 == 0:
            self._adjust_difficulty()
        
        # 记录难度信息
        if info is not None:
            info['current_difficulty'] = self.current_difficulty
            info['difficulty_adjusted_reward'] = difficulty_adjusted_reward
        
        # 应用归一化
        reward = self.normalize_reward(difficulty_adjusted_reward)
        
        # 更新指标
        self.update_metrics(reward)
        
        return reward
    
    def on_episode_end(self, episode_reward: float):
        """记录回合成功状态"""
        super().on_episode_end(episode_reward)
        
        # 简单的成功判断：正奖励为成功
        success = episode_reward > 0
        self.success_history.append(success)
    
    def _adjust_difficulty(self):
        """调整难度级别"""
        if len(self.success_history) == 0:
            return
        
        # 计算当前成功率
        current_success_rate = sum(self.success_history) / len(self.success_history)
        
        # 根据成功率与目标的差距调整难度
        success_gap = current_success_rate - self.target_success_rate
        
        if success_gap > 0.1:  # 成功率太高，增加难度
            self.current_difficulty += self.difficulty_adjustment_rate
        elif success_gap < -0.1:  # 成功率太低，降低难度
            self.current_difficulty -= self.difficulty_adjustment_rate
        
        # 限制难度范围
        self.current_difficulty = np.clip(self.current_difficulty, self.min_difficulty, self.max_difficulty)
        
        # 记录难度变化
        self.difficulty_history.append({
            'step': self.step_count,
            'episode': self.episode_count,
            'success_rate': current_success_rate,
            'difficulty': self.current_difficulty,
            'success_gap': success_gap
        })
    
    def get_difficulty_stats(self) -> Dict[str, Any]:
        """获取难度统计信息"""
        current_success_rate = sum(self.success_history) / len(self.success_history) if self.success_history else 0.0
        
        return {
            'current_difficulty': self.current_difficulty,
            'current_success_rate': current_success_rate,
            'target_success_rate': self.target_success_rate,
            'difficulty_history_length': len(self.difficulty_history),
            'min_difficulty': self.min_difficulty,
            'max_difficulty': self.max_difficulty
        }
