"""
奖励函数接口和实现

定义强化学习中的奖励函数接口和常用奖励函数实现。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math

from ..qlearning.base import AgentState


class RewardType(Enum):
    """奖励类型枚举"""
    SPARSE = "sparse"
    DENSE = "dense"
    SHAPED = "shaped"
    POTENTIAL = "potential"
    COMPOSITE = "composite"


@dataclass
class RewardComponent:
    """奖励组件"""
    name: str
    weight: float
    reward_function: Callable[[AgentState, Any, AgentState, Dict[str, Any]], float]
    description: str = ""
    enabled: bool = True


class BaseRewardFunction(ABC):
    """奖励函数基类"""
    
    def __init__(self, reward_type: RewardType, config: Dict[str, Any]):
        self.reward_type = reward_type
        self.config = config
        self.cumulative_reward = 0.0
        self.episode_rewards: List[float] = []
        self.reward_history: List[float] = []
        
    @abstractmethod
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        """计算奖励值"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置奖励函数状态（新回合开始时调用）"""
        pass
    
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        return {
            "type": self.reward_type.value,
            "config": self.config,
            "cumulative_reward": self.cumulative_reward,
            "episode_count": len(self.episode_rewards),
            "average_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "reward_history_length": len(self.reward_history)
        }
    
    def update_history(self, reward: float, episode_end: bool = False):
        """更新奖励历史"""
        self.cumulative_reward += reward
        self.reward_history.append(reward)
        
        if episode_end:
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0.0


class SparseRewardFunction(BaseRewardFunction):
    """稀疏奖励函数 - 只在特定条件下给予奖励"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(RewardType.SPARSE, config)
        self.success_reward = config.get("success_reward", 1.0)
        self.failure_penalty = config.get("failure_penalty", -1.0)
        self.step_penalty = config.get("step_penalty", 0.0)
        self.success_conditions = config.get("success_conditions", [])
        self.failure_conditions = config.get("failure_conditions", [])
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        reward = self.step_penalty
        
        # 检查成功条件
        for condition in self.success_conditions:
            if self._check_condition(next_state, info, condition):
                reward += self.success_reward
                break
        
        # 检查失败条件
        for condition in self.failure_conditions:
            if self._check_condition(next_state, info, condition):
                reward += self.failure_penalty
                break
        
        self.update_history(reward, info.get("episode_end", False))
        return reward
    
    def reset(self):
        """重置状态"""
        pass
    
    def _check_condition(self, state: AgentState, info: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        condition_type = condition.get("type", "feature")
        
        if condition_type == "feature":
            feature_name = condition.get("feature")
            if feature_name not in state.features:
                return False
            
            value = state.features[feature_name]
            operator = condition.get("operator", "==")
            threshold = condition.get("threshold")
            
            return self._apply_operator(value, operator, threshold)
        
        elif condition_type == "info":
            info_key = condition.get("key")
            if info_key not in info:
                return False
            
            value = info[info_key]
            expected = condition.get("value")
            return value == expected
        
        elif condition_type == "combined":
            # 组合条件
            sub_conditions = condition.get("conditions", [])
            operator = condition.get("logic_operator", "and")
            
            results = [self._check_condition(state, info, cond) for cond in sub_conditions]
            
            if operator == "and":
                return all(results)
            elif operator == "or":
                return any(results)
            elif operator == "not":
                return not any(results)
        
        return False
    
    def _apply_operator(self, value: float, operator: str, threshold: float) -> bool:
        """应用比较操作符"""
        if operator == "==":
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        return False


class DenseRewardFunction(BaseRewardFunction):
    """密集奖励函数 - 每步都提供奖励信号"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(RewardType.DENSE, config)
        self.step_penalty = config.get("step_penalty", -0.01)
        self.goal_reward = config.get("goal_reward", 1.0)
        self.distance_reward_scale = config.get("distance_reward_scale", 0.1)
        self.goal_positions = config.get("goal_positions", [(0, 0)])
        self.distance_metric = config.get("distance_metric", "manhattan")
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        reward = self.step_penalty
        
        # 检查是否达到目标
        if self._is_at_goal(next_state):
            reward += self.goal_reward
        
        # 计算基于距离的奖励
        if self.distance_reward_scale > 0:
            current_distance = self._calculate_min_distance(state)
            next_distance = self._calculate_min_distance(next_state)
            
            # 距离减少时给予奖励，距离增加时给予惩罚
            distance_reward = (current_distance - next_distance) * self.distance_reward_scale
            reward += distance_reward
        
        self.update_history(reward, info.get("episode_end", False))
        return reward
    
    def reset(self):
        """重置状态"""
        pass
    
    def _is_at_goal(self, state: AgentState) -> bool:
        """检查是否在目标位置"""
        current_pos = self._extract_position(state)
        if current_pos is None:
            return False
        
        for goal_pos in self.goal_positions:
            if self._calculate_distance(current_pos, goal_pos) < 0.1:
                return True
        return False
    
    def _calculate_min_distance(self, state: AgentState) -> float:
        """计算到最近目标的距离"""
        current_pos = self._extract_position(state)
        if current_pos is None:
            return 0.0
        
        min_distance = float('inf')
        for goal_pos in self.goal_positions:
            distance = self._calculate_distance(current_pos, goal_pos)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _extract_position(self, state: AgentState) -> Optional[Tuple[float, float]]:
        """从状态中提取位置信息"""
        x = state.features.get("x")
        y = state.features.get("y")
        
        if x is not None and y is not None:
            return (float(x), float(y))
        return None
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间距离"""
        if self.distance_metric == "manhattan":
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        elif self.distance_metric == "euclidean":
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        elif self.distance_metric == "chebyshev":
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        else:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class ShapedRewardFunction(BaseRewardFunction):
    """奖励塑形函数 - 结合多种奖励信号引导学习"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(RewardType.SHAPED, config)
        self.base_reward_function = self._create_base_reward(config.get("base_reward", {}))
        self.exploration_bonus = config.get("exploration_bonus", 0.01)
        self.progress_bonus = config.get("progress_bonus", 0.05)
        self.consistency_bonus = config.get("consistency_bonus", 0.02)
        
        # 探索状态跟踪
        self.visited_states: set = set()
        self.state_visit_counts: Dict[str, int] = {}
        
        # 进度跟踪
        self.best_performance = float('-inf')
        self.last_action = None
        self.action_consistency_count = 0
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        # 基础奖励
        base_reward = self.base_reward_function.calculate_reward(state, action, next_state, info)
        
        # 探索奖励
        exploration_reward = self._calculate_exploration_reward(next_state)
        
        # 进度奖励  
        progress_reward = self._calculate_progress_reward(base_reward)
        
        # 一致性奖励
        consistency_reward = self._calculate_consistency_reward(action)
        
        total_reward = base_reward + exploration_reward + progress_reward + consistency_reward
        
        self.update_history(total_reward, info.get("episode_end", False))
        return total_reward
    
    def reset(self):
        """重置状态"""
        self.base_reward_function.reset()
        self.best_performance = float('-inf')
        self.last_action = None
        self.action_consistency_count = 0
        # 注意：不清理visited_states，保持跨回合的探索记录
    
    def _create_base_reward(self, config: Dict[str, Any]) -> BaseRewardFunction:
        """创建基础奖励函数"""
        reward_type = config.get("type", "dense")
        if reward_type == "sparse":
            return SparseRewardFunction(config)
        else:
            return DenseRewardFunction(config)
    
    def _calculate_exploration_reward(self, state: AgentState) -> float:
        """计算探索奖励"""
        state_key = self._state_to_key(state)
        
        if state_key not in self.visited_states:
            self.visited_states.add(state_key)
            self.state_visit_counts[state_key] = 1
            return self.exploration_bonus
        else:
            # 减少重复访问的奖励
            self.state_visit_counts[state_key] += 1
            visit_count = self.state_visit_counts[state_key]
            return self.exploration_bonus / visit_count
    
    def _calculate_progress_reward(self, current_reward: float) -> float:
        """计算进度奖励"""
        if current_reward > self.best_performance:
            self.best_performance = current_reward
            return self.progress_bonus
        return 0.0
    
    def _calculate_consistency_reward(self, action: Any) -> float:
        """计算动作一致性奖励"""
        if self.last_action is not None and action == self.last_action:
            self.action_consistency_count += 1
            # 适度的一致性奖励，防止过度震荡
            consistency_reward = self.consistency_bonus / (1 + self.action_consistency_count * 0.1)
        else:
            self.action_consistency_count = 0
            consistency_reward = 0.0
        
        self.last_action = action
        return consistency_reward
    
    def _state_to_key(self, state: AgentState) -> str:
        """将状态转换为字符串键"""
        # 对连续状态进行离散化
        discrete_features = {}
        for name, value in state.features.items():
            if isinstance(value, (int, float)):
                # 离散化到网格
                discrete_features[name] = round(float(value), 2)
            else:
                discrete_features[name] = value
        
        return str(sorted(discrete_features.items()))


class PotentialBasedRewardFunction(BaseRewardFunction):
    """基于势函数的奖励函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(RewardType.POTENTIAL, config)
        self.step_penalty = config.get("step_penalty", -0.01)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.potential_function = self._create_potential_function(config.get("potential_config", {}))
        self.last_potential = None
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        # 基础步数惩罚
        reward = self.step_penalty
        
        # 势函数奖励: R = γ*Φ(s') - Φ(s)
        current_potential = self.potential_function(state)
        next_potential = self.potential_function(next_state)
        
        if self.last_potential is not None:
            potential_reward = self.discount_factor * next_potential - current_potential
            reward += potential_reward
        
        self.last_potential = next_potential
        
        self.update_history(reward, info.get("episode_end", False))
        return reward
    
    def reset(self):
        """重置状态"""
        self.last_potential = None
    
    def _create_potential_function(self, config: Dict[str, Any]) -> Callable[[AgentState], float]:
        """创建势函数"""
        potential_type = config.get("type", "distance_based")
        
        if potential_type == "distance_based":
            goal_positions = config.get("goal_positions", [(0, 0)])
            
            def distance_potential(state: AgentState) -> float:
                x = state.features.get("x", 0)
                y = state.features.get("y", 0)
                current_pos = (x, y)
                
                # 返回到最近目标的负距离作为势函数值
                min_distance = float('inf')
                for goal_pos in goal_positions:
                    distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
                    min_distance = min(min_distance, distance)
                
                return -min_distance
            
            return distance_potential
        
        elif potential_type == "custom":
            # 自定义势函数（通过配置定义）
            formula = config.get("formula", "0")
            
            def custom_potential(state: AgentState) -> float:
                # 简化的自定义势函数评估
                # 实际应用中可以使用更复杂的表达式解析器
                try:
                    # 将状态特征作为变量
                    local_vars = state.features.copy()
                    return float(eval(formula, {"__builtins__": {}}, local_vars))
                except:
                    return 0.0
            
            return custom_potential
        
        else:
            # 默认零势函数
            return lambda state: 0.0


class CompositeRewardFunction(BaseRewardFunction):
    """组合奖励函数 - 多个奖励组件的加权组合"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(RewardType.COMPOSITE, config)
        self.components: List[RewardComponent] = []
        self.normalization = config.get("normalization", "none")
        self.reward_bounds = config.get("reward_bounds", (-1.0, 1.0))
        
        # 创建奖励组件
        for comp_config in config.get("components", []):
            self._add_reward_component(comp_config)
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        total_reward = 0.0
        component_rewards = {}
        
        # 计算各组件奖励
        for component in self.components:
            if component.enabled:
                comp_reward = component.reward_function(state, action, next_state, info)
                weighted_reward = comp_reward * component.weight
                total_reward += weighted_reward
                component_rewards[component.name] = comp_reward
        
        # 奖励归一化
        if self.normalization == "clip":
            total_reward = np.clip(total_reward, self.reward_bounds[0], self.reward_bounds[1])
        elif self.normalization == "tanh":
            total_reward = np.tanh(total_reward)
        
        # 记录组件奖励到info中
        info["component_rewards"] = component_rewards
        
        self.update_history(total_reward, info.get("episode_end", False))
        return total_reward
    
    def reset(self):
        """重置状态"""
        # 重置有状态的组件
        for component in self.components:
            if hasattr(component.reward_function, 'reset'):
                component.reward_function.reset()
    
    def _add_reward_component(self, config: Dict[str, Any]):
        """添加奖励组件"""
        comp_type = config.get("type", "sparse")
        name = config.get("name", f"component_{len(self.components)}")
        weight = config.get("weight", 1.0)
        
        if comp_type == "sparse":
            reward_func = SparseRewardFunction(config)
        elif comp_type == "dense":
            reward_func = DenseRewardFunction(config)
        elif comp_type == "shaped":
            reward_func = ShapedRewardFunction(config)
        elif comp_type == "potential":
            reward_func = PotentialBasedRewardFunction(config)
        else:
            raise ValueError(f"不支持的奖励组件类型: {comp_type}")
        
        component = RewardComponent(
            name=name,
            weight=weight,
            reward_function=lambda s, a, ns, i, rf=reward_func: rf.calculate_reward(s, a, ns, i),
            description=config.get("description", f"{comp_type}奖励组件"),
            enabled=config.get("enabled", True)
        )
        
        self.components.append(component)
    
    def enable_component(self, name: str):
        """启用奖励组件"""
        for component in self.components:
            if component.name == name:
                component.enabled = True
                break
    
    def disable_component(self, name: str):
        """禁用奖励组件"""
        for component in self.components:
            if component.name == name:
                component.enabled = False
                break
    
    def set_component_weight(self, name: str, weight: float):
        """设置奖励组件权重"""
        for component in self.components:
            if component.name == name:
                component.weight = weight
                break


class RewardFunctionFactory:
    """奖励函数工厂类"""
    
    @staticmethod
    def create_reward_function(config: Dict[str, Any]) -> BaseRewardFunction:
        """根据配置创建奖励函数"""
        reward_type = config.get("type", "sparse").lower()
        
        if reward_type == "sparse":
            return SparseRewardFunction(config)
        elif reward_type == "dense":
            return DenseRewardFunction(config)
        elif reward_type == "shaped":
            return ShapedRewardFunction(config)
        elif reward_type == "potential":
            return PotentialBasedRewardFunction(config)
        elif reward_type == "composite":
            return CompositeRewardFunction(config)
        else:
            raise ValueError(f"不支持的奖励函数类型: {reward_type}")
    
    @staticmethod
    def create_simple_sparse_reward(success_reward: float = 1.0, 
                                  step_penalty: float = -0.01) -> SparseRewardFunction:
        """创建简单的稀疏奖励函数"""
        config = {
            "success_reward": success_reward,
            "step_penalty": step_penalty,
            "success_conditions": [{"type": "info", "key": "at_goal", "value": True}]
        }
        return SparseRewardFunction(config)
    
    @staticmethod
    def create_simple_dense_reward(goal_reward: float = 1.0, 
                                 step_penalty: float = -0.01,
                                 distance_scale: float = 0.1) -> DenseRewardFunction:
        """创建简单的密集奖励函数"""
        config = {
            "goal_reward": goal_reward,
            "step_penalty": step_penalty,
            "distance_reward_scale": distance_scale
        }
        return DenseRewardFunction(config)