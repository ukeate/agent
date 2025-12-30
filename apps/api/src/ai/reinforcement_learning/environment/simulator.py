"""
环境模拟器 - 强化学习环境核心组件

实现强化学习环境的状态转移、奖励计算和智能体交互逻辑。
"""

import copy
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.utils.timezone_utils import utc_now
from ..qlearning.base import AgentState, Experience
from .state_space import StateSpace, StateSpaceFactory
from .action_space import ActionSpace, ActionSpaceFactory

class EnvironmentStatus(Enum):
    """环境状态枚举"""
    READY = "ready"
    RUNNING = "running"
    TERMINATED = "terminated"
    TRUNCATED = "truncated"
    ERROR = "error"

@dataclass
class EnvironmentInfo:
    """环境信息"""
    env_id: str
    name: str
    description: str
    version: str = "1.0"
    max_episode_steps: int = 1000
    reward_threshold: Optional[float] = None
    entry_point: Optional[str] = None
    
@dataclass
class StepResult:
    """环境步骤执行结果"""
    next_state: AgentState
    reward: float
    terminated: bool  # 环境因为任务完成而终止
    truncated: bool   # 环境因为时间限制等外部条件而截断
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def done(self) -> bool:
        """是否结束（终止或截断）"""
        return self.terminated or self.truncated

class RewardFunction(ABC):
    """奖励函数抽象基类"""
    
    @abstractmethod
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        """计算奖励值"""
        ...
    
    @abstractmethod
    def get_reward_info(self) -> Dict[str, Any]:
        """获取奖励函数信息"""
        ...

class TransitionFunction(ABC):
    """状态转移函数抽象基类"""
    
    @abstractmethod
    def transition(self, state: AgentState, action: Any) -> Tuple[AgentState, Dict[str, Any]]:
        """执行状态转移，返回新状态和额外信息"""
        ...
    
    @abstractmethod
    def is_terminal_state(self, state: AgentState) -> bool:
        """判断是否为终止状态"""
        ...

class BaseEnvironment(ABC):
    """环境基类"""
    
    def __init__(self, env_info: EnvironmentInfo, state_space: StateSpace, 
                 action_space: ActionSpace):
        self.env_info = env_info
        self.state_space = state_space
        self.action_space = action_space
        
        # 环境状态
        self.status = EnvironmentStatus.READY
        self.current_state: Optional[AgentState] = None
        self.episode_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # 历史记录
        self.episode_history: List[Experience] = []
        self.reward_history: List[float] = []
        self.step_info_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> AgentState:
        """重置环境到初始状态"""
        ...
    
    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """执行一步环境交互"""
        ...
    
    def close(self):
        """关闭环境，清理资源"""
        self.status = EnvironmentStatus.READY
        self.current_state = None
        self.episode_step = 0
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染环境（可选实现）"""
        return None
    
    def get_valid_actions(self, state: Optional[AgentState] = None) -> List[Any]:
        """获取在给定状态下的有效动作"""
        if state is None:
            state = self.current_state
        
        # 默认返回所有动作都有效
        if hasattr(self.action_space, 'n'):
            return list(range(self.action_space.n))
        else:
            return [self.action_space.sample()]
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """获取当前回合统计信息"""
        total_reward = sum(self.reward_history)
        return {
            "episode_count": self.episode_count,
            "episode_step": self.episode_step,
            "total_steps": self.total_steps,
            "total_reward": total_reward,
            "average_reward": total_reward / max(self.episode_step, 1),
            "status": self.status.value,
            "max_episode_steps": self.env_info.max_episode_steps
        }

class AgentEnvironmentSimulator(BaseEnvironment):
    """通用智能体环境模拟器"""
    
    def __init__(self, config: Dict[str, Any]):
        # 创建环境信息
        env_info = EnvironmentInfo(
            env_id=config.get("env_id", str(uuid.uuid4())),
            name=config.get("name", "Generic Environment"),
            description=config.get("description", "通用智能体环境"),
            max_episode_steps=config.get("max_episode_steps", 1000),
            reward_threshold=config.get("reward_threshold")
        )
        
        # 创建状态空间和动作空间
        state_space = StateSpaceFactory.create_from_config(config["state_space"])
        action_space = ActionSpaceFactory.create_from_config(config["action_space"])
        
        super().__init__(env_info, state_space, action_space)
        
        # 设置转移函数和奖励函数
        self.transition_function = self._create_transition_function(config.get("transition_function", {}))
        self.reward_function = self._create_reward_function(config.get("reward_function", {}))
        
        # 环境特定配置
        self.stochastic = config.get("stochastic", True)
        self.noise_level = config.get("noise_level", 0.0)
        self.initial_state_config = config.get("initial_state", {})
        
        # 随机数生成器
        self.np_random = np.random.RandomState()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> AgentState:
        """重置环境到初始状态"""
        if seed is not None:
            self.np_random.seed(seed)
            np.random.seed(seed)
        
        # 重置环境状态
        self.status = EnvironmentStatus.RUNNING
        self.episode_step = 0
        self.episode_count += 1
        self.episode_history = []
        self.reward_history = []
        self.step_info_history = []
        
        # 生成初始状态
        if self.initial_state_config:
            # 使用配置的初始状态
            self.current_state = self._create_initial_state(self.initial_state_config, options)
        else:
            # 随机采样初始状态
            self.current_state = self.state_space.sample()
        
        return self.current_state
    
    def step(self, action: Any) -> StepResult:
        """执行一步环境交互"""
        if self.status != EnvironmentStatus.RUNNING:
            raise RuntimeError(f"环境状态不正确: {self.status}")
        
        if not self.action_space.contains(action):
            raise ValueError(f"动作不在动作空间内: {action}")
        
        old_state = copy.deepcopy(self.current_state)
        
        # 执行状态转移
        next_state, transition_info = self.transition_function.transition(self.current_state, action)
        
        # 添加噪声（如果是随机环境）
        if self.stochastic and self.noise_level > 0:
            next_state = self._add_state_noise(next_state)
        
        # 计算奖励
        reward = self.reward_function.calculate_reward(
            self.current_state, action, next_state, transition_info
        )
        
        # 检查终止条件
        terminated = self.transition_function.is_terminal_state(next_state)
        truncated = self.episode_step >= self.env_info.max_episode_steps - 1
        
        # 更新环境状态
        self.current_state = next_state
        self.episode_step += 1
        self.total_steps += 1
        
        # 记录历史
        experience = Experience(
            experience_id=str(uuid.uuid4()),
            state=old_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated or truncated,
            priority=1.0,
            timestamp=utc_now()
        )
        self.episode_history.append(experience)
        self.reward_history.append(reward)
        
        # 创建步骤信息
        info = {
            **transition_info,
            "episode_step": self.episode_step,
            "total_steps": self.total_steps,
            "is_success": terminated and reward > 0
        }
        self.step_info_history.append(info)
        
        # 更新环境状态
        if terminated:
            self.status = EnvironmentStatus.TERMINATED
        elif truncated:
            self.status = EnvironmentStatus.TRUNCATED
        
        return StepResult(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info
        )
    
    def _create_transition_function(self, config: Dict[str, Any]) -> TransitionFunction:
        """创建状态转移函数"""
        transition_type = config.get("type", "identity")
        
        if transition_type == "identity":
            return IdentityTransition(config)
        elif transition_type == "linear":
            return LinearTransition(config, self.state_space)
        elif transition_type == "grid_world":
            return GridWorldTransition(config)
        else:
            raise ValueError(f"不支持的转移函数类型: {transition_type}")
    
    def _create_reward_function(self, config: Dict[str, Any]) -> RewardFunction:
        """创建奖励函数"""
        reward_type = config.get("type", "sparse")
        
        if reward_type == "sparse":
            return SparseReward(config)
        elif reward_type == "dense":
            return DenseReward(config)
        elif reward_type == "shaped":
            return ShapedReward(config)
        else:
            raise ValueError(f"不支持的奖励函数类型: {reward_type}")
    
    def _create_initial_state(self, config: Dict[str, Any], options: Optional[Dict[str, Any]]) -> AgentState:
        """根据配置创建初始状态"""
        if config.get("type") == "fixed":
            return AgentState.create(features=config["features"])
        elif config.get("type") == "random_box":
            # 在指定范围内随机生成
            features = {}
            for name, bounds in config["bounds"].items():
                features[name] = np.random.uniform(bounds[0], bounds[1])
            return AgentState.create(features=features)
        else:
            return self.state_space.sample()
    
    def _add_state_noise(self, state: AgentState) -> AgentState:
        """为状态添加噪声"""
        noisy_features = {}
        for name, value in state.features.items():
            if isinstance(value, (int, float)):
                noise = np.random.normal(0, self.noise_level)
                noisy_features[name] = value + noise
            else:
                noisy_features[name] = value
        
        return AgentState.create(features=noisy_features, context=state.context)

# 具体的状态转移函数实现

class IdentityTransition(TransitionFunction):
    """恒等转移函数（状态不变）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.terminal_conditions = config.get("terminal_conditions", [])
    
    def transition(self, state: AgentState, action: Any) -> Tuple[AgentState, Dict[str, Any]]:
        return state, {"transition_type": "identity"}
    
    def is_terminal_state(self, state: AgentState) -> bool:
        # 检查终止条件
        for condition in self.terminal_conditions:
            if self._check_condition(state, condition):
                return True
        return False
    
    def _check_condition(self, state: AgentState, condition: Dict[str, Any]) -> bool:
        """检查终止条件"""
        feature_name = condition.get("feature")
        if feature_name not in state.features:
            return False
        
        value = state.features[feature_name]
        operator = condition.get("operator", "==")
        threshold = condition.get("value")
        
        if operator == "==":
            return value == threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        
        return False

class LinearTransition(TransitionFunction):
    """线性状态转移函数"""
    
    def __init__(self, config: Dict[str, Any], state_space: StateSpace):
        self.state_space = state_space
        self.dynamics_matrix = np.array(config.get("dynamics_matrix", np.eye(state_space.dimension)))
        self.action_matrix = np.array(config.get("action_matrix", np.zeros((state_space.dimension, 1))))
        self.noise_std = config.get("noise_std", 0.0)
        self.terminal_conditions = config.get("terminal_conditions", [])
    
    def transition(self, state: AgentState, action: Any) -> Tuple[AgentState, Dict[str, Any]]:
        # 将状态转换为向量
        state_vector = self.state_space.normalize_state(state)
        
        # 线性转移: s' = A*s + B*a + noise
        if isinstance(action, (int, float)):
            action_vector = np.array([action])
        else:
            action_vector = np.array(action)
        
        next_state_vector = (
            self.dynamics_matrix @ state_vector +
            self.action_matrix @ action_vector
        )
        
        # 添加噪声
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, next_state_vector.shape)
            next_state_vector += noise
        
        # 转换回状态格式
        next_features = {}
        for i, feature_name in enumerate(self.state_space.feature_names):
            if i < len(next_state_vector):
                next_features[feature_name] = float(next_state_vector[i])
        
        next_state = AgentState.create(features=next_features)
        
        return next_state, {
            "transition_type": "linear",
            "dynamics_matrix": self.dynamics_matrix.tolist(),
            "action_matrix": self.action_matrix.tolist()
        }
    
    def is_terminal_state(self, state: AgentState) -> bool:
        # 复用IdentityTransition的终止条件检查
        identity_transition = IdentityTransition({"terminal_conditions": self.terminal_conditions})
        return identity_transition.is_terminal_state(state)

class GridWorldTransition(TransitionFunction):
    """网格世界状态转移函数"""
    
    def __init__(self, config: Dict[str, Any]):
        self.grid_size = config.get("grid_size", (10, 10))
        self.obstacles = set(tuple(obs) for obs in config.get("obstacles", []))
        self.goal_positions = set(tuple(goal) for goal in config.get("goals", [(9, 9)]))
        self.action_effects = config.get("action_effects", {
            0: (0, -1),  # 上
            1: (0, 1),   # 下
            2: (-1, 0),  # 左
            3: (1, 0)    # 右
        })
        self.success_rate = config.get("success_rate", 0.8)  # 动作成功率
    
    def transition(self, state: AgentState, action: Any) -> Tuple[AgentState, Dict[str, Any]]:
        current_pos = (int(state.features.get("x", 0)), int(state.features.get("y", 0)))
        
        # 确定实际执行的动作（考虑动作失败）
        if np.random.random() < self.success_rate:
            executed_action = action
        else:
            # 随机选择其他动作
            executed_action = np.random.choice([a for a in self.action_effects.keys() if a != action])
        
        # 计算新位置
        if executed_action in self.action_effects:
            dx, dy = self.action_effects[executed_action]
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
        else:
            new_x, new_y = current_pos
        
        # 检查边界和障碍物
        new_x = max(0, min(new_x, self.grid_size[0] - 1))
        new_y = max(0, min(new_y, self.grid_size[1] - 1))
        
        if (new_x, new_y) in self.obstacles:
            # 撞到障碍物，位置不变
            new_x, new_y = current_pos
        
        # 创建新状态
        next_state = AgentState.create(features={
            "x": new_x,
            "y": new_y,
            **{k: v for k, v in state.features.items() if k not in ["x", "y"]}
        })
        
        info = {
            "transition_type": "grid_world",
            "intended_action": action,
            "executed_action": executed_action,
            "hit_obstacle": (new_x, new_y) == current_pos and action in self.action_effects,
            "at_goal": (new_x, new_y) in self.goal_positions
        }
        
        return next_state, info
    
    def is_terminal_state(self, state: AgentState) -> bool:
        current_pos = (int(state.features.get("x", 0)), int(state.features.get("y", 0)))
        return current_pos in self.goal_positions

# 具体的奖励函数实现

class SparseReward(RewardFunction):
    """稀疏奖励函数"""
    
    def __init__(self, config: Dict[str, Any]):
        self.goal_reward = config.get("goal_reward", 1.0)
        self.step_penalty = config.get("step_penalty", -0.01)
        self.collision_penalty = config.get("collision_penalty", -0.1)
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        reward = self.step_penalty
        
        # 目标奖励
        if info.get("at_goal", False):
            reward += self.goal_reward
        
        # 碰撞惩罚
        if info.get("hit_obstacle", False):
            reward += self.collision_penalty
        
        return reward
    
    def get_reward_info(self) -> Dict[str, Any]:
        return {
            "type": "sparse",
            "goal_reward": self.goal_reward,
            "step_penalty": self.step_penalty,
            "collision_penalty": self.collision_penalty
        }

class DenseReward(RewardFunction):
    """密集奖励函数（基于距离）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.goal_reward = config.get("goal_reward", 1.0)
        self.step_penalty = config.get("step_penalty", -0.01)
        self.distance_reward_scale = config.get("distance_reward_scale", 0.1)
        self.goal_position = config.get("goal_position", (9, 9))
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        reward = self.step_penalty
        
        # 目标奖励
        if info.get("at_goal", False):
            reward += self.goal_reward
        
        # 距离奖励
        current_pos = (state.features.get("x", 0), state.features.get("y", 0))
        next_pos = (next_state.features.get("x", 0), next_state.features.get("y", 0))
        
        current_distance = abs(current_pos[0] - self.goal_position[0]) + abs(current_pos[1] - self.goal_position[1])
        next_distance = abs(next_pos[0] - self.goal_position[0]) + abs(next_pos[1] - self.goal_position[1])
        
        # 距离减小时给予奖励
        distance_improvement = current_distance - next_distance
        reward += distance_improvement * self.distance_reward_scale
        
        return reward
    
    def get_reward_info(self) -> Dict[str, Any]:
        return {
            "type": "dense",
            "goal_reward": self.goal_reward,
            "step_penalty": self.step_penalty,
            "distance_reward_scale": self.distance_reward_scale,
            "goal_position": self.goal_position
        }

class ShapedReward(RewardFunction):
    """塑形奖励函数（结合多种奖励信号）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_reward_function = DenseReward(config)
        self.exploration_bonus = config.get("exploration_bonus", 0.01)
        self.visited_states: set = set()
    
    def calculate_reward(self, state: AgentState, action: Any, 
                        next_state: AgentState, info: Dict[str, Any]) -> float:
        # 基础奖励
        reward = self.base_reward_function.calculate_reward(state, action, next_state, info)
        
        # 探索奖励
        state_key = tuple(next_state.features.get(k, 0) for k in ["x", "y"])
        if state_key not in self.visited_states:
            reward += self.exploration_bonus
            self.visited_states.add(state_key)
        
        return reward
    
    def get_reward_info(self) -> Dict[str, Any]:
        base_info = self.base_reward_function.get_reward_info()
        base_info.update({
            "type": "shaped",
            "exploration_bonus": self.exploration_bonus,
            "visited_states_count": len(self.visited_states)
        })
        return base_info
