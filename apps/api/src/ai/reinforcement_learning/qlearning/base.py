"""
Q-Learning智能体抽象基类

定义所有Q-Learning算法的通用接口和行为。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.utils.timezone_utils import utc_now

class AlgorithmType(Enum):
    """Q-Learning算法类型枚举"""
    Q_LEARNING = "q_learning"
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"

@dataclass
class AgentState:
    """智能体状态数据结构"""
    state_id: str
    features: Dict[str, float]
    context: Dict[str, Any]
    timestamp: datetime
    episode_id: Optional[str] = None

    @classmethod
    def create(cls, features: Dict[str, float], context: Optional[Dict[str, Any]] = None, episode_id: Optional[str] = None) -> "AgentState":
        """创建新的智能体状态"""
        return cls(
            state_id=str(uuid.uuid4()),
            features=features,
            context=context or {},
            timestamp=utc_now(),
            episode_id=episode_id
        )

@dataclass 
class Experience:
    """经验回放数据结构"""
    experience_id: str
    state: AgentState
    action: str
    reward: float
    next_state: AgentState
    done: bool
    priority: float = 1.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = utc_now()

    @classmethod
    def create(cls, state: AgentState, action: str, reward: float, next_state: AgentState, done: bool, priority: float = 1.0) -> "Experience":
        """创建新的经验"""
        return cls(
            experience_id=str(uuid.uuid4()),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=priority
        )

@dataclass
class QLearningConfig:
    """Q-Learning配置参数"""
    algorithm_type: AlgorithmType
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 32
    target_update_frequency: int = 100
    
    # 神经网络架构配置（仅用于深度学习方法）
    network_architecture: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """配置参数验证"""
        if self.learning_rate <= 0:
            raise ValueError("学习率必须大于0")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("折扣因子必须在[0,1]范围内")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon参数范围错误")

@dataclass
class TrainingResults:
    """训练结果数据结构"""
    total_episodes: int
    average_reward: float
    final_epsilon: float
    training_time: float
    convergence_achieved: bool
    best_average_reward: float
    loss_history: List[float]
    reward_history: List[float]

class QLearningAgent(ABC):
    """Q-Learning智能体抽象基类"""
    
    def __init__(self, agent_id: str, state_size: int, action_size: int, config: QLearningConfig):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.epsilon = config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.training_history: Dict[str, List[float]] = {
            "rewards": [],
            "losses": [], 
            "epsilon_values": []
        }
        
    @abstractmethod
    def get_action(self, state: AgentState, exploration: bool = True) -> str:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            exploration: 是否使用探索策略
            
        Returns:
            选择的动作
        """
        ...
    
    @abstractmethod 
    def update_q_value(self, experience: Experience) -> Optional[float]:
        """
        更新Q值函数
        
        Args:
            experience: 经验数据
            
        Returns:
            训练损失（如果适用）
        """
        ...
    
    @abstractmethod
    def get_q_values(self, state: AgentState) -> Dict[str, float]:
        """
        获取状态的Q值
        
        Args:
            state: 输入状态
            
        Returns:
            动作到Q值的映射
        """
        ...
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        ...
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        ...
    
    def decay_epsilon(self) -> None:
        """衰减探索率"""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def get_optimal_action(self, state: AgentState) -> str:
        """获取最优动作（不使用探索）"""
        return self.get_action(state, exploration=False)
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验（基础实现，可被子类重写）"""
        self.step_count += 1
        self.total_reward += experience.reward
    
    def start_episode(self, episode_id: Optional[str] = None) -> None:
        """开始新的episode"""
        self.episode_count += 1
        if episode_id is None:
            episode_id = f"episode_{self.episode_count}"
    
    def end_episode(self, total_reward: float) -> None:
        """结束当前episode"""
        self.training_history["rewards"].append(total_reward)
        self.training_history["epsilon_values"].append(self.epsilon)
        self.decay_epsilon()

    def _normalize_state(self, state: Any) -> AgentState:
        """将输入状态统一为AgentState"""
        if isinstance(state, AgentState):
            return state
        if isinstance(state, dict):
            features = {str(k): float(v) for k, v in state.items()}
            return AgentState.create(features=features)
        if isinstance(state, (list, tuple, np.ndarray)):
            array_state = np.asarray(state, dtype=float).reshape(-1)
            features = {f"f{i}": float(v) for i, v in enumerate(array_state.tolist())}
            return AgentState.create(features=features)
        raise ValueError(f"不支持的状态类型: {type(state)}")

    def _action_label_to_index(self, action: Any) -> int:
        if isinstance(action, int):
            return action
        action_space = getattr(self, "action_space", None)
        if action_space and action in action_space:
            return action_space.index(action)
        try:
            return int(action)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法解析动作: {action}") from exc

    def _action_index_to_label(self, action: Any) -> str:
        if isinstance(action, str):
            return action
        action_space = getattr(self, "action_space", None)
        if action_space and isinstance(action, int) and 0 <= action < len(action_space):
            return action_space[action]
        return str(action)

    def act(self, state: Any, episode: Optional[int] = None, evaluation: bool = False) -> int:
        """兼容旧接口的动作选择"""
        agent_state = self._normalize_state(state)
        action_label = self.get_action(agent_state, exploration=not evaluation)
        return self._action_label_to_index(action_label)

    def learn(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> Optional[float]:
        """兼容旧接口的学习过程"""
        agent_state = self._normalize_state(state)
        next_agent_state = self._normalize_state(next_state)
        action_label = self._action_index_to_label(action)
        experience = Experience.create(agent_state, action_label, float(reward), next_agent_state, bool(done))
        loss = self.update_q_value(experience)
        self.decay_epsilon()
        return loss

    def on_episode_end(self, episode_reward: float) -> None:
        """回合结束处理（基础实现）"""
        self.training_history["rewards"].append(float(episode_reward))
        self.training_history["epsilon_values"].append(self.epsilon)

    def set_learning_rate(self, learning_rate: float) -> None:
        """更新学习率（基础实现）"""
        self.config.learning_rate = float(learning_rate)
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "learning_rate"):
            try:
                optimizer.learning_rate.assign(self.config.learning_rate)
            except Exception:
                optimizer.learning_rate = self.config.learning_rate
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            "agent_id": self.agent_id,
            "algorithm_type": self.config.algorithm_type.value,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "current_epsilon": self.epsilon,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(self.step_count, 1),
            "recent_rewards": self.training_history["rewards"][-10:] if self.training_history["rewards"] else []
        }
    
    def reset(self) -> None:
        """重置智能体状态"""
        self.epsilon = self.config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.training_history = {"rewards": [], "losses": [], "epsilon_values": []}

class ExplorationStrategy(ABC):
    """探索策略抽象基类"""
    
    @abstractmethod
    def select_action(self, q_values: Dict[str, float], available_actions: List[str], epsilon: float) -> str:
        """根据探索策略选择动作"""
        ...

class EpsilonGreedyStrategy(ExplorationStrategy):
    """epsilon-greedy探索策略"""
    
    def select_action(self, q_values: Dict[str, float], available_actions: List[str], epsilon: float) -> str:
        """epsilon-greedy动作选择"""
        if np.random.random() < epsilon:
            # 随机探索
            return np.random.choice(available_actions)
        else:
            # 贪婪选择
            if not q_values:
                return np.random.choice(available_actions)
            
            # 找到Q值最高的动作
            max_q_value = max(q_values.values())
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q_value and action in available_actions]
            
            if not best_actions:
                return np.random.choice(available_actions)
                
            return np.random.choice(best_actions)

class UCBStrategy(ExplorationStrategy):
    """Upper Confidence Bound探索策略"""
    
    def __init__(self, c: float = 1.0):
        self.c = c
        self.action_counts: Dict[str, int] = {}
        self.total_steps = 0
    
    def select_action(self, q_values: Dict[str, float], available_actions: List[str], epsilon: float = None) -> str:
        """UCB动作选择"""
        self.total_steps += 1
        
        if not q_values:
            action = np.random.choice(available_actions)
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            return action
        
        ucb_values = {}
        for action in available_actions:
            if action not in q_values:
                continue
                
            count = self.action_counts.get(action, 0)
            if count == 0:
                # 尚未尝试的动作给予最高优先级
                ucb_values[action] = float('inf')
            else:
                confidence_bonus = self.c * np.sqrt(np.log(self.total_steps) / count)
                ucb_values[action] = q_values[action] + confidence_bonus
        
        if not ucb_values:
            action = np.random.choice(available_actions)
        else:
            action = max(ucb_values.keys(), key=lambda x: ucb_values[x])
        
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        return action
