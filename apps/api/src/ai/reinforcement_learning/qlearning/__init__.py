"""
Q-Learning算法实现模块

包含经典Q-Learning、Deep Q-Network、Double DQN、Dueling DQN等算法变体
"""

# 基础类始终可用
from .base import QLearningAgent
from .q_learning import ClassicQLearningAgent  
from .replay_buffer import ReplayBuffer

__all__ = [
    "QLearningAgent",
    "ClassicQLearningAgent",
    "ReplayBuffer",
]

# 深度学习算法类的延迟导入，避免TensorFlow导入问题
def _import_deep_learning_classes():
    """延迟导入深度学习相关类"""
    try:
        from .dqn import DQNAgent
        from .double_dqn import DoubleDQNAgent
        from .dueling_dqn import DuelingDQNAgent
        return {
            "DQNAgent": DQNAgent,
            "DoubleDQNAgent": DoubleDQNAgent,
            "DuelingDQNAgent": DuelingDQNAgent
        }
    except ImportError as e:
        import warnings
        warnings.warn(f"深度学习算法类导入失败: {e}. 请安装tensorflow>=2.15.0")
        return {}

def __getattr__(name):
    """动态导入深度学习类"""
    deep_classes = _import_deep_learning_classes()
    if name in deep_classes:
        return deep_classes[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")