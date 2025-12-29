"""
Double DQN 算法实现

解决传统DQN的高估偏差问题，使用主网络选择动作，目标网络估计Q值。
"""

from src.core.tensorflow_config import tensorflow_lazy
from typing import Dict, List, Optional, Any, Tuple
from .dqn import DQNAgent
from .base import QLearningConfig, AlgorithmType

class DoubleDQNAgent(DQNAgent):
    """Double DQN智能体实现"""
    
    def __init__(self, 
                 agent_id: str, 
                 state_size: int, 
                 action_size: int, 
                 config: QLearningConfig,
                 action_space: List[str],
                 use_prioritized_replay: bool = False):
        
        # 强制启用Double DQN
        config.algorithm_type = AlgorithmType.DOUBLE_DQN
        super().__init__(
            agent_id=agent_id,
            state_size=state_size,
            action_size=action_size,
            config=config,
            action_space=action_space,
            use_prioritized_replay=use_prioritized_replay,
            use_double_dqn=True  # 强制启用Double DQN
        )
    
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights=None):
        """计算Double DQN损失"""
        # 当前Q值
        current_q_values = self.q_network(states, training=True)
        current_q_values = tensorflow_lazy.tf.gather(current_q_values, actions, batch_dims=1)
        
        # Double DQN核心：使用主网络选择动作
        next_q_values_main = self.q_network(next_states, training=False)
        next_actions = tensorflow_lazy.tf.argmax(next_q_values_main, axis=1)
        
        # 使用目标网络估计选中动作的Q值
        next_q_values_target = self.target_network(next_states, training=False)
        next_q_values = tensorflow_lazy.tf.gather(next_q_values_target, next_actions, batch_dims=1)
        
        # 计算目标Q值
        targets = rewards + (1.0 - tensorflow_lazy.tf.cast(dones, tensorflow_lazy.tf.float32)) * self.config.discount_factor * next_q_values
        targets = tensorflow_lazy.tf.stop_gradient(targets)
        
        # 计算TD误差
        td_errors = targets - current_q_values
        
        # 计算损失
        if weights is not None:
            # 优先级采样权重
            loss = tensorflow_lazy.tf.reduce_mean(weights * tensorflow_lazy.tf.square(td_errors))
        else:
            loss = tensorflow_lazy.tf.reduce_mean(tensorflow_lazy.tf.square(td_errors))
        
        return loss, td_errors
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息（包含Double DQN特定信息）"""
        stats = super().get_training_stats()
        stats["uses_double_dqn"] = True
        stats["overestimation_bias_reduction"] = "active"
        return stats
