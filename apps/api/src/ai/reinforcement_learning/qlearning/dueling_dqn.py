"""
Dueling DQN 算法实现

分离状态价值函数和优势函数的估计，提高学习效率和稳定性。
"""

from __future__ import annotations

from src.core.tensorflow_config import tensorflow_lazy
from typing import Dict, List, Optional, Any
from .dqn import DQNAgent
from .base import QLearningConfig, AlgorithmType

class DuelingDQNAgent(DQNAgent):
    """Dueling DQN智能体实现"""
    
    def __init__(self, 
                 agent_id: str, 
                 state_size: int, 
                 action_size: int, 
                 config: QLearningConfig,
                 action_space: List[str],
                 use_prioritized_replay: bool = False,
                 use_double_dqn: bool = False):
        if not tensorflow_lazy.available:
            raise RuntimeError("TensorFlow不可用，无法创建DuelingDQN智能体")
        
        config.algorithm_type = AlgorithmType.DUELING_DQN
        
        # 先初始化基本属性
        self.use_double_dqn = use_double_dqn
        
        # 调用父类初始化，但不构建网络
        super(DQNAgent, self).__init__(agent_id, state_size, action_size, config)
        
        self.action_space = action_space
        self.use_prioritized_replay = use_prioritized_replay
        
        # 神经网络架构配置
        self.network_config = config.network_architecture or {
            "hidden_layers": [128, 128, 64],
            "activation": "relu",
            "optimizer": "adam",
            "loss_function": "mse"
        }
        
        # 构建Dueling DQN网络
        self.q_network = self._build_dueling_network()
        self.target_network = self._build_dueling_network()
        self.optimizer = self._create_optimizer()
        
        # 初始化目标网络参数
        self.update_target_network()
        
        # 经验回放缓冲区
        from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.buffer_size,
                alpha=0.6,
                beta=0.4
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)
        
        from .base import EpsilonGreedyStrategy
        self.exploration_strategy = EpsilonGreedyStrategy()
        
        # 训练相关
        self.loss_function = tensorflow_lazy.tf.keras.losses.MeanSquaredError()
        self.train_step_counter = 0
        self.target_update_counter = 0
        
        # GPU配置
        self._configure_gpu()
    
    def _build_dueling_network(self) -> tensorflow_lazy.tf.keras.Model:
        """构建Dueling DQN网络架构"""
        # 输入层
        inputs = tensorflow_lazy.tf.keras.layers.Input(shape=(self.state_size,), name="state_input")
        
        # 共享特征提取层
        x = inputs
        for i, units in enumerate(self.network_config["hidden_layers"][:-1]):
            x = tensorflow_lazy.tf.keras.layers.Dense(
                units,
                activation=self.network_config["activation"],
                name=f"shared_dense_{i}"
            )(x)
            if i < len(self.network_config["hidden_layers"]) - 2:
                x = tensorflow_lazy.tf.keras.layers.Dropout(0.2, name=f"shared_dropout_{i}")(x)
        
        # 分离为状态价值流和优势流
        
        # 状态价值流 (Value Stream)
        value_stream = tensorflow_lazy.tf.keras.layers.Dense(
            self.network_config["hidden_layers"][-1],
            activation=self.network_config["activation"],
            name="value_dense"
        )(x)
        value_stream = tensorflow_lazy.tf.keras.layers.Dense(1, name="value_output")(value_stream)
        
        # 优势流 (Advantage Stream)  
        advantage_stream = tensorflow_lazy.tf.keras.layers.Dense(
            self.network_config["hidden_layers"][-1],
            activation=self.network_config["activation"],
            name="advantage_dense"
        )(x)
        advantage_stream = tensorflow_lazy.tf.keras.layers.Dense(
            self.action_size, 
            name="advantage_output"
        )(advantage_stream)
        
        # 组合为Q值: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        # 减去平均值确保可识别性
        advantage_mean = tensorflow_lazy.tf.reduce_mean(advantage_stream, axis=1, keepdims=True)
        advantage_normalized = advantage_stream - advantage_mean
        
        # 最终Q值输出
        q_values = tensorflow_lazy.tf.keras.layers.Add(name="q_values_output")([value_stream, advantage_normalized])
        
        model = tensorflow_lazy.tf.keras.Model(inputs=inputs, outputs=q_values, name="dueling_dqn")
        
        return model
    
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights=None):
        """计算Dueling DQN损失"""
        # 当前Q值
        current_q_values = self.q_network(states, training=True)
        current_q_values = tensorflow_lazy.tf.gather(current_q_values, actions, batch_dims=1)
        
        # 计算目标Q值
        if self.use_double_dqn:
            # 结合Double DQN: 使用主网络选择动作，目标网络估计Q值
            next_q_values_main = self.q_network(next_states, training=False)
            next_actions = tensorflow_lazy.tf.argmax(next_q_values_main, axis=1)
            next_q_values_target = self.target_network(next_states, training=False)
            next_q_values = tensorflow_lazy.tf.gather(next_q_values_target, next_actions, batch_dims=1)
        else:
            # 标准Dueling DQN
            next_q_values = tensorflow_lazy.tf.reduce_max(self.target_network(next_states, training=False), axis=1)
        
        # 计算目标
        targets = rewards + (1.0 - tensorflow_lazy.tf.cast(dones, tensorflow_lazy.tf.float32)) * self.config.discount_factor * next_q_values
        targets = tensorflow_lazy.tf.stop_gradient(targets)
        
        # 计算TD误差
        td_errors = targets - current_q_values
        
        # 计算损失
        if weights is not None:
            loss = tensorflow_lazy.tf.reduce_mean(weights * tensorflow_lazy.tf.square(td_errors))
        else:
            loss = tensorflow_lazy.tf.reduce_mean(tensorflow_lazy.tf.square(td_errors))
        
        return loss, td_errors
    
    def get_value_and_advantage(self, state) -> Dict[str, Any]:
        """获取状态价值和优势函数值（调试用）"""
        state_array = self._state_to_array(state)
        
        with tensorflow_lazy.tf.device(self.device_name):
            # 通过网络的中间层获取价值和优势
            inputs = tensorflow_lazy.tf.constant(state_array.reshape(1, -1), dtype=tensorflow_lazy.tf.float32)
            
            # 前向传播到分离点
            x = inputs
            for layer in self.q_network.layers[1:-4]:  # 到共享层结束
                x = layer(x)
            
            # 价值流
            value_layer = self.q_network.get_layer("value_dense")
            value_output_layer = self.q_network.get_layer("value_output")
            value_stream = value_layer(x)
            value = value_output_layer(value_stream)
            
            # 优势流
            advantage_layer = self.q_network.get_layer("advantage_dense") 
            advantage_output_layer = self.q_network.get_layer("advantage_output")
            advantage_stream = advantage_layer(x)
            advantages = advantage_output_layer(advantage_stream)
            
            # 归一化优势
            advantage_mean = tensorflow_lazy.tf.reduce_mean(advantages, axis=1, keepdims=True)
            advantages_normalized = advantages - advantage_mean
            
            # 最终Q值
            q_values = value + advantages_normalized
            
            return {
                "value": float(value[0, 0].numpy()),
                "advantages": advantages[0].numpy().tolist(),
                "advantages_normalized": advantages_normalized[0].numpy().tolist(),
                "q_values": q_values[0].numpy().tolist(),
                "action_space": self.action_space
            }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息（包含Dueling DQN特定信息）"""
        stats = super().get_training_stats()
        stats["uses_dueling_architecture"] = True
        stats["uses_double_dqn"] = self.use_double_dqn
        stats["value_advantage_separation"] = "active"
        return stats
    
    def get_network_summary(self) -> str:
        """获取Dueling网络结构摘要"""
        lines: List[str] = []
        lines.append("=== Dueling DQN 网络架构 ===")
        self.q_network.summary(print_fn=lines.append)
        lines.append("\n=== 网络层详情 ===")
        for i, layer in enumerate(self.q_network.layers):
            lines.append(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
            if hasattr(layer, "units"):
                lines.append(f"  Units: {layer.units}")
            if hasattr(layer, "activation"):
                lines.append(f"  Activation: {layer.activation}")
        return "\n".join(lines)
