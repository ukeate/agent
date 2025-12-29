"""
Deep Q-Network (DQN) 算法实现

使用深度神经网络近似Q函数，适用于高维状态空间的强化学习问题。
"""

from __future__ import annotations

import os
import json
import numpy as np
from src.core.tensorflow_config import tensorflow_lazy
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import random
from .base import (

    QLearningAgent,
    AgentState, 
    Experience,
    QLearningConfig,
    AlgorithmType,
    EpsilonGreedyStrategy
)
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from src.core.logging import get_logger
logger = get_logger(__name__)

class DQNAgent(QLearningAgent):
    """Deep Q-Network智能体实现"""
    
    def __init__(self, 
                 agent_id: str, 
                 state_size: int, 
                 action_size: int, 
                 config: QLearningConfig,
                 action_space: List[str],
                 use_prioritized_replay: bool = False,
                 use_double_dqn: bool = False):
        if not tensorflow_lazy.available:
            raise RuntimeError("TensorFlow不可用，无法创建DQN智能体")
        
        # 确保是DQN算法
        config.algorithm_type = AlgorithmType.DQN
        super().__init__(agent_id, state_size, action_size, config)
        
        self.action_space = action_space
        self.use_prioritized_replay = use_prioritized_replay
        self.use_double_dqn = use_double_dqn
        
        # 神经网络架构配置
        self.network_config = config.network_architecture or {
            "hidden_layers": [128, 128, 64],
            "activation": "relu",
            "optimizer": "adam",
            "loss_function": "mse"
        }
        
        # 构建神经网络
        self.q_network = self._build_network()
        self.target_network = self._build_network() 
        self.optimizer = self._create_optimizer()
        
        # 初始化目标网络参数
        self.update_target_network()
        
        # 经验回放缓冲区
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.buffer_size,
                alpha=0.6,
                beta=0.4
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)
        
        self.exploration_strategy = EpsilonGreedyStrategy()
        
        # 训练相关
        self.loss_function = tensorflow_lazy.tf.keras.losses.MeanSquaredError()
        self.train_step_counter = 0
        self.target_update_counter = 0
        
        # GPU配置
        self._configure_gpu()
    
    def _configure_gpu(self):
        """配置GPU使用"""
        physical_devices = tensorflow_lazy.tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                # 启用内存增长
                for device in physical_devices:
                    tensorflow_lazy.tf.config.experimental.set_memory_growth(device, True)
                self.device_name = '/GPU:0'
                logger.info("DQN使用GPU加速", agent_id=self.agent_id)
            except Exception as e:
                logger.error("GPU配置失败", agent_id=self.agent_id, error=str(e), exc_info=True)
                self.device_name = '/CPU:0'
        else:
            self.device_name = '/CPU:0'
            logger.info("DQN使用CPU", agent_id=self.agent_id)
    
    def _build_network(self) -> tensorflow_lazy.tf.keras.Model:
        """构建深度Q网络"""
        model = tensorflow_lazy.tf.keras.Sequential()
        
        # 输入层
        model.add(tensorflow_lazy.tf.keras.layers.Dense(
            self.network_config["hidden_layers"][0],
            input_shape=(self.state_size,),
            activation=self.network_config["activation"],
            name="input_dense"
        ))
        
        # 隐藏层
        for i, units in enumerate(self.network_config["hidden_layers"][1:], 1):
            model.add(tensorflow_lazy.tf.keras.layers.Dense(
                units,
                activation=self.network_config["activation"],
                name=f"hidden_dense_{i}"
            ))
            
            # 添加Dropout防止过拟合
            if i < len(self.network_config["hidden_layers"]) - 1:
                model.add(tensorflow_lazy.tf.keras.layers.Dropout(0.2, name=f"dropout_{i}"))
        
        # 输出层
        model.add(tensorflow_lazy.tf.keras.layers.Dense(
            self.action_size,
            activation="linear",  # Q值可以为负数
            name="output_dense"
        ))
        
        return model
    
    def _create_optimizer(self) -> tensorflow_lazy.tf.keras.optimizers.Optimizer:
        """创建优化器"""
        optimizer_name = self.network_config.get("optimizer", "adam").lower()
        
        if optimizer_name == "adam":
            return tensorflow_lazy.tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif optimizer_name == "rmsprop":
            return tensorflow_lazy.tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif optimizer_name == "sgd":
            return tensorflow_lazy.tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        else:
            return tensorflow_lazy.tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
    
    def _state_to_array(self, state: AgentState) -> np.ndarray:
        """将状态转换为神经网络输入"""
        if isinstance(state.features, dict):
            # 按键排序确保一致性
            sorted_keys = sorted(state.features.keys())
            features = [state.features[key] for key in sorted_keys]
        elif isinstance(state.features, (list, np.ndarray)):
            features = list(state.features)
        else:
            raise ValueError(f"不支持的状态特征类型: {type(state.features)}")
        
        # 确保特征维度正确
        if len(features) != self.state_size:
            raise ValueError(f"状态特征维度 {len(features)} 与预期 {self.state_size} 不匹配")
        
        return np.array(features, dtype=np.float32)
    
    def get_action(self, state: AgentState, exploration: bool = True) -> str:
        """使用epsilon-greedy策略选择动作"""
        with tensorflow_lazy.tf.device(self.device_name):
            state = self._normalize_state(state)
            state_array = self._state_to_array(state)
            
            if not exploration:
                # 贪婪策略
                q_values = self.q_network(state_array.reshape(1, -1), training=False)
                action_idx = tensorflow_lazy.tf.argmax(q_values[0]).numpy()
                return self.action_space[action_idx]
            
            # epsilon-greedy策略
            if random.random() < self.epsilon:
                return random.choice(self.action_space)
            else:
                q_values = self.q_network(state_array.reshape(1, -1), training=False)
                action_idx = tensorflow_lazy.tf.argmax(q_values[0]).numpy()
                return self.action_space[action_idx]
    
    def get_q_values(self, state: AgentState) -> Dict[str, float]:
        """获取状态的所有Q值"""
        with tensorflow_lazy.tf.device(self.device_name):
            state = self._normalize_state(state)
            state_array = self._state_to_array(state)
            q_values = self.q_network(state_array.reshape(1, -1), training=False)
            q_values_dict = {}
            
            for i, action in enumerate(self.action_space):
                q_values_dict[action] = float(q_values[0][i].numpy())
            
            return q_values_dict
    
    def update_q_value(self, experience: Experience) -> Optional[float]:
        """添加经验到replay buffer并执行训练"""
        self.replay_buffer.push(experience)
        self.add_experience(experience)
        
        # 检查是否可以开始训练
        min_buffer_size = max(self.config.batch_size * 4, 1000)
        if not self.replay_buffer.is_ready(min_buffer_size):
            return None
        
        # 执行训练
        loss = self._train_step()
        
        # 更新目标网络
        self.target_update_counter += 1
        if self.target_update_counter >= self.config.target_update_frequency:
            self.update_target_network()
            self.target_update_counter = 0
        
        return loss
    
    def _train_step(self) -> float:
        """执行一次训练步骤"""
        with tensorflow_lazy.tf.device(self.device_name):
            # 从replay buffer采样
            if self.use_prioritized_replay:
                experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
                weights = tensorflow_lazy.tf.constant(weights, dtype=tensorflow_lazy.tf.float32)
            else:
                experiences = self.replay_buffer.sample(self.config.batch_size)
                weights = None
                indices = None
            
            if not experiences:
                return 0.0
            
            # 准备训练数据
            states, actions, rewards, next_states, dones = self._prepare_training_data(experiences)
            
            # 计算损失和梯度
            with tensorflow_lazy.tf.GradientTape() as tape:
                loss, td_errors = self._compute_loss(states, actions, rewards, next_states, dones, weights)
            
            # 应用梯度
            gradients = tape.gradient(loss, self.q_network.trainable_variables)
            
            # 梯度裁剪
            gradients = [tensorflow_lazy.tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
            
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
            
            # 更新优先级（如果使用优先级replay）
            if self.use_prioritized_replay and indices is not None:
                priorities = np.abs(td_errors.numpy()) + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
            
            self.train_step_counter += 1
            loss_value = float(loss.numpy())
            self.training_history["losses"].append(loss_value)
            
            return loss_value
    
    def _prepare_training_data(self, experiences: List[Experience]) -> Tuple[tensorflow_lazy.tf.Tensor, tensorflow_lazy.tf.Tensor, tensorflow_lazy.tf.Tensor, tensorflow_lazy.tf.Tensor, tensorflow_lazy.tf.Tensor]:
        """准备训练数据"""
        batch_size = len(experiences)
        
        states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size, self.state_size), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.bool_)
        
        for i, exp in enumerate(experiences):
            states[i] = self._state_to_array(exp.state)
            actions[i] = self.action_space.index(exp.action)
            rewards[i] = exp.reward
            next_states[i] = self._state_to_array(exp.next_state)
            dones[i] = exp.done
        
        return (
            tensorflow_lazy.tf.constant(states),
            tensorflow_lazy.tf.constant(actions),
            tensorflow_lazy.tf.constant(rewards),
            tensorflow_lazy.tf.constant(next_states),
            tensorflow_lazy.tf.constant(dones)
        )
    
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights=None):
        """计算DQN损失"""
        # 当前Q值
        current_q_values = self.q_network(states, training=True)
        current_q_values = tensorflow_lazy.tf.gather(current_q_values, actions, batch_dims=1)
        
        # 计算目标Q值
        if self.use_double_dqn:
            # Double DQN: 使用主网络选择动作，目标网络估计Q值
            next_q_values_main = self.q_network(next_states, training=False)
            next_actions = tensorflow_lazy.tf.argmax(next_q_values_main, axis=1)
            next_q_values_target = self.target_network(next_states, training=False)
            next_q_values = tensorflow_lazy.tf.gather(next_q_values_target, next_actions, batch_dims=1)
        else:
            # 标准DQN: 使用目标网络
            next_q_values = tensorflow_lazy.tf.reduce_max(self.target_network(next_states, training=False), axis=1)
        
        # 计算目标
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
    
    def update_target_network(self):
        """更新目标网络参数"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存神经网络权重
        weights_path = filepath.replace('.json', '_weights.h5').replace('.pkl', '_weights.h5')
        self.q_network.save_weights(weights_path)
        
        # 保存目标网络权重
        target_weights_path = filepath.replace('.json', '_target_weights.h5').replace('.pkl', '_target_weights.h5')
        self.target_network.save_weights(target_weights_path)
        
        # 保存配置和状态
        model_data = {
            "agent_id": self.agent_id,
            "config": {
                "algorithm_type": self.config.algorithm_type.value,
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
                "buffer_size": self.config.buffer_size,
                "batch_size": self.config.batch_size,
                "target_update_frequency": self.config.target_update_frequency,
            },
            "network_config": self.network_config,
            "action_space": self.action_space,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "use_prioritized_replay": self.use_prioritized_replay,
            "use_double_dqn": self.use_double_dqn,
            "current_epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "train_step_counter": self.train_step_counter,
            "target_update_counter": self.target_update_counter,
            "training_history": self.training_history
        }
        
        # 保存为JSON
        config_path = filepath if filepath.endswith('.json') else filepath + '.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        # 加载配置
        config_path = filepath if filepath.endswith('.json') else filepath + '.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # 恢复配置
        self.network_config = model_data["network_config"]
        self.action_space = model_data["action_space"]
        self.state_size = model_data["state_size"]
        self.action_size = model_data["action_size"]
        self.use_prioritized_replay = model_data.get("use_prioritized_replay", False)
        self.use_double_dqn = model_data.get("use_double_dqn", False)
        
        # 重建网络
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        
        # 加载权重
        weights_path = filepath.replace('.json', '_weights.h5').replace('.pkl', '_weights.h5')
        if os.path.exists(weights_path):
            self.q_network.load_weights(weights_path)
        
        target_weights_path = filepath.replace('.json', '_target_weights.h5').replace('.pkl', '_target_weights.h5')
        if os.path.exists(target_weights_path):
            self.target_network.load_weights(target_weights_path)
        else:
            # 如果没有目标网络权重，复制主网络权重
            self.update_target_network()
        
        # 恢复训练状态
        self.epsilon = model_data.get("current_epsilon", self.config.epsilon_start)
        self.step_count = model_data.get("step_count", 0)
        self.episode_count = model_data.get("episode_count", 0)
        self.train_step_counter = model_data.get("train_step_counter", 0)
        self.target_update_counter = model_data.get("target_update_counter", 0)
        self.training_history = model_data.get("training_history", {"rewards": [], "losses": [], "epsilon_values": []})
    
    def get_network_summary(self) -> str:
        """获取网络结构摘要"""
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        try:
            self.q_network.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def evaluate_policy(self, test_states: List[AgentState], num_episodes: int = 100) -> Dict[str, float]:
        """评估当前策略"""
        total_rewards = []
        q_value_stats = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_q_values = []
            
            for state in test_states:
                q_values = self.get_q_values(state)
                max_q = max(q_values.values())
                episode_q_values.append(max_q)
                
                # 模拟奖励（实际使用中应该从环境获取）
                action = self.get_optimal_action(state)
                reward = random.uniform(-1, 1)  # 占位符奖励
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            q_value_stats.extend(episode_q_values)
        
        return {
            "average_reward": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "average_q_value": np.mean(q_value_stats),
            "q_value_std": np.std(q_value_stats),
            "num_episodes": num_episodes
        }
    
    def reset(self) -> None:
        """重置智能体（但保留网络权重）"""
        super().reset()
        self.train_step_counter = 0
        self.target_update_counter = 0
        
        # 重置replay buffer
        if hasattr(self.replay_buffer, 'clear'):
            self.replay_buffer.clear()
