"""
Q-Learning服务实现

提供Q-Learning智能体的管理、训练和评估服务
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from src.ai.reinforcement_learning.qlearning.base import QLearningAgent, QLearningConfig, AlgorithmType
from src.ai.reinforcement_learning.qlearning.q_learning import ClassicQLearningAgent
from src.ai.reinforcement_learning.qlearning.dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from src.ai.reinforcement_learning.environment.state_space import StateSpace, ActionSpace, DiscreteStateSpace, ContinuousStateSpace
from src.ai.reinforcement_learning.environment.simulator import AgentEnvironmentSimulator
from src.ai.reinforcement_learning.environment.grid_world import GridWorldEnvironment
from src.ai.reinforcement_learning.strategies.exploration import (
    ExplorationStrategy, ExplorationConfig, ExplorationMode,
    EpsilonGreedyStrategy, DecayingEpsilonGreedyStrategy, UCBStrategy
)
from src.ai.reinforcement_learning.strategies.training_manager import TrainingManager, TrainingConfig, TrainingMetrics
from src.ai.reinforcement_learning.rewards.base import RewardFunction
from src.ai.reinforcement_learning.rewards.basic_rewards import LinearReward, StepReward, ThresholdReward, GaussianReward

logger = logging.getLogger(__name__)


class QLearningAgentSession:
    """Q-Learning智能体会话"""
    
    def __init__(self, session_id: str, agent: QLearningAgent, environment: AgentEnvironmentSimulator,
                 exploration_strategy: ExplorationStrategy, reward_function: RewardFunction):
        self.session_id = session_id
        self.agent = agent
        self.environment = environment
        self.exploration_strategy = exploration_strategy
        self.reward_function = reward_function
        
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.training_manager: Optional[TrainingManager] = None
        self.training_metrics: Optional[TrainingMetrics] = None
        self.is_training = False
        self.training_task: Optional[asyncio.Task] = None


class QLearningService:
    """Q-Learning服务"""
    
    def __init__(self):
        self.active_sessions: Dict[str, QLearningAgentSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 预定义环境和奖励函数
        self._setup_predefined_components()
    
    def _setup_predefined_components(self):
        """设置预定义组件"""
        # 预定义状态空间
        self.predefined_state_spaces = {
            "grid_4x4": DiscreteStateSpace("grid_4x4", dimensions=[4, 4]),
            "grid_8x8": DiscreteStateSpace("grid_8x8", dimensions=[8, 8]),
            "continuous_2d": ContinuousStateSpace("continuous_2d", dimensions=2, bounds=[(-10, 10), (-10, 10)])
        }
        
        # 预定义动作空间
        self.predefined_action_spaces = {
            "discrete_4": ActionSpace("discrete_4", action_type="discrete", num_actions=4),
            "discrete_8": ActionSpace("discrete_8", action_type="discrete", num_actions=8)
        }
    
    async def create_agent_session(self, agent_config: Dict[str, Any]) -> str:
        """创建智能体会话"""
        try:
            session_id = str(uuid.uuid4())
            
            # 解析配置
            agent_type = AlgorithmType(agent_config.get("agent_type", "q_learning"))
            state_size = agent_config.get("state_size", 16)
            action_size = agent_config.get("action_size", 4)
            
            # 创建Q-Learning配置
            qlearning_config = QLearningConfig(
                agent_type=agent_type,
                learning_rate=agent_config.get("learning_rate", 0.1),
                gamma=agent_config.get("gamma", 0.99),
                epsilon=agent_config.get("epsilon", 0.1),
                epsilon_decay=agent_config.get("epsilon_decay", 0.995),
                epsilon_min=agent_config.get("epsilon_min", 0.01)
            )
            
            # 创建智能体
            agent = self._create_agent(agent_type, state_size, action_size, qlearning_config)
            
            # 创建环境
            environment_config = agent_config.get("environment", {})
            environment = self._create_environment(environment_config)
            
            # 创建探索策略
            exploration_config = agent_config.get("exploration", {})
            exploration_strategy = self._create_exploration_strategy(exploration_config, action_size)
            
            # 创建奖励函数
            reward_config = agent_config.get("reward", {})
            reward_function = self._create_reward_function(reward_config)
            
            # 创建会话
            session = QLearningAgentSession(
                session_id=session_id,
                agent=agent,
                environment=environment,
                exploration_strategy=exploration_strategy,
                reward_function=reward_function
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"创建Q-Learning会话: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"创建智能体会话失败: {e}")
            raise
    
    def _create_agent(self, agent_type: AlgorithmType, state_size: int, action_size: int, 
                     config: QLearningConfig) -> QLearningAgent:
        """创建智能体"""
        if agent_type == AlgorithmType.Q_LEARNING:
            return ClassicQLearningAgent("classic_agent", state_size, action_size, config)
        elif agent_type == AlgorithmType.DQN:
            return DQNAgent("dqn_agent", state_size, action_size, config)
        elif agent_type == AlgorithmType.DOUBLE_DQN:
            return DoubleDQNAgent("double_dqn_agent", state_size, action_size, config)
        elif agent_type == AlgorithmType.DUELING_DQN:
            return DuelingDQNAgent("dueling_dqn_agent", state_size, action_size, config)
        else:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
    
    def _create_environment(self, config: Dict[str, Any]) -> AgentEnvironmentSimulator:
        """创建环境"""
        env_type = config.get("type", "grid_world")
        
        if env_type == "grid_world":
            grid_size = config.get("grid_size", (4, 4))
            obstacles = config.get("obstacles", [])
            start_pos = config.get("start_position", (0, 0))
            goal_pos = config.get("goal_position", (3, 3))
            
            return GridWorldEnvironment(
                grid_size=grid_size,
                start_position=start_pos,
                goal_position=goal_pos,
                obstacles=obstacles
            )
        else:
            # 默认简单环境
            state_space = self.predefined_state_spaces.get("grid_4x4")
            action_space = self.predefined_action_spaces.get("discrete_4")
            return AgentEnvironmentSimulator("default", state_space, action_space)
    
    def _create_exploration_strategy(self, config: Dict[str, Any], action_size: int) -> ExplorationStrategy:
        """创建探索策略"""
        mode = ExplorationMode(config.get("mode", "epsilon_greedy"))
        
        exploration_config = ExplorationConfig(
            mode=mode,
            initial_exploration=config.get("initial_exploration", 1.0),
            final_exploration=config.get("final_exploration", 0.01),
            decay_steps=config.get("decay_steps", 10000)
        )
        
        if mode == ExplorationMode.EPSILON_GREEDY:
            return EpsilonGreedyStrategy(exploration_config, action_size)
        elif mode == ExplorationMode.DECAYING_EPSILON:
            return DecayingEpsilonGreedyStrategy(exploration_config, action_size)
        elif mode == ExplorationMode.UCB:
            return UCBStrategy(exploration_config, action_size)
        else:
            return EpsilonGreedyStrategy(exploration_config, action_size)
    
    def _create_reward_function(self, config: Dict[str, Any]) -> RewardFunction:
        """创建奖励函数"""
        from ai.reinforcement_learning.rewards.base import RewardConfig, RewardType
        
        reward_type = RewardType(config.get("type", "step"))
        
        reward_config = RewardConfig(
            reward_type=reward_type,
            parameters=config.get("parameters", {}),
            normalization=config.get("normalization", True),
            scaling_factor=config.get("scaling_factor", 1.0)
        )
        
        if reward_type == RewardType.LINEAR:
            return LinearReward(reward_config)
        elif reward_type == RewardType.STEP:
            return StepReward(reward_config)
        elif reward_type == RewardType.THRESHOLD:
            return ThresholdReward(reward_config)
        elif reward_type == RewardType.GAUSSIAN:
            return GaussianReward(reward_config)
        else:
            return StepReward(reward_config)
    
    async def start_training(self, session_id: str, training_config: Dict[str, Any]) -> bool:
        """开始训练"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"会话 {session_id} 不存在")
            
            session = self.active_sessions[session_id]
            
            if session.is_training:
                raise ValueError(f"会话 {session_id} 已经在训练中")
            
            # 创建训练配置
            train_config = TrainingConfig(
                max_episodes=training_config.get("max_episodes", 1000),
                max_steps_per_episode=training_config.get("max_steps_per_episode", 1000),
                evaluation_frequency=training_config.get("evaluation_frequency", 100),
                save_frequency=training_config.get("save_frequency", 500),
                initial_learning_rate=training_config.get("learning_rate", 0.001),
                early_stopping=training_config.get("early_stopping", True),
                patience=training_config.get("patience", 200)
            )
            
            # 创建训练管理器
            session.training_manager = TrainingManager(train_config, session.agent, session.environment)
            session.is_training = True
            
            # 在后台线程中启动训练
            session.training_task = asyncio.create_task(self._run_training(session))
            
            logger.info(f"开始训练会话: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"开始训练失败: {e}")
            raise
    
    async def _run_training(self, session: QLearningAgentSession):
        """运行训练（异步）"""
        try:
            loop = asyncio.get_event_loop()
            # 在线程池中运行训练
            session.training_metrics = await loop.run_in_executor(
                self.executor, 
                session.training_manager.train
            )
            session.is_training = False
            session.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"训练执行失败: {e}")
            session.is_training = False
            raise
    
    async def stop_training(self, session_id: str) -> bool:
        """停止训练"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"会话 {session_id} 不存在")
            
            session = self.active_sessions[session_id]
            
            if not session.is_training:
                return True
            
            if session.training_task:
                session.training_task.cancel()
                try:
                    await session.training_task
                except asyncio.CancelledError:
                    pass
            
            session.is_training = False
            session.last_updated = datetime.now()
            
            logger.info(f"停止训练会话: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"停止训练失败: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话 {session_id} 不存在")
        
        session = self.active_sessions[session_id]
        
        # 基本会话信息
        info = {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "is_training": session.is_training,
            "agent_type": session.agent.config.agent_type.value,
            "state_size": session.agent.state_size,
            "action_size": session.agent.action_size
        }
        
        # 训练指标
        if session.training_metrics:
            metrics = session.training_metrics
            info["training_metrics"] = {
                "episode": metrics.episode,
                "total_steps": metrics.total_steps,
                "training_time": metrics.training_time,
                "mean_reward": metrics.mean_reward,
                "std_reward": metrics.std_reward,
                "best_reward": metrics.best_reward,
                "best_episode": metrics.best_episode,
                "current_phase": metrics.current_phase.value,
                "early_stopped": metrics.early_stopped
            }
        
        # 智能体统计
        if hasattr(session.agent, 'get_statistics'):
            info["agent_statistics"] = session.agent.get_statistics()
        
        # 探索策略信息
        info["exploration"] = {
            "current_rate": session.exploration_strategy.get_exploration_rate(),
            "step_count": session.exploration_strategy.step_count
        }
        
        return info
    
    async def get_training_progress(self, session_id: str) -> Dict[str, Any]:
        """获取训练进度"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话 {session_id} 不存在")
        
        session = self.active_sessions[session_id]
        
        if not session.training_metrics:
            return {"progress": 0.0, "status": "not_started"}
        
        metrics = session.training_metrics
        training_manager = session.training_manager
        
        if training_manager:
            max_episodes = training_manager.config.max_episodes
            progress = min(metrics.episode / max_episodes, 1.0) if max_episodes > 0 else 0.0
        else:
            progress = 0.0
        
        # 最近的训练数据
        recent_rewards = metrics.episode_rewards[-10:] if metrics.episode_rewards else []
        recent_lengths = metrics.episode_lengths[-10:] if metrics.episode_lengths else []
        
        return {
            "progress": progress,
            "status": "training" if session.is_training else "stopped",
            "current_episode": metrics.episode,
            "total_steps": metrics.total_steps,
            "training_time": metrics.training_time,
            "current_performance": {
                "mean_reward": metrics.mean_reward,
                "std_reward": metrics.std_reward,
                "best_reward": metrics.best_reward
            },
            "recent_data": {
                "rewards": recent_rewards,
                "episode_lengths": recent_lengths,
                "learning_rates": metrics.learning_rates[-10:] if metrics.learning_rates else []
            },
            "phase": metrics.current_phase.value
        }
    
    async def predict_action(self, session_id: str, state: List[float]) -> Dict[str, Any]:
        """预测动作"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话 {session_id} 不存在")
        
        session = self.active_sessions[session_id]
        
        try:
            state_array = np.array(state)
            
            # 获取Q值
            if hasattr(session.agent, 'get_q_values'):
                q_values = session.agent.get_q_values(state_array)
            else:
                # 对于不支持直接获取Q值的智能体，使用act方法
                action = session.agent.act(state_array, evaluation=True)
                q_values = np.zeros(session.agent.action_size)
                q_values[action] = 1.0  # 简化表示
            
            # 使用探索策略选择动作
            action = session.exploration_strategy.select_action(q_values)
            
            return {
                "action": int(action),
                "q_values": q_values.tolist(),
                "exploration_rate": session.exploration_strategy.get_exploration_rate(),
                "confidence": float(np.max(q_values))
            }
            
        except Exception as e:
            logger.error(f"预测动作失败: {e}")
            raise
    
    async def evaluate_policy(self, session_id: str, num_episodes: int = 10) -> Dict[str, Any]:
        """评估策略"""
        if session_id not in self.active_sessions:
            raise ValueError(f"会话 {session_id} 不存在")
        
        session = self.active_sessions[session_id]
        
        try:
            # 在线程池中运行评估
            loop = asyncio.get_event_loop()
            evaluation_results = await loop.run_in_executor(
                self.executor,
                self._run_policy_evaluation,
                session,
                num_episodes
            )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"策略评估失败: {e}")
            raise
    
    def _run_policy_evaluation(self, session: QLearningAgentSession, num_episodes: int) -> Dict[str, Any]:
        """运行策略评估"""
        rewards = []
        episode_lengths = []
        successes = []
        
        for _ in range(num_episodes):
            state = session.environment.reset()
            total_reward = 0.0
            steps = 0
            success = False
            
            while steps < 1000:  # 最大步数限制
                action = session.agent.act(state, evaluation=True)
                next_state, reward, done, info = session.environment.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if info and info.get('success', False):
                    success = True
                
                if done:
                    break
            
            rewards.append(total_reward)
            episode_lengths.append(steps)
            successes.append(success)
        
        return {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": np.mean(successes),
            "rewards": rewards,
            "episode_lengths": episode_lengths
        }
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # 如果正在训练，先停止
        if session.is_training:
            await self.stop_training(session_id)
        
        # 删除会话
        del self.active_sessions[session_id]
        
        logger.info(f"删除会话: {session_id}")
        return True
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        sessions = []
        
        for session_id, session in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_updated": session.last_updated.isoformat(),
                "is_training": session.is_training,
                "agent_type": session.agent.config.agent_type.value
            })
        
        return sessions
    
    async def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "supported_algorithms": [
                {
                    "type": "tabular",
                    "name": "表格Q-Learning",
                    "description": "经典的表格式Q-Learning算法",
                    "suitable_for": "小规模离散状态空间"
                },
                {
                    "type": "dqn",
                    "name": "Deep Q-Network",
                    "description": "使用深度神经网络的Q-Learning",
                    "suitable_for": "大规模状态空间，连续状态"
                },
                {
                    "type": "double_dqn",
                    "name": "Double DQN",
                    "description": "解决过估计问题的DQN变体",
                    "suitable_for": "提高DQN的稳定性和性能"
                },
                {
                    "type": "dueling_dqn",
                    "name": "Dueling DQN",
                    "description": "分离状态价值和动作优势的DQN",
                    "suitable_for": "某些动作不影响状态价值的环境"
                }
            ],
            "exploration_strategies": [
                "epsilon_greedy",
                "decaying_epsilon",
                "upper_confidence_bound",
                "thompson_sampling",
                "boltzmann"
            ],
            "reward_functions": [
                "linear",
                "step",
                "threshold",
                "gaussian",
                "distance_based",
                "sparse"
            ]
        }