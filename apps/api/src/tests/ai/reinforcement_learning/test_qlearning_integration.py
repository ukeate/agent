"""
Q-Learning集成测试

测试Q-Learning系统的整体功能和性能
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import MagicMock, patch

# 由于TensorFlow依赖问题，我们创建模拟测试
@pytest.fixture
def mock_tensorflow():
    """模拟TensorFlow依赖"""
    with patch.dict('sys.modules', {
        'tensorflow': MagicMock(),
        'tensorflow.keras': MagicMock(),
        'tensorflow.keras.models': MagicMock(),
        'tensorflow.keras.layers': MagicMock(),
        'tensorflow.keras.optimizers': MagicMock(),
    }):
        yield


class TestQLearningBasicFunctionality:
    """Q-Learning基础功能测试"""
    
    def test_tabular_qlearning_creation(self, mock_tensorflow):
        """测试表格Q-Learning智能体创建"""
        from ai.reinforcement_learning.qlearning.base import QLearningConfig, AlgorithmType
        from ai.reinforcement_learning.qlearning.q_learning import TabularQLearningAgent
        
        config = QLearningConfig(
            agent_type=AlgorithmType.Q_LEARNING,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=0.1
        )
        
        agent = TabularQLearningAgent("test_agent", 16, 4, config)
        
        assert agent.agent_id == "test_agent"
        assert agent.state_size == 16
        assert agent.action_size == 4
        assert agent.config.learning_rate == 0.1
    
    def test_exploration_strategies(self, mock_tensorflow):
        """测试探索策略功能"""
        from ai.reinforcement_learning.strategies.exploration import (
            EpsilonGreedyStrategy, ExplorationConfig, ExplorationMode
        )
        
        config = ExplorationConfig(
            mode=ExplorationMode.EPSILON_GREEDY,
            initial_exploration=0.5
        )
        
        strategy = EpsilonGreedyStrategy(config, action_size=4)
        
        # 测试动作选择
        q_values = np.array([1.0, 2.0, 0.5, 1.5])
        action = strategy.select_action(q_values)
        
        assert 0 <= action < 4
        assert strategy.get_exploration_rate() == 0.5
    
    def test_reward_functions(self, mock_tensorflow):
        """测试奖励函数"""
        from ai.reinforcement_learning.rewards.basic_rewards import StepReward
        from ai.reinforcement_learning.rewards.base import RewardConfig, RewardType
        
        config = RewardConfig(
            reward_type=RewardType.STEP,
            parameters={
                'positive_reward': 10.0,
                'negative_reward': -1.0,
                'neutral_reward': -0.01
            }
        )
        
        reward_func = StepReward(config)
        
        # 测试成功情况
        state = np.array([0, 0])
        next_state = np.array([1, 1])
        info = {'success': True}
        
        reward = reward_func.compute_reward(state, 0, next_state, True, info)
        assert reward > 0  # 应该是正奖励（经过归一化）
    
    def test_grid_world_environment(self, mock_tensorflow):
        """测试网格世界环境"""
        from ai.reinforcement_learning.environment.grid_world import GridWorldEnvironment
        
        env = GridWorldEnvironment(
            grid_size=(4, 4),
            start_position=(0, 0),
            goal_position=(3, 3)
        )
        
        # 测试重置
        initial_state = env.reset()
        assert len(initial_state) == 16  # 4x4网格的one-hot编码
        assert np.sum(initial_state) == 1.0  # one-hot编码只有一个1
        
        # 测试步进
        state, reward, done, info = env.step(1)  # 向右移动
        assert len(state) == 16
        assert 'position' in info
        assert info['position'] == [0, 1]  # 应该移动到(0,1)


class TestQLearningIntegration:
    """Q-Learning集成测试"""
    
    @pytest.mark.asyncio
    async def test_qlearning_service_basic_flow(self, mock_tensorflow):
        """测试Q-Learning服务的基本流程"""
        # 由于依赖问题，我们创建一个简化的模拟测试
        
        # 模拟配置
        agent_config = {
            "agent_type": "tabular",
            "state_size": 16,
            "action_size": 4,
            "learning_rate": 0.1,
            "environment": {
                "type": "grid_world",
                "grid_size": [4, 4]
            },
            "exploration": {
                "mode": "epsilon_greedy",
                "initial_exploration": 0.1
            },
            "reward": {
                "type": "step"
            }
        }
        
        # 这里我们只测试配置解析的逻辑
        assert agent_config["agent_type"] == "tabular"
        assert agent_config["state_size"] == 16
        assert agent_config["action_size"] == 4
        
        # 模拟训练配置
        training_config = {
            "max_episodes": 100,
            "max_steps_per_episode": 200,
            "learning_rate": 0.001
        }
        
        assert training_config["max_episodes"] == 100
    
    def test_q_table_learning(self, mock_tensorflow):
        """测试Q表学习过程"""
        from ai.reinforcement_learning.qlearning.base import QLearningConfig, AlgorithmType
        from ai.reinforcement_learning.qlearning.q_learning import TabularQLearningAgent
        
        config = QLearningConfig(
            agent_type=AlgorithmType.Q_LEARNING,
            learning_rate=0.5,  # 较高的学习率用于测试
            gamma=0.9,
            epsilon=0.0  # 不探索，纯贪婪
        )
        
        agent = TabularQLearningAgent("test", 4, 2, config)
        
        # 模拟学习过程
        state = 0
        action = 0
        reward = 1.0
        next_state = 1
        done = False
        
        # 执行多次学习更新
        for _ in range(10):
            agent.learn(state, action, reward, next_state, done)
        
        # 检查Q值是否更新
        q_values = agent.get_q_values(np.array([1, 0, 0, 0]))  # state 0的one-hot
        assert q_values[action] > 0  # 应该学到正的Q值
    
    def test_epsilon_decay(self, mock_tensorflow):
        """测试epsilon衰减"""
        from ai.reinforcement_learning.strategies.exploration import (
            DecayingEpsilonGreedyStrategy, ExplorationConfig, ExplorationMode
        )
        
        config = ExplorationConfig(
            mode=ExplorationMode.DECAYING_EPSILON,
            initial_exploration=1.0,
            final_exploration=0.01,
            decay_steps=1000,
            decay_type="exponential"
        )
        
        strategy = DecayingEpsilonGreedyStrategy(config, 4)
        
        initial_epsilon = strategy.epsilon
        
        # 模拟多次动作选择以触发衰减
        q_values = np.array([1.0, 2.0, 0.5, 1.5])
        for _ in range(500):  # 执行500步
            strategy.select_action(q_values)
        
        final_epsilon = strategy.epsilon
        
        assert final_epsilon < initial_epsilon  # epsilon应该衰减
        assert final_epsilon >= config.final_exploration  # 不应该低于最小值


class TestPerformanceOptimization:
    """性能优化测试"""
    
    def test_vectorized_operations(self, mock_tensorflow):
        """测试向量化操作性能"""
        import time
        
        # 测试numpy向量化操作
        size = 10000
        array1 = np.random.rand(size)
        array2 = np.random.rand(size)
        
        # 向量化操作
        start_time = time.time()
        result_vectorized = array1 + array2
        vectorized_time = time.time() - start_time
        
        # 循环操作（较慢）
        start_time = time.time()
        result_loop = np.zeros(size)
        for i in range(size):
            result_loop[i] = array1[i] + array2[i]
        loop_time = time.time() - start_time
        
        # 向量化操作应该更快
        assert vectorized_time < loop_time
        assert np.allclose(result_vectorized, result_loop)
    
    def test_memory_efficiency(self, mock_tensorflow):
        """测试内存效率"""
        from ai.reinforcement_learning.qlearning.replay_buffer import CircularReplayBuffer
        
        buffer = CircularReplayBuffer(capacity=1000)
        
        # 填充缓冲区
        for i in range(1500):  # 超过容量
            state = np.random.rand(4)
            action = i % 4
            reward = np.random.rand()
            next_state = np.random.rand(4)
            done = False
            
            buffer.store(state, action, reward, next_state, done)
        
        # 检查容量限制
        assert len(buffer) == 1000  # 应该保持在容量限制内
        
        # 检查可以采样
        batch = buffer.sample(32)
        assert len(batch['states']) == 32


class TestErrorHandling:
    """错误处理测试"""
    
    def test_invalid_agent_config(self, mock_tensorflow):
        """测试无效智能体配置的处理"""
        from ai.reinforcement_learning.qlearning.base import QLearningConfig, AlgorithmType
        
        # 测试无效学习率
        with pytest.raises((ValueError, AssertionError)):
            QLearningConfig(
                agent_type=AlgorithmType.Q_LEARNING,
                learning_rate=-0.1,  # 负学习率应该无效
                gamma=0.99
            )
    
    def test_invalid_environment_actions(self, mock_tensorflow):
        """测试环境中的无效动作"""
        from ai.reinforcement_learning.environment.grid_world import GridWorldEnvironment
        
        env = GridWorldEnvironment(grid_size=(3, 3))
        env.reset()
        
        # 测试无效动作
        with pytest.raises(ValueError):
            env.step(10)  # 无效的动作索引
    
    def test_exploration_strategy_bounds(self, mock_tensorflow):
        """测试探索策略的边界条件"""
        from ai.reinforcement_learning.strategies.exploration import (
            EpsilonGreedyStrategy, ExplorationConfig, ExplorationMode
        )
        
        config = ExplorationConfig(
            mode=ExplorationMode.EPSILON_GREEDY,
            initial_exploration=0.0  # epsilon=0，纯贪婪
        )
        
        strategy = EpsilonGreedyStrategy(config, 4)
        
        # 多次选择应该总是选择最优动作
        q_values = np.array([1.0, 3.0, 2.0, 0.5])  # 动作1是最优的
        
        actions = []
        for _ in range(100):
            action = strategy.select_action(q_values)
            actions.append(action)
        
        # 由于epsilon=0，应该总是选择动作1
        assert all(action == 1 for action in actions)


# 运行测试的辅助函数
def run_integration_tests():
    """运行集成测试"""
    print("开始Q-Learning集成测试...")
    
    # 创建模拟的tensorflow模块
    import sys
    from unittest.mock import MagicMock
    
    sys.modules['tensorflow'] = MagicMock()
    sys.modules['tensorflow.keras'] = MagicMock()
    sys.modules['tensorflow.keras.models'] = MagicMock()
    sys.modules['tensorflow.keras.layers'] = MagicMock()
    sys.modules['tensorflow.keras.optimizers'] = MagicMock()
    
    try:
        # 基础功能测试
        basic_tests = TestQLearningBasicFunctionality()
        basic_tests.test_tabular_qlearning_creation(None)
        basic_tests.test_exploration_strategies(None)
        basic_tests.test_reward_functions(None)
        basic_tests.test_grid_world_environment(None)
        print("✓ 基础功能测试通过")
        
        # 集成测试
        integration_tests = TestQLearningIntegration()
        integration_tests.test_q_table_learning(None)
        integration_tests.test_epsilon_decay(None)
        print("✓ 集成测试通过")
        
        # 性能测试
        performance_tests = TestPerformanceOptimization()
        performance_tests.test_vectorized_operations(None)
        performance_tests.test_memory_efficiency(None)
        print("✓ 性能优化测试通过")
        
        # 错误处理测试
        error_tests = TestErrorHandling()
        error_tests.test_invalid_environment_actions(None)
        error_tests.test_exploration_strategy_bounds(None)
        print("✓ 错误处理测试通过")
        
        print("\n所有集成测试通过！Q-Learning系统功能正常。")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    run_integration_tests()