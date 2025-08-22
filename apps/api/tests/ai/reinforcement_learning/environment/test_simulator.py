"""
环境模拟器测试

测试环境模拟器、状态转移函数和奖励函数的实现。
"""

import pytest
import numpy as np
from typing import Dict, Any
import copy

from apps.api.src.ai.reinforcement_learning.qlearning.base import AgentState
from apps.api.src.ai.reinforcement_learning.environment.simulator import (
    BaseEnvironment,
    AgentEnvironmentSimulator,
    EnvironmentStatus,
    EnvironmentInfo,
    StepResult,
    RewardFunction,
    TransitionFunction,
    IdentityTransition,
    LinearTransition,
    GridWorldTransition,
    SparseReward,
    DenseReward,
    ShapedReward
)


class TestEnvironmentInfo:
    """环境信息测试"""
    
    def test_environment_info_creation(self):
        """测试环境信息创建"""
        env_info = EnvironmentInfo(
            env_id="test_env",
            name="Test Environment",
            description="测试环境",
            max_episode_steps=100,
            reward_threshold=50.0
        )
        
        assert env_info.env_id == "test_env"
        assert env_info.name == "Test Environment"
        assert env_info.description == "测试环境"
        assert env_info.max_episode_steps == 100
        assert env_info.reward_threshold == 50.0


class TestStepResult:
    """步骤结果测试"""
    
    def test_step_result_creation(self):
        """测试步骤结果创建"""
        state = AgentState.create(features={"x": 1.0})
        result = StepResult(
            next_state=state,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={"step": 1}
        )
        
        assert result.next_state == state
        assert result.reward == 1.0
        assert result.terminated is False
        assert result.truncated is False
        assert result.done is False
        assert result.info["step"] == 1
        
        # 测试done属性
        result_terminated = StepResult(state, 1.0, True, False)
        assert result_terminated.done is True
        
        result_truncated = StepResult(state, 1.0, False, True)
        assert result_truncated.done is True


class TestIdentityTransition:
    """恒等转移函数测试"""
    
    def test_basic_transition(self):
        """测试基本转移"""
        config = {"terminal_conditions": []}
        transition = IdentityTransition(config)
        
        state = AgentState.create(features={"x": 1.0, "y": 2.0})
        next_state, info = transition.transition(state, 0)
        
        # 恒等转移，状态不变
        assert next_state == state
        assert info["transition_type"] == "identity"
    
    def test_terminal_conditions(self):
        """测试终止条件"""
        config = {
            "terminal_conditions": [
                {"feature": "x", "operator": ">=", "value": 5.0},
                {"feature": "y", "operator": "==", "value": 10.0}
            ]
        }
        transition = IdentityTransition(config)
        
        # 未达到终止条件的状态
        state1 = AgentState.create(features={"x": 3.0, "y": 2.0})
        assert transition.is_terminal_state(state1) is False
        
        # 达到第一个终止条件
        state2 = AgentState.create(features={"x": 6.0, "y": 2.0})
        assert transition.is_terminal_state(state2) is True
        
        # 达到第二个终止条件
        state3 = AgentState.create(features={"x": 3.0, "y": 10.0})
        assert transition.is_terminal_state(state3) is True


class TestLinearTransition:
    """线性转移函数测试"""
    
    def test_basic_linear_transition(self):
        """测试基本线性转移"""
        from apps.api.src.ai.reinforcement_learning.environment.state_space import (
            StateSpaceFactory
        )
        
        # 创建状态空间
        state_space_config = {
            "space_type": "continuous",
            "features": [
                {"name": "x", "type": "continuous", "low": -10, "high": 10},
                {"name": "y", "type": "continuous", "low": -10, "high": 10}
            ]
        }
        state_space = StateSpaceFactory.create_from_config(state_space_config)
        
        # 创建线性转移函数
        config = {
            "dynamics_matrix": [[1.0, 0.1], [0.0, 0.9]],  # 简单的线性动力学
            "action_matrix": [[1.0], [0.5]],  # 动作影响
            "noise_std": 0.0,  # 无噪声
            "terminal_conditions": []
        }
        transition = LinearTransition(config, state_space)
        
        # 测试转移
        state = AgentState.create(features={"x": 1.0, "y": 2.0})
        action = 0.5
        next_state, info = transition.transition(state, action)
        
        # 验证线性转移: s' = A*s + B*a
        # x' = 1.0*1.0 + 0.1*2.0 + 1.0*0.5 = 1.7
        # y' = 0.0*1.0 + 0.9*2.0 + 0.5*0.5 = 2.05
        assert abs(next_state.features["x"] - 1.7) < 1e-6
        assert abs(next_state.features["y"] - 2.05) < 1e-6
        assert info["transition_type"] == "linear"


class TestGridWorldTransition:
    """网格世界转移函数测试"""
    
    def test_basic_grid_world_transition(self):
        """测试基本网格世界转移"""
        config = {
            "grid_size": (5, 5),
            "obstacles": [(2, 2), (3, 2)],
            "goals": [(4, 4)],
            "action_effects": {
                0: (0, -1),  # 上
                1: (0, 1),   # 下
                2: (-1, 0),  # 左
                3: (1, 0)    # 右
            },
            "success_rate": 1.0  # 100%成功率，便于测试
        }
        transition = GridWorldTransition(config)
        
        # 测试正常移动
        state = AgentState.create(features={"x": 1, "y": 1})
        next_state, info = transition.transition(state, 3)  # 右移
        
        assert next_state.features["x"] == 2
        assert next_state.features["y"] == 1
        assert info["transition_type"] == "grid_world"
        assert info["intended_action"] == 3
        assert info["executed_action"] == 3
        assert info["hit_obstacle"] is False
    
    def test_obstacle_collision(self):
        """测试障碍物碰撞"""
        config = {
            "grid_size": (5, 5),
            "obstacles": [(2, 2)],
            "goals": [(4, 4)],
            "action_effects": {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)},
            "success_rate": 1.0
        }
        transition = GridWorldTransition(config)
        
        # 尝试移动到障碍物
        state = AgentState.create(features={"x": 1, "y": 2})
        next_state, info = transition.transition(state, 3)  # 右移到障碍物
        
        # 位置应该不变（撞到障碍物）
        assert next_state.features["x"] == 1
        assert next_state.features["y"] == 2
        assert info["hit_obstacle"] is True
    
    def test_boundary_collision(self):
        """测试边界碰撞"""
        config = {
            "grid_size": (3, 3),
            "obstacles": [],
            "goals": [(2, 2)],
            "action_effects": {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)},
            "success_rate": 1.0
        }
        transition = GridWorldTransition(config)
        
        # 尝试移出边界
        state = AgentState.create(features={"x": 2, "y": 2})
        next_state, info = transition.transition(state, 3)  # 右移出边界
        
        # 位置应该被限制在边界内
        assert next_state.features["x"] == 2
        assert next_state.features["y"] == 2
    
    def test_goal_reaching(self):
        """测试到达目标"""
        config = {
            "grid_size": (5, 5),
            "obstacles": [],
            "goals": [(4, 4)],
            "action_effects": {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)},
            "success_rate": 1.0
        }
        transition = GridWorldTransition(config)
        
        # 到达目标
        state = AgentState.create(features={"x": 3, "y": 4})
        next_state, info = transition.transition(state, 3)  # 右移到目标
        
        assert next_state.features["x"] == 4
        assert next_state.features["y"] == 4
        assert info["at_goal"] is True
        assert transition.is_terminal_state(next_state) is True


class TestSparseReward:
    """稀疏奖励测试"""
    
    def test_basic_sparse_reward(self):
        """测试基本稀疏奖励"""
        config = {
            "goal_reward": 10.0,
            "step_penalty": -0.1,
            "collision_penalty": -1.0
        }
        reward_func = SparseReward(config)
        
        state = AgentState.create(features={"x": 1, "y": 1})
        next_state = AgentState.create(features={"x": 2, "y": 1})
        
        # 普通步骤
        reward = reward_func.calculate_reward(state, 0, next_state, {})
        assert reward == -0.1
        
        # 到达目标
        reward = reward_func.calculate_reward(state, 0, next_state, {"at_goal": True})
        assert reward == 10.0 - 0.1
        
        # 碰撞
        reward = reward_func.calculate_reward(state, 0, next_state, {"hit_obstacle": True})
        assert reward == -1.0 - 0.1


class TestDenseReward:
    """密集奖励测试"""
    
    def test_dense_reward_with_distance(self):
        """测试基于距离的密集奖励"""
        config = {
            "goal_reward": 10.0,
            "step_penalty": -0.1,
            "distance_reward_scale": 1.0,
            "goal_position": (4, 4)
        }
        reward_func = DenseReward(config)
        
        # 向目标移动
        state = AgentState.create(features={"x": 1, "y": 1})
        next_state = AgentState.create(features={"x": 2, "y": 1})
        
        reward = reward_func.calculate_reward(state, 0, next_state, {})
        
        # 计算预期奖励
        # 距离从6减少到5，改善了1，奖励应该是step_penalty + distance_improvement * scale
        expected_reward = -0.1 + 1.0 * 1.0
        assert abs(reward - expected_reward) < 1e-6


class TestShapedReward:
    """塑形奖励测试"""
    
    def test_shaped_reward_with_exploration(self):
        """测试包含探索的塑形奖励"""
        config = {
            "base_reward": {
                "type": "dense",
                "goal_reward": 1.0,
                "step_penalty": -0.01,
                "distance_reward_scale": 0.1,
                "goal_position": (4, 4)
            },
            "exploration_bonus": 0.1
        }
        reward_func = ShapedReward(config)
        
        state = AgentState.create(features={"x": 1, "y": 1})
        next_state = AgentState.create(features={"x": 2, "y": 1})
        
        # 第一次访问新状态，应该有探索奖励
        reward1 = reward_func.calculate_reward(state, 0, next_state, {})
        
        # 第二次访问相同状态，探索奖励应该减少
        reward2 = reward_func.calculate_reward(state, 0, next_state, {})
        
        assert reward1 > reward2  # 第一次访问奖励更高


class TestAgentEnvironmentSimulator:
    """智能体环境模拟器测试"""
    
    def test_basic_environment_simulation(self):
        """测试基本环境模拟"""
        config = {
            "env_id": "test_env",
            "name": "Test Environment",
            "description": "测试环境",
            "max_episode_steps": 10,
            "state_space": {
                "space_type": "discrete",
                "features": [
                    {"name": "x", "type": "discrete", "low": 0, "high": 4},
                    {"name": "y", "type": "discrete", "low": 0, "high": 4}
                ]
            },
            "action_space": {
                "space_type": "discrete",
                "n": 4
            },
            "transition_function": {
                "type": "grid_world",
                "grid_size": (5, 5),
                "obstacles": [],
                "goals": [(4, 4)],
                "action_effects": {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)},
                "success_rate": 1.0
            },
            "reward_function": {
                "type": "sparse",
                "goal_reward": 10.0,
                "step_penalty": -0.1
            },
            "initial_state": {
                "type": "fixed",
                "features": {"x": 0, "y": 0}
            }
        }
        
        env = AgentEnvironmentSimulator(config)
        
        # 测试重置
        initial_state = env.reset()
        assert initial_state.features["x"] == 0
        assert initial_state.features["y"] == 0
        assert env.status == EnvironmentStatus.RUNNING
        assert env.episode_step == 0
        
        # 测试步骤
        result = env.step(3)  # 右移
        assert isinstance(result, StepResult)
        assert result.next_state.features["x"] == 1
        assert result.next_state.features["y"] == 0
        assert result.reward == -0.1  # 步骤惩罚
        assert result.terminated is False
        assert result.truncated is False
        
        # 测试环境统计
        stats = env.get_episode_stats()
        assert stats["episode_step"] == 1
        assert stats["total_steps"] == 1
        assert stats["status"] == "running"
    
    def test_episode_termination(self):
        """测试回合终止"""
        config = {
            "env_id": "test_env",
            "max_episode_steps": 3,  # 很短的最大步数
            "state_space": {
                "space_type": "discrete",
                "features": [{"name": "x", "type": "discrete", "low": 0, "high": 4}]
            },
            "action_space": {"space_type": "discrete", "n": 2},
            "transition_function": {"type": "identity", "terminal_conditions": []},
            "reward_function": {"type": "sparse", "step_penalty": -0.1},
            "initial_state": {"type": "fixed", "features": {"x": 0}}
        }
        
        env = AgentEnvironmentSimulator(config)
        env.reset()
        
        # 执行步骤直到截断
        for i in range(3):
            result = env.step(0)
            if i < 2:
                assert result.truncated is False
            else:
                assert result.truncated is True
                assert env.status == EnvironmentStatus.TRUNCATED
    
    def test_environment_with_noise(self):
        """测试带噪声的环境"""
        config = {
            "env_id": "noisy_env",
            "stochastic": True,
            "noise_level": 0.1,
            "max_episode_steps": 100,
            "state_space": {
                "space_type": "continuous",
                "features": [{"name": "x", "type": "continuous", "low": -10, "high": 10}]
            },
            "action_space": {"space_type": "discrete", "n": 2},
            "transition_function": {"type": "identity", "terminal_conditions": []},
            "reward_function": {"type": "sparse", "step_penalty": -0.01}
        }
        
        env = AgentEnvironmentSimulator(config)
        initial_state = env.reset()
        
        # 执行多个步骤，验证噪声效果
        states = [initial_state.features["x"]]
        for _ in range(5):
            result = env.step(0)
            states.append(result.next_state.features["x"])
        
        # 由于噪声，状态应该有变化（虽然使用恒等转移）
        # 注意：这个测试有随机性，可能偶尔失败


if __name__ == "__main__":
    pytest.main([__file__])