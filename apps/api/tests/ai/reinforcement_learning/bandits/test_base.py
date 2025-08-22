"""
测试多臂老虎机基类
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ai.reinforcement_learning.bandits.base import MultiArmedBandit


class TestableMultiArmedBandit(MultiArmedBandit):
    """用于测试的可实例化的多臂老虎机实现"""
    
    def select_arm(self, context=None):
        # 简单的随机选择用于测试
        return self.rng.integers(0, self.n_arms)
    
    def _update_algorithm_state(self, arm, reward, context=None):
        pass
    
    def _get_confidence_intervals(self):
        return np.ones(self.n_arms) * 0.1
    
    def _reset_algorithm_state(self):
        pass


class TestMultiArmedBandit:
    """多臂老虎机基类测试"""
    
    def test_initialization(self):
        """测试初始化"""
        bandit = TestableMultiArmedBandit(n_arms=5, random_state=42)
        
        assert bandit.n_arms == 5
        assert bandit.random_state == 42
        assert bandit.total_pulls == 0
        assert bandit.total_reward == 0.0
        assert len(bandit.n_pulls) == 5
        assert len(bandit.rewards) == 5
        assert np.all(bandit.n_pulls == 0)
        assert np.all(bandit.rewards == 0)
    
    def test_update(self):
        """测试更新方法"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        # 第一次更新
        bandit.update(arm=1, reward=0.8)
        
        assert bandit.n_pulls[1] == 1
        assert bandit.rewards[1] == 0.8
        assert bandit.total_pulls == 1
        assert bandit.total_reward == 0.8
        assert len(bandit.reward_history) == 1
        assert len(bandit.arm_history) == 1
        assert bandit.arm_history[0] == 1
        
        # 第二次更新
        bandit.update(arm=0, reward=0.5)
        
        assert bandit.n_pulls[0] == 1
        assert bandit.n_pulls[1] == 1
        assert bandit.rewards[0] == 0.5
        assert bandit.rewards[1] == 0.8
        assert bandit.total_pulls == 2
        assert bandit.total_reward == 1.3
    
    def test_get_arm_stats(self):
        """测试获取臂统计信息"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        # 更新一些数据
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=0, reward=0.8)
        bandit.update(arm=1, reward=0.3)
        
        stats = bandit.get_arm_stats()
        
        assert 'n_pulls' in stats
        assert 'total_rewards' in stats
        assert 'estimated_rewards' in stats
        assert 'confidence_intervals' in stats
        
        np.testing.assert_array_equal(stats['n_pulls'], [2, 1, 0])
        np.testing.assert_array_equal(stats['total_rewards'], [1.4, 0.3, 0])
        np.testing.assert_array_almost_equal(stats['estimated_rewards'], [0.7, 0.3, 0])
    
    def test_get_best_arm(self):
        """测试获取最佳臂"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        bandit.update(arm=0, reward=0.5)
        bandit.update(arm=1, reward=0.8)
        bandit.update(arm=2, reward=0.3)
        
        best_arm = bandit.get_best_arm()
        assert best_arm == 1  # 臂1有最高的平均奖励
    
    def test_calculate_regret(self):
        """测试遗憾计算"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        # 没有历史记录时遗憾为0
        regret = bandit.calculate_regret(optimal_reward=1.0)
        assert regret == 0.0
        
        # 有历史记录时计算遗憾
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        
        regret = bandit.calculate_regret(optimal_reward=1.0)
        expected_regret = (1.0 - 0.6) + (1.0 - 0.8)  # 0.4 + 0.2 = 0.6
        assert regret == expected_regret
    
    def test_get_performance_metrics(self):
        """测试性能指标"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        # 没有拉取时的指标
        metrics = bandit.get_performance_metrics()
        assert metrics['total_pulls'] == 0
        assert metrics['average_reward'] == 0.0
        assert metrics['best_arm'] == -1
        
        # 有拉取后的指标
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        bandit.update(arm=0, reward=0.4)
        
        metrics = bandit.get_performance_metrics()
        assert metrics['total_pulls'] == 3
        assert metrics['average_reward'] == 1.8 / 3
        assert metrics['best_arm'] == 1
        assert 'exploration_rate' in metrics
        assert 'arm_selection_distribution' in metrics
    
    def test_reset(self):
        """测试重置"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        # 添加一些数据
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        
        # 重置
        bandit.reset()
        
        assert bandit.total_pulls == 0
        assert bandit.total_reward == 0.0
        assert np.all(bandit.n_pulls == 0)
        assert np.all(bandit.rewards == 0)
        assert len(bandit.reward_history) == 0
        assert len(bandit.arm_history) == 0
    
    def test_select_arm_returns_valid_arm(self):
        """测试选择臂返回有效的臂索引"""
        bandit = TestableMultiArmedBandit(n_arms=5, random_state=42)
        
        for _ in range(10):
            arm = bandit.select_arm()
            assert 0 <= arm < 5
    
    def test_reproducible_random_behavior(self):
        """测试随机行为的可重现性"""
        bandit1 = TestableMultiArmedBandit(n_arms=3, random_state=42)
        bandit2 = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        arms1 = [bandit1.select_arm() for _ in range(10)]
        arms2 = [bandit2.select_arm() for _ in range(10)]
        
        assert arms1 == arms2
    
    def test_context_parameter(self):
        """测试上下文参数传递"""
        bandit = TestableMultiArmedBandit(n_arms=3, random_state=42)
        
        context = {"user_id": 123, "features": [0.1, 0.2, 0.3]}
        
        # 确保方法接受上下文参数而不出错
        arm = bandit.select_arm(context=context)
        assert 0 <= arm < 3
        
        bandit.update(arm=arm, reward=0.8, context=context)
        assert bandit.total_pulls == 1