"""
测试Epsilon-Greedy多臂老虎机算法
"""

import pytest
import numpy as np

from ai.reinforcement_learning.bandits.epsilon_greedy import EpsilonGreedyBandit


class TestEpsilonGreedyBandit:
    """Epsilon-Greedy多臂老虎机算法测试"""
    
    def test_initialization(self):
        """测试初始化"""
        bandit = EpsilonGreedyBandit(
            n_arms=5,
            epsilon=0.2,
            decay_rate=0.99,
            min_epsilon=0.05,
            random_state=42
        )
        
        assert bandit.n_arms == 5
        assert bandit.initial_epsilon == 0.2
        assert bandit.epsilon == 0.2
        assert bandit.use_decay == False
        assert bandit.decay_rate == 0.99
        assert bandit.min_epsilon == 0.05
        assert bandit.random_state == 42
    
    def test_decay_initialization(self):
        """测试衰减模式初始化"""
        bandit = EpsilonGreedyBandit(
            n_arms=3,
            epsilon="decay",
            decay_rate=0.95,
            min_epsilon=0.01,
            random_state=42
        )
        
        assert bandit.initial_epsilon == "decay"
        assert bandit.epsilon == 0.1  # 默认初始值
        assert bandit.use_decay == True
        assert bandit.decay_rate == 0.95
        assert bandit.min_epsilon == 0.01
    
    def test_select_arm_returns_valid_arm(self):
        """测试选择臂返回有效的臂索引"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1, random_state=42)
        
        for _ in range(20):
            arm = bandit.select_arm()
            assert 0 <= arm < 3
    
    def test_greedy_arm_selection(self):
        """测试贪婪臂选择"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.0, random_state=42)  # 完全贪婪
        
        # 给臂1最高的奖励
        bandit.update(arm=0, reward=0.3)
        bandit.update(arm=1, reward=0.8)  # 最高奖励
        bandit.update(arm=2, reward=0.5)
        
        # 在epsilon=0的情况下，应该总是选择最佳臂
        for _ in range(10):
            arm = bandit.select_arm()
            assert arm == 1  # 应该总是选择臂1
    
    def test_exploration_vs_exploitation(self):
        """测试探索与利用的平衡"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.5, random_state=42)  # 高探索率
        
        # 给臂0最高奖励
        for _ in range(10):
            bandit.update(arm=0, reward=0.9)
        
        # 给其他臂低奖励
        for _ in range(5):
            bandit.update(arm=1, reward=0.1)
            bandit.update(arm=2, reward=0.1)
        
        # 统计100次选择中的分布
        selections = [bandit.select_arm() for _ in range(100)]
        
        # 由于epsilon=0.5，应该有相当数量的随机探索
        # 但臂0应该仍然被选择得更多（贪婪选择时）
        arm_counts = np.bincount(selections, minlength=3)
        
        # 至少应该有一些探索（不应该100%选择臂0）
        assert arm_counts[1] > 0 or arm_counts[2] > 0
    
    def test_epsilon_decay(self):
        """测试epsilon衰减"""
        bandit = EpsilonGreedyBandit(
            n_arms=3,
            epsilon="decay",
            decay_rate=0.9,
            min_epsilon=0.05,
            random_state=42
        )
        
        initial_epsilon = bandit.epsilon
        
        # 多次选择臂来触发衰减
        for _ in range(10):
            arm = bandit.select_arm()
            bandit.update(arm, reward=0.5)
        
        # epsilon应该有所衰减
        assert bandit.epsilon < initial_epsilon
        assert bandit.epsilon >= bandit.min_epsilon
    
    def test_set_epsilon(self):
        """测试设置epsilon值"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1, random_state=42)
        
        bandit.set_epsilon(0.3)
        assert bandit.epsilon == 0.3
        
        # 测试边界值
        bandit.set_epsilon(-0.1)
        assert bandit.epsilon == 0.0
        
        bandit.set_epsilon(1.5)
        assert bandit.epsilon == 1.0
    
    def test_get_current_epsilon(self):
        """测试获取当前epsilon值"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.2, random_state=42)
        
        assert bandit.get_current_epsilon() == 0.2
        
        bandit.set_epsilon(0.4)
        assert bandit.get_current_epsilon() == 0.4
    
    def test_confidence_intervals(self):
        """测试置信区间计算"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1, random_state=42)
        
        # 给臂0更多数据
        for _ in range(20):
            bandit.update(arm=0, reward=0.7)
        
        # 给臂1少量数据
        for _ in range(2):
            bandit.update(arm=1, reward=0.3)
        
        confidence_intervals = bandit._get_confidence_intervals()
        
        assert len(confidence_intervals) == 3
        assert np.all(confidence_intervals >= 0)
        
        # 臂0有更多数据，应该有更小的置信区间
        assert confidence_intervals[0] < confidence_intervals[1]
        
        # 臂2没有数据，置信区间应该为0或很小
        assert confidence_intervals[2] >= 0
    
    def test_algorithm_params(self):
        """测试算法参数获取"""
        bandit = EpsilonGreedyBandit(
            n_arms=4,
            epsilon=0.15,
            decay_rate=0.98,
            min_epsilon=0.02,
            random_state=123
        )
        
        params = bandit.get_algorithm_params()
        
        assert params['algorithm'] == 'Epsilon-Greedy'
        assert params['initial_epsilon'] == 0.15
        assert params['current_epsilon'] == 0.15
        assert params['use_decay'] == False
        assert params['decay_rate'] == 0.98
        assert params['min_epsilon'] == 0.02
        assert params['n_arms'] == 4
        assert params['random_state'] == 123
    
    def test_exploration_stats(self):
        """测试探索统计信息"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.2, random_state=42)
        
        # 模拟一些选择
        for _ in range(10):
            arm = bandit.select_arm()
            bandit.update(arm, reward=0.5)
        
        stats = bandit.get_exploration_stats()
        
        assert 'current_epsilon' in stats
        assert 'estimated_exploration_rate' in stats
        assert 'estimated_exploitation_rate' in stats
        assert 'estimated_random_actions' in stats
        assert 'estimated_greedy_actions' in stats
        assert 'total_actions' in stats
        
        assert stats['current_epsilon'] == 0.2
        assert stats['estimated_exploration_rate'] == 0.2
        assert stats['estimated_exploitation_rate'] == 0.8
        assert stats['total_actions'] == 10
    
    def test_detailed_stats(self):
        """测试详细统计信息"""
        bandit = EpsilonGreedyBandit(n_arms=2, epsilon=0.1, random_state=42)
        
        bandit.update(arm=0, reward=0.7)
        bandit.update(arm=1, reward=0.3)
        
        stats = bandit.get_detailed_stats()
        
        # 检查包含的字段
        required_fields = [
            'n_pulls', 'total_rewards', 'estimated_rewards', 'confidence_intervals',
            'current_epsilon', 'estimated_exploration_rate', 'estimated_exploitation_rate',
            'algorithm_params', 'performance_metrics'
        ]
        
        for field in required_fields:
            assert field in stats
    
    def test_reset(self):
        """测试重置功能"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.2, random_state=42)
        
        # 添加一些数据
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        bandit.set_epsilon(0.5)  # 修改epsilon
        
        # 重置前检查状态
        assert bandit.total_pulls == 2
        assert bandit.epsilon == 0.5
        
        # 重置
        bandit.reset()
        
        # 重置后检查状态
        assert bandit.total_pulls == 0
        assert bandit.epsilon == 0.2  # 恢复到初始值
        assert np.all(bandit.n_pulls == 0)
        assert np.all(bandit.rewards == 0)
    
    def test_reset_with_decay(self):
        """测试衰减模式的重置"""
        bandit = EpsilonGreedyBandit(
            n_arms=2,
            epsilon="decay",
            decay_rate=0.9,
            random_state=42
        )
        
        # 让epsilon衰减
        for _ in range(5):
            arm = bandit.select_arm()
            bandit.update(arm, reward=0.5)
        
        decayed_epsilon = bandit.epsilon
        assert decayed_epsilon < 0.1  # 应该已经衰减
        
        # 重置
        bandit.reset()
        
        # epsilon应该重置为初始值
        assert bandit.epsilon == 0.1
    
    def test_unplayed_arms_priority(self):
        """测试优先选择未探索的臂"""
        bandit = EpsilonGreedyBandit(n_arms=4, epsilon=0.0, random_state=42)  # 完全贪婪
        
        # 即使在贪婪模式下，未探索的臂应该被优先选择
        bandit.update(arm=0, reward=0.9)  # 给臂0很高的奖励
        
        # 下一次选择应该选择未探索的臂（1、2、3中的一个）
        arm = bandit.select_arm()
        assert arm in [1, 2, 3]
    
    def test_tie_breaking_in_greedy_selection(self):
        """测试贪婪选择中的平局处理"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.0, random_state=42)
        
        # 给两个臂相同的奖励
        bandit.update(arm=0, reward=0.5)
        bandit.update(arm=1, reward=0.5)  # 相同奖励
        bandit.update(arm=2, reward=0.3)  # 更低奖励
        
        # 多次选择应该在臂0和臂1之间随机选择
        selections = [bandit.select_arm() for _ in range(100)]
        
        # 应该不会选择臂2（奖励最低）
        assert 2 not in selections
        
        # 应该包含臂0和臂1
        assert 0 in selections
        assert 1 in selections
    
    def test_string_representation(self):
        """测试字符串表示"""
        bandit = EpsilonGreedyBandit(n_arms=5, epsilon=0.15, random_state=42)
        
        str_repr = str(bandit)
        assert 'EpsilonGreedyBandit' in str_repr
        assert '5' in str_repr  # n_arms
        assert '0.150' in str_repr  # epsilon (保留3位小数)
        assert '0' in str_repr  # total_pulls
    
    def test_min_epsilon_constraint(self):
        """测试最小epsilon约束"""
        bandit = EpsilonGreedyBandit(
            n_arms=2,
            epsilon="decay",
            decay_rate=0.1,  # 很快衰减
            min_epsilon=0.05,
            random_state=42
        )
        
        # 多次选择来强制衰减
        for _ in range(100):
            arm = bandit.select_arm()
            bandit.update(arm, reward=0.5)
        
        # epsilon不应该低于最小值
        assert bandit.epsilon >= bandit.min_epsilon
        assert bandit.epsilon == bandit.min_epsilon