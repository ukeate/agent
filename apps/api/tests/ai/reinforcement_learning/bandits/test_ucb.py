"""
测试UCB多臂老虎机算法
"""

import pytest
import numpy as np
import math

from ai.reinforcement_learning.bandits.ucb import UCBBandit


class TestUCBBandit:
    """UCB多臂老虎机算法测试"""
    
    def test_initialization(self):
        """测试初始化"""
        bandit = UCBBandit(n_arms=5, c=2.5, random_state=42)
        
        assert bandit.n_arms == 5
        assert bandit.c == 2.5
        assert bandit.random_state == 42
        assert bandit.total_pulls == 0
    
    def test_select_arm_unplayed_arms_first(self):
        """测试优先选择未探索的臂"""
        bandit = UCBBandit(n_arms=3, c=2.0, random_state=42)
        
        # 前3次选择应该选择每个臂一次
        selected_arms = set()
        for _ in range(3):
            arm = bandit.select_arm()
            assert 0 <= arm < 3
            selected_arms.add(arm)
            bandit.update(arm, reward=0.5)
        
        # 确保所有臂都被选择过
        assert len(selected_arms) == 3
    
    def test_ucb_calculation(self):
        """测试UCB值计算"""
        bandit = UCBBandit(n_arms=2, c=2.0, random_state=42)
        
        # 更新一些数据
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=0, reward=0.8)  # 臂0平均奖励: 0.7, 拉取2次
        bandit.update(arm=1, reward=0.3)  # 臂1平均奖励: 0.3, 拉取1次
        
        ucb_values = bandit.get_ucb_values()
        
        # 手动计算预期的UCB值
        total_pulls = 3
        log_t = math.log(total_pulls)
        
        # 臂0: 平均奖励0.7，拉取2次
        expected_ucb_0 = 0.7 + 2.0 * math.sqrt(log_t / 2)
        
        # 臂1: 平均奖励0.3，拉取1次  
        expected_ucb_1 = 0.3 + 2.0 * math.sqrt(log_t / 1)
        
        np.testing.assert_almost_equal(ucb_values[0], expected_ucb_0, decimal=6)
        np.testing.assert_almost_equal(ucb_values[1], expected_ucb_1, decimal=6)
    
    def test_select_arm_after_exploration(self):
        """测试探索阶段后的臂选择"""
        bandit = UCBBandit(n_arms=3, c=2.0, random_state=42)
        
        # 初始探索每个臂
        bandit.update(arm=0, reward=0.2)  # 低奖励
        bandit.update(arm=1, reward=0.8)  # 高奖励  
        bandit.update(arm=2, reward=0.5)  # 中等奖励
        
        # 下一次选择应该基于UCB值
        arm = bandit.select_arm()
        assert 0 <= arm < 3
        
        # 由于UCB考虑置信区间，可能不总是选择平均奖励最高的臂
        # 但应该倾向于选择奖励较高或不确定性较大的臂
    
    def test_confidence_intervals(self):
        """测试置信区间计算"""
        bandit = UCBBandit(n_arms=2, c=1.0, random_state=42)
        
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        bandit.update(arm=1, reward=0.4)  # 臂1有更多数据，置信区间应该更小
        
        confidence_intervals = bandit._get_confidence_intervals()
        
        # 臂0拉取1次，臂1拉取2次
        # 臂0的置信区间应该大于臂1的
        assert confidence_intervals[0] > confidence_intervals[1]
        
        # 置信区间应该都是正数
        assert np.all(confidence_intervals > 0)
    
    def test_algorithm_params(self):
        """测试算法参数获取"""
        bandit = UCBBandit(n_arms=4, c=1.5, random_state=123)
        
        params = bandit.get_algorithm_params()
        
        assert params['algorithm'] == 'UCB'
        assert params['c'] == 1.5
        assert params['n_arms'] == 4
        assert params['random_state'] == 123
    
    def test_detailed_stats(self):
        """测试详细统计信息"""
        bandit = UCBBandit(n_arms=2, c=2.0, random_state=42)
        
        bandit.update(arm=0, reward=0.7)
        bandit.update(arm=1, reward=0.3)
        
        stats = bandit.get_detailed_stats()
        
        # 检查包含的字段
        assert 'n_pulls' in stats
        assert 'total_rewards' in stats
        assert 'estimated_rewards' in stats
        assert 'confidence_intervals' in stats
        assert 'ucb_values' in stats
        assert 'algorithm_params' in stats
        assert 'performance_metrics' in stats
        
        # 检查UCB值
        assert len(stats['ucb_values']) == 2
        assert np.all(np.isfinite(stats['ucb_values']))
    
    def test_reset(self):
        """测试重置功能"""
        bandit = UCBBandit(n_arms=3, c=1.0, random_state=42)
        
        # 添加一些数据
        bandit.update(arm=0, reward=0.6)
        bandit.update(arm=1, reward=0.8)
        
        # 重置前检查状态
        assert bandit.total_pulls == 2
        
        # 重置
        bandit.reset()
        
        # 重置后检查状态
        assert bandit.total_pulls == 0
        assert np.all(bandit.n_pulls == 0)
        assert np.all(bandit.rewards == 0)
        
        # UCB值应该重新计算
        ucb_values = bandit.get_ucb_values()
        assert len(ucb_values) == 3
    
    def test_edge_cases(self):
        """测试边缘情况"""
        bandit = UCBBandit(n_arms=1, c=1.0, random_state=42)
        
        # 只有一个臂时应该总是选择它
        arm = bandit.select_arm()
        assert arm == 0
        
        bandit.update(arm=0, reward=0.5)
        arm = bandit.select_arm()
        assert arm == 0
    
    def test_zero_pulls_handling(self):
        """测试零拉取次数的处理"""
        bandit = UCBBandit(n_arms=2, c=1.0, random_state=42)
        
        # 没有拉取时的UCB值应该是无穷大或很大的数
        ucb_values = bandit.get_ucb_values()
        assert len(ucb_values) == 2
        assert np.all(np.isinf(ucb_values))
    
    def test_c_parameter_effect(self):
        """测试c参数对探索的影响"""
        # 创建两个c值不同的老虎机
        bandit_low_c = UCBBandit(n_arms=3, c=0.1, random_state=42)
        bandit_high_c = UCBBandit(n_arms=3, c=10.0, random_state=42)
        
        # 给两个老虎机相同的经验
        for bandit in [bandit_low_c, bandit_high_c]:
            bandit.update(arm=0, reward=0.9)  # 高奖励臂
            bandit.update(arm=1, reward=0.1)  # 低奖励臂
            bandit.update(arm=2, reward=0.1)  # 低奖励臂
        
        # 获取UCB值
        ucb_low = bandit_low_c.get_ucb_values()
        ucb_high = bandit_high_c.get_ucb_values()
        
        # 高c值应该导致更大的置信区间
        # 因此除了最佳臂外，其他臂的UCB值差异应该更大
        assert ucb_high[1] > ucb_low[1]
        assert ucb_high[2] > ucb_low[2]
    
    def test_string_representation(self):
        """测试字符串表示"""
        bandit = UCBBandit(n_arms=5, c=2.0, random_state=42)
        
        str_repr = str(bandit)
        assert 'UCBBandit' in str_repr
        assert '5' in str_repr  # n_arms
        assert '2.0' in str_repr  # c
        assert '0' in str_repr  # total_pulls