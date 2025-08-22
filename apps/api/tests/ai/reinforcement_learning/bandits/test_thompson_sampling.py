"""
测试Thompson Sampling多臂老虎机算法
"""

import pytest
import numpy as np

from ai.reinforcement_learning.bandits.thompson_sampling import ThompsonSamplingBandit


class TestThompsonSamplingBandit:
    """Thompson Sampling多臂老虎机算法测试"""
    
    def test_initialization(self):
        """测试初始化"""
        bandit = ThompsonSamplingBandit(
            n_arms=5, 
            alpha_init=2.0, 
            beta_init=3.0, 
            random_state=42
        )
        
        assert bandit.n_arms == 5
        assert bandit.alpha_init == 2.0
        assert bandit.beta_init == 3.0
        assert bandit.random_state == 42
        
        # 检查初始参数
        np.testing.assert_array_equal(bandit.alpha, [2.0] * 5)
        np.testing.assert_array_equal(bandit.beta, [3.0] * 5)
    
    def test_select_arm_returns_valid_arm(self):
        """测试选择臂返回有效的臂索引"""
        bandit = ThompsonSamplingBandit(n_arms=3, random_state=42)
        
        for _ in range(10):
            arm = bandit.select_arm()
            assert 0 <= arm < 3
    
    def test_update_with_binary_reward(self):
        """测试使用二项奖励更新"""
        bandit = ThompsonSamplingBandit(n_arms=3, alpha_init=1.0, beta_init=1.0, random_state=42)
        
        # 测试成功更新
        initial_alpha = bandit.alpha[0]
        bandit.update_with_binary_reward(arm=0, success=True)
        
        assert bandit.alpha[0] == initial_alpha + 1
        assert bandit.total_pulls == 1
        assert bandit.total_reward == 1.0
        
        # 测试失败更新
        initial_beta = bandit.beta[1]
        bandit.update_with_binary_reward(arm=1, success=False)
        
        assert bandit.beta[1] == initial_beta + 1
        assert bandit.total_pulls == 2
        assert bandit.total_reward == 1.0  # 仍然是1.0，因为第二次是失败
    
    def test_update_with_continuous_reward(self):
        """测试使用连续奖励更新"""
        bandit = ThompsonSamplingBandit(n_arms=2, alpha_init=1.0, beta_init=1.0, random_state=42)
        
        initial_alpha = bandit.alpha[0]
        initial_beta = bandit.beta[0]
        
        # 奖励0.8应该增加alpha更多
        bandit.update_with_continuous_reward(arm=0, reward=0.8, n_trials=1)
        
        assert bandit.alpha[0] > initial_alpha
        assert bandit.beta[0] > initial_beta
        assert bandit.alpha[0] - initial_alpha > bandit.beta[0] - initial_beta
    
    def test_standard_update_method(self):
        """测试标准更新方法（从基类继承）"""
        bandit = ThompsonSamplingBandit(n_arms=2, random_state=42)
        
        initial_alpha = bandit.alpha[0]
        initial_beta = bandit.beta[0]
        
        # 高奖励应该更新alpha
        bandit.update(arm=0, reward=0.8)
        assert bandit.alpha[0] == initial_alpha + 1
        assert bandit.beta[0] == initial_beta
        
        # 低奖励应该更新beta
        bandit.update(arm=1, reward=0.2)
        assert bandit.alpha[1] == initial_alpha  # 未改变
        assert bandit.beta[1] == initial_beta + 1
    
    def test_posterior_stats(self):
        """测试后验统计信息"""
        bandit = ThompsonSamplingBandit(n_arms=2, alpha_init=1.0, beta_init=1.0, random_state=42)
        
        # 更新一些数据
        bandit.update_with_binary_reward(arm=0, success=True)
        bandit.update_with_binary_reward(arm=0, success=True)
        bandit.update_with_binary_reward(arm=1, success=False)
        
        stats = bandit.get_posterior_stats()
        
        assert 'posterior_means' in stats
        assert 'posterior_variances' in stats
        assert 'alpha_params' in stats
        assert 'beta_params' in stats
        
        # 检查后验均值
        means = stats['posterior_means']
        assert len(means) == 2
        
        # 臂0有更多成功，应该有更高的后验均值
        assert means[0] > means[1]
        
        # 检查参数
        np.testing.assert_array_equal(stats['alpha_params'], [3.0, 1.0])  # 1+2成功
        np.testing.assert_array_equal(stats['beta_params'], [1.0, 2.0])   # 1+1失败
    
    def test_sample_from_posterior(self):
        """测试从后验分布采样"""
        bandit = ThompsonSamplingBandit(n_arms=2, random_state=42)
        
        samples = bandit.sample_from_posterior(n_samples=100)
        
        assert samples.shape == (100, 2)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)  # Beta分布的取值范围
    
    def test_get_arm_probabilities(self):
        """测试计算每个臂是最优臂的概率"""
        bandit = ThompsonSamplingBandit(n_arms=3, random_state=42)
        
        # 给臂0更多正面奖励
        for _ in range(10):
            bandit.update_with_binary_reward(arm=0, success=True)
        
        # 给其他臂一些负面奖励
        for _ in range(5):
            bandit.update_with_binary_reward(arm=1, success=False)
            bandit.update_with_binary_reward(arm=2, success=False)
        
        probabilities = bandit.get_arm_probabilities(n_samples=1000)
        
        assert len(probabilities) == 3
        assert np.sum(probabilities) == pytest.approx(1.0, rel=1e-2)
        
        # 臂0应该有最高的概率成为最优臂
        assert probabilities[0] > probabilities[1]
        assert probabilities[0] > probabilities[2]
    
    def test_confidence_intervals(self):
        """测试置信区间计算"""
        bandit = ThompsonSamplingBandit(n_arms=2, random_state=42)
        
        # 给一个臂更多数据来减少不确定性
        for _ in range(10):
            bandit.update_with_binary_reward(arm=0, success=True)
        
        bandit.update_with_binary_reward(arm=1, success=True)
        
        confidence_intervals = bandit._get_confidence_intervals()
        
        assert len(confidence_intervals) == 2
        assert np.all(confidence_intervals > 0)
        
        # 臂0有更多数据，应该有更小的置信区间
        assert confidence_intervals[0] < confidence_intervals[1]
    
    def test_algorithm_params(self):
        """测试算法参数获取"""
        bandit = ThompsonSamplingBandit(
            n_arms=4, 
            alpha_init=2.5, 
            beta_init=1.5, 
            random_state=123
        )
        
        params = bandit.get_algorithm_params()
        
        assert params['algorithm'] == 'Thompson Sampling'
        assert params['alpha_init'] == 2.5
        assert params['beta_init'] == 1.5
        assert params['n_arms'] == 4
        assert params['random_state'] == 123
    
    def test_detailed_stats(self):
        """测试详细统计信息"""
        bandit = ThompsonSamplingBandit(n_arms=2, random_state=42)
        
        bandit.update_with_binary_reward(arm=0, success=True)
        bandit.update_with_binary_reward(arm=1, success=False)
        
        stats = bandit.get_detailed_stats()
        
        # 检查包含的字段
        required_fields = [
            'n_pulls', 'total_rewards', 'estimated_rewards', 'confidence_intervals',
            'posterior_means', 'posterior_variances', 'alpha_params', 'beta_params',
            'arm_selection_probabilities', 'algorithm_params', 'performance_metrics'
        ]
        
        for field in required_fields:
            assert field in stats
    
    def test_reset(self):
        """测试重置功能"""
        bandit = ThompsonSamplingBandit(n_arms=2, alpha_init=2.0, beta_init=3.0, random_state=42)
        
        # 添加一些数据
        bandit.update_with_binary_reward(arm=0, success=True)
        bandit.update_with_binary_reward(arm=1, success=False)
        
        # 重置前检查状态
        assert bandit.total_pulls == 2
        assert not np.array_equal(bandit.alpha, [2.0, 2.0])
        
        # 重置
        bandit.reset()
        
        # 重置后检查状态
        assert bandit.total_pulls == 0
        np.testing.assert_array_equal(bandit.alpha, [2.0, 2.0])
        np.testing.assert_array_equal(bandit.beta, [3.0, 3.0])
    
    def test_reproducible_sampling(self):
        """测试采样的可重现性"""
        bandit1 = ThompsonSamplingBandit(n_arms=3, random_state=42)
        bandit2 = ThompsonSamplingBandit(n_arms=3, random_state=42)
        
        arms1 = [bandit1.select_arm() for _ in range(5)]
        arms2 = [bandit2.select_arm() for _ in range(5)]
        
        assert arms1 == arms2
    
    def test_edge_cases(self):
        """测试边缘情况"""
        # 单臂老虎机
        bandit = ThompsonSamplingBandit(n_arms=1, random_state=42)
        
        arm = bandit.select_arm()
        assert arm == 0
        
        bandit.update_with_binary_reward(arm=0, success=True)
        arm = bandit.select_arm()
        assert arm == 0
    
    def test_reward_clamping(self):
        """测试奖励值限制"""
        bandit = ThompsonSamplingBandit(n_arms=2, random_state=42)
        
        # 测试超出范围的奖励值
        initial_alpha = bandit.alpha[0]
        initial_beta = bandit.beta[0]
        
        # 负奖励应该被限制为0
        bandit.update_with_continuous_reward(arm=0, reward=-0.5, n_trials=1)
        
        # 应该只更新beta（因为奖励被限制为0）
        assert bandit.alpha[0] == initial_alpha
        assert bandit.beta[0] > initial_beta
        
        # 超过1的奖励应该被限制为1
        bandit.update_with_continuous_reward(arm=1, reward=1.5, n_trials=1)
        
        # 应该只更新alpha（因为奖励被限制为1）
        assert bandit.alpha[1] > initial_alpha
        assert bandit.beta[1] == initial_beta
    
    def test_string_representation(self):
        """测试字符串表示"""
        bandit = ThompsonSamplingBandit(
            n_arms=5, 
            alpha_init=2.0, 
            beta_init=1.5, 
            random_state=42
        )
        
        str_repr = str(bandit)
        assert 'ThompsonSamplingBandit' in str_repr
        assert '5' in str_repr  # n_arms
        assert '2.0' in str_repr  # alpha_init
        assert '1.5' in str_repr  # beta_init
        assert '0' in str_repr  # total_pulls