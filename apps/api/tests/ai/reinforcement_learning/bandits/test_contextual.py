"""
测试Linear Contextual Bandit多臂老虎机算法
"""

import pytest
import numpy as np

from ai.reinforcement_learning.bandits.contextual import LinearContextualBandit


class TestLinearContextualBandit:
    """Linear Contextual Bandit多臂老虎机算法测试"""
    
    def test_initialization(self):
        """测试初始化"""
        bandit = LinearContextualBandit(
            n_arms=3,
            n_features=5,
            alpha=2.0,
            lambda_reg=1.5,
            random_state=42
        )
        
        assert bandit.n_arms == 3
        assert bandit.n_features == 5
        assert bandit.alpha == 2.0
        assert bandit.lambda_reg == 1.5
        assert bandit.random_state == 42
        
        # 检查初始化的矩阵
        assert bandit.A.shape == (3, 5, 5)
        assert bandit.b.shape == (3, 5)
        assert bandit.theta.shape == (3, 5)
        
        # A应该被初始化为lambda_reg * I
        for i in range(3):
            expected_A = 1.5 * np.identity(5)
            np.testing.assert_array_equal(bandit.A[i], expected_A)
    
    def test_select_arm_requires_context(self):
        """测试选择臂需要上下文信息"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 没有上下文应该抛出异常
        with pytest.raises(ValueError, match="需要提供包含'features'的context"):
            bandit.select_arm()
        
        # 没有features字段应该抛出异常
        with pytest.raises(ValueError, match="需要提供包含'features'的context"):
            bandit.select_arm(context={})
    
    def test_select_arm_with_valid_context(self):
        """测试使用有效上下文选择臂"""
        bandit = LinearContextualBandit(n_arms=3, n_features=4, random_state=42)
        
        context = {"features": [0.1, 0.2, 0.3, 0.4]}
        
        arm = bandit.select_arm(context=context)
        assert 0 <= arm < 3
    
    def test_feature_dimension_validation(self):
        """测试特征维度验证"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 特征维度不匹配应该抛出异常
        context = {"features": [0.1, 0.2]}  # 只有2个特征，但期望3个
        
        with pytest.raises(ValueError, match="特征维度不匹配"):
            bandit.select_arm(context=context)
    
    def test_update_requires_context(self):
        """测试更新需要上下文信息"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 没有上下文的更新应该抛出异常
        with pytest.raises(ValueError, match="必须提供包含'features'的context"):
            bandit.update(arm=0, reward=0.5)
        
        # 没有features字段的更新应该抛出异常
        with pytest.raises(ValueError, match="必须提供包含'features'的context"):
            bandit.update(arm=0, reward=0.5, context={})
    
    def test_update_with_context(self):
        """测试使用上下文更新"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        context = {"features": [0.5, -0.3, 0.8]}
        features = np.array(context["features"])
        
        # 记录更新前的状态
        A_before = bandit.A[0].copy()
        b_before = bandit.b[0].copy()
        
        # 更新
        bandit.update(arm=0, reward=0.7, context=context)
        
        # 检查A矩阵更新：A = A + x*x^T
        expected_A = A_before + np.outer(features, features)
        np.testing.assert_array_almost_equal(bandit.A[0], expected_A)
        
        # 检查b向量更新：b = b + r*x
        expected_b = b_before + 0.7 * features
        np.testing.assert_array_almost_equal(bandit.b[0], expected_b)
        
        # 检查基类统计信息更新
        assert bandit.n_pulls[0] == 1
        assert bandit.rewards[0] == 0.7
        assert bandit.total_pulls == 1
    
    def test_predict_reward(self):
        """测试奖励预测"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 添加一些数据来训练模型
        context1 = {"features": [1.0, 0.0, 0.0]}
        context2 = {"features": [0.0, 1.0, 0.0]}
        
        bandit.update(arm=0, reward=0.8, context=context1)
        bandit.update(arm=0, reward=0.2, context=context2)
        
        # 预测奖励
        features_test = np.array([1.0, 0.0, 0.0])
        predicted_reward = bandit.predict_reward(arm=0, features=features_test)
        
        assert isinstance(predicted_reward, (int, float, np.number))
        
        # 测试无效臂索引
        with pytest.raises(ValueError, match="臂索引.*超出范围"):
            bandit.predict_reward(arm=5, features=features_test)
    
    def test_get_model_parameters(self):
        """测试获取模型参数"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 添加一些数据
        context = {"features": [0.5, -0.3, 0.8]}
        bandit.update(arm=0, reward=0.7, context=context)
        
        params = bandit.get_model_parameters()
        
        assert 'theta' in params
        assert 'A_matrices' in params
        assert 'b_vectors' in params
        
        assert params['theta'].shape == (2, 3)
        assert params['A_matrices'].shape == (2, 3, 3)
        assert params['b_vectors'].shape == (2, 3)
    
    def test_get_feature_importance(self):
        """测试特征重要性"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 添加一些数据
        context = {"features": [1.0, 0.5, -0.3]}
        bandit.update(arm=0, reward=0.8, context=context)
        
        importance = bandit.get_feature_importance()
        
        assert importance.shape == (2, 3)
        assert np.all(importance >= 0)  # 重要性应该是非负的（使用绝对值）
    
    def test_linear_ucb_calculation(self):
        """测试线性UCB值计算"""
        bandit = LinearContextualBandit(n_arms=2, n_features=2, alpha=1.0, random_state=42)
        
        # 简单的2D情况进行手动验证
        context = {"features": [1.0, 0.0]}
        features = np.array([1.0, 0.0])
        
        # 给臂0一些正向数据
        bandit.update(arm=0, reward=0.8, context=context)
        
        # 计算UCB值
        ucb_values = bandit._calculate_linear_ucb_values(features)
        
        assert len(ucb_values) == 2
        assert np.all(np.isfinite(ucb_values))
        
        # 臂0应该有正的预期奖励（因为有正向数据）
        # 臂1应该有0或接近0的预期奖励（没有数据）
        assert ucb_values[0] > 0  # 有数据支持的臂
    
    def test_algorithm_params(self):
        """测试算法参数获取"""
        bandit = LinearContextualBandit(
            n_arms=4,
            n_features=6,
            alpha=1.5,
            lambda_reg=2.0,
            random_state=123
        )
        
        params = bandit.get_algorithm_params()
        
        assert params['algorithm'] == 'Linear Contextual Bandit'
        assert params['n_arms'] == 4
        assert params['n_features'] == 6
        assert params['alpha'] == 1.5
        assert params['lambda_reg'] == 2.0
        assert params['random_state'] == 123
    
    def test_detailed_stats(self):
        """测试详细统计信息"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        context = {"features": [0.5, -0.3, 0.8]}
        bandit.update(arm=0, reward=0.7, context=context)
        
        stats = bandit.get_detailed_stats()
        
        # 检查包含的字段
        required_fields = [
            'n_pulls', 'total_rewards', 'estimated_rewards', 'confidence_intervals',
            'model_parameters', 'feature_importance', 'algorithm_params', 'performance_metrics'
        ]
        
        for field in required_fields:
            assert field in stats
    
    def test_reset(self):
        """测试重置功能"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, lambda_reg=1.5, random_state=42)
        
        # 添加一些数据
        context = {"features": [0.5, -0.3, 0.8]}
        bandit.update(arm=0, reward=0.7, context=context)
        
        # 重置前检查状态
        assert bandit.total_pulls == 1
        assert not np.array_equal(bandit.A[0], 1.5 * np.identity(3))
        
        # 重置
        bandit.reset()
        
        # 重置后检查状态
        assert bandit.total_pulls == 0
        np.testing.assert_array_equal(bandit.A[0], 1.5 * np.identity(3))
        np.testing.assert_array_equal(bandit.b[0], np.zeros(3))
        np.testing.assert_array_equal(bandit.theta[0], np.zeros(3))
    
    def test_singular_matrix_handling(self):
        """测试奇异矩阵的处理"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, lambda_reg=0.01, random_state=42)
        
        # 使用相同的特征多次更新，可能导致近似奇异矩阵
        context = {"features": [1.0, 0.0, 0.0]}
        
        for _ in range(100):
            bandit.update(arm=0, reward=0.5, context=context)
        
        # 即使矩阵接近奇异，选择臂也应该工作（使用伪逆）
        arm = bandit.select_arm(context=context)
        assert 0 <= arm < 2
        
        # 预测也应该工作
        features = np.array([1.0, 0.0, 0.0])
        predicted_reward = bandit.predict_reward(arm=0, features=features)
        assert np.isfinite(predicted_reward)
    
    def test_different_context_features(self):
        """测试不同上下文特征的处理"""
        bandit = LinearContextualBandit(n_arms=2, n_features=4, random_state=42)
        
        contexts = [
            {"features": [1.0, 0.0, 0.0, 0.0], "user_id": 1},
            {"features": [0.0, 1.0, 0.0, 0.0], "user_id": 2}, 
            {"features": [0.0, 0.0, 1.0, 0.0], "user_id": 3},
            {"features": [0.0, 0.0, 0.0, 1.0], "user_id": 4}
        ]
        
        rewards = [0.8, 0.6, 0.4, 0.2]
        
        # 更新不同的上下文
        for i, (context, reward) in enumerate(zip(contexts, rewards)):
            arm = bandit.select_arm(context=context)
            bandit.update(arm=arm, reward=reward, context=context)
        
        assert bandit.total_pulls == 4
        
        # 预测应该反映学到的模式
        for context, expected_reward in zip(contexts, rewards):
            for arm in range(2):
                predicted = bandit.predict_reward(arm=arm, features=np.array(context["features"]))
                assert np.isfinite(predicted)
    
    def test_confidence_intervals_context_dependent(self):
        """测试上下文相关的置信区间"""
        bandit = LinearContextualBandit(n_arms=2, n_features=3, random_state=42)
        
        # 添加一些数据
        context = {"features": [0.5, -0.3, 0.8]}
        bandit.update(arm=0, reward=0.7, context=context)
        
        confidence_intervals = bandit._get_confidence_intervals()
        
        assert len(confidence_intervals) == 2
        assert np.all(confidence_intervals >= 0)
        
        # 有数据的臂应该有不同的置信区间
        # （这是基于单位向量的平均估计）
    
    def test_string_representation(self):
        """测试字符串表示"""
        bandit = LinearContextualBandit(
            n_arms=5,
            n_features=8, 
            alpha=1.5,
            random_state=42
        )
        
        str_repr = str(bandit)
        assert 'LinearContextualBandit' in str_repr
        assert '5' in str_repr  # n_arms
        assert '8' in str_repr  # n_features
        assert '1.5' in str_repr  # alpha
        assert '0' in str_repr  # total_pulls