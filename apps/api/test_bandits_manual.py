#!/usr/bin/env python3
"""
手动测试多臂老虎机算法实现
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from ai.reinforcement_learning.bandits.ucb import UCBBandit
from ai.reinforcement_learning.bandits.thompson_sampling import ThompsonSamplingBandit
from ai.reinforcement_learning.bandits.epsilon_greedy import EpsilonGreedyBandit
from ai.reinforcement_learning.bandits.contextual import LinearContextualBandit


def test_ucb_bandit():
    """测试UCB算法"""
    print("=== 测试UCB算法 ===")
    bandit = UCBBandit(n_arms=3, c=2.0, random_state=42)
    print(f"初始化: {bandit}")
    
    # 模拟一些交互
    for i in range(10):
        arm = bandit.select_arm()
        reward = np.random.beta(2, 5) if arm == 0 else np.random.beta(1, 3)  # 臂0有更好的奖励
        bandit.update(arm, reward)
        print(f"轮次 {i+1}: 选择臂 {arm}, 奖励 {reward:.3f}")
    
    stats = bandit.get_detailed_stats()
    print(f"最终统计: 平均奖励 {stats['estimated_rewards']}")
    print(f"UCB值: {stats['ucb_values']}")
    print(f"最佳臂: {bandit.get_best_arm()}")
    print()


def test_thompson_sampling():
    """测试Thompson Sampling算法"""
    print("=== 测试Thompson Sampling算法 ===")
    bandit = ThompsonSamplingBandit(n_arms=3, random_state=42)
    print(f"初始化: {bandit}")
    
    # 模拟一些交互
    for i in range(10):
        arm = bandit.select_arm()
        success = np.random.random() < (0.7 if arm == 1 else 0.3)  # 臂1有更高成功率
        bandit.update_with_binary_reward(arm, success)
        print(f"轮次 {i+1}: 选择臂 {arm}, 成功 {success}")
    
    stats = bandit.get_detailed_stats()
    print(f"后验均值: {stats['posterior_means']}")
    print(f"选择概率: {stats['arm_selection_probabilities']}")
    print(f"最佳臂: {bandit.get_best_arm()}")
    print()


def test_epsilon_greedy():
    """测试Epsilon-Greedy算法"""
    print("=== 测试Epsilon-Greedy算法 ===")
    bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.2, random_state=42)
    print(f"初始化: {bandit}")
    
    # 模拟一些交互
    for i in range(10):
        arm = bandit.select_arm()
        reward = np.random.normal(0.8, 0.1) if arm == 2 else np.random.normal(0.4, 0.1)  # 臂2有更高奖励
        reward = max(0, min(1, reward))  # 限制在[0,1]范围内
        bandit.update(arm, reward)
        print(f"轮次 {i+1}: 选择臂 {arm}, 奖励 {reward:.3f}, epsilon {bandit.epsilon:.3f}")
    
    stats = bandit.get_detailed_stats()
    print(f"平均奖励: {stats['estimated_rewards']}")
    print(f"探索率: {stats['estimated_exploration_rate']}")
    print(f"最佳臂: {bandit.get_best_arm()}")
    print()


def test_linear_contextual():
    """测试Linear Contextual Bandit算法"""
    print("=== 测试Linear Contextual Bandit算法 ===")
    bandit = LinearContextualBandit(n_arms=2, n_features=3, alpha=1.0, random_state=42)
    print(f"初始化: {bandit}")
    
    # 模拟一些带上下文的交互
    contexts = [
        {"features": [1.0, 0.0, 0.5]},
        {"features": [0.0, 1.0, -0.3]},
        {"features": [0.5, 0.5, 0.0]},
        {"features": [1.0, 0.2, 0.8]},
        {"features": [-0.3, 1.0, 0.1]}
    ]
    
    for i, context in enumerate(contexts):
        arm = bandit.select_arm(context)
        # 模拟线性奖励：reward = features * [0.5, -0.2, 0.3] + noise
        features = np.array(context["features"])
        true_reward = np.dot(features, [0.5, -0.2, 0.3]) + 0.5
        true_reward = max(0, min(1, true_reward + np.random.normal(0, 0.1)))
        
        bandit.update(arm, true_reward, context)
        print(f"轮次 {i+1}: 特征 {features}, 选择臂 {arm}, 奖励 {true_reward:.3f}")
    
    # 测试预测
    test_context = {"features": [0.8, 0.1, 0.2]}
    test_features = np.array(test_context["features"])
    for arm in range(2):
        predicted = bandit.predict_reward(arm, test_features)
        print(f"臂 {arm} 预测奖励: {predicted:.3f}")
    
    print(f"最佳臂: {bandit.get_best_arm()}")
    print()


def test_algorithm_comparison():
    """比较不同算法的性能"""
    print("=== 算法性能比较 ===")
    n_arms = 4
    n_rounds = 50
    true_rewards = [0.2, 0.5, 0.8, 0.3]  # 真实的臂奖励
    
    bandits = {
        "UCB": UCBBandit(n_arms=n_arms, c=2.0, random_state=42),
        "Thompson": ThompsonSamplingBandit(n_arms=n_arms, random_state=42),
        "Epsilon-Greedy": EpsilonGreedyBandit(n_arms=n_arms, epsilon=0.1, random_state=42)
    }
    
    results = {}
    
    for name, bandit in bandits.items():
        print(f"测试 {name}...")
        total_reward = 0
        optimal_selections = 0
        optimal_arm = np.argmax(true_rewards)
        
        for round_num in range(n_rounds):
            if name == "Thompson":
                arm = bandit.select_arm()
                success = np.random.random() < true_rewards[arm]
                bandit.update_with_binary_reward(arm, success)
                reward = 1.0 if success else 0.0
            else:
                arm = bandit.select_arm()
                reward = np.random.normal(true_rewards[arm], 0.1)
                reward = max(0, min(1, reward))
                bandit.update(arm, reward)
            
            total_reward += reward
            if arm == optimal_arm:
                optimal_selections += 1
        
        avg_reward = total_reward / n_rounds
        optimal_rate = optimal_selections / n_rounds
        regret = bandit.calculate_regret(max(true_rewards))
        
        results[name] = {
            "平均奖励": avg_reward,
            "最优选择率": optimal_rate, 
            "累积遗憾": regret,
            "最佳臂": bandit.get_best_arm()
        }
        
        print(f"  平均奖励: {avg_reward:.3f}")
        print(f"  最优选择率: {optimal_rate:.3f}")
        print(f"  累积遗憾: {regret:.3f}")
        print(f"  识别的最佳臂: {bandit.get_best_arm()} (真实最佳臂: {optimal_arm})")
        print()


if __name__ == "__main__":
    print("多臂老虎机算法测试")
    print("=" * 50)
    
    try:
        test_ucb_bandit()
        test_thompson_sampling()
        test_epsilon_greedy() 
        test_linear_contextual()
        test_algorithm_comparison()
        
        print("✅ 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()