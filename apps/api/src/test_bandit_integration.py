#!/usr/bin/env python3
"""
多臂老虎机推荐系统集成测试

测试推荐引擎服务和API路由的完整工作流程。
"""

import asyncio
import sys
import os
from datetime import datetime
import json

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from src.services.bandit_recommendation_service import BanditRecommendationService
from src.ai.reinforcement_learning.recommendation_engine import AlgorithmType


async def test_recommendation_service():
    """测试推荐服务基本功能"""
    print("=== 测试推荐服务基本功能 ===")
    
    # 创建服务实例
    service = BanditRecommendationService()
    
    # 初始化服务
    print("1. 初始化推荐服务...")
    success = await service.initialize(
        n_items=100,
        enable_cold_start=True,
        enable_evaluation=True
    )
    
    if not success:
        print("❌ 服务初始化失败")
        return False
    
    print("✅ 服务初始化成功")
    
    # 获取推荐
    print("\n2. 测试推荐生成...")
    try:
        recommendations = await service.get_recommendations(
            user_id="test_user_1",
            num_recommendations=5,
            context={"age": 25, "location": "Beijing"},
            include_explanations=True
        )
        
        print(f"推荐结果: {len(recommendations['recommendations'])} 个物品")
        print(f"使用算法: {recommendations['algorithm_used']}")
        print(f"置信度: {recommendations['confidence_score']:.3f}")
        print(f"处理时间: {recommendations['processing_time_ms']:.2f}ms")
        
        if recommendations['explanations']:
            print("推荐解释:")
            for i, explanation in enumerate(recommendations['explanations'][:3]):
                print(f"  {i+1}. {explanation}")
        
    except Exception as e:
        print(f"❌ 推荐生成失败: {e}")
        return False
    
    print("✅ 推荐生成成功")
    
    # 测试反馈处理
    print("\n3. 测试反馈处理...")
    try:
        success = await service.process_feedback(
            user_id="test_user_1",
            item_id=recommendations['recommendations'][0]['item_id'],
            feedback_type="click",
            feedback_value=1.0,
            context={"page": "home"}
        )
        
        if success:
            print("✅ 反馈处理成功")
        else:
            print("❌ 反馈处理失败")
            return False
    except Exception as e:
        print(f"❌ 反馈处理异常: {e}")
        return False
    
    # 测试统计信息
    print("\n4. 测试统计信息...")
    try:
        stats = service.get_statistics()
        print(f"总请求数: {stats['engine_stats']['total_requests']}")
        print(f"活跃用户数: {stats['active_users']}")
        print(f"算法统计: {list(stats['algorithm_stats'].keys())}")
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        return False
    
    print("✅ 统计信息获取成功")
    
    # 测试用户上下文更新
    print("\n5. 测试用户上下文更新...")
    try:
        success = await service.update_user_context(
            "test_user_1",
            {"age": 26, "location": "Shanghai", "interests": ["tech", "sports"]}
        )
        
        if success:
            print("✅ 用户上下文更新成功")
        else:
            print("❌ 用户上下文更新失败")
            return False
    except Exception as e:
        print(f"❌ 用户上下文更新异常: {e}")
        return False
    
    # 测试物品特征更新
    print("\n6. 测试物品特征更新...")
    try:
        success = await service.update_item_features(
            "item_1",
            {"category": "electronics", "price": 299.99, "rating": 4.5}
        )
        
        if success:
            print("✅ 物品特征更新成功")
        else:
            print("❌ 物品特征更新失败")
            return False
    except Exception as e:
        print(f"❌ 物品特征更新异常: {e}")
        return False
    
    # 测试健康检查
    print("\n7. 测试健康检查...")
    try:
        health = service.get_health_status()
        print(f"服务状态: {health['status']}")
        print(f"是否初始化: {health['is_initialized']}")
        
        if health['status'] == 'healthy':
            print("✅ 健康检查通过")
        else:
            print(f"⚠️  服务状态异常: {health['status']}")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False
    
    return True


async def test_cold_start_scenario():
    """测试冷启动场景"""
    print("\n=== 测试冷启动场景 ===")
    
    service = BanditRecommendationService()
    await service.initialize(n_items=50, enable_cold_start=True)
    
    # 测试新用户冷启动
    print("1. 测试新用户冷启动...")
    try:
        recommendations = await service.get_recommendations(
            user_id="new_user_cold",
            num_recommendations=5,
            context=None  # 没有上下文信息
        )
        
        print(f"冷启动推荐: {len(recommendations['recommendations'])} 个物品")
        if recommendations.get('cold_start_strategy'):
            print(f"冷启动策略: {recommendations['cold_start_strategy']}")
        
        print("✅ 新用户冷启动测试成功")
        
    except Exception as e:
        print(f"❌ 新用户冷启动测试失败: {e}")
        return False
    
    # 测试带特征的新用户
    print("\n2. 测试带特征的新用户...")
    try:
        recommendations = await service.get_recommendations(
            user_id="new_user_with_features",
            num_recommendations=5,
            context={"age": 30, "gender": "male", "interests": ["sports", "tech"]}
        )
        
        print(f"带特征新用户推荐: {len(recommendations['recommendations'])} 个物品")
        print(f"使用算法: {recommendations['algorithm_used']}")
        
        print("✅ 带特征新用户测试成功")
        
    except Exception as e:
        print(f"❌ 带特征新用户测试失败: {e}")
        return False
    
    return True


async def test_algorithm_comparison():
    """测试不同算法的性能对比"""
    print("\n=== 测试算法性能对比 ===")
    
    # 测试不同算法配置
    algorithm_configs = {
        "ucb": {"c": 1.5, "random_state": 42},
        "thompson_sampling": {"alpha_init": 2.0, "beta_init": 2.0, "random_state": 42},
        "epsilon_greedy": {"epsilon": 0.15, "decay_rate": 0.99, "random_state": 42}
    }
    
    service = BanditRecommendationService()
    await service.initialize(
        n_items=20,
        algorithm_configs=algorithm_configs,
        enable_evaluation=True
    )
    
    # 模拟多轮推荐和反馈
    num_rounds = 20
    users = [f"user_{i}" for i in range(5)]
    
    print(f"模拟{num_rounds}轮推荐和反馈...")
    
    for round_num in range(num_rounds):
        for user_id in users:
            try:
                # 获取推荐
                recs = await service.get_recommendations(
                    user_id=user_id,
                    num_recommendations=3,
                    context={"round": round_num}
                )
                
                # 模拟用户反馈
                for rec in recs['recommendations'][:2]:  # 只对前两个推荐提供反馈
                    feedback_value = 1.0 if int(rec['item_id']) < 5 else 0.3  # 前5个物品更受欢迎
                    await service.process_feedback(
                        user_id=user_id,
                        item_id=rec['item_id'],
                        feedback_type="click",
                        feedback_value=feedback_value
                    )
                
            except Exception as e:
                print(f"轮次{round_num}，用户{user_id}测试失败: {e}")
                return False
    
    # 获取最终统计
    try:
        stats = service.get_statistics()
        print(f"\n最终统计:")
        print(f"总请求数: {stats['engine_stats']['total_requests']}")
        print(f"缓存命中数: {stats['engine_stats']['cache_hits']}")
        print(f"平均响应时间: {stats['engine_stats']['average_response_time_ms']:.2f}ms")
        
        if stats.get('evaluation_metrics'):
            eval_metrics = stats['evaluation_metrics']
            print(f"平均奖励: {eval_metrics.get('average_reward', 0):.3f}")
            print(f"点击率: {eval_metrics.get('click_through_rate', 0):.3f}")
        
        print("✅ 算法性能对比测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 获取最终统计失败: {e}")
        return False


async def test_contextual_bandit():
    """测试上下文多臂老虎机"""
    print("\n=== 测试上下文多臂老虎机 ===")
    
    # 配置上下文算法
    algorithm_configs = {
        "linear_contextual": {
            "n_features": 10,
            "alpha": 0.5,
            "lambda_reg": 0.1,
            "random_state": 42
        }
    }
    
    service = BanditRecommendationService()
    service.default_config["default_algorithm"] = AlgorithmType.LINEAR_CONTEXTUAL
    
    await service.initialize(
        n_items=15,
        algorithm_configs=algorithm_configs
    )
    
    # 测试不同上下文的推荐
    contexts = [
        {"age": 25, "income": 50000, "city": "Beijing", "interests": "tech"},
        {"age": 35, "income": 80000, "city": "Shanghai", "interests": "finance"},
        {"age": 28, "income": 60000, "city": "Guangzhou", "interests": "travel"}
    ]
    
    for i, context in enumerate(contexts):
        try:
            print(f"\n上下文{i+1}: {context}")
            
            recs = await service.get_recommendations(
                user_id=f"contextual_user_{i}",
                num_recommendations=3,
                context=context,
                include_explanations=True
            )
            
            print(f"推荐结果: {[rec['item_id'] for rec in recs['recommendations']]}")
            print(f"平均置信度: {recs['confidence_score']:.3f}")
            
            if recs.get('explanations'):
                print(f"解释: {recs['explanations'][0]}")
            
            # 提供反馈
            for rec in recs['recommendations'][:2]:
                feedback_value = 0.8 if "tech" in str(context.get('interests', '')) else 0.4
                await service.process_feedback(
                    user_id=f"contextual_user_{i}",
                    item_id=rec['item_id'],
                    feedback_type="rating",
                    feedback_value=feedback_value,
                    context=context
                )
            
        except Exception as e:
            print(f"❌ 上下文测试{i+1}失败: {e}")
            return False
    
    print("✅ 上下文多臂老虎机测试成功")
    return True


async def run_all_tests():
    """运行所有测试"""
    print("多臂老虎机推荐系统集成测试")
    print("=" * 50)
    
    tests = [
        ("推荐服务基本功能", test_recommendation_service),
        ("冷启动场景", test_cold_start_scenario),
        ("算法性能对比", test_algorithm_comparison),
        ("上下文多臂老虎机", test_contextual_bandit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*20} 测试汇总 {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"测试通过: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 所有测试通过！多臂老虎机推荐系统集成成功！")
    else:
        print(f"\n⚠️  有{total - passed}个测试失败，需要检查问题")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n测试被中断")
        sys.exit(1)
        
    except Exception as e:
        print(f"测试运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)