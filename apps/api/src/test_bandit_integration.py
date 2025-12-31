import asyncio
import sys
import os
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import json
from src.services.bandit_recommendation_service import BanditRecommendationService
from src.ai.reinforcement_learning.recommendation_engine import AlgorithmType
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
多臂老虎机推荐系统集成测试

测试推荐引擎服务和API路由的完整工作流程。
"""

sys.path.insert(0, os.path.dirname(__file__))

async def test_recommendation_service():
    """测试推荐服务基本功能"""
    logger.info("=== 测试推荐服务基本功能 ===")
    
    # 创建服务实例
    service = BanditRecommendationService()
    
    # 初始化服务
    logger.info("1. 初始化推荐服务...")
    success = await service.initialize(
        n_items=100,
        enable_cold_start=True,
        enable_evaluation=True
    )
    
    if not success:
        logger.error("❌ 服务初始化失败")
        return False
    
    logger.info("✅ 服务初始化成功")
    
    # 获取推荐
    logger.info("\n2. 测试推荐生成...")
    try:
        recommendations = await service.get_recommendations(
            user_id="test_user_1",
            num_recommendations=5,
            context={"age": 25, "location": "Beijing"},
            include_explanations=True
        )
        
        logger.info(f"推荐结果: {len(recommendations['recommendations'])} 个物品")
        logger.info(f"使用算法: {recommendations['algorithm_used']}")
        logger.info(f"置信度: {recommendations['confidence_score']:.3f}")
        logger.info(f"处理时间: {recommendations['processing_time_ms']:.2f}ms")
        
        if recommendations['explanations']:
            logger.info("推荐解释:")
            for i, explanation in enumerate(recommendations['explanations'][:3]):
                logger.info(f"  {i+1}. {explanation}")
        
    except Exception as e:
        logger.error(f"❌ 推荐生成失败: {e}")
        return False
    
    logger.info("✅ 推荐生成成功")
    
    # 测试反馈处理
    logger.info("\n3. 测试反馈处理...")
    try:
        success = await service.process_feedback(
            user_id="test_user_1",
            item_id=recommendations['recommendations'][0]['item_id'],
            feedback_type="click",
            feedback_value=1.0,
            context={"page": "home"}
        )
        
        if success:
            logger.info("✅ 反馈处理成功")
        else:
            logger.error("❌ 反馈处理失败")
            return False
    except Exception as e:
        logger.error(f"❌ 反馈处理异常: {e}")
        return False
    
    # 测试统计信息
    logger.info("\n4. 测试统计信息...")
    try:
        stats = service.get_statistics()
        logger.info(f"总请求数: {stats['engine_stats']['total_requests']}")
        logger.info(f"活跃用户数: {stats['active_users']}")
        logger.info(f"算法统计: {list(stats['algorithm_stats'].keys())}")
        
    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {e}")
        return False
    
    logger.info("✅ 统计信息获取成功")
    
    # 测试用户上下文更新
    logger.info("\n5. 测试用户上下文更新...")
    try:
        success = await service.update_user_context(
            "test_user_1",
            {"age": 26, "location": "Shanghai", "interests": ["tech", "sports"]}
        )
        
        if success:
            logger.info("✅ 用户上下文更新成功")
        else:
            logger.error("❌ 用户上下文更新失败")
            return False
    except Exception as e:
        logger.error(f"❌ 用户上下文更新异常: {e}")
        return False
    
    # 测试物品特征更新
    logger.info("\n6. 测试物品特征更新...")
    try:
        success = await service.update_item_features(
            "item_1",
            {"category": "electronics", "price": 299.99, "rating": 4.5}
        )
        
        if success:
            logger.info("✅ 物品特征更新成功")
        else:
            logger.error("❌ 物品特征更新失败")
            return False
    except Exception as e:
        logger.error(f"❌ 物品特征更新异常: {e}")
        return False
    
    # 测试健康检查
    logger.info("\n7. 测试健康检查...")
    try:
        health = service.get_health_status()
        logger.info(f"服务状态: {health['status']}")
        logger.info(f"是否初始化: {health['is_initialized']}")
        
        if health['status'] == 'healthy':
            logger.info("✅ 健康检查通过")
        else:
            logger.error(f"⚠️  服务状态异常: {health['status']}")
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return False
    
    return True

async def test_cold_start_scenario():
    """测试冷启动场景"""
    logger.info("\n=== 测试冷启动场景 ===")
    
    service = BanditRecommendationService()
    await service.initialize(n_items=50, enable_cold_start=True)
    
    # 测试新用户冷启动
    logger.info("1. 测试新用户冷启动...")
    try:
        recommendations = await service.get_recommendations(
            user_id="new_user_cold",
            num_recommendations=5,
            context=None  # 没有上下文信息
        )
        
        logger.info(f"冷启动推荐: {len(recommendations['recommendations'])} 个物品")
        if recommendations.get('cold_start_strategy'):
            logger.info(f"冷启动策略: {recommendations['cold_start_strategy']}")
        
        logger.info("✅ 新用户冷启动测试成功")
        
    except Exception as e:
        logger.error(f"❌ 新用户冷启动测试失败: {e}")
        return False
    
    # 测试带特征的新用户
    logger.info("\n2. 测试带特征的新用户...")
    try:
        recommendations = await service.get_recommendations(
            user_id="new_user_with_features",
            num_recommendations=5,
            context={"age": 30, "gender": "male", "interests": ["sports", "tech"]}
        )
        
        logger.info(f"带特征新用户推荐: {len(recommendations['recommendations'])} 个物品")
        logger.info(f"使用算法: {recommendations['algorithm_used']}")
        
        logger.info("✅ 带特征新用户测试成功")
        
    except Exception as e:
        logger.error(f"❌ 带特征新用户测试失败: {e}")
        return False
    
    return True

async def test_algorithm_comparison():
    """测试不同算法的性能对比"""
    logger.info("\n=== 测试算法性能对比 ===")
    
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
    
    logger.info(f"模拟{num_rounds}轮推荐和反馈...")
    
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
                logger.error(f"轮次{round_num}，用户{user_id}测试失败: {e}")
                return False
    
    # 获取最终统计
    try:
        stats = service.get_statistics()
        logger.info(f"\n最终统计:")
        logger.info(f"总请求数: {stats['engine_stats']['total_requests']}")
        logger.info(f"缓存命中数: {stats['engine_stats']['cache_hits']}")
        logger.info(f"平均响应时间: {stats['engine_stats']['average_response_time_ms']:.2f}ms")
        
        if stats.get('evaluation_metrics'):
            eval_metrics = stats['evaluation_metrics']
            logger.info(f"平均奖励: {eval_metrics.get('average_reward', 0):.3f}")
            logger.info(f"点击率: {eval_metrics.get('click_through_rate', 0):.3f}")
        
        logger.info("✅ 算法性能对比测试成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 获取最终统计失败: {e}")
        return False

async def test_contextual_bandit():
    """测试上下文多臂老虎机"""
    logger.info("\n=== 测试上下文多臂老虎机 ===")
    
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
            logger.info(f"\n上下文{i+1}: {context}")
            
            recs = await service.get_recommendations(
                user_id=f"contextual_user_{i}",
                num_recommendations=3,
                context=context,
                include_explanations=True
            )
            
            logger.info(f"推荐结果: {[rec['item_id'] for rec in recs['recommendations']]}")
            logger.info(f"平均置信度: {recs['confidence_score']:.3f}")
            
            if recs.get('explanations'):
                logger.info(f"解释: {recs['explanations'][0]}")
            
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
            logger.error(f"❌ 上下文测试{i+1}失败: {e}")
            return False
    
    logger.info("✅ 上下文多臂老虎机测试成功")
    return True

async def run_all_tests():
    """运行所有测试"""
    logger.info("多臂老虎机推荐系统集成测试")
    logger.info("=" * 50)
    
    tests = [
        ("推荐服务基本功能", test_recommendation_service),
        ("冷启动场景", test_cold_start_scenario),
        ("算法性能对比", test_algorithm_comparison),
        ("上下文多臂老虎机", test_contextual_bandit)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} 测试通过")
            else:
                logger.error(f"❌ {test_name} 测试失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    logger.info(f"\n{'='*20} 测试汇总 {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"测试通过: {passed}/{total}")
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    if passed == total:
        logger.info(f"\n🎉 所有测试通过！多臂老虎机推荐系统集成成功！")
    else:
        logger.error(f"\n⚠️  有{total - passed}个测试失败，需要检查问题")
    
    return passed == total

if __name__ == "__main__":
    setup_logging()
    try:
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\n测试被中断")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"测试运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
