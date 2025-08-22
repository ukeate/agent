"""
强化学习系统端到端集成测试

测试所有强化学习组件的协同工作，包括：
- 多臂老虎机推荐引擎
- Q-Learning智能体
- 用户反馈学习系统  
- A/B测试平台
- 行为分析系统
- 实时个性化引擎
"""

import pytest
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from ai.reinforcement_learning.recommendation_engine import (
    BanditRecommendationEngine,
    AlgorithmType,
    RecommendationRequest,
    FeedbackData
)
from ai.reinforcement_learning.bandits.ucb import UCBBandit
from ai.reinforcement_learning.bandits.thompson_sampling import ThompsonSamplingBandit
from ai.reinforcement_learning.bandits.epsilon_greedy import EpsilonGreedyBandit


class TestRLSystemEndToEnd:
    """强化学习系统端到端集成测试"""
    
    @pytest.fixture
    async def recommendation_engine(self):
        """创建推荐引擎实例"""
        engine = BanditRecommendationEngine(
            default_algorithm=AlgorithmType.UCB,
            enable_cold_start=True,
            enable_evaluation=True,
            cache_ttl_seconds=300,
            max_cache_size=1000
        )
        
        # 初始化算法
        await engine.initialize_algorithms(n_items=100)
        return engine
    
    @pytest.fixture
    def sample_users(self):
        """生成测试用户"""
        return [f"user_{i}" for i in range(10)]
    
    @pytest.fixture
    def sample_items(self):
        """生成测试物品"""
        return [f"item_{i}" for i in range(100)]
    
    async def test_full_recommendation_workflow(self, recommendation_engine, sample_users):
        """测试完整推荐工作流程"""
        
        # 1. 初始冷启动推荐
        user_id = sample_users[0]
        request = RecommendationRequest(
            user_id=user_id,
            context={"age": 25, "location": "beijing", "device": "mobile"},
            num_recommendations=5,
            include_explanations=True
        )
        
        response = await recommendation_engine.get_recommendations(request)
        
        # 验证推荐响应
        assert response.user_id == user_id
        assert len(response.recommendations) == 5
        assert response.algorithm_used in ["cold_start", "ucb"]
        assert response.confidence_score >= 0.0
        assert response.explanations is not None
        assert response.processing_time_ms > 0
        
        # 2. 模拟用户反馈
        feedback_data = []
        for i, rec in enumerate(response.recommendations[:3]):
            feedback = FeedbackData(
                user_id=user_id,
                item_id=rec["item_id"],
                feedback_type="click" if i == 0 else "view",
                feedback_value=0.8 if i == 0 else 0.3,
                context=request.context,
                timestamp=datetime.now()
            )
            await recommendation_engine.process_feedback(feedback)
            feedback_data.append(feedback)
        
        # 3. 再次获取推荐（应该有学习效果）
        request2 = RecommendationRequest(
            user_id=user_id,
            context={"age": 25, "location": "beijing", "device": "mobile"},
            num_recommendations=5,
            include_explanations=True
        )
        
        response2 = await recommendation_engine.get_recommendations(request2)
        
        # 验证学习效果
        assert response2.user_id == user_id
        assert response2.algorithm_used == "ucb"  # 不再是冷启动
        assert response2.confidence_score > 0
        
        # 4. 验证统计信息
        stats = recommendation_engine.get_engine_statistics()
        assert stats["engine_stats"]["total_requests"] >= 2
        assert stats["engine_stats"]["cold_start_requests"] >= 1
        assert "ucb" in stats["algorithm_stats"]
        assert stats["active_users"] >= 1
    
    async def test_multi_user_concurrent_recommendations(self, recommendation_engine, sample_users):
        """测试多用户并发推荐"""
        
        async def get_user_recommendations(user_id: str):
            """单个用户推荐流程"""
            request = RecommendationRequest(
                user_id=user_id,
                context={"preference": f"category_{hash(user_id) % 5}"},
                num_recommendations=3
            )
            return await recommendation_engine.get_recommendations(request)
        
        # 并发执行多个用户的推荐请求
        tasks = [get_user_recommendations(user) for user in sample_users[:5]]
        responses = await asyncio.gather(*tasks)
        
        # 验证所有响应
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.user_id == sample_users[i]
            assert len(response.recommendations) == 3
            assert response.processing_time_ms > 0
        
        # 验证用户算法分配
        stats = recommendation_engine.get_engine_statistics()
        assert stats["active_users"] >= 5
    
    async def test_algorithm_comparison_workflow(self, recommendation_engine, sample_users):
        """测试算法比较工作流"""
        
        # 为不同用户分配不同算法
        algorithms = ["ucb", "thompson_sampling", "epsilon_greedy"]
        algorithm_performances = {}
        
        for i, algorithm in enumerate(algorithms):
            user_id = sample_users[i]
            
            # 手动设置用户算法
            recommendation_engine.user_algorithms[user_id] = algorithm
            
            # 执行推荐和反馈循环
            total_reward = 0
            for round_num in range(10):
                # 获取推荐
                request = RecommendationRequest(
                    user_id=user_id,
                    context={"round": round_num},
                    num_recommendations=1
                )
                response = await recommendation_engine.get_recommendations(request)
                
                # 模拟反馈（随机奖励）
                import random
                reward = random.uniform(0.0, 1.0)
                total_reward += reward
                
                feedback = FeedbackData(
                    user_id=user_id,
                    item_id=response.recommendations[0]["item_id"],
                    feedback_type="rating",
                    feedback_value=reward,
                    timestamp=datetime.now()
                )
                await recommendation_engine.process_feedback(feedback)
            
            algorithm_performances[algorithm] = total_reward / 10
        
        # 验证算法性能统计
        stats = recommendation_engine.get_engine_statistics()
        for algorithm in algorithms:
            assert algorithm in stats["algorithm_stats"]
            assert stats["engine_stats"]["algorithm_usage"][algorithm] >= 10
    
    async def test_caching_mechanism(self, recommendation_engine, sample_users):
        """测试缓存机制"""
        
        user_id = sample_users[0]
        request = RecommendationRequest(
            user_id=user_id,
            context={"test": "caching"},
            num_recommendations=5
        )
        
        # 第一次请求
        start_time = time.time()
        response1 = await recommendation_engine.get_recommendations(request)
        first_request_time = time.time() - start_time
        
        # 第二次相同请求（应该使用缓存）
        start_time = time.time()
        response2 = await recommendation_engine.get_recommendations(request)
        second_request_time = time.time() - start_time
        
        # 验证缓存效果
        assert response1.user_id == response2.user_id
        assert len(response1.recommendations) == len(response2.recommendations)
        # 第二次请求应该更快（使用了缓存）
        # assert second_request_time < first_request_time  # 缓存命中
        
        # 验证缓存统计
        stats = recommendation_engine.get_engine_statistics()
        assert stats["engine_stats"]["cache_hits"] >= 1
        assert stats["cache_size"] > 0
    
    async def test_cold_start_to_personalization_transition(self, recommendation_engine):
        """测试从冷启动到个性化的转换"""
        
        new_user = "new_user_" + str(uuid.uuid4())
        
        # 1. 冷启动阶段
        request = RecommendationRequest(
            user_id=new_user,
            context={"interests": ["tech", "sports"]},
            num_recommendations=5
        )
        
        cold_start_response = await recommendation_engine.get_recommendations(request)
        assert cold_start_response.algorithm_used == "cold_start"
        assert cold_start_response.cold_start_strategy is not None
        
        # 2. 提供足够的反馈数据
        feedback_sessions = [
            ("click", 0.7),
            ("like", 0.9),
            ("purchase", 1.0),
            ("view", 0.2),
            ("click", 0.6)
        ]
        
        for feedback_type, feedback_value in feedback_sessions:
            # 随机选择推荐物品进行反馈
            item_id = cold_start_response.recommendations[0]["item_id"]
            feedback = FeedbackData(
                user_id=new_user,
                item_id=item_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                context=request.context,
                timestamp=datetime.now()
            )
            await recommendation_engine.process_feedback(feedback)
        
        # 3. 再次请求推荐（应该切换到个性化算法）
        personalized_request = RecommendationRequest(
            user_id=new_user,
            context={"interests": ["tech", "sports"]},
            num_recommendations=5
        )
        
        personalized_response = await recommendation_engine.get_recommendations(personalized_request)
        
        # 验证切换到个性化算法
        assert personalized_response.algorithm_used != "cold_start"
        assert personalized_response.algorithm_used == "ucb"  # 默认算法
        assert personalized_response.confidence_score > cold_start_response.confidence_score
    
    async def test_error_handling_and_fallback(self, recommendation_engine, sample_users):
        """测试错误处理和降级机制"""
        
        user_id = sample_users[0]
        
        # 1. 测试异常情况下的降级推荐
        with patch.object(recommendation_engine, '_generate_recommendations', side_effect=Exception("Test error")):
            request = RecommendationRequest(
                user_id=user_id,
                num_recommendations=5
            )
            
            response = await recommendation_engine.get_recommendations(request)
            
            # 验证降级推荐
            assert response.algorithm_used == "fallback"
            assert len(response.recommendations) == 5
            assert response.confidence_score == 0.1
        
        # 2. 测试无效反馈的处理
        invalid_feedback = FeedbackData(
            user_id=user_id,
            item_id="invalid_item",
            feedback_type="invalid_type",
            feedback_value=-999.0,
            timestamp=datetime.now()
        )
        
        # 应该不会崩溃
        await recommendation_engine.process_feedback(invalid_feedback)
    
    async def test_context_feature_processing(self, recommendation_engine, sample_users):
        """测试上下文特征处理"""
        
        user_id = sample_users[0]
        
        # 测试不同类型的上下文数据
        contexts = [
            {"age": 25, "income": 50000, "location": "beijing"},
            {"preferences": ["tech", "sports"], "time_of_day": "evening"},
            {"device": "mobile", "browser": "chrome", "page_views": 15},
            {}  # 空上下文
        ]
        
        for i, context in enumerate(contexts):
            request = RecommendationRequest(
                user_id=user_id,
                context=context,
                num_recommendations=3
            )
            
            response = await recommendation_engine.get_recommendations(request)
            
            # 验证响应
            assert response.user_id == user_id
            assert len(response.recommendations) == 3
            assert response.processing_time_ms > 0
    
    async def test_system_performance_under_load(self, recommendation_engine, sample_users, sample_items):
        """测试系统负载下的性能"""
        
        # 高并发推荐请求
        async def load_test_requests():
            tasks = []
            for i in range(50):  # 50个并发请求
                user_id = sample_users[i % len(sample_users)]
                request = RecommendationRequest(
                    user_id=user_id,
                    context={"load_test": True, "batch": i},
                    num_recommendations=3
                )
                tasks.append(recommendation_engine.get_recommendations(request))
            
            return await asyncio.gather(*tasks)
        
        # 执行负载测试
        start_time = time.time()
        responses = await load_test_requests()
        total_time = time.time() - start_time
        
        # 验证性能指标
        assert len(responses) == 50
        assert total_time < 10.0  # 总时间应小于10秒
        
        # 验证所有响应
        for response in responses:
            assert len(response.recommendations) == 3
            assert response.processing_time_ms < 1000  # 单个请求小于1秒
        
        # 验证系统统计
        stats = recommendation_engine.get_engine_statistics()
        assert stats["engine_stats"]["total_requests"] >= 50
        assert stats["engine_stats"]["average_response_time_ms"] < 500  # 平均响应时间
    
    async def test_real_time_feedback_learning(self, recommendation_engine, sample_users):
        """测试实时反馈学习"""
        
        user_id = sample_users[0]
        
        # 获取初始推荐
        initial_request = RecommendationRequest(
            user_id=user_id,
            num_recommendations=5
        )
        initial_response = await recommendation_engine.get_recommendations(initial_request)
        
        # 记录初始推荐的物品
        initial_items = [rec["item_id"] for rec in initial_response.recommendations]
        
        # 对第一个物品给予高奖励反馈
        high_reward_item = initial_items[0]
        feedback = FeedbackData(
            user_id=user_id,
            item_id=high_reward_item,
            feedback_type="purchase",
            feedback_value=1.0,
            timestamp=datetime.now()
        )
        await recommendation_engine.process_feedback(feedback)
        
        # 多次获取推荐，验证学习效果
        high_reward_item_appearances = 0
        total_requests = 10
        
        for i in range(total_requests):
            request = RecommendationRequest(
                user_id=user_id,
                num_recommendations=5
            )
            response = await recommendation_engine.get_recommendations(request)
            
            # 检查高奖励物品是否更频繁出现
            recommended_items = [rec["item_id"] for rec in response.recommendations]
            if high_reward_item in recommended_items:
                high_reward_item_appearances += 1
        
        # 验证学习效果：高奖励物品应该更频繁出现
        appearance_rate = high_reward_item_appearances / total_requests
        assert appearance_rate > 0.3  # 至少30%的出现率
    
    @pytest.mark.performance
    async def test_response_time_requirements(self, recommendation_engine, sample_users):
        """测试响应时间要求"""
        
        user_id = sample_users[0]
        response_times = []
        
        # 执行多次推荐请求
        for i in range(20):
            request = RecommendationRequest(
                user_id=user_id,
                context={"test_round": i},
                num_recommendations=10
            )
            
            start_time = time.time()
            response = await recommendation_engine.get_recommendations(request)
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            response_times.append(response_time)
            
            # 验证单次响应时间
            assert response_time < 200  # P99 < 200ms
            assert response.processing_time_ms < 200
        
        # 验证统计指标
        p50 = sorted(response_times)[len(response_times) // 2]
        p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        
        assert p50 < 50   # P50 < 50ms
        assert p95 < 100  # P95 < 100ms
        
        # 验证引擎统计
        stats = recommendation_engine.get_engine_statistics()
        assert stats["engine_stats"]["average_response_time_ms"] < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])