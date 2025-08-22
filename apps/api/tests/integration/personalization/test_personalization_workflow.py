"""
集成测试：个性化引擎完整工作流程
验证端到端个性化推荐流程，包括特征计算、模型推理、缓存机制和反馈处理
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
import numpy as np

from ai.personalization.engine import PersonalizationEngine
from ai.personalization.features.realtime import RealTimeFeatureEngine
from ai.personalization.models.service import DistributedModelService
from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    RealTimeFeatures
)


class TestPersonalizationWorkflow:
    """个性化引擎集成测试套件"""
    
    @pytest.fixture
    async def engine(self):
        """创建测试用的个性化引擎实例"""
        with patch('ai.personalization.engine.redis') as mock_redis, \
             patch('ai.personalization.engine.BanditManager') as mock_bandit, \
             patch('ai.personalization.engine.QLearningAgent') as mock_qlearning:
            
            # 配置模拟对象
            mock_redis.get.return_value = None
            mock_redis.setex = AsyncMock()
            mock_bandit.return_value.select_actions = AsyncMock(return_value=[1, 2, 3])
            mock_qlearning.return_value.get_action = AsyncMock(return_value=1)
            
            engine = PersonalizationEngine()
            await engine.initialize()
            return engine
    
    @pytest.fixture
    def sample_request(self) -> RecommendationRequest:
        """创建样本推荐请求"""
        return RecommendationRequest(
            user_id="test_user_123",
            context={
                "page_type": "homepage",
                "device": "mobile",
                "time_of_day": "evening"
            },
            n_recommendations=10,
            scenario="content_discovery"
        )
    
    @pytest.fixture
    def sample_user_profile(self) -> UserProfile:
        """创建样本用户画像"""
        return UserProfile(
            user_id="test_user_123",
            features={
                "age_group": 25.0,
                "activity_level": 0.8,
                "engagement_score": 0.75
            },
            preferences={
                "technology": 0.9,
                "sports": 0.3,
                "entertainment": 0.7
            },
            behavior_history=[
                {"action": "click", "item_id": "item_1", "timestamp": "2024-01-15T10:00:00"},
                {"action": "view", "item_id": "item_2", "timestamp": "2024-01-15T10:05:00"}
            ],
            last_updated="2024-01-15T10:30:00",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )

    async def test_end_to_end_recommendation_flow(self, engine, sample_request, sample_user_profile):
        """测试端到端推荐流程"""
        
        # 模拟用户画像数据
        with patch.object(engine, '_get_user_profile', return_value=sample_user_profile):
            
            # 执行推荐请求
            start_time = time.time()
            response = await engine.get_recommendations(sample_request)
            end_time = time.time()
            
            # 验证响应结构
            assert isinstance(response, RecommendationResponse)
            assert response.user_id == sample_request.user_id
            assert len(response.recommendations) <= sample_request.n_recommendations
            assert response.latency_ms > 0
            assert response.model_version is not None
            
            # 验证响应时间要求 (P99 < 100ms)
            response_time_ms = (end_time - start_time) * 1000
            assert response_time_ms < 100, f"响应时间 {response_time_ms}ms 超过100ms要求"

    async def test_realtime_feature_computation_performance(self, engine, sample_request):
        """测试实时特征计算性能 (<10ms)"""
        
        # 执行多次特征计算测试
        computation_times = []
        
        for _ in range(10):
            start_time = time.time()
            features = await engine.feature_engine.compute_features(
                sample_request.user_id,
                sample_request.context
            )
            end_time = time.time()
            
            computation_time_ms = (end_time - start_time) * 1000
            computation_times.append(computation_time_ms)
            
            # 验证特征结构
            assert isinstance(features, dict)
            assert "temporal" in features
            assert "behavioral" in features
            assert "contextual" in features
        
        # 验证特征计算延迟要求
        avg_time = sum(computation_times) / len(computation_times)
        max_time = max(computation_times)
        
        assert avg_time < 10, f"平均特征计算时间 {avg_time}ms 超过10ms要求"
        assert max_time < 15, f"最大特征计算时间 {max_time}ms 超过15ms阈值"

    async def test_caching_effectiveness(self, engine, sample_request):
        """测试缓存机制有效性"""
        
        # 第一次请求 - 缓存未命中
        start_time = time.time()
        response1 = await engine.get_recommendations(sample_request)
        first_request_time = time.time() - start_time
        
        # 第二次相同请求 - 应该命中缓存
        start_time = time.time()
        response2 = await engine.get_recommendations(sample_request)
        cached_request_time = time.time() - start_time
        
        # 验证缓存命中效果
        assert response1.request_id != response2.request_id  # 不同的请求ID
        assert len(response1.recommendations) == len(response2.recommendations)
        
        # 缓存请求应该显著更快
        speed_improvement = first_request_time / cached_request_time
        assert speed_improvement > 2, f"缓存加速比 {speed_improvement} 不足2倍"

    async def test_concurrent_recommendation_requests(self, engine):
        """测试并发推荐请求处理"""
        
        # 创建多个并发请求
        requests = [
            RecommendationRequest(
                user_id=f"user_{i}",
                context={"session": f"session_{i}"},
                n_recommendations=5,
                scenario="test"
            )
            for i in range(20)
        ]
        
        # 并发执行推荐请求
        start_time = time.time()
        tasks = [engine.get_recommendations(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 验证所有请求成功处理
        successful_responses = [r for r in responses if isinstance(r, RecommendationResponse)]
        assert len(successful_responses) == len(requests), "部分并发请求失败"
        
        # 验证并发处理性能
        avg_response_time = total_time / len(requests)
        assert avg_response_time < 0.1, f"平均并发响应时间 {avg_response_time}s 过长"

    async def test_feedback_processing_and_learning(self, engine, sample_user_profile):
        """测试反馈处理和在线学习"""
        
        # 提交用户反馈
        feedback_data = {
            "user_id": sample_user_profile["user_id"],
            "item_id": "recommended_item_1",
            "feedback_type": "click",
            "rating": 4.5,
            "context": {"source": "homepage"},
            "timestamp": "2024-01-15T11:00:00"
        }
        
        # 处理反馈
        await engine.process_feedback(
            sample_user_profile["user_id"],
            feedback_data
        )
        
        # 验证反馈是否影响后续推荐
        request = RecommendationRequest(
            user_id=sample_user_profile["user_id"],
            context={"session": "post_feedback"},
            n_recommendations=5,
            scenario="personalized"
        )
        
        response = await engine.get_recommendations(request)
        
        # 验证推荐系统已学习用户反馈
        assert len(response.recommendations) > 0
        assert response.explanation is not None  # 应该有推荐解释

    async def test_model_ensemble_prediction(self, engine, sample_request):
        """测试集成模型预测"""
        
        with patch.object(engine.model_service, 'ensemble_predict') as mock_ensemble:
            mock_ensemble.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
            
            response = await engine.get_recommendations(sample_request)
            
            # 验证集成模型被调用
            assert mock_ensemble.called
            assert len(response.recommendations) > 0

    async def test_fallback_and_error_handling(self, engine, sample_request):
        """测试降级策略和错误处理"""
        
        # 模拟模型服务故障
        with patch.object(engine.model_service, 'predict', side_effect=Exception("模型服务异常")):
            
            response = await engine.get_recommendations(sample_request)
            
            # 验证降级策略生效
            assert isinstance(response, RecommendationResponse)
            assert len(response.recommendations) > 0  # 应该返回默认推荐
            assert "fallback" in response.explanation.lower()

    async def test_recommendation_diversity_and_quality(self, engine, sample_request):
        """测试推荐多样性和质量"""
        
        response = await engine.get_recommendations(sample_request)
        
        # 验证推荐多样性
        recommended_items = [rec.item_id for rec in response.recommendations]
        unique_items = set(recommended_items)
        
        # 推荐应该有足够的多样性
        diversity_ratio = len(unique_items) / len(recommended_items)
        assert diversity_ratio > 0.8, f"推荐多样性 {diversity_ratio} 不足"
        
        # 验证推荐质量
        for rec in response.recommendations:
            assert rec.score >= 0 and rec.score <= 1, "推荐分数超出范围"
            assert rec.confidence >= 0 and rec.confidence <= 1, "置信度超出范围"

    async def test_user_profile_realtime_update(self, engine, sample_user_profile):
        """测试用户画像实时更新"""
        
        user_id = sample_user_profile["user_id"]
        
        # 获取初始用户画像
        initial_profile = await engine._get_user_profile(user_id)
        
        # 模拟用户行为更新
        behavior_update = {
            "action": "purchase",
            "item_id": "premium_item",
            "value": 99.99,
            "timestamp": "2024-01-15T12:00:00"
        }
        
        await engine._update_user_profile(user_id, behavior_update)
        
        # 获取更新后的用户画像
        updated_profile = await engine._get_user_profile(user_id)
        
        # 验证画像已更新
        assert updated_profile["last_updated"] != initial_profile["last_updated"]
        assert len(updated_profile["behavior_history"]) > len(initial_profile["behavior_history"])

    @pytest.mark.performance
    async def test_high_throughput_load(self, engine):
        """测试高吞吐量负载 (目标: 10,000+ QPS)"""
        
        # 创建大量并发请求模拟高负载
        num_requests = 1000  # 简化测试，实际应该更大
        requests = [
            RecommendationRequest(
                user_id=f"load_test_user_{i % 100}",  # 100个不同用户
                context={"load_test": True, "batch": i // 100},
                n_recommendations=3,  # 减少推荐数量以提高吞吐量
                scenario="load_test"
            )
            for i in range(num_requests)
        ]
        
        # 分批执行以模拟真实负载
        batch_size = 100
        total_start_time = time.time()
        
        for i in range(0, num_requests, batch_size):
            batch = requests[i:i + batch_size]
            batch_start = time.time()
            
            tasks = [engine.get_recommendations(req) for req in batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_time = time.time() - batch_start
            batch_qps = len(batch) / batch_time
            
            # 验证批次QPS
            assert batch_qps > 100, f"批次QPS {batch_qps} 低于最低要求"
            
            # 验证成功率
            successful = [r for r in responses if isinstance(r, RecommendationResponse)]
            success_rate = len(successful) / len(batch)
            assert success_rate > 0.95, f"成功率 {success_rate} 低于95%"
        
        total_time = time.time() - total_start_time
        overall_qps = num_requests / total_time
        
        # 验证整体性能
        print(f"整体QPS: {overall_qps}")
        assert overall_qps > 500, f"整体QPS {overall_qps} 低于预期"  # 在测试环境中的保守目标

    async def test_memory_usage_stability(self, engine, sample_request):
        """测试内存使用稳定性"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量请求测试内存泄漏
        for i in range(500):
            request = RecommendationRequest(
                user_id=f"memory_test_user_{i % 50}",
                context={"iteration": i},
                n_recommendations=5,
                scenario="memory_test"
            )
            await engine.get_recommendations(request)
            
            # 每100次检查一次内存
            if i % 100 == 99:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # 内存增长不应超过200MB
                assert memory_growth < 200, f"内存增长 {memory_growth}MB 过多，可能存在内存泄漏"

    async def test_recommendation_explanation_quality(self, engine, sample_request):
        """测试推荐解释质量"""
        
        response = await engine.get_recommendations(sample_request)
        
        if response.explanation:
            explanation = response.explanation.lower()
            
            # 验证解释包含关键信息
            expected_terms = ["user", "preference", "similarity", "behavior", "score"]
            found_terms = sum(1 for term in expected_terms if term in explanation)
            
            assert found_terms >= 2, "推荐解释缺少关键信息"
            assert len(explanation) > 50, "推荐解释过于简短"
            assert len(explanation) < 500, "推荐解释过于冗长"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])