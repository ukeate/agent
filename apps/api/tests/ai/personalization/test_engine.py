import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from redis.asyncio import Redis

from ai.personalization.engine import PersonalizationEngine
from ai.personalization.features.realtime import FeatureConfig
from ai.personalization.cache.feature_cache import CacheConfig
from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    UserFeedback,
    RecommendationScenario
)


@pytest.fixture
async def mock_redis():
    """Mock Redis客户端"""
    redis_mock = AsyncMock(spec=Redis)
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.zrevrange.return_value = []
    redis_mock.smembers.return_value = set()
    redis_mock.sadd.return_value = 1
    redis_mock.expire.return_value = True
    return redis_mock


@pytest.fixture
async def engine(mock_redis):
    """创建个性化引擎实例"""
    feature_config = FeatureConfig()
    cache_config = CacheConfig()
    
    engine = PersonalizationEngine(
        redis_client=mock_redis,
        feature_config=feature_config,
        cache_config=cache_config
    )
    
    # Mock模型服务
    engine.model_service.initialize = AsyncMock()
    engine.model_service.get_active_version = AsyncMock(return_value="v1.0.0")
    engine.model_service.batch_predict = AsyncMock(return_value=np.array([0.8, 0.6, 0.7]))
    
    # Mock特征引擎
    engine.feature_engine.start = AsyncMock()
    engine.feature_engine.compute_features = AsyncMock()
    
    await engine.start()
    return engine


@pytest.mark.asyncio
class TestPersonalizationEngine:
    """个性化引擎测试"""
    
    async def test_engine_initialization(self, mock_redis):
        """测试引擎初始化"""
        engine = PersonalizationEngine(mock_redis)
        
        assert engine.redis == mock_redis
        assert engine.feature_engine is not None
        assert engine.model_service is not None
        assert engine.feature_cache is not None
        assert engine.result_cache is not None
        assert engine.metrics["total_requests"] == 0
    
    async def test_get_recommendations_success(self, engine):
        """测试成功获取推荐"""
        # 准备测试数据
        request = RecommendationRequest(
            user_id="test_user",
            scenario=RecommendationScenario.CONTENT,
            n_recommendations=5,
            context={"source": "test"}
        )
        
        # Mock特征计算
        from models.schemas.personalization import RealTimeFeatures
        mock_features = RealTimeFeatures(
            temporal={"hour": 14.0},
            behavioral={"clicks": 10.0},
            contextual={"device": 1.0},
            aggregated={"avg_score": 0.8}
        )
        engine.feature_engine.compute_features.return_value = mock_features
        
        # Mock用户画像
        engine._get_user_profile = AsyncMock(return_value=None)
        
        # Mock候选获取
        engine._get_candidates = AsyncMock(return_value=[
            {"item_id": "item1", "type": "article"},
            {"item_id": "item2", "type": "article"},
            {"item_id": "item3", "type": "article"}
        ])
        
        # 执行推荐
        response = await engine.get_recommendations(request)
        
        # 验证结果
        assert isinstance(response, RecommendationResponse)
        assert response.user_id == "test_user"
        assert len(response.recommendations) <= 5
        assert response.latency_ms >= 0
        assert response.model_version == "v1.0.0"
        assert response.scenario == RecommendationScenario.CONTENT
        
        # 验证指标更新
        assert engine.metrics["total_requests"] > 0
    
    async def test_get_recommendations_with_cache(self, engine, mock_redis):
        """测试缓存命中的推荐"""
        request = RecommendationRequest(
            user_id="test_user",
            scenario=RecommendationScenario.CONTENT,
            use_cache=True
        )
        
        # Mock缓存命中
        cached_response = RecommendationResponse(
            request_id="cached_id",
            user_id="test_user",
            recommendations=[],
            latency_ms=0.0,
            model_version="v1.0.0",
            scenario=RecommendationScenario.CONTENT
        )
        engine.result_cache.get_cached_result = AsyncMock(return_value=cached_response)
        
        response = await engine.get_recommendations(request)
        
        assert response.request_id == "cached_id"
        assert engine.metrics["cache_hits"] > 0
    
    async def test_get_recommendations_fallback(self, engine, mock_redis):
        """测试降级策略"""
        request = RecommendationRequest(
            user_id="test_user",
            scenario=RecommendationScenario.CONTENT
        )
        
        # Mock热门推荐
        mock_redis.zrevrange.return_value = [
            (b"popular_item1", 0.9),
            (b"popular_item2", 0.8)
        ]
        
        # Mock特征计算失败
        engine.feature_engine.compute_features.side_effect = Exception("Feature error")
        
        response = await engine.get_recommendations(request)
        
        # 应该返回降级响应
        assert isinstance(response, RecommendationResponse)
        assert response.user_id == "test_user"
        # 在降级模式下，可能返回空推荐
    
    async def test_process_feedback(self, engine):
        """测试反馈处理"""
        feedback = UserFeedback(
            user_id="test_user",
            item_id="item1",
            feedback_type="like",
            feedback_value=1.0,
            context={"source": "test"}
        )
        
        # Mock方法
        engine._update_user_profile = AsyncMock()
        engine.result_cache.invalidate_user_cache = AsyncMock()
        
        # 处理反馈
        await engine.process_feedback(feedback)
        
        # 验证调用
        engine._update_user_profile.assert_called_once_with(feedback)
        engine.result_cache.invalidate_user_cache.assert_called_once_with("test_user")
    
    async def test_circuit_breaker(self, engine):
        """测试熔断器功能"""
        # 模拟多次错误
        engine.circuit_breaker_threshold = 0.5
        engine.metrics["total_requests"] = 10
        
        # 触发多个错误
        for _ in range(6):
            engine._record_error()
        
        # 验证熔断器状态
        assert engine.circuit_breaker_open is True
    
    async def test_performance_metrics(self, engine):
        """测试性能指标"""
        # 更新一些指标
        engine._update_metrics(100.0, success=True)
        engine._update_metrics(200.0, success=True)
        engine._update_metrics(150.0, success=False)
        
        metrics = engine.get_metrics()
        
        assert "avg_latency_ms" in metrics
        assert "p99_latency_ms" in metrics
        assert "total_requests" in metrics
        assert "cache_hit_rate" in metrics
        assert "circuit_breaker_open" in metrics
    
    async def test_feature_vector_preparation(self, engine):
        """测试特征向量准备"""
        from models.schemas.personalization import RealTimeFeatures, UserProfile
        
        features = RealTimeFeatures(
            temporal={"hour": 14.0, "day": 1.0},
            behavioral={"clicks": 10.0, "views": 20.0},
            contextual={"device": 1.0},
            aggregated={"avg": 0.8}
        )
        
        user_profile = UserProfile(
            user_id="test_user",
            features={"age": 25.0, "gender": 1.0},
            preferences={"news": 0.8, "sports": 0.6}
        )
        
        feature_vector = engine._prepare_feature_vector(features, user_profile)
        
        assert isinstance(feature_vector, np.ndarray)
        assert feature_vector.dtype == np.float32
        assert len(feature_vector) > 0
    
    async def test_diversity_application(self, engine):
        """测试多样性策略"""
        from models.schemas.personalization import RecommendationItem
        
        recommendations = [
            RecommendationItem(
                item_id="item1",
                score=0.9,
                confidence=0.8,
                metadata={"type": "news"}
            ),
            RecommendationItem(
                item_id="item2", 
                score=0.8,
                confidence=0.7,
                metadata={"type": "news"}
            ),
            RecommendationItem(
                item_id="item3",
                score=0.7,
                confidence=0.6,
                metadata={"type": "sports"}
            )
        ]
        
        diverse_recs = await engine._apply_diversity(recommendations, 0.5)
        
        assert len(diverse_recs) == 3
        assert all(isinstance(rec, RecommendationItem) for rec in diverse_recs)
        # 验证分数被调整
        assert diverse_recs[0].score <= recommendations[0].score
    
    async def test_reward_calculation(self, engine):
        """测试奖励计算"""
        feedback_like = UserFeedback(
            user_id="user1",
            item_id="item1", 
            feedback_type="like",
            feedback_value=1.0
        )
        
        feedback_purchase = UserFeedback(
            user_id="user1",
            item_id="item1",
            feedback_type="purchase", 
            feedback_value=1.0
        )
        
        feedback_dislike = UserFeedback(
            user_id="user1",
            item_id="item1",
            feedback_type="dislike",
            feedback_value=1.0
        )
        
        assert engine._calculate_reward(feedback_like) == 0.3
        assert engine._calculate_reward(feedback_purchase) == 1.0
        assert engine._calculate_reward(feedback_dislike) == -0.5
    
    async def test_engine_lifecycle(self, engine):
        """测试引擎生命周期"""
        # 引擎已经在fixture中启动
        assert engine._running is True
        
        # 停止引擎
        await engine.stop()
        assert engine._running is False


@pytest.mark.asyncio
async def test_engine_error_handling(mock_redis):
    """测试错误处理"""
    engine = PersonalizationEngine(mock_redis)
    
    # Mock初始化失败
    engine.model_service.initialize = AsyncMock(side_effect=Exception("Init failed"))
    
    # 应该能够处理初始化错误
    await engine.start()
    
    # 即使初始化失败，引擎仍应该可用（降级模式）
    request = RecommendationRequest(user_id="test_user")
    response = await engine.get_recommendations(request)
    
    assert isinstance(response, RecommendationResponse)


@pytest.mark.asyncio
async def test_concurrent_requests(engine):
    """测试并发请求处理"""
    requests = [
        RecommendationRequest(
            user_id=f"user_{i}",
            scenario=RecommendationScenario.CONTENT
        )
        for i in range(10)
    ]
    
    # Mock必要的方法
    engine._get_user_profile = AsyncMock(return_value=None)
    engine._get_candidates = AsyncMock(return_value=[
        {"item_id": "item1", "type": "article"}
    ])
    
    # 并发执行推荐
    tasks = [engine.get_recommendations(req) for req in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 验证所有请求都得到处理
    assert len(responses) == 10
    assert all(isinstance(r, RecommendationResponse) for r in responses if not isinstance(r, Exception))
    
    # 验证请求计数
    assert engine.metrics["total_requests"] >= 10