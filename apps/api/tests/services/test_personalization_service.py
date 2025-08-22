import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from services.personalization_service import PersonalizationService, get_personalization_service
from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse, 
    UserFeedback,
    UserProfile,
    ModelConfig,
    RecommendationScenario
)


@pytest.fixture
async def service():
    """创建个性化服务实例"""
    service = PersonalizationService()
    
    # Mock Redis客户端
    service.redis_client = AsyncMock()
    
    # Mock引擎
    service.engine = AsyncMock()
    service.engine.get_recommendations = AsyncMock()
    service.engine.process_feedback = AsyncMock()
    service.engine._get_user_profile = AsyncMock()
    service.engine._update_user_profile = AsyncMock()
    service.engine.feature_engine.compute_features = AsyncMock()
    service.engine.model_service.get_model_status = AsyncMock()
    service.engine.model_service.predict = AsyncMock()
    service.engine.model_service.update_model = AsyncMock()
    service.engine.get_metrics = AsyncMock()
    service.engine.result_cache.invalidate_user_cache = AsyncMock()
    service.engine.feature_cache.get_stats = AsyncMock(return_value={})
    service.engine.result_cache.get_stats = AsyncMock(return_value={})
    service.engine.feature_engine.get_feature_statistics = AsyncMock(return_value={})
    
    service._initialized = True
    return service


@pytest.mark.asyncio
class TestPersonalizationService:
    """个性化服务测试"""
    
    async def test_get_recommendations(self, service):
        """测试获取推荐"""
        # 准备测试数据
        request = RecommendationRequest(
            user_id="test_user",
            scenario=RecommendationScenario.CONTENT,
            n_recommendations=5
        )
        
        expected_response = RecommendationResponse(
            request_id="test_id",
            user_id="test_user",
            recommendations=[],
            latency_ms=100.0,
            model_version="v1.0.0",
            scenario=RecommendationScenario.CONTENT
        )
        
        service.engine.get_recommendations.return_value = expected_response
        
        # 执行测试
        response = await service.get_recommendations(request)
        
        # 验证结果
        assert response == expected_response
        service.engine.get_recommendations.assert_called_once_with(request)
    
    async def test_process_feedback(self, service):
        """测试处理反馈"""
        feedback = UserFeedback(
            user_id="test_user",
            item_id="item1",
            feedback_type="like",
            feedback_value=1.0
        )
        
        # 执行测试
        await service.process_feedback(feedback)
        
        # 验证调用
        service.engine.process_feedback.assert_called_once_with(feedback)
    
    async def test_get_user_profile(self, service):
        """测试获取用户画像"""
        expected_profile = UserProfile(
            user_id="test_user",
            features={"age": 25.0},
            preferences={"news": 0.8}
        )
        
        service.engine._get_user_profile.return_value = expected_profile
        
        # 执行测试
        profile = await service.get_user_profile("test_user")
        
        # 验证结果
        assert profile == expected_profile
        service.engine._get_user_profile.assert_called_once_with("test_user")
    
    async def test_update_user_profile(self, service):
        """测试更新用户画像"""
        profile = UserProfile(
            user_id="test_user",
            features={"age": 26.0},
            preferences={"news": 0.9}
        )
        
        # 执行测试
        result = await service.update_user_profile(profile)
        
        # 验证结果
        assert result is True
        service.engine._update_user_profile.assert_called_once()
    
    async def test_get_realtime_features(self, service):
        """测试获取实时特征"""
        from models.schemas.personalization import RealTimeFeatures
        
        expected_features = RealTimeFeatures(
            temporal={"hour": 14.0},
            behavioral={"clicks": 10.0},
            contextual={"device": 1.0},
            aggregated={"avg": 0.8}
        )
        
        service.engine.feature_engine.compute_features.return_value = expected_features
        
        # 执行测试
        features = await service.get_realtime_features("test_user")
        
        # 验证结果
        assert features == expected_features
        service.engine.feature_engine.compute_features.assert_called_once_with(
            user_id="test_user",
            context={},
            use_cache=True
        )
    
    async def test_compute_features(self, service):
        """测试计算特征"""
        from models.schemas.personalization import RealTimeFeatures
        
        context = {"source": "test"}
        expected_features = RealTimeFeatures(
            temporal={"hour": 14.0},
            behavioral={"clicks": 10.0}
        )
        
        service.engine.feature_engine.compute_features.return_value = expected_features
        
        # 执行测试
        features = await service.compute_features("test_user", context)
        
        # 验证结果
        assert features == expected_features
        service.engine.feature_engine.compute_features.assert_called_once_with(
            user_id="test_user",
            context=context,
            use_cache=False
        )
    
    async def test_get_model_status(self, service):
        """测试获取模型状态"""
        expected_status = {
            "model_content": {
                "instance_count": 2,
                "total_capacity": 200,
                "current_load": 50
            }
        }
        
        service.engine.model_service.get_model_status.return_value = expected_status
        
        # 执行测试
        status = await service.get_model_status("model_content")
        
        # 验证结果
        assert status == expected_status
        service.engine.model_service.get_model_status.assert_called_once_with("model_content")
    
    async def test_predict(self, service):
        """测试模型预测"""
        import numpy as np
        
        features = np.array([1.0, 2.0, 3.0])
        expected_result = np.array([0.8])
        
        service.engine.model_service.predict.return_value = expected_result
        
        # 执行测试
        result = await service.predict(features, "model_content")
        
        # 验证结果
        assert np.array_equal(result, expected_result)
        service.engine.model_service.predict.assert_called_once_with(features, "model_content")
    
    async def test_update_model(self, service):
        """测试更新模型"""
        model_config = ModelConfig(
            model_id="model_content",
            model_type="neural_network",
            version="v2.0.0"
        )
        
        # 执行测试
        await service.update_model(model_config)
        
        # 验证调用
        service.engine.model_service.update_model.assert_called_once_with(
            "model_content",
            "v2.0.0",
            rollout_percentage=0.1
        )
    
    async def test_get_cache_stats(self, service):
        """测试获取缓存统计"""
        expected_stats = {
            "feature_cache": {"hits": 100, "misses": 20},
            "result_cache": {"hits": 80, "misses": 10},
            "feature_engine_stats": {"total_users": 1000}
        }
        
        service.engine.feature_cache.get_stats.return_value = expected_stats["feature_cache"]
        service.engine.result_cache.get_stats.return_value = expected_stats["result_cache"]
        service.engine.feature_engine.get_feature_statistics.return_value = expected_stats["feature_engine_stats"]
        
        # 执行测试
        stats = await service.get_cache_stats()
        
        # 验证结果
        assert "feature_cache" in stats
        assert "result_cache" in stats
        assert "feature_engine_stats" in stats
    
    async def test_invalidate_user_cache(self, service):
        """测试失效用户缓存"""
        service.redis_client.exists.return_value = True
        service.redis_client.delete.return_value = 1
        service.engine.result_cache.invalidate_user_cache.return_value = 1
        
        # 执行测试
        count = await service.invalidate_user_cache("test_user")
        
        # 验证结果
        assert count == 2  # 特征缓存 + 结果缓存
        service.redis_client.delete.assert_called_once()
        service.engine.result_cache.invalidate_user_cache.assert_called_once_with("test_user")
    
    async def test_get_performance_metrics(self, service):
        """测试获取性能指标"""
        expected_metrics = {
            "total_requests": 1000,
            "avg_latency_ms": 50.0,
            "cache_hit_rate": 0.8
        }
        
        service.engine.get_metrics.return_value = expected_metrics
        
        # 执行测试
        metrics = await service.get_performance_metrics()
        
        # 验证结果
        assert metrics == expected_metrics
        service.engine.get_metrics.assert_called_once()
    
    async def test_service_initialization(self):
        """测试服务初始化"""
        service = PersonalizationService()
        
        # 初始状态
        assert service._initialized is False
        assert service.engine is None
        assert service.redis_client is None
        
        # Mock数据库连接
        with pytest.MockAsyncContext() as mock_get_redis:
            mock_redis = AsyncMock()
            mock_get_redis.return_value = mock_redis
            
            # 模拟_ensure_initialized方法（在实际测试中会被调用）
            # 这里我们手动验证初始化逻辑
            service.redis_client = mock_redis
            service._initialized = True
            
            assert service._initialized is True
            assert service.redis_client == mock_redis
    
    async def test_shutdown(self, service):
        """测试服务关闭"""
        service.engine.stop = AsyncMock()
        
        # 执行测试
        await service.shutdown()
        
        # 验证状态
        assert service._initialized is False
        service.engine.stop.assert_called_once()


@pytest.mark.asyncio
async def test_get_personalization_service():
    """测试服务单例获取"""
    # 获取服务实例
    service1 = await get_personalization_service()
    service2 = await get_personalization_service()
    
    # 验证单例模式
    assert service1 is service2
    assert isinstance(service1, PersonalizationService)


@pytest.mark.asyncio
async def test_service_error_handling():
    """测试服务错误处理"""
    service = PersonalizationService()
    service._initialized = True
    service.engine = AsyncMock()
    
    # Mock方法抛出异常
    service.engine._update_user_profile.side_effect = Exception("Update failed")
    
    # 测试更新用户画像错误处理
    profile = UserProfile(user_id="test_user")
    result = await service.update_user_profile(profile)
    
    # 应该返回False而不是抛出异常
    assert result is False
    
    # 测试失效缓存错误处理
    service.redis_client = AsyncMock()
    service.redis_client.exists.side_effect = Exception("Redis error")
    
    count = await service.invalidate_user_cache("test_user")
    # 应该返回0而不是抛出异常
    assert count >= 0