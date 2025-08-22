"""
个性化API端点测试
使用FastAPI TestClient测试所有个性化推荐API端点
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from datetime import datetime
from typing import Dict, Any

from api.v1.personalization import router
from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    UserFeedback,
    ModelConfig,
    RecommendationItem,
    RecommendationScenario
)


@pytest.fixture
def app():
    """创建测试FastAPI应用"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """创建模拟个性化服务"""
    service = Mock()
    
    # 模拟推荐响应
    service.get_recommendations = AsyncMock(return_value=RecommendationResponse(
        request_id="test_request_123",
        user_id="test_user",
        recommendations=[
            RecommendationItem(
                item_id="item_1",
                score=0.95,
                confidence=0.88,
                explanation="基于用户历史偏好推荐"
            ),
            RecommendationItem(
                item_id="item_2", 
                score=0.87,
                confidence=0.82,
                explanation="相似用户喜欢的内容"
            )
        ],
        latency_ms=45.2,
        model_version="v1.2.3",
        explanation="基于协同过滤和内容匹配的混合推荐",
        cache_hit=False
    ))
    
    # 模拟用户画像
    service.get_user_profile = AsyncMock(return_value=UserProfile(
        user_id="test_user",
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
        last_updated=datetime.utcnow(),
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
    ))
    
    # 模拟其他方法
    service.update_user_profile = AsyncMock(return_value=True)
    service.process_feedback = AsyncMock()
    service.get_realtime_features = AsyncMock(return_value={
        "temporal": {"hour_of_day": 14, "day_of_week": 2},
        "behavioral": {"recent_clicks": 5, "session_duration": 180},
        "contextual": {"device": "mobile", "location": "home"},
        "aggregated": {"engagement_score": 0.75}
    })
    service.compute_features = AsyncMock(return_value={
        "temporal": {"hour_of_day": 14},
        "behavioral": {"recent_clicks": 5},
        "contextual": {"device": "mobile"}
    })
    service.get_model_status = AsyncMock(return_value={
        "model_id": "default",
        "status": "active",
        "version": "v1.2.3",
        "last_updated": datetime.utcnow().isoformat()
    })
    service.predict = AsyncMock(return_value=[0.8, 0.7, 0.6])
    service.update_model = AsyncMock()
    service.get_cache_stats = AsyncMock(return_value={
        "hit_rate": 0.85,
        "total_requests": 10000,
        "cache_hits": 8500,
        "cache_misses": 1500
    })
    service.invalidate_user_cache = AsyncMock(return_value=5)
    service.get_performance_metrics = AsyncMock(return_value={
        "avg_latency_ms": 42.5,
        "p95_latency_ms": 78.2,
        "p99_latency_ms": 95.1,
        "qps": 1250.5,
        "error_rate": 0.001
    })
    
    return service


@pytest.fixture
def mock_redis():
    """创建模拟Redis客户端"""
    redis = Mock()
    redis.setex = AsyncMock()
    redis.hincrby = AsyncMock()
    return redis


class TestPersonalizationAPI:
    """个性化API测试套件"""
    
    def test_get_recommendations_success(self, client, mock_service, mock_redis):
        """测试获取推荐 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            request_data = {
                "user_id": "test_user",
                "context": {"page": "homepage", "device": "mobile"},
                "n_recommendations": 10,
                "scenario": "content_discovery"
            }
            
            response = client.post("/api/v1/personalization/recommend", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test_user"
            assert data["request_id"] == "test_request_123"
            assert len(data["recommendations"]) == 2
            assert data["latency_ms"] == 45.2
            assert data["model_version"] == "v1.2.3"
            
            # 验证推荐项结构
            rec = data["recommendations"][0]
            assert rec["item_id"] == "item_1"
            assert rec["score"] == 0.95
            assert rec["confidence"] == 0.88
            assert "explanation" in rec

    def test_get_recommendations_validation_error(self, client):
        """测试获取推荐 - 请求验证错误"""
        
        # 缺少必需字段
        request_data = {
            "context": {"page": "homepage"},
            "n_recommendations": 10
            # 缺少user_id
        }
        
        response = client.post("/api/v1/personalization/recommend", json=request_data)
        
        assert response.status_code == 422  # Validation error
        assert "user_id" in response.text

    def test_get_recommendations_invalid_scenario(self, client):
        """测试获取推荐 - 无效场景"""
        
        request_data = {
            "user_id": "test_user",
            "context": {"page": "homepage"},
            "n_recommendations": 10,
            "scenario": "invalid_scenario"  # 无效场景
        }
        
        response = client.post("/api/v1/personalization/recommend", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_get_user_profile_success(self, client, mock_service, mock_redis):
        """测试获取用户画像 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/user/test_user/profile")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "test_user"
            assert "features" in data
            assert "preferences" in data
            assert "behavior_history" in data
            assert data["features"]["age_group"] == 25.0

    def test_get_user_profile_not_found(self, client, mock_service, mock_redis):
        """测试获取用户画像 - 用户不存在"""
        
        mock_service.get_user_profile.return_value = None
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/user/nonexistent_user/profile")
            
            assert response.status_code == 404
            assert "用户画像不存在" in response.json()["detail"]

    def test_update_user_profile_success(self, client, mock_service, mock_redis):
        """测试更新用户画像 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            profile_data = {
                "user_id": "test_user",
                "features": {"age_group": 30.0, "activity_level": 0.9},
                "preferences": {"technology": 0.95, "sports": 0.4},
                "behavior_history": [],
                "last_updated": datetime.utcnow().isoformat(),
                "embedding": [0.1, 0.2, 0.3]
            }
            
            response = client.put("/api/v1/personalization/user/test_user/profile", json=profile_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "更新成功" in data["message"]

    def test_submit_feedback_success(self, client, mock_service, mock_redis):
        """测试提交反馈 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            feedback_data = {
                "user_id": "test_user",
                "item_id": "item_123",
                "feedback_type": "click",
                "feedback_value": 1.0,
                "context": {"source": "homepage"},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = client.post("/api/v1/personalization/feedback", json=feedback_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "已接收" in data["message"]

    def test_get_realtime_features_success(self, client, mock_service, mock_redis):
        """测试获取实时特征 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/features/realtime/test_user")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "temporal" in data
            assert "behavioral" in data
            assert "contextual" in data
            assert "aggregated" in data

    def test_compute_features_batch_success(self, client, mock_service, mock_redis):
        """测试批量计算特征 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            request_data = ["user_1", "user_2", "user_3"]
            
            response = client.post("/api/v1/personalization/features/compute", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "user_1" in data
            assert "user_2" in data
            assert "user_3" in data

    def test_get_model_status_success(self, client, mock_service, mock_redis):
        """测试获取模型状态 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/models/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model_id"] == "default"
            assert data["status"] == "active"
            assert data["version"] == "v1.2.3"

    def test_model_predict_success(self, client, mock_service, mock_redis):
        """测试模型预测 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            request_data = {
                "model_id": "test_model",
                "features": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            
            response = client.post("/api/v1/personalization/models/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model_id"] == "test_model"
            assert "prediction" in data
            assert "timestamp" in data

    def test_update_model_success(self, client, mock_service, mock_redis):
        """测试更新模型 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            model_config = {
                "model_id": "test_model",
                "version": "v2.0.0",
                "config": {"learning_rate": 0.01, "batch_size": 32},
                "description": "Updated model configuration"
            }
            
            response = client.put("/api/v1/personalization/models/update", json=model_config)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "accepted"
            assert data["model_id"] == "test_model"

    def test_get_cache_stats_success(self, client, mock_service, mock_redis):
        """测试获取缓存统计 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/cache/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["hit_rate"] == 0.85
            assert data["total_requests"] == 10000
            assert data["cache_hits"] == 8500

    def test_invalidate_user_cache_success(self, client, mock_service, mock_redis):
        """测试失效用户缓存 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.post("/api/v1/personalization/cache/invalidate/test_user")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "5 个缓存项" in data["message"]

    def test_get_performance_metrics_success(self, client, mock_service, mock_redis):
        """测试获取性能指标 - 成功场景"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response = client.get("/api/v1/personalization/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["avg_latency_ms"] == 42.5
            assert data["p95_latency_ms"] == 78.2
            assert data["qps"] == 1250.5

    def test_api_error_handling(self, client, mock_service, mock_redis):
        """测试API错误处理"""
        
        # 模拟服务异常
        mock_service.get_recommendations.side_effect = Exception("服务异常")
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            request_data = {
                "user_id": "test_user",
                "context": {"page": "homepage"},
                "n_recommendations": 10,
                "scenario": "content_discovery"
            }
            
            response = client.post("/api/v1/personalization/recommend", json=request_data)
            
            assert response.status_code == 500
            assert "服务异常" in response.json()["detail"]

    def test_request_rate_limiting(self, client, mock_service, mock_redis):
        """测试请求频率限制（如果实现）"""
        # 这个测试取决于是否实现了频率限制
        # 可以根据实际实现进行调整
        pass

    def test_concurrent_requests(self, client, mock_service, mock_redis):
        """测试并发请求处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            # 创建多个并发请求
            import threading
            import time
            
            results = []
            errors = []
            
            def make_request():
                try:
                    request_data = {
                        "user_id": f"test_user_{threading.current_thread().ident}",
                        "context": {"page": "homepage"},
                        "n_recommendations": 5,
                        "scenario": "content_discovery"
                    }
                    
                    response = client.post("/api/v1/personalization/recommend", json=request_data)
                    results.append(response.status_code)
                except Exception as e:
                    errors.append(str(e))
            
            # 创建并启动多个线程
            threads = []
            for i in range(10):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证结果
            assert len(errors) == 0, f"并发请求出现错误: {errors}"
            assert all(status == 200 for status in results), f"部分请求失败: {results}"

    def test_api_response_format_consistency(self, client, mock_service, mock_redis):
        """测试API响应格式一致性"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            # 测试多个端点的响应格式
            endpoints = [
                ("/api/v1/personalization/cache/stats", "GET"),
                ("/api/v1/personalization/metrics", "GET"),
                ("/api/v1/personalization/models/status", "GET"),
                ("/api/v1/personalization/features/realtime/test_user", "GET")
            ]
            
            for endpoint, method in endpoints:
                if method == "GET":
                    response = client.get(endpoint)
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/json"
                
                # 验证响应是有效的JSON
                data = response.json()
                assert isinstance(data, dict)

    def test_input_sanitization(self, client, mock_service, mock_redis):
        """测试输入清理和安全性"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            # 测试SQL注入尝试
            malicious_user_id = "'; DROP TABLE users; --"
            
            response = client.get(f"/api/v1/personalization/user/{malicious_user_id}/profile")
            
            # 应该正常处理，不会导致系统错误
            assert response.status_code in [200, 404, 500]  # 不应该导致系统崩溃
            
            # 测试XSS尝试
            malicious_context = {
                "page": "<script>alert('xss')</script>",
                "data": "../../etc/passwd"
            }
            
            request_data = {
                "user_id": "test_user",
                "context": malicious_context,
                "n_recommendations": 5,
                "scenario": "content_discovery"
            }
            
            response = client.post("/api/v1/personalization/recommend", json=request_data)
            
            # 应该正常处理或返回验证错误
            assert response.status_code in [200, 400, 422]


class TestPersonalizationAPIPerformance:
    """API性能测试"""
    
    def test_api_response_time(self, client, mock_service, mock_redis):
        """测试API响应时间"""
        import time
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            request_data = {
                "user_id": "test_user",
                "context": {"page": "homepage"},
                "n_recommendations": 10,
                "scenario": "content_discovery"
            }
            
            # 测量响应时间
            start_time = time.time()
            response = client.post("/api/v1/personalization/recommend", json=request_data)
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            assert response.status_code == 200
            assert response_time_ms < 1000  # API层响应时间应该在1秒内

    def test_large_payload_handling(self, client, mock_service, mock_redis):
        """测试大负载处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            # 创建大的上下文数据
            large_context = {
                f"field_{i}": f"value_{i}" * 100 for i in range(100)
            }
            
            request_data = {
                "user_id": "test_user",
                "context": large_context,
                "n_recommendations": 50,  # 请求更多推荐
                "scenario": "content_discovery"
            }
            
            response = client.post("/api/v1/personalization/recommend", json=request_data)
            
            # 应该能正常处理大负载
            assert response.status_code in [200, 400, 413]  # 200成功，400验证错误，413负载过大


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])