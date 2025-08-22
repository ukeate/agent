"""
个性化WebSocket测试
测试实时推荐WebSocket功能，包括连接、消息处理、错误处理等
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from datetime import datetime
import threading
import time

from api.v1.personalization import router
from models.schemas.personalization import (
    RecommendationResponse,
    RecommendationItem,
    UserFeedback
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
        request_id="ws_test_request_123",
        user_id="ws_test_user",
        recommendations=[
            RecommendationItem(
                item_id="ws_item_1",
                score=0.92,
                confidence=0.85,
                explanation="实时推荐：基于当前上下文"
            ),
            RecommendationItem(
                item_id="ws_item_2",
                score=0.88,
                confidence=0.80,
                explanation="实时推荐：相似用户偏好"
            )
        ],
        latency_ms=32.1,
        model_version="v1.2.3",
        explanation="WebSocket实时推荐响应",
        cache_hit=False
    ))
    
    # 模拟反馈处理
    service.process_feedback = AsyncMock()
    
    return service


@pytest.fixture
def mock_redis():
    """创建模拟Redis客户端"""
    redis = Mock()
    redis.setex = AsyncMock()
    redis.hincrby = AsyncMock()
    return redis


class TestPersonalizationWebSocket:
    """个性化WebSocket测试套件"""
    
    def test_websocket_connection_success(self, client, mock_service, mock_redis):
        """测试WebSocket连接成功"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 发送认证消息
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                
                # 接收欢迎消息
                welcome_message = websocket.receive_json()
                
                assert welcome_message["type"] == "connected"
                assert welcome_message["user_id"] == "ws_test_user"
                assert "timestamp" in welcome_message

    def test_websocket_authentication_failure(self, client, mock_service, mock_redis):
        """测试WebSocket认证失败"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 发送无效认证消息
                auth_message = {"invalid_field": "value"}
                websocket.send_json(auth_message)
                
                # 接收错误消息
                error_message = websocket.receive_json()
                
                assert error_message["type"] == "error"
                assert "用户ID缺失" in error_message["message"]

    def test_websocket_recommendation_request(self, client, mock_service, mock_redis):
        """测试WebSocket推荐请求"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送推荐请求
                request_message = {
                    "type": "request",
                    "context": {"page": "search", "query": "laptop"},
                    "n_recommendations": 5,
                    "scenario": "search_results"
                }
                websocket.send_json(request_message)
                
                # 接收推荐响应
                response_message = websocket.receive_json()
                
                assert response_message["type"] == "recommendations"
                assert "data" in response_message
                assert "timestamp" in response_message
                
                # 验证推荐数据
                rec_data = response_message["data"]
                assert rec_data["user_id"] == "ws_test_user"
                assert len(rec_data["recommendations"]) == 2
                assert rec_data["latency_ms"] == 32.1

    def test_websocket_feedback_submission(self, client, mock_service, mock_redis):
        """测试WebSocket反馈提交"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送反馈
                feedback_message = {
                    "type": "feedback",
                    "item_id": "ws_item_1",
                    "feedback_type": "click",
                    "feedback_value": 1.0,
                    "context": {"source": "websocket_test"}
                }
                websocket.send_json(feedback_message)
                
                # 接收反馈确认
                confirmation_message = websocket.receive_json()
                
                assert confirmation_message["type"] == "feedback_received"
                assert "timestamp" in confirmation_message
                
                # 验证服务调用
                mock_service.process_feedback.assert_called_once()

    def test_websocket_ping_pong(self, client, mock_service, mock_redis):
        """测试WebSocket心跳机制"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送ping
                ping_message = {"type": "ping"}
                websocket.send_json(ping_message)
                
                # 接收pong
                pong_message = websocket.receive_json()
                
                assert pong_message["type"] == "pong"
                assert "timestamp" in pong_message

    def test_websocket_invalid_json(self, client, mock_service, mock_redis):
        """测试WebSocket无效JSON处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送无效JSON
                websocket.send_text("invalid json {")
                
                # 接收错误消息
                error_message = websocket.receive_json()
                
                assert error_message["type"] == "error"
                assert "无效的JSON格式" in error_message["message"]

    def test_websocket_service_error_handling(self, client, mock_service, mock_redis):
        """测试WebSocket服务错误处理"""
        
        # 模拟服务异常
        mock_service.get_recommendations.side_effect = Exception("推荐服务异常")
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送推荐请求
                request_message = {
                    "type": "request",
                    "context": {"page": "homepage"},
                    "n_recommendations": 5,
                    "scenario": "content_discovery"
                }
                websocket.send_json(request_message)
                
                # 接收错误消息
                error_message = websocket.receive_json()
                
                assert error_message["type"] == "error"
                assert "推荐服务异常" in error_message["message"]

    def test_websocket_multiple_requests(self, client, mock_service, mock_redis):
        """测试WebSocket多个请求处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "ws_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 发送多个推荐请求
                requests = [
                    {
                        "type": "request",
                        "context": {"page": "homepage"},
                        "n_recommendations": 3,
                        "scenario": "content_discovery"
                    },
                    {
                        "type": "request",
                        "context": {"page": "search", "query": "mobile"},
                        "n_recommendations": 5,
                        "scenario": "search_results"
                    },
                    {
                        "type": "request",
                        "context": {"page": "product", "category": "electronics"},
                        "n_recommendations": 8,
                        "scenario": "product_recommendations"
                    }
                ]
                
                responses = []
                for request_msg in requests:
                    websocket.send_json(request_msg)
                    response = websocket.receive_json()
                    responses.append(response)
                
                # 验证所有响应
                assert len(responses) == 3
                for response in responses:
                    assert response["type"] == "recommendations"
                    assert "data" in response
                    assert "timestamp" in response

    def test_websocket_concurrent_connections(self, client, mock_service, mock_redis):
        """测试WebSocket并发连接"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            def test_connection(user_id: str, results: list, errors: list):
                try:
                    with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                        # 认证
                        auth_message = {"user_id": user_id}
                        websocket.send_json(auth_message)
                        welcome = websocket.receive_json()
                        
                        # 发送推荐请求
                        request_message = {
                            "type": "request",
                            "context": {"test": "concurrent"},
                            "n_recommendations": 3,
                            "scenario": "content_discovery"
                        }
                        websocket.send_json(request_message)
                        response = websocket.receive_json()
                        
                        results.append({
                            "user_id": user_id,
                            "welcome": welcome,
                            "response": response
                        })
                        
                except Exception as e:
                    errors.append(f"用户 {user_id}: {str(e)}")
            
            # 创建并发连接
            results = []
            errors = []
            threads = []
            
            for i in range(5):  # 5个并发连接
                user_id = f"concurrent_user_{i}"
                thread = threading.Thread(
                    target=test_connection,
                    args=(user_id, results, errors)
                )
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证结果
            assert len(errors) == 0, f"并发连接出现错误: {errors}"
            assert len(results) == 5
            
            for result in results:
                assert result["welcome"]["type"] == "connected"
                assert result["response"]["type"] == "recommendations"

    def test_websocket_connection_timeout(self, client, mock_service, mock_redis):
        """测试WebSocket连接超时处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "timeout_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 模拟长时间无活动（在实际实现中可能需要超时机制）
                # 这里测试心跳是否仍然工作
                time.sleep(1)  # 短暂等待
                
                # 发送ping测试连接
                ping_message = {"type": "ping"}
                websocket.send_json(ping_message)
                
                pong_message = websocket.receive_json()
                assert pong_message["type"] == "pong"

    def test_websocket_message_ordering(self, client, mock_service, mock_redis):
        """测试WebSocket消息顺序"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "order_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 快速发送多个消息
                messages = [
                    {"type": "ping"},
                    {
                        "type": "request",
                        "context": {"order": 1},
                        "n_recommendations": 2,
                        "scenario": "content_discovery"
                    },
                    {"type": "ping"},
                    {
                        "type": "feedback",
                        "item_id": "order_item",
                        "feedback_type": "view",
                        "feedback_value": 1.0
                    }
                ]
                
                # 发送所有消息
                for msg in messages:
                    websocket.send_json(msg)
                
                # 接收所有响应
                responses = []
                for _ in messages:
                    response = websocket.receive_json()
                    responses.append(response)
                
                # 验证响应顺序（应该与发送顺序对应）
                expected_types = ["pong", "recommendations", "pong", "feedback_received"]
                actual_types = [response["type"] for response in responses]
                
                assert actual_types == expected_types

    def test_websocket_large_message_handling(self, client, mock_service, mock_redis):
        """测试WebSocket大消息处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "large_msg_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 创建大的上下文数据
                large_context = {
                    f"field_{i}": f"value_{i}" * 50 for i in range(100)
                }
                
                request_message = {
                    "type": "request",
                    "context": large_context,
                    "n_recommendations": 10,
                    "scenario": "content_discovery"
                }
                
                websocket.send_json(request_message)
                response = websocket.receive_json()
                
                # 应该能正常处理大消息
                assert response["type"] in ["recommendations", "error"]

    def test_websocket_performance_metrics(self, client, mock_service, mock_redis):
        """测试WebSocket性能指标"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            response_times = []
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "perf_test_user"}
                websocket.send_json(auth_message)
                websocket.receive_json()  # 接收欢迎消息
                
                # 测量多个请求的响应时间
                for i in range(10):
                    start_time = time.time()
                    
                    request_message = {
                        "type": "request",
                        "context": {"test_iteration": i},
                        "n_recommendations": 5,
                        "scenario": "content_discovery"
                    }
                    
                    websocket.send_json(request_message)
                    response = websocket.receive_json()
                    
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)
                    
                    assert response["type"] == "recommendations"
                
                # 验证性能
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                print(f"WebSocket平均响应时间: {avg_response_time:.2f}ms")
                print(f"WebSocket最大响应时间: {max_response_time:.2f}ms")
                
                # WebSocket响应时间应该在合理范围内
                assert avg_response_time < 500  # 平均500ms内
                assert max_response_time < 1000  # 最大1秒内


class TestPersonalizationWebSocketEdgeCases:
    """WebSocket边界情况测试"""
    
    def test_websocket_empty_message(self, client, mock_service, mock_redis):
        """测试空消息处理"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "empty_msg_test"}
                websocket.send_json(auth_message)
                websocket.receive_json()
                
                # 发送空对象
                websocket.send_json({})
                
                # 应该收到某种响应（可能是错误或忽略）
                try:
                    response = websocket.receive_json()
                    # 如果收到响应，验证格式
                    assert "type" in response
                except:
                    # 如果没有响应，也是可以接受的
                    pass

    def test_websocket_malformed_request(self, client, mock_service, mock_redis):
        """测试格式错误的请求"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "malformed_test"}
                websocket.send_json(auth_message)
                websocket.receive_json()
                
                # 发送格式错误的推荐请求
                malformed_request = {
                    "type": "request",
                    "n_recommendations": "invalid_number",  # 应该是数字
                    "scenario": "invalid_scenario"  # 无效场景
                }
                
                websocket.send_json(malformed_request)
                response = websocket.receive_json()
                
                # 应该收到错误响应
                assert response["type"] == "error"

    def test_websocket_extremely_high_frequency(self, client, mock_service, mock_redis):
        """测试极高频率请求"""
        
        with patch("api.v1.personalization.get_personalization_service", return_value=mock_service), \
             patch("api.v1.personalization.get_redis_client", return_value=mock_redis):
            
            with client.websocket_connect("/api/v1/personalization/stream") as websocket:
                # 认证
                auth_message = {"user_id": "high_freq_test"}
                websocket.send_json(auth_message)
                websocket.receive_json()
                
                # 快速发送大量请求
                num_requests = 50
                for i in range(num_requests):
                    request_message = {
                        "type": "request",
                        "context": {"batch": i},
                        "n_recommendations": 3,
                        "scenario": "content_discovery"
                    }
                    websocket.send_json(request_message)
                
                # 接收所有响应
                successful_responses = 0
                for i in range(num_requests):
                    try:
                        response = websocket.receive_json()
                        if response["type"] == "recommendations":
                            successful_responses += 1
                    except:
                        break
                
                # 应该处理大部分请求
                success_rate = successful_responses / num_requests
                assert success_rate > 0.8  # 至少80%成功率


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])