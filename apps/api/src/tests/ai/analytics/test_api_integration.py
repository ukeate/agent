"""
行为分析API集成测试
"""

import pytest
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from src.main import app
from src.ai.analytics.models import BehaviorEvent

class TestAnalyticsAPI:
    """分析API测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.client = TestClient(app)
        
        # 示例事件数据
        self.sample_events = [
            {
                "event_id": "test-event-1",
                "user_id": "user-123",
                "session_id": "session-456",
                "event_type": "page_view",
                "timestamp": utc_now().isoformat(),
                "properties": {"page": "/dashboard"},
                "context": {"user_agent": "Mozilla/5.0"}
            },
            {
                "event_id": "test-event-2", 
                "user_id": "user-123",
                "session_id": "session-456",
                "event_type": "click",
                "timestamp": (utc_now() + timedelta(minutes=1)).isoformat(),
                "properties": {"element": "button"},
                "context": {"user_agent": "Mozilla/5.0"}
            }
        ]
    
    def test_submit_events_success(self):
        """测试成功提交事件"""
        request_data = {
            "events": self.sample_events,
            "batch_id": "test-batch-1"
        }
        
        response = self.client.post(
            "/api/v1/analytics/events",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["event_count"] == 2
        assert data["batch_id"] == "test-batch-1"
    
    def test_submit_events_validation_error(self):
        """测试事件提交验证错误"""
        # 空事件列表
        response = self.client.post(
            "/api/v1/analytics/events",
            json={"events": []}
        )
        
        assert response.status_code == 400
        assert "事件列表不能为空" in response.json()["detail"]
    
    def test_submit_too_many_events(self):
        """测试提交过多事件"""
        # 创建超过限制的事件数量
        large_events = [self.sample_events[0]] * 1001
        
        response = self.client.post(
            "/api/v1/analytics/events",
            json={"events": large_events}
        )
        
        assert response.status_code == 400
        assert "单次提交事件数量不能超过1000" in response.json()["detail"]
    
    @patch('src.ai.analytics.storage.event_store.EventStore.get_events')
    def test_get_events_success(self, mock_get_events):
        """测试成功获取事件"""
        # 模拟返回数据
        mock_events = [
            BehaviorEvent(
                event_id="test-1",
                user_id="user-123",
                event_type="click",
                timestamp=utc_now()
            )
        ]
        mock_get_events.return_value = mock_events
        
        response = self.client.get("/api/v1/analytics/events?user_id=user-123")
        
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert data["total_count"] >= 0
    
    def test_get_events_with_filters(self):
        """测试带过滤条件获取事件"""
        params = {
            "user_id": "user-123",
            "event_type": "click",
            "limit": 50,
            "offset": 0
        }
        
        with patch('src.ai.analytics.storage.event_store.EventStore.get_events') as mock_get:
            mock_get.return_value = []
            
            response = self.client.get("/api/v1/analytics/events", params=params)
            
            assert response.status_code == 200
            data = response.json()
            assert data["limit"] == 50
            assert data["offset"] == 0
    
    @patch('src.ai.analytics.storage.event_store.EventStore.get_events')
    def test_analyze_behavior_success(self, mock_get_events):
        """测试成功执行行为分析"""
        # 模拟事件数据
        mock_get_events.return_value = [
            BehaviorEvent(
                event_id="test-1",
                user_id="user-123",
                event_type="click",
                timestamp=utc_now()
            )
        ]
        
        request_data = {
            "user_id": "user-123",
            "analysis_types": ["patterns", "anomalies"]
        }
        
        with patch('src.ai.analytics.behavior.pattern_recognition.PatternRecognitionEngine.analyze_patterns') as mock_patterns:
            with patch('src.ai.analytics.behavior.anomaly_detection.AnomalyDetectionEngine.detect_anomalies') as mock_anomalies:
                mock_patterns.return_value = {"patterns": []}
                mock_anomalies.return_value = {"anomalies": []}
                
                response = self.client.post(
                    "/api/v1/analytics/analyze",
                    json=request_data
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "results" in data
                assert "patterns" in data["results"]
                assert "anomalies" in data["results"]
    
    @patch('src.ai.analytics.storage.event_store.EventStore.get_events')
    def test_analyze_behavior_no_data(self, mock_get_events):
        """测试分析无数据情况"""
        mock_get_events.return_value = []
        
        request_data = {
            "user_id": "nonexistent-user",
            "analysis_types": ["patterns"]
        }
        
        response = self.client.post(
            "/api/v1/analytics/analyze",
            json=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_data"
    
    def test_generate_report_success(self):
        """测试成功生成报告"""
        request_data = {
            "report_type": "comprehensive",
            "format": "json",
            "include_visualizations": True
        }
        
        with patch('src.ai.analytics.storage.event_store.EventStore.get_events') as mock_get:
            mock_get.return_value = [
                BehaviorEvent(
                    event_id="test-1",
                    user_id="user-123",
                    event_type="click",
                    timestamp=utc_now()
                )
            ]
            
            response = self.client.post(
                "/api/v1/analytics/reports/generate",
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"
            assert "report_id" in data
    
    def test_generate_report_invalid_type(self):
        """测试生成报告时使用无效类型"""
        request_data = {
            "report_type": "invalid_type",
            "format": "json"
        }
        
        response = self.client.post(
            "/api/v1/analytics/reports/generate",
            json=request_data
        )
        
        assert response.status_code == 400
        assert "无效的报告类型" in response.json()["detail"]
    
    def test_get_dashboard_stats(self):
        """测试获取仪表板统计"""
        with patch('src.ai.analytics.storage.event_store.EventStore.get_event_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_events": 100,
                "unique_users": 10,
                "event_types": {"click": 50, "view": 50}
            }
            
            response = self.client.get("/api/v1/analytics/dashboard/stats?time_range=24h")
            
            assert response.status_code == 200
            data = response.json()
            assert "stats" in data
            assert data["time_range"] == "24h"
    
    def test_get_dashboard_stats_invalid_range(self):
        """测试无效时间范围"""
        response = self.client.get("/api/v1/analytics/dashboard/stats?time_range=invalid")
        
        assert response.status_code == 400
        assert "无效的时间范围" in response.json()["detail"]
    
    def test_export_events_json(self):
        """测试导出事件数据为JSON格式"""
        with patch('src.ai.analytics.storage.event_store.EventStore.get_events') as mock_get:
            mock_get.return_value = [
                BehaviorEvent(
                    event_id="test-1",
                    user_id="user-123", 
                    event_type="click",
                    timestamp=utc_now()
                )
            ]
            
            response = self.client.get("/api/v1/analytics/export/events?format=json")
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            assert "attachment" in response.headers.get("content-disposition", "")
    
    def test_export_events_csv(self):
        """测试导出事件数据为CSV格式"""
        with patch('src.ai.analytics.storage.event_store.EventStore.get_events') as mock_get:
            mock_get.return_value = [
                BehaviorEvent(
                    event_id="test-1",
                    user_id="user-123",
                    event_type="click", 
                    timestamp=utc_now()
                )
            ]
            
            response = self.client.get("/api/v1/analytics/export/events?format=csv")
            
            assert response.status_code == 200
            assert "text/csv" in response.headers["content-type"]
    
    def test_export_events_unsupported_format(self):
        """测试不支持的导出格式"""
        response = self.client.get("/api/v1/analytics/export/events?format=xml")
        
        assert response.status_code == 400
        assert "不支持的导出格式" in response.json()["detail"]
    
    def test_export_events_no_data(self):
        """测试导出无数据情况"""
        with patch('src.ai.analytics.storage.event_store.EventStore.get_events') as mock_get:
            mock_get.return_value = []
            
            response = self.client.get("/api/v1/analytics/export/events?format=json")
            
            assert response.status_code == 404
            assert "没有找到匹配的数据" in response.json()["detail"]
    
    def test_health_check(self):
        """测试健康检查"""
        response = self.client.get("/api/v1/analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_websocket_stats(self):
        """测试WebSocket统计"""
        response = self.client.get("/api/v1/analytics/ws/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "stats" in data
    
    def test_broadcast_message(self):
        """测试广播消息"""
        request_data = {
            "message_type": "test_message",
            "data": {"test": "data"},
            "user_id": "user-123"
        }
        
        response = self.client.post(
            "/api/v1/analytics/realtime/broadcast",
            params=request_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

class TestAnalyticsAPIValidation:
    """API验证测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.client = TestClient(app)
    
    def test_events_query_parameter_validation(self):
        """测试事件查询参数验证"""
        # 测试无效的limit
        response = self.client.get("/api/v1/analytics/events?limit=2000")
        assert response.status_code == 422  # Validation error
        
        # 测试负数offset
        response = self.client.get("/api/v1/analytics/events?offset=-1")
        assert response.status_code == 422
    
    def test_analysis_request_validation(self):
        """测试分析请求验证"""
        # 空的analysis_types
        request_data = {
            "analysis_types": []
        }
        
        response = self.client.post(
            "/api/v1/analytics/analyze",
            json=request_data
        )
        
        # 应该处理空的分析类型
        assert response.status_code in [200, 400]
    
    def test_datetime_parameter_parsing(self):
        """测试日期时间参数解析"""
        # 有效的datetime
        valid_datetime = utc_now().isoformat()
        response = self.client.get(f"/api/v1/analytics/events?start_time={valid_datetime}")
        
        # 应该能够解析有效的datetime
        assert response.status_code == 200
        
        # 无效的datetime格式
        response = self.client.get("/api/v1/analytics/events?start_time=invalid-date")
        assert response.status_code == 422
