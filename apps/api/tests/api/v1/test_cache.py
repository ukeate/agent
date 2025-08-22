"""
缓存API端点测试
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app


class TestCacheAPI:
    """缓存API测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @patch('src.ai.langgraph.cache_monitor.get_cache_monitor')
    def test_get_cache_stats(self, mock_get_monitor, client):
        """测试获取缓存统计"""
        # 模拟监控器
        mock_monitor = AsyncMock()
        mock_monitor.get_detailed_stats.return_value = {
            "hit_count": 100,
            "miss_count": 20,
            "hit_rate": 0.83,
            "cache_entries": 50
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/api/v1/cache/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["hit_count"] == 100
        assert data["hit_rate"] == 0.83
    
    @patch('src.ai.langgraph.cache_factory.get_node_cache')
    @patch('src.ai.langgraph.cache_monitor.CacheHealthChecker')
    def test_check_cache_health_healthy(self, mock_health_checker_class, mock_get_cache, client):
        """测试缓存健康检查 - 健康状态"""
        # 模拟健康检查器
        mock_health_checker = AsyncMock()
        mock_health_checker.health_check.return_value = {
            "status": "healthy",
            "checks": {
                "connection": {"status": "pass", "message": "连接正常"}
            }
        }
        mock_health_checker_class.return_value = mock_health_checker
        
        response = client.get("/api/v1/cache/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
    
    @patch('src.ai.langgraph.cache_factory.get_node_cache')
    def test_clear_cache(self, mock_get_cache, client):
        """测试清理缓存"""
        # 模拟缓存实例
        mock_cache = AsyncMock()
        mock_cache.clear.return_value = 15  # 清理了15个条目
        mock_get_cache.return_value = mock_cache
        
        response = client.delete("/api/v1/cache/clear?pattern=test:*")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["cleared_count"] == 15
        assert data["pattern"] == "test:*"
    
    @patch('src.ai.langgraph.cache_factory.get_node_cache')
    def test_invalidate_node_cache(self, mock_get_cache, client):
        """测试失效节点缓存"""
        # 模拟缓存实例
        mock_cache = AsyncMock()
        mock_cache.clear.return_value = 3  # 清理了3个相关条目
        mock_get_cache.return_value = mock_cache
        
        response = client.delete("/api/v1/cache/invalidate/test_node")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["node_name"] == "test_node"
    
    @patch('src.ai.langgraph.cache_monitor.get_cache_monitor')
    def test_get_cache_summary(self, mock_get_monitor, client):
        """测试获取缓存摘要"""
        # 模拟监控器
        mock_monitor = AsyncMock()
        mock_monitor.get_summary.return_value = {
            "hit_rate": "85.5%",
            "total_operations": 1000,
            "avg_latency_ms": "12.5",
            "current_entries": 250
        }
        mock_get_monitor.return_value = mock_monitor
        
        response = client.get("/api/v1/cache/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert data["hit_rate"] == "85.5%"
        assert data["total_operations"] == 1000
    
    @patch('src.ai.langgraph.cache_factory.get_node_cache')
    def test_get_cache_config(self, mock_get_cache, client):
        """测试获取缓存配置"""
        # 模拟缓存实例
        mock_cache = AsyncMock()
        mock_cache.config.enabled = True
        mock_cache.config.backend = "redis"
        mock_cache.config.ttl_default = 3600
        mock_cache.__class__.__name__ = "RedisNodeCache"
        
        # 设置config的__dict__属性
        mock_cache.config.__dict__ = {
            "enabled": True,
            "backend": "redis", 
            "ttl_default": 3600,
            "max_entries": 10000
        }
        
        mock_get_cache.return_value = mock_cache
        
        response = client.get("/api/v1/cache/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["backend"] == "RedisNodeCache"
        assert data["config"]["enabled"] is True
        assert data["config"]["ttl_default"] == 3600