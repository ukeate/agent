"""监控系统测试"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from ai.platform_integration.monitoring import MonitoringSystem, PROMETHEUS_AVAILABLE

@pytest.fixture
def monitoring_config():
    """监控配置"""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0
    }

@pytest.fixture
def monitoring_system(monitoring_config):
    """监控系统实例"""
    with patch('redis.Redis'):
        system = MonitoringSystem(monitoring_config)
        return system

class TestMonitoringSystem:
    """监控系统测试类"""

    def test_init_with_prometheus(self, monitoring_config):
        """测试初始化（有Prometheus）"""
        with patch('redis.Redis') as mock_redis, \
             patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True), \
             patch('ai.platform_integration.monitoring.CollectorRegistry') as mock_registry:
            
            mock_registry.return_value = MagicMock()
            
            system = MonitoringSystem(monitoring_config)
            
            assert system.config == monitoring_config
            assert system.metrics == {}
            assert system.alert_rules == []
            mock_redis.assert_called_once()

    def test_init_without_prometheus(self, monitoring_config):
        """测试初始化（无Prometheus）"""
        with patch('redis.Redis'), \
             patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', False):
            
            system = MonitoringSystem(monitoring_config)
            
            assert system.registry is None
            assert system.metrics == {}

    def test_initialize_metrics(self, monitoring_config):
        """测试初始化指标"""
        with patch('redis.Redis'), \
             patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True):
            
            # Mock prometheus_client components
            with patch.dict('ai.platform_integration.monitoring.__dict__', {
                'Counter': MagicMock(),
                'Histogram': MagicMock(),
                'Gauge': MagicMock(),
                'CollectorRegistry': MagicMock(),
                'generate_latest': MagicMock()
            }):
                system = MonitoringSystem(monitoring_config)
                
                # 验证系统初始化成功
                assert system.config == monitoring_config
                assert hasattr(system, 'metrics')

    @pytest.mark.asyncio
    async def test_setup_monitoring(self, monitoring_system):
        """测试设置监控系统"""
        with patch.object(monitoring_system, '_setup_prometheus_metrics') as mock_prometheus, \
             patch.object(monitoring_system, '_setup_grafana_dashboards') as mock_grafana, \
             patch.object(monitoring_system, '_setup_alerting_rules') as mock_alerting, \
             patch.object(monitoring_system, '_setup_health_checks') as mock_health:
            
            mock_prometheus.return_value = {"status": "configured"}
            mock_grafana.return_value = {"status": "configured"}
            mock_alerting.return_value = {"status": "configured"}
            mock_health.return_value = {"status": "configured"}
            
            result = await monitoring_system.setup_monitoring()
            
            assert result["status"] == "monitoring_active"
            assert "prometheus" in result
            assert "grafana" in result
            assert "alerting" in result
            assert "health_checks" in result

    @pytest.mark.asyncio
    async def test_setup_prometheus_metrics_available(self, monitoring_system):
        """测试设置Prometheus指标（可用）"""
        with patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True):
            monitoring_system.metrics = {"test_metric": Mock()}
            
            result = await monitoring_system._setup_prometheus_metrics()
            
            assert result["status"] == "configured"
            assert "metrics" in result
            assert result["endpoint"] == "/metrics"

    @pytest.mark.asyncio
    async def test_setup_prometheus_metrics_unavailable(self, monitoring_system):
        """测试设置Prometheus指标（不可用）"""
        with patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', False):
            result = await monitoring_system._setup_prometheus_metrics()
            
            assert result["status"] == "unavailable"
            assert "Prometheus client not installed" in result["message"]

    @pytest.mark.asyncio
    async def test_setup_grafana_dashboards(self, monitoring_system):
        """测试设置Grafana仪表板"""
        result = await monitoring_system._setup_grafana_dashboards()
        
        assert result["status"] == "configured"
        assert "dashboards" in result
        assert "total_panels" in result
        assert "config" in result
        
        # 验证包含预期的仪表板
        expected_dashboards = ["platform_overview", "training_monitoring", "component_health"]
        assert all(dashboard in result["dashboards"] for dashboard in expected_dashboards)

    @pytest.mark.asyncio
    async def test_setup_alerting_rules(self, monitoring_system):
        """测试设置告警规则"""
        result = await monitoring_system._setup_alerting_rules()
        
        assert result["status"] == "configured"
        assert "total_rules" in result
        assert "alert_rules" in result
        assert result["total_rules"] > 0
        
        # 验证告警规则被设置到实例中
        assert len(monitoring_system.alert_rules) > 0
        
        # 验证包含预期的告警规则
        rule_names = [rule["name"] for rule in monitoring_system.alert_rules]
        expected_rules = ["HighErrorRate", "HighResponseTime", "ComponentUnhealthy"]
        assert all(rule in rule_names for rule in expected_rules)

    @pytest.mark.asyncio
    async def test_setup_health_checks(self, monitoring_system):
        """测试设置健康检查"""
        result = await monitoring_system._setup_health_checks()
        
        assert result["status"] == "configured"
        assert "config" in result
        
        config = result["config"]
        assert "endpoints" in config
        assert "database" in config
        assert "redis" in config
        assert "external_services" in config
        
        # 验证健康检查端点配置
        endpoints = config["endpoints"]
        assert any(ep["path"] == "/health" for ep in endpoints)
        assert any(ep["path"] == "/ready" for ep in endpoints)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_record_request(self, monitoring_system):
        """测试记录请求指标"""
        # Mock metrics
        mock_counter = Mock()
        mock_histogram = Mock()
        monitoring_system.metrics = {
            'request_counter': mock_counter,
            'request_duration': mock_histogram
        }
        
        monitoring_system.record_request("GET", "/api/test", 200, 1.5)
        
        mock_counter.labels.assert_called_with(method="GET", endpoint="/api/test", status="200")
        mock_counter.labels.return_value.inc.assert_called_once()
        
        mock_histogram.labels.assert_called_with(method="GET", endpoint="/api/test")
        mock_histogram.labels.return_value.observe.assert_called_with(1.5)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', False)
    def test_record_request_no_prometheus(self, monitoring_system):
        """测试记录请求指标（无Prometheus）"""
        # 应该不抛出异常
        monitoring_system.record_request("GET", "/api/test", 200, 1.5)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_memory_usage(self, monitoring_system):
        """测试更新内存使用量"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'memory_usage': mock_gauge}
        
        monitoring_system.update_memory_usage("api", 1024.0)
        
        mock_gauge.labels.assert_called_with(component="api")
        mock_gauge.labels.return_value.set.assert_called_with(1024.0)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_cpu_usage(self, monitoring_system):
        """测试更新CPU使用率"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'cpu_usage': mock_gauge}
        
        monitoring_system.update_cpu_usage("worker", 75.5)
        
        mock_gauge.labels.assert_called_with(component="worker")
        mock_gauge.labels.return_value.set.assert_called_with(75.5)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_active_jobs(self, monitoring_system):
        """测试更新活跃任务数"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'active_training_jobs': mock_gauge}
        
        monitoring_system.update_active_jobs(5)
        
        mock_gauge.set.assert_called_with(5)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_evaluation_score(self, monitoring_system):
        """测试更新模型评估分数"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'model_evaluation_score': mock_gauge}
        
        monitoring_system.update_evaluation_score("model_123", "accuracy", 0.95)
        
        mock_gauge.labels.assert_called_with(model_id="model_123", metric_type="accuracy")
        mock_gauge.labels.return_value.set.assert_called_with(0.95)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_workflow_success_rate(self, monitoring_system):
        """测试更新工作流成功率"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'workflow_success_rate': mock_gauge}
        
        monitoring_system.update_workflow_success_rate("fine_tuning", 0.92)
        
        mock_gauge.labels.assert_called_with(workflow_type="fine_tuning")
        mock_gauge.labels.return_value.set.assert_called_with(0.92)

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_update_component_health(self, monitoring_system):
        """测试更新组件健康状态"""
        mock_gauge = Mock()
        monitoring_system.metrics = {'component_health': mock_gauge}
        
        monitoring_system.update_component_health("comp_123", "fine_tuning", True)
        
        mock_gauge.labels.assert_called_with(component_id="comp_123", component_type="fine_tuning")
        mock_gauge.labels.return_value.set.assert_called_with(1)
        
        # 测试不健康状态
        monitoring_system.update_component_health("comp_456", "evaluation", False)
        mock_gauge.labels.return_value.set.assert_called_with(0)

    def test_get_metrics(self, monitoring_system):
        """测试获取指标"""
        with patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True):
            with patch.dict('ai.platform_integration.monitoring.__dict__', {
                'generate_latest': MagicMock(return_value=b'metrics_data')
            }):
                monitoring_system.registry = Mock()
                
                metrics_data = monitoring_system.get_metrics()
                
                assert metrics_data == b'metrics_data'

    @patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', False)
    def test_get_metrics_no_prometheus(self, monitoring_system):
        """测试获取指标（无Prometheus）"""
        metrics_data = monitoring_system.get_metrics()
        assert metrics_data == b""

    @pytest.mark.asyncio
    async def test_generate_monitoring_report(self, monitoring_system):
        """测试生成监控报告"""
        with patch.object(monitoring_system, '_get_recent_metrics') as mock_recent, \
             patch.object(monitoring_system, '_calculate_statistics') as mock_stats, \
             patch.object(monitoring_system, '_check_alert_status') as mock_alerts, \
             patch.object(monitoring_system, '_generate_recommendations') as mock_recommendations:
            
            mock_recent.return_value = {"metric1": {"value": 100}}
            mock_stats.return_value = {
                "request_rate": 50.0,
                "error_rate": 0.02,
                "avg_response_time": 1.2
            }
            mock_alerts.return_value = {"active_alerts": [], "status": "healthy"}
            mock_recommendations.return_value = ["System is operating normally"]
            
            report = await monitoring_system.generate_monitoring_report()
            
            assert "report_generated_at" in report
            assert "metrics_summary" in report
            assert "alert_status" in report
            assert "recommendations" in report
            assert "health_score" in report

    @pytest.mark.asyncio
    async def test_get_recent_metrics(self, monitoring_system):
        """测试获取最近指标"""
        mock_keys = ["metrics:cpu", "metrics:memory", "metrics:disk"]
        mock_values = [
            '{"value": 75.0, "timestamp": "2023-01-01T10:00:00Z"}',
            '{"value": 60.0, "timestamp": "2023-01-01T10:00:00Z"}',
            '{"value": 85.0, "timestamp": "2023-01-01T10:00:00Z"}'
        ]
        
        with patch.object(monitoring_system.redis_client, 'keys', return_value=mock_keys), \
             patch.object(monitoring_system.redis_client, 'get', side_effect=mock_values):
            
            metrics = await monitoring_system._get_recent_metrics()
            
            assert len(metrics) == 3
            assert metrics["metrics:cpu"]["value"] == 75.0

    @pytest.mark.asyncio
    async def test_get_recent_metrics_invalid_json(self, monitoring_system):
        """测试获取最近指标（无效JSON）"""
        mock_keys = ["metrics:invalid"]
        mock_values = ["invalid json"]
        
        with patch.object(monitoring_system.redis_client, 'keys', return_value=mock_keys), \
             patch.object(monitoring_system.redis_client, 'get', side_effect=mock_values):
            
            metrics = await monitoring_system._get_recent_metrics()
            
            # 无效JSON应该被跳过
            assert len(metrics) == 0

    @pytest.mark.asyncio
    async def test_calculate_statistics(self, monitoring_system):
        """测试计算统计信息"""
        mock_metrics = {
            "metrics:requests": {"count": 1000},
            "metrics:errors": {"count": 20},
            "metrics:response_time": {"avg": 1.5}
        }
        
        with patch.object(monitoring_system, '_calculate_request_rate', return_value=100.0), \
             patch.object(monitoring_system, '_calculate_error_rate', return_value=0.02), \
             patch.object(monitoring_system, '_calculate_avg_response_time', return_value=1.2), \
             patch.object(monitoring_system, '_calculate_avg_memory', return_value=4.5), \
             patch.object(monitoring_system, '_calculate_avg_cpu', return_value=45.0), \
             patch.object(monitoring_system, '_calculate_workflow_success', return_value=0.92):
            
            stats = await monitoring_system._calculate_statistics(mock_metrics)
            
            assert stats["request_rate"] == 100.0
            assert stats["error_rate"] == 0.02
            assert stats["avg_response_time"] == 1.2
            assert stats["memory_usage_avg"] == 4.5
            assert stats["cpu_usage_avg"] == 45.0
            assert stats["workflow_success_rate"] == 0.92

    @pytest.mark.asyncio
    async def test_check_alert_status(self, monitoring_system):
        """测试检查告警状态"""
        result = await monitoring_system._check_alert_status()
        
        assert "active_alerts" in result
        assert "total_alerts" in result
        assert "status" in result
        assert result["status"] == "healthy"  # 简化实现返回healthy

    @pytest.mark.asyncio
    async def test_generate_recommendations_normal(self, monitoring_system):
        """测试生成建议（正常状态）"""
        stats = {
            "error_rate": 0.01,
            "cpu_usage_avg": 50.0,
            "memory_usage_avg": 4.0,
            "workflow_success_rate": 0.95
        }
        alerts = {"active_alerts": []}
        
        recommendations = await monitoring_system._generate_recommendations(stats, alerts)
        
        assert "System is operating within normal parameters" in recommendations

    @pytest.mark.asyncio
    async def test_generate_recommendations_issues(self, monitoring_system):
        """测试生成建议（有问题）"""
        stats = {
            "error_rate": 0.08,  # High error rate
            "cpu_usage_avg": 85.0,  # High CPU
            "memory_usage_avg": 8.5,  # High memory
            "workflow_success_rate": 0.75  # Low success rate
        }
        alerts = {"active_alerts": []}
        
        recommendations = await monitoring_system._generate_recommendations(stats, alerts)
        
        assert len(recommendations) == 4  # Should have recommendations for all issues
        assert any("error rate" in rec for rec in recommendations)
        assert any("scaling up" in rec for rec in recommendations)
        assert any("Memory usage" in rec for rec in recommendations)
        assert any("Workflow success" in rec for rec in recommendations)

    def test_calculate_health_score_perfect(self, monitoring_system):
        """测试计算完美健康评分"""
        stats = {
            "error_rate": 0.01,
            "avg_response_time": 0.5
        }
        alerts = {"active_alerts": []}
        
        score = monitoring_system._calculate_health_score(stats, alerts)
        assert score == 100.0

    def test_calculate_health_score_with_issues(self, monitoring_system):
        """测试计算有问题的健康评分"""
        stats = {
            "error_rate": 0.15,  # Very high error rate
            "avg_response_time": 4.0  # Very high response time
        }
        alerts = {"active_alerts": ["alert1", "alert2", "alert3"]}  # 3 active alerts
        
        score = monitoring_system._calculate_health_score(stats, alerts)
        
        # Should deduct: 30 (high error rate) + 20 (high response time) + 30 (3 alerts * 10)
        expected_score = 100 - 30 - 20 - 30
        assert score == expected_score

    @pytest.mark.asyncio
    async def test_export_metrics_history(self, monitoring_system):
        """测试导出指标历史"""
        result = await monitoring_system.export_metrics_history(hours=12)
        
        assert "start_time" in result
        assert "end_time" in result
        assert "duration_hours" in result
        assert "data_points" in result
        assert "metrics" in result
        assert result["duration_hours"] == 12

        # 验证时间范围
        start_time = datetime.fromisoformat(result["start_time"])
        end_time = datetime.fromisoformat(result["end_time"])
        duration = end_time - start_time
        assert abs(duration.total_seconds() - 12 * 3600) < 60  # 允许1分钟误差

class TestMonitoringSystemIntegration:
    """监控系统集成测试"""

    @pytest.mark.asyncio
    async def test_full_monitoring_setup(self, monitoring_config):
        """测试完整监控设置"""
        with patch('redis.Redis'), \
             patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True):
            
            # Mock prometheus_client components
            with patch.dict('ai.platform_integration.monitoring.__dict__', {
                'Counter': MagicMock(),
                'Histogram': MagicMock(), 
                'Gauge': MagicMock(),
                'CollectorRegistry': MagicMock(),
                'generate_latest': MagicMock()
            }):
                system = MonitoringSystem(monitoring_config)
                
                # 设置完整监控
                setup_result = await system.setup_monitoring()
                assert setup_result["status"] == "monitoring_active"
                
                # 记录一些指标
                system.record_request("GET", "/health", 200, 0.1)
                system.update_memory_usage("api", 512.0)
                system.update_cpu_usage("api", 25.0)
                system.update_active_jobs(3)
                
                # 生成报告
                report = await system.generate_monitoring_report()
                assert "health_score" in report
                assert 0 <= report["health_score"] <= 100

    @pytest.mark.asyncio
    async def test_metrics_collection_workflow(self, monitoring_config):
        """测试指标收集工作流"""
        with patch('redis.Redis') as mock_redis, \
             patch('ai.platform_integration.monitoring.PROMETHEUS_AVAILABLE', True):
            
            # Mock prometheus_client components
            with patch.dict('ai.platform_integration.monitoring.__dict__', {
                'Counter': MagicMock(),
                'Histogram': MagicMock(),
                'Gauge': MagicMock(),
                'CollectorRegistry': MagicMock(),
                'generate_latest': MagicMock()
            }):
                system = MonitoringSystem(monitoring_config)
                
                # 模拟指标数据收集
                mock_redis_instance = mock_redis.return_value
                mock_redis_instance.keys.return_value = ["metrics:test"]
                mock_redis_instance.get.return_value = '{"value": 100, "timestamp": "2023-01-01T10:00:00Z"}'
                
                # 获取最近指标
                recent_metrics = await system._get_recent_metrics()
                assert "metrics:test" in recent_metrics
                
                # 计算统计信息
                stats = await system._calculate_statistics(recent_metrics)
                assert "request_rate" in stats
                
                # 检查告警状态
                alert_status = await system._check_alert_status()
            assert alert_status["status"] == "healthy"
            
            # 生成建议
            recommendations = await system._generate_recommendations(stats, alert_status)
            assert isinstance(recommendations, list)
