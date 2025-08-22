"""
监控仪表板测试
测试指标收集、告警规则、仪表板服务等功能
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch

from src.ai.autogen.monitoring_dashboard import (
    MetricType, AlertLevel, MetricPoint, MetricSeries, Alert, AlertRule,
    MetricCollector, DashboardServer, get_metric_collector
)


class TestMetricPoint:
    """指标数据点测试"""
    
    def test_metric_point_creation(self):
        """测试指标数据点创建"""
        timestamp = datetime.now(timezone.utc)
        point = MetricPoint(
            timestamp=timestamp,
            value=42.5,
            labels={"host": "server1", "service": "api"}
        )
        
        assert point.timestamp == timestamp
        assert point.value == 42.5
        assert point.labels["host"] == "server1"
        assert point.labels["service"] == "api"
    
    def test_metric_point_to_dict(self):
        """测试指标数据点转换为字典"""
        timestamp = datetime.now(timezone.utc)
        point = MetricPoint(
            timestamp=timestamp,
            value=100,
            labels={"environment": "prod"}
        )
        
        point_dict = point.to_dict()
        
        assert point_dict["timestamp"] == timestamp.isoformat()
        assert point_dict["value"] == 100
        assert point_dict["labels"]["environment"] == "prod"


class TestMetricSeries:
    """指标序列测试"""
    
    @pytest.fixture
    def metric_series(self):
        """创建指标序列"""
        return MetricSeries(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="测试指标",
            unit="count"
        )
    
    def test_metric_series_creation(self, metric_series):
        """测试指标序列创建"""
        assert metric_series.name == "test_metric"
        assert metric_series.metric_type == MetricType.GAUGE
        assert metric_series.description == "测试指标"
        assert metric_series.unit == "count"
        assert len(metric_series.points) == 0
    
    def test_add_point(self, metric_series):
        """测试添加数据点"""
        metric_series.add_point(42, {"label": "value"})
        
        assert len(metric_series.points) == 1
        point = metric_series.points[0]
        assert point.value == 42
        assert point.labels["label"] == "value"
        assert isinstance(point.timestamp, datetime)
    
    def test_get_latest_value(self, metric_series):
        """测试获取最新值"""
        # 空序列
        assert metric_series.get_latest_value() is None
        
        # 添加数据点
        metric_series.add_point(10)
        assert metric_series.get_latest_value() == 10
        
        metric_series.add_point(20)
        assert metric_series.get_latest_value() == 20
    
    def test_get_values_in_range(self, metric_series):
        """测试获取范围内的值"""
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(minutes=10)
        end_time = now + timedelta(minutes=10)
        
        # 添加一些数据点
        metric_series.add_point(10)
        time.sleep(0.001)  # 确保时间戳不同
        metric_series.add_point(20)
        
        points_in_range = metric_series.get_values_in_range(start_time, end_time)
        assert len(points_in_range) == 2
        assert points_in_range[0].value == 10
        assert points_in_range[1].value == 20
    
    def test_get_statistics(self, metric_series):
        """测试获取统计信息"""
        # 空序列
        stats = metric_series.get_statistics()
        assert stats == {}
        
        # 添加数据点
        values = [10, 20, 30, 40, 50]
        for value in values:
            metric_series.add_point(value)
            time.sleep(0.001)
        
        stats = metric_series.get_statistics(60)  # 最近60分钟
        
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["avg"] == 30
        assert stats["median"] == 30
        assert stats["latest"] == 50


class TestAlert:
    """告警测试"""
    
    @pytest.fixture
    def sample_alert(self):
        """创建样例告警"""
        return Alert(
            id="alert_001",
            metric_name="cpu_usage",
            level=AlertLevel.WARNING,
            message="CPU使用率过高",
            timestamp=datetime.now(timezone.utc),
            threshold=80.0,
            actual_value=85.5,
            labels={"host": "server1"}
        )
    
    def test_alert_creation(self, sample_alert):
        """测试告警创建"""
        assert sample_alert.id == "alert_001"
        assert sample_alert.metric_name == "cpu_usage"
        assert sample_alert.level == AlertLevel.WARNING
        assert sample_alert.message == "CPU使用率过高"
        assert sample_alert.threshold == 80.0
        assert sample_alert.actual_value == 85.5
        assert sample_alert.labels["host"] == "server1"
        assert sample_alert.resolved is False
        assert sample_alert.resolved_at is None
    
    def test_alert_to_dict(self, sample_alert):
        """测试告警转换为字典"""
        alert_dict = sample_alert.to_dict()
        
        assert alert_dict["id"] == "alert_001"
        assert alert_dict["metric_name"] == "cpu_usage"
        assert alert_dict["level"] == AlertLevel.WARNING.value
        assert alert_dict["message"] == "CPU使用率过高"
        assert alert_dict["threshold"] == 80.0
        assert alert_dict["actual_value"] == 85.5
        assert alert_dict["labels"]["host"] == "server1"
        assert alert_dict["resolved"] is False
        assert alert_dict["resolved_at"] is None


class TestAlertRule:
    """告警规则测试"""
    
    @pytest.fixture
    def alert_rule(self):
        """创建告警规则"""
        return AlertRule(
            name="high_cpu",
            metric_name="cpu_usage",
            condition=">",
            threshold=80.0,
            level=AlertLevel.WARNING,
            message_template="CPU使用率过高: {actual_value:.1f}% > {threshold}%",
            duration_minutes=5
        )
    
    @pytest.fixture
    def metric_series_high_cpu(self):
        """创建高CPU使用率的指标序列"""
        series = MetricSeries(
            name="cpu_usage",
            metric_type=MetricType.GAUGE,
            description="CPU使用率",
            unit="%"
        )
        
        # 添加持续高CPU使用率的数据点
        for _ in range(10):
            series.add_point(85.0)  # 高于阈值80
            time.sleep(0.001)
        
        return series
    
    @pytest.fixture
    def metric_series_normal_cpu(self):
        """创建正常CPU使用率的指标序列"""
        series = MetricSeries(
            name="cpu_usage",
            metric_type=MetricType.GAUGE,
            description="CPU使用率",
            unit="%"
        )
        
        series.add_point(50.0)  # 低于阈值80
        return series
    
    def test_alert_rule_creation(self, alert_rule):
        """测试告警规则创建"""
        assert alert_rule.name == "high_cpu"
        assert alert_rule.metric_name == "cpu_usage"
        assert alert_rule.condition == ">"
        assert alert_rule.threshold == 80.0
        assert alert_rule.level == AlertLevel.WARNING
        assert alert_rule.duration_minutes == 5
        assert alert_rule.enabled is True
    
    def test_check_condition_greater_than(self, alert_rule):
        """测试大于条件检查"""
        assert alert_rule._check_condition(85.0) is True
        assert alert_rule._check_condition(75.0) is False
        assert alert_rule._check_condition(80.0) is False
    
    def test_evaluate_alert_triggered(self, alert_rule, metric_series_high_cpu):
        """测试告警触发"""
        # 模拟持续时间内的高CPU使用率
        alert = alert_rule.evaluate(metric_series_high_cpu)
        
        assert alert is not None
        assert alert.metric_name == "cpu_usage"
        assert alert.level == AlertLevel.WARNING
        assert alert.threshold == 80.0
        assert alert.actual_value == 85.0
        assert "CPU使用率过高" in alert.message
    
    def test_evaluate_no_alert(self, alert_rule, metric_series_normal_cpu):
        """测试无告警情况"""
        alert = alert_rule.evaluate(metric_series_normal_cpu)
        assert alert is None
    
    def test_evaluate_disabled_rule(self, alert_rule, metric_series_high_cpu):
        """测试禁用规则"""
        alert_rule.enabled = False
        alert = alert_rule.evaluate(metric_series_high_cpu)
        assert alert is None
    
    def test_different_conditions(self):
        """测试不同的条件类型"""
        rule_lt = AlertRule("test", "metric", "<", 50.0, AlertLevel.INFO, "message")
        assert rule_lt._check_condition(40.0) is True
        assert rule_lt._check_condition(60.0) is False
        
        rule_eq = AlertRule("test", "metric", "==", 50.0, AlertLevel.INFO, "message")
        assert rule_eq._check_condition(50.0) is True
        assert rule_eq._check_condition(49.9999) is False
        
        rule_ne = AlertRule("test", "metric", "!=", 50.0, AlertLevel.INFO, "message")
        assert rule_ne._check_condition(51.0) is True
        assert rule_ne._check_condition(50.0) is False


class TestMetricCollector:
    """指标收集器测试"""
    
    @pytest.fixture
    def metric_collector(self):
        """创建指标收集器"""
        return MetricCollector()
    
    def test_metric_collector_initialization(self, metric_collector):
        """测试指标收集器初始化"""
        assert isinstance(metric_collector.metrics, dict)
        assert isinstance(metric_collector.alert_rules, list)
        assert isinstance(metric_collector.active_alerts, dict)
        
        # 应该有内置指标
        assert len(metric_collector.metrics) > 0
        assert "system_cpu_usage" in metric_collector.metrics
        assert "agent_pool_size" in metric_collector.metrics
        
        # 应该有默认告警规则
        assert len(metric_collector.alert_rules) > 0
    
    def test_register_metric(self, metric_collector):
        """测试注册指标"""
        metric_series = metric_collector.register_metric(
            "test_metric",
            MetricType.COUNTER,
            "测试计数器",
            "count"
        )
        
        assert metric_series.name == "test_metric"
        assert metric_series.metric_type == MetricType.COUNTER
        assert "test_metric" in metric_collector.metrics
    
    def test_record_metric(self, metric_collector):
        """测试记录指标"""
        # 先注册指标
        metric_collector.register_metric("test_gauge", MetricType.GAUGE, "测试仪表")
        
        # 记录指标值
        metric_collector.record_metric("test_gauge", 42.5, {"label": "test"})
        
        metric_series = metric_collector.get_metric("test_gauge")
        assert metric_series.get_latest_value() == 42.5
        assert len(metric_series.points) == 1
        assert metric_series.points[0].labels["label"] == "test"
    
    def test_record_unknown_metric(self, metric_collector):
        """测试记录未知指标"""
        # 记录未注册的指标应该不会崩溃
        metric_collector.record_metric("unknown_metric", 123)
        
        # 指标不应该被创建
        assert "unknown_metric" not in metric_collector.metrics
    
    def test_get_metric(self, metric_collector):
        """测试获取指标"""
        # 获取存在的指标
        cpu_metric = metric_collector.get_metric("system_cpu_usage")
        assert cpu_metric is not None
        assert cpu_metric.name == "system_cpu_usage"
        
        # 获取不存在的指标
        unknown_metric = metric_collector.get_metric("unknown")
        assert unknown_metric is None
    
    def test_add_alert_rule(self, metric_collector):
        """测试添加告警规则"""
        initial_rule_count = len(metric_collector.alert_rules)
        
        new_rule = AlertRule(
            name="test_rule",
            metric_name="test_metric",
            condition=">",
            threshold=100.0,
            level=AlertLevel.CRITICAL,
            message_template="测试告警"
        )
        
        metric_collector.add_alert_rule(new_rule)
        assert len(metric_collector.alert_rules) == initial_rule_count + 1
    
    def test_evaluate_alerts(self, metric_collector):
        """测试评估告警"""
        # 注册测试指标
        metric_collector.register_metric("test_alert_metric", MetricType.GAUGE, "测试指标")
        
        # 添加测试告警规则
        rule = AlertRule(
            name="test_alert",
            metric_name="test_alert_metric",
            condition=">",
            threshold=50.0,
            level=AlertLevel.WARNING,
            message_template="测试告警触发: {actual_value}",
            duration_minutes=0  # 立即触发
        )
        metric_collector.add_alert_rule(rule)
        
        # 记录触发告警的指标值
        metric_collector.record_metric("test_alert_metric", 75.0)
        
        # 评估告警
        initial_alert_count = len(metric_collector.active_alerts)
        metric_collector.evaluate_alerts()
        
        # 应该有新告警
        assert len(metric_collector.active_alerts) == initial_alert_count + 1
        assert len(metric_collector.alert_history) > 0
    
    def test_resolve_alert(self, metric_collector):
        """测试解决告警"""
        # 先创建一个告警
        alert = Alert(
            id="test_resolve_alert",
            metric_name="test_metric",
            level=AlertLevel.INFO,
            message="测试解决",
            timestamp=datetime.now(timezone.utc),
            threshold=0.0,
            actual_value=1.0
        )
        
        metric_collector.active_alerts["test_key"] = alert
        
        # 解决告警
        metric_collector.resolve_alert("test_resolve_alert")
        
        # 告警应该被解决
        assert "test_key" not in metric_collector.active_alerts
        assert alert.resolved is True
        assert alert.resolved_at is not None
    
    def test_alert_callback(self, metric_collector):
        """测试告警回调"""
        callback_called = False
        received_alert = None
        
        def test_callback(alert):
            nonlocal callback_called, received_alert
            callback_called = True
            received_alert = alert
        
        metric_collector.add_alert_callback(test_callback)
        
        # 触发告警
        metric_collector.register_metric("callback_test_metric", MetricType.GAUGE, "回调测试")
        rule = AlertRule(
            name="callback_test",
            metric_name="callback_test_metric",
            condition=">",
            threshold=10.0,
            level=AlertLevel.INFO,
            message_template="回调测试",
            duration_minutes=0
        )
        metric_collector.add_alert_rule(rule)
        
        metric_collector.record_metric("callback_test_metric", 20.0)
        metric_collector.evaluate_alerts()
        
        # 回调应该被调用
        assert callback_called is True
        assert received_alert is not None
        assert received_alert.metric_name == "callback_test_metric"


class TestDashboardServer:
    """仪表板服务器测试"""
    
    @pytest.fixture
    def dashboard_server(self):
        """创建仪表板服务器"""
        collector = MetricCollector()
        return DashboardServer(collector, port=8081)
    
    @pytest.mark.asyncio
    async def test_dashboard_server_startup_shutdown(self, dashboard_server):
        """测试仪表板服务器启动关闭"""
        await dashboard_server.start()
        assert dashboard_server.running is True
        
        await dashboard_server.stop()
        assert dashboard_server.running is False
    
    def test_generate_dashboard_data(self, dashboard_server):
        """测试生成仪表板数据"""
        # 添加一些测试数据
        dashboard_server.metric_collector.record_metric("system_cpu_usage", 75.0)
        dashboard_server.metric_collector.record_metric("agent_pool_size", 5)
        
        dashboard_data = dashboard_server._generate_dashboard_data()
        
        assert "timestamp" in dashboard_data
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "summary" in dashboard_data
        
        # 检查指标数据
        assert "system_cpu_usage" in dashboard_data["metrics"]
        cpu_metric = dashboard_data["metrics"]["system_cpu_usage"]
        assert cpu_metric["current_value"] == 75.0
        assert cpu_metric["unit"] == "%"
        
        # 检查摘要
        summary = dashboard_data["summary"]
        assert "system_status" in summary
        assert "total_metrics" in summary
        assert "key_metrics" in summary
    
    def test_get_metric_data(self, dashboard_server):
        """测试获取特定指标数据"""
        # 添加测试数据
        collector = dashboard_server.metric_collector
        collector.record_metric("system_memory_usage", 60.0)
        collector.record_metric("system_memory_usage", 65.0)
        
        metric_data = dashboard_server.get_metric_data("system_memory_usage", 60)
        
        assert metric_data is not None
        assert metric_data["name"] == "system_memory_usage"
        assert metric_data["unit"] == "%"
        assert len(metric_data["points"]) == 2
        assert "statistics" in metric_data
    
    def test_get_metric_data_unknown(self, dashboard_server):
        """测试获取未知指标数据"""
        metric_data = dashboard_server.get_metric_data("unknown_metric")
        assert metric_data is None
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, dashboard_server):
        """测试收集系统指标"""
        await dashboard_server._collect_system_metrics()
        
        # 应该有系统指标被记录
        collector = dashboard_server.metric_collector
        assert collector.get_metric("system_cpu_usage").get_latest_value() is not None
        assert collector.get_metric("system_memory_usage").get_latest_value() is not None
        assert collector.get_metric("system_disk_usage").get_latest_value() is not None
    
    def test_on_alert(self, dashboard_server):
        """测试告警处理"""
        alert = Alert(
            id="test_dashboard_alert",
            metric_name="test_metric",
            level=AlertLevel.WARNING,
            message="测试仪表板告警",
            timestamp=datetime.now(timezone.utc),
            threshold=100.0,
            actual_value=150.0
        )
        
        # 调用告警处理方法（应该不会抛出异常）
        dashboard_server._on_alert(alert)
        
        # 这里可以添加更多的断言，比如检查告警是否被正确记录


@pytest.mark.integration
class TestMonitoringIntegration:
    """监控系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """测试完整的监控工作流"""
        # 1. 创建监控组件
        collector = MetricCollector()
        dashboard = DashboardServer(collector, port=8082)
        
        # 2. 启动仪表板
        await dashboard.start()
        
        try:
            # 3. 注册自定义指标
            collector.register_metric("test_workflow_metric", MetricType.GAUGE, "工作流测试指标", "units")
            
            # 4. 添加自定义告警规则
            rule = AlertRule(
                name="workflow_test_alert",
                metric_name="test_workflow_metric",
                condition=">",
                threshold=100.0,
                level=AlertLevel.CRITICAL,
                message_template="工作流测试告警: {actual_value}",
                duration_minutes=0
            )
            collector.add_alert_rule(rule)
            
            # 5. 记录一些指标数据
            for i in range(10):
                collector.record_metric("test_workflow_metric", 50 + i * 10)
                await asyncio.sleep(0.01)
            
            # 6. 评估告警
            collector.evaluate_alerts()
            
            # 7. 验证告警被触发
            active_alerts = collector.get_active_alerts()
            workflow_alerts = [a for a in active_alerts if a.metric_name == "test_workflow_metric"]
            assert len(workflow_alerts) > 0
            
            # 8. 获取仪表板数据
            dashboard_data = dashboard.get_dashboard_data()
            assert "test_workflow_metric" in dashboard_data["metrics"]
            
            # 9. 获取特定指标数据
            metric_data = dashboard.get_metric_data("test_workflow_metric")
            assert metric_data is not None
            assert len(metric_data["points"]) == 10
            
            # 10. 验证统计数据
            stats = metric_data["statistics"]
            assert stats["min"] == 50
            assert stats["max"] == 140
            assert stats["latest"] == 140
            
        finally:
            await dashboard.stop()


@pytest.mark.performance
class TestMonitoringPerformance:
    """监控系统性能测试"""
    
    def test_metric_recording_performance(self):
        """测试指标记录性能"""
        collector = MetricCollector()
        collector.register_metric("perf_test_metric", MetricType.COUNTER, "性能测试指标")
        
        # 记录大量指标
        start_time = time.time()
        for i in range(1000):
            collector.record_metric("perf_test_metric", i)
        end_time = time.time()
        
        # 应该能够快速完成
        duration = end_time - start_time
        assert duration < 1.0  # 1000个指标应该在1秒内完成
        
        # 验证数据正确性
        metric = collector.get_metric("perf_test_metric")
        assert len(metric.points) == 1000
        assert metric.get_latest_value() == 999
    
    def test_alert_evaluation_performance(self):
        """测试告警评估性能"""
        collector = MetricCollector()
        
        # 创建多个指标和告警规则
        for i in range(100):
            metric_name = f"perf_metric_{i}"
            collector.register_metric(metric_name, MetricType.GAUGE, f"性能指标{i}")
            
            rule = AlertRule(
                name=f"rule_{i}",
                metric_name=metric_name,
                condition=">",
                threshold=50.0,
                level=AlertLevel.WARNING,
                message_template=f"规则{i}告警",
                duration_minutes=0
            )
            collector.add_alert_rule(rule)
            
            # 记录一些数据
            collector.record_metric(metric_name, 25.0)  # 不触发告警
        
        # 评估告警性能
        start_time = time.time()
        collector.evaluate_alerts()
        end_time = time.time()
        
        # 应该能够快速完成
        duration = end_time - start_time
        assert duration < 0.5  # 100个规则应该在0.5秒内评估完成


if __name__ == "__main__":
    pytest.main([__file__])