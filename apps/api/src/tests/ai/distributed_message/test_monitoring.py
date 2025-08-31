"""
监控和性能优化测试
测试性能指标监控、健康检查、告警系统和性能优化功能
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.ai.distributed_message.monitoring import (
    MonitoringManager, MetricCollector, HealthChecker, AlertManager, PerformanceOptimizer,
    PerformanceMetric, Alert, HealthStatus, AlertLevel, MetricType
)


class TestMetricCollector:
    """指标收集器测试"""
    
    @pytest.fixture
    def metric_collector(self):
        return MetricCollector()
    
    def test_increment_counter(self, metric_collector):
        """测试增加计数器"""
        metric_collector.increment_counter("test_counter", 5.0, {"service": "test"})
        metric_collector.increment_counter("test_counter", 3.0, {"service": "test"})
        
        history = metric_collector.get_metric_history("test_counter", {"service": "test"})
        assert len(history) == 2
        assert history[0].value == 5.0  # 第一次记录的累积值
        assert history[1].value == 8.0  # 第二次记录的累积值
    
    def test_set_gauge(self, metric_collector):
        """测试设置仪表盘值"""
        metric_collector.set_gauge("test_gauge", 100.0)
        metric_collector.set_gauge("test_gauge", 150.0)
        
        history = metric_collector.get_metric_history("test_gauge")
        assert len(history) == 2
        assert history[0].value == 100.0
        assert history[1].value == 150.0
    
    def test_record_timer(self, metric_collector):
        """测试记录计时器"""
        metric_collector.record_timer("test_timer", 0.5)
        metric_collector.record_timer("test_timer", 1.2)
        
        history = metric_collector.get_metric_history("test_timer")
        assert len(history) == 2
        assert history[0].value == 0.5
        assert history[1].value == 1.2
    
    def test_get_metric_statistics(self, metric_collector):
        """测试获取指标统计信息"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            metric_collector.set_gauge("test_stats", value)
        
        stats = metric_collector.get_metric_statistics("test_stats")
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
    
    def test_cleanup_old_metrics(self, metric_collector):
        """测试清理过期指标"""
        # 设置较短的保留期
        metric_collector.retention_period = timedelta(seconds=1)
        
        # 记录指标
        metric_collector.set_gauge("test_cleanup", 1.0)
        
        # 等待过期
        time.sleep(1.1)
        
        # 记录新指标
        metric_collector.set_gauge("test_cleanup", 2.0)
        
        # 执行清理
        metric_collector.cleanup_old_metrics()
        
        # 验证只有新指标保留
        history = metric_collector.get_metric_history("test_cleanup")
        assert len(history) == 1
        assert history[0].value == 2.0


class TestHealthChecker:
    """健康检查器测试"""
    
    @pytest.fixture
    def health_checker(self):
        return HealthChecker()
    
    def test_register_health_check(self, health_checker):
        """测试注册健康检查"""
        def test_check():
            return HealthStatus("test_component", True, "测试正常")
        
        health_checker.register_health_check("test_component", test_check)
        assert "test_component" in health_checker.health_checks
    
    @pytest.mark.asyncio
    async def test_run_health_check(self, health_checker):
        """测试运行健康检查"""
        def healthy_check():
            return HealthStatus("healthy_component", True, "正常")
        
        def unhealthy_check():
            return HealthStatus("unhealthy_component", False, "异常")
        
        health_checker.register_health_check("healthy_component", healthy_check)
        health_checker.register_health_check("unhealthy_component", unhealthy_check)
        
        # 测试健康的组件
        result = await health_checker.run_health_check("healthy_component")
        assert result.healthy is True
        assert result.component == "healthy_component"
        
        # 测试不健康的组件
        result = await health_checker.run_health_check("unhealthy_component")
        assert result.healthy is False
        assert result.component == "unhealthy_component"
    
    @pytest.mark.asyncio
    async def test_run_all_health_checks(self, health_checker):
        """测试运行所有健康检查"""
        def check1():
            return HealthStatus("component1", True, "正常")
        
        def check2():
            return HealthStatus("component2", False, "异常")
        
        health_checker.register_health_check("component1", check1)
        health_checker.register_health_check("component2", check2)
        
        results = await health_checker.run_all_health_checks()
        
        assert len(results) == 2
        assert results["component1"].healthy is True
        assert results["component2"].healthy is False
    
    def test_get_overall_health(self, health_checker):
        """测试获取整体健康状态"""
        # 无检查项时
        overall = health_checker.get_overall_health()
        assert overall.healthy is True
        
        # 有健康组件时
        health_checker.last_check_results["comp1"] = HealthStatus("comp1", True, "正常")
        overall = health_checker.get_overall_health()
        assert overall.healthy is True
        
        # 有不健康组件时
        health_checker.last_check_results["comp2"] = HealthStatus("comp2", False, "异常")
        overall = health_checker.get_overall_health()
        assert overall.healthy is False
        assert "comp2" in overall.message


class TestAlertManager:
    """告警管理器测试"""
    
    @pytest.fixture
    def alert_manager(self):
        return AlertManager()
    
    def test_create_alert(self, alert_manager):
        """测试创建告警"""
        alert = alert_manager.create_alert(
            name="test_alert",
            level=AlertLevel.WARNING,
            message="测试告警消息",
            metadata={"test": "data"}
        )
        
        assert alert.name == "test_alert"
        assert alert.level == AlertLevel.WARNING
        assert alert.resolved is False
        assert alert.alert_id in alert_manager.alerts
    
    def test_resolve_alert(self, alert_manager):
        """测试解决告警"""
        alert = alert_manager.create_alert("test", AlertLevel.INFO, "测试")
        
        success = alert_manager.resolve_alert(alert.alert_id)
        assert success is True
        assert alert.resolved is True
        assert alert.resolved_at is not None
    
    def test_add_alert_rule(self, alert_manager):
        """测试添加告警规则"""
        alert_manager.add_alert_rule(
            "high_cpu",
            "cpu_percent",
            "gt",
            80.0,
            AlertLevel.WARNING
        )
        
        assert "high_cpu" in alert_manager.rules
        rule = alert_manager.rules["high_cpu"]
        assert rule["metric_name"] == "cpu_percent"
        assert rule["condition"] == "gt"
        assert rule["threshold"] == 80.0
    
    def test_check_metric_alerts(self, alert_manager):
        """测试检查指标告警"""
        # 添加告警规则
        alert_manager.add_alert_rule(
            "high_cpu",
            "cpu_percent",
            "gt",
            80.0,
            AlertLevel.WARNING
        )
        
        # 创建触发告警的指标
        metric = PerformanceMetric(
            name="cpu_percent",
            value=85.0,
            metric_type=MetricType.GAUGE
        )
        
        initial_alert_count = len(alert_manager.alerts)
        alert_manager.check_metric_alerts(metric)
        
        # 验证告警被触发
        assert len(alert_manager.alerts) == initial_alert_count + 1
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert "high_cpu" in active_alerts[0].name
    
    def test_get_alert_summary(self, alert_manager):
        """测试获取告警摘要"""
        # 创建不同级别的告警
        alert_manager.create_alert("warning1", AlertLevel.WARNING, "警告1")
        alert_manager.create_alert("error1", AlertLevel.ERROR, "错误1")
        alert_manager.create_alert("critical1", AlertLevel.CRITICAL, "严重1")
        
        # 解决一个告警
        alerts = list(alert_manager.alerts.keys())
        alert_manager.resolve_alert(alerts[0])
        
        summary = alert_manager.get_alert_summary()
        
        assert summary["total_alerts"] == 3
        assert summary["active_alerts"] == 2
        assert summary["resolved_alerts"] == 1
        assert summary["by_level"]["warning"] == 1
        assert summary["by_level"]["error"] == 1
        assert summary["by_level"]["critical"] == 1
    
    def test_max_alerts_limit(self, alert_manager):
        """测试告警数量限制"""
        alert_manager.max_alerts = 3
        
        # 创建3个告警
        for i in range(3):
            alert_manager.create_alert(f"alert{i}", AlertLevel.INFO, f"消息{i}")
        
        assert len(alert_manager.alerts) == 3
        
        # 创建第4个告警，应该删除最老的
        alert_manager.create_alert("alert3", AlertLevel.INFO, "消息3")
        assert len(alert_manager.alerts) == 3


class TestPerformanceOptimizer:
    """性能优化器测试"""
    
    @pytest.fixture
    def optimizer(self):
        return PerformanceOptimizer()
    
    def test_enable_disable_compression(self, optimizer):
        """测试启用/禁用压缩"""
        assert optimizer.compression_enabled is False
        
        optimizer.enable_compression()
        assert optimizer.compression_enabled is True
        
        optimizer.disable_compression()
        assert optimizer.compression_enabled is False
    
    def test_compress_decompress_data(self, optimizer):
        """测试数据压缩和解压缩"""
        test_data = "这是测试数据".encode('utf-8') * 100
        
        # 未启用压缩时
        compressed = optimizer.compress_data(test_data)
        assert compressed == test_data
        
        # 启用压缩后
        optimizer.enable_compression()
        compressed = optimizer.compress_data(test_data)
        assert len(compressed) < len(test_data)  # 压缩后应该更小
        
        decompressed = optimizer.decompress_data(compressed)
        assert decompressed == test_data
    
    def test_batch_processing(self, optimizer):
        """测试批处理"""
        optimizer.batch_size = 3
        
        # 添加项目到批次
        assert optimizer.add_to_batch("test_batch", "item1") is False
        assert optimizer.add_to_batch("test_batch", "item2") is False
        assert optimizer.add_to_batch("test_batch", "item3") is True  # 达到批次大小
        
        # 获取并清空批次
        batch = optimizer.get_and_clear_batch("test_batch")
        assert batch == ["item1", "item2", "item3"]
        assert len(optimizer.message_batches["test_batch"]) == 0
    
    def test_batch_timeout(self, optimizer):
        """测试批处理超时"""
        optimizer.batch_timeout = 0.1  # 100ms超时
        
        # 添加一个项目
        assert optimizer.add_to_batch("timeout_batch", "item1") is False
        
        # 等待超时
        time.sleep(0.15)
        
        # 再添加一个项目，应该触发超时
        assert optimizer.add_to_batch("timeout_batch", "item2") is True
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_get_system_performance(self, mock_net, mock_disk, mock_memory, mock_cpu, optimizer):
        """测试获取系统性能信息"""
        # 模拟系统信息
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(total=8000000000, available=4000000000, percent=50.0, used=4000000000)
        mock_disk.return_value = Mock(total=1000000000000, used=600000000000, free=400000000000)
        mock_net.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000, packets_sent=5000, packets_recv=7000)
        
        perf = optimizer.get_system_performance()
        
        assert perf["cpu_percent"] == 45.5
        assert perf["memory"]["percent"] == 50.0
        assert perf["disk"]["total"] == 1000000000000
        assert perf["network"]["bytes_sent"] == 1000000


class TestMonitoringManager:
    """监控管理器测试"""
    
    @pytest.fixture
    def monitoring_manager(self):
        return MonitoringManager(check_interval=0.1)  # 快速检查间隔用于测试
    
    @pytest.mark.asyncio
    async def test_start_stop(self, monitoring_manager):
        """测试启动和停止"""
        assert monitoring_manager.is_running is False
        
        await monitoring_manager.start()
        assert monitoring_manager.is_running is True
        assert len(monitoring_manager.monitoring_tasks) > 0
        
        await monitoring_manager.stop()
        assert monitoring_manager.is_running is False
        assert len(monitoring_manager.monitoring_tasks) == 0
    
    def test_record_message_metric(self, monitoring_manager):
        """测试记录消息指标"""
        monitoring_manager.record_message_metric("messages_sent_count", 10.0)
        monitoring_manager.record_message_metric("message_process_duration", 0.5)
        monitoring_manager.record_message_metric("queue_size", 25.0)
        
        # 验证指标被记录
        history = monitoring_manager.metric_collector.get_metric_history("messages_sent_count")
        assert len(history) == 1
        assert history[0].value == 10.0
        
        history = monitoring_manager.metric_collector.get_metric_history("message_process_duration")
        assert len(history) == 1
        assert history[0].value == 0.5
        
        history = monitoring_manager.metric_collector.get_metric_history("queue_size")
        assert len(history) == 1
        assert history[0].value == 25.0
    
    def test_get_monitoring_dashboard(self, monitoring_manager):
        """测试获取监控仪表板"""
        # 记录一些指标和告警
        monitoring_manager.record_message_metric("test_metric", 100.0)
        monitoring_manager.alert_manager.create_alert("test_alert", AlertLevel.INFO, "测试告警")
        
        dashboard = monitoring_manager.get_monitoring_dashboard()
        
        assert "health" in dashboard
        assert "alerts" in dashboard
        assert "performance" in dashboard
        assert "metrics" in dashboard
        
        # 验证健康状态
        assert "overall" in dashboard["health"]
        assert "components" in dashboard["health"]
        
        # 验证告警信息
        assert "summary" in dashboard["alerts"]
        assert "active" in dashboard["alerts"]
        assert dashboard["alerts"]["summary"]["total_alerts"] == 1
        
        # 验证性能信息
        assert "system" in dashboard["performance"]
        assert "compression_enabled" in dashboard["performance"]
        
        # 验证指标信息
        assert "total_metrics" in dashboard["metrics"]
    
    @pytest.mark.asyncio
    async def test_alert_handler_registration(self, monitoring_manager):
        """测试告警处理器注册"""
        handled_alerts = []
        
        def test_alert_handler(alert: Alert):
            handled_alerts.append(alert)
        
        monitoring_manager.alert_manager.register_alert_handler(test_alert_handler)
        
        # 创建告警
        alert = monitoring_manager.alert_manager.create_alert(
            "test", AlertLevel.WARNING, "测试告警"
        )
        
        # 验证处理器被调用
        assert len(handled_alerts) == 1
        assert handled_alerts[0].alert_id == alert.alert_id
    
    @patch('src.ai.distributed_message.monitoring.psutil.cpu_percent')
    @patch('src.ai.distributed_message.monitoring.psutil.virtual_memory')
    @patch('src.ai.distributed_message.monitoring.psutil.disk_usage')
    @patch('src.ai.distributed_message.monitoring.psutil.net_io_counters')
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, mock_net, mock_disk, mock_memory, mock_cpu, monitoring_manager):
        """测试系统指标收集"""
        # 模拟系统信息
        mock_cpu.return_value = 75.0
        mock_memory.return_value = Mock(total=8000000000, available=2000000000, percent=75.0, used=6000000000)
        mock_disk.return_value = Mock(total=1000000000000, used=500000000000, free=500000000000)
        mock_net.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000, packets_sent=5000, packets_recv=7000)
        
        # 收集指标
        await monitoring_manager._collect_system_metrics()
        
        # 验证指标被收集
        cpu_history = monitoring_manager.metric_collector.get_metric_history("system.cpu_percent")
        assert len(cpu_history) == 1
        assert cpu_history[0].value == 75.0
        
        memory_history = monitoring_manager.metric_collector.get_metric_history("system.memory_percent")
        assert len(memory_history) == 1
        assert memory_history[0].value == 75.0
    
    def test_default_alert_rules(self, monitoring_manager):
        """测试默认告警规则"""
        # 验证默认规则已注册
        assert "high_cpu_usage" in monitoring_manager.alert_manager.rules
        assert "high_memory_usage" in monitoring_manager.alert_manager.rules
        assert "message_error_rate_high" in monitoring_manager.alert_manager.rules
        
        # 测试CPU告警规则
        cpu_rule = monitoring_manager.alert_manager.rules["high_cpu_usage"]
        assert cpu_rule["metric_name"] == "system.cpu_percent"
        assert cpu_rule["condition"] == "gt"
        assert cpu_rule["threshold"] == 80.0
    
    @pytest.mark.asyncio
    async def test_monitoring_with_real_interval(self, monitoring_manager):
        """测试真实间隔的监控"""
        await monitoring_manager.start()
        
        # 让监控运行一小段时间
        await asyncio.sleep(0.3)  # 运行几个检查周期
        
        await monitoring_manager.stop()
        
        # 验证健康检查被执行
        assert len(monitoring_manager.health_checker.last_check_results) > 0
        
        # 验证系统指标被收集
        cpu_history = monitoring_manager.metric_collector.get_metric_history("system.cpu_percent")
        assert len(cpu_history) > 0  # 应该有至少一次指标收集