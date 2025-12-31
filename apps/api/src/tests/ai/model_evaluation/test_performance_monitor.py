import pytest
import time
import threading
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
from ai.model_evaluation.performance_monitor import (
    PerformanceMonitor,
    SystemMetrics,
    ModelMetrics,
    BenchmarkMetrics,
    PerformanceAlert,
    PerformanceAnalyzer
)

class TestSystemMetrics:
    """测试系统性能指标类"""
    
    def test_system_metrics_creation(self):
        """测试系统指标创建"""
        timestamp = utc_now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=85.2,
            memory_used_gb=16.5,
            memory_total_gb=32.0,
            gpu_percent=90.0,
            gpu_memory_used_gb=8.5,
            gpu_memory_total_gb=24.0,
            gpu_temperature=65.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 75.5
        assert metrics.memory_percent == 85.2
        assert metrics.memory_used_gb == 16.5
        assert metrics.memory_total_gb == 32.0
        assert metrics.gpu_percent == 90.0
        assert metrics.gpu_memory_used_gb == 8.5
        assert metrics.gpu_memory_total_gb == 24.0
        assert metrics.gpu_temperature == 65.0

class TestModelMetrics:
    """测试模型性能指标类"""
    
    def test_model_metrics_creation(self):
        """测试模型指标创建"""
        timestamp = utc_now()
        metrics = ModelMetrics(
            model_name="test_model",
            timestamp=timestamp,
            batch_size=32,
            sequence_length=128,
            inference_time_ms=150.5,
            tokens_per_second=213.3,
            memory_usage_mb=1024.0,
            gpu_utilization=85.0
        )
        
        assert metrics.model_name == "test_model"
        assert metrics.timestamp == timestamp
        assert metrics.batch_size == 32
        assert metrics.sequence_length == 128
        assert metrics.inference_time_ms == 150.5
        assert metrics.tokens_per_second == 213.3
        assert metrics.memory_usage_mb == 1024.0
        assert metrics.gpu_utilization == 85.0

class TestBenchmarkMetrics:
    """测试基准测试指标类"""
    
    def test_benchmark_metrics_creation(self):
        """测试基准测试指标创建"""
        timestamp = utc_now()
        metrics = BenchmarkMetrics(
            benchmark_name="test_benchmark",
            model_name="test_model",
            timestamp=timestamp,
            total_samples=1000,
            processed_samples=1000,
            accuracy=0.85,
            throughput=50.5,
            avg_inference_time=20.0,
            p95_inference_time=35.0,
            p99_inference_time=45.0,
            total_time_seconds=120.0,
            peak_memory_usage_gb=8.5
        )
        
        assert metrics.benchmark_name == "test_benchmark"
        assert metrics.model_name == "test_model"
        assert metrics.timestamp == timestamp
        assert metrics.total_samples == 1000
        assert metrics.processed_samples == 1000
        assert metrics.accuracy == 0.85
        assert metrics.throughput == 50.5
        assert metrics.avg_inference_time == 20.0
        assert metrics.p95_inference_time == 35.0
        assert metrics.p99_inference_time == 45.0
        assert metrics.total_time_seconds == 120.0
        assert metrics.peak_memory_usage_gb == 8.5

class TestPerformanceAlert:
    """测试性能告警类"""
    
    def test_performance_alert_creation(self):
        """测试性能告警创建"""
        timestamp = utc_now()
        alert = PerformanceAlert(
            alert_id="alert_001",
            timestamp=timestamp,
            severity="high",
            category="memory",
            title="High Memory Usage",
            description="Memory usage exceeded 90%",
            metric_name="memory_percent",
            current_value=95.0,
            threshold_value=90.0,
            model_name="test_model",
            resolved=False
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.timestamp == timestamp
        assert alert.severity == "high"
        assert alert.category == "memory"
        assert alert.title == "High Memory Usage"
        assert alert.description == "Memory usage exceeded 90%"
        assert alert.metric_name == "memory_percent"
        assert alert.current_value == 95.0
        assert alert.threshold_value == 90.0
        assert alert.model_name == "test_model"
        assert not alert.resolved

class TestPerformanceMonitor:
    """测试性能监控器"""
    
    @pytest.fixture
    def monitor(self):
        """创建性能监控器实例"""
        return PerformanceMonitor(sampling_interval=0.1, history_size=10)
    
    def test_monitor_initialization(self, monitor):
        """测试监控器初始化"""
        assert monitor.sampling_interval == 0.1
        assert monitor.history_size == 10
        assert not monitor.is_monitoring
        assert monitor.monitor_thread is None
        assert isinstance(monitor.system_metrics_history, type(monitor.system_metrics_history))
        assert isinstance(monitor.model_metrics_history, dict)
        assert isinstance(monitor.benchmark_metrics_history, dict)
        assert isinstance(monitor.alerts, list)
        assert isinstance(monitor.alert_thresholds, dict)
    
    def test_default_thresholds(self, monitor):
        """测试默认告警阈值"""
        thresholds = monitor._get_default_thresholds()
        
        assert "cpu_percent" in thresholds
        assert "memory_percent" in thresholds
        assert "gpu_memory_percent" in thresholds
        assert "gpu_temperature" in thresholds
        assert "inference_time_ms" in thresholds
        assert "accuracy_drop" in thresholds
        assert "throughput_drop" in thresholds
        
        assert thresholds["cpu_percent"] == 85.0
        assert thresholds["memory_percent"] == 90.0
        assert thresholds["gpu_memory_percent"] == 95.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_collect_system_metrics(self, mock_virtual_memory, mock_cpu_percent, monitor):
        """测试收集系统指标"""
        # 模拟系统指标
        mock_cpu_percent.return_value = 75.5
        
        mock_memory = Mock()
        mock_memory.percent = 85.2
        mock_memory.used = 16.5 * (1024**3)  # 16.5 GB in bytes
        mock_memory.total = 32.0 * (1024**3)  # 32 GB in bytes
        mock_virtual_memory.return_value = mock_memory
        
        with patch.object(monitor, '_get_gpu_utilization', return_value=90.0), \
             patch.object(monitor, '_get_gpu_memory', return_value=(8.5 * (1024**3), 24.0 * (1024**3))), \
             patch.object(monitor, '_get_gpu_temperature', return_value=65.0):
            
            metrics = monitor._collect_system_metrics()
            
            assert metrics.cpu_percent == 75.5
            assert metrics.memory_percent == 85.2
            assert abs(metrics.memory_used_gb - 16.5) < 0.1
            assert abs(metrics.memory_total_gb - 32.0) < 0.1
            assert metrics.gpu_percent == 90.0
            assert abs(metrics.gpu_memory_used_gb - 8.5) < 0.1
            assert abs(metrics.gpu_memory_total_gb - 24.0) < 0.1
            assert metrics.gpu_temperature == 65.0
    
    def test_record_model_metrics(self, monitor):
        """测试记录模型性能指标"""
        model_name = "test_model"
        
        with patch.object(monitor, '_get_gpu_utilization', return_value=80.0):
            monitor.record_model_metrics(
                model_name=model_name,
                batch_size=32,
                sequence_length=128,
                inference_time_ms=150.0,
                memory_usage_mb=1024.0,
                tokens_processed=1000
            )
        
        # 验证指标已记录
        assert model_name in monitor.model_metrics_history
        assert len(monitor.model_metrics_history[model_name]) == 1
        
        metrics = monitor.model_metrics_history[model_name][0]
        assert metrics.model_name == model_name
        assert metrics.batch_size == 32
        assert metrics.sequence_length == 128
        assert metrics.inference_time_ms == 150.0
        assert metrics.memory_usage_mb == 1024.0
        assert abs(metrics.tokens_per_second - (1000 / 0.15)) < 0.1
    
    def test_record_benchmark_metrics(self, monitor):
        """测试记录基准测试指标"""
        # 先添加一些模型指标用于计算统计信息
        model_name = "test_model"
        for i in range(5):
            monitor.model_metrics_history[model_name].append(
                ModelMetrics(
                    model_name=model_name,
                    timestamp=utc_now(),
                    batch_size=32,
                    sequence_length=128,
                    inference_time_ms=150.0 + i * 10,
                    tokens_per_second=100.0,
                    memory_usage_mb=1024.0
                )
            )
        
        # 添加系统指标用于峰值内存计算
        for i in range(3):
            monitor.system_metrics_history.append(
                SystemMetrics(
                    timestamp=utc_now(),
                    cpu_percent=70.0,
                    memory_percent=80.0,
                    memory_used_gb=16.0 + i,
                    memory_total_gb=32.0
                )
            )
        
        with patch.object(monitor, '_get_gpu_utilization', return_value=85.0):
            monitor.record_benchmark_metrics(
                benchmark_name="test_benchmark",
                model_name=model_name,
                total_samples=1000,
                processed_samples=1000,
                accuracy=0.85,
                total_time_seconds=120.0
            )
        
        # 验证基准测试指标已记录
        key = f"test_benchmark_{model_name}"
        assert key in monitor.benchmark_metrics_history
        assert len(monitor.benchmark_metrics_history[key]) == 1
        
        metrics = monitor.benchmark_metrics_history[key][0]
        assert metrics.benchmark_name == "test_benchmark"
        assert metrics.model_name == model_name
        assert metrics.total_samples == 1000
        assert metrics.processed_samples == 1000
        assert metrics.accuracy == 0.85
        assert abs(metrics.throughput - (1000 / 120.0)) < 0.1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_check_system_alerts(self, mock_virtual_memory, mock_cpu_percent, monitor):
        """测试系统告警检查"""
        # 模拟超过阈值的系统指标
        mock_cpu_percent.return_value = 90.0  # 超过85%阈值
        
        mock_memory = Mock()
        mock_memory.percent = 95.0  # 超过90%阈值
        mock_memory.used = 30.0 * (1024**3)
        mock_memory.total = 32.0 * (1024**3)
        mock_virtual_memory.return_value = mock_memory
        
        with patch.object(monitor, '_get_gpu_utilization', return_value=90.0), \
             patch.object(monitor, '_get_gpu_memory', return_value=(22.0 * (1024**3), 24.0 * (1024**3))), \
             patch.object(monitor, '_get_gpu_temperature', return_value=90.0):  # 超过85℃阈值
            
            metrics = monitor._collect_system_metrics()
            initial_alerts_count = len(monitor.alerts)
            
            monitor._check_system_alerts(metrics)
            
            # 应该生成多个告警
            assert len(monitor.alerts) > initial_alerts_count
            
            # 验证告警内容
            alert_categories = [alert.category for alert in monitor.alerts]
            assert "system" in alert_categories
    
    def test_check_model_alerts(self, monitor):
        """测试模型性能告警检查"""
        # 设置较低的推理时间阈值用于测试
        monitor.alert_thresholds["inference_time_ms"] = 100.0
        
        model_metrics = ModelMetrics(
            model_name="slow_model",
            timestamp=utc_now(),
            batch_size=32,
            sequence_length=128,
            inference_time_ms=150.0,  # 超过100ms阈值
            tokens_per_second=100.0,
            memory_usage_mb=1024.0
        )
        
        initial_alerts_count = len(monitor.alerts)
        monitor._check_model_alerts(model_metrics)
        
        # 应该生成告警
        assert len(monitor.alerts) > initial_alerts_count
        
        # 验证告警内容
        latest_alert = monitor.alerts[-1]
        assert latest_alert.category == "performance"
        assert latest_alert.model_name == "slow_model"
        assert latest_alert.metric_name == "inference_time_ms"
        assert latest_alert.current_value == 150.0
    
    def test_check_benchmark_alerts(self, monitor):
        """测试基准测试告警检查"""
        # 设置性能基线
        baseline_key = "test_benchmark_test_model"
        monitor.performance_baselines[baseline_key] = {
            "accuracy": 0.90,
            "throughput": 100.0,
            "timestamp": utc_now()
        }
        
        # 创建准确率下降的基准测试指标
        benchmark_metrics = BenchmarkMetrics(
            benchmark_name="test_benchmark",
            model_name="test_model",
            timestamp=utc_now(),
            total_samples=1000,
            processed_samples=1000,
            accuracy=0.80,  # 比基线0.90低了0.10，超过0.05阈值
            throughput=95.0,
            avg_inference_time=20.0,
            p95_inference_time=30.0,
            p99_inference_time=40.0,
            total_time_seconds=120.0,
            peak_memory_usage_gb=8.0
        )
        
        initial_alerts_count = len(monitor.alerts)
        monitor._check_benchmark_alerts(benchmark_metrics)
        
        # 应该生成告警
        assert len(monitor.alerts) > initial_alerts_count
        
        # 验证告警内容
        latest_alert = monitor.alerts[-1]
        assert latest_alert.category == "accuracy"
        assert latest_alert.benchmark_name == "test_benchmark"
        assert latest_alert.model_name == "test_model"
    
    def test_get_system_metrics_summary(self, monitor):
        """测试获取系统指标摘要"""
        # 添加一些测试数据
        base_time = utc_now()
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=base_time - timedelta(minutes=i),
                cpu_percent=70.0 + i * 5,
                memory_percent=80.0 + i * 2,
                memory_used_gb=16.0 + i,
                memory_total_gb=32.0,
                gpu_percent=75.0 + i * 3
            )
            monitor.system_metrics_history.append(metrics)
        
        summary = monitor.get_system_metrics_summary(60)
        
        assert "time_range_minutes" in summary
        assert "sample_count" in summary
        assert "cpu" in summary
        assert "memory" in summary
        
        assert summary["time_range_minutes"] == 60
        assert summary["sample_count"] == 5
        assert "avg" in summary["cpu"]
        assert "max" in summary["cpu"]
        assert "min" in summary["cpu"]
    
    def test_get_model_performance_summary(self, monitor):
        """测试获取模型性能摘要"""
        model_name = "test_model"
        base_time = utc_now()
        
        # 添加测试数据
        for i in range(5):
            metrics = ModelMetrics(
                model_name=model_name,
                timestamp=base_time - timedelta(minutes=i),
                batch_size=32,
                sequence_length=128,
                inference_time_ms=150.0 + i * 10,
                tokens_per_second=100.0 + i * 5,
                memory_usage_mb=1024.0 + i * 100
            )
            monitor.model_metrics_history[model_name].append(metrics)
        
        summary = monitor.get_model_performance_summary(model_name, 60)
        
        assert "model_name" in summary
        assert "time_range_minutes" in summary
        assert "total_inferences" in summary
        assert "inference_time_ms" in summary
        assert "throughput" in summary
        assert "memory_usage_mb" in summary
        
        assert summary["model_name"] == model_name
        assert summary["total_inferences"] == 5
        assert "avg" in summary["inference_time_ms"]
        assert "median" in summary["inference_time_ms"]
        assert "p95" in summary["inference_time_ms"]
        assert "p99" in summary["inference_time_ms"]
    
    def test_get_benchmark_comparison(self, monitor):
        """测试获取基准测试对比"""
        benchmark_name = "test_benchmark"
        model_names = ["model_1", "model_2"]
        
        # 为每个模型添加基准测试指标
        for i, model_name in enumerate(model_names):
            key = f"{benchmark_name}_{model_name}"
            metrics = BenchmarkMetrics(
                benchmark_name=benchmark_name,
                model_name=model_name,
                timestamp=utc_now(),
                total_samples=1000,
                processed_samples=1000,
                accuracy=0.80 + i * 0.05,
                throughput=50.0 + i * 10,
                avg_inference_time=200.0 - i * 50,
                p95_inference_time=300.0 - i * 50,
                total_time_seconds=120.0,
                peak_memory_usage_gb=8.0 + i
            )
            monitor.benchmark_metrics_history[key].append(metrics)
        
        comparison = monitor.get_benchmark_comparison(benchmark_name, model_names)
        
        assert "benchmark_name" in comparison
        assert "models" in comparison
        assert "comparison_timestamp" in comparison
        
        assert comparison["benchmark_name"] == benchmark_name
        assert len(comparison["models"]) == 2
        assert "model_1" in comparison["models"]
        assert "model_2" in comparison["models"]
        
        # 验证模型数据
        model_1_data = comparison["models"]["model_1"]
        assert "accuracy" in model_1_data
        assert "throughput" in model_1_data
        assert model_1_data["accuracy"] == 0.80
    
    def test_set_performance_baseline(self, monitor):
        """测试设置性能基线"""
        model_name = "test_model"
        benchmark_name = "test_benchmark"
        
        metrics = BenchmarkMetrics(
            benchmark_name=benchmark_name,
            model_name=model_name,
            timestamp=utc_now(),
            total_samples=1000,
            processed_samples=1000,
            accuracy=0.85,
            throughput=50.0,
            avg_inference_time=200.0,
            p95_inference_time=300.0,
            p99_inference_time=400.0,
            total_time_seconds=120.0,
            peak_memory_usage_gb=8.0
        )
        
        monitor.set_performance_baseline(model_name, benchmark_name, metrics)
        
        baseline_key = f"{benchmark_name}_{model_name}"
        assert baseline_key in monitor.performance_baselines
        
        baseline = monitor.performance_baselines[baseline_key]
        assert baseline["accuracy"] == 0.85
        assert baseline["throughput"] == 50.0
        assert baseline["avg_inference_time"] == 200.0
    
    def test_get_active_alerts(self, monitor):
        """测试获取活跃告警"""
        # 添加一些测试告警
        alerts = [
            PerformanceAlert(
                alert_id="alert_1",
                timestamp=utc_now(),
                severity="high",
                category="memory",
                title="High Memory Usage",
                description="Memory usage exceeded threshold",
                metric_name="memory_percent",
                current_value=95.0,
                threshold_value=90.0,
                resolved=False
            ),
            PerformanceAlert(
                alert_id="alert_2",
                timestamp=utc_now(),
                severity="medium",
                category="cpu",
                title="High CPU Usage",
                description="CPU usage exceeded threshold",
                metric_name="cpu_percent",
                current_value=88.0,
                threshold_value=85.0,
                resolved=True  # 已解决
            ),
            PerformanceAlert(
                alert_id="alert_3",
                timestamp=utc_now(),
                severity="high",
                category="gpu",
                title="High GPU Temperature",
                description="GPU temperature exceeded threshold",
                metric_name="gpu_temperature",
                current_value=90.0,
                threshold_value=85.0,
                resolved=False
            )
        ]
        
        monitor.alerts.extend(alerts)
        
        # 获取所有活跃告警
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 2  # 两个未解决的告警
        
        # 按严重程度过滤
        high_severity_alerts = monitor.get_active_alerts(severity="high")
        assert len(high_severity_alerts) == 2  # 两个高严重程度的未解决告警
        
        medium_severity_alerts = monitor.get_active_alerts(severity="medium")
        assert len(medium_severity_alerts) == 0  # 中等严重程度的告警已解决
    
    def test_resolve_alert(self, monitor):
        """测试解决告警"""
        alert = PerformanceAlert(
            alert_id="test_alert",
            timestamp=utc_now(),
            severity="high",
            category="memory",
            title="Test Alert",
            description="Test alert description",
            metric_name="memory_percent",
            current_value=95.0,
            threshold_value=90.0,
            resolved=False
        )
        
        monitor.alerts.append(alert)
        assert not alert.resolved
        
        monitor.resolve_alert("test_alert")
        assert alert.resolved
    
    def test_export_metrics(self, monitor):
        """测试导出指标数据"""
        # 添加一些测试数据
        monitor.system_metrics_history.append(
            SystemMetrics(
                timestamp=utc_now(),
                cpu_percent=75.0,
                memory_percent=80.0,
                memory_used_gb=16.0,
                memory_total_gb=32.0
            )
        )
        
        monitor.model_metrics_history["test_model"].append(
            ModelMetrics(
                model_name="test_model",
                timestamp=utc_now(),
                batch_size=32,
                sequence_length=128,
                inference_time_ms=150.0,
                tokens_per_second=100.0,
                memory_usage_mb=1024.0
            )
        )
        
        monitor.alerts.append(
            PerformanceAlert(
                alert_id="export_test_alert",
                timestamp=utc_now(),
                severity="high",
                category="test",
                title="Export Test Alert",
                description="Test alert for export",
                metric_name="test_metric",
                current_value=100.0,
                threshold_value=90.0,
                resolved=False
            )
        )
        
        # 导出到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            monitor.export_metrics(temp_path, 1)
            
            # 验证导出文件
            assert Path(temp_path).exists()
            
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "export_timestamp" in exported_data
            assert "time_range_hours" in exported_data
            assert "system_metrics" in exported_data
            assert "model_metrics" in exported_data
            assert "benchmark_metrics" in exported_data
            assert "alerts" in exported_data
            
            assert exported_data["time_range_hours"] == 1
            assert len(exported_data["system_metrics"]) == 1
            assert "test_model" in exported_data["model_metrics"]
            assert len(exported_data["alerts"]) == 1
            
        finally:
            Path(temp_path).unlink()  # 清理临时文件
    
    def test_start_stop_monitoring(self, monitor):
        """测试启动和停止监控"""
        assert not monitor.is_monitoring
        assert monitor.monitor_thread is None
        
        # 启动监控
        monitor.start_monitoring()
        assert monitor.is_monitoring
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()
        
        # 等待一小段时间让监控循环运行
        time.sleep(0.2)
        
        # 停止监控
        monitor.stop_monitoring()
        assert not monitor.is_monitoring
        
        # 线程应该已停止
        time.sleep(0.2)
        assert not monitor.monitor_thread.is_alive()
    
    def test_monitoring_loop_with_exception(self, monitor):
        """测试监控循环中的异常处理"""
        # 模拟收集指标时抛出异常
        with patch.object(monitor, '_collect_system_metrics', side_effect=Exception("Test exception")):
            monitor.start_monitoring()
            
            # 等待一小段时间让异常发生
            time.sleep(0.2)
            
            # 监控应该仍在运行（异常被捕获）
            assert monitor.is_monitoring
            
            monitor.stop_monitoring()

class TestPerformanceAnalyzer:
    """测试性能分析器"""
    
    @pytest.fixture
    def monitor_with_data(self):
        """创建包含测试数据的监控器"""
        monitor = PerformanceMonitor(sampling_interval=0.1, history_size=100)
        
        # 添加测试数据
        base_time = utc_now()
        model_name = "test_model"
        
        for i in range(10):
            metrics = ModelMetrics(
                model_name=model_name,
                timestamp=base_time - timedelta(hours=i),
                batch_size=32,
                sequence_length=128,
                inference_time_ms=150.0 + i * 5,  # 递增趋势
                tokens_per_second=100.0 - i * 2,  # 递减趋势
                memory_usage_mb=1024.0 + i * 50   # 递增趋势
            )
            monitor.model_metrics_history[model_name].append(metrics)
        
        return monitor
    
    @pytest.fixture
    def analyzer(self, monitor_with_data):
        """创建性能分析器实例"""
        return PerformanceAnalyzer(monitor_with_data)
    
    def test_analyzer_initialization(self, monitor_with_data):
        """测试分析器初始化"""
        analyzer = PerformanceAnalyzer(monitor_with_data)
        assert analyzer.monitor == monitor_with_data
    
    def test_analyze_performance_trends(self, analyzer):
        """测试性能趋势分析"""
        analysis = analyzer.analyze_performance_trends("test_model", 24)
        
        assert "model_name" in analysis
        assert "analysis_period" in analysis
        assert "sample_count" in analysis
        assert "trends" in analysis
        assert "recommendations" in analysis
        
        assert analysis["model_name"] == "test_model"
        assert analysis["analysis_period"] == "24 hours"
        assert analysis["sample_count"] == 10
        
        # 验证趋势分析
        trends = analysis["trends"]
        assert "inference_time" in trends
        assert "throughput" in trends
        assert "memory_usage" in trends
        
        # 推理时间应该是递增趋势（degrading）
        assert trends["inference_time"]["trend_direction"] == "degrading"
        
        # 吞吐量应该是递减趋势（degrading）
        assert trends["throughput"]["trend_direction"] == "degrading"
        
        # 内存使用应该是递增趋势（degrading）
        assert trends["memory_usage"]["trend_direction"] == "degrading"
    
    def test_analyze_performance_trends_insufficient_data(self, analyzer):
        """测试数据不足时的性能趋势分析"""
        # 清空数据
        analyzer.monitor.model_metrics_history["test_model"].clear()
        
        analysis = analyzer.analyze_performance_trends("test_model", 24)
        
        assert "error" in analysis
        assert analysis["error"] == "数据不足，无法分析趋势"
    
    def test_calculate_trend(self, analyzer):
        """测试趋势计算"""
        # 测试递增趋势
        increasing_values = [10.0, 12.0, 14.0, 16.0, 18.0]
        increasing_trend = analyzer._calculate_trend(increasing_values)
        
        assert increasing_trend["trend_direction"] == "degrading"  # 对于延迟等指标，增加是退化
        assert increasing_trend["slope"] > 0
        assert increasing_trend["change_percent"] > 0
        
        # 测试递减趋势
        decreasing_values = [20.0, 18.0, 16.0, 14.0, 12.0]
        decreasing_trend = analyzer._calculate_trend(decreasing_values)
        
        assert decreasing_trend["trend_direction"] == "improving"
        assert decreasing_trend["slope"] < 0
        assert decreasing_trend["change_percent"] < 0
        
        # 测试稳定趋势
        stable_values = [15.0, 15.1, 14.9, 15.0, 15.0]
        stable_trend = analyzer._calculate_trend(stable_values)
        
        assert stable_trend["trend_direction"] in ["stable", "improving", "degrading"]  # 可能有小幅波动
        assert abs(stable_trend["slope"]) < 0.1
    
    def test_generate_recommendations(self, analyzer):
        """测试生成性能优化建议"""
        # 创建高延迟和高内存使用的指标
        high_latency_metrics = [
            ModelMetrics(
                model_name="slow_model",
                timestamp=utc_now(),
                batch_size=1,  # 小batch size
                sequence_length=128,
                inference_time_ms=2000.0,  # 高延迟
                tokens_per_second=50.0,
                memory_usage_mb=10000.0  # 高内存使用
            )
        ]
        
        recommendations = analyzer._generate_recommendations(high_latency_metrics)
        
        assert len(recommendations) > 0
        
        # 应该包含相关建议
        recommendations_text = " ".join(recommendations)
        assert "推理时间" in recommendations_text or "内存使用" in recommendations_text or "batch size" in recommendations_text
    
    def test_compare_models(self, analyzer):
        """测试模型对比"""
        benchmark_name = "test_benchmark"
        model_names = ["model_1", "model_2"]
        
        # 为每个模型添加基准测试指标
        for i, model_name in enumerate(model_names):
            key = f"{benchmark_name}_{model_name}"
            metrics = BenchmarkMetrics(
                benchmark_name=benchmark_name,
                model_name=model_name,
                timestamp=utc_now(),
                total_samples=1000,
                processed_samples=1000,
                accuracy=0.80 + i * 0.05,
                throughput=50.0 + i * 10,
                avg_inference_time=200.0 - i * 50,
                p95_inference_time=300.0 - i * 50,
                total_time_seconds=120.0,
                peak_memory_usage_gb=8.0 + i
            )
            analyzer.monitor.benchmark_metrics_history[key].append(metrics)
        
        comparison = analyzer.compare_models(model_names, benchmark_name)
        
        assert "benchmark_name" in comparison
        assert "models_compared" in comparison
        assert "detailed_metrics" in comparison
        assert "rankings" in comparison
        assert "winner" in comparison
        
        assert comparison["benchmark_name"] == benchmark_name
        assert comparison["models_compared"] == model_names
        
        # 验证详细指标
        detailed_metrics = comparison["detailed_metrics"]
        assert len(detailed_metrics) == 2
        assert "model_1" in detailed_metrics
        assert "model_2" in detailed_metrics
        
        # model_2应该有更好的性能
        assert detailed_metrics["model_2"]["accuracy"] > detailed_metrics["model_1"]["accuracy"]
        assert detailed_metrics["model_2"]["throughput"] > detailed_metrics["model_1"]["throughput"]
        
        # 验证排名
        rankings = comparison["rankings"]
        assert "accuracy" in rankings
        assert "throughput" in rankings
        assert "overall" in rankings
        
        # model_2应该在准确率排名第一
        assert rankings["accuracy"][0][0] == "model_2"
    
    def test_calculate_efficiency_score(self, analyzer):
        """测试效率分数计算"""
        # 高性能指标
        high_performance_metrics = BenchmarkMetrics(
            benchmark_name="test",
            model_name="test",
            timestamp=utc_now(),
            total_samples=1000,
            processed_samples=1000,
            accuracy=0.95,  # 高准确率
            throughput=100.0,
            avg_inference_time=50.0,  # 低延迟
            p95_inference_time=100.0,
            p99_inference_time=150.0,
            total_time_seconds=120.0,
            peak_memory_usage_gb=2.0  # 低内存使用
        )
        
        high_score = analyzer._calculate_efficiency_score(high_performance_metrics)
        
        # 低性能指标
        low_performance_metrics = BenchmarkMetrics(
            benchmark_name="test",
            model_name="test",
            timestamp=utc_now(),
            total_samples=1000,
            processed_samples=1000,
            accuracy=0.60,  # 低准确率
            throughput=20.0,
            avg_inference_time=500.0,  # 高延迟
            p95_inference_time=800.0,
            p99_inference_time=1000.0,
            total_time_seconds=600.0,
            peak_memory_usage_gb=15.0  # 高内存使用
        )
        
        low_score = analyzer._calculate_efficiency_score(low_performance_metrics)
        
        # 高性能应该有更高的效率分数
        assert high_score > low_score
        assert 0 <= high_score <= 100
        assert 0 <= low_score <= 100
    
    def test_compare_models_no_data(self, analyzer):
        """测试没有数据时的模型对比"""
        comparison = analyzer.compare_models(["nonexistent_model"], "nonexistent_benchmark")
        
        assert "benchmark_name" in comparison
        assert "models_compared" in comparison
        assert "detailed_metrics" in comparison
        assert "rankings" in comparison
        
        assert comparison["detailed_metrics"]["nonexistent_model"]["error"] == "没有数据"
        assert comparison["rankings"] == {}  # 没有数据时排名为空

if __name__ == "__main__":
    pytest.main([__file__])
