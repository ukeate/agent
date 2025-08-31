"""
性能监控器的轻量级测试
避免重依赖项导入，专注测试核心逻辑
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# 模拟重依赖项
sys.modules['torch'] = Mock()
sys.modules['psutil'] = Mock()
sys.modules['matplotlib'] = Mock()
sys.modules['matplotlib.pyplot'] = Mock()
sys.modules['seaborn'] = Mock()

class TestPerformanceMonitorMinimal:
    """性能监控器轻量级测试"""
    
    def setup_method(self):
        """测试前设置"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        self.config = MonitorConfig(
            update_interval=0.5,
            history_size=100,
            alert_cpu_threshold=80.0,
            enable_gpu_monitoring=False  # 简化测试
        )
    
    def test_monitor_config_creation(self):
        """测试监控器配置创建"""
        assert self.config.update_interval == 0.5
        assert self.config.history_size == 100
        assert self.config.alert_cpu_threshold == 80.0
        assert self.config.enable_gpu_monitoring is False
        assert self.config.save_metrics_to_file is True
    
    def test_monitor_config_defaults(self):
        """测试监控器配置默认值"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        default_config = MonitorConfig()
        assert default_config.update_interval == 1.0
        assert default_config.history_size == 1000
        assert default_config.alert_cpu_threshold == 85.0
        assert default_config.alert_memory_threshold == 90.0
        assert default_config.alert_gpu_threshold == 95.0
        assert default_config.enable_gpu_monitoring is True
        assert default_config.metrics_file_path == "logs/performance_metrics.json"
    
    def test_performance_monitor_initialization(self):
        """测试性能监控器初始化"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(self.config)
        
        assert monitor.config == self.config
        assert monitor.sampling_interval == self.config.update_interval
        assert monitor.history_size == self.config.history_size
        assert monitor.is_monitoring is False
        assert len(monitor.system_metrics_history) == 0
        assert len(monitor.alerts) == 0
    
    def test_system_metrics_creation(self):
        """测试系统指标创建"""
        from ai.model_evaluation.performance_monitor import SystemMetrics
        
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=45.2,
            memory_percent=68.5,
            memory_used_gb=8.5,
            memory_total_gb=16.0,
            gpu_percent=75.0,
            gpu_memory_used_gb=3.2,
            gpu_memory_total_gb=8.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 68.5
        assert metrics.gpu_percent == 75.0
        assert metrics.gpu_memory_used_gb == 3.2
    
    def test_model_metrics_creation(self):
        """测试模型指标创建"""
        from ai.model_evaluation.performance_monitor import ModelMetrics
        
        timestamp = datetime.now()
        metrics = ModelMetrics(
            model_name="test_model",
            timestamp=timestamp,
            batch_size=4,
            sequence_length=512,
            inference_time_ms=125.5,
            tokens_per_second=205.3,
            memory_usage_mb=1024.0,
            gpu_utilization=85.2
        )
        
        assert metrics.model_name == "test_model"
        assert metrics.batch_size == 4
        assert metrics.sequence_length == 512
        assert metrics.inference_time_ms == 125.5
        assert metrics.tokens_per_second == 205.3
        assert metrics.gpu_utilization == 85.2
    
    
    def test_add_system_metrics(self):
        """测试添加系统指标"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, SystemMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        timestamp = datetime.now()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=9.6,
            memory_total_gb=16.0
        )
        
        monitor.add_system_metrics(metrics)
        
        assert len(monitor.system_metrics_history) == 1
        assert monitor.system_metrics_history[0] == metrics
    
    
    def test_add_model_metrics(self):
        """测试添加模型指标"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, ModelMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        timestamp = datetime.now()
        metrics = ModelMetrics(
            model_name="gpt_model",
            timestamp=timestamp,
            batch_size=2,
            sequence_length=256,
            inference_time_ms=89.2,
            tokens_per_second=145.7,
            memory_usage_mb=512.0
        )
        
        monitor.add_model_metrics(metrics)
        
        assert len(monitor.model_metrics_history["gpt_model"]) == 1
        assert monitor.model_metrics_history["gpt_model"][0] == metrics
    
    
    def test_history_size_limit(self):
        """测试历史数据大小限制"""
        from ai.model_evaluation.performance_monitor import (
            PerformanceMonitor, SystemMetrics, MonitorConfig
        )
        
        # 使用小的历史大小进行测试
        config = MonitorConfig(history_size=3)
        monitor = PerformanceMonitor(config)
        
        # 添加超过限制的指标
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=float(i * 10),
                memory_percent=float(i * 15),
                memory_used_gb=float(i * 2),
                memory_total_gb=16.0
            )
            monitor.add_system_metrics(metrics)
        
        # 验证只保留最新的3条记录
        assert len(monitor.system_metrics_history) == 3
        assert monitor.system_metrics_history[0].cpu_percent == 20.0  # 第3条记录
        assert monitor.system_metrics_history[-1].cpu_percent == 40.0  # 最后一条记录
    
    
    def test_get_latest_system_metrics(self):
        """测试获取最新系统指标"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, SystemMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        # 没有数据时返回None
        latest = monitor.get_latest_system_metrics()
        assert latest is None
        
        # 添加指标
        timestamp1 = datetime.now()
        metrics1 = SystemMetrics(
            timestamp=timestamp1,
            cpu_percent=30.0,
            memory_percent=40.0,
            memory_used_gb=6.4,
            memory_total_gb=16.0
        )
        monitor.add_system_metrics(metrics1)
        
        timestamp2 = datetime.now() + timedelta(seconds=1)
        metrics2 = SystemMetrics(
            timestamp=timestamp2,
            cpu_percent=35.0,
            memory_percent=45.0,
            memory_used_gb=7.2,
            memory_total_gb=16.0
        )
        monitor.add_system_metrics(metrics2)
        
        # 获取最新指标
        latest = monitor.get_latest_system_metrics()
        assert latest == metrics2
        assert latest.cpu_percent == 35.0
    
    
    def test_get_metrics_summary(self):
        """测试获取指标摘要"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, SystemMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        # 添加多个指标用于计算平均值
        cpu_values = [20.0, 40.0, 60.0, 80.0]
        memory_values = [30.0, 50.0, 70.0, 90.0]
        
        for i, (cpu, memory) in enumerate(zip(cpu_values, memory_values)):
            metrics = SystemMetrics(
                timestamp=datetime.now() + timedelta(seconds=i),
                cpu_percent=cpu,
                memory_percent=memory,
                memory_used_gb=memory * 16.0 / 100.0,
                memory_total_gb=16.0
            )
            monitor.add_system_metrics(metrics)
        
        # 获取摘要
        summary = monitor.get_system_metrics_summary()
        
        assert 'avg_cpu_percent' in summary
        assert 'avg_memory_percent' in summary
        assert 'max_cpu_percent' in summary
        assert 'max_memory_percent' in summary
        assert 'min_cpu_percent' in summary
        assert 'min_memory_percent' in summary
        
        assert summary['avg_cpu_percent'] == 50.0  # (20+40+60+80)/4
        assert summary['avg_memory_percent'] == 60.0  # (30+50+70+90)/4
        assert summary['max_cpu_percent'] == 80.0
        assert summary['min_cpu_percent'] == 20.0
    
    
    def test_alert_generation_cpu_threshold(self):
        """测试CPU阈值告警生成"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, SystemMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        # 添加超过CPU阈值的指标
        high_cpu_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # 超过阈值80.0
            memory_percent=50.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0
        )
        
        monitor.add_system_metrics(high_cpu_metrics)
        monitor._check_and_generate_alerts()
        
        assert len(monitor.alerts) > 0
        cpu_alerts = [alert for alert in monitor.alerts if 'CPU' in alert.get('message', '')]
        assert len(cpu_alerts) > 0
    
    
    def test_alert_generation_memory_threshold(self):
        """测试内存阈值告警生成"""
        from ai.model_evaluation.performance_monitor import (
            PerformanceMonitor, SystemMetrics, MonitorConfig
        )
        
        config = MonitorConfig(alert_memory_threshold=70.0)
        monitor = PerformanceMonitor(config)
        
        # 添加超过内存阈值的指标
        high_memory_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=75.0,  # 超过阈值70.0
            memory_used_gb=12.0,
            memory_total_gb=16.0
        )
        
        monitor.add_system_metrics(high_memory_metrics)
        monitor._check_and_generate_alerts()
        
        assert len(monitor.alerts) > 0
        memory_alerts = [alert for alert in monitor.alerts if '内存' in alert.get('message', '')]
        assert len(memory_alerts) > 0
    
    
    def test_clear_old_metrics(self):
        """测试清理旧指标"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, SystemMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        # 添加旧的指标
        old_timestamp = datetime.now() - timedelta(hours=25)  # 超过24小时
        old_metrics = SystemMetrics(
            timestamp=old_timestamp,
            cpu_percent=30.0,
            memory_percent=40.0,
            memory_used_gb=6.4,
            memory_total_gb=16.0
        )
        monitor.add_system_metrics(old_metrics)
        
        # 添加新的指标
        new_timestamp = datetime.now()
        new_metrics = SystemMetrics(
            timestamp=new_timestamp,
            cpu_percent=35.0,
            memory_percent=45.0,
            memory_used_gb=7.2,
            memory_total_gb=16.0
        )
        monitor.add_system_metrics(new_metrics)
        
        assert len(monitor.system_metrics_history) == 2
        
        # 清理旧指标
        monitor._cleanup_old_metrics()
        
        # 验证只保留新指标
        assert len(monitor.system_metrics_history) == 1
        assert monitor.system_metrics_history[0] == new_metrics
    
    def test_benchmark_metrics_creation(self):
        """测试基准测试指标创建"""
        from ai.model_evaluation.performance_monitor import BenchmarkMetrics
        
        timestamp = datetime.now()
        metrics = BenchmarkMetrics(
            benchmark_name="glue",
            model_name="bert_model",
            timestamp=timestamp,
            task_name="cola",
            accuracy=0.85,
            f1_score=0.82,
            inference_time_ms=45.2,
            samples_processed=1000,
            memory_usage_mb=2048.0
        )
        
        assert metrics.benchmark_name == "glue"
        assert metrics.model_name == "bert_model"
        assert metrics.task_name == "cola"
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.82
        assert metrics.samples_processed == 1000
    
    
    def test_benchmark_metrics_tracking(self):
        """测试基准测试指标跟踪"""
        from ai.model_evaluation.performance_monitor import PerformanceMonitor, BenchmarkMetrics
        
        monitor = PerformanceMonitor(self.config)
        
        timestamp = datetime.now()
        metrics = BenchmarkMetrics(
            benchmark_name="mmlu",
            model_name="gpt_model",
            timestamp=timestamp,
            task_name="abstract_algebra",
            accuracy=0.78,
            inference_time_ms=112.5,
            samples_processed=500,
            memory_usage_mb=1536.0
        )
        
        monitor.add_benchmark_metrics(metrics)
        
        assert len(monitor.benchmark_metrics_history["mmlu"]) == 1
        assert monitor.benchmark_metrics_history["mmlu"][0] == metrics

class TestMonitorConfigValidation:
    """监控器配置验证测试"""
    
    def test_update_interval_validation(self):
        """测试更新间隔验证"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        # 有效的更新间隔
        valid_intervals = [0.1, 0.5, 1.0, 2.0, 5.0]
        for interval in valid_intervals:
            config = MonitorConfig(update_interval=interval)
            assert config.update_interval == interval
    
    def test_threshold_validation(self):
        """测试阈值验证"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        config = MonitorConfig(
            alert_cpu_threshold=75.0,
            alert_memory_threshold=85.0,
            alert_gpu_threshold=90.0,
            alert_disk_threshold=95.0
        )
        
        assert config.alert_cpu_threshold == 75.0
        assert config.alert_memory_threshold == 85.0
        assert config.alert_gpu_threshold == 90.0
        assert config.alert_disk_threshold == 95.0
    
    def test_monitoring_features_configuration(self):
        """测试监控功能配置"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        config = MonitorConfig(
            enable_gpu_monitoring=False,
            enable_network_monitoring=False,
            enable_disk_monitoring=True,
            save_metrics_to_file=False
        )
        
        assert config.enable_gpu_monitoring is False
        assert config.enable_network_monitoring is False
        assert config.enable_disk_monitoring is True
        assert config.save_metrics_to_file is False
    
    def test_cleanup_configuration(self):
        """测试清理配置"""
        from ai.model_evaluation.performance_monitor import MonitorConfig
        
        config = MonitorConfig(
            auto_cleanup_hours=48,
            max_alert_frequency=10
        )
        
        assert config.auto_cleanup_hours == 48
        assert config.max_alert_frequency == 10

if __name__ == "__main__":
    pytest.main([__file__])