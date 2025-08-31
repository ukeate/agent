import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import threading
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class MonitorConfig:
    """性能监控器配置"""
    update_interval: float = 1.0  # 监控更新间隔(秒)
    history_size: int = 1000  # 历史数据保存数量
    alert_cpu_threshold: float = 85.0  # CPU使用率告警阈值(%)
    alert_memory_threshold: float = 90.0  # 内存使用率告警阈值(%)
    alert_gpu_threshold: float = 95.0  # GPU使用率告警阈值(%)
    alert_disk_threshold: float = 90.0  # 磁盘使用率告警阈值(%)
    enable_gpu_monitoring: bool = True  # 启用GPU监控
    enable_network_monitoring: bool = True  # 启用网络监控
    enable_disk_monitoring: bool = True  # 启用磁盘监控
    save_metrics_to_file: bool = True  # 保存指标到文件
    metrics_file_path: str = "logs/performance_metrics.json"
    max_alert_frequency: int = 5  # 相同告警最大频率(分钟)
    auto_cleanup_hours: int = 24  # 自动清理旧数据(小时)
    enable_detailed_logging: bool = False  # 启用详细日志

@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None

@dataclass
class ModelMetrics:
    """模型性能指标"""
    model_name: str
    timestamp: datetime
    batch_size: int
    sequence_length: int
    inference_time_ms: float
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization: Optional[float] = None
    energy_consumption_j: Optional[float] = None
    flops: Optional[int] = None

@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    benchmark_name: str
    model_name: str
    timestamp: datetime
    # 基础字段
    total_samples: int = 0
    processed_samples: int = 0
    samples_processed: int = 0  # 与processed_samples同义
    accuracy: float = 0.0
    f1_score: Optional[float] = None
    throughput: float = 0.0  # samples per second
    
    # 性能字段
    avg_inference_time: float = 0.0
    inference_time_ms: float = 0.0  # 与avg_inference_time同义
    p95_inference_time: float = 0.0
    p99_inference_time: float = 0.0
    total_time_seconds: float = 0.0
    
    # 资源使用
    peak_memory_usage_gb: float = 0.0
    memory_usage_mb: float = 0.0
    avg_gpu_utilization: Optional[float] = None
    
    # 可选字段，支持测试需要
    task_name: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理，同步相关字段"""
        # 同步processed_samples和samples_processed
        if self.samples_processed > 0 and self.processed_samples == 0:
            self.processed_samples = self.samples_processed
        elif self.processed_samples > 0 and self.samples_processed == 0:
            self.samples_processed = self.processed_samples

@dataclass
class PerformanceAlert:
    """性能告警"""
    alert_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "memory", "gpu", "performance", "accuracy"
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    model_name: Optional[str] = None
    benchmark_name: Optional[str] = None
    resolved: bool = False

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.sampling_interval = self.config.update_interval
        self.history_size = self.config.history_size
        self.system_metrics_history = deque(maxlen=self.history_size)
        self.model_metrics_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.benchmark_metrics_history = defaultdict(lambda: deque(maxlen=self.history_size))
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.alerts = []
        self.alert_thresholds = self._get_default_thresholds()
        
        # 性能基线
        self.performance_baselines = {}
        
        # GPU可用性检查
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"GPU monitoring enabled: {self.gpu_count} GPU(s) detected")
        else:
            logger.info("GPU monitoring disabled: No CUDA devices found")
    
    def _get_default_thresholds(self) -> Dict[str, float]:
        """获取默认告警阈值"""
        return {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "gpu_memory_percent": 95.0,
            "gpu_temperature": 85.0,
            "inference_time_ms": 5000.0,
            "accuracy_drop": 0.05,  # 5% accuracy drop
            "throughput_drop": 0.20  # 20% throughput drop
        }
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            logger.warning("监控已在运行")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                self._check_system_alerts(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        memory = psutil.virtual_memory()
        
        metrics = SystemMetrics(
            timestamp=utc_now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3)
        )
        
        # GPU指标
        if self.gpu_available:
            try:
                metrics.gpu_percent = self._get_gpu_utilization()
                gpu_mem_used, gpu_mem_total = self._get_gpu_memory()
                metrics.gpu_memory_used_gb = gpu_mem_used / (1024**3)
                metrics.gpu_memory_total_gb = gpu_mem_total / (1024**3)
                metrics.gpu_temperature = self._get_gpu_temperature()
            except Exception as e:
                logger.debug(f"GPU指标收集失败: {e}")
        
        # IO指标
        try:
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            if disk_io:
                metrics.disk_io_read_mb = disk_io.read_bytes / (1024**2)
                metrics.disk_io_write_mb = disk_io.write_bytes / (1024**2)
            if net_io:
                metrics.network_sent_mb = net_io.bytes_sent / (1024**2)
                metrics.network_recv_mb = net_io.bytes_recv / (1024**2)
        except Exception as e:
            logger.debug(f"IO指标收集失败: {e}")
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        if not self.gpu_available:
            return 0.0
        # 这里可以集成nvidia-ml-py来获取更详细的GPU信息
        return torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
    
    def _get_gpu_memory(self) -> Tuple[int, int]:
        """获取GPU内存使用情况"""
        if not self.gpu_available:
            return 0, 0
        return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """获取GPU温度"""
        # 需要nvidia-ml-py或其他库来获取温度信息
        return None
    
    def record_model_metrics(self, model_name: str, 
                           batch_size: int,
                           sequence_length: int,
                           inference_time_ms: float,
                           memory_usage_mb: float,
                           tokens_processed: int = 0):
        """记录模型性能指标"""
        tokens_per_second = tokens_processed / (inference_time_ms / 1000.0) if inference_time_ms > 0 else 0
        
        metrics = ModelMetrics(
            model_name=model_name,
            timestamp=utc_now(),
            batch_size=batch_size,
            sequence_length=sequence_length,
            inference_time_ms=inference_time_ms,
            tokens_per_second=tokens_per_second,
            memory_usage_mb=memory_usage_mb,
            gpu_utilization=self._get_gpu_utilization() if self.gpu_available else None
        )
        
        self.model_metrics_history[model_name].append(metrics)
        self._check_model_alerts(metrics)
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """添加系统指标到历史记录"""
        self.system_metrics_history.append(metrics)
        self._check_system_alerts(metrics)
    
    def add_model_metrics(self, metrics: ModelMetrics):
        """添加模型指标到历史记录"""
        self.model_metrics_history[metrics.model_name].append(metrics)
        self._check_model_alerts(metrics)
    
    def add_benchmark_metrics(self, metrics: 'BenchmarkMetrics'):
        """添加基准测试指标到历史记录"""
        self.benchmark_metrics_history[metrics.benchmark_name].append(metrics)
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """获取最新的系统指标"""
        return self.system_metrics_history[-1] if self.system_metrics_history else None
    
    def get_system_metrics_summary(self) -> Dict[str, float]:
        """获取系统指标摘要统计"""
        if not self.system_metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.system_metrics_history]
        memory_values = [m.memory_percent for m in self.system_metrics_history]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'min_cpu_percent': min(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'min_memory_percent': min(memory_values)
        }
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统指标告警"""
        alerts = []
        
        if metrics.cpu_percent > self.config.alert_cpu_threshold:
            alerts.append({
                'type': 'system_cpu_high',
                'message': f'CPU使用率过高: {metrics.cpu_percent:.1f}% > {self.config.alert_cpu_threshold}%',
                'timestamp': metrics.timestamp,
                'value': metrics.cpu_percent
            })
        
        if metrics.memory_percent > self.config.alert_memory_threshold:
            alerts.append({
                'type': 'system_memory_high',
                'message': f'内存使用率过高: {metrics.memory_percent:.1f}% > {self.config.alert_memory_threshold}%',
                'timestamp': metrics.timestamp,
                'value': metrics.memory_percent
            })
        
        self.alerts.extend(alerts)
    
    def _check_and_generate_alerts(self):
        """检查并生成告警"""
        if self.system_metrics_history:
            latest_metrics = self.get_latest_system_metrics()
            if latest_metrics:
                self._check_system_alerts(latest_metrics)
    
    def _cleanup_old_metrics(self):
        """清理旧的指标数据"""
        cutoff_time = utc_now() - timedelta(hours=self.config.auto_cleanup_hours)
        
        # 清理系统指标
        self.system_metrics_history = deque(
            [m for m in self.system_metrics_history if m.timestamp > cutoff_time],
            maxlen=self.history_size
        )
        
        # 清理模型指标
        for model_name in self.model_metrics_history:
            self.model_metrics_history[model_name] = deque(
                [m for m in self.model_metrics_history[model_name] if m.timestamp > cutoff_time],
                maxlen=self.history_size
            )
    
    def record_benchmark_metrics(self, benchmark_name: str,
                                model_name: str,
                                total_samples: int,
                                processed_samples: int, 
                                accuracy: float,
                                total_time_seconds: float):
        """记录基准测试指标"""
        throughput = processed_samples / total_time_seconds if total_time_seconds > 0 else 0
        
        # 计算推理时间统计
        model_history = self.model_metrics_history.get(model_name, [])
        recent_times = [m.inference_time_ms for m in list(model_history)[-100:]]  # 最近100次
        
        avg_inference_time = np.mean(recent_times) if recent_times else 0
        p95_inference_time = np.percentile(recent_times, 95) if recent_times else 0
        p99_inference_time = np.percentile(recent_times, 99) if recent_times else 0
        
        # 获取峰值内存使用
        peak_memory = 0
        if self.system_metrics_history:
            recent_system_metrics = list(self.system_metrics_history)[-100:]
            peak_memory = max(m.memory_used_gb for m in recent_system_metrics)
        
        metrics = BenchmarkMetrics(
            benchmark_name=benchmark_name,
            model_name=model_name,
            timestamp=utc_now(),
            total_samples=total_samples,
            processed_samples=processed_samples,
            accuracy=accuracy,
            throughput=throughput,
            avg_inference_time=avg_inference_time,
            p95_inference_time=p95_inference_time,
            p99_inference_time=p99_inference_time,
            total_time_seconds=total_time_seconds,
            peak_memory_usage_gb=peak_memory,
            avg_gpu_utilization=self._get_gpu_utilization() if self.gpu_available else None
        )
        
        self.benchmark_metrics_history[f"{benchmark_name}_{model_name}"].append(metrics)
        self._check_benchmark_alerts(metrics)
    
    
    def _check_model_alerts(self, metrics: ModelMetrics):
        """检查模型性能告警"""
        if metrics.inference_time_ms > self.alert_thresholds.get("inference_time_ms", float('inf')):
            self._create_alert(
                severity="medium",
                category="performance", 
                title="推理时间过长",
                description=f"模型 {metrics.model_name} 推理时间: {metrics.inference_time_ms:.0f}ms",
                metric_name="inference_time_ms",
                current_value=metrics.inference_time_ms,
                threshold_value=self.alert_thresholds["inference_time_ms"],
                model_name=metrics.model_name
            )
    
    def _check_benchmark_alerts(self, metrics: BenchmarkMetrics):
        """检查基准测试告警"""
        # 检查准确率下降
        baseline_key = f"{metrics.benchmark_name}_{metrics.model_name}"
        if baseline_key in self.performance_baselines:
            baseline_accuracy = self.performance_baselines[baseline_key].get("accuracy", 0)
            accuracy_drop = baseline_accuracy - metrics.accuracy
            
            if accuracy_drop > self.alert_thresholds.get("accuracy_drop", float('inf')):
                self._create_alert(
                    severity="high",
                    category="accuracy",
                    title="准确率显著下降",
                    description=f"基准测试 {metrics.benchmark_name} 准确率下降 {accuracy_drop:.1%}",
                    metric_name="accuracy",
                    current_value=metrics.accuracy,
                    threshold_value=baseline_accuracy - self.alert_thresholds["accuracy_drop"],
                    model_name=metrics.model_name,
                    benchmark_name=metrics.benchmark_name
                )
    
    def _create_alert(self, severity: str, category: str, title: str, 
                     description: str, metric_name: str, current_value: float,
                     threshold_value: float, model_name: Optional[str] = None,
                     benchmark_name: Optional[str] = None):
        """创建告警"""
        alert_id = f"{category}_{metric_name}_{utc_now().timestamp()}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            timestamp=utc_now(),
            severity=severity,
            category=category,
            title=title,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            model_name=model_name,
            benchmark_name=benchmark_name
        )
        
        self.alerts.append(alert)
        logger.warning(f"性能告警: {title} - {description}")
        
        # 保持告警列表大小
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]  # 保留最近500个告警
    
    
    def _get_gpu_summary(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """获取GPU指标摘要"""
        gpu_metrics = [m for m in metrics if m.gpu_percent is not None]
        if not gpu_metrics:
            return {"error": "没有GPU指标数据"}
        
        return {
            "utilization": {
                "avg": np.mean([m.gpu_percent for m in gpu_metrics]),
                "max": max(m.gpu_percent for m in gpu_metrics)
            },
            "memory": {
                "avg_used_gb": np.mean([m.gpu_memory_used_gb for m in gpu_metrics if m.gpu_memory_used_gb]),
                "max_used_gb": max(m.gpu_memory_used_gb for m in gpu_metrics if m.gpu_memory_used_gb),
                "total_gb": gpu_metrics[0].gpu_memory_total_gb if gpu_metrics[0].gpu_memory_total_gb else 0
            }
        }
    
    def get_model_performance_summary(self, model_name: str, 
                                    time_range_minutes: int = 60) -> Dict[str, Any]:
        """获取模型性能摘要"""
        cutoff_time = utc_now() - timedelta(minutes=time_range_minutes)
        model_metrics = [m for m in self.model_metrics_history[model_name] 
                        if m.timestamp > cutoff_time]
        
        if not model_metrics:
            return {"error": f"没有模型 {model_name} 的性能数据"}
        
        inference_times = [m.inference_time_ms for m in model_metrics]
        throughputs = [m.tokens_per_second for m in model_metrics if m.tokens_per_second > 0]
        
        return {
            "model_name": model_name,
            "time_range_minutes": time_range_minutes,
            "total_inferences": len(model_metrics),
            "inference_time_ms": {
                "avg": np.mean(inference_times),
                "median": np.median(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
                "min": min(inference_times),
                "max": max(inference_times)
            },
            "throughput": {
                "avg_tokens_per_second": np.mean(throughputs) if throughputs else 0,
                "max_tokens_per_second": max(throughputs) if throughputs else 0
            },
            "memory_usage_mb": {
                "avg": np.mean([m.memory_usage_mb for m in model_metrics]),
                "max": max(m.memory_usage_mb for m in model_metrics)
            }
        }
    
    def get_benchmark_comparison(self, benchmark_name: str, 
                               model_names: List[str]) -> Dict[str, Any]:
        """获取基准测试对比"""
        comparison = {
            "benchmark_name": benchmark_name,
            "models": {},
            "comparison_timestamp": utc_now()
        }
        
        for model_name in model_names:
            key = f"{benchmark_name}_{model_name}"
            metrics_list = list(self.benchmark_metrics_history[key])
            
            if metrics_list:
                latest_metrics = metrics_list[-1]  # 获取最新指标
                comparison["models"][model_name] = {
                    "accuracy": latest_metrics.accuracy,
                    "throughput": latest_metrics.throughput,
                    "avg_inference_time": latest_metrics.avg_inference_time,
                    "p95_inference_time": latest_metrics.p95_inference_time,
                    "peak_memory_usage_gb": latest_metrics.peak_memory_usage_gb,
                    "total_time_seconds": latest_metrics.total_time_seconds,
                    "timestamp": latest_metrics.timestamp
                }
            else:
                comparison["models"][model_name] = {"error": "没有数据"}
        
        return comparison
    
    def set_performance_baseline(self, model_name: str, benchmark_name: str, 
                               metrics: BenchmarkMetrics):
        """设置性能基线"""
        baseline_key = f"{benchmark_name}_{model_name}"
        self.performance_baselines[baseline_key] = {
            "accuracy": metrics.accuracy,
            "throughput": metrics.throughput,
            "avg_inference_time": metrics.avg_inference_time,
            "timestamp": metrics.timestamp
        }
        logger.info(f"设置性能基线: {baseline_key}")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """获取活跃告警"""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if severity:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"告警已解决: {alert_id}")
                break
    
    def export_metrics(self, output_path: str, time_range_hours: int = 24):
        """导出指标数据"""
        cutoff_time = utc_now() - timedelta(hours=time_range_hours)
        
        export_data = {
            "export_timestamp": utc_now().isoformat(),
            "time_range_hours": time_range_hours,
            "system_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_gb": m.memory_used_gb,
                    "gpu_percent": m.gpu_percent,
                    "gpu_memory_used_gb": m.gpu_memory_used_gb
                }
                for m in self.system_metrics_history
                if m.timestamp > cutoff_time
            ],
            "model_metrics": {},
            "benchmark_metrics": {},
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "category": alert.category,
                    "title": alert.title,
                    "description": alert.description,
                    "resolved": alert.resolved
                }
                for alert in self.alerts
                if alert.timestamp > cutoff_time
            ]
        }
        
        # 导出模型指标
        for model_name, metrics_list in self.model_metrics_history.items():
            recent_metrics = [m for m in metrics_list if m.timestamp > cutoff_time]
            export_data["model_metrics"][model_name] = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "batch_size": m.batch_size,
                    "sequence_length": m.sequence_length,
                    "inference_time_ms": m.inference_time_ms,
                    "tokens_per_second": m.tokens_per_second,
                    "memory_usage_mb": m.memory_usage_mb
                }
                for m in recent_metrics
            ]
        
        # 导出基准测试指标
        for key, metrics_list in self.benchmark_metrics_history.items():
            recent_metrics = [m for m in metrics_list if m.timestamp > cutoff_time]
            export_data["benchmark_metrics"][key] = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_samples": m.total_samples,
                    "accuracy": m.accuracy,
                    "throughput": m.throughput,
                    "total_time_seconds": m.total_time_seconds
                }
                for m in recent_metrics
            ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"指标数据已导出到: {output_path}")

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    def analyze_performance_trends(self, model_name: str, 
                                 time_range_hours: int = 24) -> Dict[str, Any]:
        """分析性能趋势"""
        cutoff_time = utc_now() - timedelta(hours=time_range_hours)
        model_metrics = [m for m in self.monitor.model_metrics_history[model_name]
                        if m.timestamp > cutoff_time]
        
        if len(model_metrics) < 2:
            return {"error": "数据不足，无法分析趋势"}
        
        # 按时间排序
        model_metrics.sort(key=lambda x: x.timestamp)
        
        # 分析趋势
        times = [m.inference_time_ms for m in model_metrics]
        throughputs = [m.tokens_per_second for m in model_metrics if m.tokens_per_second > 0]
        memory_usage = [m.memory_usage_mb for m in model_metrics]
        
        return {
            "model_name": model_name,
            "analysis_period": f"{time_range_hours} hours",
            "sample_count": len(model_metrics),
            "trends": {
                "inference_time": self._calculate_trend(times),
                "throughput": self._calculate_trend(throughputs) if throughputs else None,
                "memory_usage": self._calculate_trend(memory_usage)
            },
            "recommendations": self._generate_recommendations(model_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """计算趋势"""
        if len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # 使用简单线性回归计算趋势
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        # 计算变化百分比
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half != 0 else 0
        
        trend_direction = "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
        
        return {
            "trend_direction": trend_direction,
            "slope": slope,
            "change_percent": change_percent,
            "current_avg": np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
        }
    
    def _generate_recommendations(self, metrics: List) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        recent_metrics = metrics[-20:] if len(metrics) >= 20 else metrics
        avg_inference_time = np.mean([m.inference_time_ms for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        
        if avg_inference_time > 1000:  # >1秒
            recommendations.append("推理时间较长，建议优化模型或使用更快的硬件")
        
        if avg_memory > 8000:  # >8GB
            recommendations.append("内存使用过高，考虑模型压缩或量化")
        
        # 检查batch size优化空间
        batch_sizes = [m.batch_size for m in recent_metrics]
        if len(set(batch_sizes)) == 1 and batch_sizes[0] == 1:
            recommendations.append("当前使用batch size为1，考虑增加batch size以提高吞吐量")
        
        if not recommendations:
            recommendations.append("当前性能表现良好，无需特殊优化")
        
        return recommendations
    
    def compare_models(self, model_names: List[str], 
                      benchmark_name: str) -> Dict[str, Any]:
        """对比多个模型的性能"""
        comparison_data = {}
        
        for model_name in model_names:
            key = f"{benchmark_name}_{model_name}"
            metrics_list = list(self.monitor.benchmark_metrics_history[key])
            
            if metrics_list:
                latest = metrics_list[-1]
                comparison_data[model_name] = {
                    "accuracy": latest.accuracy,
                    "throughput": latest.throughput,
                    "avg_inference_time": latest.avg_inference_time,
                    "peak_memory_gb": latest.peak_memory_usage_gb,
                    "efficiency_score": self._calculate_efficiency_score(latest)
                }
        
        # 排名
        if comparison_data:
            rankings = {
                "accuracy": sorted(comparison_data.items(), key=lambda x: x[1]["accuracy"], reverse=True),
                "throughput": sorted(comparison_data.items(), key=lambda x: x[1]["throughput"], reverse=True),
                "speed": sorted(comparison_data.items(), key=lambda x: x[1]["avg_inference_time"]),
                "memory_efficiency": sorted(comparison_data.items(), key=lambda x: x[1]["peak_memory_gb"]),
                "overall": sorted(comparison_data.items(), key=lambda x: x[1]["efficiency_score"], reverse=True)
            }
        else:
            rankings = {}
        
        return {
            "benchmark_name": benchmark_name,
            "models_compared": model_names,
            "detailed_metrics": comparison_data,
            "rankings": rankings,
            "winner": rankings.get("overall", [("unknown", {})])[0][0] if rankings else None
        }
    
    def _calculate_efficiency_score(self, metrics: BenchmarkMetrics) -> float:
        """计算效率分数"""
        # 综合考虑准确率、速度和内存使用的效率分数
        accuracy_score = metrics.accuracy * 100  # 0-100
        speed_score = max(0, 100 - (metrics.avg_inference_time / 100))  # 越快分数越高
        memory_score = max(0, 100 - (metrics.peak_memory_usage_gb * 10))  # 内存使用越少分数越高
        
        # 加权平均 (准确率权重最高)
        efficiency_score = (accuracy_score * 0.5 + speed_score * 0.3 + memory_score * 0.2)
        return max(0, min(100, efficiency_score))  # 限制在0-100范围内