"""监控和分析系统"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import threading
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)

class AlertSeverity(str, Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    
    def update_latency_percentiles(self, latencies: List[float]):
        """更新延迟百分位数"""
        if latencies:
            latencies.sort()
            self.p50_latency_ms = np.percentile(latencies, 50)
            self.p95_latency_ms = np.percentile(latencies, 95)
            self.p99_latency_ms = np.percentile(latencies, 99)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.RLock()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """记录指标"""
        with self.lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
                metric_type=metric_type
            )
            self.metrics[name].append(point)
    
    def get_metrics(self, name: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[MetricPoint]:
        """获取指标数据"""
        with self.lock:
            points = list(self.metrics.get(name, []))
        
        if start_time or end_time:
            filtered_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
            return filtered_points
        
        return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """获取最新值"""
        with self.lock:
            points = self.metrics.get(name)
            if points:
                return points[-1].value
        return None
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=window_minutes)
        points = self.get_metrics(name, start_time, end_time)
        
        if not points:
            return {"name": name, "count": 0}
        
        values = [p.value for p in points]
        
        return {
            "name": name,
            "count": len(values),
            "latest": values[-1] if values else None,
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
    
    def clear_old_metrics(self, retention_hours: int = 24):
        """清理旧指标数据"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        
        with self.lock:
            for name, points in self.metrics.items():
                # 从左侧移除旧数据点
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # 标准差的倍数
        self.metric_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
    
    def add_data_point(self, metric_name: str, value: float) -> bool:
        """添加数据点并检测异常"""
        window = self.metric_windows[metric_name]
        window.append(value)
        
        # 需要足够的历史数据才能检测异常
        if len(window) < self.window_size // 2:
            return False
        
        # 更新基线统计
        values = list(window)
        self.baseline_stats[metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }
        
        # 检测异常
        stats = self.baseline_stats[metric_name]
        if stats["std"] > 0:
            z_score = abs((value - stats["mean"]) / stats["std"])
            return z_score > self.sensitivity
        
        return False
    
    def detect_anomalies_batch(self, metric_name: str, values: List[float]) -> List[bool]:
        """批量检测异常"""
        results = []
        for value in values:
            is_anomaly = self.add_data_point(metric_name, value)
            results.append(is_anomaly)
        return results
    
    def get_anomaly_threshold(self, metric_name: str) -> Optional[Tuple[float, float]]:
        """获取异常阈值范围"""
        if metric_name not in self.baseline_stats:
            return None
        
        stats = self.baseline_stats[metric_name]
        if stats["std"] == 0:
            return None
        
        mean = stats["mean"]
        std = stats["std"]
        threshold = self.sensitivity * std
        
        return (mean - threshold, mean + threshold)

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """添加告警规则"""
        required_fields = ["name", "metric", "condition", "threshold", "severity"]
        if not all(field in rule for field in required_fields):
            raise ValueError(f"告警规则缺少必要字段: {required_fields}")
        
        with self.lock:
            self.alert_rules.append(rule)
        
        logger.info(f"添加告警规则: {rule['name']}")
    
    def evaluate_rules(self, metrics_collector: MetricsCollector):
        """评估告警规则"""
        with self.lock:
            for rule in self.alert_rules:
                try:
                    self._evaluate_single_rule(rule, metrics_collector)
                except Exception as e:
                    logger.error(f"评估告警规则失败 {rule['name']}: {e}")
    
    def _evaluate_single_rule(self, rule: Dict[str, Any], metrics_collector: MetricsCollector):
        """评估单个告警规则"""
        metric_name = rule["metric"]
        condition = rule["condition"]  # "gt", "lt", "eq", "ne"
        threshold = rule["threshold"]
        severity = AlertSeverity(rule["severity"])
        
        # 获取最新指标值
        latest_value = metrics_collector.get_latest_value(metric_name)
        if latest_value is None:
            return
        
        # 检查条件
        triggered = False
        if condition == "gt" and latest_value > threshold:
            triggered = True
        elif condition == "lt" and latest_value < threshold:
            triggered = True
        elif condition == "eq" and abs(latest_value - threshold) < 1e-6:
            triggered = True
        elif condition == "ne" and abs(latest_value - threshold) >= 1e-6:
            triggered = True
        
        alert_key = f"{rule['name']}:{metric_name}"
        
        if triggered:
            if alert_key not in self.alerts or self.alerts[alert_key].resolved:
                # 创建新告警
                alert = Alert(
                    alert_id=f"{alert_key}:{int(time.time())}",
                    name=rule["name"],
                    description=rule.get("description", f"{metric_name} {condition} {threshold}"),
                    severity=severity,
                    metric_name=metric_name,
                    threshold_value=threshold,
                    actual_value=latest_value,
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.alerts[alert_key] = alert
                self._trigger_alert(alert)
        else:
            # 解除告警
            if alert_key in self.alerts and not self.alerts[alert_key].resolved:
                alert = self.alerts[alert_key]
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                self._resolve_alert(alert)
    
    def _trigger_alert(self, alert: Alert):
        """触发告警"""
        logger.warning(f"告警触发: {alert.name} - {alert.description}")
        
        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")
    
    def _resolve_alert(self, alert: Alert):
        """解除告警"""
        logger.info(f"告警解除: {alert.name}")
        
        # 调用回调函数（由回调根据resolved状态处理）
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"解除告警回调失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self, hours: int = 24) -> List[Alert]:
        """获取所有告警"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        with self.lock:
            return [alert for alert in self.alerts.values() 
                   if alert.timestamp >= cutoff_time]

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.last_collection_time = time.time()
    
    def collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric("system.memory_percent", memory.percent)
            self.metrics_collector.record_metric("system.memory_used_gb", memory.used / (1024**3))
            self.metrics_collector.record_metric("system.memory_available_gb", memory.available / (1024**3))
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric("system.disk_percent", disk.percent)
            self.metrics_collector.record_metric("system.disk_used_gb", disk.used / (1024**3))
            self.metrics_collector.record_metric("system.disk_free_gb", disk.free / (1024**3))
            
            # 网络IO（如果可用）
            try:
                net_io = psutil.net_io_counters()
                current_time = time.time()
                time_delta = current_time - self.last_collection_time
                
                if hasattr(self, 'last_bytes_sent'):
                    bytes_sent_rate = (net_io.bytes_sent - self.last_bytes_sent) / time_delta
                    bytes_recv_rate = (net_io.bytes_recv - self.last_bytes_recv) / time_delta
                    
                    self.metrics_collector.record_metric("system.network_sent_mbps", bytes_sent_rate / (1024**2))
                    self.metrics_collector.record_metric("system.network_recv_mbps", bytes_recv_rate / (1024**2))
                
                self.last_bytes_sent = net_io.bytes_sent
                self.last_bytes_recv = net_io.bytes_recv
                self.last_collection_time = current_time
                
            except Exception as e:
                logger.debug(f"网络指标收集失败: {e}")
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def collect_gpu_metrics(self):
        """收集GPU指标"""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU使用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.metrics_collector.record_metric(
                    "gpu.utilization_percent", 
                    util.gpu, 
                    labels={"device": str(i)}
                )
                
                # 显存使用情况
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.metrics_collector.record_metric(
                    "gpu.memory_used_gb", 
                    mem_info.used / (1024**3),
                    labels={"device": str(i)}
                )
                self.metrics_collector.record_metric(
                    "gpu.memory_total_gb", 
                    mem_info.total / (1024**3),
                    labels={"device": str(i)}
                )
                
                # GPU温度
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.metrics_collector.record_metric(
                        "gpu.temperature_celsius", 
                        temp,
                        labels={"device": str(i)}
                    )
                except Exception:
                    logger.exception("获取GPU温度失败", exc_info=True)
            
        except ImportError:
            logger.debug("pynvml不可用，无法收集GPU指标")
        except Exception as e:
            logger.debug(f"收集GPU指标失败: {e}")

class ModelPerformanceMonitor:
    """模型性能监控器"""
    
    def __init__(self):
        self.performance_data: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.latency_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.RLock()
    
    def record_inference(
        self, 
        model_key: str, 
        latency_ms: float, 
        success: bool = True,
        batch_size: int = 1
    ):
        """记录推理性能"""
        with self.lock:
            metrics = self.performance_data[model_key]
            latency_buffer = self.latency_buffers[model_key]
            
            # 更新基本计数
            metrics.request_count += batch_size
            if success:
                metrics.success_count += batch_size
            else:
                metrics.error_count += batch_size
            
            # 更新延迟统计
            metrics.total_latency_ms += latency_ms
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            
            # 添加到延迟缓冲区
            latency_buffer.append(latency_ms)
            
            # 更新百分位数（每100次请求更新一次）
            if metrics.request_count % 100 == 0:
                metrics.update_latency_percentiles(list(latency_buffer))
            
            # 更新错误率
            if metrics.request_count > 0:
                metrics.error_rate = metrics.error_count / metrics.request_count
    
    def calculate_throughput(self, model_key: str, window_seconds: int = 60) -> float:
        """计算吞吐量"""
        # 这里简化实现，实际应该基于时间窗口计算
        with self.lock:
            metrics = self.performance_data.get(model_key)
            if not metrics or metrics.request_count == 0:
                return 0.0
            
            # 假设平均分布在时间窗口内
            return metrics.request_count / window_seconds
    
    def get_model_metrics(self, model_key: str) -> Optional[Dict[str, Any]]:
        """获取模型指标"""
        with self.lock:
            if model_key not in self.performance_data:
                return None
            
            metrics = self.performance_data[model_key]
            
            return {
                "model_key": model_key,
                "request_count": metrics.request_count,
                "success_count": metrics.success_count,
                "error_count": metrics.error_count,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.total_latency_ms / max(metrics.request_count, 1),
                "min_latency_ms": metrics.min_latency_ms if metrics.min_latency_ms != float('inf') else 0,
                "max_latency_ms": metrics.max_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "throughput_qps": self.calculate_throughput(model_key)
            }
    
    def get_all_model_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型指标"""
        result = {}
        with self.lock:
            for model_key in self.performance_data:
                metrics = self.get_model_metrics(model_key)
                if metrics:
                    result[model_key] = metrics
        return result

class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, storage_path: str = "/tmp/monitoring"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 组件初始化
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.resource_monitor = ResourceMonitor()
        self.model_monitor = ModelPerformanceMonitor()
        
        # 后台任务控制
        self._running = False
        self._background_task = None
        
        # 设置默认告警规则
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        default_rules = [
            {
                "name": "High CPU Usage",
                "metric": "system.cpu_percent",
                "condition": "gt",
                "threshold": 90.0,
                "severity": "warning",
                "description": "CPU使用率超过90%"
            },
            {
                "name": "High Memory Usage",
                "metric": "system.memory_percent",
                "condition": "gt",
                "threshold": 85.0,
                "severity": "warning",
                "description": "内存使用率超过85%"
            },
            {
                "name": "High Error Rate",
                "metric": "model.error_rate",
                "condition": "gt",
                "threshold": 0.05,
                "severity": "error",
                "description": "模型错误率超过5%"
            },
            {
                "name": "High Latency",
                "metric": "model.p95_latency_ms",
                "condition": "gt",
                "threshold": 1000.0,
                "severity": "warning",
                "description": "P95延迟超过1000ms"
            }
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    async def start_monitoring(self, collection_interval: int = 60):
        """开始监控"""
        if self._running:
            return
        
        self._running = True
        self._background_task = create_task_with_logging(
            self._monitoring_loop(collection_interval)
        )
        
        logger.info("监控系统已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                raise
        
        logger.info("监控系统已停止")
    
    async def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self._running:
            try:
                # 收集系统指标
                self.resource_monitor.collect_system_metrics()
                self.resource_monitor.collect_gpu_metrics()
                
                # 从资源监控器复制指标到主收集器
                for name, points in self.resource_monitor.metrics_collector.metrics.items():
                    if points:
                        latest_point = points[-1]
                        self.metrics_collector.record_metric(
                            latest_point.name,
                            latest_point.value,
                            latest_point.labels,
                            latest_point.metric_type
                        )
                
                # 异常检测
                self._run_anomaly_detection()
                
                # 评估告警规则
                self.alert_manager.evaluate_rules(self.metrics_collector)
                
                # 清理旧数据
                self.metrics_collector.clear_old_metrics()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(interval)
    
    def _run_anomaly_detection(self):
        """运行异常检测"""
        # 对关键指标进行异常检测
        critical_metrics = [
            "system.cpu_percent",
            "system.memory_percent", 
            "model.p95_latency_ms",
            "model.error_rate"
        ]
        
        for metric_name in critical_metrics:
            latest_value = self.metrics_collector.get_latest_value(metric_name)
            if latest_value is not None:
                is_anomaly = self.anomaly_detector.add_data_point(metric_name, latest_value)
                
                if is_anomaly:
                    # 记录异常指标
                    self.metrics_collector.record_metric(
                        f"{metric_name}.anomaly",
                        1.0,
                        metric_type=MetricType.COUNTER
                    )
                    
                    logger.warning(f"检测到异常: {metric_name}={latest_value}")
    
    def record_model_inference(
        self, 
        model_key: str,
        latency_ms: float,
        success: bool = True,
        batch_size: int = 1
    ):
        """记录模型推理性能"""
        # 记录到模型性能监控器
        self.model_monitor.record_inference(model_key, latency_ms, success, batch_size)
        
        # 记录到指标收集器
        self.metrics_collector.record_metric(
            "model.latency_ms",
            latency_ms,
            labels={"model": model_key}
        )
        
        if success:
            self.metrics_collector.record_metric(
                "model.success_count",
                1,
                labels={"model": model_key},
                metric_type=MetricType.COUNTER
            )
        else:
            self.metrics_collector.record_metric(
                "model.error_count", 
                1,
                labels={"model": model_key},
                metric_type=MetricType.COUNTER
            )
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        overview = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_metrics": {},
            "model_metrics": {},
            "alerts": {
                "active_count": len(self.alert_manager.get_active_alerts()),
                "total_count": len(self.alert_manager.get_all_alerts())
            },
            "anomalies": {}
        }
        
        # 系统指标摘要
        system_metrics = [
            "system.cpu_percent",
            "system.memory_percent", 
            "system.disk_percent"
        ]
        
        for metric in system_metrics:
            summary = self.metrics_collector.get_metric_summary(metric, window_minutes=15)
            if summary["count"] > 0:
                overview["system_metrics"][metric] = {
                    "current": summary["latest"],
                    "avg": summary["avg"],
                    "max": summary["max"]
                }
        
        # 模型性能摘要
        overview["model_metrics"] = self.model_monitor.get_all_model_metrics()
        
        # 异常检测阈值
        for metric in system_metrics:
            threshold = self.anomaly_detector.get_anomaly_threshold(metric)
            if threshold:
                overview["anomalies"][metric] = {
                    "lower_bound": threshold[0],
                    "upper_bound": threshold[1]
                }
        
        return overview
    
    def get_metrics_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """获取仪表板数据"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        dashboard = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "metrics": {},
            "alerts": self.alert_manager.get_all_alerts(hours),
            "performance": self.model_monitor.get_all_model_metrics()
        }
        
        # 获取时间序列数据
        key_metrics = [
            "system.cpu_percent",
            "system.memory_percent",
            "model.latency_ms",
            "model.error_rate"
        ]
        
        for metric in key_metrics:
            points = self.metrics_collector.get_metrics(metric, start_time, end_time)
            dashboard["metrics"][metric] = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "value": p.value,
                    "labels": p.labels
                }
                for p in points
            ]
        
        return dashboard
    
    def add_custom_alert_rule(self, rule: Dict[str, Any]):
        """添加自定义告警规则"""
        self.alert_manager.add_alert_rule(rule)
    
    def get_resource_recommendations(self) -> List[Dict[str, Any]]:
        """获取资源优化建议"""
        recommendations = []
        
        # CPU建议
        cpu_summary = self.metrics_collector.get_metric_summary("system.cpu_percent", 60)
        if cpu_summary["count"] > 0 and cpu_summary["avg"] > 80:
            recommendations.append({
                "type": "cpu",
                "severity": "warning",
                "message": f"CPU平均使用率{cpu_summary['avg']:.1f}%，建议增加CPU资源或优化模型",
                "current_value": cpu_summary["avg"],
                "threshold": 80
            })
        
        # 内存建议
        memory_summary = self.metrics_collector.get_metric_summary("system.memory_percent", 60)
        if memory_summary["count"] > 0 and memory_summary["avg"] > 75:
            recommendations.append({
                "type": "memory", 
                "severity": "warning",
                "message": f"内存平均使用率{memory_summary['avg']:.1f}%，建议增加内存或优化模型加载",
                "current_value": memory_summary["avg"],
                "threshold": 75
            })
        
        # 模型性能建议
        for model_key, metrics in self.model_monitor.get_all_model_metrics().items():
            if metrics["error_rate"] > 0.03:  # 3%错误率
                recommendations.append({
                    "type": "model_error",
                    "severity": "error", 
                    "message": f"模型{model_key}错误率{metrics['error_rate']:.1%}过高，需要检查模型或数据质量",
                    "model": model_key,
                    "current_value": metrics["error_rate"],
                    "threshold": 0.03
                })
            
            if metrics["p95_latency_ms"] > 500:  # P95延迟超过500ms
                recommendations.append({
                    "type": "model_latency",
                    "severity": "warning",
                    "message": f"模型{model_key} P95延迟{metrics['p95_latency_ms']:.0f}ms过高，建议优化模型或增加资源",
                    "model": model_key,
                    "current_value": metrics["p95_latency_ms"],
                    "threshold": 500
                })
        
        return recommendations
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        health_status = "healthy"
        if critical_alerts:
            health_status = "critical"
        elif len(active_alerts) > 5:
            health_status = "warning"
        
        return {
            "status": health_status,
            "monitoring_active": self._running,
            "total_metrics": sum(len(deque_obj) for deque_obj in self.metrics_collector.metrics.values()),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "monitored_models": len(self.model_monitor.performance_data)
        }
from src.core.logging import get_logger

from src.core.utils.async_utils import create_task_with_logging
