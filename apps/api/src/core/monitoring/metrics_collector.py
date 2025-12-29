"""
指标收集器
"""

from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics

from src.core.logging import get_logger
logger = get_logger(__name__)

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"  # 摘要

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricStats:
    """指标统计"""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, window_size: int = 1000, retention_hours: int = 24):
        self.metrics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))
        self.window_size = window_size
        self.retention_hours = retention_hours
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def start(self):
        """启动收集器"""
        self._cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
    
    async def stop(self):
        """停止收集器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def record_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """记录计数器"""
        async with self._lock:
            label_key = self._labels_to_key(labels)
            point = MetricPoint(utc_now(), value, labels or {})
            self.metrics[name][label_key].append(point)
    
    async def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录仪表"""
        async with self._lock:
            label_key = self._labels_to_key(labels)
            point = MetricPoint(utc_now(), value, labels or {})
            self.metrics[name][label_key].append(point)
    
    async def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录直方图"""
        async with self._lock:
            label_key = self._labels_to_key(labels)
            point = MetricPoint(utc_now(), value, labels or {})
            self.metrics[name][label_key].append(point)
    
    async def record_timing(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """记录时间"""
        await self.record_histogram(f"{name}_duration_seconds", duration, labels)

    def increment(self, name: str, value: float = 1, tags: Optional[Dict[str, Any]] = None):
        labels = {k: str(v) for k, v in (tags or {}).items()}
        try:
            asyncio.get_running_loop().create_task(self.record_counter(name, value, labels))
        except RuntimeError:
            return

    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, Any]] = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            labels = {k: str(v) for k, v in (tags or {}).items()}
            try:
                asyncio.get_running_loop().create_task(self.record_timing(name, duration, labels))
            except RuntimeError:
                logger.debug("事件循环不可用，跳过计时记录", exc_info=True)
    
    async def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[MetricPoint]:
        """获取指标"""
        async with self._lock:
            label_key = self._labels_to_key(labels)
            return list(self.metrics.get(name, {}).get(label_key, []))
    
    async def get_metric_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[MetricStats]:
        """获取指标统计"""
        points = await self.get_metric(name, labels)
        if not points:
            return None
        
        values = [p.value for p in points]
        sorted_values = sorted(values)
        
        return MetricStats(
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            p95=sorted_values[int(len(sorted_values) * 0.95)] if values else 0,
            p99=sorted_values[int(len(sorted_values) * 0.99)] if values else 0
        )
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        async with self._lock:
            result = {}
            for metric_name, label_data in self.metrics.items():
                result[metric_name] = {}
                for label_key, points in label_data.items():
                    if points:
                        values = [p.value for p in points]
                        result[metric_name][label_key] = {
                            "count": len(values),
                            "last": values[-1],
                            "mean": statistics.mean(values) if values else 0,
                            "min": min(values) if values else 0,
                            "max": max(values) if values else 0
                        }
            return result
    
    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """标签转换为键"""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    async def _cleanup_old_metrics(self):
        """清理旧指标"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                cutoff_time = utc_now() - timedelta(hours=self.retention_hours)
                
                async with self._lock:
                    for metric_name in list(self.metrics.keys()):
                        for label_key in list(self.metrics[metric_name].keys()):
                            points = self.metrics[metric_name][label_key]
                            # 过滤掉旧数据点
                            filtered_points = deque(
                                (p for p in points if p.timestamp > cutoff_time),
                                maxlen=self.window_size
                            )
                            if filtered_points:
                                self.metrics[metric_name][label_key] = filtered_points
                            else:
                                del self.metrics[metric_name][label_key]
                        
                        # 如果指标没有数据，删除它
                        if not self.metrics[metric_name]:
                            del self.metrics[metric_name]
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("清理指标时出错", error=str(e), exc_info=True)

class RequestMetrics:
    """请求指标"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        experiment_id: Optional[str] = None
    ):
        """记录请求"""
        labels = {
            "method": method,
            "path": path,
            "status": str(status_code),
        }
        if experiment_id:
            labels["experiment_id"] = experiment_id
        
        # 请求计数
        await self.collector.record_counter("http_requests_total", 1, labels)
        
        # 请求延迟
        await self.collector.record_histogram("http_request_duration_seconds", duration, labels)
        
        # 错误率
        if status_code >= 400:
            await self.collector.record_counter("http_errors_total", 1, labels)

class ExperimentMetrics:
    """实验指标"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def record_experiment_created(self, experiment_id: str):
        """记录实验创建"""
        await self.collector.record_counter(
            "experiments_created_total",
            1,
            {"experiment_id": experiment_id}
        )
    
    async def record_experiment_started(self, experiment_id: str):
        """记录实验启动"""
        await self.collector.record_counter(
            "experiments_started_total",
            1,
            {"experiment_id": experiment_id}
        )
    
    async def record_experiment_completed(self, experiment_id: str):
        """记录实验完成"""
        await self.collector.record_counter(
            "experiments_completed_total",
            1,
            {"experiment_id": experiment_id}
        )
    
    async def record_variant_assignment(self, experiment_id: str, variant_id: str):
        """记录变体分配"""
        await self.collector.record_counter(
            "variant_assignments_total",
            1,
            {"experiment_id": experiment_id, "variant_id": variant_id}
        )
    
    async def record_event(self, experiment_id: str, event_type: str):
        """记录事件"""
        await self.collector.record_counter(
            "experiment_events_total",
            1,
            {"experiment_id": experiment_id, "event_type": event_type}
        )
    
    async def record_conversion(self, experiment_id: str, variant_id: str, value: float):
        """记录转化"""
        await self.collector.record_histogram(
            "experiment_conversions",
            value,
            {"experiment_id": experiment_id, "variant_id": variant_id}
        )

class SystemMetrics:
    """系统指标"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._last_cpu_time = None
        self._last_check_time = None
    
    async def collect_system_metrics(self):
        """收集系统指标"""
        import psutil
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        await self.collector.record_gauge("system_cpu_usage_percent", cpu_percent)
        
        # 内存使用
        memory = psutil.virtual_memory()
        await self.collector.record_gauge("system_memory_usage_percent", memory.percent)
        await self.collector.record_gauge("system_memory_used_bytes", memory.used)
        await self.collector.record_gauge("system_memory_available_bytes", memory.available)
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        await self.collector.record_gauge("system_disk_usage_percent", disk.percent)
        await self.collector.record_gauge("system_disk_used_bytes", disk.used)
        await self.collector.record_gauge("system_disk_free_bytes", disk.free)
        
        # 网络IO
        net_io = psutil.net_io_counters()
        await self.collector.record_counter("system_network_bytes_sent", net_io.bytes_sent)
        await self.collector.record_counter("system_network_bytes_recv", net_io.bytes_recv)
        await self.collector.record_counter("system_network_packets_sent", net_io.packets_sent)
        await self.collector.record_counter("system_network_packets_recv", net_io.packets_recv)
    
    async def start_monitoring(self, interval: int = 60):
        """开始监控"""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("收集系统指标时出错", error=str(e), exc_info=True)
                await asyncio.sleep(interval)

class DatabaseMetrics:
    """数据库指标"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def record_query(self, query_type: str, duration: float, success: bool):
        """记录查询"""
        labels = {
            "query_type": query_type,
            "status": "success" if success else "error"
        }
        
        await self.collector.record_counter("database_queries_total", 1, labels)
        await self.collector.record_histogram("database_query_duration_seconds", duration, labels)
    
    async def record_connection_pool(self, active: int, idle: int, total: int):
        """记录连接池状态"""
        await self.collector.record_gauge("database_connections_active", active)
        await self.collector.record_gauge("database_connections_idle", idle)
        await self.collector.record_gauge("database_connections_total", total)
    
    async def record_transaction(self, duration: float, success: bool):
        """记录事务"""
        labels = {"status": "success" if success else "error"}
        await self.collector.record_counter("database_transactions_total", 1, labels)
        await self.collector.record_histogram("database_transaction_duration_seconds", duration, labels)

# 全局指标收集器实例
metrics_collector = MetricsCollector()
request_metrics = RequestMetrics(metrics_collector)
experiment_metrics = ExperimentMetrics(metrics_collector)
system_metrics = SystemMetrics(metrics_collector)
database_metrics = DatabaseMetrics(metrics_collector)

async def init_metrics():
    """初始化指标收集"""
    await metrics_collector.start()
    # 启动系统监控
    asyncio.create_task(system_metrics.start_monitoring())

async def cleanup_metrics():
    """清理指标收集"""
    await metrics_collector.stop()
