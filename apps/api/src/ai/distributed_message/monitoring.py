"""
分布式消息系统监控和性能优化
实现性能指标监控、健康检查、告警系统和性能优化功能
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import time
import statistics
import psutil
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from src.core.logging import get_logger
logger = get_logger(__name__)

class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"         # 仪表盘
    HISTOGRAM = "histogram" # 直方图
    TIMER = "timer"         # 计时器

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }

@dataclass
class Alert:
    """告警"""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve(self):
        """解决告警"""
        self.resolved = True
        self.resolved_at = utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }

@dataclass
class HealthStatus:
    """健康状态"""
    component: str
    healthy: bool
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "component": self.component,
            "healthy": self.healthy,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }

class MetricCollector:
    """指标收集器"""
    
    def __init__(self, retention_period: timedelta = timedelta(hours=24)):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_period = retention_period
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        
    def record_metric(self, metric: PerformanceMetric):
        """记录指标"""
        metric_key = f"{metric.name}:{','.join(f'{k}={v}' for k, v in metric.labels.items())}"
        
        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric_key] += metric.value
            metric.value = self.counters[metric_key]
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric_key] = metric.value
        
        self.metrics[metric_key].append(metric)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """设置仪表盘值"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """记录计时器"""
        metric = PerformanceMetric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            labels=labels or {}
        )
        self.record_metric(metric)
    
    def get_metric_history(self, name: str, labels: Optional[Dict[str, str]] = None) -> List[PerformanceMetric]:
        """获取指标历史"""
        metric_key = f"{name}:{','.join(f'{k}={v}' for k, v in (labels or {}).items())}"
        return list(self.metrics.get(metric_key, []))
    
    def get_metric_statistics(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """获取指标统计信息"""
        history = self.get_metric_history(name, labels)
        if not history:
            return {}
        
        values = [m.value for m in history]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def cleanup_old_metrics(self):
        """清理过期指标"""
        cutoff_time = utc_now() - self.retention_period
        
        for metric_key in list(self.metrics.keys()):
            metrics_deque = self.metrics[metric_key]
            
            # 从左侧移除过期的指标
            while metrics_deque and metrics_deque[0].timestamp < cutoff_time:
                metrics_deque.popleft()
            
            # 如果队列为空，删除键
            if not metrics_deque:
                del self.metrics[metric_key]

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.last_check_results: Dict[str, HealthStatus] = {}
        
    def register_health_check(self, component: str, check_function: Callable[[], HealthStatus]):
        """注册健康检查"""
        self.health_checks[component] = check_function
        logger.info(f"注册健康检查: {component}")
    
    def unregister_health_check(self, component: str):
        """取消注册健康检查"""
        if component in self.health_checks:
            del self.health_checks[component]
            if component in self.last_check_results:
                del self.last_check_results[component]
            logger.info(f"取消注册健康检查: {component}")
    
    async def run_health_check(self, component: str) -> Optional[HealthStatus]:
        """运行单个健康检查"""
        if component not in self.health_checks:
            return None
        
        try:
            check_function = self.health_checks[component]
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            self.last_check_results[component] = result
            return result
            
        except Exception as e:
            logger.error(f"健康检查 {component} 执行失败: {e}")
            error_status = HealthStatus(
                component=component,
                healthy=False,
                message=f"健康检查执行失败: {str(e)}"
            )
            self.last_check_results[component] = error_status
            return error_status
    
    async def run_all_health_checks(self) -> Dict[str, HealthStatus]:
        """运行所有健康检查"""
        results = {}
        
        for component in self.health_checks:
            result = await self.run_health_check(component)
            if result:
                results[component] = result
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """获取整体健康状态"""
        if not self.last_check_results:
            return HealthStatus(
                component="system",
                healthy=True,
                message="无健康检查项"
            )
        
        unhealthy_components = [
            comp for comp, status in self.last_check_results.items() 
            if not status.healthy
        ]
        
        if not unhealthy_components:
            return HealthStatus(
                component="system",
                healthy=True,
                message="所有组件健康",
                details={"total_components": len(self.last_check_results)}
            )
        else:
            return HealthStatus(
                component="system",
                healthy=False,
                message=f"存在不健康的组件: {', '.join(unhealthy_components)}",
                details={
                    "total_components": len(self.last_check_results),
                    "unhealthy_components": unhealthy_components
                }
            )

class AlertManager:
    """告警管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.max_alerts = max_alerts
        
        # 告警规则
        self.rules: Dict[str, Dict[str, Any]] = {}
        
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """注册告警处理器"""
        self.alert_handlers.append(handler)
        logger.info("注册告警处理器")
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        message_template: str = "指标 {metric_name} {condition} {threshold}"
    ):
        """添加告警规则"""
        self.rules[rule_name] = {
            "metric_name": metric_name,
            "condition": condition,  # "gt", "lt", "eq", "gte", "lte"
            "threshold": threshold,
            "level": level,
            "message_template": message_template
        }
        logger.info(f"添加告警规则: {rule_name}")
    
    def create_alert(
        self,
        name: str,
        level: AlertLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """创建告警"""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=name,
            level=level,
            message=message,
            metadata=metadata or {}
        )
        
        # 限制告警数量
        if len(self.alerts) >= self.max_alerts:
            # 删除最老的已解决告警
            oldest_resolved = None
            for alert_id, old_alert in self.alerts.items():
                if old_alert.resolved:
                    if oldest_resolved is None or old_alert.timestamp < oldest_resolved[1].timestamp:
                        oldest_resolved = (alert_id, old_alert)
            
            if oldest_resolved:
                del self.alerts[oldest_resolved[0]]
            else:
                # 如果没有已解决的告警，删除最老的告警
                oldest = min(self.alerts.items(), key=lambda x: x[1].timestamp)
                del self.alerts[oldest[0]]
        
        self.alerts[alert.alert_id] = alert
        
        # 通知告警处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
        
        logger.warning(f"创建告警: {name} - {message}")
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolve()
            logger.info(f"解决告警: {alert_id}")
            return True
        return False
    
    def check_metric_alerts(self, metric: PerformanceMetric):
        """检查指标告警"""
        for rule_name, rule in self.rules.items():
            if rule["metric_name"] == metric.name:
                threshold = rule["threshold"]
                condition = rule["condition"]
                
                triggered = False
                if condition == "gt" and metric.value > threshold:
                    triggered = True
                elif condition == "lt" and metric.value < threshold:
                    triggered = True
                elif condition == "gte" and metric.value >= threshold:
                    triggered = True
                elif condition == "lte" and metric.value <= threshold:
                    triggered = True
                elif condition == "eq" and metric.value == threshold:
                    triggered = True
                
                if triggered:
                    message = rule["message_template"].format(
                        metric_name=metric.name,
                        condition=condition,
                        threshold=threshold,
                        value=metric.value
                    )
                    
                    self.create_alert(
                        name=rule_name,
                        level=rule["level"],
                        message=message,
                        metadata={
                            "metric_name": metric.name,
                            "metric_value": metric.value,
                            "threshold": threshold,
                            "condition": condition,
                            "labels": metric.labels
                        }
                    )
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        active_alerts = self.get_active_alerts()
        level_counts = defaultdict(int)
        
        # 统计所有告警的级别分布（不仅仅是活跃告警）
        for alert in self.alerts.values():
            level_counts[alert.level.value.lower()] += 1
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(self.alerts) - len(active_alerts),
            "by_level": dict(level_counts)
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.connection_pools: Dict[str, Any] = {}
        self.message_batches: Dict[str, List[Any]] = defaultdict(list)
        self.batch_timers: Dict[str, float] = {}
        self.compression_enabled = False
        self.compression_threshold = 1024
        self.batching_enabled = False
        self.batch_size = 100
        self.batch_timeout = 1.0  # 秒
        
    def enable_compression(self, threshold: int = 1024):
        """启用消息压缩"""
        self.compression_enabled = True
        self.compression_threshold = threshold
        logger.info(f"启用消息压缩，阈值: {threshold} 字节")
    
    def disable_compression(self):
        """禁用消息压缩"""
        self.compression_enabled = False
        logger.info("禁用消息压缩")
    
    def enable_batching(self, batch_size: int = 10, timeout: float = 1.0):
        """启用批处理"""
        self.batching_enabled = True
        self.batch_size = batch_size
        self.batch_timeout = timeout
        logger.info(f"启用批处理，大小: {batch_size}, 超时: {timeout}s")
    
    def disable_batching(self):
        """禁用批处理"""
        self.batching_enabled = False
        logger.info("禁用批处理")
    
    def compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        if not self.compression_enabled:
            return data
        
        try:
            import gzip
            return gzip.compress(data)
        except Exception as e:
            logger.error(f"数据压缩失败: {e}")
            return data
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """解压缩数据"""
        if not self.compression_enabled:
            return compressed_data
        
        try:
            import gzip
            return gzip.decompress(compressed_data)
        except Exception as e:
            logger.error(f"数据解压缩失败: {e}")
            return compressed_data
    
    def add_to_batch(self, batch_key: str, item: Any) -> bool:
        """添加到批处理队列"""
        self.message_batches[batch_key].append(item)
        
        # 设置批处理计时器
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = time.time()
        
        # 检查是否需要立即处理批次
        batch = self.message_batches[batch_key]
        if len(batch) >= self.batch_size:
            return True  # 需要立即处理
        
        # 检查超时
        if time.time() - self.batch_timers[batch_key] >= self.batch_timeout:
            return True  # 需要立即处理
        
        return False  # 继续收集
    
    def get_and_clear_batch(self, batch_key: str) -> List[Any]:
        """获取并清空批处理队列"""
        batch = self.message_batches[batch_key].copy()
        self.message_batches[batch_key].clear()
        if batch_key in self.batch_timers:
            del self.batch_timers[batch_key]
        return batch
    
    def get_system_performance(self) -> Dict[str, Any]:
        """获取系统性能信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
        except Exception as e:
            logger.error(f"获取系统性能信息失败: {e}")
            return {}

class MonitoringManager:
    """监控管理器"""
    
    def __init__(self, check_interval: float = 30.0):
        self.metric_collector = MetricCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.performance_optimizer = PerformanceOptimizer()
        
        self.check_interval = check_interval
        self.is_running = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # 注册默认健康检查
        self._register_default_health_checks()
        
        # 注册默认告警规则
        self._register_default_alert_rules()
    
    def _register_default_health_checks(self):
        """注册默认健康检查"""
        def system_health_check() -> HealthStatus:
            try:
                perf = self.performance_optimizer.get_system_performance()
                
                # 检查CPU使用率
                cpu_threshold = 90.0
                memory_threshold = 85.0
                disk_threshold = 90.0
                
                issues = []
                if perf.get("cpu_percent", 0) > cpu_threshold:
                    issues.append(f"CPU使用率过高: {perf['cpu_percent']:.1f}%")
                
                if perf.get("memory", {}).get("percent", 0) > memory_threshold:
                    issues.append(f"内存使用率过高: {perf['memory']['percent']:.1f}%")
                
                if perf.get("disk", {}).get("percent", 0) > disk_threshold:
                    issues.append(f"磁盘使用率过高: {perf['disk']['percent']:.1f}%")
                
                if issues:
                    return HealthStatus(
                        component="system_resources",
                        healthy=False,
                        message="; ".join(issues),
                        details=perf
                    )
                else:
                    return HealthStatus(
                        component="system_resources",
                        healthy=True,
                        message="系统资源正常",
                        details=perf
                    )
                    
            except Exception as e:
                return HealthStatus(
                    component="system_resources",
                    healthy=False,
                    message=f"系统资源检查失败: {str(e)}"
                )
        
        self.health_checker.register_health_check("system_resources", system_health_check)
    
    def _register_default_alert_rules(self):
        """注册默认告警规则"""
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "system.cpu_percent",
            "gt",
            80.0,
            AlertLevel.WARNING,
            "CPU使用率过高: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "system.memory_percent",
            "gt",
            85.0,
            AlertLevel.WARNING,
            "内存使用率过高: {value:.1f}% > {threshold}%"
        )
        
        self.alert_manager.add_alert_rule(
            "message_error_rate_high",
            "messages.error_rate",
            "gt",
            0.05,
            AlertLevel.ERROR,
            "消息错误率过高: {value:.3f} > {threshold}"
        )
    
    async def start(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动监控任务
        self.monitoring_tasks.append(
            asyncio.create_task(self._monitoring_loop())
        )
        self.monitoring_tasks.append(
            asyncio.create_task(self._cleanup_loop())
        )
        
        logger.info("监控管理器已启动")
    
    async def stop(self):
        """停止监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止所有监控任务
        for task in self.monitoring_tasks:
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("监控管理器已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 运行健康检查
                await self.health_checker.run_all_health_checks()
                
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 等待下一次检查
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                # 清理过期指标
                self.metric_collector.cleanup_old_metrics()
                
                # 每小时清理一次
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                await asyncio.sleep(300)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            perf = self.performance_optimizer.get_system_performance()
            
            if perf:
                # 记录CPU指标
                cpu_metric = PerformanceMetric(
                    name="system.cpu_percent",
                    value=perf["cpu_percent"],
                    metric_type=MetricType.GAUGE
                )
                self.metric_collector.record_metric(cpu_metric)
                self.alert_manager.check_metric_alerts(cpu_metric)
                
                # 记录内存指标
                memory_metric = PerformanceMetric(
                    name="system.memory_percent",
                    value=perf["memory"]["percent"],
                    metric_type=MetricType.GAUGE
                )
                self.metric_collector.record_metric(memory_metric)
                self.alert_manager.check_metric_alerts(memory_metric)
                
                # 记录磁盘指标
                disk_metric = PerformanceMetric(
                    name="system.disk_percent",
                    value=perf["disk"]["percent"],
                    metric_type=MetricType.GAUGE
                )
                self.metric_collector.record_metric(disk_metric)
                
                # 记录网络指标
                self.metric_collector.set_gauge("system.network.bytes_sent", perf["network"]["bytes_sent"])
                self.metric_collector.set_gauge("system.network.bytes_recv", perf["network"]["bytes_recv"])
                
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def record_message_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录消息相关指标"""
        if metric_name.endswith("_count"):
            self.metric_collector.increment_counter(metric_name, value, labels)
        elif metric_name.endswith("_duration"):
            self.metric_collector.record_timer(metric_name, value, labels)
        else:
            self.metric_collector.set_gauge(metric_name, value, labels)
        
        # 检查告警
        metric = PerformanceMetric(
            name=metric_name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        self.alert_manager.check_metric_alerts(metric)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        return {
            "health": {
                "overall_health": self.health_checker.get_overall_health().healthy,
                "overall": self.health_checker.get_overall_health().to_dict(),
                "checks": [
                    {**status.to_dict(), "name": comp} 
                    for comp, status in self.health_checker.last_check_results.items()
                ]
            },
            "alerts": {
                "summary": self.alert_manager.get_alert_summary(),
                "active": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()]
            },
            "performance": {
                **self.performance_optimizer.get_system_performance(),
                "compression_enabled": self.performance_optimizer.compression_enabled,
                "batching_enabled": self.performance_optimizer.batching_enabled,
                "batch_size": self.performance_optimizer.batch_size,
                "batch_timeout": self.performance_optimizer.batch_timeout
            },
            "metrics": self._get_all_metrics(),
            "metrics_summary": {
                "total_metrics": len(self.metric_collector.metrics),
                "recent_metrics": self._get_recent_metrics_summary()
            }
        }
    
    def _get_recent_metrics_summary(self) -> Dict[str, Any]:
        """获取最近指标摘要"""
        summary = {}
        
        # 获取最近5分钟的指标
        recent_cutoff = utc_now() - timedelta(minutes=5)
        
        for metric_key, metrics_deque in self.metric_collector.metrics.items():
            recent_metrics = [m for m in metrics_deque if m.timestamp >= recent_cutoff]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary[metric_key] = {
                    "count": len(values),
                    "latest": values[-1] if values else 0,
                    "avg": sum(values) / len(values) if values else 0
                }
        
        return summary
    
    def register_alert_handler(self, handler: Callable[[Any], None]) -> None:
        """注册告警处理器"""
        self.alert_manager.register_alert_handler(handler)
    
    def _get_all_metrics(self) -> List[PerformanceMetric]:
        """获取所有指标"""
        all_metrics = []
        for metrics_deque in self.metric_collector.metrics.values():
            all_metrics.extend(metrics_deque)
        return sorted(all_metrics, key=lambda x: x.timestamp, reverse=True)
