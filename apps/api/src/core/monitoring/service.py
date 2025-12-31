"""系统监控和指标收集（服务实现）"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
import json
from src.core.redis import get_redis

logger = get_logger(__name__)

@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.metrics_buffer = deque(maxlen=max_buffer_size)
        self.counters = {}
        self.gauges = {}
        self.histograms = {}
        self.timers = {}
        self._lock = asyncio.Lock()
    
    async def record_counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """记录计数器指标"""
        async with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] = self.counters.get(key, 0) + value
            
            # 添加到缓冲区
            self.metrics_buffer.append(MetricPoint(
                name=f"counter.{name}",
                value=self.counters[key],
                tags=tags or {}
            ))
    
    async def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录仪表指标"""
        async with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            self.metrics_buffer.append(MetricPoint(
                name=f"gauge.{name}",
                value=value,
                tags=tags or {}
            ))
    
    async def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录直方图指标"""
        async with self._lock:
            key = self._make_key(name, tags)
            
            if key not in self.histograms:
                self.histograms[key] = []
            
            self.histograms[key].append(value)
            
            # 保持最近的1000个值
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            self.metrics_buffer.append(MetricPoint(
                name=f"histogram.{name}",
                value=value,
                tags=tags or {}
            ))
    
    async def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """开始计时"""
        timer_id = f"{name}_{time.time()}_{id(tags)}"
        self.timers[timer_id] = {
            "name": name,
            "start_time": time.time(),
            "tags": tags or {}
        }
        return timer_id
    
    async def stop_timer(self, timer_id: str):
        """停止计时并记录"""
        if timer_id in self.timers:
            timer = self.timers.pop(timer_id)
            duration = time.time() - timer["start_time"]
            
            await self.record_histogram(
                f"timer.{timer['name']}",
                duration * 1000,  # 转换为毫秒
                timer["tags"]
            )
            
            return duration
        return None
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """生成指标键"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name},{tag_str}"
    
    async def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        async with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }
            
            # 计算直方图统计
            for key, values in self.histograms.items():
                if values:
                    sorted_values = sorted(values)
                    summary["histograms"][key] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": sorted_values[int(len(values) * 0.5)],
                        "p95": sorted_values[int(len(values) * 0.95)],
                        "p99": sorted_values[int(len(values) * 0.99)]
                    }
            
            return summary
    
    async def export_metrics(self) -> List[MetricPoint]:
        """导出所有指标"""
        async with self._lock:
            return list(self.metrics_buffer)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)
        self.error_counts = {}
        self.active_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    async def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """记录请求"""
        self.total_requests += 1
        
        self.request_times.append({
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # 记录错误
        if status_code >= 400:
            error_key = f"{method}_{endpoint}_{status_code}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    async def increment_active_requests(self):
        """增加活动请求计数"""
        self.active_requests += 1
    
    async def decrement_active_requests(self):
        """减少活动请求计数"""
        self.active_requests = max(0, self.active_requests - 1)
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        uptime = time.time() - self.start_time
        
        # 计算请求率
        recent_requests = [
            r for r in self.request_times 
            if r["timestamp"] > time.time() - 60
        ]
        
        requests_per_minute = len(recent_requests)
        
        # 计算平均响应时间
        if recent_requests:
            avg_duration = sum(r["duration"] for r in recent_requests) / len(recent_requests)
        else:
            avg_duration = 0
        
        # 计算错误率
        recent_errors = sum(
            1 for r in recent_requests 
            if r["status_code"] >= 400
        )
        
        error_rate = recent_errors / len(recent_requests) if recent_requests else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "requests_per_minute": requests_per_minute,
            "average_response_time_ms": avg_duration * 1000,
            "recent_error_count": recent_errors,
            "error_rate": error_rate,
            "error_counts": dict(self.error_counts)
        }

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = []
        self.notification_handlers = []
    
    def add_rule(self, name: str, condition, message: str, severity: str = "warning"):
        """添加告警规则"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "message": message,
            "severity": severity
        })
    
    def add_notification_handler(self, handler):
        """添加通知处理器"""
        self.notification_handlers.append(handler)
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """检查告警条件"""
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    alert = {
                        "name": rule["name"],
                        "message": rule["message"],
                        "severity": rule["severity"],
                        "timestamp": utc_now().isoformat(),
                        "metrics": metrics
                    }
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    
                    # 发送通知
                    await self._send_notifications(alert)
                    
            except Exception as e:
                logger.error("检查告警规则失败", rule_name=rule["name"], error=str(e), exc_info=True)
        
        return new_alerts
    
    async def _send_notifications(self, alert: Dict[str, Any]):
        """发送告警通知"""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error("发送告警通知失败", error=str(e), exc_info=True)
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活动告警"""
        # 只返回最近24小时的告警
        cutoff_time = utc_now() - timedelta(hours=24)
        
        active_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        return active_alerts

class MonitoringService:
    """监控服务"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self._setup_default_alerts()
        self._export_task = None
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        # CPU使用率告警
        self.alert_manager.add_rule(
            name="high_cpu_usage",
            condition=lambda m: m.get("system", {}).get("cpu_percent", 0) > 80,
            message="CPU usage is above 80%",
            severity="warning"
        )
        
        # 内存使用率告警
        self.alert_manager.add_rule(
            name="high_memory_usage",
            condition=lambda m: m.get("system", {}).get("memory_percent", 0) > 85,
            message="Memory usage is above 85%",
            severity="warning"
        )
        
        # 错误率告警
        self.alert_manager.add_rule(
            name="high_error_rate",
            condition=lambda m: m.get("performance", {}).get("error_rate", 0) > 0.05,
            message="Error rate is above 5%",
            severity="critical"
        )
        
        # 响应时间告警
        self.alert_manager.add_rule(
            name="slow_response_time",
            condition=lambda m: m.get("performance", {}).get("average_response_time_ms", 0) > 1000,
            message="Average response time is above 1000ms",
            severity="warning"
        )
    
    async def start(self):
        """启动监控服务"""
        if not self._export_task:
            self._export_task = create_task_with_logging(self._export_metrics_loop())
            logger.info("监控服务已启动")
    
    async def stop(self):
        """停止监控服务"""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                raise
            self._export_task = None
            logger.info("监控服务已停止")
    
    async def _export_metrics_loop(self):
        """定期导出指标"""
        while True:
            try:
                redis = get_redis()
                
                # 收集所有指标
                metrics = await self.collect_all_metrics()
                
                # 检查告警
                await self.alert_manager.check_alerts(metrics)
                
                # 导出到Redis（用于持久化或其他服务消费）
                if redis:
                    await redis.set(
                        "monitoring:metrics:latest",
                        json.dumps(metrics),
                        ex=300  # 5分钟过期
                    )
                
                # 等待下一个周期
                await asyncio.sleep(60)  # 每分钟导出一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("指标导出循环异常", error=str(e), exc_info=True)
                await asyncio.sleep(60)
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """收集所有指标"""
        import psutil

        network_connections = None
        network_connections_error = None
        try:
            network_connections = len(psutil.net_connections())
        except Exception as e:
            network_connections_error = str(e)
        
        # 系统指标
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "network_connections": network_connections,
            "network_connections_error": network_connections_error,
        }
        
        # 性能指标
        performance_metrics = await self.performance_monitor.get_stats()
        
        # 自定义指标
        custom_metrics = await self.metrics_collector.get_summary()
        
        return {
            "timestamp": utc_now().isoformat(),
            "system": system_metrics,
            "performance": performance_metrics,
            "custom": custom_metrics
        }

# 全局监控服务实例
monitoring_service = MonitoringService()
from src.core.logging import get_logger
