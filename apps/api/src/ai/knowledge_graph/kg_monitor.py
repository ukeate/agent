"""
知识图谱监控和日志系统 - 性能监控、健康检查、日志记录和告警
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import logging.handlers
from pathlib import Path

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    Counter = Histogram = Gauge = CollectorRegistry = None

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertLevel(Enum):
    """告警级别"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    timestamp: datetime = None
    description: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = utc_now()
        if self.labels is None:
            self.labels = {}


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: str
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    condition: str
    threshold: float
    level: AlertLevel
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    callback: Optional[Callable] = None


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    rule_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = time.time()
        self.last_gc_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Prometheus指标（如果可用）
        if HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self.request_counter = Counter(
                'kg_requests_total', 
                'Total requests', 
                ['method', 'endpoint'],
                registry=self.registry
            )
            self.request_duration = Histogram(
                'kg_request_duration_seconds',
                'Request duration',
                ['method', 'endpoint'],
                registry=self.registry
            )
            self.memory_usage = Gauge(
                'kg_memory_usage_bytes',
                'Memory usage in bytes',
                registry=self.registry
            )
            self.active_connections = Gauge(
                'kg_active_connections',
                'Active connections',
                registry=self.registry
            )
    
    def record_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """记录请求"""
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
        
        # 记录到内存
        self.metrics['request_duration'].append({
            'timestamp': time.time(),
            'method': method,
            'endpoint': endpoint,
            'duration': duration,
            'status_code': status_code
        })
        
        # 记录到Prometheus
        if HAS_PROMETHEUS:
            self.request_counter.labels(method=method, endpoint=endpoint).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_query(self, query_type: str, duration: float, result_count: int):
        """记录查询"""
        self.metrics['query_performance'].append({
            'timestamp': time.time(),
            'query_type': query_type,
            'duration': duration,
            'result_count': result_count
        })
    
    def record_memory_usage(self, usage_bytes: int):
        """记录内存使用"""
        self.metrics['memory_usage'].append({
            'timestamp': time.time(),
            'usage_bytes': usage_bytes
        })
        
        if HAS_PROMETHEUS:
            self.memory_usage.set(usage_bytes)
    
    def record_system_metrics(self):
        """记录系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent
            })
            
            # 内存使用
            memory = psutil.virtual_memory()
            self.record_memory_usage(memory.used)
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].append({
                'timestamp': time.time(),
                'disk_percent': disk.percent,
                'disk_used': disk.used,
                'disk_free': disk.free
            })
            
            # 网络IO
            network = psutil.net_io_counters()
            self.metrics['network_io'].append({
                'timestamp': time.time(),
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            })
            
        except Exception as e:
            logger.warning(f"记录系统指标失败: {e}")
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """获取指标摘要"""
        cutoff_time = time.time() - (minutes * 60)
        
        summary = {
            'uptime_seconds': time.time() - self.start_time,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'timestamp': utc_now().isoformat()
        }
        
        # 请求性能
        recent_requests = [
            m for m in self.metrics['request_duration']
            if m['timestamp'] > cutoff_time
        ]
        
        if recent_requests:
            durations = [r['duration'] for r in recent_requests]
            summary['request_metrics'] = {
                'count': len(recent_requests),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'requests_per_minute': len(recent_requests) / minutes
            }
        
        # 查询性能
        recent_queries = [
            m for m in self.metrics['query_performance']
            if m['timestamp'] > cutoff_time
        ]
        
        if recent_queries:
            query_durations = [q['duration'] for q in recent_queries]
            summary['query_metrics'] = {
                'count': len(recent_queries),
                'avg_duration': sum(query_durations) / len(query_durations),
                'avg_result_count': sum(q['result_count'] for q in recent_queries) / len(recent_queries)
            }
        
        # 系统资源
        recent_cpu = [
            m for m in self.metrics['cpu_usage']
            if m['timestamp'] > cutoff_time
        ]
        
        if recent_cpu:
            cpu_values = [c['cpu_percent'] for c in recent_cpu]
            summary['system_metrics'] = {
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
                'max_cpu_percent': max(cpu_values)
            }
        
        recent_memory = [
            m for m in self.metrics['memory_usage']
            if m['timestamp'] > cutoff_time
        ]
        
        if recent_memory:
            memory_values = [m['usage_bytes'] for m in recent_memory]
            summary['system_metrics'] = summary.get('system_metrics', {})
            summary['system_metrics'].update({
                'avg_memory_bytes': sum(memory_values) / len(memory_values),
                'max_memory_bytes': max(memory_values)
            })
        
        return summary


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """注册健康检查"""
        self.checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """运行单个健康检查"""
        if name not in self.checks:
            return HealthCheckResult(
                component=name,
                status="ERROR",
                message="检查不存在",
                timestamp=utc_now(),
                duration_ms=0
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = result.get('status', 'OK')
                message = result.get('message', '健康')
                metadata = result.get('metadata')
            else:
                status = 'OK' if result else 'ERROR'
                message = '健康' if result else '检查失败'
                metadata = None
            
            health_result = HealthCheckResult(
                component=name,
                status=status,
                message=message,
                timestamp=utc_now(),
                duration_ms=duration,
                metadata=metadata
            )
            
            self.last_results[name] = health_result
            return health_result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            error_result = HealthCheckResult(
                component=name,
                status="ERROR",
                message=f"检查异常: {str(e)}",
                timestamp=utc_now(),
                duration_ms=duration
            )
            
            self.last_results[name] = error_result
            return error_result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查"""
        tasks = [self.run_check(name) for name in self.checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        check_results = {}
        for name, result in zip(self.checks.keys(), results):
            if isinstance(result, Exception):
                check_results[name] = HealthCheckResult(
                    component=name,
                    status="ERROR",
                    message=f"检查异常: {str(result)}",
                    timestamp=utc_now(),
                    duration_ms=0
                )
            else:
                check_results[name] = result
        
        return check_results
    
    def get_overall_status(self) -> str:
        """获取整体健康状态"""
        if not self.last_results:
            return "UNKNOWN"
        
        statuses = [result.status for result in self.last_results.values()]
        
        if any(status == "ERROR" for status in statuses):
            return "ERROR"
        elif any(status == "WARNING" for status in statuses):
            return "WARNING"
        else:
            return "OK"


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
    
    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    async def check_rules(self, metrics: Dict[str, Any]):
        """检查告警规则"""
        current_time = utc_now()
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 检查冷却时间
            if (rule.last_triggered and 
                (current_time - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60):
                continue
            
            try:
                if await self._evaluate_condition(rule, metrics):
                    await self._trigger_alert(rule, current_time)
            except Exception as e:
                logger.error(f"评估告警规则 {rule.rule_id} 失败: {e}")
    
    async def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """评估告警条件"""
        # 简化的条件评估实现
        # 实际项目中可以使用更复杂的表达式引擎
        
        condition = rule.condition
        threshold = rule.threshold
        
        if "error_rate" in condition and "error_rate" in metrics:
            return metrics["error_rate"] > threshold
        elif "memory_usage" in condition and "system_metrics" in metrics:
            memory_bytes = metrics["system_metrics"].get("avg_memory_bytes", 0)
            # 假设阈值是MB
            return memory_bytes > threshold * 1024 * 1024
        elif "cpu_usage" in condition and "system_metrics" in metrics:
            cpu_percent = metrics["system_metrics"].get("avg_cpu_percent", 0)
            return cpu_percent > threshold
        elif "response_time" in condition and "request_metrics" in metrics:
            avg_duration = metrics["request_metrics"].get("avg_duration", 0)
            return avg_duration > threshold
        
        return False
    
    async def _trigger_alert(self, rule: AlertRule, timestamp: datetime):
        """触发告警"""
        alert_id = f"alert_{int(timestamp.timestamp())}_{rule.rule_id}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            level=rule.level,
            message=f"告警规则 '{rule.name}' 被触发: {rule.condition} > {rule.threshold}",
            timestamp=timestamp
        )
        
        self.alerts[alert_id] = alert
        rule.last_triggered = timestamp
        
        # 调用告警处理器
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
        
        # 调用规则回调
        if rule.callback:
            try:
                if asyncio.iscoroutinefunction(rule.callback):
                    await rule.callback(alert)
                else:
                    rule.callback(alert)
            except Exception as e:
                logger.error(f"告警规则回调执行失败: {e}")
        
        logger.warning(f"告警触发: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolved_at = utc_now()
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts.values() if not alert.resolved]


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str, log_dir: str = "/tmp/kg_logs", max_file_size_mb: int = 100):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=5
        )
        
        # JSON格式化器
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_event(self, level: LogLevel, event_type: str, message: str, **kwargs):
        """记录结构化事件"""
        log_data = {
            "event_type": event_type,
            "message": message,
            **kwargs
        }
        
        log_message = json.dumps(log_data, ensure_ascii=False, default=str)
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
    
    def log_query(self, query: str, duration: float, result_count: int, error: str = None):
        """记录查询日志"""
        self.log_event(
            LogLevel.INFO if not error else LogLevel.ERROR,
            "sparql_query",
            f"SPARQL查询{'成功' if not error else '失败'}",
            query=query,
            duration_ms=duration * 1000,
            result_count=result_count,
            error=error
        )
    
    def log_operation(self, operation: str, resource: str, user_id: str, success: bool, **kwargs):
        """记录操作日志"""
        self.log_event(
            LogLevel.INFO if success else LogLevel.ERROR,
            "kg_operation",
            f"{operation} {resource} {'成功' if success else '失败'}",
            operation=operation,
            resource=resource,
            user_id=user_id,
            success=success,
            **kwargs
        )


class KnowledgeGraphMonitor:
    """知识图谱监控系统"""
    
    def __init__(self, log_dir: str = "/tmp/kg_logs"):
        self.performance_monitor = PerformanceMonitor()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.logger = StructuredLogger("kg_monitor", log_dir)
        
        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # 设置默认健康检查
        self._setup_default_health_checks()
        
        # 设置默认告警规则
        self._setup_default_alert_rules()
    
    def _setup_default_health_checks(self):
        """设置默认健康检查"""
        
        def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {"status": "ERROR", "message": f"内存使用率过高: {memory.percent}%"}
            elif memory.percent > 80:
                return {"status": "WARNING", "message": f"内存使用率较高: {memory.percent}%"}
            else:
                return {"status": "OK", "message": f"内存使用率正常: {memory.percent}%"}
        
        def check_disk():
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return {"status": "ERROR", "message": f"磁盘使用率过高: {disk.percent}%"}
            elif disk.percent > 80:
                return {"status": "WARNING", "message": f"磁盘使用率较高: {disk.percent}%"}
            else:
                return {"status": "OK", "message": f"磁盘使用率正常: {disk.percent}%"}
        
        async def check_response_time():
            # 模拟响应时间检查
            start = time.time()
            await asyncio.sleep(0.01)  # 模拟操作
            duration = time.time() - start
            
            if duration > 1.0:
                return {"status": "ERROR", "message": f"响应时间过长: {duration:.3f}s"}
            elif duration > 0.5:
                return {"status": "WARNING", "message": f"响应时间较长: {duration:.3f}s"}
            else:
                return {"status": "OK", "message": f"响应时间正常: {duration:.3f}s"}
        
        self.health_checker.register_check("memory", check_memory)
        self.health_checker.register_check("disk", check_disk)
        self.health_checker.register_check("response_time", check_response_time)
    
    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        
        # 错误率告警
        error_rate_rule = AlertRule(
            rule_id="high_error_rate",
            name="高错误率告警",
            condition="error_rate > threshold",
            threshold=0.05,  # 5%
            level=AlertLevel.HIGH,
            cooldown_minutes=10
        )
        
        # 内存使用告警
        memory_rule = AlertRule(
            rule_id="high_memory_usage",
            name="内存使用告警",
            condition="memory_usage > threshold",
            threshold=1024,  # 1GB
            level=AlertLevel.MEDIUM,
            cooldown_minutes=5
        )
        
        # CPU使用告警
        cpu_rule = AlertRule(
            rule_id="high_cpu_usage",
            name="CPU使用告警",
            condition="cpu_usage > threshold",
            threshold=80,  # 80%
            level=AlertLevel.MEDIUM,
            cooldown_minutes=5
        )
        
        # 响应时间告警
        response_time_rule = AlertRule(
            rule_id="slow_response",
            name="慢响应告警",
            condition="response_time > threshold",
            threshold=2.0,  # 2秒
            level=AlertLevel.HIGH,
            cooldown_minutes=5
        )
        
        self.alert_manager.add_rule(error_rate_rule)
        self.alert_manager.add_rule(memory_rule)
        self.alert_manager.add_rule(cpu_rule)
        self.alert_manager.add_rule(response_time_rule)
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        
        self.logger.log_event(
            LogLevel.INFO,
            "monitor_start",
            "监控系统已启动",
            interval_seconds=interval_seconds
        )
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.log_event(LogLevel.INFO, "monitor_stop", "监控系统已停止")
    
    async def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 记录系统指标
                self.performance_monitor.record_system_metrics()
                
                # 获取指标摘要
                metrics = self.performance_monitor.get_metrics_summary()
                
                # 运行健康检查
                health_results = await self.health_checker.run_all_checks()
                
                # 检查告警规则
                await self.alert_manager.check_rules(metrics)
                
                # 记录监控日志
                self.logger.log_event(
                    LogLevel.DEBUG,
                    "monitoring_cycle",
                    "监控周期完成",
                    metrics_summary=metrics,
                    health_status=self.health_checker.get_overall_status(),
                    active_alerts=len(self.alert_manager.get_active_alerts())
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.log_event(
                    LogLevel.ERROR,
                    "monitoring_error",
                    f"监控循环异常: {str(e)}"
                )
                await asyncio.sleep(5)  # 短暂等待后重试
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        # 获取性能指标
        metrics = self.performance_monitor.get_metrics_summary()
        
        # 运行健康检查
        health_results = await self.health_checker.run_all_checks()
        
        # 获取活跃告警
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": utc_now().isoformat(),
            "overall_status": self.health_checker.get_overall_status(),
            "performance_metrics": metrics,
            "health_checks": {
                name: asdict(result) for name, result in health_results.items()
            },
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "monitoring_enabled": self.is_monitoring
        }
    
    def record_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """记录请求"""
        self.performance_monitor.record_request(method, endpoint, duration, status_code)
    
    def record_query(self, query: str, duration: float, result_count: int, error: str = None):
        """记录查询"""
        self.performance_monitor.record_query("sparql", duration, result_count)
        self.logger.log_query(query, duration, result_count, error)
    
    def record_operation(self, operation: str, resource: str, user_id: str, success: bool, **kwargs):
        """记录操作"""
        self.logger.log_operation(operation, resource, user_id, success, **kwargs)


# 全局监控实例
kg_monitor = KnowledgeGraphMonitor()


# 装饰器
def monitor_performance(operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                kg_monitor.performance_monitor.metrics['operations'].append({
                    'timestamp': start_time,
                    'operation': op_name,
                    'duration': duration,
                    'success': True
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                kg_monitor.performance_monitor.metrics['operations'].append({
                    'timestamp': start_time,
                    'operation': op_name,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })
                
                kg_monitor.logger.log_event(
                    LogLevel.ERROR,
                    "operation_error",
                    f"操作 {op_name} 失败",
                    operation=op_name,
                    duration=duration,
                    error=str(e)
                )
                
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试监控系统
    async def test_monitoring():
        print("测试监控系统...")
        
        monitor = KnowledgeGraphMonitor("/tmp/test_kg_monitor")
        
        # 启动监控
        await monitor.start_monitoring(interval_seconds=5)
        
        try:
            # 模拟一些操作
            monitor.record_request("GET", "/api/entities", 0.1, 200)
            monitor.record_request("POST", "/api/query", 0.5, 200)
            monitor.record_request("GET", "/api/status", 0.05, 200)
            monitor.record_request("POST", "/api/import", 2.1, 500)  # 慢请求和错误
            
            monitor.record_query("SELECT * WHERE { ?s ?p ?o }", 0.3, 100)
            monitor.record_query("SELECT * WHERE { ?s rdf:type ?type }", 1.5, 50)
            
            monitor.record_operation("create", "entity:john", "user123", True)
            monitor.record_operation("delete", "entity:invalid", "user456", False, error="实体不存在")
            
            # 等待监控周期
            await asyncio.sleep(6)
            
            # 获取系统状态
            status = await monitor.get_system_status()
            print(f"系统状态: {status['overall_status']}")
            print(f"活跃告警: {len(status['active_alerts'])}")
            print(f"性能指标: 请求数={status['performance_metrics'].get('total_requests', 0)}")
            
            # 测试告警
            print("测试告警...")
            alert_handler = lambda alert: print(f"收到告警: {alert.message}")
            monitor.alert_manager.add_alert_handler(alert_handler)
            
            # 手动检查告警（模拟高错误率）
            test_metrics = {
                "error_rate": 0.1,  # 10% 错误率，超过阈值
                "system_metrics": {
                    "avg_cpu_percent": 85,  # 85% CPU使用率
                    "avg_memory_bytes": 2 * 1024 * 1024 * 1024  # 2GB内存
                }
            }
            
            await monitor.alert_manager.check_rules(test_metrics)
            active_alerts = monitor.alert_manager.get_active_alerts()
            print(f"触发的告警数: {len(active_alerts)}")
            
        finally:
            await monitor.stop_monitoring()
        
        print("监控系统测试完成")
    
    asyncio.run(test_monitoring())