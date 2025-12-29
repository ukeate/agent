"""
情感智能系统监控和故障恢复机制
提供系统健康检查、性能监控、异常检测和自动恢复功能
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import json
import time
import psutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import traceback
from .core_interfaces import EmotionalSystemMonitor
from .communication_protocol import CommunicationProtocol, ModuleType, Priority

from src.core.logging import get_logger
class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

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
    TIMER = "timer"

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    module_type: ModuleType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time: float
    details: Dict[str, Any] = None

@dataclass
class SystemMetric:
    """系统指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = None
    description: str = ""

@dataclass
class Alert:
    """系统告警"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    source: ModuleType
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, protocol: CommunicationProtocol):
        self._protocol = protocol
        self._logger = get_logger(__name__)
        self._check_timeout = 10.0
    
    async def check_module_health(self, module_type: ModuleType) -> HealthCheckResult:
        """检查单个模块健康状态"""
        start_time = time.time()
        
        try:
            response = await self._protocol.send_request(
                target_module=module_type,
                payload={"action": "health_check"},
                priority=Priority.HIGH,
                timeout=self._check_timeout
            )
            
            response_time = time.time() - start_time
            
            if response and response.get("success"):
                status = HealthStatus.HEALTHY
                message = "Module is healthy"
                details = response.get("details", {})
            elif response:
                status = HealthStatus.WARNING
                message = response.get("message", "Module responded but with warnings")
                details = response.get("details", {})
            else:
                status = HealthStatus.DOWN
                message = "Module did not respond"
                details = {"error": "No response"}
                
        except asyncio.TimeoutError:
            response_time = self._check_timeout
            status = HealthStatus.CRITICAL
            message = f"Module health check timeout after {self._check_timeout}s"
            details = {"error": "Timeout"}
            
        except Exception as e:
            response_time = time.time() - start_time
            status = HealthStatus.CRITICAL
            message = f"Health check failed: {str(e)}"
            details = {"error": str(e), "traceback": traceback.format_exc()}
        
        return HealthCheckResult(
            module_type=module_type,
            status=status,
            message=message,
            timestamp=utc_now(),
            response_time=response_time,
            details=details
        )
    
    async def check_all_modules(self) -> Dict[ModuleType, HealthCheckResult]:
        """检查所有模块健康状态"""
        check_tasks = []
        modules_to_check = [
            module for module in ModuleType 
            if module != ModuleType.SYSTEM_MONITOR
        ]
        
        for module_type in modules_to_check:
            task = self.check_module_health(module_type)
            check_tasks.append((module_type, task))
        
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in check_tasks], 
            return_exceptions=True
        )
        
        for (module_type, _), result in zip(check_tasks, completed_tasks):
            if isinstance(result, Exception):
                results[module_type] = HealthCheckResult(
                    module_type=module_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check exception: {str(result)}",
                    timestamp=utc_now(),
                    response_time=0.0,
                    details={"error": str(result)}
                )
            else:
                results[module_type] = result
        
        return results

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._metrics_history: Dict[str, List[SystemMetric]] = {}
        self._max_history_size = 1000
    
    def record_metric(self, metric: SystemMetric):
        """记录指标"""
        metric_key = f"{metric.name}:{':'.join(f'{k}={v}' for k, v in (metric.tags or {}).items())}"
        
        if metric_key not in self._metrics_history:
            self._metrics_history[metric_key] = []
        
        self._metrics_history[metric_key].append(metric)
        
        # 保持历史记录大小限制
        if len(self._metrics_history[metric_key]) > self._max_history_size:
            self._metrics_history[metric_key] = self._metrics_history[metric_key][-self._max_history_size:]
    
    def get_system_metrics(self) -> List[SystemMetric]:
        """获取系统资源指标"""
        timestamp = utc_now()
        metrics = []
        
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(SystemMetric(
                name="system.cpu.usage",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="CPU usage percentage"
            ))
            
            # 内存使用
            memory = psutil.virtual_memory()
            metrics.append(SystemMetric(
                name="system.memory.usage",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Memory usage percentage"
            ))
            
            metrics.append(SystemMetric(
                name="system.memory.available",
                value=memory.available / (1024 * 1024 * 1024),  # GB
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Available memory in GB"
            ))
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            metrics.append(SystemMetric(
                name="system.disk.usage",
                value=(disk.used / disk.total) * 100,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Disk usage percentage"
            ))
            
            # 网络统计
            net_io = psutil.net_io_counters()
            metrics.append(SystemMetric(
                name="system.network.bytes_sent",
                value=net_io.bytes_sent,
                metric_type=MetricType.COUNTER,
                timestamp=timestamp,
                description="Total bytes sent"
            ))
            
            metrics.append(SystemMetric(
                name="system.network.bytes_recv",
                value=net_io.bytes_recv,
                metric_type=MetricType.COUNTER,
                timestamp=timestamp,
                description="Total bytes received"
            ))
            
        except Exception as e:
            self._logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def get_metric_history(
        self, 
        metric_name: str, 
        tags: Optional[Dict[str, str]] = None,
        since: Optional[datetime] = None
    ) -> List[SystemMetric]:
        """获取指标历史"""
        metric_key = f"{metric_name}:{':'.join(f'{k}={v}' for k, v in (tags or {}).items())}"
        
        history = self._metrics_history.get(metric_key, [])
        
        if since:
            history = [m for m in history if m.timestamp >= since]
        
        return history
    
    def calculate_metric_statistics(
        self, 
        metric_name: str,
        tags: Optional[Dict[str, str]] = None,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """计算指标统计信息"""
        since = utc_now() - timedelta(minutes=window_minutes)
        history = self.get_metric_history(metric_name, tags, since)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0.0
        }

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self._metrics_collector = metrics_collector
        self._logger = get_logger(__name__)
        
        # 阈值配置
        self._thresholds = {
            "system.cpu.usage": {"warning": 70.0, "critical": 90.0},
            "system.memory.usage": {"warning": 80.0, "critical": 95.0},
            "system.disk.usage": {"warning": 80.0, "critical": 95.0},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.10}
        }
    
    def detect_metric_anomalies(self, metrics: List[SystemMetric]) -> List[Dict[str, Any]]:
        """检测指标异常"""
        anomalies = []
        
        for metric in metrics:
            threshold_config = self._thresholds.get(metric.name)
            if not threshold_config:
                continue
            
            anomaly = None
            
            if metric.value >= threshold_config.get("critical", float('inf')):
                anomaly = {
                    "type": "threshold_critical",
                    "metric": metric.name,
                    "value": metric.value,
                    "threshold": threshold_config["critical"],
                    "severity": AlertSeverity.CRITICAL,
                    "message": f"{metric.name} is critically high: {metric.value}"
                }
            elif metric.value >= threshold_config.get("warning", float('inf')):
                anomaly = {
                    "type": "threshold_warning",
                    "metric": metric.name,
                    "value": metric.value,
                    "threshold": threshold_config["warning"],
                    "severity": AlertSeverity.WARNING,
                    "message": f"{metric.name} is high: {metric.value}"
                }
            
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_trend_anomalies(
        self, 
        metric_name: str,
        window_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """检测趋势异常"""
        anomalies = []
        
        try:
            stats = self._metrics_collector.calculate_metric_statistics(
                metric_name, window_minutes=window_minutes
            )
            
            if not stats or stats["count"] < 3:
                return anomalies
            
            # 检测快速增长趋势
            recent_stats = self._metrics_collector.calculate_metric_statistics(
                metric_name, window_minutes=5
            )
            
            if recent_stats and recent_stats["avg"] > stats["avg"] * 1.5:
                anomalies.append({
                    "type": "rapid_increase",
                    "metric": metric_name,
                    "current_avg": recent_stats["avg"],
                    "baseline_avg": stats["avg"],
                    "severity": AlertSeverity.WARNING,
                    "message": f"{metric_name} showing rapid increase trend"
                })
            
        except Exception as e:
            self._logger.error(f"Error detecting trend anomalies: {e}")
        
        return anomalies
    
    def detect_health_anomalies(
        self, 
        health_results: Dict[ModuleType, HealthCheckResult]
    ) -> List[Dict[str, Any]]:
        """检测健康状态异常"""
        anomalies = []
        
        for module_type, result in health_results.items():
            if result.status == HealthStatus.CRITICAL:
                anomalies.append({
                    "type": "module_critical",
                    "module": module_type.value,
                    "status": result.status.value,
                    "message": result.message,
                    "severity": AlertSeverity.CRITICAL
                })
            elif result.status == HealthStatus.DOWN:
                anomalies.append({
                    "type": "module_down",
                    "module": module_type.value,
                    "status": result.status.value,
                    "message": result.message,
                    "severity": AlertSeverity.CRITICAL
                })
            elif result.status == HealthStatus.WARNING:
                anomalies.append({
                    "type": "module_warning",
                    "module": module_type.value,
                    "status": result.status.value,
                    "message": result.message,
                    "severity": AlertSeverity.WARNING
                })
            elif result.response_time > 5.0:  # 响应时间过长
                anomalies.append({
                    "type": "slow_response",
                    "module": module_type.value,
                    "response_time": result.response_time,
                    "severity": AlertSeverity.WARNING,
                    "message": f"{module_type.value} response time is slow: {result.response_time:.2f}s"
                })
        
        return anomalies

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._max_history_size = 10000
        
        # 告警规则
        self._suppression_rules = {}
        self._escalation_rules = {}
    
    def create_alert(
        self, 
        title: str,
        description: str,
        severity: AlertSeverity,
        source: ModuleType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建告警"""
        alert_id = f"{source.value}_{int(time.time())}_{hash(title) & 0xffffffff:08x}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            source=source,
            timestamp=utc_now(),
            metadata=metadata or {}
        )
        
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # 保持历史记录大小限制
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]
        
        self._logger.warning(f"Alert created: {alert.title} ({alert.severity.value})")
        
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """解决告警"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = utc_now()
            
            if resolution_note:
                alert.metadata["resolution_note"] = resolution_note
            
            del self._active_alerts[alert_id]
            
            self._logger.info(f"Alert resolved: {alert.title}")
            return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        active_alerts = list(self._active_alerts.values())
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in active_alerts if a.severity == severity
            ])
        
        return {
            "active_count": len(active_alerts),
            "total_count": len(self._alert_history),
            "severity_distribution": severity_counts,
            "oldest_active": min(
                (a.timestamp for a in active_alerts), default=None
            )
        }

class RecoveryManager:
    """故障恢复管理器"""
    
    def __init__(self, protocol: CommunicationProtocol):
        self._protocol = protocol
        self._logger = get_logger(__name__)
        self._recovery_strategies = {}
        self._recovery_history: List[Dict[str, Any]] = []
    
    def register_recovery_strategy(
        self, 
        module_type: ModuleType, 
        strategy_func: callable
    ):
        """注册恢复策略"""
        self._recovery_strategies[module_type] = strategy_func
    
    async def attempt_recovery(
        self, 
        module_type: ModuleType,
        issue_description: str
    ) -> bool:
        """尝试恢复模块"""
        self._logger.info(f"Attempting recovery for {module_type.value}: {issue_description}")
        
        recovery_record = {
            "module": module_type.value,
            "issue": issue_description,
            "timestamp": utc_now(),
            "success": False,
            "actions_taken": []
        }
        
        try:
            # 通用恢复策略
            success = await self._execute_general_recovery(module_type, recovery_record)
            
            # 模块特定恢复策略
            if not success and module_type in self._recovery_strategies:
                strategy_func = self._recovery_strategies[module_type]
                success = await strategy_func(module_type, issue_description, recovery_record)
            
            recovery_record["success"] = success
            
            if success:
                self._logger.info(f"Recovery successful for {module_type.value}")
            else:
                self._logger.error(f"Recovery failed for {module_type.value}")
                
        except Exception as e:
            self._logger.error(f"Recovery attempt failed: {e}")
            recovery_record["error"] = str(e)
        
        self._recovery_history.append(recovery_record)
        return recovery_record["success"]
    
    async def _execute_general_recovery(
        self, 
        module_type: ModuleType,
        recovery_record: Dict[str, Any]
    ) -> bool:
        """执行通用恢复策略"""
        
        # 1. 发送重启请求
        try:
            response = await self._protocol.send_request(
                target_module=module_type,
                payload={"action": "restart"},
                priority=Priority.CRITICAL,
                timeout=30.0
            )
            
            recovery_record["actions_taken"].append("restart_request_sent")
            
            if response and response.get("success"):
                recovery_record["actions_taken"].append("restart_successful")
                
                # 等待模块重新上线
                await asyncio.sleep(5)
                
                # 验证恢复
                health_response = await self._protocol.send_request(
                    target_module=module_type,
                    payload={"action": "health_check"},
                    priority=Priority.HIGH,
                    timeout=10.0
                )
                
                if health_response and health_response.get("success"):
                    recovery_record["actions_taken"].append("health_check_passed")
                    return True
                else:
                    recovery_record["actions_taken"].append("health_check_failed")
                    
        except Exception as e:
            recovery_record["actions_taken"].append(f"restart_error: {str(e)}")
        
        return False
    
    def get_recovery_history(
        self, 
        module_type: Optional[ModuleType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取恢复历史"""
        history = self._recovery_history
        
        if module_type:
            history = [r for r in history if r["module"] == module_type.value]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]

class EmotionalSystemMonitorImpl(EmotionalSystemMonitor):
    """情感系统监控器实现"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._protocol = CommunicationProtocol(ModuleType.SYSTEM_MONITOR)
        
        # 核心组件
        self._health_checker = HealthChecker(self._protocol)
        self._metrics_collector = MetricsCollector()
        self._anomaly_detector = AnomalyDetector(self._metrics_collector)
        self._alert_manager = AlertManager()
        self._recovery_manager = RecoveryManager(self._protocol)
        
        # 监控配置
        self._monitoring_interval = 30  # 秒
        self._health_check_interval = 60  # 秒
        self._is_running = False
        
        # 系统状态
        self._system_status = HealthStatus.HEALTHY
        self._last_health_check: Optional[Dict[ModuleType, HealthCheckResult]] = None
    
    async def start(self):
        """启动系统监控"""
        self._is_running = True
        self._logger.info("Starting emotional system monitor")
        
        # 启动监控任务
        await asyncio.gather(
            self._monitoring_loop(),
            self._health_check_loop(),
            self._protocol.start(),
            return_exceptions=True
        )
    
    async def stop(self):
        """停止系统监控"""
        self._is_running = False
        await self._protocol.stop()
        self._logger.info("Emotional system monitor stopped")
    
    async def collect_performance_metrics(self) -> Dict[str, float]:
        """收集性能指标"""
        metrics_data = {}
        
        try:
            # 收集系统指标
            system_metrics = self._metrics_collector.get_system_metrics()
            for metric in system_metrics:
                self._metrics_collector.record_metric(metric)
                metrics_data[metric.name] = metric.value
            
            # 收集模块特定指标
            module_metrics = await self._collect_module_metrics()
            metrics_data.update(module_metrics)
            
        except Exception as e:
            self._logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics_data
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        try:
            # 收集最新指标
            system_metrics = self._metrics_collector.get_system_metrics()
            
            # 检测指标异常
            metric_anomalies = self._anomaly_detector.detect_metric_anomalies(system_metrics)
            anomalies.extend(metric_anomalies)
            
            # 检测趋势异常
            for metric_name in ["system.cpu.usage", "system.memory.usage", "response_time"]:
                trend_anomalies = self._anomaly_detector.detect_trend_anomalies(metric_name)
                anomalies.extend(trend_anomalies)
            
            # 检测健康状态异常
            if self._last_health_check:
                health_anomalies = self._anomaly_detector.detect_health_anomalies(
                    self._last_health_check
                )
                anomalies.extend(health_anomalies)
            
        except Exception as e:
            self._logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        try:
            # 执行健康检查
            health_results = await self._health_checker.check_all_modules()
            self._last_health_check = health_results
            
            # 收集性能指标
            performance_metrics = await self.collect_performance_metrics()
            
            # 检测异常
            anomalies = await self.detect_anomalies()
            
            # 获取告警统计
            alert_stats = self._alert_manager.get_alert_statistics()
            
            # 计算整体系统状态
            overall_status = self._calculate_overall_status(health_results)
            self._system_status = overall_status
            
            report = {
                "timestamp": utc_now().isoformat(),
                "overall_status": overall_status.value,
                "module_health": {
                    module.value: {
                        "status": result.status.value,
                        "response_time": result.response_time,
                        "message": result.message
                    }
                    for module, result in health_results.items()
                },
                "performance_metrics": performance_metrics,
                "anomalies": anomalies,
                "alert_statistics": alert_stats,
                "active_alerts": [
                    {
                        "id": alert.alert_id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self._alert_manager.get_active_alerts()
                ],
                "system_uptime": self._get_system_uptime(),
                "recovery_history": self._recovery_manager.get_recovery_history(limit=10)
            }
            
            return report
            
        except Exception as e:
            self._logger.error(f"Error generating health report: {e}")
            return {
                "error": str(e),
                "timestamp": utc_now().isoformat(),
                "overall_status": HealthStatus.CRITICAL.value
            }
    
    async def trigger_alerts(self, alert_data: Dict[str, Any]) -> bool:
        """触发告警"""
        try:
            alert_id = self._alert_manager.create_alert(
                title=alert_data.get("title", "System Alert"),
                description=alert_data.get("description", ""),
                severity=AlertSeverity(alert_data.get("severity", AlertSeverity.WARNING)),
                source=ModuleType(alert_data.get("source", ModuleType.SYSTEM_MONITOR)),
                metadata=alert_data.get("metadata")
            )
            
            # 如果是关键告警，尝试自动恢复
            if alert_data.get("severity") == AlertSeverity.CRITICAL:
                source_module = ModuleType(alert_data.get("source", ModuleType.SYSTEM_MONITOR))
                await self._recovery_manager.attempt_recovery(
                    source_module,
                    alert_data.get("description", "Critical alert triggered")
                )
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error triggering alert: {e}")
            return False
    
    async def _monitoring_loop(self):
        """监控主循环"""
        while self._is_running:
            try:
                # 收集指标
                await self.collect_performance_metrics()
                
                # 检测异常
                anomalies = await self.detect_anomalies()
                
                # 处理异常
                for anomaly in anomalies:
                    await self.trigger_alerts({
                        "title": f"Anomaly Detected: {anomaly['type']}",
                        "description": anomaly['message'],
                        "severity": anomaly['severity'],
                        "source": ModuleType.SYSTEM_MONITOR,
                        "metadata": anomaly
                    })
                
                await asyncio.sleep(self._monitoring_interval)
                
            except Exception as e:
                self._logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self._monitoring_interval)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._is_running:
            try:
                # 执行健康检查
                health_results = await self._health_checker.check_all_modules()
                self._last_health_check = health_results
                
                # 处理不健康的模块
                for module_type, result in health_results.items():
                    if result.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                        await self._recovery_manager.attempt_recovery(
                            module_type,
                            f"Module status: {result.status.value} - {result.message}"
                        )
                
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                self._logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _collect_module_metrics(self) -> Dict[str, float]:
        """收集模块指标"""
        module_metrics = {}
        
        # 通过协议收集各模块指标
        for module_type in ModuleType:
            if module_type == ModuleType.SYSTEM_MONITOR:
                continue
            
            try:
                response = await self._protocol.send_request(
                    target_module=module_type,
                    payload={"action": "get_metrics"},
                    priority=Priority.NORMAL,
                    timeout=5.0
                )
                
                if response and response.get("metrics"):
                    metrics = response["metrics"]
                    for metric_name, value in metrics.items():
                        key = f"{module_type.value}.{metric_name}"
                        module_metrics[key] = value
                        
            except Exception as e:
                self._logger.debug(f"Could not collect metrics from {module_type.value}: {e}")
        
        return module_metrics
    
    def _calculate_overall_status(
        self, 
        health_results: Dict[ModuleType, HealthCheckResult]
    ) -> HealthStatus:
        """计算整体系统状态"""
        if not health_results:
            return HealthStatus.DOWN
        
        statuses = [result.status for result in health_results.values()]
        
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _get_system_uptime(self) -> str:
        """获取系统运行时间"""
        # 这里可以记录系统启动时间并计算运行时长
        return "Unknown"  # 简化实现
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "is_running": self._is_running,
            "system_status": self._system_status.value,
            "monitoring_interval": self._monitoring_interval,
            "health_check_interval": self._health_check_interval,
            "active_alerts": len(self._alert_manager.get_active_alerts())
        }
