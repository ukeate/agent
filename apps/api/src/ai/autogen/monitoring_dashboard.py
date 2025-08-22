"""
监控指标仪表板
提供企业级系统的实时监控数据可视化和告警系统
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import structlog
from collections import deque, defaultdict
import statistics

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"         # 计数器：累计值
    GAUGE = "gauge"            # 仪表：瞬时值
    HISTOGRAM = "histogram"    # 直方图：分布统计
    SUMMARY = "summary"        # 摘要：分位数统计


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"             # 信息
    WARNING = "warning"       # 警告
    CRITICAL = "critical"     # 严重
    EMERGENCY = "emergency"   # 紧急


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class MetricSeries:
    """指标序列"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """添加数据点"""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        self.points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """获取最新值"""
        return self.points[-1].value if self.points else None
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricPoint]:
        """获取时间范围内的数据点"""
        return [
            point for point in self.points
            if start_time <= point.timestamp <= end_time
        ]
    
    def get_statistics(self, duration_minutes: int = 60) -> Dict[str, float]:
        """获取统计信息"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        recent_values = [
            point.value for point in self.points
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "avg": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "latest": recent_values[-1] if recent_values else 0
        }


@dataclass
class Alert:
    """告警信息"""
    id: str
    metric_name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    threshold: float
    actual_value: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "labels": self.labels,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    level: AlertLevel
    message_template: str
    duration_minutes: int = 5  # 持续时间
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self, metric_series: MetricSeries) -> Optional[Alert]:
        """评估告警条件"""
        if not self.enabled:
            return None
        
        latest_value = metric_series.get_latest_value()
        if latest_value is None:
            return None
        
        # 检查阈值条件
        condition_met = self._check_condition(latest_value)
        if not condition_met:
            return None
        
        # 检查持续时间
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.duration_minutes)
        recent_points = [
            point for point in metric_series.points
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return None
        
        # 检查是否在整个持续时间内都满足条件
        all_meet_condition = all(
            self._check_condition(point.value) for point in recent_points
        )
        
        if not all_meet_condition:
            return None
        
        # 生成告警
        alert_id = f"{self.name}_{int(time.time())}"
        message = self.message_template.format(
            metric_name=self.metric_name,
            threshold=self.threshold,
            actual_value=latest_value,
            **metric_series.labels
        )
        
        return Alert(
            id=alert_id,
            metric_name=self.metric_name,
            level=self.level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            threshold=self.threshold,
            actual_value=latest_value,
            labels={**self.labels, **metric_series.labels}
        )
    
    def _check_condition(self, value: float) -> bool:
        """检查条件是否满足"""
        if self.condition == ">":
            return value > self.threshold
        elif self.condition == "<":
            return value < self.threshold
        elif self.condition == ">=":
            return value >= self.threshold
        elif self.condition == "<=":
            return value <= self.threshold
        elif self.condition == "==":
            return abs(value - self.threshold) < 1e-6
        elif self.condition == "!=":
            return abs(value - self.threshold) >= 1e-6
        else:
            return False


class MetricCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, MetricSeries] = {}
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 内置指标
        self._init_builtin_metrics()
        self._init_default_alert_rules()
    
    def _init_builtin_metrics(self):
        """初始化内置指标"""
        builtin_metrics = [
            # 系统指标
            ("system_cpu_usage", MetricType.GAUGE, "CPU使用率", "%"),
            ("system_memory_usage", MetricType.GAUGE, "内存使用率", "%"),
            ("system_disk_usage", MetricType.GAUGE, "磁盘使用率", "%"),
            ("system_network_io", MetricType.COUNTER, "网络IO", "bytes"),
            
            # 应用指标
            ("agent_pool_size", MetricType.GAUGE, "智能体池大小", "count"),
            ("agent_pool_active", MetricType.GAUGE, "活跃智能体数量", "count"),
            ("agent_pool_idle", MetricType.GAUGE, "空闲智能体数量", "count"),
            ("task_queue_size", MetricType.GAUGE, "任务队列大小", "count"),
            ("task_completion_rate", MetricType.GAUGE, "任务完成率", "%"),
            ("task_failure_rate", MetricType.GAUGE, "任务失败率", "%"),
            ("task_avg_duration", MetricType.GAUGE, "任务平均耗时", "ms"),
            
            # 流控指标
            ("flow_control_throughput", MetricType.GAUGE, "流控吞吐量", "tasks/sec"),
            ("flow_control_queue_size", MetricType.GAUGE, "流控队列大小", "count"),
            ("flow_control_drop_rate", MetricType.GAUGE, "流控丢弃率", "%"),
            ("flow_control_backpressure", MetricType.GAUGE, "背压状态", "boolean"),
            
            # 事件指标
            ("event_bus_messages", MetricType.COUNTER, "事件总数", "count"),
            ("event_bus_errors", MetricType.COUNTER, "事件错误数", "count"),
            ("event_processing_latency", MetricType.HISTOGRAM, "事件处理延迟", "ms"),
            
            # 错误指标
            ("error_count_by_category", MetricType.COUNTER, "按类别分组的错误数", "count"),
            ("error_count_by_severity", MetricType.COUNTER, "按严重性分组的错误数", "count")
        ]
        
        for name, metric_type, description, unit in builtin_metrics:
            self.register_metric(name, metric_type, description, unit)
    
    def _init_default_alert_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            # 系统资源告警
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_usage",
                condition=">",
                threshold=90.0,
                level=AlertLevel.WARNING,
                message_template="CPU使用率过高: {actual_value:.1f}% > {threshold}%",
                duration_minutes=2
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name="system_cpu_usage",
                condition=">",
                threshold=95.0,
                level=AlertLevel.CRITICAL,
                message_template="CPU使用率严重过高: {actual_value:.1f}% > {threshold}%",
                duration_minutes=1
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_usage",
                condition=">",
                threshold=85.0,
                level=AlertLevel.WARNING,
                message_template="内存使用率过高: {actual_value:.1f}% > {threshold}%",
                duration_minutes=5
            ),
            
            # 业务指标告警
            AlertRule(
                name="high_task_failure_rate",
                metric_name="task_failure_rate",
                condition=">",
                threshold=10.0,
                level=AlertLevel.CRITICAL,
                message_template="任务失败率过高: {actual_value:.1f}% > {threshold}%",
                duration_minutes=3
            ),
            AlertRule(
                name="no_available_agents",
                metric_name="agent_pool_idle",
                condition="==",
                threshold=0.0,
                level=AlertLevel.CRITICAL,
                message_template="没有可用的智能体",
                duration_minutes=1
            ),
            AlertRule(
                name="large_task_queue",
                metric_name="task_queue_size",
                condition=">",
                threshold=1000.0,
                level=AlertLevel.WARNING,
                message_template="任务队列过大: {actual_value} > {threshold}",
                duration_minutes=5
            ),
            
            # 流控告警
            AlertRule(
                name="high_flow_control_drop_rate",
                metric_name="flow_control_drop_rate",
                condition=">",
                threshold=5.0,
                level=AlertLevel.WARNING,
                message_template="流控丢弃率过高: {actual_value:.1f}% > {threshold}%",
                duration_minutes=3
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def register_metric(self, name: str, metric_type: MetricType, 
                       description: str, unit: str = "") -> MetricSeries:
        """注册指标"""
        if name in self.metrics:
            logger.warning(f"指标已存在，将被覆盖: {name}")
        
        metric_series = MetricSeries(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit
        )
        
        self.metrics[name] = metric_series
        logger.debug(f"注册指标: {name}")
        
        return metric_series
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Optional[Dict[str, str]] = None):
        """记录指标值"""
        if name not in self.metrics:
            logger.warning(f"未知指标: {name}")
            return
        
        self.metrics[name].add_point(value, labels)
        logger.debug(f"记录指标: {name} = {value}")
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """获取指标"""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """获取所有指标"""
        return self.metrics.copy()
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def evaluate_alerts(self):
        """评估所有告警规则"""
        new_alerts = []
        
        for rule in self.alert_rules:
            if rule.metric_name not in self.metrics:
                continue
            
            metric_series = self.metrics[rule.metric_name]
            alert = rule.evaluate(metric_series)
            
            if alert:
                # 检查是否已存在相同告警
                existing_alert_key = f"{rule.name}_{rule.metric_name}"
                if existing_alert_key not in self.active_alerts:
                    self.active_alerts[existing_alert_key] = alert
                    self.alert_history.append(alert)
                    new_alerts.append(alert)
                    logger.warning(f"触发告警: {alert.message}")
        
        # 执行告警回调
        for alert in new_alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"告警回调执行失败: {e}")
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        for key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                del self.active_alerts[key]
                logger.info(f"告警已解决: {alert_id}")
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return list(self.alert_history)[-limit:]


class DashboardServer:
    """仪表板服务器"""
    
    def __init__(self, metric_collector: MetricCollector, port: int = 8080):
        self.metric_collector = metric_collector
        self.port = port
        self.running = False
        self.update_interval = 1.0  # 更新间隔（秒）
        self.dashboard_data = {}
        
        # 注册告警回调
        self.metric_collector.add_alert_callback(self._on_alert)
    
    async def start(self):
        """启动仪表板服务"""
        if self.running:
            return
        
        self.running = True
        
        # 启动数据更新任务
        asyncio.create_task(self._update_dashboard_data())
        
        # 启动告警评估任务
        asyncio.create_task(self._evaluate_alerts_loop())
        
        logger.info(f"监控仪表板服务器启动，端口: {self.port}")
    
    async def stop(self):
        """停止仪表板服务"""
        self.running = False
        logger.info("监控仪表板服务器已停止")
    
    async def _update_dashboard_data(self):
        """更新仪表板数据"""
        while self.running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 更新仪表板数据
                self.dashboard_data = self._generate_dashboard_data()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"更新仪表板数据失败: {e}")
                await asyncio.sleep(5)
    
    async def _evaluate_alerts_loop(self):
        """告警评估循环"""
        while self.running:
            try:
                self.metric_collector.evaluate_alerts()
                await asyncio.sleep(30)  # 每30秒评估一次
            except Exception as e:
                logger.error(f"告警评估失败: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_usage = psutil.cpu_percent()
            self.metric_collector.record_metric("system_cpu_usage", cpu_usage)
            
            # 内存使用率
            memory_info = psutil.virtual_memory()
            self.metric_collector.record_metric("system_memory_usage", memory_info.percent)
            
            # 磁盘使用率
            disk_info = psutil.disk_usage('/')
            disk_usage = (disk_info.used / disk_info.total) * 100
            self.metric_collector.record_metric("system_disk_usage", disk_usage)
            
        except ImportError:
            # psutil不可用时使用模拟数据
            import random
            self.metric_collector.record_metric("system_cpu_usage", random.uniform(10, 80))
            self.metric_collector.record_metric("system_memory_usage", random.uniform(20, 70))
            self.metric_collector.record_metric("system_disk_usage", random.uniform(30, 60))
        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")
    
    def _generate_dashboard_data(self) -> Dict[str, Any]:
        """生成仪表板数据"""
        dashboard = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "alerts": {
                "active": [alert.to_dict() for alert in self.metric_collector.get_active_alerts()],
                "recent": [alert.to_dict() for alert in self.metric_collector.get_alert_history(10)]
            },
            "summary": {}
        }
        
        # 收集指标数据
        for name, metric_series in self.metric_collector.get_all_metrics().items():
            stats = metric_series.get_statistics(60)  # 最近60分钟
            
            dashboard["metrics"][name] = {
                "name": name,
                "type": metric_series.metric_type.value,
                "description": metric_series.description,
                "unit": metric_series.unit,
                "current_value": metric_series.get_latest_value(),
                "statistics": stats,
                "recent_points": [
                    point.to_dict() for point in list(metric_series.points)[-60:]  # 最近60个点
                ]
            }
        
        # 生成摘要
        dashboard["summary"] = self._generate_summary()
        
        return dashboard
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成系统摘要"""
        summary = {
            "system_status": "healthy",
            "total_metrics": len(self.metric_collector.metrics),
            "active_alerts": len(self.metric_collector.active_alerts),
            "critical_alerts": len([
                alert for alert in self.metric_collector.get_active_alerts()
                if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
            ]),
            "key_metrics": {}
        }
        
        # 关键指标
        key_metric_names = [
            "system_cpu_usage", "system_memory_usage", 
            "agent_pool_active", "task_completion_rate",
            "flow_control_throughput"
        ]
        
        for name in key_metric_names:
            if name in self.metric_collector.metrics:
                metric = self.metric_collector.metrics[name]
                summary["key_metrics"][name] = {
                    "value": metric.get_latest_value(),
                    "unit": metric.unit
                }
        
        # 根据告警数量调整系统状态
        if summary["critical_alerts"] > 0:
            summary["system_status"] = "critical"
        elif summary["active_alerts"] > 0:
            summary["system_status"] = "warning"
        
        return summary
    
    def _on_alert(self, alert: Alert):
        """告警回调处理"""
        logger.info(f"仪表板接收到告警: {alert.message}")
        # 这里可以添加告警通知逻辑，如发送邮件、短信等
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.dashboard_data.copy()
    
    def get_metric_data(self, metric_name: str, 
                       duration_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """获取特定指标数据"""
        if metric_name not in self.metric_collector.metrics:
            return None
        
        metric_series = self.metric_collector.metrics[metric_name]
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        
        recent_points = [
            point.to_dict() for point in metric_series.points
            if point.timestamp >= cutoff_time
        ]
        
        return {
            "name": metric_name,
            "type": metric_series.metric_type.value,
            "description": metric_series.description,
            "unit": metric_series.unit,
            "points": recent_points,
            "statistics": metric_series.get_statistics(duration_minutes)
        }


# 全局监控实例
_metric_collector: Optional[MetricCollector] = None
_dashboard_server: Optional[DashboardServer] = None

def get_metric_collector() -> MetricCollector:
    """获取全局指标收集器"""
    global _metric_collector
    if _metric_collector is None:
        _metric_collector = MetricCollector()
    return _metric_collector

def get_dashboard_server() -> Optional[DashboardServer]:
    """获取全局仪表板服务器"""
    return _dashboard_server

async def init_monitoring_dashboard(port: int = 8080):
    """初始化监控仪表板"""
    global _dashboard_server
    collector = get_metric_collector()
    _dashboard_server = DashboardServer(collector, port)
    await _dashboard_server.start()

async def shutdown_monitoring_dashboard():
    """关闭监控仪表板"""
    global _dashboard_server
    if _dashboard_server:
        await _dashboard_server.stop()
        _dashboard_server = None