"""
资源监控和指标收集器

负责智能体集群的实时监控、指标收集、历史数据存储和趋势分析。
基于Prometheus监控模型和Kubernetes metrics-server设计。
"""

import asyncio
import logging
import time
import json
import httpx
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import statistics

from .topology import AgentInfo, ResourceUsage, AgentStatus
from .state_manager import ClusterStateManager


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"              # 计数器指标
    GAUGE = "gauge"                  # 瞬时值指标
    HISTOGRAM = "histogram"          # 直方图指标
    SUMMARY = "summary"              # 摘要指标


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """指标数据点"""
    metric_name: str
    agent_id: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "agent_id": self.agent_id,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class MetricSeries:
    """指标时间序列"""
    metric_name: str
    agent_id: str
    points: List[MetricPoint] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: float, timestamp: Optional[float] = None):
        """添加数据点"""
        if timestamp is None:
            timestamp = time.time()
        
        point = MetricPoint(
            metric_name=self.metric_name,
            agent_id=self.agent_id,
            value=value,
            timestamp=timestamp,
            labels=self.labels.copy()
        )
        self.points.append(point)
        
        # 保持最近1000个点
        if len(self.points) > 1000:
            self.points = self.points[-1000:]
    
    def get_latest_value(self) -> Optional[float]:
        """获取最新值"""
        return self.points[-1].value if self.points else None
    
    def get_average(self, duration_seconds: float = 300) -> Optional[float]:
        """获取指定时间内的平均值"""
        if not self.points:
            return None
        
        cutoff_time = time.time() - duration_seconds
        recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        
        if not recent_points:
            return None
        
        return statistics.mean(p.value for p in recent_points)
    
    def get_percentile(self, percentile: float, duration_seconds: float = 300) -> Optional[float]:
        """获取指定时间内的百分位数"""
        if not self.points:
            return None
        
        cutoff_time = time.time() - duration_seconds
        recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        
        if not recent_points:
            return None
        
        values = [p.value for p in recent_points]
        return statistics.quantiles(values, n=100)[int(percentile) - 1]


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    duration_seconds: float = 300  # 持续时间
    level: AlertLevel = AlertLevel.WARNING
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def evaluate(self, metric_series: MetricSeries) -> bool:
        """评估告警规则"""
        if not self.enabled or not metric_series.points:
            return False
        
        # 获取指定时间内的平均值
        avg_value = metric_series.get_average(self.duration_seconds)
        if avg_value is None:
            return False
        
        # 评估条件
        if self.condition == ">":
            return avg_value > self.threshold
        elif self.condition == "<":
            return avg_value < self.threshold
        elif self.condition == ">=":
            return avg_value >= self.threshold
        elif self.condition == "<=":
            return avg_value <= self.threshold
        elif self.condition == "==":
            return abs(avg_value - self.threshold) < 0.001
        elif self.condition == "!=":
            return abs(avg_value - self.threshold) >= 0.001
        
        return False


@dataclass
class AlertEvent:
    """告警事件"""
    alert_id: str
    rule_id: str
    agent_id: str
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def resolve(self):
        """解决告警"""
        self.resolved = True
        self.resolved_at = time.time()


class MetricsCollector:
    """资源监控和指标收集器
    
    提供智能体集群的实时监控功能，包括：
    - 实时资源使用情况收集
    - 自定义业务指标支持
    - 历史数据存储和查询
    - 告警规则管理和通知
    - 趋势分析和预测
    """
    
    def __init__(
        self, 
        cluster_manager: ClusterStateManager,
        storage_backend: Optional[Any] = None,
        collection_interval: float = 30.0
    ):
        self.cluster_manager = cluster_manager
        self.storage_backend = storage_backend
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # HTTP客户端用于从智能体收集指标
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # 指标存储
        self.metric_series: Dict[str, Dict[str, MetricSeries]] = {}  # {metric_name: {agent_id: series}}
        self.cluster_metrics: Dict[str, MetricSeries] = {}  # 集群级别指标
        
        # 告警管理
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        self.max_alert_history = 10000
        
        # 自定义指标收集器
        self.custom_collectors: List[Callable] = []
        
        # 收集任务
        self.collection_task: Optional[asyncio.Task] = None
        self.alert_check_task: Optional[asyncio.Task] = None
        
        # 性能指标
        self.collector_metrics = {
            "collections_performed": 0,
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "collection_errors": 0,
            "avg_collection_time": 0.0
        }
        
        # 预定义的标准指标
        self.standard_metrics = [
            "cpu_usage_percent",
            "memory_usage_percent", 
            "storage_usage_percent",
            "gpu_usage_percent",
            "network_io_mbps",
            "active_tasks",
            "total_requests",
            "failed_requests",
            "avg_response_time",
            "error_rate"
        ]
        
        # 预定义告警规则
        self._create_default_alert_rules()
        
        self.logger.info("MetricsCollector initialized")
    
    async def start(self):
        """启动指标收集器"""
        try:
            # 启动指标收集任务
            self.collection_task = asyncio.create_task(self._collection_loop())
            
            # 启动告警检查任务
            self.alert_check_task = asyncio.create_task(self._alert_check_loop())
            
            self.logger.info("MetricsCollector started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start MetricsCollector: {e}")
            raise
    
    async def stop(self):
        """停止指标收集器"""
        try:
            # 停止收集任务
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    pass
            
            # 停止告警检查任务
            if self.alert_check_task:
                self.alert_check_task.cancel()
                try:
                    await self.alert_check_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭HTTP客户端
            await self.http_client.aclose()
            
            # 最终持久化指标
            await self._persist_metrics()
            
            self.logger.info("MetricsCollector stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MetricsCollector: {e}")
    
    # 指标收集
    async def collect_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """收集单个智能体的指标"""
        
        agent = await self.cluster_manager.get_agent_info(agent_id)
        if not agent:
            return {}
        
        try:
            # 从智能体获取指标
            response = await self.http_client.get(
                f"{agent.endpoint}/metrics",
                timeout=10.0
            )
            
            if response.status_code == 200:
                metrics_data = response.json()
                
                # 标准化指标格式
                standardized_metrics = self._standardize_metrics(metrics_data, agent_id)
                
                # 存储指标
                await self._store_metrics(agent_id, standardized_metrics)
                
                # 更新智能体资源使用情况
                resource_usage = self._create_resource_usage_from_metrics(standardized_metrics)
                await self.cluster_manager.update_agent_resource_usage(agent_id, resource_usage)
                
                return standardized_metrics
            else:
                self.logger.warning(f"Failed to collect metrics from agent {agent_id}: HTTP {response.status_code}")
                return {}
                
        except httpx.ConnectError:
            self.logger.debug(f"Cannot connect to agent {agent_id} for metrics collection")
            return {}
        except httpx.TimeoutException:
            self.logger.warning(f"Timeout collecting metrics from agent {agent_id}")
            return {}
        except Exception as e:
            self.logger.error(f"Error collecting metrics from agent {agent_id}: {e}")
            self.collector_metrics["collection_errors"] += 1
            return {}
    
    async def collect_cluster_metrics(self) -> Dict[str, Any]:
        """收集集群级别指标"""
        
        try:
            topology = await self.cluster_manager.get_cluster_topology()
            
            # 计算集群聚合指标
            cluster_metrics = {
                "total_agents": topology.total_agents,
                "running_agents": topology.running_agents,
                "healthy_agents": topology.healthy_agents,
                "health_score": topology.cluster_health_score,
                "groups_count": len(topology.groups)
            }
            
            # 计算资源使用聚合
            cluster_usage = topology.cluster_resource_usage
            cluster_metrics.update({
                "cluster_cpu_usage": cluster_usage.cpu_usage_percent,
                "cluster_memory_usage": cluster_usage.memory_usage_percent,
                "cluster_storage_usage": cluster_usage.storage_usage_percent,
                "cluster_gpu_usage": cluster_usage.gpu_usage_percent,
                "cluster_network_io": cluster_usage.network_io_mbps,
                "cluster_active_tasks": cluster_usage.active_tasks,
                "cluster_total_requests": cluster_usage.total_requests,
                "cluster_failed_requests": cluster_usage.failed_requests,
                "cluster_error_rate": cluster_usage.error_rate,
                "cluster_avg_response_time": cluster_usage.avg_response_time
            })
            
            # 存储集群指标
            await self._store_cluster_metrics(cluster_metrics)
            
            return cluster_metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting cluster metrics: {e}")
            return {}
    
    async def add_custom_metric(
        self, 
        agent_id: str, 
        metric_name: str, 
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """添加自定义指标"""
        
        try:
            # 获取或创建指标序列
            if metric_name not in self.metric_series:
                self.metric_series[metric_name] = {}
            
            if agent_id not in self.metric_series[metric_name]:
                self.metric_series[metric_name][agent_id] = MetricSeries(
                    metric_name=metric_name,
                    agent_id=agent_id,
                    labels=labels or {}
                )
            
            # 添加数据点
            self.metric_series[metric_name][agent_id].add_point(value)
            self.collector_metrics["metrics_collected"] += 1
            
            self.logger.debug(f"Added custom metric {metric_name} for agent {agent_id}: {value}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom metric: {e}")
    
    # 指标查询
    async def get_agent_metrics(
        self, 
        agent_id: str,
        metric_names: Optional[List[str]] = None,
        duration_seconds: float = 3600
    ) -> Dict[str, List[MetricPoint]]:
        """获取智能体指标"""
        
        result = {}
        cutoff_time = time.time() - duration_seconds
        
        # 如果没有指定指标名称，返回所有指标
        if metric_names is None:
            metric_names = list(self.metric_series.keys())
        
        for metric_name in metric_names:
            if (metric_name in self.metric_series and 
                agent_id in self.metric_series[metric_name]):
                
                series = self.metric_series[metric_name][agent_id]
                recent_points = [
                    p for p in series.points 
                    if p.timestamp >= cutoff_time
                ]
                result[metric_name] = recent_points
        
        return result
    
    async def get_cluster_metrics(
        self, 
        metric_names: Optional[List[str]] = None,
        duration_seconds: float = 3600
    ) -> Dict[str, List[MetricPoint]]:
        """获取集群指标"""
        
        result = {}
        cutoff_time = time.time() - duration_seconds
        
        if metric_names is None:
            metric_names = list(self.cluster_metrics.keys())
        
        for metric_name in metric_names:
            if metric_name in self.cluster_metrics:
                series = self.cluster_metrics[metric_name]
                recent_points = [
                    p for p in series.points 
                    if p.timestamp >= cutoff_time
                ]
                result[metric_name] = recent_points
        
        return result
    
    async def get_metrics_summary(
        self, 
        agent_id: Optional[str] = None,
        duration_seconds: float = 300
    ) -> Dict[str, Any]:
        """获取指标摘要"""
        
        summary = {}
        
        if agent_id:
            # 单个智能体摘要
            for metric_name, agent_series in self.metric_series.items():
                if agent_id in agent_series:
                    series = agent_series[agent_id]
                    latest = series.get_latest_value()
                    average = series.get_average(duration_seconds)
                    
                    if latest is not None and average is not None:
                        summary[metric_name] = {
                            "latest": latest,
                            "average": average,
                            "agent_id": agent_id
                        }
        else:
            # 集群摘要
            for metric_name, series in self.cluster_metrics.items():
                latest = series.get_latest_value()
                average = series.get_average(duration_seconds)
                
                if latest is not None and average is not None:
                    summary[metric_name] = {
                        "latest": latest,
                        "average": average,
                        "scope": "cluster"
                    }
        
        return summary
    
    # 告警管理
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Alert rule added: {rule.name} ({rule.rule_id})")
    
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Alert rule removed: {rule_id}")
    
    def get_alert_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        return list(self.alert_rules.values())
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[AlertEvent]:
        """获取告警历史"""
        sorted_history = sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)
        return sorted_history[:limit]
    
    # 趋势分析
    async def analyze_trend(
        self, 
        metric_name: str,
        agent_id: Optional[str] = None,
        duration_seconds: float = 3600
    ) -> Dict[str, Any]:
        """分析指标趋势"""
        
        try:
            # 获取指标数据
            if agent_id:
                if (metric_name not in self.metric_series or 
                    agent_id not in self.metric_series[metric_name]):
                    return {"error": "Metric or agent not found"}
                
                series = self.metric_series[metric_name][agent_id]
            else:
                if metric_name not in self.cluster_metrics:
                    return {"error": "Cluster metric not found"}
                
                series = self.cluster_metrics[metric_name]
            
            # 获取指定时间内的数据点
            cutoff_time = time.time() - duration_seconds
            recent_points = [p for p in series.points if p.timestamp >= cutoff_time]
            
            if len(recent_points) < 2:
                return {"error": "Insufficient data points"}
            
            # 计算趋势统计
            values = [p.value for p in recent_points]
            timestamps = [p.timestamp for p in recent_points]
            
            # 基本统计
            min_val = min(values)
            max_val = max(values)
            avg_val = statistics.mean(values)
            median_val = statistics.median(values)
            
            # 线性趋势计算（简单线性回归）
            n = len(values)
            sum_x = sum(timestamps)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(timestamps, values))
            sum_x2 = sum(x * x for x in timestamps)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # 趋势方向
            if abs(slope) < 0.001:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # 变化率
            if len(values) >= 2:
                change_rate = (values[-1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
            else:
                change_rate = 0
            
            return {
                "metric_name": metric_name,
                "agent_id": agent_id,
                "duration_seconds": duration_seconds,
                "data_points": len(recent_points),
                "statistics": {
                    "min": min_val,
                    "max": max_val,
                    "average": avg_val,
                    "median": median_val
                },
                "trend": {
                    "direction": trend_direction,
                    "slope": slope,
                    "change_rate_percent": change_rate
                },
                "analysis_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return {"error": str(e)}
    
    # 内部方法
    async def _collection_loop(self):
        """指标收集循环"""
        
        while True:
            try:
                await asyncio.sleep(self.collection_interval)
                await self._perform_collection()
                
            except asyncio.CancelledError:
                self.logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                self.collector_metrics["collection_errors"] += 1
    
    async def _perform_collection(self):
        """执行指标收集"""
        
        start_time = time.time()
        
        try:
            # 获取所有健康的智能体
            healthy_agents = await self.cluster_manager.get_healthy_agents()
            
            # 并发收集智能体指标
            tasks = []
            for agent in healthy_agents:
                task = self.collect_agent_metrics(agent.agent_id)
                tasks.append(task)
            
            # 等待所有收集任务完成
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集集群级别指标
            await self.collect_cluster_metrics()
            
            # 执行自定义收集器
            for collector in self.custom_collectors:
                try:
                    if asyncio.iscoroutinefunction(collector):
                        await collector()
                    else:
                        collector()
                except Exception as e:
                    self.logger.error(f"Error in custom collector: {e}")
            
            # 更新统计
            collection_time = (time.time() - start_time) * 1000
            self.collector_metrics["collections_performed"] += 1
            
            # 更新平均收集时间
            old_avg = self.collector_metrics["avg_collection_time"]
            new_avg = (
                (old_avg * (self.collector_metrics["collections_performed"] - 1) + collection_time) /
                self.collector_metrics["collections_performed"]
            )
            self.collector_metrics["avg_collection_time"] = new_avg
            
            self.logger.debug(f"Metrics collection completed in {collection_time:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error performing metrics collection: {e}")
            self.collector_metrics["collection_errors"] += 1
    
    async def _alert_check_loop(self):
        """告警检查循环"""
        
        while True:
            try:
                await asyncio.sleep(60.0)  # 每分钟检查一次告警
                await self._check_alerts()
                
            except asyncio.CancelledError:
                self.logger.info("Alert check loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in alert check loop: {e}")
    
    async def _check_alerts(self):
        """检查告警规则"""
        
        try:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # 检查每个智能体的指标
                if rule.metric_name in self.metric_series:
                    for agent_id, series in self.metric_series[rule.metric_name].items():
                        alert_key = f"{rule.rule_id}-{agent_id}"
                        
                        # 评估规则
                        is_alerting = rule.evaluate(series)
                        
                        if is_alerting:
                            # 如果还没有活跃告警，创建新告警
                            if alert_key not in self.active_alerts:
                                alert = AlertEvent(
                                    alert_id=alert_key,
                                    rule_id=rule.rule_id,
                                    agent_id=agent_id,
                                    metric_name=rule.metric_name,
                                    level=rule.level,
                                    message=f"{rule.name}: {rule.description}",
                                    value=series.get_latest_value() or 0.0,
                                    threshold=rule.threshold,
                                    timestamp=time.time()
                                )
                                
                                self.active_alerts[alert_key] = alert
                                self.alert_history.append(alert)
                                self.collector_metrics["alerts_triggered"] += 1
                                
                                self.logger.warning(
                                    f"Alert triggered: {rule.name} for agent {agent_id} "
                                    f"(value: {alert.value}, threshold: {rule.threshold})"
                                )
                        else:
                            # 如果有活跃告警但条件不再满足，解决告警
                            if alert_key in self.active_alerts:
                                alert = self.active_alerts[alert_key]
                                alert.resolve()
                                del self.active_alerts[alert_key]
                                
                                self.logger.info(
                                    f"Alert resolved: {rule.name} for agent {agent_id}"
                                )
                
                # 检查集群级别指标
                if rule.metric_name in self.cluster_metrics:
                    series = self.cluster_metrics[rule.metric_name]
                    alert_key = f"{rule.rule_id}-cluster"
                    
                    is_alerting = rule.evaluate(series)
                    
                    if is_alerting and alert_key not in self.active_alerts:
                        alert = AlertEvent(
                            alert_id=alert_key,
                            rule_id=rule.rule_id,
                            agent_id="cluster",
                            metric_name=rule.metric_name,
                            level=rule.level,
                            message=f"Cluster {rule.name}: {rule.description}",
                            value=series.get_latest_value() or 0.0,
                            threshold=rule.threshold,
                            timestamp=time.time()
                        )
                        
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        self.collector_metrics["alerts_triggered"] += 1
                    
                    elif not is_alerting and alert_key in self.active_alerts:
                        alert = self.active_alerts[alert_key]
                        alert.resolve()
                        del self.active_alerts[alert_key]
            
            # 保持告警历史在限制内
            if len(self.alert_history) > self.max_alert_history:
                self.alert_history = self.alert_history[-self.max_alert_history:]
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _standardize_metrics(self, raw_metrics: Dict[str, Any], agent_id: str) -> Dict[str, float]:
        """标准化指标格式"""
        
        standardized = {}
        
        try:
            # 映射标准指标名称
            metric_mapping = {
                "cpu": "cpu_usage_percent",
                "cpu_percent": "cpu_usage_percent",
                "cpu_usage": "cpu_usage_percent",
                "memory": "memory_usage_percent", 
                "memory_percent": "memory_usage_percent",
                "memory_usage": "memory_usage_percent",
                "storage": "storage_usage_percent",
                "disk": "storage_usage_percent",
                "disk_usage": "storage_usage_percent",
                "gpu": "gpu_usage_percent",
                "gpu_usage": "gpu_usage_percent",
                "network": "network_io_mbps",
                "network_io": "network_io_mbps",
                "tasks": "active_tasks",
                "active_tasks": "active_tasks",
                "requests": "total_requests",
                "total_requests": "total_requests",
                "errors": "failed_requests",
                "failed_requests": "failed_requests",
                "response_time": "avg_response_time",
                "avg_response_time": "avg_response_time"
            }
            
            # 处理原始指标
            for key, value in raw_metrics.items():
                try:
                    # 转换为标准名称
                    standard_key = metric_mapping.get(key.lower(), key)
                    
                    # 确保值是数字
                    if isinstance(value, (int, float)):
                        standardized[standard_key] = float(value)
                    elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        standardized[standard_key] = float(value)
                        
                except (ValueError, TypeError):
                    self.logger.debug(f"Could not convert metric {key}={value} to float")
                    continue
            
            # 计算派生指标
            if ("total_requests" in standardized and 
                "failed_requests" in standardized and
                standardized["total_requests"] > 0):
                standardized["error_rate"] = standardized["failed_requests"] / standardized["total_requests"]
            
            return standardized
            
        except Exception as e:
            self.logger.error(f"Error standardizing metrics for agent {agent_id}: {e}")
            return {}
    
    def _create_resource_usage_from_metrics(self, metrics: Dict[str, float]) -> ResourceUsage:
        """从指标创建资源使用对象"""
        
        return ResourceUsage(
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0.0),
            memory_usage_percent=metrics.get("memory_usage_percent", 0.0),
            storage_usage_percent=metrics.get("storage_usage_percent", 0.0),
            gpu_usage_percent=metrics.get("gpu_usage_percent", 0.0),
            network_io_mbps=metrics.get("network_io_mbps", 0.0),
            active_tasks=int(metrics.get("active_tasks", 0)),
            total_requests=int(metrics.get("total_requests", 0)),
            failed_requests=int(metrics.get("failed_requests", 0)),
            avg_response_time=metrics.get("avg_response_time", 0.0),
            timestamp=time.time()
        )
    
    async def _store_metrics(self, agent_id: str, metrics: Dict[str, float]):
        """存储智能体指标"""
        
        try:
            for metric_name, value in metrics.items():
                # 获取或创建指标序列
                if metric_name not in self.metric_series:
                    self.metric_series[metric_name] = {}
                
                if agent_id not in self.metric_series[metric_name]:
                    self.metric_series[metric_name][agent_id] = MetricSeries(
                        metric_name=metric_name,
                        agent_id=agent_id
                    )
                
                # 添加数据点
                self.metric_series[metric_name][agent_id].add_point(value)
                self.collector_metrics["metrics_collected"] += 1
                
        except Exception as e:
            self.logger.error(f"Error storing metrics for agent {agent_id}: {e}")
    
    async def _store_cluster_metrics(self, metrics: Dict[str, Any]):
        """存储集群指标"""
        
        try:
            for metric_name, value in metrics.items():
                # 确保值是数字
                if not isinstance(value, (int, float)):
                    continue
                
                # 获取或创建集群指标序列
                if metric_name not in self.cluster_metrics:
                    self.cluster_metrics[metric_name] = MetricSeries(
                        metric_name=metric_name,
                        agent_id="cluster"
                    )
                
                # 添加数据点
                self.cluster_metrics[metric_name].add_point(float(value))
                
        except Exception as e:
            self.logger.error(f"Error storing cluster metrics: {e}")
    
    async def _persist_metrics(self):
        """持久化指标数据"""
        
        if not self.storage_backend:
            return
        
        try:
            # 准备指标数据
            metrics_data = {
                "agent_metrics": {},
                "cluster_metrics": {},
                "collector_metrics": self.collector_metrics,
                "timestamp": time.time()
            }
            
            # 序列化智能体指标
            for metric_name, agent_series in self.metric_series.items():
                metrics_data["agent_metrics"][metric_name] = {}
                for agent_id, series in agent_series.items():
                    # 只保存最近的指标点
                    recent_points = series.points[-100:] if series.points else []
                    metrics_data["agent_metrics"][metric_name][agent_id] = [
                        p.to_dict() for p in recent_points
                    ]
            
            # 序列化集群指标
            for metric_name, series in self.cluster_metrics.items():
                recent_points = series.points[-100:] if series.points else []
                metrics_data["cluster_metrics"][metric_name] = [
                    p.to_dict() for p in recent_points
                ]
            
            # 保存到存储后端
            await self.storage_backend.save_metrics_data(metrics_data)
            
            self.logger.debug("Metrics data persisted successfully")
            
        except Exception as e:
            self.logger.error(f"Error persisting metrics data: {e}")
    
    def _create_default_alert_rules(self):
        """创建默认告警规则"""
        
        # CPU使用率过高告警
        self.alert_rules["high_cpu"] = AlertRule(
            rule_id="high_cpu",
            name="High CPU Usage",
            metric_name="cpu_usage_percent",
            condition=">",
            threshold=80.0,
            level=AlertLevel.WARNING,
            description="CPU usage exceeds 80%"
        )
        
        # 内存使用率过高告警
        self.alert_rules["high_memory"] = AlertRule(
            rule_id="high_memory",
            name="High Memory Usage",
            metric_name="memory_usage_percent",
            condition=">",
            threshold=85.0,
            level=AlertLevel.WARNING,
            description="Memory usage exceeds 85%"
        )
        
        # 错误率过高告警
        self.alert_rules["high_error_rate"] = AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            metric_name="error_rate",
            condition=">",
            threshold=0.1,
            level=AlertLevel.ERROR,
            description="Error rate exceeds 10%"
        )
        
        # 响应时间过长告警
        self.alert_rules["slow_response"] = AlertRule(
            rule_id="slow_response",
            name="Slow Response Time",
            metric_name="avg_response_time",
            condition=">",
            threshold=5000.0,  # 5秒
            level=AlertLevel.WARNING,
            description="Average response time exceeds 5 seconds"
        )
    
    def add_custom_collector(self, collector: Callable):
        """添加自定义指标收集器"""
        self.custom_collectors.append(collector)
    
    def remove_custom_collector(self, collector: Callable):
        """移除自定义指标收集器"""
        if collector in self.custom_collectors:
            self.custom_collectors.remove(collector)
    
    def get_collector_metrics(self) -> Dict[str, Any]:
        """获取收集器性能指标"""
        return {
            **self.collector_metrics,
            "active_series_count": sum(len(agents) for agents in self.metric_series.values()),
            "cluster_series_count": len(self.cluster_metrics),
            "active_alert_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "active_alerts": len(self.active_alerts),
            "alert_history_size": len(self.alert_history)
        }