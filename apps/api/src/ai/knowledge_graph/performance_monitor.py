"""
SPARQL查询性能监控器

提供全面的性能监控和分析功能：
- 查询执行时间监控
- 资源使用统计
- 性能基准测试
- 瓶颈识别和告警
- 性能报告生成
"""

import time
import asyncio
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """指标类型"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUERY_COUNT = "query_count"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryProfile:
    """查询性能配置文件"""
    query_id: str
    query_text: str
    start_time: float
    end_time: Optional[float] = None
    execution_time_ms: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    result_count: int = 0
    cache_hit: bool = False
    error_message: Optional[str] = None
    optimization_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self, success: bool = True, error: str = None):
        """完成性能分析"""
        if self.end_time is None:
            self.end_time = time.time()
        
        if self.execution_time_ms is None:
            self.execution_time_ms = (self.end_time - self.start_time) * 1000
        
        if not success and error:
            self.error_message = error


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_type: MetricType
    threshold: float
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    level: AlertLevel
    enabled: bool = True
    window_size: int = 10  # 滑动窗口大小
    callback: Optional[Callable] = None


@dataclass
class Alert:
    """告警"""
    rule_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_size: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_size))
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def record_metric(self, metric: PerformanceMetric):
        """记录指标"""
        with self._lock:
            self.metrics[metric.name].append(metric)
    
    def get_metrics(
        self, 
        metric_name: str, 
        limit: int = None,
        since: float = None
    ) -> List[PerformanceMetric]:
        """获取指标"""
        with self._lock:
            if metric_name not in self.metrics:
                return []
            
            metrics = list(self.metrics[metric_name])
            
            # 时间过滤
            if since is not None:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            # 限制数量
            if limit is not None:
                metrics = metrics[-limit:]
            
            return metrics
    
    def get_metric_stats(
        self, 
        metric_name: str,
        window_minutes: int = 60
    ) -> Dict[str, float]:
        """获取指标统计"""
        since = time.time() - (window_minutes * 60)
        metrics = self.get_metrics(metric_name, since=since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """计算百分位数"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def clear_metrics(self, metric_name: str = None):
        """清空指标"""
        with self._lock:
            if metric_name:
                if metric_name in self.metrics:
                    self.metrics[metric_name].clear()
            else:
                self.metrics.clear()


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """检查告警"""
        with self._lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    self._check_single_rule(rule, metrics_collector)
                except Exception as e:
                    logger.error(f"检查告警规则 {rule_name} 失败: {e}")
    
    def _check_single_rule(self, rule: AlertRule, metrics_collector: MetricsCollector):
        """检查单个规则"""
        # 获取最近的指标
        metrics = metrics_collector.get_metrics(
            rule.metric_type.value,
            limit=rule.window_size
        )
        
        if not metrics:
            return
        
        # 计算当前值（使用平均值）
        current_value = statistics.mean([m.value for m in metrics])
        
        # 检查条件
        triggered = self._evaluate_condition(
            current_value, 
            rule.threshold, 
            rule.condition
        )
        
        alert_key = f"{rule.name}_{rule.metric_type.value}"
        
        if triggered:
            if alert_key not in self.active_alerts:
                # 创建新告警
                alert = Alert(
                    rule_name=rule.name,
                    level=rule.level,
                    message=f"{rule.metric_type.value} {rule.condition} {rule.threshold}",
                    value=current_value,
                    threshold=rule.threshold,
                    timestamp=time.time()
                )
                
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
                
                # 回调处理
                if rule.callback:
                    try:
                        rule.callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调执行失败: {e}")
                
                logger.warning(f"告警触发: {alert.message} (值: {current_value})")
        
        else:
            # 解决告警
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                del self.active_alerts[alert_key]
                
                logger.info(f"告警已解决: {alert.message}")
    
    def _evaluate_condition(self, value: float, threshold: float, condition: str) -> bool:
        """评估条件"""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        else:
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            return list(self.alert_history)[-limit:]


class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_task = None
    
    def start_monitoring(self, interval: float = 5.0):
        """开始监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval)
        )
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            try:
                # CPU使用率
                cpu_percent = self.process.cpu_percent()
                
                # 内存使用
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 记录系统指标
                # 这里应该与MetricsCollector集成
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"系统资源监控失败: {e}")
                await asyncio.sleep(interval)
    
    def get_current_stats(self) -> Dict[str, float]:
        """获取当前统计"""
        try:
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_mb,
                "threads": self.process.num_threads(),
                "open_files": len(self.process.open_files())
            }
        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            return {}


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self._lock = threading.Lock()
    
    def start_profile(self, query_id: str, query_text: str) -> QueryProfile:
        """开始性能分析"""
        profile = QueryProfile(
            query_id=query_id,
            query_text=query_text,
            start_time=time.time()
        )
        
        with self._lock:
            self.active_profiles[query_id] = profile
        
        return profile
    
    def end_profile(
        self, 
        query_id: str, 
        success: bool = True,
        result_count: int = 0,
        cache_hit: bool = False,
        error: str = None
    ):
        """结束性能分析"""
        with self._lock:
            if query_id not in self.active_profiles:
                logger.warning(f"找不到查询分析: {query_id}")
                return
            
            profile = self.active_profiles[query_id]
            profile.finalize(success, error)
            profile.result_count = result_count
            profile.cache_hit = cache_hit
            
            # 记录指标
            self._record_profile_metrics(profile)
            
            del self.active_profiles[query_id]
    
    def _record_profile_metrics(self, profile: QueryProfile):
        """记录分析指标"""
        timestamp = profile.end_time
        
        # 执行时间
        if profile.execution_time_ms is not None:
            self.metrics_collector.record_metric(PerformanceMetric(
                name=MetricType.EXECUTION_TIME.value,
                value=profile.execution_time_ms,
                timestamp=timestamp,
                tags={"success": str(profile.error_message is None)}
            ))
        
        # 查询计数
        self.metrics_collector.record_metric(PerformanceMetric(
            name=MetricType.QUERY_COUNT.value,
            value=1,
            timestamp=timestamp,
            tags={"cache_hit": str(profile.cache_hit)}
        ))
        
        # 错误率
        if profile.error_message:
            self.metrics_collector.record_metric(PerformanceMetric(
                name=MetricType.ERROR_RATE.value,
                value=1,
                timestamp=timestamp
            ))
        
        # 缓存命中率
        if profile.cache_hit:
            self.metrics_collector.record_metric(PerformanceMetric(
                name=MetricType.CACHE_HIT_RATE.value,
                value=1,
                timestamp=timestamp
            ))


class SPARQLPerformanceMonitor:
    """SPARQL性能监控器主类"""
    
    def __init__(self, 
                 max_metrics: int = 10000,
                 monitoring_interval: float = 5.0):
        self.metrics_collector = MetricsCollector(max_metrics)
        self.alert_manager = AlertManager()
        self.resource_monitor = SystemResourceMonitor()
        self.profiler = PerformanceProfiler(self.metrics_collector)
        
        self.monitoring_interval = monitoring_interval
        self._monitoring_task = None
        self._running = False
        
        # 默认告警规则
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        # 查询执行时间告警
        self.alert_manager.add_rule(AlertRule(
            name="slow_query",
            metric_type=MetricType.EXECUTION_TIME,
            threshold=5000,  # 5秒
            condition="gt",
            level=AlertLevel.WARNING
        ))
        
        # 错误率告警
        self.alert_manager.add_rule(AlertRule(
            name="high_error_rate",
            metric_type=MetricType.ERROR_RATE,
            threshold=0.1,  # 10%
            condition="gt",
            level=AlertLevel.ERROR
        ))
        
        # 内存使用告警
        self.alert_manager.add_rule(AlertRule(
            name="high_memory_usage",
            metric_type=MetricType.MEMORY_USAGE,
            threshold=1024,  # 1GB
            condition="gt",
            level=AlertLevel.WARNING
        ))
    
    async def start_monitoring(self):
        """开始监控"""
        if self._running:
            return
        
        self._running = True
        
        # 启动资源监控
        self.resource_monitor.start_monitoring(self.monitoring_interval)
        
        # 启动告警检查任务
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop()
        )
        
        logger.info("SPARQL性能监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止资源监控
        self.resource_monitor.stop_monitoring()
        
        # 停止告警检查
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        logger.info("SPARQL性能监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 检查告警
                self.alert_manager.check_alerts(self.metrics_collector)
                
                # 记录系统资源指标
                system_stats = self.resource_monitor.get_current_stats()
                timestamp = time.time()
                
                for metric_name, value in system_stats.items():
                    self.metrics_collector.record_metric(PerformanceMetric(
                        name=metric_name,
                        value=value,
                        timestamp=timestamp
                    ))
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def start_query_profile(self, query_id: str, query_text: str) -> QueryProfile:
        """开始查询性能分析"""
        return self.profiler.start_profile(query_id, query_text)
    
    def end_query_profile(
        self, 
        query_id: str,
        success: bool = True,
        result_count: int = 0,
        cache_hit: bool = False,
        error: str = None
    ):
        """结束查询性能分析"""
        self.profiler.end_profile(
            query_id, success, result_count, cache_hit, error
        )
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        # 主要性能指标统计
        for metric_type in MetricType:
            stats = self.metrics_collector.get_metric_stats(
                metric_type.value, 
                window_minutes
            )
            if stats:
                summary[metric_type.value] = stats
        
        # 活跃告警
        summary["active_alerts"] = len(self.alert_manager.get_active_alerts())
        
        # 系统资源
        summary["system_stats"] = self.resource_monitor.get_current_stats()
        
        return summary
    
    def get_detailed_report(self, window_minutes: int = 60) -> Dict[str, Any]:
        """获取详细性能报告"""
        report = {
            "timestamp": time.time(),
            "window_minutes": window_minutes,
            "performance_summary": self.get_performance_summary(window_minutes),
            "active_alerts": self.alert_manager.get_active_alerts(),
            "recent_alerts": self.alert_manager.get_alert_history(20),
            "top_slow_queries": self._get_slow_queries(10),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _get_slow_queries(self, limit: int) -> List[Dict[str, Any]]:
        """获取慢查询列表"""
        # 这里应该从查询历史中提取慢查询
        # 简化实现返回空列表
        return []
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # 基于当前指标生成建议
        exec_time_stats = self.metrics_collector.get_metric_stats(
            MetricType.EXECUTION_TIME.value, 60
        )
        
        if exec_time_stats and exec_time_stats.get("mean", 0) > 1000:
            recommendations.append("平均查询时间较高，建议检查查询优化和索引使用")
        
        cache_stats = self.metrics_collector.get_metric_stats(
            MetricType.CACHE_HIT_RATE.value, 60
        )
        
        if cache_stats and cache_stats.get("mean", 0) < 0.5:
            recommendations.append("缓存命中率较低，建议检查缓存配置和策略")
        
        return recommendations
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_manager.add_rule(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        self.alert_manager.remove_rule(rule_name)
    
    def clear_metrics(self, metric_name: str = None):
        """清空指标"""
        self.metrics_collector.clear_metrics(metric_name)


# 创建默认性能监控器实例
default_performance_monitor = SPARQLPerformanceMonitor()


async def start_sparql_monitoring():
    """启动SPARQL性能监控的便捷函数"""
    await default_performance_monitor.start_monitoring()


async def stop_sparql_monitoring():
    """停止SPARQL性能监控的便捷函数"""
    await default_performance_monitor.stop_monitoring()


def get_performance_report(window_minutes: int = 60) -> Dict[str, Any]:
    """获取性能报告的便捷函数"""
    return default_performance_monitor.get_detailed_report(window_minutes)