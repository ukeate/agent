"""
性能监控系统

实时监控行为分析系统的性能指标，提供预警和优化建议。
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # 'system', 'database', 'analysis', 'websocket'
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def is_warning(self) -> bool:
        """是否达到警告阈值"""
        return self.threshold_warning is not None and self.value >= self.threshold_warning
    
    def is_critical(self) -> bool:
        """是否达到严重阈值"""
        return self.threshold_critical is not None and self.value >= self.threshold_critical

@dataclass
class PerformanceAlert:
    """性能告警"""
    metric_name: str
    current_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: datetime
    category: str

class PerformanceCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.collection_interval = 5  # 5秒收集间隔
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动性能收集"""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("性能监控已启动")
    
    async def stop(self):
        """停止性能收集"""
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("性能监控已停止")
    
    async def _collection_loop(self):
        """性能收集循环"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"性能收集错误: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        timestamp = datetime.utcnow()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_metric = PerformanceMetric(
            name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp,
            category="system",
            threshold_warning=70.0,
            threshold_critical=90.0
        )
        self.metrics_history["cpu_usage"].append(cpu_metric)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_metric = PerformanceMetric(
            name="memory_usage",
            value=memory.percent,
            unit="percent", 
            timestamp=timestamp,
            category="system",
            threshold_warning=80.0,
            threshold_critical=95.0
        )
        self.metrics_history["memory_usage"].append(memory_metric)
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_read_metric = PerformanceMetric(
                name="disk_read_bytes",
                value=disk_io.read_bytes,
                unit="bytes",
                timestamp=timestamp,
                category="system"
            )
            self.metrics_history["disk_read_bytes"].append(disk_read_metric)
            
            disk_write_metric = PerformanceMetric(
                name="disk_write_bytes", 
                value=disk_io.write_bytes,
                unit="bytes",
                timestamp=timestamp,
                category="system"
            )
            self.metrics_history["disk_write_bytes"].append(disk_write_metric)
        
        # 网络IO
        net_io = psutil.net_io_counters()
        if net_io:
            net_sent_metric = PerformanceMetric(
                name="network_sent_bytes",
                value=net_io.bytes_sent,
                unit="bytes",
                timestamp=timestamp,
                category="system"
            )
            self.metrics_history["network_sent_bytes"].append(net_sent_metric)
            
            net_recv_metric = PerformanceMetric(
                name="network_recv_bytes",
                value=net_io.bytes_recv,
                unit="bytes", 
                timestamp=timestamp,
                category="system"
            )
            self.metrics_history["network_recv_bytes"].append(net_recv_metric)
    
    async def _collect_application_metrics(self):
        """收集应用程序指标"""
        timestamp = datetime.utcnow()
        
        # 这里可以添加具体的应用程序指标收集
        # 例如：事件处理速率、数据库查询时间、WebSocket连接数等
        
        # 示例：模拟事件处理速率
        # 实际实现中应该从事件收集器获取真实数据
        event_rate = 100  # 每秒事件数，应该从实际系统获取
        event_rate_metric = PerformanceMetric(
            name="event_processing_rate",
            value=event_rate,
            unit="events/second",
            timestamp=timestamp,
            category="analysis",
            threshold_warning=1000.0,
            threshold_critical=2000.0
        )
        self.metrics_history["event_processing_rate"].append(event_rate_metric)
    
    def get_latest_metrics(self, category: Optional[str] = None) -> Dict[str, PerformanceMetric]:
        """获取最新的性能指标"""
        latest_metrics = {}
        
        for metric_name, history in self.metrics_history.items():
            if history:
                latest_metric = history[-1]
                if category is None or latest_metric.category == category:
                    latest_metrics[metric_name] = latest_metric
        
        return latest_metrics
    
    def get_metric_history(self, metric_name: str, duration: timedelta) -> List[PerformanceMetric]:
        """获取指定时间段内的指标历史"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.utcnow() - duration
        history = self.metrics_history[metric_name]
        
        return [metric for metric in history if metric.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, metric_name: str, duration: timedelta) -> Dict[str, float]:
        """获取指标统计信息"""
        history = self.get_metric_history(metric_name, duration)
        
        if not history:
            return {}
        
        values = [metric.value for metric in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.alert_handlers: List[Callable[[PerformanceAlert], None]] = []
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """分析当前性能状态"""
        latest_metrics = self.collector.get_latest_metrics()
        alerts = []
        
        # 检查每个指标是否超过阈值
        for metric_name, metric in latest_metrics.items():
            if metric.is_critical():
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    current_value=metric.value,
                    threshold=metric.threshold_critical,
                    severity="critical",
                    message=f"{metric_name} 达到严重阈值: {metric.value}{metric.unit}",
                    timestamp=metric.timestamp,
                    category=metric.category
                )
                alerts.append(alert)
                await self._handle_alert(alert)
                
            elif metric.is_warning():
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    current_value=metric.value,
                    threshold=metric.threshold_warning,
                    severity="warning",
                    message=f"{metric_name} 达到警告阈值: {metric.value}{metric.unit}",
                    timestamp=metric.timestamp,
                    category=metric.category
                )
                alerts.append(alert)
                await self._handle_alert(alert)
        
        # 生成性能分析报告
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": self._get_overall_status(alerts),
            "active_alerts": [asdict(alert) for alert in alerts],
            "metrics_summary": self._get_metrics_summary(latest_metrics),
            "performance_score": self._calculate_performance_score(latest_metrics),
            "recommendations": await self._generate_recommendations(latest_metrics)
        }
    
    async def _handle_alert(self, alert: PerformanceAlert):
        """处理告警"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"告警处理器错误: {e}")
    
    def _get_overall_status(self, alerts: List[PerformanceAlert]) -> str:
        """获取整体状态"""
        if any(alert.severity == "critical" for alert in alerts):
            return "critical"
        elif any(alert.severity == "warning" for alert in alerts):
            return "warning"
        else:
            return "healthy"
    
    def _get_metrics_summary(self, metrics: Dict[str, PerformanceMetric]) -> Dict[str, Any]:
        """获取指标摘要"""
        categories = defaultdict(list)
        for metric in metrics.values():
            categories[metric.category].append(metric)
        
        summary = {}
        for category, category_metrics in categories.items():
            summary[category] = {
                "metric_count": len(category_metrics),
                "alerts": sum(1 for m in category_metrics if m.is_warning() or m.is_critical()),
                "critical_alerts": sum(1 for m in category_metrics if m.is_critical())
            }
        
        return summary
    
    def _calculate_performance_score(self, metrics: Dict[str, PerformanceMetric]) -> float:
        """计算性能分数 (0-100)"""
        if not metrics:
            return 100.0
        
        total_score = 0.0
        count = 0
        
        for metric in metrics.values():
            if metric.threshold_critical is not None:
                # 根据当前值相对于阈值的比例计算分数
                ratio = min(metric.value / metric.threshold_critical, 1.0)
                score = max(0, 100 - (ratio * 100))
                total_score += score
                count += 1
        
        return total_score / count if count > 0 else 100.0
    
    async def _generate_recommendations(self, metrics: Dict[str, PerformanceMetric]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        # CPU使用率建议
        cpu_metric = metrics.get("cpu_usage")
        if cpu_metric and cpu_metric.value > 80:
            recommendations.append("CPU使用率过高，建议优化算法或增加并发处理能力")
        
        # 内存使用建议
        memory_metric = metrics.get("memory_usage")
        if memory_metric and memory_metric.value > 85:
            recommendations.append("内存使用率过高，建议优化数据结构或增加内存")
        
        # 事件处理速率建议
        event_rate_metric = metrics.get("event_processing_rate")
        if event_rate_metric and event_rate_metric.value > 1500:
            recommendations.append("事件处理速率过高，建议增加缓冲区大小或优化处理逻辑")
        
        return recommendations

class PerformanceMonitor:
    """性能监控主类"""
    
    def __init__(self):
        self.collector = PerformanceCollector()
        self.analyzer = PerformanceAnalyzer(self.collector)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # 30秒分析间隔
        
        # 添加默认告警处理器
        self.analyzer.add_alert_handler(self._log_alert)
    
    async def start(self):
        """启动性能监控"""
        await self.collector.start()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("性能监控系统已启动")
    
    async def stop(self):
        """停止性能监控"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self.collector.stop()
        logger.info("性能监控系统已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                await self.analyzer.analyze_performance()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能监控循环错误: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _log_alert(self, alert: PerformanceAlert):
        """记录告警"""
        if alert.severity == "critical":
            logger.critical(f"性能严重告警: {alert.message}")
        else:
            logger.warning(f"性能告警: {alert.message}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return await self.analyzer.analyze_performance()
    
    def get_current_metrics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """获取当前指标"""
        metrics = self.collector.get_latest_metrics(category)
        return {name: asdict(metric) for name, metric in metrics.items()}
    
    def get_metrics_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """获取指标历史"""
        duration = timedelta(hours=hours)
        history = self.collector.get_metric_history(metric_name, duration)
        return [asdict(metric) for metric in history]
    
    def get_metrics_statistics(self, metric_name: str, hours: int = 1) -> Dict[str, float]:
        """获取指标统计"""
        duration = timedelta(hours=hours)
        return self.collector.get_metric_statistics(metric_name, duration)

# 全局监控实例
performance_monitor = PerformanceMonitor()