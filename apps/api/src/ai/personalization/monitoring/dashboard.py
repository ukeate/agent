"""监控仪表板配置和实时监控"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import redis.asyncio as redis_async
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import numpy as np
from .performance import PerformanceMonitor, PerformanceOptimizer
from .tracing import DistributedTracer, TraceAnalyzer

logger = get_logger(__name__)

class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self,
                 monitor: PerformanceMonitor,
                 optimizer: PerformanceOptimizer,
                 tracer: DistributedTracer,
                 redis: redis_async.Redis):
        
        self.monitor = monitor
        self.optimizer = optimizer
        self.tracer = tracer
        self.redis = redis
        
        # WebSocket连接管理
        self.active_connections: List[WebSocket] = []
        
        # 监控任务
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 告警配置
        self.alert_config = {
            'latency_threshold': 200,  # ms
            'error_rate_threshold': 0.01,  # 1%
            'cpu_threshold': 80,  # %
            'memory_threshold': 85,  # %
            'alert_cooldown': 300  # 5分钟冷却时间
        }
        
        # 告警历史
        self.alert_history: List[Dict[str, Any]] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
    async def connect_websocket(self, websocket: WebSocket):
        """连接WebSocket客户端"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # 发送初始数据
        await self.send_initial_data(websocket)
        
    def disconnect_websocket(self, websocket: WebSocket):
        """断开WebSocket连接"""
        self.active_connections.remove(websocket)
        
    async def send_initial_data(self, websocket: WebSocket):
        """发送初始监控数据"""
        metrics = await self.monitor.get_current_metrics()
        
        initial_data = {
            "type": "initial",
            "data": {
                "metrics": self._serialize_metrics(metrics),
                "alerts": await self.monitor.check_performance_alerts(),
                "optimization_status": self.optimizer.optimization_history[-1] if self.optimizer.optimization_history else None
            }
        }
        
        await websocket.send_json(initial_data)
        
    async def broadcast_metrics(self, data: Dict[str, Any]):
        """广播指标到所有连接的客户端"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {e}")
                disconnected.append(connection)
                
        # 清理断开的连接
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                
    async def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring_task and not self.monitoring_task.done():
            return
            
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """停止监控"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                raise
                
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                # 收集指标
                metrics = await self.monitor.get_current_metrics()
                
                # 检查告警
                alerts = await self._check_alerts(metrics)
                
                # 自动优化
                optimizations = None
                if self._should_optimize(metrics):
                    optimizations = await self.optimizer.auto_optimize()
                    
                # 准备广播数据
                broadcast_data = {
                    "type": "update",
                    "timestamp": utc_now().isoformat(),
                    "data": {
                        "metrics": self._serialize_metrics(metrics),
                        "alerts": alerts,
                        "optimizations": optimizations
                    }
                }
                
                # 广播到所有客户端
                await self.broadcast_metrics(broadcast_data)
                
                # 存储历史数据
                await self._store_metrics_history(metrics)
                
                # 等待下次更新
                await asyncio.sleep(5)  # 5秒更新一次
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
                
    async def _check_alerts(self, metrics) -> List[Dict[str, Any]]:
        """检查告警条件"""
        alerts = []
        now = utc_now()
        
        # 延迟告警
        if metrics.p99_latency > self.alert_config['latency_threshold']:
            alert_key = "high_latency"
            if self._can_send_alert(alert_key, now):
                alert = {
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"P99延迟过高: {metrics.p99_latency:.2f}ms",
                    "value": metrics.p99_latency,
                    "threshold": self.alert_config['latency_threshold'],
                    "timestamp": now.isoformat()
                }
                alerts.append(alert)
                self._record_alert(alert_key, now, alert)
                
        # 错误率告警
        if metrics.error_rate > self.alert_config['error_rate_threshold']:
            alert_key = "high_error_rate"
            if self._can_send_alert(alert_key, now):
                alert = {
                    "type": "high_error_rate",
                    "severity": "critical",
                    "message": f"错误率过高: {metrics.error_rate:.2%}",
                    "value": metrics.error_rate,
                    "threshold": self.alert_config['error_rate_threshold'],
                    "timestamp": now.isoformat()
                }
                alerts.append(alert)
                self._record_alert(alert_key, now, alert)
                
        # CPU告警
        if metrics.cpu_usage > self.alert_config['cpu_threshold']:
            alert_key = "high_cpu"
            if self._can_send_alert(alert_key, now):
                alert = {
                    "type": "high_cpu",
                    "severity": "warning",
                    "message": f"CPU使用率过高: {metrics.cpu_usage:.1f}%",
                    "value": metrics.cpu_usage,
                    "threshold": self.alert_config['cpu_threshold'],
                    "timestamp": now.isoformat()
                }
                alerts.append(alert)
                self._record_alert(alert_key, now, alert)
                
        # 内存告警
        if metrics.memory_usage > self.alert_config['memory_threshold']:
            alert_key = "high_memory"
            if self._can_send_alert(alert_key, now):
                alert = {
                    "type": "high_memory",
                    "severity": "warning",
                    "message": f"内存使用率过高: {metrics.memory_usage:.1f}%",
                    "value": metrics.memory_usage,
                    "threshold": self.alert_config['memory_threshold'],
                    "timestamp": now.isoformat()
                }
                alerts.append(alert)
                self._record_alert(alert_key, now, alert)
                
        return alerts
        
    def _can_send_alert(self, alert_key: str, now: datetime) -> bool:
        """检查是否可以发送告警（避免告警风暴）"""
        if alert_key not in self.last_alert_time:
            return True
            
        last_time = self.last_alert_time[alert_key]
        cooldown = timedelta(seconds=self.alert_config['alert_cooldown'])
        
        return now - last_time > cooldown
        
    def _record_alert(self, alert_key: str, now: datetime, alert: Dict[str, Any]):
        """记录告警"""
        self.last_alert_time[alert_key] = now
        self.alert_history.append(alert)
        
        # 限制历史记录大小
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
            
    def _should_optimize(self, metrics) -> bool:
        """判断是否需要自动优化"""
        # 基于指标判断是否需要优化
        return (
            metrics.p99_latency > 150 or
            metrics.error_rate > 0.005 or
            metrics.cpu_usage > 70 or
            metrics.memory_usage > 80
        )
        
    def _serialize_metrics(self, metrics) -> Dict[str, Any]:
        """序列化指标对象"""
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "latency": {
                "p50": metrics.p50_latency,
                "p95": metrics.p95_latency,
                "p99": metrics.p99_latency
            },
            "throughput": metrics.throughput,
            "error_rate": metrics.error_rate,
            "cache_hit_rate": metrics.cache_hit_rate,
            "resources": {
                "cpu": metrics.cpu_usage,
                "memory": metrics.memory_usage
            },
            "connections": metrics.active_connections,
            "queue_size": metrics.queue_size
        }
        
    async def _store_metrics_history(self, metrics):
        """存储指标历史"""
        # 存储到Redis时间序列
        timestamp = int(metrics.timestamp.timestamp())
        
        # 存储各项指标
        await self.redis.zadd(
            "metrics:latency:p99",
            {str(timestamp): metrics.p99_latency}
        )
        
        await self.redis.zadd(
            "metrics:throughput",
            {str(timestamp): metrics.throughput}
        )
        
        await self.redis.zadd(
            "metrics:error_rate",
            {str(timestamp): metrics.error_rate}
        )
        
        await self.redis.zadd(
            "metrics:cpu",
            {str(timestamp): metrics.cpu_usage}
        )
        
        await self.redis.zadd(
            "metrics:memory",
            {str(timestamp): metrics.memory_usage}
        )
        
        # 设置过期时间（保留7天数据）
        expire_time = 7 * 24 * 3600
        await self.redis.expire("metrics:latency:p99", expire_time)
        await self.redis.expire("metrics:throughput", expire_time)
        await self.redis.expire("metrics:error_rate", expire_time)
        await self.redis.expire("metrics:cpu", expire_time)
        await self.redis.expire("metrics:memory", expire_time)
        
    async def get_metrics_history(self,
                                 metric_type: str,
                                 start_time: datetime,
                                 end_time: datetime) -> List[Dict[str, Any]]:
        """获取指标历史数据"""
        
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        
        # 从Redis获取数据
        key = f"metrics:{metric_type}"
        data = await self.redis.zrangebyscore(
            key,
            start_ts,
            end_ts,
            withscores=True
        )
        
        # 格式化数据
        result = []
        for value, timestamp in data:
            result.append({
                "timestamp": datetime.fromtimestamp(float(timestamp)).isoformat(),
                "value": float(value)
            })
            
        return result
        
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """获取仪表板摘要"""
        
        # 获取当前指标
        current_metrics = await self.monitor.get_current_metrics()
        
        # 获取最近1小时的历史数据
        end_time = utc_now()
        start_time = end_time - timedelta(hours=1)
        
        latency_history = await self.get_metrics_history(
            "latency:p99",
            start_time,
            end_time
        )
        
        throughput_history = await self.get_metrics_history(
            "throughput",
            start_time,
            end_time
        )
        
        # 获取告警统计
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > start_time
        ]
        
        alert_stats = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            if alert_type not in alert_stats:
                alert_stats[alert_type] = 0
            alert_stats[alert_type] += 1
            
        # 获取优化历史
        recent_optimizations = [
            opt for opt in self.optimizer.optimization_history
            if opt['timestamp'] > start_time
        ]
        
        return {
            "current_metrics": self._serialize_metrics(current_metrics),
            "history": {
                "latency": latency_history[-20:],  # 最近20个数据点
                "throughput": throughput_history[-20:]
            },
            "alerts": {
                "recent": recent_alerts[-10:],  # 最近10个告警
                "statistics": alert_stats
            },
            "optimizations": {
                "recent": recent_optimizations[-5:],  # 最近5次优化
                "total": len(self.optimizer.optimization_history)
            },
            "system_status": self._get_system_status(current_metrics)
        }
        
    def _get_system_status(self, metrics) -> str:
        """获取系统状态"""
        
        if metrics.error_rate > 0.05:
            return "critical"
        elif metrics.error_rate > 0.01 or metrics.p99_latency > 500:
            return "degraded"
        elif metrics.p99_latency > 200 or metrics.cpu_usage > 80:
            return "warning"
        else:
            return "healthy"

def setup_monitoring_endpoints(app: FastAPI, dashboard: MonitoringDashboard):
    """设置监控端点"""
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus指标端点"""
        registry = CollectorRegistry()
        # 这里应该注册所有的Prometheus collectors
        metrics = generate_latest(registry)
        return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
        
    @app.websocket("/ws/monitoring")
    async def websocket_monitoring(websocket: WebSocket):
        """WebSocket实时监控"""
        await dashboard.connect_websocket(websocket)
        
        try:
            while True:
                # 保持连接活跃
                await websocket.receive_text()
        except WebSocketDisconnect:
            dashboard.disconnect_websocket(websocket)
            
    @app.get("/api/v1/monitoring/summary")
    async def get_monitoring_summary():
        """获取监控摘要"""
        return await dashboard.get_dashboard_summary()
        
    @app.get("/api/v1/monitoring/alerts")
    async def get_alerts():
        """获取告警列表"""
        return {
            "alerts": dashboard.alert_history[-50:],  # 最近50个告警
            "active": await dashboard.monitor.check_performance_alerts()
        }
        
    @app.post("/api/v1/monitoring/optimize")
    async def trigger_optimization():
        """手动触发优化"""
        optimizations = await dashboard.optimizer.auto_optimize()
        return {
            "status": "success",
            "optimizations": optimizations
        }
        
    @app.get("/api/v1/monitoring/traces/{trace_id}")
    async def get_trace_analysis(trace_id: str):
        """获取追踪分析"""
        analyzer = TraceAnalyzer(dashboard.tracer)
        analysis = analyzer.analyze_trace(trace_id)
        
        return {
            "trace_id": trace_id,
            "analysis": analysis,
            "report": analyzer.generate_trace_report(trace_id)
        }

# Grafana仪表板配置
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "Personalization Engine Monitoring",
        "panels": [
            {
                "title": "Request Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.99, personalization_recommendation_latency_seconds_bucket)",
                        "legendFormat": "P99"
                    },
                    {
                        "expr": "histogram_quantile(0.95, personalization_recommendation_latency_seconds_bucket)",
                        "legendFormat": "P95"
                    },
                    {
                        "expr": "histogram_quantile(0.50, personalization_recommendation_latency_seconds_bucket)",
                        "legendFormat": "P50"
                    }
                ]
            },
            {
                "title": "Throughput",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(personalization_throughput_rps_count[1m])",
                        "legendFormat": "RPS"
                    }
                ]
            },
            {
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(personalization_errors_total[5m])",
                        "legendFormat": "{{error_type}}"
                    }
                ]
            },
            {
                "title": "Cache Hit Rate",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "personalization_cache_hit_rate",
                        "legendFormat": "{{cache_level}}"
                    }
                ]
            },
            {
                "title": "Active Users",
                "type": "stat",
                "targets": [
                    {
                        "expr": "personalization_active_users"
                    }
                ]
            },
            {
                "title": "System Resources",
                "type": "graph",
                "targets": [
                    {
                        "expr": "process_resident_memory_bytes / 1024 / 1024",
                        "legendFormat": "Memory (MB)"
                    },
                    {
                        "expr": "rate(process_cpu_seconds_total[1m]) * 100",
                        "legendFormat": "CPU %"
                    }
                ]
            }
        ]
    }
}
from src.core.logging import get_logger
