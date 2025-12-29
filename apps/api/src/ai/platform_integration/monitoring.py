"""监控系统实现"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import redis
from .models import MonitoringConfig
from src.core.utils.timezone_utils import utc_now

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None

class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Redis客户端
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=True
        )
        
        # Prometheus指标
        self.metrics = {}
        self.alert_rules = []
        self.registry = None
        
        # 初始化指标
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._initialize_metrics()
        else:
            self.logger.warning("Prometheus 客户端不可用，已禁用指标导出")
    
    def _initialize_metrics(self):
        """初始化Prometheus指标"""
        
        if not PROMETHEUS_AVAILABLE:
            return
        
        # 导入prometheus_client模块
        try:
            from prometheus_client import Counter, Histogram, Gauge
        except ImportError:
            self.logger.warning("Failed to import prometheus_client metrics")
            return
        
        # 请求计数器
        self.metrics['request_counter'] = Counter(
            'platform_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        # 请求延迟直方图
        self.metrics['request_duration'] = Histogram(
            'platform_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # 内存使用量表
        self.metrics['memory_usage'] = Gauge(
            'platform_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        # CPU使用率
        self.metrics['cpu_usage'] = Gauge(
            'platform_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
        
        # 活跃训练任务数
        self.metrics['active_training_jobs'] = Gauge(
            'platform_active_training_jobs',
            'Number of active training jobs',
            registry=self.registry
        )
        
        # 模型评估分数
        self.metrics['model_evaluation_score'] = Gauge(
            'platform_model_evaluation_score',
            'Model evaluation score',
            ['model_id', 'metric_type'],
            registry=self.registry
        )
        
        # 工作流成功率
        self.metrics['workflow_success_rate'] = Gauge(
            'platform_workflow_success_rate',
            'Workflow success rate',
            ['workflow_type'],
            registry=self.registry
        )
        
        # 组件健康状态
        self.metrics['component_health'] = Gauge(
            'platform_component_health',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component_id', 'component_type'],
            registry=self.registry
        )
    
    async def setup_monitoring(self) -> Dict[str, Any]:
        """设置监控系统"""
        raise RuntimeError("Monitoring setup requires Prometheus/Grafana integration")
    
    async def _setup_prometheus_metrics(self) -> Dict[str, Any]:
        """设置Prometheus指标"""
        
        if not PROMETHEUS_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Prometheus client not installed"
            }
        
        return {
            "status": "configured",
            "metrics": list(self.metrics.keys()),
            "endpoint": "/metrics",
            "scrape_interval": "15s",
            "evaluation_interval": "15s"
        }
    
    async def _setup_grafana_dashboards(self) -> Dict[str, Any]:
        """设置Grafana仪表板"""
        
        dashboards = {
            "platform_overview": {
                "title": "Platform Overview",
                "uid": "platform-overview",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": ["rate(platform_requests_total[5m])"],
                        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
                    },
                    {
                        "id": 2,
                        "title": "Response Time (p95)",
                        "type": "graph",
                        "targets": ["histogram_quantile(0.95, platform_request_duration_seconds)"],
                        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
                    },
                    {
                        "id": 3,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": ["platform_memory_usage_bytes"],
                        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
                    },
                    {
                        "id": 4,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": ["platform_cpu_usage_percent"],
                        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
                    }
                ]
            },
            "training_monitoring": {
                "title": "Training Monitoring",
                "uid": "training-monitoring",
                "panels": [
                    {
                        "id": 1,
                        "title": "Active Training Jobs",
                        "type": "singlestat",
                        "targets": ["platform_active_training_jobs"],
                        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4}
                    },
                    {
                        "id": 2,
                        "title": "Workflow Success Rate",
                        "type": "gauge",
                        "targets": ["platform_workflow_success_rate"],
                        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4}
                    },
                    {
                        "id": 3,
                        "title": "Model Evaluation Scores",
                        "type": "graph",
                        "targets": ["platform_model_evaluation_score"],
                        "gridPos": {"x": 0, "y": 4, "w": 24, "h": 8}
                    }
                ]
            },
            "component_health": {
                "title": "Component Health",
                "uid": "component-health",
                "panels": [
                    {
                        "id": 1,
                        "title": "Component Status",
                        "type": "heatmap",
                        "targets": ["platform_component_health"],
                        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 12}
                    }
                ]
            }
        }
        
        return {
            "status": "configured",
            "dashboards": list(dashboards.keys()),
            "total_panels": sum(len(d["panels"]) for d in dashboards.values()),
            "config": dashboards
        }
    
    async def _setup_alerting_rules(self) -> Dict[str, Any]:
        """设置告警规则"""
        
        alert_rules = [
            {
                "name": "HighErrorRate",
                "expr": 'rate(platform_requests_total{status=~"5.."}[5m]) > 0.1',
                "for": "5m",
                "labels": {"severity": "warning", "team": "platform"},
                "annotations": {
                    "summary": "High error rate detected",
                    "description": "Error rate is {{ $value }} requests per second"
                }
            },
            {
                "name": "HighResponseTime",
                "expr": "histogram_quantile(0.95, platform_request_duration_seconds) > 2",
                "for": "5m",
                "labels": {"severity": "warning", "team": "platform"},
                "annotations": {
                    "summary": "High response time detected",
                    "description": "95th percentile response time is {{ $value }} seconds"
                }
            },
            {
                "name": "HighMemoryUsage",
                "expr": "platform_memory_usage_bytes / 1024 / 1024 / 1024 > 8",
                "for": "10m",
                "labels": {"severity": "critical", "team": "platform"},
                "annotations": {
                    "summary": "High memory usage detected",
                    "description": "Memory usage is {{ $value }} GB"
                }
            },
            {
                "name": "HighCPUUsage",
                "expr": "platform_cpu_usage_percent > 80",
                "for": "10m",
                "labels": {"severity": "warning", "team": "platform"},
                "annotations": {
                    "summary": "High CPU usage detected",
                    "description": "CPU usage is {{ $value }}%"
                }
            },
            {
                "name": "ComponentUnhealthy",
                "expr": "platform_component_health == 0",
                "for": "3m",
                "labels": {"severity": "critical", "team": "platform"},
                "annotations": {
                    "summary": "Component unhealthy",
                    "description": "Component {{ $labels.component_id }} is unhealthy"
                }
            },
            {
                "name": "LowWorkflowSuccessRate",
                "expr": "platform_workflow_success_rate < 0.8",
                "for": "15m",
                "labels": {"severity": "warning", "team": "platform"},
                "annotations": {
                    "summary": "Low workflow success rate",
                    "description": "Workflow {{ $labels.workflow_type }} success rate is {{ $value }}"
                }
            }
        ]
        
        self.alert_rules = alert_rules
        
        return {
            "status": "configured",
            "total_rules": len(alert_rules),
            "alert_rules": alert_rules
        }
    
    async def _setup_health_checks(self) -> Dict[str, Any]:
        """设置健康检查"""
        
        health_check_config = {
            "endpoints": [
                {
                    "path": "/health",
                    "interval": 30,
                    "timeout": 5,
                    "description": "Basic health check"
                },
                {
                    "path": "/ready",
                    "interval": 30,
                    "timeout": 5,
                    "description": "Readiness check"
                },
                {
                    "path": "/metrics",
                    "interval": 60,
                    "timeout": 10,
                    "description": "Prometheus metrics"
                }
            ],
            "database": {
                "check_query": "SELECT 1",
                "interval": 60,
                "timeout": 5
            },
            "redis": {
                "check_command": "ping",
                "interval": 60,
                "timeout": 5
            },
            "external_services": [
                {
                    "name": "fine_tuning_service",
                    "url": "http://fine-tuning:8001/health",
                    "interval": 60,
                    "timeout": 5
                },
                {
                    "name": "evaluation_service",
                    "url": "http://evaluation:8002/health",
                    "interval": 60,
                    "timeout": 5
                }
            ]
        }
        
        return {
            "status": "configured",
            "config": health_check_config
        }
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录请求指标"""
        if PROMETHEUS_AVAILABLE and 'request_counter' in self.metrics:
            self.metrics['request_counter'].labels(
                method=method,
                endpoint=endpoint,
                status=str(status)
            ).inc()
            
            self.metrics['request_duration'].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def update_memory_usage(self, component: str, usage_bytes: float):
        """更新内存使用量"""
        if PROMETHEUS_AVAILABLE and 'memory_usage' in self.metrics:
            self.metrics['memory_usage'].labels(component=component).set(usage_bytes)
    
    def update_cpu_usage(self, component: str, usage_percent: float):
        """更新CPU使用率"""
        if PROMETHEUS_AVAILABLE and 'cpu_usage' in self.metrics:
            self.metrics['cpu_usage'].labels(component=component).set(usage_percent)
    
    def update_active_jobs(self, count: int):
        """更新活跃任务数"""
        if PROMETHEUS_AVAILABLE and 'active_training_jobs' in self.metrics:
            self.metrics['active_training_jobs'].set(count)
    
    def update_evaluation_score(self, model_id: str, metric_type: str, score: float):
        """更新模型评估分数"""
        if PROMETHEUS_AVAILABLE and 'model_evaluation_score' in self.metrics:
            self.metrics['model_evaluation_score'].labels(
                model_id=model_id,
                metric_type=metric_type
            ).set(score)
    
    def update_workflow_success_rate(self, workflow_type: str, success_rate: float):
        """更新工作流成功率"""
        if PROMETHEUS_AVAILABLE and 'workflow_success_rate' in self.metrics:
            self.metrics['workflow_success_rate'].labels(
                workflow_type=workflow_type
            ).set(success_rate)
    
    def update_component_health(self, component_id: str, component_type: str, is_healthy: bool):
        """更新组件健康状态"""
        if PROMETHEUS_AVAILABLE and 'component_health' in self.metrics:
            self.metrics['component_health'].labels(
                component_id=component_id,
                component_type=component_type
            ).set(1 if is_healthy else 0)
    
    def get_metrics(self) -> bytes:
        """获取Prometheus格式的指标"""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry)
        return b""
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        
        # 获取最近的指标数据
        recent_metrics = await self._get_recent_metrics()
        
        # 计算统计信息
        statistics = await self._calculate_statistics(recent_metrics)
        
        # 检查告警状态
        alert_status = await self._check_alert_status()
        
        # 生成建议
        recommendations = await self._generate_recommendations(statistics, alert_status)
        
        return {
            "report_generated_at": utc_now().isoformat(),
            "metrics_summary": statistics,
            "alert_status": alert_status,
            "recommendations": recommendations,
            "health_score": self._calculate_health_score(statistics, alert_status)
        }
    
    async def _get_recent_metrics(self) -> Dict[str, Any]:
        """获取最近的指标数据"""
        from src.core.monitoring import monitoring_service
        return await monitoring_service.collect_all_metrics()
    
    async def _calculate_statistics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        system = metrics.get("system", {})
        performance = metrics.get("performance", {})

        request_rate = self._calculate_request_rate(performance)
        error_rate = self._calculate_error_rate(performance)
        avg_response_time = self._calculate_avg_response_time(performance)
        cpu_usage_avg = self._calculate_avg_cpu(system)
        memory_usage_avg = self._calculate_avg_memory(system)
        workflow_success_rate = self._calculate_workflow_success()

        return {
            "request_rate": request_rate,
            "requests_per_minute": performance.get("requests_per_minute", 0),
            "total_requests": performance.get("total_requests", 0),
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "cpu_usage_avg": cpu_usage_avg,
            "memory_usage_avg": memory_usage_avg,
            "workflow_success_rate": workflow_success_rate,
        }
    
    def _calculate_request_rate(self, performance: Dict[str, Any]) -> float:
        """计算请求速率"""
        rpm = performance.get("requests_per_minute", 0)
        try:
            return float(rpm) / 60
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_error_rate(self, performance: Dict[str, Any]) -> float:
        """计算错误率"""
        try:
            return float(performance.get("error_rate", 0))
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_avg_response_time(self, performance: Dict[str, Any]) -> float:
        """计算平均响应时间"""
        ms = performance.get("average_response_time_ms", 0)
        try:
            return float(ms) / 1000
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_avg_memory(self, system: Dict[str, Any]) -> float:
        """计算平均内存使用"""
        try:
            return float(system.get("memory_percent", 0))
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_avg_cpu(self, system: Dict[str, Any]) -> float:
        """计算平均CPU使用"""
        try:
            return float(system.get("cpu_percent", 0))
        except (TypeError, ValueError):
            return 0.0
    
    def _calculate_workflow_success(self) -> float:
        """计算工作流成功率"""
        total = 0
        success = 0
        for key in self.redis_client.scan_iter("workflow:*", count=1000):
            raw = self.redis_client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            status = str(data.get("status", "")).lower()
            if not status:
                continue
            if status in {"completed", "failed", "error"}:
                total += 1
                if status == "completed":
                    success += 1
        if total == 0:
            return 1.0
        return success / total
    
    async def _check_alert_status(self) -> Dict[str, Any]:
        """检查告警状态"""
        raw = self.redis_client.get("platform:monitoring:alerts")
        if not raw:
            return {"active_alerts": [], "total_alerts": 0, "status": "healthy"}
        try:
            alerts = json.loads(raw)
        except json.JSONDecodeError:
            alerts = []
        active_alerts = [a for a in alerts if isinstance(a, dict) and a.get("status") == "active"]
        return {
            "active_alerts": active_alerts,
            "total_alerts": len(active_alerts),
            "status": "healthy" if len(active_alerts) == 0 else "warning"
        }
    
    async def _generate_recommendations(
        self, 
        statistics: Dict[str, Any], 
        alert_status: Dict[str, Any]
    ) -> List[str]:
        """生成建议"""
        
        recommendations = []
        
        if statistics.get("error_rate", 0) > 0.05:
            recommendations.append("错误率较高，建议优先排查最近的异常请求与依赖服务状态")

        if statistics.get("cpu_usage_avg", 0) > 80:
            recommendations.append("CPU 使用率偏高，建议检查热点接口或扩容计算资源")

        if statistics.get("memory_usage_avg", 0) > 85:
            recommendations.append("内存使用率偏高，建议检查缓存策略与内存泄漏风险")

        if statistics.get("workflow_success_rate", 1) < 0.9:
            recommendations.append("工作流成功率偏低，建议检查组件健康检查与工作流执行日志")

        if not recommendations:
            recommendations.append("系统运行在正常范围内")
        
        return recommendations
    
    def _calculate_health_score(
        self, 
        statistics: Dict[str, Any], 
        alert_status: Dict[str, Any]
    ) -> float:
        """计算健康评分"""
        
        score = 100.0
        
        # 错误率扣分
        error_rate = statistics.get("error_rate", 0)
        if error_rate > 0.1:
            score -= 30
        elif error_rate > 0.05:
            score -= 15
        elif error_rate > 0.02:
            score -= 5
        
        # 响应时间扣分
        response_time = statistics.get("avg_response_time", 0)
        if response_time > 3:
            score -= 20
        elif response_time > 2:
            score -= 10
        elif response_time > 1.5:
            score -= 5
        
        # 活跃告警扣分
        active_alerts = len(alert_status.get("active_alerts", []))
        score -= active_alerts * 10
        
        return max(0, min(100, score))
    
    async def export_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """导出指标历史"""
        
        end_time = utc_now()
        start_time = end_time - timedelta(hours=hours)
        
        # 从Redis获取历史数据
        history = []
        
        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": hours,
            "data_points": len(history),
            "metrics": history
        }
from src.core.logging import get_logger
