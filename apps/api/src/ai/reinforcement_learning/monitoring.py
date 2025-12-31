"""
强化学习系统OpenTelemetry监控

集成全面的监控和可观测性，包括：
- OpenTelemetry链路追踪
- Prometheus指标收集
- 自定义业务指标
- 告警规则配置
- 性能监控仪表板
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass
from enum import Enum
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize
from opentelemetry.semconv.trace import SpanAttributes

from src.core.logging import get_logger
# OpenTelemetry imports

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    UP_DOWN_COUNTER = "up_down_counter"

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    duration_seconds: int
    severity: str  # "critical", "warning", "info"
    description: str

class RLSystemMonitoring:
    """强化学习系统监控"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # 初始化OpenTelemetry
        self._setup_telemetry()
        
        # 获取tracer和meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)
        
        # 创建指标
        self._create_metrics()
        
        # 告警规则
        self.alert_rules = []
        self._setup_alert_rules()
        
        # 监控状态
        self.monitoring_active = True
        self.last_health_check = utc_now()
    
    def _setup_telemetry(self):
        """设置OpenTelemetry"""
        
        # 设置Trace Provider
        trace_provider = TracerProvider()
        trace.set_tracer_provider(trace_provider)
        
        # 添加控制台导出器（开发环境）
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        trace_provider.add_span_processor(span_processor)
        
        # 设置Metrics Provider with Prometheus
        prometheus_reader = PrometheusMetricReader()
        metrics_provider = MeterProvider(metric_readers=[prometheus_reader])
        metrics.set_meter_provider(metrics_provider)
        
        self.logger.info("OpenTelemetry初始化完成")
    
    def _create_metrics(self):
        """创建业务指标"""
        
        # 推荐请求指标
        self.recommendation_request_counter = self.meter.create_counter(
            name="rl_recommendation_requests_total",
            description="推荐请求总数",
            unit="1"
        )
        
        self.recommendation_latency_histogram = self.meter.create_histogram(
            name="rl_recommendation_latency_seconds",
            description="推荐请求延迟",
            unit="s"
        )
        
        self.recommendation_errors_counter = self.meter.create_counter(
            name="rl_recommendation_errors_total", 
            description="推荐错误总数",
            unit="1"
        )
        
        # 算法性能指标
        self.algorithm_selection_counter = self.meter.create_counter(
            name="rl_algorithm_selections_total",
            description="算法选择次数",
            unit="1"
        )
        
        self.bandit_arm_selection_counter = self.meter.create_counter(
            name="rl_bandit_arm_selections_total",
            description="老虎机臂选择次数",
            unit="1"
        )
        
        # 反馈处理指标
        self.feedback_processed_counter = self.meter.create_counter(
            name="rl_feedback_processed_total",
            description="处理的反馈总数",
            unit="1"
        )
        
        self.feedback_processing_latency = self.meter.create_histogram(
            name="rl_feedback_processing_latency_seconds",
            description="反馈处理延迟",
            unit="s"
        )
        
        # 缓存指标
        self.cache_hits_counter = self.meter.create_counter(
            name="rl_cache_hits_total",
            description="缓存命中总数",
            unit="1"
        )
        
        self.cache_misses_counter = self.meter.create_counter(
            name="rl_cache_misses_total",
            description="缓存未命中总数",
            unit="1"
        )
        
        # 系统健康指标
        self.active_users_gauge = self.meter.create_gauge(
            name="rl_active_users",
            description="活跃用户数",
            unit="1"
        )
        
        self.model_accuracy_gauge = self.meter.create_gauge(
            name="rl_model_accuracy",
            description="模型准确率",
            unit="1"
        )
        
        # A/B测试指标
        self.ab_test_assignments_counter = self.meter.create_counter(
            name="rl_ab_test_assignments_total",
            description="A/B测试分配总数",
            unit="1"
        )
        
        self.ab_test_conversions_counter = self.meter.create_counter(
            name="rl_ab_test_conversions_total",
            description="A/B测试转化总数",
            unit="1"
        )
        
        self.logger.info("业务指标创建完成")
    
    def _setup_alert_rules(self):
        """设置告警规则"""
        
        self.alert_rules = [
            AlertRule(
                name="高响应延迟",
                metric_name="rl_recommendation_latency_seconds",
                condition="gt",
                threshold=0.1,  # 100ms
                duration_seconds=300,  # 5分钟
                severity="warning",
                description="推荐响应时间超过100ms持续5分钟"
            ),
            AlertRule(
                name="高错误率",
                metric_name="rl_recommendation_errors_rate",
                condition="gt", 
                threshold=0.05,  # 5%
                duration_seconds=180,  # 3分钟
                severity="critical",
                description="推荐错误率超过5%持续3分钟"
            ),
            AlertRule(
                name="缓存命中率低",
                metric_name="rl_cache_hit_rate",
                condition="lt",
                threshold=0.7,  # 70%
                duration_seconds=600,  # 10分钟
                severity="warning",
                description="缓存命中率低于70%持续10分钟"
            ),
            AlertRule(
                name="模型准确率下降",
                metric_name="rl_model_accuracy",
                condition="lt",
                threshold=0.6,  # 60%
                duration_seconds=1800,  # 30分钟
                severity="critical",
                description="模型准确率低于60%持续30分钟"
            ),
            AlertRule(
                name="活跃用户数异常",
                metric_name="rl_active_users",
                condition="lt",
                threshold=100,
                duration_seconds=900,  # 15分钟
                severity="warning",
                description="活跃用户数低于100持续15分钟"
            )
        ]
        
        self.logger.info(f"配置了{len(self.alert_rules)}个告警规则")
    
    def trace_recommendation_request(self, user_id: str, algorithm: str):
        """追踪推荐请求"""
        
        span = self.tracer.start_span("recommendation_request")
        span.set_attributes({
            SpanAttributes.USER_ID: user_id,
            "rl.algorithm": algorithm,
            "rl.component": "recommendation_engine"
        })
        return span
    
    def trace_algorithm_execution(self, algorithm: str, arm_id: int):
        """追踪算法执行"""
        
        span = self.tracer.start_span("algorithm_execution")
        span.set_attributes({
            "rl.algorithm": algorithm,
            "rl.arm_id": arm_id,
            "rl.component": "bandit_algorithm"
        })
        return span
    
    def trace_feedback_processing(self, user_id: str, item_id: str, feedback_type: str):
        """追踪反馈处理"""
        
        span = self.tracer.start_span("feedback_processing")
        span.set_attributes({
            SpanAttributes.USER_ID: user_id,
            "rl.item_id": item_id,
            "rl.feedback_type": feedback_type,
            "rl.component": "feedback_processor"
        })
        return span
    
    def record_recommendation_request(
        self, 
        user_id: str, 
        algorithm: str, 
        latency_seconds: float,
        success: bool = True,
        num_recommendations: int = 0
    ):
        """记录推荐请求指标"""
        
        attributes = {
            "algorithm": algorithm,
            "user_type": "new" if user_id.startswith("new_") else "existing"
        }
        
        # 记录请求计数
        self.recommendation_request_counter.add(1, attributes)
        
        # 记录延迟
        self.recommendation_latency_histogram.record(latency_seconds, attributes)
        
        # 记录错误
        if not success:
            self.recommendation_errors_counter.add(1, attributes)
        
        # 记录算法使用
        self.algorithm_selection_counter.add(1, {"algorithm": algorithm})
    
    def record_bandit_arm_selection(self, algorithm: str, arm_id: int, reward: Optional[float] = None):
        """记录老虎机臂选择"""
        
        attributes = {
            "algorithm": algorithm,
            "arm_id": str(arm_id)
        }
        
        if reward is not None:
            attributes["reward_range"] = self._categorize_reward(reward)
        
        self.bandit_arm_selection_counter.add(1, attributes)
    
    def record_feedback_processing(
        self, 
        feedback_type: str, 
        processing_latency_seconds: float,
        success: bool = True
    ):
        """记录反馈处理指标"""
        
        attributes = {"feedback_type": feedback_type}
        
        # 记录处理计数
        self.feedback_processed_counter.add(1, attributes)
        
        # 记录处理延迟
        self.feedback_processing_latency.record(processing_latency_seconds, attributes)
    
    def record_cache_metrics(self, hit: bool, cache_type: str = "recommendation"):
        """记录缓存指标"""
        
        attributes = {"cache_type": cache_type}
        
        if hit:
            self.cache_hits_counter.add(1, attributes)
        else:
            self.cache_misses_counter.add(1, attributes)
    
    def record_ab_test_assignment(self, experiment_id: str, variant: str, user_id: str):
        """记录A/B测试分配"""
        
        attributes = {
            "experiment_id": experiment_id,
            "variant": variant
        }
        
        self.ab_test_assignments_counter.add(1, attributes)
    
    def record_ab_test_conversion(self, experiment_id: str, variant: str, conversion_type: str):
        """记录A/B测试转化"""
        
        attributes = {
            "experiment_id": experiment_id,
            "variant": variant,
            "conversion_type": conversion_type
        }
        
        self.ab_test_conversions_counter.add(1, attributes)
    
    def update_system_health_metrics(self, active_users: int, model_accuracy: float):
        """更新系统健康指标"""
        
        self.active_users_gauge.set(active_users)
        self.model_accuracy_gauge.set(model_accuracy)
        
        self.last_health_check = utc_now()
    
    def _categorize_reward(self, reward: float) -> str:
        """分类奖励值"""
        
        if reward < 0.3:
            return "low"
        elif reward < 0.7:
            return "medium"
        else:
            return "high"
    
    async def generate_monitoring_dashboard_config(self) -> Dict[str, Any]:
        """生成Grafana监控仪表板配置"""
        
        dashboard_config = {
            "dashboard": {
                "title": "强化学习系统监控",
                "tags": ["reinforcement-learning", "ai", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "推荐请求QPS",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(rl_recommendation_requests_total[5m])",
                            "legendFormat": "QPS"
                        }],
                        "yAxes": [{"label": "requests/sec"}]
                    },
                    {
                        "title": "推荐延迟分布",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.50, rate(rl_recommendation_latency_seconds_bucket[5m]))",
                                "legendFormat": "P50"
                            },
                            {
                                "expr": "histogram_quantile(0.95, rate(rl_recommendation_latency_seconds_bucket[5m]))",
                                "legendFormat": "P95"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(rl_recommendation_latency_seconds_bucket[5m]))",
                                "legendFormat": "P99"
                            }
                        ],
                        "yAxes": [{"label": "seconds"}]
                    },
                    {
                        "title": "错误率",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(rl_recommendation_errors_total[5m]) / rate(rl_recommendation_requests_total[5m])",
                            "legendFormat": "Error Rate"
                        }],
                        "yAxes": [{"label": "percent", "max": 1}]
                    },
                    {
                        "title": "算法使用分布",
                        "type": "piechart",
                        "targets": [{
                            "expr": "increase(rl_algorithm_selections_total[1h])",
                            "legendFormat": "{{algorithm}}"
                        }]
                    },
                    {
                        "title": "缓存命中率",
                        "type": "stat",
                        "targets": [{
                            "expr": "rate(rl_cache_hits_total[5m]) / (rate(rl_cache_hits_total[5m]) + rate(rl_cache_misses_total[5m]))",
                            "legendFormat": "Cache Hit Rate"
                        }],
                        "fieldConfig": {
                            "min": 0,
                            "max": 1,
                            "unit": "percentunit"
                        }
                    },
                    {
                        "title": "活跃用户数",
                        "type": "stat",
                        "targets": [{
                            "expr": "rl_active_users",
                            "legendFormat": "Active Users"
                        }]
                    },
                    {
                        "title": "模型准确率",
                        "type": "gauge",
                        "targets": [{
                            "expr": "rl_model_accuracy",
                            "legendFormat": "Model Accuracy"
                        }],
                        "fieldConfig": {
                            "min": 0,
                            "max": 1,
                            "unit": "percentunit",
                            "thresholds": [
                                {"color": "red", "value": 0.5},
                                {"color": "yellow", "value": 0.7},
                                {"color": "green", "value": 0.8}
                            ]
                        }
                    },
                    {
                        "title": "A/B测试转化率",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(rl_ab_test_conversions_total[5m]) / rate(rl_ab_test_assignments_total[5m])",
                            "legendFormat": "{{experiment_id}} - {{variant}}"
                        }],
                        "yAxes": [{"label": "conversion rate"}]
                    }
                ]
            }
        }
        
        return dashboard_config
    
    async def generate_prometheus_config(self) -> str:
        """生成Prometheus配置"""
        
        config = """
# Prometheus配置文件
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: 'rl-system'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s
"""
        
        return config
    
    async def generate_alert_rules_config(self) -> str:
        """生成Prometheus告警规则配置"""
        
        rules = []
        
        for rule in self.alert_rules:
            prometheus_rule = {
                "alert": rule.name.replace(" ", "_"),
                "expr": self._convert_alert_rule_to_promql(rule),
                "for": f"{rule.duration_seconds}s",
                "labels": {
                    "severity": rule.severity,
                    "component": "rl-system"
                },
                "annotations": {
                    "summary": rule.description,
                    "description": f"{{{{ $labels.instance }}}} {rule.description}"
                }
            }
            rules.append(prometheus_rule)
        
        config = f"""
groups:
  - name: rl_system_alerts
    rules:
{self._format_prometheus_rules(rules)}
"""
        
        return config
    
    def _convert_alert_rule_to_promql(self, rule: AlertRule) -> str:
        """将告警规则转换为PromQL"""
        
        if rule.metric_name == "rl_recommendation_latency_seconds":
            return f"histogram_quantile(0.95, rate(rl_recommendation_latency_seconds_bucket[5m])) > {rule.threshold}"
        elif rule.metric_name == "rl_recommendation_errors_rate":
            return f"rate(rl_recommendation_errors_total[5m]) / rate(rl_recommendation_requests_total[5m]) > {rule.threshold}"
        elif rule.metric_name == "rl_cache_hit_rate":
            return f"rate(rl_cache_hits_total[5m]) / (rate(rl_cache_hits_total[5m]) + rate(rl_cache_misses_total[5m])) < {rule.threshold}"
        elif rule.metric_name == "rl_model_accuracy":
            return f"rl_model_accuracy < {rule.threshold}"
        elif rule.metric_name == "rl_active_users":
            return f"rl_active_users < {rule.threshold}"
        else:
            return f"{rule.metric_name} {self._condition_to_operator(rule.condition)} {rule.threshold}"
    
    def _condition_to_operator(self, condition: str) -> str:
        """转换条件到运算符"""
        mapping = {"gt": ">", "lt": "<", "eq": "==", "ne": "!="}
        return mapping.get(condition, ">")
    
    def _format_prometheus_rules(self, rules: List[Dict]) -> str:
        """格式化Prometheus规则"""
        formatted_rules = []
        
        for rule in rules:
            rule_str = f"""      - alert: {rule['alert']}
        expr: {rule['expr']}
        for: {rule['for']}
        labels:
          severity: {rule['labels']['severity']}
          component: {rule['labels']['component']}
        annotations:
          summary: "{rule['annotations']['summary']}"
          description: "{rule['annotations']['description']}\""""
            formatted_rules.append(rule_str)
        
        return "\n".join(formatted_rules)
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        
        health_status = {
            "status": "healthy",
            "timestamp": utc_now().isoformat(),
            "monitoring_active": self.monitoring_active,
            "last_health_check": self.last_health_check.isoformat(),
            "components": {
                "telemetry": "healthy",
                "metrics": "healthy", 
                "alerts": "healthy"
            },
            "metrics_summary": {
                "total_alert_rules": len(self.alert_rules),
                "monitoring_uptime_seconds": (utc_now() - self.last_health_check).total_seconds()
            }
        }
        
        return health_status
    
    def shutdown(self):
        """关闭监控系统"""
        
        self.monitoring_active = False
        self.logger.info("监控系统已关闭")

# 全局监控实例
monitoring = RLSystemMonitoring()

# 装饰器用于自动追踪和监控
def trace_recommendation(func):
    """推荐请求追踪装饰器"""
    
    async def wrapper(*args, **kwargs):
        user_id = kwargs.get('user_id', 'unknown')
        algorithm = kwargs.get('algorithm', 'unknown')
        
        with monitoring.trace_recommendation_request(user_id, algorithm) as span:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                
                monitoring.record_recommendation_request(
                    user_id, algorithm, latency, True,
                    len(result.recommendations) if hasattr(result, 'recommendations') else 0
                )
                
                span.set_attribute("rl.success", True)
                span.set_attribute("rl.latency_seconds", latency)
                
                return result
                
            except Exception as e:
                latency = time.time() - start_time
                monitoring.record_recommendation_request(
                    user_id, algorithm, latency, False
                )
                
                span.set_attribute("rl.success", False)
                span.set_attribute("rl.error", str(e))
                
                raise
    
    return wrapper

def trace_feedback(func):
    """反馈处理追踪装饰器"""
    
    async def wrapper(*args, **kwargs):
        user_id = kwargs.get('user_id', 'unknown')
        item_id = kwargs.get('item_id', 'unknown')
        feedback_type = kwargs.get('feedback_type', 'unknown')
        
        with monitoring.trace_feedback_processing(user_id, item_id, feedback_type) as span:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                
                monitoring.record_feedback_processing(feedback_type, latency, True)
                
                span.set_attribute("rl.success", True)
                span.set_attribute("rl.latency_seconds", latency)
                
                return result
                
            except Exception as e:
                latency = time.time() - start_time
                monitoring.record_feedback_processing(feedback_type, latency, False)
                
                span.set_attribute("rl.success", False)
                span.set_attribute("rl.error", str(e))
                
                raise
    
    return wrapper
