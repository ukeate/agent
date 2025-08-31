"""
告警规则引擎服务

实现灵活的告警规则配置和触发机制
"""
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import asyncio
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict

from ..core.database import get_db_session
from ..services.anomaly_detection_service import AnomalyDetectionService, Anomaly, AnomalyType
from ..services.realtime_metrics_service import RealtimeMetricsService


class AlertSeverity(str, Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """告警通道"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DASHBOARD = "dashboard"


class RuleOperator(str, Enum):
    """规则操作符"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    REGEX = "regex"


class RuleAggregation(str, Enum):
    """规则聚合方式"""
    ALL = "all"  # 所有条件都满足
    ANY = "any"  # 任一条件满足
    NONE = "none"  # 所有条件都不满足
    CUSTOM = "custom"  # 自定义表达式


@dataclass
class AlertCondition:
    """告警条件"""
    field: str  # 字段名
    operator: RuleOperator  # 操作符
    value: Any  # 阈值
    description: Optional[str] = None
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """评估条件"""
        field_value = self._get_nested_value(data, self.field)
        
        if field_value is None:
            return False
            
        try:
            if self.operator == RuleOperator.GREATER_THAN:
                return field_value > self.value
            elif self.operator == RuleOperator.LESS_THAN:
                return field_value < self.value
            elif self.operator == RuleOperator.EQUAL:
                return field_value == self.value
            elif self.operator == RuleOperator.NOT_EQUAL:
                return field_value != self.value
            elif self.operator == RuleOperator.GREATER_EQUAL:
                return field_value >= self.value
            elif self.operator == RuleOperator.LESS_EQUAL:
                return field_value <= self.value
            elif self.operator == RuleOperator.CONTAINS:
                return self.value in str(field_value)
            elif self.operator == RuleOperator.NOT_CONTAINS:
                return self.value not in str(field_value)
            elif self.operator == RuleOperator.IN:
                return field_value in self.value
            elif self.operator == RuleOperator.NOT_IN:
                return field_value not in self.value
            elif self.operator == RuleOperator.BETWEEN:
                return self.value[0] <= field_value <= self.value[1]
            elif self.operator == RuleOperator.REGEX:
                return bool(re.match(self.value, str(field_value)))
            else:
                return False
        except Exception:
            return False
            
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """获取嵌套字段值"""
        keys = field.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
                
        return value


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    experiment_id: Optional[str]  # 可选，用于特定实验
    metric_name: Optional[str]  # 可选，用于特定指标
    conditions: List[AlertCondition]
    aggregation: RuleAggregation
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 5  # 冷却时间，避免重复告警
    max_alerts_per_hour: int = 10  # 每小时最大告警数
    custom_expression: Optional[str] = None  # 自定义表达式
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """评估规则"""
        if not self.enabled:
            return False
            
        # 检查是否匹配实验和指标
        if self.experiment_id and data.get("experiment_id") != self.experiment_id:
            return False
        if self.metric_name and data.get("metric_name") != self.metric_name:
            return False
            
        # 评估条件
        results = [cond.evaluate(data) for cond in self.conditions]
        
        if self.aggregation == RuleAggregation.ALL:
            return all(results)
        elif self.aggregation == RuleAggregation.ANY:
            return any(results)
        elif self.aggregation == RuleAggregation.NONE:
            return not any(results)
        elif self.aggregation == RuleAggregation.CUSTOM and self.custom_expression:
            return self._evaluate_custom_expression(results)
        else:
            return False
            
    def _evaluate_custom_expression(self, results: List[bool]) -> bool:
        """评估自定义表达式"""
        try:
            # 将结果映射到变量
            context = {f"c{i}": r for i, r in enumerate(results)}
            # 安全评估表达式
            return eval(self.custom_expression, {"__builtins__": {}}, context)
        except Exception:
            return False


@dataclass
class Alert:
    """告警实例"""
    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    description: str
    data: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertRulesEngine:
    """告警规则引擎"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, datetime] = {}
        self.rate_limiter: Dict[str, List[datetime]] = defaultdict(list)
        self.anomaly_service = AnomalyDetectionService()
        self.metrics_service = RealtimeMetricsService()
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # 注册默认通知处理器
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """注册默认通知处理器"""
        self.notification_handlers[AlertChannel.EMAIL] = self._send_email
        self.notification_handlers[AlertChannel.SLACK] = self._send_slack
        self.notification_handlers[AlertChannel.WEBHOOK] = self._send_webhook
        self.notification_handlers[AlertChannel.DASHBOARD] = self._send_dashboard
        
    async def add_rule(self, rule: AlertRule) -> bool:
        """添加规则"""
        self.rules[rule.id] = rule
        return True
        
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新规则"""
        if rule_id not in self.rules:
            return False
            
        rule = self.rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        rule.updated_at = utc_now()
        
        return True
        
    async def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
        
    async def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """评估所有规则"""
        triggered_alerts = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
                
            # 检查冷却时间
            if self._is_in_cooldown(rule.id):
                continue
                
            # 检查速率限制
            if not self._check_rate_limit(rule.id, rule.max_alerts_per_hour):
                continue
                
            # 评估规则
            if rule.evaluate(data):
                alert = await self._create_alert(rule, data)
                triggered_alerts.append(alert)
                
                # 发送通知
                await self._send_notifications(alert, rule.channels)
                
                # 更新冷却时间
                self._update_cooldown(rule.id, rule.cooldown_minutes)
                
                # 更新速率限制
                self._update_rate_limit(rule.id)
                
        return triggered_alerts
        
    async def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """创建告警"""
        alert_id = f"alert_{rule.id}_{utc_now().timestamp()}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            title=self._generate_alert_title(rule, data),
            description=self._generate_alert_description(rule, data),
            data=data,
            triggered_at=utc_now()
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        return alert
        
    def _generate_alert_title(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """生成告警标题"""
        experiment_id = data.get("experiment_id", "unknown")
        metric_name = data.get("metric_name", "unknown")
        
        if rule.severity == AlertSeverity.CRITICAL:
            prefix = "🚨 严重"
        elif rule.severity == AlertSeverity.ERROR:
            prefix = "❌ 错误"
        elif rule.severity == AlertSeverity.WARNING:
            prefix = "⚠️ 警告"
        else:
            prefix = "ℹ️ 信息"
            
        return f"{prefix}: {rule.name} - 实验{experiment_id} 指标{metric_name}"
        
    def _generate_alert_description(self, rule: AlertRule, data: Dict[str, Any]) -> str:
        """生成告警描述"""
        desc = f"{rule.description}\n\n"
        desc += "触发条件:\n"
        
        for condition in rule.conditions:
            field_value = condition._get_nested_value(data, condition.field)
            desc += f"- {condition.field} {condition.operator.value} {condition.value} "
            desc += f"(当前值: {field_value})\n"
            
        return desc
        
    async def _send_notifications(self, alert: Alert, channels: List[AlertChannel]):
        """发送通知"""
        for channel in channels:
            handler = self.notification_handlers.get(channel)
            if handler:
                try:
                    await handler(alert)
                    alert.notifications_sent.append(channel.value)
                except Exception as e:
                    print(f"发送{channel.value}通知失败: {e}")
                    
    async def _send_email(self, alert: Alert):
        """发送邮件通知"""
        # 这里应该实现实际的邮件发送逻辑
        print(f"发送邮件: {alert.title}")
        
    async def _send_slack(self, alert: Alert):
        """发送Slack通知"""
        # 这里应该实现实际的Slack发送逻辑
        print(f"发送Slack: {alert.title}")
        
    async def _send_webhook(self, alert: Alert):
        """发送Webhook通知"""
        # 这里应该实现实际的Webhook调用逻辑
        print(f"发送Webhook: {alert.title}")
        
    async def _send_dashboard(self, alert: Alert):
        """发送到仪表板"""
        # 这里应该实现实际的仪表板更新逻辑
        print(f"更新仪表板: {alert.title}")
        
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """检查是否在冷却期"""
        if rule_id not in self.cooldown_tracker:
            return False
            
        last_alert = self.cooldown_tracker[rule_id]
        return utc_now() < last_alert
        
    def _update_cooldown(self, rule_id: str, cooldown_minutes: int):
        """更新冷却时间"""
        self.cooldown_tracker[rule_id] = utc_now() + timedelta(minutes=cooldown_minutes)
        
    def _check_rate_limit(self, rule_id: str, max_per_hour: int) -> bool:
        """检查速率限制"""
        now = utc_now()
        hour_ago = now - timedelta(hours=1)
        
        # 清理过期记录
        self.rate_limiter[rule_id] = [
            t for t in self.rate_limiter[rule_id]
            if t > hour_ago
        ]
        
        return len(self.rate_limiter[rule_id]) < max_per_hour
        
    def _update_rate_limit(self, rule_id: str):
        """更新速率限制"""
        self.rate_limiter[rule_id].append(utc_now())
        
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """确认告警"""
        if alert_id not in self.alerts:
            return False
            
        alert = self.alerts[alert_id]
        alert.acknowledged_at = utc_now()
        alert.acknowledged_by = user_id
        
        return True
        
    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id not in self.alerts:
            return False
            
        alert = self.alerts[alert_id]
        alert.resolved_at = utc_now()
        
        return True
        
    async def get_active_alerts(
        self,
        experiment_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """获取活跃告警"""
        active_alerts = [
            alert for alert in self.alerts.values()
            if alert.resolved_at is None
        ]
        
        if experiment_id:
            active_alerts = [
                a for a in active_alerts
                if a.data.get("experiment_id") == experiment_id
            ]
            
        if severity:
            active_alerts = [
                a for a in active_alerts
                if a.severity == severity
            ]
            
        return active_alerts
        
    async def get_alert_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取告警统计"""
        if not start_time:
            start_time = utc_now() - timedelta(days=7)
        if not end_time:
            end_time = utc_now()
            
        # 过滤时间范围内的告警
        filtered_alerts = [
            a for a in self.alert_history
            if start_time <= a.triggered_at <= end_time
        ]
        
        # 统计
        stats = {
            "total_alerts": len(filtered_alerts),
            "by_severity": {},
            "by_rule": {},
            "by_experiment": {},
            "mean_time_to_acknowledge": None,
            "mean_time_to_resolve": None,
            "top_rules": []
        }
        
        # 按严重级别统计
        for alert in filtered_alerts:
            if alert.severity not in stats["by_severity"]:
                stats["by_severity"][alert.severity] = 0
            stats["by_severity"][alert.severity] += 1
            
            # 按规则统计
            if alert.rule_name not in stats["by_rule"]:
                stats["by_rule"][alert.rule_name] = 0
            stats["by_rule"][alert.rule_name] += 1
            
            # 按实验统计
            exp_id = alert.data.get("experiment_id", "unknown")
            if exp_id not in stats["by_experiment"]:
                stats["by_experiment"][exp_id] = 0
            stats["by_experiment"][exp_id] += 1
            
        # 计算平均确认时间
        ack_times = [
            (a.acknowledged_at - a.triggered_at).total_seconds()
            for a in filtered_alerts
            if a.acknowledged_at
        ]
        if ack_times:
            stats["mean_time_to_acknowledge"] = sum(ack_times) / len(ack_times)
            
        # 计算平均解决时间
        resolve_times = [
            (a.resolved_at - a.triggered_at).total_seconds()
            for a in filtered_alerts
            if a.resolved_at
        ]
        if resolve_times:
            stats["mean_time_to_resolve"] = sum(resolve_times) / len(resolve_times)
            
        # 获取触发最多的规则
        stats["top_rules"] = sorted(
            stats["by_rule"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return stats


class AlertRuleBuilder:
    """告警规则构建器"""
    
    def __init__(self):
        self.rule = None
        
    def create(self, rule_id: str, name: str) -> 'AlertRuleBuilder':
        """创建新规则"""
        self.rule = AlertRule(
            id=rule_id,
            name=name,
            description="",
            experiment_id=None,
            metric_name=None,
            conditions=[],
            aggregation=RuleAggregation.ALL,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.DASHBOARD]
        )
        return self
        
    def description(self, desc: str) -> 'AlertRuleBuilder':
        """设置描述"""
        self.rule.description = desc
        return self
        
    def for_experiment(self, experiment_id: str) -> 'AlertRuleBuilder':
        """设置实验ID"""
        self.rule.experiment_id = experiment_id
        return self
        
    def for_metric(self, metric_name: str) -> 'AlertRuleBuilder':
        """设置指标名称"""
        self.rule.metric_name = metric_name
        return self
        
    def add_condition(
        self,
        field: str,
        operator: RuleOperator,
        value: Any,
        description: Optional[str] = None
    ) -> 'AlertRuleBuilder':
        """添加条件"""
        condition = AlertCondition(field, operator, value, description)
        self.rule.conditions.append(condition)
        return self
        
    def aggregation(self, agg: RuleAggregation) -> 'AlertRuleBuilder':
        """设置聚合方式"""
        self.rule.aggregation = agg
        return self
        
    def severity(self, sev: AlertSeverity) -> 'AlertRuleBuilder':
        """设置严重级别"""
        self.rule.severity = sev
        return self
        
    def channels(self, *channels: AlertChannel) -> 'AlertRuleBuilder':
        """设置通知渠道"""
        self.rule.channels = list(channels)
        return self
        
    def cooldown(self, minutes: int) -> 'AlertRuleBuilder':
        """设置冷却时间"""
        self.rule.cooldown_minutes = minutes
        return self
        
    def rate_limit(self, max_per_hour: int) -> 'AlertRuleBuilder':
        """设置速率限制"""
        self.rule.max_alerts_per_hour = max_per_hour
        return self
        
    def build(self) -> AlertRule:
        """构建规则"""
        return self.rule


# 预定义规则模板
class AlertRuleTemplates:
    """告警规则模板"""
    
    @staticmethod
    def metric_spike_rule(metric_name: str, threshold: float) -> AlertRule:
        """指标突增规则"""
        return (AlertRuleBuilder()
                .create(f"spike_{metric_name}", f"{metric_name}突增告警")
                .description(f"当{metric_name}增长超过{threshold*100}%时触发")
                .for_metric(metric_name)
                .add_condition("relative_change", RuleOperator.GREATER_THAN, threshold)
                .severity(AlertSeverity.WARNING)
                .channels(AlertChannel.EMAIL, AlertChannel.DASHBOARD)
                .cooldown(10)
                .build())
                
    @staticmethod
    def metric_drop_rule(metric_name: str, threshold: float) -> AlertRule:
        """指标突降规则"""
        return (AlertRuleBuilder()
                .create(f"drop_{metric_name}", f"{metric_name}突降告警")
                .description(f"当{metric_name}下降超过{threshold*100}%时触发")
                .for_metric(metric_name)
                .add_condition("relative_change", RuleOperator.LESS_THAN, -threshold)
                .severity(AlertSeverity.ERROR)
                .channels(AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD)
                .cooldown(10)
                .build())
                
    @staticmethod
    def srm_rule(experiment_id: str) -> AlertRule:
        """SRM告警规则"""
        return (AlertRuleBuilder()
                .create(f"srm_{experiment_id}", f"实验{experiment_id} SRM告警")
                .description("检测到样本比例不匹配")
                .for_experiment(experiment_id)
                .add_condition("anomaly_type", RuleOperator.EQUAL, AnomalyType.SAMPLE_RATIO_MISMATCH)
                .severity(AlertSeverity.CRITICAL)
                .channels(AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY)
                .cooldown(60)
                .build())
                
    @staticmethod
    def data_quality_rule() -> AlertRule:
        """数据质量规则"""
        return (AlertRuleBuilder()
                .create("data_quality", "数据质量告警")
                .description("数据质量问题检测")
                .add_condition("missing_rate", RuleOperator.GREATER_THAN, 0.1)
                .add_condition("duplicate_rate", RuleOperator.GREATER_THAN, 0.05)
                .aggregation(RuleAggregation.ANY)
                .severity(AlertSeverity.WARNING)
                .channels(AlertChannel.DASHBOARD)
                .cooldown(30)
                .build())