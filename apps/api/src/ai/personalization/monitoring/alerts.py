"""
个性化引擎告警系统
实现生产就绪的监控告警功能，包括性能指标、错误率、资源使用等
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect
from abc import ABC, abstractmethod
import aiohttp
from pydantic import BaseModel, Field
import redis.asyncio as redis_async

from src.core.logging import get_logger
logger = get_logger(__name__)

class AlertSeverity(Enum):
    """告警严重级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # ">", "<", ">=", "<=", "=="
    severity: AlertSeverity
    duration: int = 300  # 持续时间（秒）
    evaluation_interval: int = 60  # 评估间隔（秒）
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self, current_value: float, previous_values: List[float]) -> bool:
        """评估告警条件"""
        if self.comparison == ">":
            return current_value > self.threshold
        elif self.comparison == "<":
            return current_value < self.threshold
        elif self.comparison == ">=":
            return current_value >= self.threshold
        elif self.comparison == "<=":
            return current_value <= self.threshold
        elif self.comparison == "==":
            return current_value == self.threshold
        return False

class Alert(BaseModel):
    """告警实例"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    current_value: float
    threshold: float
    metric_name: str
    start_time: datetime
    last_update: datetime
    resolved_time: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class AlertChannel(ABC):
    """告警通道抽象类"""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        ...
    
    @abstractmethod
    async def send_resolution(self, alert: Alert) -> bool:
        """发送解决通知"""
        ...

class WebhookAlertChannel(AlertChannel):
    """Webhook告警通道"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警到Webhook"""
        try:
            payload = {
                "type": "alert",
                "alert": alert.model_dump(),
                "timestamp": utc_now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
            return False
    
    async def send_resolution(self, alert: Alert) -> bool:
        """发送解决通知到Webhook"""
        try:
            payload = {
                "type": "resolution",
                "alert": alert.model_dump(),
                "timestamp": utc_now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Webhook解决通知失败: {e}")
            return False

class SlackAlertChannel(AlertChannel):
    """Slack告警通道"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警到Slack"""
        try:
            # 根据严重级别设置颜色
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # 绿色
                AlertSeverity.MEDIUM: "#ff9900",   # 橙色
                AlertSeverity.HIGH: "#ff6600",     # 红橙色
                AlertSeverity.CRITICAL: "#ff0000"  # 红色
            }
            
            payload = {
                "channel": self.channel,
                "username": "PersonalizationAlert",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"🚨 {alert.severity.value.upper()} - {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "指标",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "当前值",
                            "value": f"{alert.current_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "阈值",
                            "value": f"{alert.threshold:.2f}",
                            "short": True
                        },
                        {
                            "title": "开始时间",
                            "value": alert.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True
                        }
                    ],
                    "footer": "个性化引擎监控",
                    "ts": int(alert.start_time.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")
            return False
    
    async def send_resolution(self, alert: Alert) -> bool:
        """发送解决通知到Slack"""
        try:
            payload = {
                "channel": self.channel,
                "username": "PersonalizationAlert",
                "icon_emoji": ":white_check_mark:",
                "attachments": [{
                    "color": "#36a64f",
                    "title": f"✅ RESOLVED - {alert.title}",
                    "text": f"告警已解决: {alert.description}",
                    "fields": [
                        {
                            "title": "解决时间",
                            "value": alert.resolved_time.strftime("%Y-%m-%d %H:%M:%S") if alert.resolved_time else "N/A",
                            "short": True
                        },
                        {
                            "title": "持续时间",
                            "value": str(alert.resolved_time - alert.start_time) if alert.resolved_time else "N/A",
                            "short": True
                        }
                    ],
                    "footer": "个性化引擎监控",
                    "ts": int(alert.resolved_time.timestamp()) if alert.resolved_time else int(utc_now().timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
                    
        except Exception as e:
            logger.error(f"发送Slack解决通知失败: {e}")
            return False

class EmailAlertChannel(AlertChannel):
    """邮件告警通道"""
    
    def __init__(self, smtp_config: Dict[str, Any], recipients: List[str]):
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警邮件"""
        try:
            import smtplib
            from email.mime.text import MimeText
            from email.mime.multipart import MimeMultipart
            
            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from']
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] 个性化引擎告警: {alert.title}"
            
            # 邮件内容
            body = f"""
            告警详情:
            
            告警名称: {alert.title}
            严重级别: {alert.severity.value.upper()}
            描述: {alert.description}
            
            指标信息:
            - 指标名称: {alert.metric_name}
            - 当前值: {alert.current_value:.2f}
            - 阈值: {alert.threshold:.2f}
            
            时间信息:
            - 开始时间: {alert.start_time.strftime('%Y-%m-%d %H:%M:%S')}
            - 最后更新: {alert.last_update.strftime('%Y-%m-%d %H:%M:%S')}
            
            标签: {json.dumps(alert.tags, ensure_ascii=False, indent=2)}
            
            ---
            个性化引擎监控系统
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.smtp_config['from'], self.recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False
    
    async def send_resolution(self, alert: Alert) -> bool:
        """发送解决通知邮件"""
        try:
            import smtplib
            from email.mime.text import MimeText
            from email.mime.multipart import MimeMultipart
            
            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from']
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[RESOLVED] 个性化引擎告警已解决: {alert.title}"
            
            # 邮件内容
            duration = alert.resolved_time - alert.start_time if alert.resolved_time else timedelta(0)
            
            body = f"""
            告警已解决:
            
            告警名称: {alert.title}
            描述: {alert.description}
            
            时间信息:
            - 开始时间: {alert.start_time.strftime('%Y-%m-%d %H:%M:%S')}
            - 解决时间: {alert.resolved_time.strftime('%Y-%m-%d %H:%M:%S') if alert.resolved_time else 'N/A'}
            - 持续时间: {str(duration)}
            
            ---
            个性化引擎监控系统
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(self.smtp_config['username'], self.smtp_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.smtp_config['from'], self.recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"发送解决通知邮件失败: {e}")
            return False

class AlertManager:
    """告警管理器"""
    
    def __init__(self, redis_client: redis_async.Redis, alert_channels: List[AlertChannel]):
        self.redis = redis_client
        self.alert_channels = alert_channels
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.metric_history: Dict[str, List[tuple]] = {}  # (timestamp, value)
        self._running = False
        self._tasks = []
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")
    
    async def update_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """更新指标值"""
        timestamp = timestamp or utc_now()
        
        # 保存到历史记录
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # 保持历史记录在合理范围内（最近1小时）
        cutoff_time = timestamp - timedelta(hours=1)
        self.metric_history[metric_name] = [
            (t, v) for t, v in self.metric_history[metric_name] 
            if t > cutoff_time
        ]
        
        # 保存到Redis
        await self.redis.zadd(
            f"metric:{metric_name}",
            {str(value): timestamp.timestamp()}
        )
        
        # 设置过期时间
        await self.redis.expire(f"metric:{metric_name}", 3600)  # 1小时
        
        # 触发规则评估
        await self._evaluate_rules_for_metric(metric_name, value)
    
    async def _evaluate_rules_for_metric(self, metric_name: str, current_value: float):
        """为特定指标评估告警规则"""
        for rule_name, rule in self.alert_rules.items():
            if rule.metric_name == metric_name and rule.enabled:
                await self._evaluate_single_rule(rule, current_value)
    
    async def _evaluate_single_rule(self, rule: AlertRule, current_value: float):
        """评估单个告警规则"""
        # 获取历史值
        history = self.metric_history.get(rule.metric_name, [])
        previous_values = [v for t, v in history]
        
        # 评估告警条件
        is_triggered = rule.evaluate(current_value, previous_values)
        alert_id = f"{rule.name}:{rule.metric_name}"
        
        if is_triggered:
            if alert_id not in self.active_alerts:
                # 创建新告警
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    title=f"{rule.name} - {rule.metric_name}告警",
                    description=rule.description,
                    current_value=current_value,
                    threshold=rule.threshold,
                    metric_name=rule.metric_name,
                    start_time=utc_now(),
                    last_update=utc_now(),
                    tags=rule.tags
                )
                
                self.active_alerts[alert_id] = alert
                await self._send_alert(alert)
                
            else:
                # 更新现有告警
                alert = self.active_alerts[alert_id]
                alert.current_value = current_value
                alert.last_update = utc_now()
                
        else:
            if alert_id in self.active_alerts:
                # 解决告警
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_time = utc_now()
                alert.last_update = utc_now()
                
                await self._send_resolution(alert)
                del self.active_alerts[alert_id]
    
    async def _send_alert(self, alert: Alert):
        """发送告警通知"""
        logger.warning(f"触发告警: {alert.title} - {alert.description}")
        
        # 保存到Redis
        await self.redis.hset(
            "active_alerts",
            alert.id,
            alert.model_dump_json()
        )
        
        # 发送到所有通道
        for channel in self.alert_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                logger.error(f"发送告警到通道失败: {e}")
    
    async def _send_resolution(self, alert: Alert):
        """发送解决通知"""
        logger.info(f"解决告警: {alert.title}")
        
        # 从Redis移除
        await self.redis.hdel("active_alerts", alert.id)
        
        # 保存到历史记录
        await self.redis.hset(
            "resolved_alerts",
            alert.id,
            alert.model_dump_json()
        )
        
        # 发送到所有通道
        for channel in self.alert_channels:
            try:
                await channel.send_resolution(alert)
            except Exception as e:
                logger.error(f"发送解决通知到通道失败: {e}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.last_update = utc_now()
            
            # 更新Redis
            await self.redis.hset(
                "active_alerts",
                alert_id,
                alert.model_dump_json()
            )
            
            logger.info(f"告警已确认: {alert_id} by {acknowledged_by}")
    
    async def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    async def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        try:
            resolved_alerts_data = await self.redis.hgetall("resolved_alerts")
            alerts = []
            
            cutoff_time = utc_now() - timedelta(hours=hours)
            
            for alert_data in resolved_alerts_data.values():
                alert = Alert.model_validate_json(alert_data)
                if alert.start_time > cutoff_time:
                    alerts.append(alert)
            
            return sorted(alerts, key=lambda a: a.start_time, reverse=True)
            
        except Exception as e:
            logger.error(f"获取告警历史失败: {e}")
            return []
    
    async def start(self):
        """启动告警管理器"""
        if self._running:
            return
        
        self._running = True
        logger.info("启动告警管理器")
        
        # 启动周期性任务
        self._tasks = [
            asyncio.create_task(self._cleanup_old_metrics()),
            asyncio.create_task(self._health_check())
        ]
    
    async def stop(self):
        """停止告警管理器"""
        if not self._running:
            return
        
        self._running = False
        logger.info("停止告警管理器")
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
    
    async def _cleanup_old_metrics(self):
        """清理旧指标数据"""
        while self._running:
            try:
                # 每10分钟清理一次
                await asyncio.sleep(600)
                
                cutoff_time = utc_now() - timedelta(hours=2)
                
                for metric_name in list(self.metric_history.keys()):
                    self.metric_history[metric_name] = [
                        (t, v) for t, v in self.metric_history[metric_name]
                        if t > cutoff_time
                    ]
                    
                    # 如果没有数据了，删除该指标
                    if not self.metric_history[metric_name]:
                        del self.metric_history[metric_name]
                
                logger.debug("已清理旧指标数据")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理旧指标数据失败: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再试
    
    async def _health_check(self):
        """健康检查"""
        while self._running:
            try:
                # 每5分钟检查一次
                await asyncio.sleep(300)
                
                # 检查Redis连接
                await self.redis.ping()
                
                # 检查告警通道
                for i, channel in enumerate(self.alert_channels):
                    # 这里可以添加通道健康检查逻辑
                    health_check = getattr(channel, "health_check", None)
                    if health_check:
                        result = health_check()
                        if inspect.isawaitable(result):
                            await result
                
                logger.debug("告警管理器健康检查通过")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"告警管理器健康检查失败: {e}")
                await asyncio.sleep(60)

# 预定义的告警规则
def get_default_alert_rules() -> List[AlertRule]:
    """获取默认告警规则"""
    return [
        # 推荐延迟告警
        AlertRule(
            name="high_recommendation_latency",
            description="推荐响应延迟过高",
            metric_name="recommendation_latency_p99",
            threshold=100.0,
            comparison=">",
            severity=AlertSeverity.HIGH,
            duration=300,
            tags={"component": "recommendation", "metric_type": "latency"}
        ),
        
        # 特征计算延迟告警
        AlertRule(
            name="high_feature_computation_latency",
            description="特征计算延迟过高",
            metric_name="feature_computation_latency_avg",
            threshold=10.0,
            comparison=">",
            severity=AlertSeverity.CRITICAL,
            duration=180,
            tags={"component": "feature", "metric_type": "latency"}
        ),
        
        # 缓存命中率告警
        AlertRule(
            name="low_cache_hit_rate",
            description="缓存命中率过低",
            metric_name="cache_hit_rate",
            threshold=0.8,
            comparison="<",
            severity=AlertSeverity.MEDIUM,
            duration=600,
            tags={"component": "cache", "metric_type": "hit_rate"}
        ),
        
        # 错误率告警
        AlertRule(
            name="high_error_rate",
            description="系统错误率过高",
            metric_name="error_rate",
            threshold=0.01,
            comparison=">",
            severity=AlertSeverity.HIGH,
            duration=180,
            tags={"component": "system", "metric_type": "error_rate"}
        ),
        
        # QPS告警
        AlertRule(
            name="low_qps",
            description="系统QPS异常低",
            metric_name="requests_per_second",
            threshold=100.0,
            comparison="<",
            severity=AlertSeverity.MEDIUM,
            duration=300,
            tags={"component": "system", "metric_type": "throughput"}
        ),
        
        # 内存使用告警
        AlertRule(
            name="high_memory_usage",
            description="内存使用率过高",
            metric_name="memory_usage_percent",
            threshold=85.0,
            comparison=">",
            severity=AlertSeverity.HIGH,
            duration=300,
            tags={"component": "system", "metric_type": "resource"}
        ),
        
        # CPU使用告警
        AlertRule(
            name="high_cpu_usage",
            description="CPU使用率过高",
            metric_name="cpu_usage_percent",
            threshold=80.0,
            comparison=">",
            severity=AlertSeverity.MEDIUM,
            duration=600,
            tags={"component": "system", "metric_type": "resource"}
        ),
        
        # 模型预测失败告警
        AlertRule(
            name="model_prediction_failures",
            description="模型预测失败率过高",
            metric_name="model_prediction_failure_rate",
            threshold=0.05,
            comparison=">",
            severity=AlertSeverity.CRITICAL,
            duration=180,
            tags={"component": "model", "metric_type": "failure_rate"}
        )
    ]
