"""
分布式安全框架 - 安全审计系统
支持威胁检测、安全告警和实时监控
"""

import asyncio
import time
import json
import hashlib
import secrets
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from src.core.logging import get_logger
class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    SYSTEM_CHANGE = "system_change"
    NETWORK_ACTIVITY = "network_activity"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class SecurityEvent:
    event_id: str
    event_type: EventType
    timestamp: float
    source_agent_id: str
    target_resource: Optional[str]
    action: str
    result: str  # success, failure, blocked
    details: Dict[str, Any] = field(default_factory=dict)
    threat_level: ThreatLevel = ThreatLevel.LOW
    risk_score: float = 0.0

@dataclass
class ThreatPattern:
    pattern_id: str
    name: str
    description: str
    event_types: List[EventType]
    conditions: Dict[str, Any]
    threat_level: ThreatLevel
    confidence_threshold: float

@dataclass
class SecurityAlert:
    alert_id: str
    threat_pattern_id: str
    triggered_events: List[str]  # event_ids
    timestamp: float
    threat_level: ThreatLevel
    confidence_score: float
    description: str
    recommended_actions: List[str]
    resolved: bool = False

class SecurityAuditSystem:
    """安全审计系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events: deque = deque(maxlen=config.get('max_events', 100000))
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.agent_behaviors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.redis = None
        
        # 威胁检测配置
        self.detection_window = config.get('detection_window', 300)  # 5分钟
        self.anomaly_threshold = config.get('anomaly_threshold', 0.8)
        self.max_failed_attempts = config.get('max_failed_attempts', 5)
        
        # 设置日志记录器
        self.logger = get_logger(__name__)
        
    async def initialize(self):
        """初始化安全审计系统"""
        self.redis = redis.from_url(self.config.get('redis_url', 'redis://localhost:6379'))
        
        await self._load_threat_patterns()
        await self._setup_monitoring()
        
        self.logger.info("Security audit system initialized")
        
    async def _load_threat_patterns(self):
        """加载威胁模式"""
        # 暴力破解攻击模式
        self.threat_patterns['brute_force'] = ThreatPattern(
            pattern_id='brute_force',
            name='暴力破解攻击',
            description='检测短时间内多次失败的认证尝试',
            event_types=[EventType.AUTHENTICATION],
            conditions={
                'time_window': 300,
                'min_failures': 5,
                'failure_rate_threshold': 0.8
            },
            threat_level=ThreatLevel.HIGH,
            confidence_threshold=0.9
        )
        
        # 权限升级攻击模式
        self.threat_patterns['privilege_escalation'] = ThreatPattern(
            pattern_id='privilege_escalation',
            name='权限升级攻击',
            description='检测异常的权限获取或使用行为',
            event_types=[EventType.AUTHORIZATION],
            conditions={
                'unusual_resource_access': True,
                'role_change_frequency_threshold': 3
            },
            threat_level=ThreatLevel.CRITICAL,
            confidence_threshold=0.8
        )
        
        # 数据泄露模式
        self.threat_patterns['data_exfiltration'] = ThreatPattern(
            pattern_id='data_exfiltration',
            name='数据泄露攻击',
            description='检测异常的数据访问和传输行为',
            event_types=[EventType.DATA_ACCESS],
            conditions={
                'large_data_access': True,
                'unusual_time_access': True,
                'data_volume_threshold': 1000000  # 1MB
            },
            threat_level=ThreatLevel.CRITICAL,
            confidence_threshold=0.7
        )
        
        # 横向移动模式
        self.threat_patterns['lateral_movement'] = ThreatPattern(
            pattern_id='lateral_movement',
            name='横向移动攻击',
            description='检测在网络中的异常移动行为',
            event_types=[EventType.NETWORK_ACTIVITY],
            conditions={
                'unusual_network_connections': True,
                'connection_frequency_threshold': 10,
                'time_window': 600
            },
            threat_level=ThreatLevel.HIGH,
            confidence_threshold=0.8
        )
        
        self.logger.info(f"Loaded {len(self.threat_patterns)} threat patterns")
    
    async def log_security_event(self, event: SecurityEvent):
        """记录安全事件"""
        try:
            # 计算风险分数
            event.risk_score = await self._calculate_risk_score(event)
            
            # 确定威胁级别
            event.threat_level = await self._determine_threat_level(event)
            
            # 存储事件
            self.events.append(event)
            
            # 更新智能体行为模式
            self.agent_behaviors[event.source_agent_id].append({
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'action': event.action,
                'result': event.result,
                'risk_score': event.risk_score
            })
            
            # 缓存到Redis
            await self._cache_event(event)
            
            # 实时威胁检测
            await self._detect_threats(event)
            
            # 记录到日志
            self.logger.info(f"Security event logged: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
    
    async def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """计算风险分数"""
        risk_score = 0.0
        
        # 基础风险分数
        base_scores = {
            EventType.AUTHENTICATION: 0.3,
            EventType.AUTHORIZATION: 0.5,
            EventType.DATA_ACCESS: 0.6,
            EventType.SYSTEM_CHANGE: 0.8,
            EventType.NETWORK_ACTIVITY: 0.4,
            EventType.SECURITY_VIOLATION: 1.0
        }
        
        risk_score = base_scores.get(event.event_type, 0.5)
        
        # 结果调整
        if event.result == 'failure':
            risk_score += 0.2
        elif event.result == 'blocked':
            risk_score += 0.3
        
        # 时间因子（非工作时间增加风险）
        hour = datetime.fromtimestamp(event.timestamp).hour
        if hour < 6 or hour > 22:  # 非工作时间
            risk_score += 0.1
        
        # 智能体历史行为
        agent_history = self.agent_behaviors.get(event.source_agent_id, deque())
        if len(agent_history) > 0:
            recent_failures = sum(1 for h in list(agent_history)[-10:] if h['result'] == 'failure')
            if recent_failures > 3:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    async def _determine_threat_level(self, event: SecurityEvent) -> ThreatLevel:
        """确定威胁级别"""
        if event.risk_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif event.risk_score >= 0.7:
            return ThreatLevel.HIGH
        elif event.risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    async def _cache_event(self, event: SecurityEvent):
        """缓存事件到Redis"""
        try:
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp,
                'source_agent_id': event.source_agent_id,
                'target_resource': event.target_resource,
                'action': event.action,
                'result': event.result,
                'details': json.dumps(event.details),
                'threat_level': event.threat_level.value,
                'risk_score': event.risk_score
            }
            
            # 缓存单个事件
            await self.redis.hset(f"security_event:{event.event_id}", mapping=event_data)
            await self.redis.expire(f"security_event:{event.event_id}", 86400 * 30)  # 30天过期
            
            # 添加到时间序列索引
            await self.redis.zadd(
                "security_events_timeline", 
                {event.event_id: event.timestamp}
            )
            
            # 添加到智能体索引
            await self.redis.sadd(f"agent_events:{event.source_agent_id}", event.event_id)
            
        except Exception as e:
            self.logger.error(f"Failed to cache event: {e}")
    
    async def _detect_threats(self, event: SecurityEvent):
        """实时威胁检测"""
        for pattern in self.threat_patterns.values():
            if event.event_type in pattern.event_types:
                if await self._evaluate_threat_pattern(pattern, event):
                    await self._generate_alert(pattern, event)
    
    async def _evaluate_threat_pattern(
        self, 
        pattern: ThreatPattern, 
        event: SecurityEvent
    ) -> bool:
        """评估威胁模式"""
        try:
            if pattern.pattern_id == 'brute_force':
                return await self._detect_brute_force(pattern, event)
            elif pattern.pattern_id == 'privilege_escalation':
                return await self._detect_privilege_escalation(pattern, event)
            elif pattern.pattern_id == 'data_exfiltration':
                return await self._detect_data_exfiltration(pattern, event)
            elif pattern.pattern_id == 'lateral_movement':
                return await self._detect_lateral_movement(pattern, event)
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to evaluate threat pattern {pattern.pattern_id}: {e}")
            return False
    
    async def _detect_brute_force(
        self, 
        pattern: ThreatPattern, 
        event: SecurityEvent
    ) -> bool:
        """检测暴力破解攻击"""
        if event.event_type != EventType.AUTHENTICATION or event.result != 'failure':
            return False
        
        # 查找时间窗口内的失败认证事件
        time_window = pattern.conditions['time_window']
        min_failures = pattern.conditions['min_failures']
        
        cutoff_time = event.timestamp - time_window
        recent_events = [
            e for e in self.events 
            if (e.timestamp >= cutoff_time and 
                e.source_agent_id == event.source_agent_id and
                e.event_type == EventType.AUTHENTICATION and
                e.result == 'failure')
        ]
        
        if len(recent_events) >= min_failures:
            # 计算失败率
            total_auth_events = [
                e for e in self.events
                if (e.timestamp >= cutoff_time and
                    e.source_agent_id == event.source_agent_id and
                    e.event_type == EventType.AUTHENTICATION)
            ]
            
            if total_auth_events:
                failure_rate = len(recent_events) / len(total_auth_events)
                return failure_rate >= pattern.conditions['failure_rate_threshold']
        
        return False
    
    async def _detect_privilege_escalation(
        self, 
        pattern: ThreatPattern, 
        event: SecurityEvent
    ) -> bool:
        """检测权限升级攻击"""
        if event.event_type != EventType.AUTHORIZATION:
            return False
        
        # 检查是否访问了异常资源
        agent_history = list(self.agent_behaviors.get(event.source_agent_id, []))
        if not agent_history:
            return False
        
        # 获取历史访问的资源
        historical_resources = set()
        for h in agent_history[-50:]:  # 检查最近50个事件
            if 'target_resource' in h:
                historical_resources.add(h['target_resource'])
        
        # 检查当前访问的资源是否异常
        if event.target_resource and event.target_resource not in historical_resources:
            # 进一步检查是否是高权限资源
            if await self._is_high_privilege_resource(event.target_resource):
                return True
        
        return False
    
    async def _detect_data_exfiltration(
        self, 
        pattern: ThreatPattern, 
        event: SecurityEvent
    ) -> bool:
        """检测数据泄露攻击"""
        if event.event_type != EventType.DATA_ACCESS:
            return False
        
        conditions = pattern.conditions
        
        # 检查数据访问量
        data_size = event.details.get('data_size', 0)
        if data_size > conditions['data_volume_threshold']:
            return True
        
        # 检查访问时间是否异常
        if conditions['unusual_time_access']:
            hour = datetime.fromtimestamp(event.timestamp).hour
            if hour < 6 or hour > 22:  # 非工作时间
                return True
        
        # 检查访问频率
        recent_data_access = [
            e for e in list(self.events)[-100:]  # 检查最近100个事件
            if (e.source_agent_id == event.source_agent_id and
                e.event_type == EventType.DATA_ACCESS and
                e.timestamp >= event.timestamp - 3600)  # 1小时内
        ]
        
        if len(recent_data_access) > 10:  # 1小时内超过10次数据访问
            return True
        
        return False
    
    async def _detect_lateral_movement(
        self, 
        pattern: ThreatPattern, 
        event: SecurityEvent
    ) -> bool:
        """检测横向移动攻击"""
        if event.event_type != EventType.NETWORK_ACTIVITY:
            return False
        
        conditions = pattern.conditions
        time_window = conditions['time_window']
        
        # 检查网络连接频率
        cutoff_time = event.timestamp - time_window
        recent_network_events = [
            e for e in self.events
            if (e.timestamp >= cutoff_time and
                e.source_agent_id == event.source_agent_id and
                e.event_type == EventType.NETWORK_ACTIVITY)
        ]
        
        if len(recent_network_events) > conditions['connection_frequency_threshold']:
            # 检查连接的目标是否异常多样化
            target_hosts = set()
            for e in recent_network_events:
                target_host = e.details.get('target_host')
                if target_host:
                    target_hosts.add(target_host)
            
            # 如果连接了过多不同的主机，可能是横向移动
            if len(target_hosts) > 5:
                return True
        
        return False
    
    async def _generate_alert(self, pattern: ThreatPattern, triggering_event: SecurityEvent):
        """生成安全告警"""
        alert_id = f"alert_{pattern.pattern_id}_{int(time.time())}_{secrets.token_hex(4)}"
        
        # 收集相关事件
        related_events = await self._collect_related_events(pattern, triggering_event)
        
        # 计算置信度分数
        confidence_score = await self._calculate_confidence_score(pattern, related_events)
        
        if confidence_score >= pattern.confidence_threshold:
            alert = SecurityAlert(
                alert_id=alert_id,
                threat_pattern_id=pattern.pattern_id,
                triggered_events=[e.event_id for e in related_events],
                timestamp=time.time(),
                threat_level=pattern.threat_level,
                confidence_score=confidence_score,
                description=f"{pattern.name}: {pattern.description}",
                recommended_actions=await self._get_recommended_actions(pattern)
            )
            
            self.active_alerts[alert_id] = alert
            
            # 缓存告警
            await self._cache_alert(alert)
            
            # 发送通知
            await self._send_alert_notification(alert)
            
            self.logger.warning(f"Security alert generated: {alert_id} - {pattern.name}")
    
    async def _collect_related_events(
        self, 
        pattern: ThreatPattern, 
        triggering_event: SecurityEvent
    ) -> List[SecurityEvent]:
        """收集相关事件"""
        related_events = [triggering_event]
        
        # 根据模式类型收集相关事件
        if pattern.pattern_id == 'brute_force':
            # 收集相同智能体的认证失败事件
            cutoff_time = triggering_event.timestamp - pattern.conditions['time_window']
            for event in reversed(list(self.events)):
                if event.timestamp < cutoff_time:
                    break
                if (event.source_agent_id == triggering_event.source_agent_id and
                    event.event_type == EventType.AUTHENTICATION and
                    event.result == 'failure' and
                    event.event_id != triggering_event.event_id):
                    related_events.append(event)
        
        return related_events
    
    async def _calculate_confidence_score(
        self, 
        pattern: ThreatPattern, 
        related_events: List[SecurityEvent]
    ) -> float:
        """计算置信度分数"""
        if not related_events:
            return 0.0
        
        # 基础置信度
        base_confidence = 0.5
        
        # 事件数量因子
        event_count_factor = min(len(related_events) / 10.0, 0.3)
        
        # 风险分数因子
        avg_risk_score = sum(e.risk_score for e in related_events) / len(related_events)
        risk_factor = avg_risk_score * 0.4
        
        # 时间聚集因子
        if len(related_events) > 1:
            time_span = max(e.timestamp for e in related_events) - min(e.timestamp for e in related_events)
            if time_span < 300:  # 5分钟内聚集
                time_factor = 0.2
            else:
                time_factor = 0.0
        else:
            time_factor = 0.0
        
        confidence = base_confidence + event_count_factor + risk_factor + time_factor
        return min(confidence, 1.0)
    
    async def _get_recommended_actions(self, pattern: ThreatPattern) -> List[str]:
        """获取推荐的响应行动"""
        actions = []
        
        if pattern.pattern_id == 'brute_force':
            actions = [
                "临时封锁源IP地址",
                "增加认证延迟",
                "要求额外的验证因子",
                "通知系统管理员"
            ]
        elif pattern.pattern_id == 'privilege_escalation':
            actions = [
                "立即撤销异常权限",
                "重置智能体访问令牌",
                "启动深度安全审计",
                "隔离受影响的智能体"
            ]
        elif pattern.pattern_id == 'data_exfiltration':
            actions = [
                "阻断数据传输",
                "启动数据泄露响应程序",
                "审查数据访问日志",
                "通知数据保护官员"
            ]
        elif pattern.pattern_id == 'lateral_movement':
            actions = [
                "限制网络访问权限",
                "启动网络流量分析",
                "隔离可疑网络段",
                "部署蜜罐陷阱"
            ]
        
        return actions
    
    async def get_security_dashboard(self, time_range: int = 86400) -> Dict[str, Any]:
        """获取安全仪表板数据"""
        cutoff_time = time.time() - time_range
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # 事件统计
        event_stats = defaultdict(int)
        threat_level_stats = defaultdict(int)
        result_stats = defaultdict(int)
        
        for event in recent_events:
            event_stats[event.event_type.value] += 1
            threat_level_stats[event.threat_level.value] += 1
            result_stats[event.result] += 1
        
        # 活跃告警
        active_alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        # 风险最高的智能体
        agent_risk_scores = defaultdict(list)
        for event in recent_events:
            agent_risk_scores[event.source_agent_id].append(event.risk_score)
        
        high_risk_agents = []
        for agent_id, scores in agent_risk_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.6:
                high_risk_agents.append({
                    'agent_id': agent_id,
                    'avg_risk_score': avg_score,
                    'event_count': len(scores)
                })
        
        high_risk_agents.sort(key=lambda x: x['avg_risk_score'], reverse=True)
        
        return {
            'total_events': len(recent_events),
            'event_by_type': dict(event_stats),
            'events_by_threat_level': dict(threat_level_stats),
            'events_by_result': dict(result_stats),
            'active_alerts_count': len(active_alerts),
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'threat_level': alert.threat_level.value,
                    'confidence_score': alert.confidence_score,
                    'description': alert.description
                }
                for alert in active_alerts[:10]  # 最多显示10个
            ],
            'high_risk_agents': high_risk_agents[:10],  # 最多显示10个
            'time_range_hours': time_range / 3600
        }

    async def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取活跃告警列表"""
        active = [a for a in self.active_alerts.values() if not a.resolved]
        active.sort(key=lambda a: (a.threat_level.value, a.confidence_score, a.timestamp), reverse=True)
        return [
            {
                "alert_id": alert.alert_id,
                "threat_pattern_id": alert.threat_pattern_id,
                "triggered_events": alert.triggered_events,
                "timestamp": alert.timestamp,
                "threat_level": alert.threat_level.value,
                "confidence_score": alert.confidence_score,
                "description": alert.description,
                "recommended_actions": alert.recommended_actions,
                "resolved": alert.resolved,
            }
            for alert in active[:limit]
        ]
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str) -> bool:
        """处理告警"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                
                # 记录处理日志
                resolution_event = SecurityEvent(
                    event_id=f"resolution_{alert_id}_{int(time.time())}",
                    event_type=EventType.SECURITY_VIOLATION,
                    timestamp=time.time(),
                    source_agent_id='system',
                    target_resource=alert_id,
                    action='resolve_alert',
                    result='success',
                    details={
                        'alert_id': alert_id,
                        'resolution_notes': resolution_notes
                    }
                )
                
                await self.log_security_event(resolution_event)
                
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {e}")
        
        return False
    
    async def _is_high_privilege_resource(self, resource_id: str) -> bool:
        """检查是否是高权限资源"""
        high_privilege_patterns = [
            r'admin/*',
            r'*/config',
            r'*/system',
            r'*/security'
        ]
        
        for pattern in high_privilege_patterns:
            if re.match(pattern.replace('*', '.*'), resource_id):
                return True
        return False
    
    async def _cache_alert(self, alert: SecurityAlert):
        """缓存告警"""
        try:
            alert_data = {
                'alert_id': alert.alert_id,
                'threat_pattern_id': alert.threat_pattern_id,
                'timestamp': alert.timestamp,
                'threat_level': alert.threat_level.value,
                'confidence_score': alert.confidence_score,
                'description': alert.description,
                'resolved': alert.resolved
            }
            
            await self.redis.hset(f"security_alert:{alert.alert_id}", mapping=alert_data)
            await self.redis.expire(f"security_alert:{alert.alert_id}", 86400 * 7)  # 7天过期
            
        except Exception as e:
            self.logger.error(f"Failed to cache alert: {e}")
    
    async def _send_alert_notification(self, alert: SecurityAlert):
        """发送告警通知"""
        # 这里应该实现实际的通知机制（邮件、短信、Webhook等）
        self.logger.critical(f"SECURITY ALERT: {alert.description} (Confidence: {alert.confidence_score:.2f})")
    
    async def _setup_monitoring(self):
        """设置监控"""
        self.logger.info("Security monitoring setup completed")

class ThreatIntelligenceEngine:
    """威胁情报引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threat_feeds = {}
        self.ioc_cache = {}  # Indicators of Compromise
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """初始化威胁情报引擎"""
        await self._load_threat_feeds()
        self.logger.info("Threat intelligence engine initialized")
    
    async def _load_threat_feeds(self):
        """加载威胁情报源"""
        # 这里应该集成实际的威胁情报源
        self.threat_feeds['malicious_ips'] = {
            '192.168.100.100',
            '10.0.0.1'  # 示例恶意IP
        }
        
        self.threat_feeds['malicious_domains'] = {
            'malicious-domain.com',
            'phishing-site.net'
        }
    
    async def check_threat_intelligence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """检查威胁情报"""
        threat_indicators = []
        
        # 检查IP地址
        client_ip = context.get('client_ip')
        if client_ip in self.threat_feeds.get('malicious_ips', set()):
            threat_indicators.append({
                'type': 'malicious_ip',
                'value': client_ip,
                'severity': 'high'
            })
        
        # 检查域名
        domain = context.get('domain')
        if domain in self.threat_feeds.get('malicious_domains', set()):
            threat_indicators.append({
                'type': 'malicious_domain',
                'value': domain,
                'severity': 'high'
            })
        
        return {
            'threat_detected': len(threat_indicators) > 0,
            'threat_indicators': threat_indicators,
            'risk_score': len(threat_indicators) * 0.3
        }

class SecurityMetricsCollector:
    """安全指标收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = defaultdict(int)
        self.logger = get_logger(__name__)
    
    async def collect_metrics(self, audit_system: SecurityAuditSystem) -> Dict[str, Any]:
        """收集安全指标"""
        # 认证指标
        auth_events = [e for e in audit_system.events if e.event_type == EventType.AUTHENTICATION]
        auth_success_rate = len([e for e in auth_events if e.result == 'success']) / len(auth_events) if auth_events else 0
        
        # 告警指标
        total_alerts = len(audit_system.active_alerts)
        critical_alerts = len([a for a in audit_system.active_alerts.values() if a.threat_level == ThreatLevel.CRITICAL])
        
        # 智能体风险分布
        risk_distribution = defaultdict(int)
        for agent_id, behaviors in audit_system.agent_behaviors.items():
            if behaviors:
                avg_risk = sum(b['risk_score'] for b in behaviors) / len(behaviors)
                if avg_risk >= 0.8:
                    risk_distribution['high'] += 1
                elif avg_risk >= 0.5:
                    risk_distribution['medium'] += 1
                else:
                    risk_distribution['low'] += 1
        
        return {
            'authentication_success_rate': auth_success_rate,
            'total_security_events': len(audit_system.events),
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'agent_risk_distribution': dict(risk_distribution),
            'collection_timestamp': time.time()
        }
