"""
自动化安全响应系统
实现威胁自动检测、分类和响应机制
"""
import asyncio
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog

from .trism import ThreatLevel, SecurityEvent
from .attack_detection import AttackType, DetectionResult

logger = structlog.get_logger(__name__)


class ResponseAction(str, Enum):
    """响应动作"""
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ALERT = "alert"
    LOG = "log"
    ESCALATE = "escalate"
    RATE_LIMIT = "rate_limit"
    REQUIRE_REVIEW = "require_review"
    DISABLE_AGENT = "disable_agent"
    ROLLBACK = "rollback"
    RETRAIN = "retrain"


class ResponseStatus(str, Enum):
    """响应状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResponseRule:
    """响应规则"""
    id: str
    name: str
    conditions: Dict[str, Any]
    actions: List[ResponseAction]
    priority: int = 0
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def matches(self, event: SecurityEvent) -> bool:
        """检查是否匹配条件"""
        try:
            # 检查威胁级别
            threat_condition = self.conditions.get("threat_level")
            if threat_condition and event.threat_level.value not in threat_condition:
                return False
            
            # 检查事件类型
            type_condition = self.conditions.get("event_type")
            if type_condition and event.event_type not in type_condition:
                return False
            
            # 检查信任影响
            trust_condition = self.conditions.get("trust_impact")
            if trust_condition:
                min_impact = trust_condition.get("min", 0.0)
                max_impact = trust_condition.get("max", 1.0)
                if not (min_impact <= event.trust_impact <= max_impact):
                    return False
            
            # 检查风险评分
            risk_condition = self.conditions.get("risk_score")
            if risk_condition:
                min_risk = risk_condition.get("min", 0.0)
                max_risk = risk_condition.get("max", 1.0)
                if not (min_risk <= event.risk_score <= max_risk):
                    return False
            
            # 检查智能体条件
            agent_condition = self.conditions.get("agent_id")
            if agent_condition and event.source_agent not in agent_condition:
                return False
            
            return True
            
        except Exception as e:
            logger.error("规则匹配检查失败", rule_id=self.id, error=str(e))
            return False


@dataclass
class ResponseExecution:
    """响应执行记录"""
    id: str
    rule_id: str
    event_id: str
    actions: List[ResponseAction]
    status: ResponseStatus = ResponseStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "event_id": self.event_id,
            "actions": [action.value for action in self.actions],
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results,
            "errors": self.errors
        }


class ActionExecutor:
    """动作执行器基类"""
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行动作"""
        raise NotImplementedError


class BlockExecutor(ActionExecutor):
    """阻止执行器"""
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """阻止智能体输出"""
        try:
            agent_id = event.source_agent
            
            # 阻止当前输出
            blocked_output_id = context.get("output_id")
            
            # 记录阻止动作
            result = {
                "action": "block",
                "agent_id": agent_id,
                "blocked_output_id": blocked_output_id,
                "reason": f"威胁级别: {event.threat_level.value}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("执行阻止动作", agent_id=agent_id, threat_level=event.threat_level)
            
            return result
            
        except Exception as e:
            logger.error("阻止动作执行失败", error=str(e))
            return {"action": "block", "error": str(e)}


class QuarantineExecutor(ActionExecutor):
    """隔离执行器"""
    
    def __init__(self):
        self.quarantined_agents: Set[str] = set()
        self.quarantine_duration = timedelta(hours=1)  # 默认隔离1小时
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """隔离智能体"""
        try:
            agent_id = event.source_agent
            
            # 添加到隔离列表
            self.quarantined_agents.add(agent_id)
            
            # 设置隔离到期时间
            quarantine_until = datetime.now(timezone.utc) + self.quarantine_duration
            
            result = {
                "action": "quarantine",
                "agent_id": agent_id,
                "quarantine_until": quarantine_until.isoformat(),
                "reason": f"安全事件: {event.event_type}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.warning("执行隔离动作", agent_id=agent_id, until=quarantine_until)
            
            return result
            
        except Exception as e:
            logger.error("隔离动作执行失败", error=str(e))
            return {"action": "quarantine", "error": str(e)}
    
    def is_quarantined(self, agent_id: str) -> bool:
        """检查智能体是否被隔离"""
        return agent_id in self.quarantined_agents
    
    def release_quarantine(self, agent_id: str) -> bool:
        """解除隔离"""
        if agent_id in self.quarantined_agents:
            self.quarantined_agents.remove(agent_id)
            logger.info("解除隔离", agent_id=agent_id)
            return True
        return False


class AlertExecutor(ActionExecutor):
    """告警执行器"""
    
    def __init__(self):
        self.alert_handlers: List[Callable] = []
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """发送告警"""
        try:
            alert_data = {
                "alert_id": str(uuid.uuid4()),
                "event_id": event.event_id,
                "agent_id": event.source_agent,
                "threat_level": event.threat_level.value,
                "event_type": event.event_type,
                "details": event.details,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # 调用所有告警处理器
            handler_results = []
            for handler in self.alert_handlers:
                try:
                    handler_result = await handler(alert_data)
                    handler_results.append(handler_result)
                except Exception as e:
                    logger.error("告警处理器失败", error=str(e))
                    handler_results.append({"error": str(e)})
            
            result = {
                "action": "alert",
                "alert_id": alert_data["alert_id"],
                "handlers_notified": len(self.alert_handlers),
                "handler_results": handler_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("执行告警动作", alert_id=alert_data["alert_id"])
            
            return result
            
        except Exception as e:
            logger.error("告警动作执行失败", error=str(e))
            return {"action": "alert", "error": str(e)}
    
    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers.append(handler)


class RateLimitExecutor(ActionExecutor):
    """限流执行器"""
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.default_limit = 10  # 默认每分钟10次
        self.default_window = 60  # 60秒窗口
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用限流"""
        try:
            agent_id = event.source_agent
            current_time = datetime.now(timezone.utc)
            
            # 获取或创建限流记录
            if agent_id not in self.rate_limits:
                self.rate_limits[agent_id] = {
                    "limit": self.default_limit,
                    "window": self.default_window,
                    "requests": [],
                    "blocked_until": None
                }
            
            rate_limit = self.rate_limits[agent_id]
            
            # 根据威胁级别调整限制
            if event.threat_level == ThreatLevel.CRITICAL:
                rate_limit["limit"] = 1
                rate_limit["window"] = 300  # 5分钟
            elif event.threat_level == ThreatLevel.HIGH:
                rate_limit["limit"] = 3
                rate_limit["window"] = 180  # 3分钟
            
            # 设置阻止时间
            block_duration = timedelta(seconds=rate_limit["window"])
            rate_limit["blocked_until"] = current_time + block_duration
            
            result = {
                "action": "rate_limit",
                "agent_id": agent_id,
                "new_limit": rate_limit["limit"],
                "window_seconds": rate_limit["window"],
                "blocked_until": rate_limit["blocked_until"].isoformat(),
                "timestamp": current_time.isoformat()
            }
            
            logger.info("执行限流动作", agent_id=agent_id, limit=rate_limit["limit"])
            
            return result
            
        except Exception as e:
            logger.error("限流动作执行失败", error=str(e))
            return {"action": "rate_limit", "error": str(e)}
    
    def is_rate_limited(self, agent_id: str) -> bool:
        """检查是否被限流"""
        if agent_id not in self.rate_limits:
            return False
        
        rate_limit = self.rate_limits[agent_id]
        blocked_until = rate_limit.get("blocked_until")
        
        if blocked_until and datetime.now(timezone.utc) < blocked_until:
            return True
        
        return False


class EscalateExecutor(ActionExecutor):
    """升级执行器"""
    
    def __init__(self):
        self.escalation_handlers: List[Callable] = []
    
    async def execute(
        self, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """升级处理"""
        try:
            escalation_data = {
                "escalation_id": str(uuid.uuid4()),
                "event_id": event.event_id,
                "agent_id": event.source_agent,
                "threat_level": event.threat_level.value,
                "escalation_reason": "自动安全响应升级",
                "event_details": event.details,
                "recommended_actions": event.mitigation_actions,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # 调用升级处理器
            handler_results = []
            for handler in self.escalation_handlers:
                try:
                    handler_result = await handler(escalation_data)
                    handler_results.append(handler_result)
                except Exception as e:
                    logger.error("升级处理器失败", error=str(e))
                    handler_results.append({"error": str(e)})
            
            result = {
                "action": "escalate",
                "escalation_id": escalation_data["escalation_id"],
                "handlers_notified": len(self.escalation_handlers),
                "handler_results": handler_results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.warning("执行升级动作", escalation_id=escalation_data["escalation_id"])
            
            return result
            
        except Exception as e:
            logger.error("升级动作执行失败", error=str(e))
            return {"action": "escalate", "error": str(e)}
    
    def add_escalation_handler(self, handler: Callable):
        """添加升级处理器"""
        self.escalation_handlers.append(handler)


class SecurityResponseManager:
    """安全响应管理器"""
    
    def __init__(self):
        self.rules: Dict[str, ResponseRule] = {}
        self.executors: Dict[ResponseAction, ActionExecutor] = {
            ResponseAction.BLOCK: BlockExecutor(),
            ResponseAction.QUARANTINE: QuarantineExecutor(),
            ResponseAction.ALERT: AlertExecutor(),
            ResponseAction.RATE_LIMIT: RateLimitExecutor(),
            ResponseAction.ESCALATE: EscalateExecutor()
        }
        self.executions: Dict[str, ResponseExecution] = {}
        self.event_history: List[SecurityEvent] = []
        self.max_history = 10000
        
        # 初始化默认规则
        self._initialize_default_rules()
        
        logger.info("安全响应管理器初始化完成")
    
    def _initialize_default_rules(self):
        """初始化默认响应规则"""
        # 关键威胁立即阻止
        critical_rule = ResponseRule(
            id="critical_threat_block",
            name="关键威胁阻止",
            conditions={
                "threat_level": ["critical"]
            },
            actions=[ResponseAction.BLOCK, ResponseAction.ALERT, ResponseAction.ESCALATE],
            priority=100
        )
        self.add_rule(critical_rule)
        
        # 高威胁隔离和告警
        high_threat_rule = ResponseRule(
            id="high_threat_quarantine",
            name="高威胁隔离",
            conditions={
                "threat_level": ["high"]
            },
            actions=[ResponseAction.QUARANTINE, ResponseAction.ALERT, ResponseAction.RATE_LIMIT],
            priority=80
        )
        self.add_rule(high_threat_rule)
        
        # 中等威胁限流
        medium_threat_rule = ResponseRule(
            id="medium_threat_limit",
            name="中等威胁限流",
            conditions={
                "threat_level": ["medium"]
            },
            actions=[ResponseAction.RATE_LIMIT, ResponseAction.ALERT],
            priority=60
        )
        self.add_rule(medium_threat_rule)
        
        # 数据泄露特殊处理
        data_leakage_rule = ResponseRule(
            id="data_leakage_response",
            name="数据泄露响应",
            conditions={
                "event_type": ["data_leakage"]
            },
            actions=[ResponseAction.BLOCK, ResponseAction.QUARANTINE, ResponseAction.ESCALATE],
            priority=90
        )
        self.add_rule(data_leakage_rule)
    
    async def handle_security_event(
        self, 
        event: SecurityEvent, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[ResponseExecution]:
        """处理安全事件"""
        try:
            if context is None:
                context = {}
            
            # 记录事件
            self._record_event(event)
            
            # 查找匹配的规则
            matching_rules = self._find_matching_rules(event)
            
            if not matching_rules:
                logger.info("未找到匹配规则", event_id=event.event_id)
                return []
            
            # 按优先级排序
            matching_rules.sort(key=lambda r: r.priority, reverse=True)
            
            # 执行响应
            executions = []
            for rule in matching_rules:
                if rule.enabled:
                    execution = await self._execute_rule(rule, event, context)
                    executions.append(execution)
            
            logger.info(
                "安全事件处理完成",
                event_id=event.event_id,
                executions=len(executions)
            )
            
            return executions
            
        except Exception as e:
            logger.error("安全事件处理失败", event_id=event.event_id, error=str(e))
            return []
    
    async def _execute_rule(
        self, 
        rule: ResponseRule, 
        event: SecurityEvent, 
        context: Dict[str, Any]
    ) -> ResponseExecution:
        """执行响应规则"""
        execution_id = str(uuid.uuid4())
        execution = ResponseExecution(
            id=execution_id,
            rule_id=rule.id,
            event_id=event.event_id,
            actions=rule.actions,
            started_at=datetime.now(timezone.utc)
        )
        
        self.executions[execution_id] = execution
        execution.status = ResponseStatus.IN_PROGRESS
        
        try:
            # 执行所有动作
            for action in rule.actions:
                if action in self.executors:
                    executor = self.executors[action]
                    result = await executor.execute(event, context)
                    execution.results[action.value] = result
                else:
                    execution.errors.append(f"未找到执行器: {action.value}")
            
            execution.status = ResponseStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            logger.info("规则执行完成", rule_id=rule.id, execution_id=execution_id)
            
        except Exception as e:
            execution.status = ResponseStatus.FAILED
            execution.errors.append(str(e))
            execution.completed_at = datetime.now(timezone.utc)
            
            logger.error("规则执行失败", rule_id=rule.id, error=str(e))
        
        return execution
    
    def _find_matching_rules(self, event: SecurityEvent) -> List[ResponseRule]:
        """查找匹配的规则"""
        matching_rules = []
        
        for rule in self.rules.values():
            if rule.enabled and rule.matches(event):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _record_event(self, event: SecurityEvent):
        """记录安全事件"""
        self.event_history.append(event)
        
        # 保持历史记录在合理大小
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history // 2:]
        
        logger.debug("记录安全事件", event_id=event.event_id, event_type=event.event_type)
    
    def add_rule(self, rule: ResponseRule):
        """添加响应规则"""
        self.rules[rule.id] = rule
        logger.info("添加响应规则", rule_id=rule.id, name=rule.name)
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除响应规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info("移除响应规则", rule_id=rule_id)
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用响应规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info("启用响应规则", rule_id=rule_id)
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用响应规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info("禁用响应规则", rule_id=rule_id)
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[ResponseRule]:
        """获取响应规则"""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[ResponseRule]:
        """列出所有规则"""
        return list(self.rules.values())
    
    def get_execution(self, execution_id: str) -> Optional[ResponseExecution]:
        """获取执行记录"""
        return self.executions.get(execution_id)
    
    def list_executions(
        self, 
        rule_id: Optional[str] = None,
        status: Optional[ResponseStatus] = None,
        limit: int = 100
    ) -> List[ResponseExecution]:
        """列出执行记录"""
        executions = list(self.executions.values())
        
        # 应用过滤器
        if rule_id:
            executions = [e for e in executions if e.rule_id == rule_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        # 按时间排序
        executions.sort(key=lambda e: e.started_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        
        return executions[:limit]
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        status = {
            "agent_id": agent_id,
            "is_quarantined": False,
            "is_rate_limited": False,
            "recent_events": 0,
            "threat_level": "low"
        }
        
        # 检查隔离状态
        quarantine_executor = self.executors.get(ResponseAction.QUARANTINE)
        if isinstance(quarantine_executor, QuarantineExecutor):
            status["is_quarantined"] = quarantine_executor.is_quarantined(agent_id)
        
        # 检查限流状态
        rate_limit_executor = self.executors.get(ResponseAction.RATE_LIMIT)
        if isinstance(rate_limit_executor, RateLimitExecutor):
            status["is_rate_limited"] = rate_limit_executor.is_rate_limited(agent_id)
        
        # 统计最近事件
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_events = [
            e for e in self.event_history 
            if e.source_agent == agent_id and e.timestamp > recent_cutoff
        ]
        
        status["recent_events"] = len(recent_events)
        
        if recent_events:
            max_threat = max(recent_events, key=lambda e: ["low", "medium", "high", "critical"].index(e.threat_level.value))
            status["threat_level"] = max_threat.threat_level.value
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)
        total_executions = len(self.executions)
        
        # 执行状态统计
        status_counts = {}
        for status in ResponseStatus:
            status_counts[status.value] = sum(
                1 for execution in self.executions.values() 
                if execution.status == status
            )
        
        # 威胁级别统计
        threat_counts = {}
        for threat_level in ThreatLevel:
            threat_counts[threat_level.value] = sum(
                1 for event in self.event_history 
                if event.threat_level == threat_level
            )
        
        # 最近24小时事件
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_events = [
            e for e in self.event_history 
            if e.timestamp > recent_cutoff
        ]
        
        return {
            "rules": {
                "total": total_rules,
                "enabled": enabled_rules,
                "disabled": total_rules - enabled_rules
            },
            "executions": {
                "total": total_executions,
                "by_status": status_counts
            },
            "events": {
                "total": len(self.event_history),
                "recent_24h": len(recent_events),
                "by_threat_level": threat_counts
            },
            "active_agents": len(set(e.source_agent for e in recent_events))
        }
    
    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        alert_executor = self.executors.get(ResponseAction.ALERT)
        if isinstance(alert_executor, AlertExecutor):
            alert_executor.add_alert_handler(handler)
    
    def add_escalation_handler(self, handler: Callable):
        """添加升级处理器"""
        escalate_executor = self.executors.get(ResponseAction.ESCALATE)
        if isinstance(escalate_executor, EscalateExecutor):
            escalate_executor.add_escalation_handler(handler)