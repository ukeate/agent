"""
实时安全监控系统
"""

import asyncio
import re
import uuid
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from fastapi import Request
from pydantic import BaseModel
from src.core.config import get_settings
from src.core.redis import get_redis

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class ThreatLevel(str, Enum):
    """威胁级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityEventType(str, Enum):
    """安全事件类型"""
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_REQUEST = "suspicious_request"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL = "path_traversal"
    BRUTE_FORCE = "brute_force"
    DDOS_ATTEMPT = "ddos_attempt"
    DATA_BREACH = "data_breach"
    ANOMALY = "anomaly"

class SecurityAssessment(BaseModel):
    """安全评估结果"""
    risk_score: float  # 0.0 - 1.0
    threat_level: ThreatLevel
    detected_threats: List[SecurityEventType]
    details: Dict[str, Any]
    recommendations: List[str]

class SecurityAlert(BaseModel):
    """安全告警"""
    id: str
    alert_type: SecurityEventType
    threat_level: ThreatLevel
    description: str
    affected_resource: str
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    status: str  # active, investigating, resolved, false_positive
    auto_blocked: bool
    action_taken: List[str]

class SecurityMonitor:
    """安全监控器"""
    
    def __init__(self):
        self.redis = None
        self.alert_queue: List[SecurityAlert] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        
    def _load_suspicious_patterns(self) -> Dict[str, re.Pattern]:
        """加载可疑模式"""
        return {
            "sql_injection": re.compile(
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b|--|;|\*|\/\*|\*\/)",
                re.IGNORECASE
            ),
            "xss": re.compile(
                r"(<script|<iframe|javascript:|onerror=|onload=|alert\(|prompt\(|confirm\()",
                re.IGNORECASE
            ),
            "path_traversal": re.compile(
                r"(\.\./|\.\.\\|%2e%2e|%252e%252e)"
            ),
            "command_injection": re.compile(
                r"(;|\||&&|\$\(|`|>|<|\n|\r)"
            ),
            "ldap_injection": re.compile(
                r"(\*|\(|\)|\\|NULL|\x00)"
            )
        }
    
    async def assess_request(self, request: Request) -> SecurityAssessment:
        """评估请求安全性"""
        risk_score = 0.0
        detected_threats = []
        details = {}
        recommendations = []
        
        # 获取请求信息
        client_ip = request.client.host if request.client else "unknown"
        path = str(request.url.path)
        query_params = dict(request.query_params) if request.query_params else {}
        headers = dict(request.headers)
        
        # 白名单本地IP和开发环境IP
        trusted_ips = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
        if client_ip in trusted_ips:
            return SecurityAssessment(
                risk_score=0.0,
                threat_level=ThreatLevel.LOW,
                detected_threats=[],
                details={"trusted_ip": True, "client_ip": client_ip},
                recommendations=["Request from trusted IP"]
            )
        
        # 1. 检查IP是否在黑名单
        if client_ip in self.blocked_ips:
            risk_score = 1.0
            detected_threats.append(SecurityEventType.UNAUTHORIZED_ACCESS)
            details["blocked_ip"] = client_ip
            recommendations.append("IP is blocked. Deny access.")
        
        # 2. 检查请求频率
        rate_check = await self._check_request_rate(client_ip)
        if rate_check["exceeded"]:
            risk_score = max(risk_score, 0.7)
            detected_threats.append(SecurityEventType.RATE_LIMIT)
            details["request_rate"] = rate_check["rate"]
            recommendations.append(f"Rate limit exceeded: {rate_check['rate']} req/min")
        
        # 3. 检查SQL注入
        sql_risk = self._check_sql_injection(path, query_params)
        if sql_risk > 0:
            risk_score = max(risk_score, sql_risk)
            detected_threats.append(SecurityEventType.SQL_INJECTION)
            details["sql_injection_risk"] = sql_risk
            recommendations.append("Potential SQL injection detected")
        
        # 4. 检查XSS攻击
        xss_risk = self._check_xss(path, query_params, headers)
        if xss_risk > 0:
            risk_score = max(risk_score, xss_risk)
            detected_threats.append(SecurityEventType.XSS_ATTEMPT)
            details["xss_risk"] = xss_risk
            recommendations.append("Potential XSS attack detected")
        
        # 5. 检查路径遍历
        if self._check_path_traversal(path):
            risk_score = max(risk_score, 0.8)
            detected_threats.append(SecurityEventType.PATH_TRAVERSAL)
            details["path_traversal"] = True
            recommendations.append("Path traversal attempt detected")
        
        # 6. 检查异常请求模式
        anomaly_score = await self._check_anomalies(request)
        if anomaly_score > 0.5:
            risk_score = max(risk_score, anomaly_score)
            detected_threats.append(SecurityEventType.ANOMALY)
            details["anomaly_score"] = anomaly_score
            recommendations.append("Anomalous request pattern detected")
        
        # 7. 检查暴力破解
        if await self._check_brute_force(client_ip, path):
            risk_score = max(risk_score, 0.9)
            detected_threats.append(SecurityEventType.BRUTE_FORCE)
            details["brute_force"] = True
            recommendations.append("Potential brute force attack")
        
        # 确定威胁级别
        if risk_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif risk_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW
        
        # 如果风险过高，自动阻断
        if risk_score >= settings.AUTO_BLOCK_THRESHOLD:
            await self._auto_block(client_ip, detected_threats)
            recommendations.append(f"IP {client_ip} has been automatically blocked")
        
        return SecurityAssessment(
            risk_score=risk_score,
            threat_level=threat_level,
            detected_threats=detected_threats,
            details=details,
            recommendations=recommendations
        )
    
    async def _check_request_rate(self, client_ip: str) -> Dict[str, Any]:
        """检查请求频率"""
        now = utc_now()
        minute_ago = now - timedelta(minutes=1)
        
        # 清理旧记录
        self.request_history[client_ip] = [
            t for t in self.request_history[client_ip]
            if t > minute_ago
        ]
        
        # 添加当前请求
        self.request_history[client_ip].append(now)
        
        # 计算请求频率
        rate = len(self.request_history[client_ip])
        max_rate = settings.MAX_REQUESTS_PER_MINUTE
        
        return {
            "rate": rate,
            "exceeded": rate > max_rate,
            "limit": max_rate
        }
    
    def _check_sql_injection(self, path: str, params: Dict[str, Any]) -> float:
        """检查SQL注入风险"""
        risk_score = 0.0
        pattern = self.suspicious_patterns["sql_injection"]
        
        # 检查路径
        if pattern.search(path):
            risk_score = max(risk_score, 0.7)
        
        # 检查参数
        for key, value in params.items():
            if isinstance(value, str) and pattern.search(value):
                risk_score = max(risk_score, 0.9)
        
        return risk_score
    
    def _check_xss(self, path: str, params: Dict[str, Any], headers: Dict[str, str]) -> float:
        """检查XSS风险"""
        risk_score = 0.0
        pattern = self.suspicious_patterns["xss"]
        
        # 检查路径
        if pattern.search(path):
            risk_score = max(risk_score, 0.6)
        
        # 检查参数
        for key, value in params.items():
            if isinstance(value, str) and pattern.search(value):
                risk_score = max(risk_score, 0.8)
        
        # 检查特定头部
        dangerous_headers = ["Referer", "User-Agent"]
        for header in dangerous_headers:
            if header in headers and pattern.search(headers[header]):
                risk_score = max(risk_score, 0.5)
        
        return risk_score
    
    def _check_path_traversal(self, path: str) -> bool:
        """检查路径遍历"""
        pattern = self.suspicious_patterns["path_traversal"]
        return bool(pattern.search(path))
    
    async def _check_anomalies(self, request: Request) -> float:
        """检查异常请求模式"""
        anomaly_score = 0.0
        
        # 检查请求大小
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > settings.MAX_REQUEST_SIZE:
                    anomaly_score = max(anomaly_score, 0.6)
            except ValueError:
                anomaly_score = max(anomaly_score, 0.4)
        
        # 检查异常User-Agent
        user_agent = request.headers.get("user-agent", "")
        suspicious_agents = ["bot", "scanner", "crawler", "nikto", "sqlmap"]
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            anomaly_score = max(anomaly_score, 0.5)
        
        # 检查缺失的标准头部
        expected_headers = ["user-agent", "accept", "accept-language"]
        missing_headers = [h for h in expected_headers if h not in request.headers]
        if len(missing_headers) >= 2:
            anomaly_score = max(anomaly_score, 0.3)
        
        return anomaly_score
    
    async def _check_brute_force(self, client_ip: str, path: str) -> bool:
        """检查暴力破解尝试"""
        # 检查认证相关路径
        auth_paths = ["/login", "/api/v1/auth", "/token", "/signin"]
        if not any(auth_path in path for auth_path in auth_paths):
            return False
        
        if not self.redis:
            return False

        key = f"security:auth_attempts:{client_ip}"
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, 300)
        return count >= 20
    
    async def _auto_block(self, client_ip: str, threats: List[SecurityEventType]):
        """自动阻断IP"""
        self.blocked_ips.add(client_ip)
        
        # 在Redis中设置阻断记录（24小时过期）
        if self.redis:
            key = f"blocked_ip:{client_ip}"
            await self.redis.setex(key, 86400, "blocked")
        
        # 创建安全告警
        alert = SecurityAlert(
            id=str(uuid.uuid4()),
            alert_type=threats[0] if threats else SecurityEventType.ANOMALY,
            threat_level=ThreatLevel.CRITICAL,
            description=f"IP {client_ip} automatically blocked due to security threats",
            affected_resource="api",
            source_ip=client_ip,
            user_id=None,
            timestamp=utc_now(),
            status="active",
            auto_blocked=True,
            action_taken=["ip_blocked", "alert_created"]
        )
        
        self.alert_queue.append(alert)
        
        logger.warning(
            "IP automatically blocked",
            ip=client_ip,
            threats=threats
        )
    
    async def get_active_alerts(self) -> List[SecurityAlert]:
        """获取活跃告警"""
        return [
            alert for alert in self.alert_queue
            if alert.status in ["active", "investigating"]
        ]
    
    async def resolve_alert(self, alert_id: str, resolution: str = "resolved"):
        """解决告警"""
        for alert in self.alert_queue:
            if alert.id == alert_id:
                alert.status = resolution
                logger.info(
                    "Security alert resolved",
                    alert_id=alert_id,
                    resolution=resolution
                )
                break
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """获取安全指标"""
        now = utc_now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # 计算指标
        total_requests = sum(len(reqs) for reqs in self.request_history.values())
        
        recent_alerts = [
            alert for alert in self.alert_queue
            if alert.timestamp > hour_ago
        ]
        
        critical_alerts = [
            alert for alert in recent_alerts
            if alert.threat_level == ThreatLevel.CRITICAL
        ]
        
        return {
            "total_requests_last_hour": total_requests,
            "blocked_ips": len(self.blocked_ips),
            "active_alerts": len([a for a in self.alert_queue if a.status == "active"]),
            "alerts_last_hour": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "threat_distribution": self._get_threat_distribution(),
            "top_blocked_ips": list(self.blocked_ips)[:10]
        }
    
    def _get_threat_distribution(self) -> Dict[str, int]:
        """获取威胁分布"""
        distribution = defaultdict(int)
        for alert in self.alert_queue:
            distribution[alert.alert_type] += 1
        return dict(distribution)
    
    async def perform_risk_assessment(self) -> Dict[str, Any]:
        """执行风险评估"""
        metrics = await self.get_security_metrics()
        
        # 计算总体风险分数
        risk_factors = {
            "blocked_ips": min(metrics["blocked_ips"] * 0.1, 0.3),
            "critical_alerts": min(metrics["critical_alerts"] * 0.2, 0.4),
            "active_alerts": min(metrics["active_alerts"] * 0.05, 0.2),
            "request_volume": 0.1 if metrics["total_requests_last_hour"] > 10000 else 0
        }
        
        overall_risk = sum(risk_factors.values())
        
        # 确定风险级别
        if overall_risk >= 0.7:
            risk_level = "critical"
        elif overall_risk >= 0.5:
            risk_level = "high"
        elif overall_risk >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "metrics": metrics,
            "recommendations": self._generate_recommendations(risk_level, metrics)
        }
    
    def _generate_recommendations(self, risk_level: str, metrics: Dict[str, Any]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if risk_level in ["critical", "high"]:
            recommendations.append("Consider enabling enhanced monitoring")
            recommendations.append("Review and update security policies")
        
        if metrics["blocked_ips"] > 10:
            recommendations.append("High number of blocked IPs detected. Consider implementing CAPTCHA")
        
        if metrics["critical_alerts"] > 5:
            recommendations.append("Multiple critical alerts. Immediate investigation required")
        
        if not recommendations:
            recommendations.append("System security is within normal parameters")
        
        return recommendations

# 全局实例
security_monitor = SecurityMonitor()
