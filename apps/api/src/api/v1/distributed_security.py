"""
分布式安全框架 API接口
提供身份认证、访问控制、安全审计等服务的REST API
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from pydantic import Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from src.core.config import get_settings
from src.api.base_model import ApiBaseModel
from src.ai.autogen.security.identity_authentication import (
    IdentityAuthenticationService, AuthenticationMethod, AuthenticationResult
)
from src.ai.autogen.security.access_control import (
    AccessControlEngine, AccessRequest, ResourceType, AccessDecision, AccessPolicy
)
from src.ai.autogen.security.security_audit import (
    SecurityAuditSystem, SecurityEvent, EventType, ThreatLevel
)
from src.ai.autogen.security.encrypted_communication import (
    EncryptedCommunicationFramework, MessageType

)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/distributed-security", tags=["distributed-security"])
security = HTTPBearer()

# Pydantic 模型定义

class AuthenticationRequest(ApiBaseModel):
    agent_id: str = Field(..., description="智能体ID")
    credentials: Dict[str, Any] = Field(..., description="认证凭据")
    authentication_methods: List[str] = Field(..., description="认证方法列表")

class AuthenticationResponse(ApiBaseModel):
    authenticated: bool
    session_token: Optional[str] = None
    trust_score: float
    error_message: Optional[str] = None

class AccessControlRequest(ApiBaseModel):
    subject_id: str = Field(..., description="主体ID")
    resource_id: str = Field(..., description="资源ID")
    action: str = Field(..., description="操作类型")
    resource_type: str = Field(..., description="资源类型")
    context: Dict[str, Any] = Field(default_factory=dict, description="请求上下文")

class AccessControlResponse(ApiBaseModel):
    decision: str
    reason: str
    request_id: str
    evaluation_time_ms: float

class SecurityEventRequest(ApiBaseModel):
    event_type: str = Field(..., description="事件类型")
    source_agent_id: str = Field(..., description="源智能体ID")
    target_resource: Optional[str] = Field(None, description="目标资源")
    action: str = Field(..., description="操作类型")
    result: str = Field(..., description="操作结果")
    details: Dict[str, Any] = Field(default_factory=dict, description="事件详情")

class SecurityEventResponse(ApiBaseModel):
    event_id: str
    logged: bool
    message: str

class SecureCommunicationRequest(ApiBaseModel):
    sender_id: str = Field(..., description="发送方ID")
    recipient_id: str = Field(..., description="接收方ID")
    message: Dict[str, Any] = Field(..., description="消息内容")
    session_id: Optional[str] = Field(None, description="通信会话ID")

class SecureCommunicationResponse(ApiBaseModel):
    session_id: str
    message_id: str
    encrypted: bool
    message: str

class PolicyRequest(ApiBaseModel):
    policy_id: str
    name: str
    description: str
    target: Dict[str, Any]
    rules: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True

class AlertResponse(ApiBaseModel):
    alert_id: str
    threat_level: str
    confidence_score: float
    description: str
    timestamp: float

class SecurityDashboardResponse(ApiBaseModel):
    total_events: int
    event_by_type: Dict[str, int]
    events_by_threat_level: Dict[str, int]
    events_by_result: Dict[str, int]
    active_alerts_count: int
    active_alerts: List[AlertResponse]
    high_risk_agents: List[Dict[str, Any]]
    time_range_hours: float

# 全局服务实例（在实际部署中应该通过依赖注入管理）
_auth_service = None
_access_control = None
_audit_system = None
_comm_framework = None

# 依赖注入
async def get_auth_service():
    """获取身份认证服务"""
    global _auth_service
    if _auth_service is None:
        try:
            settings = get_settings()
            config = {
                'redis_url': settings.REDIS_URL,
                'jwt_secret': settings.SECRET_KEY,
                'session_timeout': 3600,
                'min_trust_score': 0.6,
                'ca_certificates': []
            }
            _auth_service = IdentityAuthenticationService(config)
            await _auth_service.initialize()
        except Exception as e:
            logger.error(f"身份认证服务初始化失败: {e}")
            raise HTTPException(status_code=503, detail="认证服务不可用")
    return _auth_service

async def get_access_control():
    """获取访问控制引擎"""
    global _access_control
    if _access_control is None:
        try:
            config = {
                'max_log_entries': 10000,
                'max_time_skew': 300
            }
            _access_control = AccessControlEngine(config)
            await _access_control.initialize()
        except Exception as e:
            logger.error(f"访问控制引擎初始化失败: {e}")
            raise HTTPException(status_code=503, detail="访问控制服务不可用")
    return _access_control

async def get_audit_system():
    """获取安全审计系统"""
    global _audit_system
    if _audit_system is None:
        try:
            config = {
                'redis_url': 'redis://localhost:6379',
                'max_events': 100000,
                'detection_window': 300,
                'anomaly_threshold': 0.8,
                'max_failed_attempts': 5
            }
            _audit_system = SecurityAuditSystem(config)
            await _audit_system.initialize()
        except Exception as e:
            logger.error(f"安全审计系统初始化失败: {e}")
            raise HTTPException(status_code=503, detail="审计系统不可用")
    return _audit_system

async def get_communication_framework():
    """获取加密通信框架"""
    global _comm_framework
    if _comm_framework is None:
        try:
            config = {
                'key_rotation_interval': 3600,
                'message_ttl': 300
            }
            _comm_framework = EncryptedCommunicationFramework(config)
            await _comm_framework.initialize()
        except Exception as e:
            logger.error(f"加密通信框架初始化失败: {e}")
            raise HTTPException(status_code=503, detail="加密通信框架不可用")
    return _comm_framework

# API路由定义

@router.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_agent(
    request: AuthenticationRequest,
    auth_service = Depends(get_auth_service)
):
    """智能体身份认证"""
    try:
        result = await auth_service.authenticate_agent(
            request.agent_id,
            request.credentials,
            request.authentication_methods
        )
        
        return AuthenticationResponse(
            authenticated=result.authenticated,
            session_token=result.session_token,
            trust_score=result.trust_score,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"认证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/authorize", response_model=AccessControlResponse)
async def authorize_access(
    request: AccessControlRequest,
    access_control = Depends(get_access_control)
):
    """访问授权检查"""
    try:
        access_request = AccessRequest(
            subject_id=request.subject_id,
            resource_id=request.resource_id,
            action=request.action,
            resource_type=request.resource_type,
            context=request.context
        )
        result = await access_control.evaluate_access(access_request)
        
        return AccessControlResponse(
            decision=result.decision.value if hasattr(result, "decision") else result["decision"].value,
            reason=result.reason if hasattr(result, "reason") else result["reason"],
            request_id=result.request_id if hasattr(result, "request_id") else result["request_id"],
            evaluation_time_ms=result.evaluation_time_ms if hasattr(result, "evaluation_time_ms") else result["evaluation_time_ms"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"授权失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events", response_model=SecurityEventResponse)
async def log_security_event(
    request: SecurityEventRequest,
    audit_system = Depends(get_audit_system)
):
    """记录安全事件"""
    try:
        try:
            event_type = EventType(request.event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的事件类型")

        event = SecurityEvent(
            event_id=f"evt_{int(time.time())}_{request.source_agent_id}",
            event_type=event_type,
            timestamp=time.time(),
            source_agent_id=request.source_agent_id,
            target_resource=request.target_resource,
            action=request.action,
            result=request.result,
            details=request.details
        )
        await audit_system.log_security_event(event)
        return SecurityEventResponse(
            event_id=event.event_id,
            logged=True,
            message="安全事件记录成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"记录安全事件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard", response_model=SecurityDashboardResponse)
async def get_security_dashboard(
    time_range: int = 86400,  # 24小时
    audit_system = Depends(get_audit_system)
):
    """获取安全仪表板数据"""
    try:
        dashboard_data = await audit_system.get_security_dashboard(time_range)
        alerts = []
        for alert in dashboard_data.get('active_alerts', []):
            alerts.append(AlertResponse(
                alert_id=alert.get('alert_id'),
                threat_level=alert.get('threat_level'),
                confidence_score=alert.get('confidence_score', 0.0),
                description=alert.get('description', ''),
                timestamp=alert.get('timestamp', time.time())
            ))
        return SecurityDashboardResponse(
            total_events=dashboard_data.get('total_events', 0),
            event_by_type=dashboard_data.get('event_by_type', {}),
            events_by_threat_level=dashboard_data.get('events_by_threat_level', {}),
            events_by_result=dashboard_data.get('events_by_result', {}),
            active_alerts_count=dashboard_data.get('active_alerts_count', len(alerts)),
            active_alerts=alerts,
            high_risk_agents=dashboard_data.get('high_risk_agents', []),
            time_range_hours=dashboard_data.get('time_range_hours', time_range / 3600)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取安全仪表盘失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_active_alerts(
    limit: int = 50,
    audit_system = Depends(get_audit_system)
):
    """获取活跃告警"""
    try:
        alerts = await audit_system.get_active_alerts(limit=limit)
        return {
            'alerts': alerts,
            'total_count': len(alerts)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_notes: str,
    audit_system = Depends(get_audit_system)
):
    """处理安全告警"""
    try:
        success = await audit_system.resolve_alert(alert_id, resolution_notes)
        
        if success:
            return {"message": f"告警 {alert_id} 已成功解决"}
        else:
            raise HTTPException(status_code=404, detail="告警未找到")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/communication/encrypt", response_model=SecureCommunicationResponse)
async def encrypt_message(
    request: SecureCommunicationRequest,
    comm_framework = Depends(get_communication_framework)
):
    """加密消息通信"""
    try:
        session_id = request.session_id
        
        # 如果没有会话ID，建立新的安全通道
        if not session_id:
            sender_public_key = await comm_framework.ensure_agent_public_key(request.sender_id)
            recipient_public_key = await comm_framework.ensure_agent_public_key(request.recipient_id)
            
            session_id = await comm_framework.establish_secure_channel(
                request.sender_id,
                request.recipient_id,
                sender_public_key,
                recipient_public_key
            )
        
        # 加密消息
        encrypted_message = await comm_framework.encrypt_message(
            session_id,
            request.sender_id,
            request.recipient_id,
            request.message,
            'encrypted_message'
        )
        
        return SecureCommunicationResponse(
            session_id=session_id,
            message_id=encrypted_message.message_id,
            encrypted=True,
            message="消息加密成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加密消息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/policies")
async def add_security_policy(
    request: PolicyRequest,
    access_control = Depends(get_access_control)
):
    """添加安全策略"""
    try:
        policy = AccessPolicy(
            policy_id=request.policy_id,
            name=request.name,
            description=request.description,
            target=request.target,
            rules=request.rules,
            priority=request.priority,
            enabled=request.enabled,
        )
        success = await access_control.add_policy(policy)
        if not success:
            raise HTTPException(status_code=400, detail="添加策略失败")
        return {"message": f"策略 {request.policy_id} 已成功添加"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加策略失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/policies")
async def get_security_policies(
    access_control = Depends(get_access_control)
):
    """获取安全策略列表"""
    try:
        policies = [
            {
                "policy_id": p.policy_id,
                "name": p.name,
                "description": p.description,
                "target": p.target,
                "rules": p.rules,
                "priority": p.priority,
                "enabled": p.enabled,
                "created_at": p.created_at,
            }
            for p in access_control.policies.values()
        ]
        return {"policies": policies, "total_count": len(policies)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/access-logs")
async def get_access_logs(
    subject_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    limit: int = 1000,
    access_control = Depends(get_access_control)
):
    """获取访问日志"""
    try:
        logs = await access_control.get_access_logs(subject_id, resource_id, limit)
        return {
            'logs': logs,
            'total_count': len(logs),
            'filters': {
                'subject_id': subject_id,
                'resource_id': resource_id,
                'limit': limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取访问日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/revoke-access")
async def revoke_agent_access(
    agent_id: str,
    reason: str,
    auth_service = Depends(get_auth_service)
):
    """撤销智能体访问权限"""
    try:
        await auth_service.revoke_agent_access(agent_id, reason)
        
        return {"message": f"已撤销智能体 {agent_id} 的访问权限"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"撤销访问权限失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def security_health_check(
    auth_service = Depends(get_auth_service),
    access_control = Depends(get_access_control),
    audit_system = Depends(get_audit_system),
    comm_framework = Depends(get_communication_framework),
):
    """安全服务健康检查"""
    return {
        "status": "healthy",
        "components": {
            "auth_service": bool(auth_service),
            "access_control": bool(access_control),
            "audit_system": bool(audit_system),
            "communication_framework": bool(comm_framework),
        },
        "timestamp": time.time(),
    }

@router.get("/metrics")
async def get_security_metrics(
    auth_service = Depends(get_auth_service),
    access_control = Depends(get_access_control),
    audit_system = Depends(get_audit_system),
    comm_framework = Depends(get_communication_framework),
):
    """获取安全指标"""
    return {
        "timestamp": time.time(),
        "authentication": {
            "ca_certificates": len(getattr(auth_service, "ca_certificates", {}) or {}),
            "revoked_certificates": len(getattr(auth_service, "revoked_certificates", set()) or set()),
        },
        "access_control": {
            "policies": len(getattr(access_control, "policies", {}) or {}),
            "roles": len(getattr(access_control, "roles", {}) or {}),
            "subjects": len(getattr(access_control, "subjects", {}) or {}),
            "access_logs": len(getattr(access_control, "access_logs", []) or []),
        },
        "audit": {
            "events": len(getattr(audit_system, "events", []) or []),
            "threat_patterns": len(getattr(audit_system, "threat_patterns", {}) or {}),
            "active_alerts": len(getattr(audit_system, "active_alerts", {}) or {}),
        },
        "communication": {
            "sessions": len(getattr(comm_framework, "sessions", {}) or {}),
        },
    }
