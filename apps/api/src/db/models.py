"""
数据库模型定义
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from src.core.utils.timezone_utils import utc_now, utc_factory

class ConversationStatus(str, Enum):
    """对话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"

class SenderType(str, Enum):
    """发送者类型枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Conversation:
    """对话模型"""
    id: str
    user_id: str
    title: str
    agent_type: str = "react"
    status: ConversationStatus = ConversationStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class Message:
    """消息模型"""
    id: str
    conversation_id: str
    content: str
    sender_type: SenderType
    message_type: MessageType = MessageType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_factory)

@dataclass
class Task:
    """任务模型"""
    id: str
    conversation_id: str
    description: str
    task_type: str = "general"
    status: str = "pending"
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    completed_at: Optional[datetime] = None

# ============================================================================
# 平台集成相关模型
# ============================================================================

class ComponentTypeEnum(str, Enum):
    """组件类型枚举"""
    FINE_TUNING = "fine_tuning"
    COMPRESSION = "compression"
    HYPERPARAMETER = "hyperparameter"
    EVALUATION = "evaluation"
    DATA_MANAGEMENT = "data_management"
    MODEL_SERVICE = "model_service"
    CUSTOM = "custom"

class ComponentStatusEnum(str, Enum):
    """组件状态枚举"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"

class WorkflowStatusEnum(str, Enum):
    """工作流状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class PlatformComponent:
    """平台组件模型"""
    id: str
    component_id: str
    component_type: ComponentTypeEnum
    name: str
    version: str
    status: ComponentStatusEnum
    health_endpoint: str
    api_endpoint: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=utc_factory)
    last_heartbeat: datetime = field(default_factory=utc_factory)
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class WorkflowExecution:
    """工作流执行模型"""
    id: str
    workflow_id: str
    workflow_type: str
    status: WorkflowStatusEnum
    parameters: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0
    started_at: datetime = field(default_factory=utc_factory)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowStep:
    """工作流步骤模型"""
    id: str
    workflow_execution_id: str
    step_name: str
    step_order: int
    status: WorkflowStatusEnum
    component_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
    """性能指标模型"""
    id: str
    metric_name: str
    metric_value: float
    metric_unit: str
    component_id: Optional[str] = None
    workflow_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_factory)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringAlert:
    """监控告警模型"""
    id: str
    alert_name: str
    alert_level: str  # info, warning, critical
    message: str
    component_id: Optional[str] = None
    workflow_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    is_resolved: bool = False
    created_at: datetime = field(default_factory=utc_factory)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlatformConfiguration:
    """平台配置模型"""
    id: str
    config_key: str
    config_value: str
    config_type: str = "string"  # string, integer, float, boolean, json
    description: Optional[str] = None
    is_sensitive: bool = False
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    updated_by: Optional[str] = None

@dataclass
class ComponentDependency:
    """组件依赖关系模型"""
    id: str
    component_id: str
    depends_on_component_id: str
    dependency_type: str = "required"  # required, optional, preferred
    created_at: datetime = field(default_factory=utc_factory)

@dataclass
class AuditLog:
    """审计日志模型"""
    id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=utc_factory)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# 分布式安全框架模型
# ============================================================================

class SecurityEventTypeEnum(str, Enum):
    """安全事件类型枚举"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILED = "auth_failed"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    THREAT_DETECTED = "threat_detected"
    SECURITY_SCAN = "security_scan"
    CONFIGURATION_CHANGE = "config_change"
    AUDIT_LOG_ACCESS = "audit_log_access"

class ThreatLevelEnum(str, Enum):
    """威胁级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatusEnum(str, Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AuthenticationMethodEnum(str, Enum):
    """认证方法枚举"""
    PASSWORD = "password"
    PKI_CERTIFICATE = "pki_certificate"
    OAUTH2 = "oauth2"
    MFA = "mfa"
    BIOMETRIC = "biometric"
    API_KEY = "api_key"

class AccessControlModelEnum(str, Enum):
    """访问控制模型枚举"""
    RBAC = "rbac"  # Role-Based Access Control
    ABAC = "abac"  # Attribute-Based Access Control
    HYBRID = "hybrid"

class EncryptionAlgorithmEnum(str, Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    ECDH_P256 = "ecdh_p256"

@dataclass
class AgentIdentity:
    """智能体身份模型"""
    id: str
    agent_id: str
    display_name: str
    agent_type: str
    public_key: str
    certificate_data: Optional[str] = None
    biometric_templates: Dict[str, Any] = field(default_factory=dict)
    authentication_methods: list[AuthenticationMethodEnum] = field(default_factory=list)
    trust_score: float = 100.0
    last_authentication: Optional[datetime] = None
    failed_attempts: int = 0
    is_locked: bool = False
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class SecurityRole:
    """安全角色模型"""
    id: str
    role_name: str
    description: str
    permissions: list[str] = field(default_factory=list)
    is_system_role: bool = False
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    created_by: Optional[str] = None

@dataclass
class AccessPolicy:
    """访问策略模型"""
    id: str
    policy_name: str
    description: str
    resource_pattern: str
    action_pattern: str
    policy_type: AccessControlModelEnum
    rules: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    created_by: Optional[str] = None

@dataclass
class AgentRoleAssignment:
    """智能体角色分配模型"""
    id: str
    agent_id: str
    role_id: str
    granted_by: str
    granted_at: datetime = field(default_factory=utc_factory)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """安全事件模型"""
    id: str
    event_type: SecurityEventTypeEnum
    agent_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    outcome: str = "unknown"  # success, failure, blocked, allowed
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_factory)

@dataclass
class SecurityAlert:
    """安全告警模型"""
    id: str
    alert_type: str
    title: str
    description: str
    threat_level: ThreatLevelEnum
    status: AlertStatusEnum = AlertStatusEnum.ACTIVE
    agent_id: Optional[str] = None
    source_event_ids: list[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestions: list[str] = field(default_factory=list)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommunicationSession:
    """通信会话模型"""
    id: str
    session_id: str
    initiator_agent_id: str
    target_agent_id: str
    encryption_algorithm: EncryptionAlgorithmEnum
    session_key_hash: str
    start_time: datetime = field(default_factory=utc_factory)
    end_time: Optional[datetime] = None
    messages_count: int = 0
    bytes_transferred: int = 0
    is_active: bool = True
    integrity_verified: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessLog:
    """访问日志模型"""
    id: str
    agent_id: str
    resource: str
    action: str
    access_granted: bool
    policy_decisions: Dict[str, Any] = field(default_factory=dict)
    context_attributes: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    duration_ms: Optional[int] = None
    timestamp: datetime = field(default_factory=utc_factory)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatPattern:
    """威胁模式模型"""
    id: str
    pattern_name: str
    description: str
    pattern_type: str  # brute_force, privilege_escalation, data_exfiltration, etc.
    detection_rules: Dict[str, Any] = field(default_factory=dict)
    severity_score: float = 0.0
    is_active: bool = True
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    created_by: Optional[str] = None

@dataclass
class AgentKey:
    """智能体密钥模型"""
    id: str
    agent_id: str
    key_type: str  # public, private, symmetric
    key_algorithm: EncryptionAlgorithmEnum
    key_data_hash: str  # 密钥数据的哈希值，不存储明文密钥
    key_purpose: str  # authentication, encryption, signing
    created_at: datetime = field(default_factory=utc_factory)
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Certificate:
    """证书模型"""
    id: str
    certificate_id: str
    agent_id: str
    certificate_type: str  # x509, jwt, custom
    issuer: str
    subject: str
    serial_number: str
    certificate_data: str  # PEM格式证书数据
    public_key: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    is_valid: bool = True
    trust_chain: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_factory)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditConfiguration:
    """审计配置模型"""
    id: str
    config_name: str
    event_types: list[SecurityEventTypeEnum] = field(default_factory=list)
    retention_days: int = 90
    log_level: str = "INFO"
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_settings: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    updated_by: Optional[str] = None
