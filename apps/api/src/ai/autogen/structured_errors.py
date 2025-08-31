"""
结构化错误处理系统
提供统一的错误代码、详细错误上下文和本地化错误消息
"""

import json
import traceback
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import structlog

logger = structlog.get_logger(__name__)


class ErrorCategory(str, Enum):
    """错误分类"""
    SYSTEM = "system"               # 系统错误
    CONFIGURATION = "configuration" # 配置错误
    VALIDATION = "validation"       # 验证错误
    AUTHENTICATION = "authentication" # 认证错误
    AUTHORIZATION = "authorization"  # 授权错误
    RESOURCE = "resource"           # 资源错误
    NETWORK = "network"             # 网络错误
    TIMEOUT = "timeout"             # 超时错误
    RATE_LIMIT = "rate_limit"       # 限流错误
    BUSINESS_LOGIC = "business_logic" # 业务逻辑错误


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    LOW = "low"         # 低：警告级别，不影响核心功能
    MEDIUM = "medium"   # 中：错误级别，影响部分功能
    HIGH = "high"       # 高：严重错误，影响主要功能
    CRITICAL = "critical" # 严重：系统级错误，影响整体运行


@dataclass
class ErrorContext:
    """错误上下文信息"""
    timestamp: datetime = field(default_factory=lambda: utc_now())
    node_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    version: str = "1.0.0"
    environment: str = "development"
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "conversation_id": self.conversation_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "component": self.component,
            "version": self.version,
            "environment": self.environment,
            "additional_data": self.additional_data
        }


@dataclass
class StructuredError:
    """结构化错误"""
    code: str                              # 错误代码
    message: str                           # 错误消息
    category: ErrorCategory                # 错误分类
    severity: ErrorSeverity                # 严重程度
    context: ErrorContext                  # 错误上下文
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    suggestions: List[str] = field(default_factory=list)   # 解决建议
    related_errors: List[str] = field(default_factory=list) # 相关错误
    stacktrace: Optional[str] = None       # 堆栈跟踪
    inner_exception: Optional[str] = None   # 内部异常
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "details": self.details,
            "suggestions": self.suggestions,
            "related_errors": self.related_errors,
            "stacktrace": self.stacktrace,
            "inner_exception": self.inner_exception
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ErrorCodes:
    """错误代码常量"""
    
    # 系统错误 (SYS-xxxx)
    SYSTEM_STARTUP_FAILED = "SYS-0001"
    SYSTEM_SHUTDOWN_FAILED = "SYS-0002"
    SYSTEM_RESOURCE_EXHAUSTED = "SYS-0003"
    SYSTEM_INTERNAL_ERROR = "SYS-0004"
    SYSTEM_DEPENDENCY_UNAVAILABLE = "SYS-0005"
    
    # 配置错误 (CFG-xxxx)
    CONFIG_FILE_NOT_FOUND = "CFG-0001"
    CONFIG_INVALID_FORMAT = "CFG-0002"
    CONFIG_MISSING_REQUIRED = "CFG-0003"
    CONFIG_VALUE_OUT_OF_RANGE = "CFG-0004"
    CONFIG_VALIDATION_FAILED = "CFG-0005"
    
    # 验证错误 (VAL-xxxx)
    VALIDATION_INPUT_INVALID = "VAL-0001"
    VALIDATION_REQUIRED_FIELD_MISSING = "VAL-0002"
    VALIDATION_FORMAT_INVALID = "VAL-0003"
    VALIDATION_LENGTH_INVALID = "VAL-0004"
    VALIDATION_CONSTRAINT_VIOLATED = "VAL-0005"
    
    # 认证错误 (AUTH-xxxx)
    AUTH_TOKEN_INVALID = "AUTH-0001"
    AUTH_TOKEN_EXPIRED = "AUTH-0002"
    AUTH_CREDENTIALS_INVALID = "AUTH-0003"
    AUTH_USER_NOT_FOUND = "AUTH-0004"
    AUTH_SESSION_EXPIRED = "AUTH-0005"
    
    # 授权错误 (AUTHZ-xxxx)
    AUTHZ_ACCESS_DENIED = "AUTHZ-0001"
    AUTHZ_INSUFFICIENT_PERMISSIONS = "AUTHZ-0002"
    AUTHZ_RESOURCE_FORBIDDEN = "AUTHZ-0003"
    AUTHZ_OPERATION_NOT_ALLOWED = "AUTHZ-0004"
    
    # 资源错误 (RES-xxxx)
    RESOURCE_NOT_FOUND = "RES-0001"
    RESOURCE_ALREADY_EXISTS = "RES-0002"
    RESOURCE_IN_USE = "RES-0003"
    RESOURCE_QUOTA_EXCEEDED = "RES-0004"
    RESOURCE_LOCKED = "RES-0005"
    
    # 网络错误 (NET-xxxx)
    NETWORK_CONNECTION_FAILED = "NET-0001"
    NETWORK_CONNECTION_TIMEOUT = "NET-0002"
    NETWORK_CONNECTION_LOST = "NET-0003"
    NETWORK_DNS_RESOLUTION_FAILED = "NET-0004"
    NETWORK_SERVICE_UNAVAILABLE = "NET-0005"
    
    # 超时错误 (TMO-xxxx)
    TIMEOUT_REQUEST_TIMEOUT = "TMO-0001"
    TIMEOUT_OPERATION_TIMEOUT = "TMO-0002"
    TIMEOUT_DEADLINE_EXCEEDED = "TMO-0003"
    TIMEOUT_LOCK_TIMEOUT = "TMO-0004"
    
    # 限流错误 (RATE-xxxx)
    RATE_LIMIT_EXCEEDED = "RATE-0001"
    RATE_LIMIT_QUOTA_EXCEEDED = "RATE-0002"
    RATE_LIMIT_CONCURRENT_LIMIT = "RATE-0003"
    
    # 业务逻辑错误 (BIZ-xxxx)
    BUSINESS_AGENT_NOT_AVAILABLE = "BIZ-0001"
    BUSINESS_TASK_FAILED = "BIZ-0002"
    BUSINESS_INVALID_STATE = "BIZ-0003"
    BUSINESS_WORKFLOW_ERROR = "BIZ-0004"
    BUSINESS_POOL_EXHAUSTED = "BIZ-0005"


class ErrorMessages:
    """错误消息模板"""
    
    MESSAGES = {
        # 系统错误消息
        ErrorCodes.SYSTEM_STARTUP_FAILED: "系统启动失败：{reason}",
        ErrorCodes.SYSTEM_SHUTDOWN_FAILED: "系统关闭失败：{reason}",
        ErrorCodes.SYSTEM_RESOURCE_EXHAUSTED: "系统资源不足：{resource_type}",
        ErrorCodes.SYSTEM_INTERNAL_ERROR: "系统内部错误：{details}",
        ErrorCodes.SYSTEM_DEPENDENCY_UNAVAILABLE: "依赖服务不可用：{service_name}",
        
        # 配置错误消息
        ErrorCodes.CONFIG_FILE_NOT_FOUND: "配置文件未找到：{file_path}",
        ErrorCodes.CONFIG_INVALID_FORMAT: "配置文件格式无效：{format_error}",
        ErrorCodes.CONFIG_MISSING_REQUIRED: "缺少必需的配置项：{config_key}",
        ErrorCodes.CONFIG_VALUE_OUT_OF_RANGE: "配置值超出范围：{config_key} = {value}，范围：{min_value}-{max_value}",
        ErrorCodes.CONFIG_VALIDATION_FAILED: "配置验证失败：{validation_errors}",
        
        # 验证错误消息
        ErrorCodes.VALIDATION_INPUT_INVALID: "输入数据无效：{field_name} = {value}",
        ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING: "缺少必需字段：{field_name}",
        ErrorCodes.VALIDATION_FORMAT_INVALID: "格式无效：{field_name}，期望格式：{expected_format}",
        ErrorCodes.VALIDATION_LENGTH_INVALID: "长度无效：{field_name}，当前长度：{current_length}，期望长度：{expected_length}",
        ErrorCodes.VALIDATION_CONSTRAINT_VIOLATED: "约束条件违反：{constraint_name}",
        
        # 认证错误消息
        ErrorCodes.AUTH_TOKEN_INVALID: "认证令牌无效：{token_type}",
        ErrorCodes.AUTH_TOKEN_EXPIRED: "认证令牌已过期，过期时间：{expired_at}",
        ErrorCodes.AUTH_CREDENTIALS_INVALID: "认证凭据无效：{credential_type}",
        ErrorCodes.AUTH_USER_NOT_FOUND: "用户不存在：{user_identifier}",
        ErrorCodes.AUTH_SESSION_EXPIRED: "会话已过期，过期时间：{expired_at}",
        
        # 授权错误消息
        ErrorCodes.AUTHZ_ACCESS_DENIED: "访问被拒绝：{resource}",
        ErrorCodes.AUTHZ_INSUFFICIENT_PERMISSIONS: "权限不足，需要权限：{required_permissions}",
        ErrorCodes.AUTHZ_RESOURCE_FORBIDDEN: "资源访问被禁止：{resource_id}",
        ErrorCodes.AUTHZ_OPERATION_NOT_ALLOWED: "操作不被允许：{operation}",
        
        # 资源错误消息
        ErrorCodes.RESOURCE_NOT_FOUND: "资源未找到：{resource_type} {resource_id}",
        ErrorCodes.RESOURCE_ALREADY_EXISTS: "资源已存在：{resource_type} {resource_id}",
        ErrorCodes.RESOURCE_IN_USE: "资源正在使用中：{resource_type} {resource_id}",
        ErrorCodes.RESOURCE_QUOTA_EXCEEDED: "资源配额已超限：{resource_type}，当前：{current_usage}，限制：{quota_limit}",
        ErrorCodes.RESOURCE_LOCKED: "资源已被锁定：{resource_type} {resource_id}，锁定者：{locker}",
        
        # 网络错误消息
        ErrorCodes.NETWORK_CONNECTION_FAILED: "网络连接失败：{target_host}:{target_port}",
        ErrorCodes.NETWORK_CONNECTION_TIMEOUT: "网络连接超时：{target_host}，超时时间：{timeout_seconds}秒",
        ErrorCodes.NETWORK_CONNECTION_LOST: "网络连接丢失：{connection_id}",
        ErrorCodes.NETWORK_DNS_RESOLUTION_FAILED: "DNS解析失败：{hostname}",
        ErrorCodes.NETWORK_SERVICE_UNAVAILABLE: "服务不可用：{service_name}",
        
        # 超时错误消息
        ErrorCodes.TIMEOUT_REQUEST_TIMEOUT: "请求超时：{request_id}，超时时间：{timeout_seconds}秒",
        ErrorCodes.TIMEOUT_OPERATION_TIMEOUT: "操作超时：{operation}，超时时间：{timeout_seconds}秒",
        ErrorCodes.TIMEOUT_DEADLINE_EXCEEDED: "截止时间已过：{deadline}",
        ErrorCodes.TIMEOUT_LOCK_TIMEOUT: "锁获取超时：{lock_name}，超时时间：{timeout_seconds}秒",
        
        # 限流错误消息
        ErrorCodes.RATE_LIMIT_EXCEEDED: "请求频率超限：{current_rate}/{rate_limit}，重置时间：{reset_time}",
        ErrorCodes.RATE_LIMIT_QUOTA_EXCEEDED: "配额已用完：{current_usage}/{quota_limit}",
        ErrorCodes.RATE_LIMIT_CONCURRENT_LIMIT: "并发限制：当前并发：{current_concurrent}，限制：{max_concurrent}",
        
        # 业务逻辑错误消息
        ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE: "智能体不可用：{agent_id}，状态：{status}",
        ErrorCodes.BUSINESS_TASK_FAILED: "任务执行失败：{task_id}，失败原因：{failure_reason}",
        ErrorCodes.BUSINESS_INVALID_STATE: "状态无效：{current_state}，期望状态：{expected_states}",
        ErrorCodes.BUSINESS_WORKFLOW_ERROR: "工作流错误：{workflow_id}，步骤：{step_name}，错误：{error_details}",
        ErrorCodes.BUSINESS_POOL_EXHAUSTED: "资源池已耗尽：{pool_id}，可用数量：{available_count}"
    }
    
    @classmethod
    def get_message(cls, error_code: str, **kwargs) -> str:
        """获取错误消息"""
        template = cls.MESSAGES.get(error_code, "未知错误：{error_code}")
        try:
            return template.format(error_code=error_code, **kwargs)
        except KeyError as e:
            return f"错误消息模板缺少参数：{e}，错误代码：{error_code}"


class ErrorBuilder:
    """错误构建器"""
    
    def __init__(self):
        self._code: Optional[str] = None
        self._message: Optional[str] = None
        self._category: Optional[ErrorCategory] = None
        self._severity: Optional[ErrorSeverity] = None
        self._context: Optional[ErrorContext] = None
        self._details: Dict[str, Any] = {}
        self._suggestions: List[str] = []
        self._related_errors: List[str] = []
        self._include_stacktrace: bool = False
        self._inner_exception: Optional[Exception] = None
    
    def code(self, error_code: str) -> 'ErrorBuilder':
        """设置错误代码"""
        self._code = error_code
        return self
    
    def message(self, message: str, **kwargs) -> 'ErrorBuilder':
        """设置错误消息"""
        if self._code and not message:
            self._message = ErrorMessages.get_message(self._code, **kwargs)
        else:
            self._message = message.format(**kwargs) if kwargs else message
        return self
    
    def category(self, category: ErrorCategory) -> 'ErrorBuilder':
        """设置错误分类"""
        self._category = category
        return self
    
    def severity(self, severity: ErrorSeverity) -> 'ErrorBuilder':
        """设置错误严重程度"""
        self._severity = severity
        return self
    
    def context(self, context: ErrorContext) -> 'ErrorBuilder':
        """设置错误上下文"""
        self._context = context
        return self
    
    def detail(self, key: str, value: Any) -> 'ErrorBuilder':
        """添加详细信息"""
        self._details[key] = value
        return self
    
    def details(self, details: Dict[str, Any]) -> 'ErrorBuilder':
        """设置详细信息"""
        self._details.update(details)
        return self
    
    def suggestion(self, suggestion: str) -> 'ErrorBuilder':
        """添加解决建议"""
        self._suggestions.append(suggestion)
        return self
    
    def suggestions(self, suggestions: List[str]) -> 'ErrorBuilder':
        """设置解决建议"""
        self._suggestions.extend(suggestions)
        return self
    
    def related_error(self, error_code: str) -> 'ErrorBuilder':
        """添加相关错误"""
        self._related_errors.append(error_code)
        return self
    
    def with_stacktrace(self, include: bool = True) -> 'ErrorBuilder':
        """包含堆栈跟踪"""
        self._include_stacktrace = include
        return self
    
    def inner_exception(self, exception: Exception) -> 'ErrorBuilder':
        """设置内部异常"""
        self._inner_exception = exception
        return self
    
    def build(self) -> StructuredError:
        """构建结构化错误"""
        # 自动推断错误分类和严重程度
        if self._category is None and self._code:
            self._category = self._infer_category(self._code)
        
        if self._severity is None and self._code:
            self._severity = self._infer_severity(self._code)
        
        # 如果没有提供消息，根据错误代码生成
        if self._message is None and self._code:
            self._message = ErrorMessages.get_message(self._code, **self._details)
        
        # 创建默认上下文
        if self._context is None:
            self._context = ErrorContext()
        
        # 获取堆栈跟踪
        stacktrace = None
        if self._include_stacktrace:
            stacktrace = traceback.format_exc()
        
        # 处理内部异常
        inner_exception = None
        if self._inner_exception:
            inner_exception = str(self._inner_exception)
        
        return StructuredError(
            code=self._code or "UNKNOWN",
            message=self._message or "未知错误",
            category=self._category or ErrorCategory.SYSTEM,
            severity=self._severity or ErrorSeverity.MEDIUM,
            context=self._context,
            details=self._details,
            suggestions=self._suggestions,
            related_errors=self._related_errors,
            stacktrace=stacktrace,
            inner_exception=inner_exception
        )
    
    def _infer_category(self, error_code: str) -> ErrorCategory:
        """根据错误代码推断错误分类"""
        prefix = error_code.split('-')[0] if '-' in error_code else error_code[:3]
        
        category_mapping = {
            'SYS': ErrorCategory.SYSTEM,
            'CFG': ErrorCategory.CONFIGURATION,
            'VAL': ErrorCategory.VALIDATION,
            'AUTH': ErrorCategory.AUTHENTICATION,
            'AUTHZ': ErrorCategory.AUTHORIZATION,
            'RES': ErrorCategory.RESOURCE,
            'NET': ErrorCategory.NETWORK,
            'TMO': ErrorCategory.TIMEOUT,
            'RATE': ErrorCategory.RATE_LIMIT,
            'BIZ': ErrorCategory.BUSINESS_LOGIC
        }
        
        return category_mapping.get(prefix, ErrorCategory.SYSTEM)
    
    def _infer_severity(self, error_code: str) -> ErrorSeverity:
        """根据错误代码推断错误严重程度"""
        # 根据错误类型推断严重程度的默认规则
        category = self._infer_category(error_code)
        
        severity_mapping = {
            ErrorCategory.SYSTEM: ErrorSeverity.HIGH,
            ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
            ErrorCategory.VALIDATION: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorCategory.AUTHORIZATION: ErrorSeverity.HIGH,
            ErrorCategory.RESOURCE: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.LOW,
            ErrorCategory.BUSINESS_LOGIC: ErrorSeverity.MEDIUM
        }
        
        return severity_mapping.get(category, ErrorSeverity.MEDIUM)


class StructuredException(Exception):
    """结构化异常"""
    
    def __init__(self, structured_error: StructuredError):
        self.structured_error = structured_error
        super().__init__(structured_error.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.structured_error.to_dict()
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return self.structured_error.to_json()


class ErrorFactory:
    """错误工厂"""
    
    @staticmethod
    def create_system_error(message: str, details: Optional[Dict[str, Any]] = None, 
                          context: Optional[ErrorContext] = None) -> StructuredException:
        """创建系统错误"""
        error = (ErrorBuilder()
                .code(ErrorCodes.SYSTEM_INTERNAL_ERROR)
                .message(message)
                .category(ErrorCategory.SYSTEM)
                .severity(ErrorSeverity.HIGH)
                .context(context or ErrorContext())
                .details(details or {})
                .with_stacktrace(True)
                .build())
        
        return StructuredException(error)
    
    @staticmethod
    def create_validation_error(field_name: str, value: Any, 
                              expected_format: Optional[str] = None,
                              context: Optional[ErrorContext] = None) -> StructuredException:
        """创建验证错误"""
        details = {"field_name": field_name, "value": value}
        if expected_format:
            details["expected_format"] = expected_format
        
        error = (ErrorBuilder()
                .code(ErrorCodes.VALIDATION_INPUT_INVALID)
                .category(ErrorCategory.VALIDATION)
                .severity(ErrorSeverity.MEDIUM)
                .context(context or ErrorContext())
                .details(details)
                .suggestion("请检查输入格式是否正确")
                .build())
        
        return StructuredException(error)
    
    @staticmethod
    def create_resource_not_found_error(resource_type: str, resource_id: str,
                                       context: Optional[ErrorContext] = None) -> StructuredException:
        """创建资源未找到错误"""
        error = (ErrorBuilder()
                .code(ErrorCodes.RESOURCE_NOT_FOUND)
                .category(ErrorCategory.RESOURCE)
                .severity(ErrorSeverity.MEDIUM)
                .context(context or ErrorContext())
                .details({"resource_type": resource_type, "resource_id": resource_id})
                .suggestion(f"请确认{resource_type}标识符是否正确")
                .suggestion("请检查资源是否已被删除或迁移")
                .build())
        
        return StructuredException(error)
    
    @staticmethod
    def create_rate_limit_error(current_rate: int, rate_limit: int, reset_time: str,
                              context: Optional[ErrorContext] = None) -> StructuredException:
        """创建限流错误"""
        error = (ErrorBuilder()
                .code(ErrorCodes.RATE_LIMIT_EXCEEDED)
                .category(ErrorCategory.RATE_LIMIT)
                .severity(ErrorSeverity.LOW)
                .context(context or ErrorContext())
                .details({
                    "current_rate": current_rate, 
                    "rate_limit": rate_limit, 
                    "reset_time": reset_time
                })
                .suggestion(f"请在{reset_time}后重试")
                .suggestion("考虑降低请求频率或联系管理员提升限制")
                .build())
        
        return StructuredException(error)


# 全局错误处理器
def handle_structured_error(func):
    """结构化错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StructuredException:
            # 重新抛出结构化异常
            raise
        except Exception as e:
            # 将普通异常转换为结构化异常
            context = ErrorContext()
            if hasattr(e, '__self__'):
                context.component = e.__self__.__class__.__name__
            
            structured_error = (ErrorBuilder()
                              .code(ErrorCodes.SYSTEM_INTERNAL_ERROR)
                              .message(str(e))
                              .category(ErrorCategory.SYSTEM)
                              .severity(ErrorSeverity.HIGH)
                              .context(context)
                              .inner_exception(e)
                              .with_stacktrace(True)
                              .build())
            
            logger.error("捕获未处理异常", structured_error=structured_error.to_dict())
            raise StructuredException(structured_error)
    
    return wrapper