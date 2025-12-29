"""
结构化日志系统
"""

from typing import Any, Dict, Optional
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.core.logging import get_logger
class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, name: str):
        self.logger = get_logger(name)

    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """错误日志"""
        if exception:
            self.logger.error(message, error=str(exception), **kwargs)
            return
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)

    def audit(self, action: str, user: str, resource: str, result: str, **kwargs):
        """审计日志"""
        self.logger.info(
            "AUDIT",
            audit=True,
            action=action,
            user=user,
            resource=resource,
            result=result,
            **kwargs,
        )

    def performance(self, operation: str, duration: float, **kwargs):
        """性能日志"""
        self.logger.info(
            "PERFORMANCE",
            performance=True,
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs,
        )

    def experiment_event(self, experiment_id: str, event_type: str, **kwargs):
        """实验事件日志"""
        self.logger.info(
            "EXPERIMENT_EVENT",
            experiment_id=experiment_id,
            event_type=event_type,
            **kwargs,
        )

class RequestLogger:
    """请求日志记录器"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        request_id: str,
        user_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """记录HTTP请求"""
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration * 1000,
            "user_id": user_id,
            "experiment_id": experiment_id,
        }

        if error:
            log_data["error"] = error
            self.logger.error(f"Request failed: {method} {path}", **log_data)
        elif status_code >= 500:
            self.logger.error(f"Server error: {method} {path}", **log_data)
        elif status_code >= 400:
            self.logger.warning(f"Client error: {method} {path}", **log_data)
        else:
            self.logger.info(f"Request completed: {method} {path}", **log_data)

    def log_slow_request(self, method: str, path: str, duration: float, threshold: float = 1.0):
        """记录慢请求"""
        if duration > threshold:
            self.logger.warning(
                f"Slow request detected: {method} {path}",
                method=method,
                path=path,
                duration_ms=duration * 1000,
                threshold_ms=threshold * 1000,
            )

class ExperimentLogger:
    """实验日志记录器"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_experiment_created(self, experiment_id: str, name: str, creator: str):
        """记录实验创建"""
        self.logger.experiment_event(
            experiment_id,
            "CREATED",
            name=name,
            creator=creator,
        )

    def log_experiment_started(self, experiment_id: str):
        """记录实验启动"""
        self.logger.experiment_event(experiment_id, "STARTED")

    def log_experiment_stopped(self, experiment_id: str, reason: str):
        """记录实验停止"""
        self.logger.experiment_event(
            experiment_id,
            "STOPPED",
            reason=reason,
        )

    def log_variant_assignment(self, experiment_id: str, user_id: str, variant_id: str):
        """记录变体分配"""
        self.logger.debug(
            f"User {user_id} assigned to variant {variant_id}",
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
        )

    def log_conversion(self, experiment_id: str, user_id: str, variant_id: str, value: float):
        """记录转化"""
        self.logger.info(
            f"Conversion recorded for user {user_id}",
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
            conversion_value=value,
        )

    def log_srm_check(self, experiment_id: str, passed: bool, p_value: float):
        """记录SRM检查"""
        level = "info" if passed else "warning"
        getattr(self.logger, level)(
            f"SRM check {'passed' if passed else 'failed'}",
            experiment_id=experiment_id,
            srm_passed=passed,
            srm_p_value=p_value,
        )

class SecurityLogger:
    """安全日志记录器"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_authentication(self, user: str, success: bool, method: str, ip: str):
        """记录认证"""
        self.logger.audit(
            "AUTHENTICATION",
            user,
            "auth_system",
            "SUCCESS" if success else "FAILURE",
            method=method,
            ip_address=ip,
        )

    def log_authorization(self, user: str, resource: str, action: str, allowed: bool):
        """记录授权"""
        self.logger.audit(
            "AUTHORIZATION",
            user,
            resource,
            "ALLOWED" if allowed else "DENIED",
            action=action,
        )

    def log_data_access(self, user: str, data_type: str, operation: str, records: int):
        """记录数据访问"""
        self.logger.audit(
            "DATA_ACCESS",
            user,
            data_type,
            "SUCCESS",
            operation=operation,
            record_count=records,
        )

    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """记录安全事件"""
        log_method = {
            "LOW": self.logger.info,
            "MEDIUM": self.logger.warning,
            "HIGH": self.logger.error,
            "CRITICAL": self.logger.critical,
        }.get(severity, self.logger.warning)

        log_method(
            f"Security event: {event_type}",
            security_event=True,
            event_type=event_type,
            severity=severity,
            **details,
        )

class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def log_database_query(self, query_type: str, duration: float, rows: int):
        """记录数据库查询"""
        self.logger.performance(
            f"database_{query_type}",
            duration,
            rows_affected=rows,
        )

    def log_cache_operation(self, operation: str, key: str, hit: bool, duration: float):
        """记录缓存操作"""
        self.logger.debug(
            f"Cache {operation}: {'HIT' if hit else 'MISS'}",
            cache_operation=operation,
            cache_key=key,
            cache_hit=hit,
            duration_ms=duration * 1000,
        )

    def log_external_api_call(self, service: str, endpoint: str, duration: float, status: int):
        """记录外部API调用"""
        self.logger.performance(
            f"external_api_{service}",
            duration,
            endpoint=endpoint,
            status_code=status,
        )

    def log_batch_processing(self, batch_type: str, size: int, duration: float, success: bool):
        """记录批处理"""
        self.logger.performance(
            f"batch_{batch_type}",
            duration,
            batch_size=size,
            success=success,
        )

# 创建全局日志实例
app_logger = StructuredLogger("ab_testing_platform")
request_logger = RequestLogger(app_logger)
experiment_logger = ExperimentLogger(app_logger)
security_logger = SecurityLogger(app_logger)
performance_logger = PerformanceLogger(app_logger)

def set_request_context(**kwargs):
    """设置请求上下文"""
    bind_contextvars(**kwargs)

def clear_request_context():
    """清除请求上下文"""
    clear_contextvars()
