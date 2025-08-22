"""
结构化日志系统
"""
import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger


# 请求上下文
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_dir: str = "logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        json_format: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # 清除现有处理器
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # JSON格式化器
        if json_format:
            formatter = jsonlogger.JsonFormatter(
                fmt='%(timestamp)s %(level)s %(name)s %(message)s',
                rename_fields={'levelname': 'level', 'name': 'logger'}
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（按大小轮转）
        file_handler = RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志文件
        error_handler = RotatingFileHandler(
            log_path / f"{name}.error.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # 按日期轮转的日志（用于审计）
        audit_handler = TimedRotatingFileHandler(
            log_path / f"{name}.audit.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        audit_handler.setFormatter(formatter)
        self.logger.addHandler(audit_handler)
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """添加上下文信息"""
        context = request_context.get()
        extra_with_context = {
            'timestamp': datetime.utcnow().isoformat(),
            **context,
            **extra
        }
        return {'extra': extra_with_context}
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **self._add_context(kwargs))
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **self._add_context(kwargs))
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **self._add_context(kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """错误日志"""
        if exception:
            kwargs['exception'] = str(exception)
            kwargs['traceback'] = traceback.format_exc()
        self.logger.error(message, **self._add_context(kwargs))
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **self._add_context(kwargs))
    
    def audit(self, action: str, user: str, resource: str, result: str, **kwargs):
        """审计日志"""
        audit_data = {
            'audit': True,
            'action': action,
            'user': user,
            'resource': resource,
            'result': result,
            **kwargs
        }
        self.logger.info(f"AUDIT: {action} on {resource} by {user}", **self._add_context(audit_data))
    
    def performance(self, operation: str, duration: float, **kwargs):
        """性能日志"""
        perf_data = {
            'performance': True,
            'operation': operation,
            'duration_ms': duration * 1000,
            **kwargs
        }
        self.logger.info(f"PERFORMANCE: {operation} took {duration:.3f}s", **self._add_context(perf_data))
    
    def experiment_event(self, experiment_id: str, event_type: str, **kwargs):
        """实验事件日志"""
        event_data = {
            'experiment_id': experiment_id,
            'event_type': event_type,
            **kwargs
        }
        self.logger.info(f"EXPERIMENT_EVENT: {event_type} for {experiment_id}", **self._add_context(event_data))


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
        error: Optional[str] = None
    ):
        """记录HTTP请求"""
        log_data = {
            'request_id': request_id,
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration * 1000,
            'user_id': user_id,
            'experiment_id': experiment_id
        }
        
        if error:
            log_data['error'] = error
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
                threshold_ms=threshold * 1000
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
            creator=creator
        )
    
    def log_experiment_started(self, experiment_id: str):
        """记录实验启动"""
        self.logger.experiment_event(experiment_id, "STARTED")
    
    def log_experiment_stopped(self, experiment_id: str, reason: str):
        """记录实验停止"""
        self.logger.experiment_event(
            experiment_id,
            "STOPPED",
            reason=reason
        )
    
    def log_variant_assignment(self, experiment_id: str, user_id: str, variant_id: str):
        """记录变体分配"""
        self.logger.debug(
            f"User {user_id} assigned to variant {variant_id}",
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id
        )
    
    def log_conversion(self, experiment_id: str, user_id: str, variant_id: str, value: float):
        """记录转化"""
        self.logger.info(
            f"Conversion recorded for user {user_id}",
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
            conversion_value=value
        )
    
    def log_srm_check(self, experiment_id: str, passed: bool, p_value: float):
        """记录SRM检查"""
        level = "info" if passed else "warning"
        getattr(self.logger, level)(
            f"SRM check {'passed' if passed else 'failed'}",
            experiment_id=experiment_id,
            srm_passed=passed,
            srm_p_value=p_value
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
            ip_address=ip
        )
    
    def log_authorization(self, user: str, resource: str, action: str, allowed: bool):
        """记录授权"""
        self.logger.audit(
            "AUTHORIZATION",
            user,
            resource,
            "ALLOWED" if allowed else "DENIED",
            action=action
        )
    
    def log_data_access(self, user: str, data_type: str, operation: str, records: int):
        """记录数据访问"""
        self.logger.audit(
            "DATA_ACCESS",
            user,
            data_type,
            "SUCCESS",
            operation=operation,
            record_count=records
        )
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """记录安全事件"""
        log_method = {
            "LOW": self.logger.info,
            "MEDIUM": self.logger.warning,
            "HIGH": self.logger.error,
            "CRITICAL": self.logger.critical
        }.get(severity, self.logger.warning)
        
        log_method(
            f"Security event: {event_type}",
            security_event=True,
            event_type=event_type,
            severity=severity,
            **details
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
            rows_affected=rows
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, duration: float):
        """记录缓存操作"""
        self.logger.debug(
            f"Cache {operation}: {'HIT' if hit else 'MISS'}",
            cache_operation=operation,
            cache_key=key,
            cache_hit=hit,
            duration_ms=duration * 1000
        )
    
    def log_external_api_call(self, service: str, endpoint: str, duration: float, status: int):
        """记录外部API调用"""
        self.logger.performance(
            f"external_api_{service}",
            duration,
            endpoint=endpoint,
            status_code=status
        )
    
    def log_batch_processing(self, batch_type: str, size: int, duration: float, success: bool):
        """记录批处理"""
        self.logger.performance(
            f"batch_{batch_type}",
            duration,
            batch_size=size,
            success=success
        )


# 创建全局日志实例
app_logger = StructuredLogger("ab_testing_platform")
request_logger = RequestLogger(app_logger)
experiment_logger = ExperimentLogger(app_logger)
security_logger = SecurityLogger(app_logger)
performance_logger = PerformanceLogger(app_logger)


def set_request_context(**kwargs):
    """设置请求上下文"""
    current_context = request_context.get()
    updated_context = {**current_context, **kwargs}
    request_context.set(updated_context)


def clear_request_context():
    """清除请求上下文"""
    request_context.set({})


# 日志配置
def configure_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    json_format: bool = True
):
    """配置日志系统"""
    global app_logger, request_logger, experiment_logger, security_logger, performance_logger
    
    app_logger = StructuredLogger(
        "ab_testing_platform",
        level=level,
        log_dir=log_dir,
        json_format=json_format
    )
    
    request_logger = RequestLogger(app_logger)
    experiment_logger = ExperimentLogger(app_logger)
    security_logger = SecurityLogger(app_logger)
    performance_logger = PerformanceLogger(app_logger)