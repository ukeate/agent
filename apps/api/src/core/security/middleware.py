"""
安全中间件系统
"""

import time
import uuid
from typing import Any, Dict, Optional

import structlog
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src.core.config import get_settings
from src.core.security.audit import AuditLogger
from src.core.security.monitoring import SecurityMonitor

logger = structlog.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """统一的安全中间件"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.settings = get_settings()
        self.security_monitor = SecurityMonitor()
        self.audit_logger = AuditLogger()
        
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 安全检查：异常请求检测
            security_check = await self.security_monitor.assess_request(request)
            
            if security_check.risk_score > self.settings.SECURITY_THRESHOLD:
                # 记录安全事件
                await self.audit_logger.log_security_event(
                    event_type="blocked_request",
                    request=request,
                    risk_score=security_check.risk_score,
                    details=security_check.details
                )
                
                # 返回阻断响应
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Request blocked by security policy",
                        "request_id": request_id,
                        "risk_score": security_check.risk_score
                    }
                )
            
            # 添加安全头
            response = await call_next(request)
            
            # 设置安全响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = self.settings.CSP_HEADER
            
            # 记录请求完成
            process_time = time.time() - start_time
            
            # 审计日志
            await self.audit_logger.log_api_call(
                request=request,
                response=response,
                process_time=process_time,
                request_id=request_id
            )
            
            # 添加处理时间头
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 记录错误
            logger.error(
                "Security middleware error",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            
            # 审计错误事件
            await self.audit_logger.log_security_event(
                event_type="middleware_error",
                request=request,
                details={"error": str(e)}
            )
            
            raise


class RateLimitMiddleware:
    """频率限制中间件"""
    
    def __init__(self):
        self.settings = get_settings()
        self.limiter = Limiter(
            key_func=self._get_rate_limit_key,
            default_limits=[self.settings.DEFAULT_RATE_LIMIT],
            storage_uri=self.settings.REDIS_URL,
            strategy="fixed-window"
        )
        
    def _get_rate_limit_key(self, request: Request) -> str:
        """获取频率限制的键"""
        # 优先使用认证用户ID，否则使用IP地址
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        return get_remote_address(request)
    
    def get_limiter(self):
        """获取限制器实例"""
        return self.limiter


class CompressionMiddleware(GZipMiddleware):
    """响应压缩中间件"""
    
    def __init__(self, app, minimum_size: int = 1000, compresslevel: int = 6):
        """
        初始化压缩中间件
        
        Args:
            minimum_size: 最小压缩大小（字节）
            compresslevel: 压缩级别 (1-9)
        """
        super().__init__(app, minimum_size=minimum_size, compresslevel=compresslevel)


class SecureHeadersMiddleware(BaseHTTPMiddleware):
    """安全头中间件"""
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        self.settings = get_settings()
        
    async def dispatch(self, request: Request, call_next):
        """添加安全响应头"""
        response = await call_next(request)
        
        # 添加安全头
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # 在生产环境添加HSTS
        if not self.settings.DEBUG:
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # 添加CSP头（如果配置了）
        if hasattr(self.settings, "CSP_HEADER") and self.settings.CSP_HEADER:
            security_headers["Content-Security-Policy"] = self.settings.CSP_HEADER
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


def setup_security_middleware(app):
    """设置所有安全中间件"""
    settings = get_settings()
    
    # 1. HTTPS重定向（仅生产环境）
    if not settings.DEBUG and settings.FORCE_HTTPS:
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # 2. 可信主机验证
    trusted_hosts = ["localhost", "127.0.0.1", "localhost:8000", "127.0.0.1:8000"]
    if not settings.DEBUG:
        # 在生产环境中使用更严格的主机验证
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
    
    # 3. 响应压缩
    app.add_middleware(
        CompressionMiddleware,
        minimum_size=1000,
        compresslevel=6
    )
    
    # 4. 安全头
    app.add_middleware(SecureHeadersMiddleware)
    
    # 5. 频率限制
    rate_limiter = RateLimitMiddleware()
    limiter = rate_limiter.get_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # 6. 统一安全中间件
    app.add_middleware(SecurityMiddleware)
    
    logger.info("Security middleware initialized successfully")