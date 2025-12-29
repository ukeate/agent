"""
安全中间件系统
"""

import time
import uuid
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send
from src.core.config import get_settings
from src.core.security.audit import AuditLogger
from src.core.security.monitoring import SecurityMonitor

from src.core.logging import get_logger
logger = get_logger(__name__)

# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

class SecurityMiddleware:
    """统一的安全中间件"""
    
    def __init__(self, app, **kwargs):
        self.app = app
        self.settings = get_settings()
        self.security_monitor = SecurityMonitor()
        self.audit_logger = AuditLogger()
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """处理请求"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_id = scope.get("state", {}).get("request_id") or str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        
        # 记录请求开始时间
        start_time = time.perf_counter()

        response_status = None
        response_headers = None
        
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
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Request blocked by security policy",
                        "request_id": request_id,
                        "risk_score": security_check.risk_score
                    }
                )
                response_status = response.status_code
                response_headers = list(response.headers.raw)
                await response(scope, receive, send)
                return
            
            async def send_wrapper(message):
                nonlocal response_status, response_headers
                if message["type"] == "http.response.start":
                    response_status = message["status"]
                    response_headers = message.get("headers", [])
                await send(message)

            await self.app(scope, receive, send_wrapper)
            
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
        finally:
            process_time = time.perf_counter() - start_time
            status_code = response_status if response_status is not None else 500
            response = JSONResponse(status_code=status_code, content={"status": "recorded"})
            if response_headers:
                for raw_key, raw_value in response_headers:
                    response.headers[raw_key.decode()] = raw_value.decode()
            await self.audit_logger.log_api_call(
                request=request,
                response=response,
                process_time=process_time,
                request_id=request_id,
            )

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

class SecureHeadersMiddleware:
    """安全头中间件"""
    
    def __init__(self, app, **kwargs):
        self.app = app
        self.settings = get_settings()
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """添加安全响应头"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
                headers["X-Permitted-Cross-Domain-Policies"] = "none"
                if self.settings.FORCE_HTTPS or scope.get("scheme") == "https":
                    headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
                if self.settings.CSP_HEADER:
                    headers["Content-Security-Policy"] = self.settings.CSP_HEADER
            await send(message)

        await self.app(scope, receive, send_wrapper)

def setup_security_middleware(app):
    """设置所有安全中间件"""
    settings = get_settings()
    
    # 1. HTTPS重定向（仅生产环境）
    if not settings.DEBUG and settings.FORCE_HTTPS:
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # 2. 可信主机验证
    if settings.TRUSTED_HOSTS and not settings.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.TRUSTED_HOSTS,
            www_redirect=settings.TRUSTED_HOSTS_WWW_REDIRECT,
        )
    
    # 3. 响应压缩
    app.add_middleware(
        CompressionMiddleware,
        minimum_size=settings.GZIP_MINIMUM_SIZE,
        compresslevel=settings.GZIP_COMPRESS_LEVEL,
    )
    
    # 4. 安全头
    app.add_middleware(SecureHeadersMiddleware)
    
    # 5. 统一安全中间件
    app.add_middleware(SecurityMiddleware)
    
    logger.info("Security middleware initialized successfully")
