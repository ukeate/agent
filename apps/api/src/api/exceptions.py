"""
异常处理模块
"""

from typing import Any

import structlog
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class ApiError(BaseModel):
    """标准API错误响应格式"""

    error: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class BaseAPIException(HTTPException):
    """基础API异常类"""

    def __init__(
        self,
        error: str,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.error = error
        self.message = message
        self.details = details
        super().__init__(status_code=status_code, detail=message)


class ValidationError(BaseAPIException):
    """验证错误"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            error="VALIDATION_ERROR",
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class NotFoundError(BaseAPIException):
    """资源未找到错误"""

    def __init__(self, resource: str, identifier: str = ""):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            error="NOT_FOUND",
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "identifier": identifier},
        )


class ConflictError(BaseAPIException):
    """资源冲突错误"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            error="CONFLICT",
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class UnauthorizedError(BaseAPIException):
    """未授权错误"""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(
            error="UNAUTHORIZED",
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class ForbiddenError(BaseAPIException):
    """权限不足错误"""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(
            error="FORBIDDEN", message=message, status_code=status.HTTP_403_FORBIDDEN
        )


class InternalServerError(BaseAPIException):
    """内部服务器错误"""

    def __init__(
        self,
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            error="INTERNAL_SERVER_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class ServiceUnavailableError(BaseAPIException):
    """服务不可用错误"""

    def __init__(self, service: str):
        super().__init__(
            error="SERVICE_UNAVAILABLE",
            message=f"Service unavailable: {service}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service},
        )


async def api_exception_handler(
    request: Request, exc: BaseAPIException
) -> JSONResponse:
    """API异常处理器"""
    request_id = getattr(request.state, "request_id", None)

    # 记录异常日志
    logger.error(
        "API exception occurred",
        request_id=request_id,
        error=exc.error,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
        url=str(request.url),
        method=request.method,
    )

    # 构造错误响应
    error_response = ApiError(
        error=exc.error, message=exc.message, details=exc.details, request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", None)

    # 记录异常日志
    logger.error(
        "HTTP exception occurred",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url),
        method=request.method,
    )

    # 构造错误响应
    error_response = ApiError(
        error="HTTP_ERROR", message=str(exc.detail), request_id=request_id
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    request_id = getattr(request.state, "request_id", None)

    # 记录异常日志
    logger.error(
        "Unhandled exception occurred",
        request_id=request_id,
        error=str(exc),
        exc_info=True,
        url=str(request.url),
        method=request.method,
    )

    # 构造错误响应
    error_response = ApiError(
        error="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True),
    )
