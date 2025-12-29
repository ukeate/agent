"""
错误处理器模块
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from src.core.logging import get_logger
logger = get_logger(__name__)

def add_exception_handlers(app: FastAPI):
    """添加全局异常处理器"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理HTTP异常"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """处理值错误"""
        logger.error(f"ValueError: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "type": "value_error",
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理一般异常"""
        logger.error(f"Unexpected error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": type(exc).__name__,
                "path": request.url.path
            }
        )
