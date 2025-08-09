"""
AI Agent System - FastAPI应用主入口
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.exceptions import (
    BaseAPIException,
    ServiceUnavailableError,
    api_exception_handler,
    general_exception_handler,
    http_exception_handler,
)
from .core.config import get_settings
from .core.database import close_database, init_database, test_database_connection
from .core.logging import setup_logging
from .core.redis import close_redis, init_redis, test_redis_connection


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 设置结构化日志
    setup_logging()
    logger = structlog.get_logger(__name__)

    # 启动时执行
    logger.info("AI Agent System API starting up", stage="startup")

    try:
        settings = get_settings()

        # 跳过测试模式下的服务初始化
        if not settings.TESTING:
            # 初始化数据库连接
            await init_database()

            # 初始化Redis连接
            await init_redis()

            # 测试所有连接
            db_ok = await test_database_connection()
            redis_ok = await test_redis_connection()

            if not db_ok:
                logger.error("Database connection failed during startup")
                raise ServiceUnavailableError("database")

            if not redis_ok:
                logger.error("Redis connection failed during startup")
                raise ServiceUnavailableError("redis")
            
            # 启动任务调度器
            from .services.task_scheduler import task_scheduler
            await task_scheduler.start()
            logger.info("Task scheduler started successfully")

            # 初始化RAG服务
            from .services.rag_service import initialize_rag_service
            rag_initialized = await initialize_rag_service()
            if rag_initialized:
                logger.info("RAG service initialized successfully")
            else:
                logger.warning("RAG service initialization failed")

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e), exc_info=True)
        if not get_settings().TESTING:
            raise

    yield

    # 关闭时执行
    logger.info("AI Agent System API shutting down", stage="shutdown")

    try:
        settings = get_settings()

        if not settings.TESTING:
            # 停止任务调度器
            from .services.task_scheduler import task_scheduler
            await task_scheduler.stop()
            logger.info("Task scheduler stopped successfully")
            
            # 关闭数据库连接
            await close_database()

            # 关闭Redis连接
            await close_redis()

        logger.info("All services closed successfully")

    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)


async def request_logging_middleware(request: Request, call_next):
    """请求日志中间件 - 添加请求ID和结构化日志"""
    request_id = str(uuid.uuid4())
    logger = structlog.get_logger(__name__)

    # 将请求ID添加到请求状态
    request.state.request_id = request_id

    # 记录请求开始
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)

        # 记录请求完成
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
        )

        # 添加请求ID到响应头
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as exc:
        # 记录请求错误
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        raise


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()

    app = FastAPI(
        title="AI Agent System API",
        description="基于多智能体架构的企业级AI开发平台",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # CORS中间件（必须最先添加）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 异常处理器
    app.add_exception_handler(BaseAPIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 请求日志中间件（在CORS之后添加）
    app.middleware("http")(request_logging_middleware)

    # Story 1.5: 添加性能监控和频率限制中间件（但要避免CORS冲突）
    # 注意：这些中间件必须在CORS之后添加
    from .api.middleware import ErrorHandlingMiddleware
    
    # 只添加不会与CORS冲突的中间件
    app.add_middleware(ErrorHandlingMiddleware)
    # 暂时不添加性能监控中间件，避免重复处理请求

    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        if settings.TESTING:
            # 测试模式下直接返回健康状态
            return JSONResponse(
                content={
                    "status": "healthy",
                    "service": "ai-agent-api",
                    "version": "0.1.0",
                    "services": {"database": "healthy", "redis": "healthy"},
                    "timestamp": None,
                }
            )

        from .core.database import test_database_connection
        from .core.redis import test_redis_connection

        # 测试服务连接状态
        db_status = await test_database_connection()
        redis_status = await test_redis_connection()

        overall_status = "healthy" if db_status and redis_status else "degraded"

        return JSONResponse(
            content={
                "status": overall_status,
                "service": "ai-agent-api",
                "version": "0.1.0",
                "services": {
                    "database": "healthy" if db_status else "unhealthy",
                    "redis": "healthy" if redis_status else "unhealthy",
                },
                "timestamp": None,  # Will be added by JSON renderer
            }
        )

    # 根路径
    @app.get("/")
    async def root():
        """根路径端点"""
        return JSONResponse(
            content={
                "message": "AI Agent System API",
                "version": "0.1.0",
                "docs": "/docs" if settings.DEBUG else "Docs disabled in production",
            }
        )

    # 注册API路由
    from .api import v1_router

    app.include_router(v1_router)

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )
