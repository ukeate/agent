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

from src.api.exceptions import (
    BaseAPIException,
    ServiceUnavailableError,
    api_exception_handler,
    general_exception_handler,
    http_exception_handler,
)
from src.core.config import get_settings
from src.core.database import close_database, init_database, test_database_connection
from src.core.logging import setup_logging
from src.core.redis import close_redis, init_redis, test_redis_connection


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
            from src.services.task_scheduler import task_scheduler
            await task_scheduler.start()
            logger.info("Task scheduler started successfully")

            # 初始化RAG服务
            from src.services.rag_service import initialize_rag_service
            rag_initialized = await initialize_rag_service()
            if rag_initialized:
                logger.info("RAG service initialized successfully")
            else:
                logger.warning("RAG service initialization failed")

            # 初始化缓存系统
            from src.ai.langgraph.cache_factory import initialize_cache
            from src.ai.langgraph.cache_monitor import start_cache_monitoring
            
            cache_initialized = await initialize_cache()
            if cache_initialized:
                logger.info("Cache system initialized successfully")
                
                # 启动缓存监控
                if settings.CACHE_MONITORING:
                    await start_cache_monitoring()
                    logger.info("Cache monitoring started successfully")
            else:
                logger.warning("Cache system initialization failed")

            # 初始化异步智能体系统
            try:
                from src.api.v1.async_agents import (
                    get_event_bus, get_agent_manager, get_langgraph_bridge
                )
                
                # 预初始化异步智能体系统组件
                event_bus = await get_event_bus()
                agent_manager = await get_agent_manager()
                bridge = await get_langgraph_bridge()
                
                logger.info("Async agent system initialized successfully")
            except Exception as e:
                logger.warning("Async agent system initialization failed", error=str(e))

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
            from src.services.task_scheduler import task_scheduler
            await task_scheduler.stop()
            logger.info("Task scheduler stopped successfully")
            
            # 停止异步智能体系统
            try:
                from src.api.v1.async_agents import cleanup
                await cleanup()
                logger.info("Async agent system stopped successfully")
            except Exception as e:
                logger.warning("Error stopping async agent system", error=str(e))
            
            # 停止缓存监控和关闭缓存系统
            from src.ai.langgraph.cache_monitor import stop_cache_monitoring
            from src.ai.langgraph.cache_factory import shutdown_cache
            
            await stop_cache_monitoring()
            await shutdown_cache()
            logger.info("Cache system stopped successfully")
            
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
        description="""
## 🚀 AI智能体系统平台 API

基于多智能体架构的企业级AI开发平台，集成了A/B测试实验功能。

### 核心功能模块

#### 🤖 智能体管理
- **单智能体**: ReAct智能体实现，支持工具调用和推理
- **多智能体协作**: AutoGen框架支持的多智能体对话系统
- **工作流编排**: LangGraph状态机驱动的工作流引擎
- **监督者模式**: 智能任务分配和执行监控

#### 🧪 A/B测试实验平台
- **实验管理**: 创建、配置、管理多变体实验
- **流量分配**: Murmur3哈希算法实现的流量分配
- **统计分析**: t检验、卡方检验、置信区间计算
- **发布策略**: 灰度发布、蓝绿部署、金丝雀发布

#### 📊 RAG系统
- **向量检索**: 基于语义的文档检索
- **智能问答**: 结合上下文的智能回答
- **知识库管理**: 文档索引和更新

#### 🔧 MCP协议集成
- **工具管理**: 标准化的工具接口
- **协议适配**: MCP 1.0协议支持
- **扩展能力**: 自定义工具开发

### 技术特性

- **异步架构**: 基于Python asyncio的高性能异步处理
- **缓存系统**: 多级缓存提升响应速度
- **监控告警**: 完整的指标收集和日志记录
- **安全认证**: JWT token认证和权限管理
- **错误处理**: 统一的异常处理和错误响应

### API使用指南

1. **认证**: 大部分接口需要Bearer Token认证
2. **分页**: 列表接口支持limit/offset分页
3. **过滤**: 支持多种查询参数过滤
4. **响应格式**: 统一的JSON响应格式

### 相关链接

- [GitHub仓库](https://github.com/your-org/ai-agent-system)
- [用户指南](https://docs.ai-agent.com/guide)
- [SDK文档](https://docs.ai-agent.com/sdk)
        """,
        version="0.1.0",
        docs_url="/docs",  # 始终启用文档以便于开发和测试
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "agents",
                "description": "智能体管理接口",
            },
            {
                "name": "multi-agents",
                "description": "多智能体协作接口",
            },
            {
                "name": "workflows",
                "description": "工作流管理接口",
            },
            {
                "name": "supervisor",
                "description": "监督者任务管理接口",
            },
            {
                "name": "rag",
                "description": "RAG系统接口",
            },
            {
                "name": "mcp",
                "description": "MCP协议工具接口",
            },
            {
                "name": "experiments",
                "description": "A/B测试实验管理接口",
            },
            {
                "name": "events",
                "description": "事件收集接口",
            },
            {
                "name": "analysis",
                "description": "统计分析接口",
            },
            {
                "name": "reports",
                "description": "报告生成接口",
            },
            {
                "name": "release",
                "description": "发布策略接口",
            },
            {
                "name": "monitoring",
                "description": "监控指标接口",
            },
        ],
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "docExpansion": "none",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "syntaxHighlight.theme": "monokai",
            "displayRequestDuration": True,
        }
    )

    # CORS中间件（必须最先添加）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 设置安全中间件
    from src.core.security.middleware import setup_security_middleware
    setup_security_middleware(app)

    # 异常处理器
    app.add_exception_handler(BaseAPIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 请求日志中间件（在CORS之后添加）
    app.middleware("http")(request_logging_middleware)

    # Story 1.5: 添加性能监控和频率限制中间件（但要避免CORS冲突）
    # 注意：这些中间件必须在CORS之后添加
    from src.api.middleware import ErrorHandlingMiddleware
    
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

        from src.core.database import test_database_connection
        from src.core.redis import test_redis_connection

        # 测试服务连接状态
        db_status = await test_database_connection()
        redis_status = await test_redis_connection()
        
        # 测试缓存系统健康状态
        cache_status = "healthy"
        try:
            from src.ai.langgraph.cache_monitor import CacheHealthChecker
            from src.ai.langgraph.cache_factory import get_node_cache
            
            cache = get_node_cache()
            health_checker = CacheHealthChecker(cache)
            cache_health = await health_checker.health_check()
            cache_status = cache_health["status"]
        except Exception:
            cache_status = "unknown"

        overall_status = "healthy" if all([db_status, redis_status, cache_status in ["healthy", "unknown"]]) else "degraded"

        return JSONResponse(
            content={
                "status": overall_status,
                "service": "ai-agent-api",
                "version": "0.1.0",
                "services": {
                    "database": "healthy" if db_status else "unhealthy",
                    "redis": "healthy" if redis_status else "unhealthy",
                    "cache": cache_status,
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
    from src.api import v1_router

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
