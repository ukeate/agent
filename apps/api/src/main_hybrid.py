"""
AI Agent System - 混合FastAPI应用
包含完整的API路由但跳过需要数据库的服务初始化
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
from src.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理 - 简化版本"""
    # 设置结构化日志
    setup_logging()
    logger = structlog.get_logger(__name__)

    # 启动时执行
    logger.info("AI Agent System API starting up (hybrid mode)", stage="startup")

    # 跳过数据库和Redis初始化，直接标记为成功
    logger.info("All services initialized successfully (hybrid mode)")

    yield

    # 关闭时执行
    logger.info("AI Agent System API shutting down (hybrid mode)", stage="shutdown")
    logger.info("All services closed successfully (hybrid mode)")


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
        title="AI Agent System API (Hybrid)",
        description="""
## 🚀 AI智能体系统平台 API (混合版本)

基于多智能体架构的企业级AI开发平台，集成了完整的API路由但跳过数据库依赖。

### 核心功能模块

#### 🤖 智能体管理
- **单智能体**: ReAct智能体实现，支持工具调用和推理
- **多智能体协作**: AutoGen框架支持的多智能体对话系统
- **工作流编排**: LangGraph状态机驱动的工作流引擎
- **监督者模式**: 智能任务分配和执行监控

#### 🔄 分布式任务协调
- **Raft共识算法**: 分布式任务状态同步
- **智能任务分解**: 8种分解策略支持
- **智能任务分配**: 8种分配策略支持
- **冲突解决**: 自动冲突检测和解决

#### 📊 流式处理系统
- **SSE流式响应**: 服务器推送事件
- **WebSocket支持**: 双向实时通信
- **背压控制**: 自动流量控制
- **队列监控**: 实时队列状态监控

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
        """,
        version="0.1.0-hybrid",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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

    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return JSONResponse(
            content={
                "status": "healthy",
                "service": "ai-agent-api-hybrid",
                "version": "0.1.0-hybrid",
                "services": {"database": "bypassed", "redis": "bypassed"},
                "timestamp": None,
            }
        )

    # 根路径
    @app.get("/")
    async def root():
        """根路径端点"""
        return JSONResponse(
            content={
                "message": "AI Agent System API (Hybrid)",
                "version": "0.1.0-hybrid",
                "docs": "/docs",
                "note": "This is a hybrid version with complete API routes but simplified service initialization",
            }
        )

    # 注册完整的API路由
    from src.api import v1_router
    app.include_router(v1_router)

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main_hybrid:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )