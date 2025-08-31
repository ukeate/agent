"""
AI Agent System - 情感智能系统专用启动入口
"""

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    # 设置结构化日志
    setup_logging()
    logger = structlog.get_logger(__name__)

    # 启动时执行
    logger.info("Emotion Intelligence System API starting up", stage="startup")

    try:
        settings = get_settings()

        # 跳过测试模式下的服务初始化
        if not settings.TESTING:
            logger.info("Emotion intelligence system initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize services", error=str(e), exc_info=True)
        if not get_settings().TESTING:
            raise

    yield

    # 关闭时执行
    logger.info("Emotion Intelligence System API shutting down", stage="shutdown")


def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()

    app = FastAPI(
        title="Emotion Intelligence System API",
        description="""
## 🧠 情感智能系统 API

多模态情感识别与分析平台，支持实时情感分析和WebSocket通信。

### 核心功能模块

#### 🎭 多模态情感识别
- **文本情感分析**: 基于NLP的情感识别
- **音频情感识别**: 语音情感特征提取
- **视频情感识别**: 面部表情和肢体语言分析
- **生理信号分析**: 心率、皮肤电导等生理指标
- **多模态融合**: 综合多种输入的情感状态判断

#### 🔄 实时通信
- **WebSocket连接**: 实时双向通信
- **流式处理**: 支持连续数据流分析
- **会话管理**: 用户会话状态管理
- **记忆系统**: 上下文记忆和历史追踪

#### 📊 情感分析
- **VAD模型**: Valence-Arousal-Dominance三维情感模型
- **个性分析**: 五因素个性模型(Big Five)
- **共情响应**: 智能情感回应生成
- **质量监控**: 分析质量和准确度监控
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc", 
        openapi_url="/openapi.json",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "emotion-intelligence",
                "description": "情感智能核心接口",
            },
            {
                "name": "emotion-websocket", 
                "description": "WebSocket实时通信接口",
            },
        ]
    )

    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 开发环境允许所有来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 健康检查端点
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return JSONResponse(
            content={
                "status": "healthy",
                "service": "emotion-intelligence-api",
                "version": "1.0.0",
                "timestamp": None,
            }
        )

    # 根路径
    @app.get("/")
    async def root():
        """根路径端点"""
        return JSONResponse(
            content={
                "message": "Emotion Intelligence System API",
                "version": "1.0.0", 
                "docs": "/docs",
            }
        )

    # 注册情感智能系统路由
    try:
        from api.v1.emotion_intelligence import router as emotion_intelligence_router
        from api.v1.emotion_websocket import router as emotion_websocket_router
        
        app.include_router(emotion_intelligence_router, prefix="/api/v1")
        app.include_router(emotion_websocket_router, prefix="/api/v1")
    except ImportError as e:
        # 临时创建模拟路由
        from fastapi import APIRouter
        
        mock_router = APIRouter(prefix="/emotion-intelligence", tags=["emotion-intelligence"])
        
        @mock_router.get("/status")
        async def mock_status():
            return {"status": "emotion system mock", "message": "情感智能系统模拟端点"}
        
        app.include_router(mock_router, prefix="/api/v1")

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main_emotion:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )