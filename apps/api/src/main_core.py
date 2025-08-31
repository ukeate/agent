"""
AI Agent System - 核心版本FastAPI应用主入口
仅包含基础API功能，用于快速修复404问题
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
    logger = structlog.get_logger()
    logger.info("启动AI Agent System Core")
    
    yield
    
    logger.info("关闭AI Agent System Core")


# 创建FastAPI应用
app = FastAPI(
    title="AI Agent System Core",
    description="个人AI智能体学习项目 - 核心版本",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "message": "AI Agent System Core is running"}

@app.get("/")
async def root():
    """根路径"""
    return {"message": "Welcome to AI Agent System Core", "version": "1.0.0"}

# 导入并注册核心API路由
try:
    from src.api.v1.core_init import core_v1_router
    app.include_router(core_v1_router)
except Exception as e:
    logger = structlog.get_logger()
    logger.warning("无法加载核心API路由", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)