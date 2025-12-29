import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from src.core.config import get_settings
from src.core.security.middleware import SecureHeadersMiddleware
from src.core.utils.timezone_utils import utc_now
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
AI Agent System - FastAPI应用主入口
基于simple_main.py的工作版本，逐步添加核心功能
"""

ENV_DEFAULTS = {
    # 完全禁用TensorFlow
    'DISABLE_TENSORFLOW': '1',
    'NO_TENSORFLOW': '1',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'CUDA_VISIBLE_DEVICES': '',
    # 禁用Python字节码生成
    'PYTHONDONTWRITEBYTECODE': '1',
    # 强制UTF-8编码
    'PYTHONIOENCODING': 'utf-8',
    # 禁用tokenizers并行
    'TOKENIZERS_PARALLELISM': 'false',
    # 禁用所有数学库多线程
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'NUMEXPR_MAX_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'GOTO_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    # 解决KMP重复库问题
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    # 禁用Intel MKL
    'MKL_THREADING_LAYER': 'sequential',
    'MKL_SERVICE_FORCE_INTEL': '1',
    # 禁用HuggingFace离线模式
    'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
}

for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

setup_logging()
settings = get_settings()

# 核心API模块配置
CORE_API_MODULES = [
    # 健康检查和系统监控
    ("health", "api.v1.health", "健康检查模块"),
    ("monitoring", "api.v1.monitoring", "监控模块"),
    
    # 核心多智能体功能
    ("multi_agent", "api.v1.multi_agent", "多智能体模块"),
    ("agent_interface", "api.v1.agent_interface", "智能体接口模块"),
    
    # 批处理和实验
    ("batch", "api.v1.batch", "批处理模块"),
    ("experiments", "api.v1.experiments", "实验模块"),
    
    # 流式处理
    ("streaming", "api.v1.streaming", "流式处理模块"),
    
    # 分析和报告
    ("analytics", "api.v1.analytics", "分析模块"),
    ("report_generation", "api.v1.report_generation", "报告生成模块"),
    
    # 核心服务
    ("workflow", "api.v1.workflow", "工作流模块"),
    ("security", "api.v1.security", "安全模块"),
]

# 全局服务实例占位（必须由真实实现注入）
health_service = None

# 创建FastAPI应用实例
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    logger.info("🚀 AI Agent System 启动中...")
    
    # 启动时初始化
    startup_info = {
        "app_name": "AI Agent System",
        "version": "1.0.0", 
        "environment": os.getenv("ENVIRONMENT", "development"),
        "loaded_modules": len(CORE_API_MODULES)
    }
    
    logger.info("应用启动完成", **startup_info)
    
    yield
    
    # 关闭时清理
    logger.info("🛑 AI Agent System 正在关闭...")

# 创建FastAPI应用
app = FastAPI(
    title="AI Agent System",
    description="个人AI智能体学习平台 - 核心版本",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=settings.CORS_EXPOSE_HEADERS,
)

if settings.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

if settings.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS,
        www_redirect=settings.TRUSTED_HOSTS_WWW_REDIRECT,
    )

app.add_middleware(
    GZipMiddleware,
    minimum_size=settings.GZIP_MINIMUM_SIZE,
    compresslevel=settings.GZIP_COMPRESS_LEVEL,
)
app.add_middleware(SecureHeadersMiddleware)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求"""
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    process_time = time.perf_counter() - start_time
    
    logger.info(
        "HTTP请求处理完成",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(
        "未处理的异常",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "message": "系统遇到了一个问题，请稍后重试"
        }
    )

# 加载核心API模块
def load_api_modules():
    """安全加载API模块"""
    loaded_modules = []
    failed_modules = []
    
    for module_name, module_path, description in CORE_API_MODULES:
        try:
            # 动态导入模块
            module = __import__(f"src.{module_path}", fromlist=["router"])
            
            if hasattr(module, "router"):
                # 添加路由到应用
                app.include_router(
                    module.router,
                    prefix=f"/api/v1",
                    tags=[module_name]
                )
                loaded_modules.append({
                    "name": module_name,
                    "path": module_path,
                    "description": description
                })
                logger.info(f"✅ 模块加载成功: {module_name}")
            else:
                logger.warning(f"⚠️ 模块没有router属性: {module_name}")
                failed_modules.append({
                    "name": module_name,
                    "error": "没有router属性"
                })
                
        except ImportError as e:
            logger.warning(f"❌ 模块导入失败: {module_name} - {e}")
            failed_modules.append({
                "name": module_name,
                "error": str(e)
            })
        except Exception as e:
            logger.error(f"💥 模块加载异常: {module_name} - {e}")
            failed_modules.append({
                "name": module_name,
                "error": str(e)
            })
    
    logger.info(
        "模块加载完成",
        loaded_count=len(loaded_modules),
        failed_count=len(failed_modules),
        total_modules=len(CORE_API_MODULES)
    )
    
    return loaded_modules, failed_modules

# 加载API模块
loaded_modules, failed_modules = load_api_modules()

# 根路径
@app.get("/")
async def root():
    """根路径欢迎信息"""
    return {
        "message": "欢迎使用 AI Agent System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "loaded_modules": len(loaded_modules),
        "failed_modules": len(failed_modules),
        "timestamp": utc_now().isoformat()
    }

# 基本健康检查
@app.get("/health")
async def health_check():
    """基本健康检查"""
    if not health_service:
        raise HTTPException(status_code=503, detail="Health service not initialized")
    health_info = await health_service.check_system_health()
    return {
        "success": True,
        "data": health_info
    }

# 系统状态
@app.get("/status")
async def system_status():
    """系统状态信息"""
    return {
        "success": True,
        "data": {
            "app_name": "AI Agent System",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "loaded_modules": loaded_modules,
            "failed_modules": failed_modules,
            "uptime": "刚启动",
            "timestamp": utc_now().isoformat()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_working:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
