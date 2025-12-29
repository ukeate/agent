"""
API v1核心路由模块 - 仅包含基础功能
"""

from fastapi import APIRouter

# 仅导入核心和基础的路由，避免复杂依赖
try:
    from src.api.v1.auth import router as auth_router
except ImportError:
    auth_router = APIRouter()

try:
    from src.api.v1.rag import router as rag_router
except ImportError:
    rag_router = APIRouter()

try:
    from src.api.v1.workflows import router as workflows_router
except ImportError:
    workflows_router = APIRouter()

try:
    from src.api.v1.supervisor import router as supervisor_router
except ImportError:
    supervisor_router = APIRouter()

try:
    from src.api.v1.events import router as events_router
except ImportError:
    events_router = APIRouter()

try:
    from src.api.v1.streaming import router as streaming_router
except ImportError:
    streaming_router = APIRouter()

try:
    from src.api.v1.multi_agents import router as multi_agents_router
except ImportError:
    multi_agents_router = APIRouter()

# 创建核心v1 API路由器
core_v1_router = APIRouter(prefix="/api/v1")

# 注册核心路由
core_v1_router.include_router(auth_router)
core_v1_router.include_router(rag_router)
core_v1_router.include_router(workflows_router)
core_v1_router.include_router(supervisor_router)
core_v1_router.include_router(events_router)
core_v1_router.include_router(streaming_router)
core_v1_router.include_router(multi_agents_router)

__all__ = ["core_v1_router"]
