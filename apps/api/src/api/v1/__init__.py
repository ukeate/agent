"""
API v1路由模块
"""

from fastapi import APIRouter

from .test import router as test_router
from .mcp import router as mcp_router
from .agents import router as agents_router
from .agent_interface import router as agent_interface_router
from .multi_agents import router as multi_agents_router
from .workflows import router as workflows_router
from .supervisor import router as supervisor_router
from .rag import router as rag_router

# 创建v1 API路由器
v1_router = APIRouter(prefix="/api/v1")

# 注册子路由
v1_router.include_router(test_router)
v1_router.include_router(mcp_router)
v1_router.include_router(agents_router)
v1_router.include_router(agent_interface_router)
v1_router.include_router(multi_agents_router)
v1_router.include_router(workflows_router)
v1_router.include_router(supervisor_router)
v1_router.include_router(rag_router)

__all__ = ["v1_router"]
