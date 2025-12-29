"""
API v1路由模块（精简版）

说明：
- 这里仅提供一个空的 `v1_router` 占位，避免在导入 `api.v1` 时
  触发大量依赖（如 langchain_community）导致导入失败。
- 实际的各业务路由在 `src.main` 中通过动态模块加载并注册到
  自己定义的 `v1_router` 上，不依赖本文件中的路由注册。
"""

from fastapi import APIRouter

v1_router = APIRouter(prefix="/api/v1")

__all__ = ["v1_router"]
