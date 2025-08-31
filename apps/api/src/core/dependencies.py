"""
FastAPI依赖注入
"""

from typing import Optional, AsyncGenerator
from fastapi import HTTPException, Header, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from ai.fault_tolerance import FaultToleranceSystem
from services.fault_tolerance_service import FaultToleranceService
from .database import get_db_session
from .config import get_settings

async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> str:
    """获取当前用户，简化版本用于开发"""
    # 在实际应用中，这里应该验证JWT token或其他认证方式
    # 现在简化为返回固定用户ID用于开发测试
    if authorization and authorization.startswith("Bearer "):
        # 简单的用户ID提取，实际应用中需要解析JWT
        return "test_user_123"
    
    # 开发模式下允许无认证访问
    return "anonymous_user"

async def get_api_key(
    x_api_key: Optional[str] = Header(None)
) -> str:
    """获取API密钥"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API密钥缺失"
        )
    
    # 这里应该验证API密钥
    # 简化版本，直接返回
    return x_api_key

# 全局实例（在实际应用中应该使用依赖注入容器）
_fault_tolerance_system: Optional[FaultToleranceSystem] = None
_fault_tolerance_service: Optional[FaultToleranceService] = None

class MockComponent:
    """模拟组件用于开发测试"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

def initialize_fault_tolerance_system(
    cluster_manager=None,
    task_coordinator=None, 
    lifecycle_manager=None,
    metrics_collector=None,
    config=None
):
    """初始化容错系统"""
    global _fault_tolerance_system, _fault_tolerance_service
    
    # 为开发环境提供默认模拟组件
    _fault_tolerance_system = FaultToleranceSystem(
        cluster_manager=cluster_manager or MockComponent(),
        task_coordinator=task_coordinator or MockComponent(),
        lifecycle_manager=lifecycle_manager or MockComponent(),
        metrics_collector=metrics_collector or MockComponent(),
        config=config or {}
    )
    
    _fault_tolerance_service = FaultToleranceService(_fault_tolerance_system)
 
async def get_fault_tolerance_system() -> FaultToleranceSystem:
    """获取容错系统实例"""
    if _fault_tolerance_system is None:
        # 自动初始化容错系统
        initialize_fault_tolerance_system()
    
    if _fault_tolerance_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fault tolerance system not initialized"
        )
    return _fault_tolerance_system
 
async def get_fault_tolerance_service() -> FaultToleranceService:
    """获取容错系统服务实例"""
    if _fault_tolerance_service is None:
        # 自动初始化容错系统
        initialize_fault_tolerance_system()
    
    if _fault_tolerance_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fault tolerance service not initialized"
        )
    return _fault_tolerance_service


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话的依赖注入函数"""
    async with get_db_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """获取Redis连接的依赖注入函数"""
    settings = get_settings()
    redis_client = redis.from_url(settings.REDIS_URL)
    try:
        yield redis_client
    finally:
        await redis_client.close()