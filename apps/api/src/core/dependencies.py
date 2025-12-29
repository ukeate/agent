"""
FastAPI依赖注入
"""

import asyncio
from typing import Optional
from fastapi import HTTPException, Header, status, Request
import redis.asyncio as redis
from src.ai.fault_tolerance import FaultToleranceSystem
from src.services.fault_tolerance_service import FaultToleranceService
from .config import get_settings
from .redis import get_redis as get_redis_client
from src.core.security.auth import jwt_manager, is_token_revoked

async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> str:
    """获取当前用户ID（JWT或匿名client_id）"""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            token_data = jwt_manager.decode_token(token)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的访问令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if token_data.token_type != "access" or not token_data.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的访问令牌",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if await is_token_revoked(token_data.jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="访问令牌已失效",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return token_data.user_id

    client_id = getattr(request.state, "client_id", None)
    if client_id:
        return client_id

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少用户标识")

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
_fault_tolerance_lock = asyncio.Lock()

def initialize_fault_tolerance_system(
    cluster_manager=None,
    task_coordinator=None, 
    lifecycle_manager=None,
    metrics_collector=None,
    config=None
):
    """初始化容错系统"""
    global _fault_tolerance_system, _fault_tolerance_service
    
    if not (cluster_manager and task_coordinator and lifecycle_manager and metrics_collector):
        raise RuntimeError("fault tolerance system missing required components")
    _fault_tolerance_system = FaultToleranceSystem(
        cluster_manager=cluster_manager,
        task_coordinator=task_coordinator,
        lifecycle_manager=lifecycle_manager,
        metrics_collector=metrics_collector,
        config=config or {}
    )
    _fault_tolerance_service = FaultToleranceService(_fault_tolerance_system)
 
async def get_fault_tolerance_system() -> FaultToleranceSystem:
    """获取容错系统实例"""
    global _fault_tolerance_system, _fault_tolerance_service
    if _fault_tolerance_system is not None:
        return _fault_tolerance_system
    async with _fault_tolerance_lock:
        if _fault_tolerance_system is not None:
            return _fault_tolerance_system

        settings = get_settings()
        redis_client = get_redis_client()
        if not redis_client:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis not initialized")

        from pathlib import Path
        import socket

        from src.ai.cluster import ClusterStateManager, LifecycleManager, MetricsCollector, AgentInfo, AgentStatus
        from src.ai.distributed_task import DistributedTaskCoordinationEngine
        from src.ai.service_discovery.config import default_config
        from src.ai.service_discovery.core import AgentServiceDiscoverySystem
        from src.services.fault_tolerance_storage_backend import FaultToleranceStorageBackend

        service_discovery = AgentServiceDiscoverySystem(
            redis_client=redis_client,
            prefix=default_config.redis_prefix,
            ttl_seconds=default_config.agent_ttl_seconds,
        )
        await service_discovery.initialize()

        node_id = socket.gethostname()
        task_coordinator = DistributedTaskCoordinationEngine(
            node_id=node_id,
            cluster_nodes=[node_id],
            service_registry=service_discovery.registry,
            load_balancer=service_discovery.load_balancer,
        )
        await task_coordinator.start()

        cluster_manager = ClusterStateManager(cluster_id="fault-tolerance")
        await cluster_manager.start()

        api_agent = AgentInfo(agent_id="api", name="api", host="localhost", port=settings.PORT)
        api_agent.update_status(AgentStatus.RUNNING)
        await cluster_manager.register_agent(api_agent)

        lifecycle_manager = LifecycleManager(cluster_manager)
        metrics_collector = MetricsCollector(cluster_manager)

        storage_backend = FaultToleranceStorageBackend(
            redis_client=redis_client,
            cluster_manager=cluster_manager,
            task_coordinator=task_coordinator,
        )

        backup_dir = str(Path(settings.OFFLINE_STORAGE_PATH) / "fault_tolerance_backups")
        config = {
            "backup": {"backup_location": backup_dir},
            "consistency": {},
            "fault_detection": {},
            "recovery": {},
        }

        _fault_tolerance_system = FaultToleranceSystem(
            cluster_manager=cluster_manager,
            task_coordinator=task_coordinator,
            lifecycle_manager=lifecycle_manager,
            metrics_collector=metrics_collector,
            config=config,
        )
        _fault_tolerance_system.backup_manager.storage_backend = storage_backend
        _fault_tolerance_system.consistency_manager.storage_backend = storage_backend
        await _fault_tolerance_system.start()
        _fault_tolerance_service = FaultToleranceService(_fault_tolerance_system)
    return _fault_tolerance_system
 
async def get_fault_tolerance_service() -> FaultToleranceService:
    """获取容错系统服务实例"""
    system = await get_fault_tolerance_system()
    if _fault_tolerance_service is None:
        _fault_tolerance_service = FaultToleranceService(system)
    return _fault_tolerance_service

def get_redis() -> redis.Redis:
    """获取Redis连接的依赖注入函数"""
    client = get_redis_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis not initialized"
        )
    return client
