"""健康检查API路由"""

from fastapi import APIRouter, Query
from typing import Dict, Any

from src.core.health import get_health_status, check_readiness, check_liveness, HealthStatus
from src.core.monitoring import monitoring_service
from src.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check(detailed: bool = Query(False, description="是否返回详细信息")) -> Dict[str, Any]:
    """
    基础健康检查端点
    
    Args:
        detailed: 是否返回详细的健康信息
    
    Returns:
        健康状态信息
    """
    try:
        health_status = await get_health_status(detailed=detailed)
        
        # 根据健康状态设置响应
        if health_status["status"] == HealthStatus.UNHEALTHY:
            # 不健康状态，但仍返回200以便监控系统可以读取详细信息
            logger.warning(f"Health check returned UNHEALTHY: {health_status}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY,
            "error": str(e)
        }


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    存活性检查端点（用于K8s liveness probe）
    
    简单检查服务是否存活，不检查依赖项
    
    Returns:
        存活状态
    """
    try:
        is_alive = await check_liveness()
        
        if is_alive:
            return {"status": "alive"}
        else:
            # 返回503 Service Unavailable
            return {"status": "dead"}
            
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return {"status": "dead", "error": str(e)}


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    就绪性检查端点（用于K8s readiness probe）
    
    检查服务及其依赖项是否准备好接收流量
    
    Returns:
        就绪状态
    """
    try:
        is_ready = await check_readiness()
        
        if is_ready:
            return {"status": "ready"}
        else:
            # 获取详细信息以了解哪些组件未就绪
            health_status = await get_health_status(detailed=False)
            
            return {
                "status": "not_ready",
                "components": health_status.get("components", {}),
                "failed_components": health_status.get("failed_components", [])
            }
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    获取系统指标
    
    Returns:
        系统性能和资源指标
    """
    try:
        metrics = await monitoring_service.collect_all_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        return {"error": str(e)}


@router.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    """
    获取活动告警
    
    Returns:
        当前活动的系统告警
    """
    try:
        alerts = await monitoring_service.alert_manager.get_active_alerts()
        
        return {
            "total_alerts": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return {"error": str(e)}