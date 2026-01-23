"""健康检查API路由"""

from datetime import datetime, timezone
from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from src.core.health import get_health_status, check_liveness, HealthStatus
from src.core.utils.timezone_utils import utc_now, parse_iso_string
from src.core.monitoring import get_monitoring_service
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
            "timestamp": utc_now().isoformat(),
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
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "dead"},
            )
            
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "dead", "error": str(e)},
        )

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    就绪性检查端点（用于K8s readiness probe）
    
    检查服务及其依赖项是否准备好接收流量
    
    Returns:
        就绪状态
    """
    try:
        health_status = await get_health_status(detailed=False)
        is_ready = health_status.get("status") != HealthStatus.UNHEALTHY
        
        if is_ready:
            return {"status": "ready"}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "components": health_status.get("components", {}),
                    "failed_components": health_status.get("failed_components", []),
                },
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)},
        )

@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    获取系统指标
    
    Returns:
        系统性能和资源指标
    """
    try:
        monitoring_service = get_monitoring_service()
        metrics = await monitoring_service.collect_all_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        return {"error": str(e)}

@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="告警级别过滤"),
    component: Optional[str] = Query(None, description="组件或规则名过滤"),
    resolved: Optional[bool] = Query(None, description="是否已解决"),
    limit: Optional[int] = Query(None, ge=1, le=200, description="返回数量上限"),
) -> Dict[str, Any]:
    """
    获取活动告警
    
    Returns:
        当前活动的系统告警
    """
    try:
        monitoring_service = get_monitoring_service()
        alerts = await monitoring_service.alert_manager.get_active_alerts()
        normalized_severity = severity.strip().lower() if severity else None
        normalized_component = component.strip().lower() if component else None

        def parse_alert_timestamp(alert: Dict[str, Any]) -> datetime:
            raw = alert.get("timestamp")
            if isinstance(raw, str):
                parsed = parse_iso_string(raw)
                if parsed:
                    return parsed
            return datetime.min.replace(tzinfo=timezone.utc)

        filtered = []
        for alert in alerts:
            if normalized_severity:
                alert_severity = str(alert.get("severity", "")).lower()
                if alert_severity != normalized_severity:
                    continue
            if normalized_component:
                name = str(alert.get("name", "")).lower()
                metric_component = str(
                    (alert.get("metrics") or {}).get("component", "")
                ).lower()
                if normalized_component not in name and normalized_component not in metric_component:
                    continue
            if resolved is not None:
                alert_resolved = bool(alert.get("resolved"))
                if alert_resolved != resolved:
                    continue
            filtered.append(alert)

        filtered.sort(key=parse_alert_timestamp, reverse=True)
        total_alerts = len(filtered)
        if limit:
            filtered = filtered[:limit]

        return {
            "total_alerts": total_alerts,
            "alerts": filtered
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return {"error": str(e)}
