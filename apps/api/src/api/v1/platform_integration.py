"""平台集成API端点"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import Response
from typing import Dict, List, Any, Optional
import json
import time
import uuid
import psutil
from src.ai.platform_integration.integrator import PlatformIntegrator
from src.ai.platform_integration.optimizer import PerformanceOptimizer
from src.ai.platform_integration.monitoring import MonitoringSystem
from src.ai.platform_integration.documentation import DocumentationGenerator
from src.ai.platform_integration.models import (
    ComponentRegistration,
    WorkflowRequest,
    PlatformHealthStatus,
    PerformanceMetrics,
    MonitoringConfig
)
from src.core.utils.timezone_utils import from_timestamp, utc_now
from fastapi import Response

logger = get_logger(__name__)

router = APIRouter(prefix="/platform", tags=["Platform Integration"])

# 全局实例 (实际项目中应该通过依赖注入管理)
_platform_integrator = None
_performance_optimizer = None
_monitoring_system = None
_documentation_generator = None

def get_platform_integrator() -> PlatformIntegrator:
    """获取平台集成器实例"""
    global _platform_integrator
    if _platform_integrator is None:
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0
        }
        _platform_integrator = PlatformIntegrator(config)
    return _platform_integrator

def get_performance_optimizer() -> PerformanceOptimizer:
    """获取性能优化器实例"""
    global _performance_optimizer
    if _performance_optimizer is None:
        config = {
            'cache': {
                'enabled': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        }
        _performance_optimizer = PerformanceOptimizer(config)
    return _performance_optimizer

def get_monitoring_system() -> MonitoringSystem:
    """获取监控系统实例"""
    global _monitoring_system
    if _monitoring_system is None:
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0
        }
        _monitoring_system = MonitoringSystem(config)
    return _monitoring_system

def get_documentation_generator() -> DocumentationGenerator:
    """获取文档生成器实例"""
    global _documentation_generator
    if _documentation_generator is None:
        config = {
            'docs_output_dir': './docs/generated',
            'template_dir': './templates'
        }
        _documentation_generator = DocumentationGenerator(config)
    return _documentation_generator

# ============================================================================
# 组件管理接口
# ============================================================================

@router.post("/components/register")
async def register_component(
    component: ComponentRegistration,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """注册新组件"""
    try:
        component_info = await integrator._register_component_from_registration(component)
        
        logger.info(f"Component {component.component_id} registered successfully")
        
        return {
            "status": "success",
            "component_id": component.component_id,
            "message": "Component registered successfully",
            "component_status": component_info.status.value
        }
        
    except Exception as e:
        logger.error(f"Error registering component {component.component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/components/{component_id}")
async def unregister_component(
    component_id: str,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """注销组件"""
    try:
        await integrator._unregister_component(component_id)
        
        logger.info(f"Component {component_id} unregistered successfully")
        
        return {
            "status": "success",
            "component_id": component_id,
            "message": "Component unregistered successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error unregistering component {component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components")
async def list_components(
    component_type: Optional[str] = Query(None, description="按组件类型过滤"),
    status: Optional[str] = Query(None, description="按状态过滤"),
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """列出所有组件"""
    try:
        components = {}
        
        for comp_id, comp_info in integrator.components.items():
            # 应用过滤条件
            if component_type and comp_info.component_type.value != component_type:
                continue
            if status and comp_info.status.value != status:
                continue
                
            components[comp_id] = comp_info.to_dict()
        
        return {
            "status": "success",
            "total_components": len(components),
            "components": components
        }
        
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components/{component_id}")
async def get_component_details(
    component_id: str,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """获取组件详情"""
    try:
        if component_id not in integrator.components:
            raise HTTPException(status_code=404, detail=f"Component {component_id} not found")
        
        component_info = integrator.components[component_id]
        
        # 执行实时健康检查
        is_healthy = await integrator._check_component_health(component_info)
        
        component_data = component_info.to_dict()
        component_data["current_health"] = "healthy" if is_healthy else "unhealthy"
        
        return {
            "status": "success",
            "component": component_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component details for {component_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 工作流管理接口
# ============================================================================

@router.post("/workflows/run")
async def run_workflow(
    workflow: WorkflowRequest,
    background_tasks: BackgroundTasks,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """执行端到端工作流"""
    try:
        workflow_id = f"workflow_{int(__import__('time').time())}"
        
        # 验证工作流类型
        valid_types = ["full_fine_tuning", "model_optimization", "evaluation_only", "data_processing"]
        if workflow.workflow_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid workflow type. Valid types: {valid_types}"
            )
        
        # 在后台任务中执行工作流
        background_tasks.add_task(
            integrator._execute_workflow_background,
            workflow_id,
            workflow
        )
        
        logger.info(f"Workflow {workflow_id} started with type {workflow.workflow_type}")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "workflow_type": workflow.workflow_type,
            "message": "Workflow execution started",
            "estimated_duration": _estimate_workflow_duration(workflow.workflow_type)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """获取工作流状态"""
    try:
        status = await integrator._get_workflow_status(workflow_id)
        
        return {
            "status": "success",
            "workflow": status
        }
        
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    except Exception as e:
        logger.error(f"Error getting workflow status for {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """取消工作流执行"""
    try:
        await integrator._cancel_workflow(workflow_id)
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Workflow cancellation requested"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def list_workflows(
    status: Optional[str] = Query(None, description="按状态过滤"),
    workflow_type: Optional[str] = Query(None, description="按类型过滤"),
    limit: int = Query(10, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    integrator: PlatformIntegrator = Depends(get_platform_integrator),
):
    """列出工作流"""
    try:
        workflows = []
        for key in integrator.redis_client.scan_iter("workflow:*", count=1000):
            value = integrator.redis_client.get(key)
            if not value:
                continue
            try:
                workflows.append(json.loads(value))
            except json.JSONDecodeError:
                continue

        workflows.sort(key=lambda w: w.get("started_at", ""), reverse=True)
        if status:
            workflows = [w for w in workflows if w.get("status") == status]
        if workflow_type:
            workflows = [w for w in workflows if w.get("workflow_type") == workflow_type]
        total = len(workflows)
        workflows = workflows[offset:offset + limit]
        
        return {
            "status": "success",
            "total": total,
            "workflows": workflows,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(workflows) < total
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 健康检查和监控接口
# ============================================================================

@router.get("/health", response_model=PlatformHealthStatus)
async def platform_health(
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """平台健康检查"""
    try:
        health_status = await integrator._check_platform_health()
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking platform health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_platform_metrics(
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    """获取Prometheus格式的指标"""
    try:
        metrics_data = monitoring.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

_MONITORING_METRICS_KEY = "platform:monitoring:metrics"
_MONITORING_RULES_KEY = "platform:monitoring:rules"
_MONITORING_ALERTS_KEY = "platform:monitoring:alerts"
_MONITORING_ALLOWED_METRICS = {
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_in",
    "network_out",
    "active_connections",
    "response_time",
    "throughput",
    "error_count",
    "error_rate",
}
_MONITORING_ALLOWED_OPERATORS = {"gt", "lt", "eq", "ne"}
_MONITORING_ALLOWED_SEVERITIES = {"info", "warning", "error", "critical"}

def _load_json(redis_client, key: str, default):
    raw = redis_client.get(key)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default

def _save_json(redis_client, key: str, value, ttl: int = 86400):
    redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))

def _evaluate_rule(operator: str, current_value: float, threshold: float) -> bool:
    if operator == "gt":
        return current_value > threshold
    if operator == "lt":
        return current_value < threshold
    if operator == "eq":
        return current_value == threshold
    if operator == "ne":
        return current_value != threshold
    return False

def _default_monitoring_rules() -> List[Dict[str, Any]]:
    return [
        {
            "rule_id": "default_high_cpu",
            "name": "CPU使用率过高",
            "metric": "cpu_usage",
            "operator": "gt",
            "threshold": 80,
            "duration": 10,
            "enabled": True,
            "severity": "warning",
            "description": "CPU使用率持续高于80%",
        },
        {
            "rule_id": "default_high_memory",
            "name": "内存使用率过高",
            "metric": "memory_usage",
            "operator": "gt",
            "threshold": 85,
            "duration": 10,
            "enabled": True,
            "severity": "warning",
            "description": "内存使用率持续高于85%",
        },
        {
            "rule_id": "default_high_disk",
            "name": "磁盘使用率过高",
            "metric": "disk_usage",
            "operator": "gt",
            "threshold": 90,
            "duration": 10,
            "enabled": True,
            "severity": "warning",
            "description": "磁盘使用率持续高于90%",
        },
        {
            "rule_id": "default_slow_response",
            "name": "响应时间过高",
            "metric": "response_time",
            "operator": "gt",
            "threshold": 1000,
            "duration": 10,
            "enabled": True,
            "severity": "warning",
            "description": "平均响应时间持续高于1000ms",
        },
        {
            "rule_id": "default_high_error_rate",
            "name": "错误率过高",
            "metric": "error_rate",
            "operator": "gt",
            "threshold": 5,
            "duration": 10,
            "enabled": True,
            "severity": "critical",
            "description": "错误率持续高于5%",
        },
    ]

@router.get("/monitoring/metrics")
async def get_monitoring_metrics(
    limit: int = Query(120, ge=1, le=500),
    monitoring: MonitoringSystem = Depends(get_monitoring_system),
):
    """获取系统监控指标（用于UI展示）"""
    from src.core.monitoring import monitoring_service

    stats = await monitoring_service.performance_monitor.get_stats()
    now_ts = time.time()
    net = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    timestamp = utc_now().isoformat()

    history = _load_json(monitoring.redis_client, _MONITORING_METRICS_KEY, [])
    last = history[-1] if history else None
    last_ts = float(last.get("_ts") or 0) if isinstance(last, dict) else 0
    last_in = float(last.get("_net_in") or 0) if isinstance(last, dict) else 0
    last_out = float(last.get("_net_out") or 0) if isinstance(last, dict) else 0

    dt = now_ts - last_ts if last_ts > 0 else 0
    in_rate = (max(0, net.bytes_recv - last_in) / dt / (1024 * 1024)) if dt > 0 else 0
    out_rate = (max(0, net.bytes_sent - last_out) / dt / (1024 * 1024)) if dt > 0 else 0

    sample = {
        "timestamp": timestamp,
        "cpu_usage": cpu,
        "memory_usage": mem,
        "disk_usage": disk,
        "network_in": in_rate,
        "network_out": out_rate,
        "active_connections": int(stats.get("active_requests") or 0),
        "response_time": float(stats.get("average_response_time_ms") or 0),
        "throughput": float(stats.get("requests_per_minute") or 0) / 60,
        "error_count": int(stats.get("recent_error_count") or 0),
        "error_rate": float(stats.get("error_rate") or 0) * 100,
        "_ts": now_ts,
        "_net_in": float(net.bytes_recv),
        "_net_out": float(net.bytes_sent),
    }

    history.append(sample)
    history = history[-500:]
    _save_json(monitoring.redis_client, _MONITORING_METRICS_KEY, history)

    rules = _load_json(monitoring.redis_client, _MONITORING_RULES_KEY, [])
    if not rules:
        rules = _default_monitoring_rules()
        _save_json(monitoring.redis_client, _MONITORING_RULES_KEY, rules)

    alerts = _load_json(monitoring.redis_client, _MONITORING_ALERTS_KEY, [])
    active_by_name = {a.get("name") for a in alerts if a.get("status") == "active"}

    for rule in rules:
        if not rule.get("enabled"):
            continue
        metric_name = rule.get("metric")
        operator = rule.get("operator")
        threshold = rule.get("threshold")
        duration = int(rule.get("duration") or 0)

        if metric_name not in sample:
            continue
        try:
            current_value = float(sample.get(metric_name) or 0)
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            continue

        if duration > 0:
            cutoff = now_ts - duration
            window = [m for m in history if isinstance(m, dict) and float(m.get("_ts") or 0) >= cutoff]
            if not window:
                continue
            triggered = True
            for m in window:
                try:
                    v = float(m.get(metric_name) or 0)
                except (TypeError, ValueError):
                    triggered = False
                    break
                if not _evaluate_rule(operator, v, threshold_value):
                    triggered = False
                    break
        else:
            triggered = _evaluate_rule(operator, current_value, threshold_value)

        if not triggered or rule.get("name") in active_by_name:
            continue

        component = "system" if metric_name in {"cpu_usage", "memory_usage", "disk_usage"} else "api"
        alerts.append({
            "alert_id": uuid.uuid4().hex,
            "name": rule.get("name"),
            "severity": rule.get("severity"),
            "status": "active",
            "description": rule.get("description") or "",
            "component": component,
            "triggered_at": timestamp,
            "threshold_value": threshold_value,
            "current_value": current_value
        })
        active_by_name.add(rule.get("name"))

    _save_json(monitoring.redis_client, _MONITORING_ALERTS_KEY, alerts)
    return {"status": "success", "metrics": history[-limit:]}

@router.get("/monitoring/rules")
async def list_monitoring_rules(
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    rules = _load_json(monitoring.redis_client, _MONITORING_RULES_KEY, [])
    if not rules:
        rules = _default_monitoring_rules()
        _save_json(monitoring.redis_client, _MONITORING_RULES_KEY, rules)
    return {"status": "success", "rules": rules}

@router.post("/monitoring/rules")
async def create_monitoring_rule(
    payload: Dict[str, Any],
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name不能为空")
    metric = payload.get("metric")
    if metric not in _MONITORING_ALLOWED_METRICS:
        raise HTTPException(status_code=400, detail="metric不支持")
    operator = payload.get("operator")
    if operator not in _MONITORING_ALLOWED_OPERATORS:
        raise HTTPException(status_code=400, detail="operator不支持")
    severity = payload.get("severity")
    if severity not in _MONITORING_ALLOWED_SEVERITIES:
        raise HTTPException(status_code=400, detail="severity不支持")
    try:
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="threshold无效")
    try:
        duration = int(payload.get("duration") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="duration无效")
    if duration < 0:
        raise HTTPException(status_code=400, detail="duration无效")

    rules = _load_json(monitoring.redis_client, _MONITORING_RULES_KEY, [])
    rule = {
        "rule_id": uuid.uuid4().hex,
        "name": name,
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "duration": duration,
        "enabled": True,
        "severity": severity,
        "description": str(payload.get("description") or "")
    }
    rules.append(rule)
    _save_json(monitoring.redis_client, _MONITORING_RULES_KEY, rules)
    return {"status": "success", "rule_id": rule["rule_id"]}

@router.patch("/monitoring/rules/{rule_id}")
async def update_monitoring_rule(
    rule_id: str,
    payload: Dict[str, Any],
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    if "enabled" not in payload:
        raise HTTPException(status_code=400, detail="enabled必填")
    rules = _load_json(monitoring.redis_client, _MONITORING_RULES_KEY, [])
    updated = False
    for rule in rules:
        if rule.get("rule_id") == rule_id:
            rule["enabled"] = bool(payload["enabled"])
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Rule not found")
    _save_json(monitoring.redis_client, _MONITORING_RULES_KEY, rules)
    return {"status": "success"}

@router.get("/monitoring/alerts")
async def list_monitoring_alerts(
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    alerts = _load_json(monitoring.redis_client, _MONITORING_ALERTS_KEY, [])
    return {"status": "success", "alerts": alerts}

@router.post("/monitoring/alerts/{alert_id}/resolve")
async def resolve_monitoring_alert(
    alert_id: str,
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    alerts = _load_json(monitoring.redis_client, _MONITORING_ALERTS_KEY, [])
    now = utc_now().isoformat()
    updated = False
    for alert in alerts:
        if alert.get("alert_id") == alert_id:
            alert["status"] = "resolved"
            alert["resolved_at"] = now
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found")
    _save_json(monitoring.redis_client, _MONITORING_ALERTS_KEY, alerts)
    return {"status": "success"}

@router.get("/monitoring/services")
async def list_monitoring_services(
    integrator: PlatformIntegrator = Depends(get_platform_integrator),
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    from src.core.database import test_database_connection
    from src.core.redis import test_redis_connection

    def update_check(name: str, ok: bool, response_time_ms: float) -> Dict[str, Any]:
        key = f"platform:monitoring:service_check:{name}"
        data = _load_json(monitoring.redis_client, key, {"total": 0, "failures": 0})
        data["total"] = int(data.get("total") or 0) + 1
        data["failures"] = int(data.get("failures") or 0) + (0 if ok else 1)
        data["last_check"] = utc_now().isoformat()
        data["last_response_time_ms"] = round(response_time_ms, 2)
        _save_json(monitoring.redis_client, key, data)
        return data

    services = []
    for comp in integrator.components.values():
        meta = comp.metadata or {}
        total = int(meta.get("health_check_total") or 0)
        failures = int(meta.get("health_check_failures") or 0)
        error_rate = (failures / total * 100) if total > 0 else 0
        last_check = meta.get("last_health_check_at") or comp.last_heartbeat.isoformat()
        response_time = float(meta.get("last_health_check_ms") or 0)
        uptime = max(0, (utc_now() - comp.registered_at).total_seconds())
        status = comp.status.value
        services.append({
            "service_name": comp.name or comp.component_id,
            "status": "healthy" if status == "healthy" else "degraded" if status in {"starting", "stopping"} else "unhealthy",
            "response_time": response_time,
            "uptime": uptime,
            "last_check": last_check,
            "error_rate": round(error_rate, 2)
        })

    start = time.perf_counter()
    db_ok = await test_database_connection()
    db_ms = (time.perf_counter() - start) * 1000
    db_check = update_check("database", db_ok, db_ms)
    db_error_rate = (db_check["failures"] / db_check["total"] * 100) if db_check["total"] else 0
    services.append({
        "service_name": "database",
        "status": "healthy" if db_ok else "unhealthy",
        "response_time": round(db_ms, 2),
        "uptime": 0,
        "last_check": db_check["last_check"],
        "error_rate": round(db_error_rate, 2)
    })

    start = time.perf_counter()
    redis_ok = await test_redis_connection()
    redis_ms = (time.perf_counter() - start) * 1000
    redis_check = update_check("redis", redis_ok, redis_ms)
    redis_error_rate = (redis_check["failures"] / redis_check["total"] * 100) if redis_check["total"] else 0
    services.append({
        "service_name": "redis",
        "status": "healthy" if redis_ok else "unhealthy",
        "response_time": round(redis_ms, 2),
        "uptime": 0,
        "last_check": redis_check["last_check"],
        "error_rate": round(redis_error_rate, 2)
    })

    return {"status": "success", "services": services}

@router.get("/monitoring/report")
async def get_monitoring_report(
    monitoring: MonitoringSystem = Depends(get_monitoring_system)
):
    """生成监控报告"""
    try:
        report = await monitoring.generate_monitoring_report()
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating monitoring report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 性能优化接口
# ============================================================================

@router.post("/optimization/run")
async def run_performance_optimization(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """运行性能优化"""
    try:
        optimization_results = await optimizer.optimize_system_performance()
        
        logger.info("Performance optimization completed successfully")
        
        return {
            "status": "success",
            "optimization_results": optimization_results
        }
        
    except Exception as e:
        logger.error(f"Error running performance optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/metrics")
async def get_performance_metrics(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """获取性能指标"""
    try:
        metrics = await optimizer.collect_metrics()
        
        return {
            "status": "success",
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_usage": metrics.disk_usage,
                "network_usage": metrics.network_usage,
                "bottlenecks": metrics.bottlenecks,
                "timestamp": metrics.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimization/profile/{profile_name}")
async def apply_optimization_profile(
    profile_name: str,
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """应用优化配置文件"""
    try:
        result = await optimizer.apply_optimization_profile(profile_name)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Applied optimization profile: {profile_name}")
        
        return {
            "status": "success",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying optimization profile {profile_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/report")
async def get_performance_report(
    optimizer: PerformanceOptimizer = Depends(get_performance_optimizer)
):
    """获取性能报告"""
    try:
        report = await optimizer.generate_performance_report()
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 文档生成接口
# ============================================================================

@router.post("/documentation/generate")
async def generate_documentation(
    background_tasks: BackgroundTasks,
    doc_generator: DocumentationGenerator = Depends(get_documentation_generator)
):
    """生成完整文档"""
    try:
        # 在后台任务中生成文档
        background_tasks.add_task(
            _generate_documentation_background,
            doc_generator
        )
        
        return {
            "status": "success",
            "message": "Documentation generation started",
            "estimated_duration": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting documentation generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documentation/status")
async def get_documentation_status():
    """获取文档生成状态"""
    try:
        redis_client = get_platform_integrator().redis_client
        raw = redis_client.get("platform:documentation:status")
        if not raw:
            return {"status": "success", "documentation": {"status": "idle"}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"status": "unknown", "raw": raw}
        return {"status": "success", "documentation": data}
    except Exception as e:
        logger.error(f"Error getting documentation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documentation/training-materials")
async def generate_training_materials(
    background_tasks: BackgroundTasks,
    doc_generator: DocumentationGenerator = Depends(get_documentation_generator)
):
    """生成培训材料"""
    try:
        # 在后台任务中生成培训材料
        background_tasks.add_task(
            _generate_training_materials_background,
            doc_generator
        )
        
        return {
            "status": "success",
            "message": "Training materials generation started",
            "estimated_duration": "3-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error starting training materials generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 系统配置接口
# ============================================================================

@router.get("/config")
async def get_platform_config():
    """获取平台配置"""
    try:
        # 返回平台配置信息（敏感信息已脱敏）
        config = {
            "version": "1.0.0",
            "build": "20250115",
            "features": {
                "component_registration": True,
                "workflow_execution": True,
                "performance_optimization": True,
                "monitoring": True,
                "documentation_generation": True
            },
            "limits": {
                "max_components": 100,
                "max_concurrent_workflows": 10,
                "workflow_timeout_seconds": 3600
            },
            "supported_workflow_types": [
                "full_fine_tuning",
                "model_optimization", 
                "evaluation_only",
                "data_processing"
            ],
            "supported_component_types": [
                "fine_tuning",
                "compression",
                "hyperparameter",
                "evaluation",
                "data_management",
                "model_service"
            ]
        }
        
        return {
            "status": "success",
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Error getting platform config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_platform_stats(
    integrator: PlatformIntegrator = Depends(get_platform_integrator)
):
    """获取平台统计信息"""
    try:
        # 组件统计
        component_stats = {
            "total": len(integrator.components),
            "healthy": len([c for c in integrator.components.values() if c.status.value == "healthy"]),
            "unhealthy": len([c for c in integrator.components.values() if c.status.value == "unhealthy"]),
            "by_type": {}
        }
        
        # 按类型统计
        for comp in integrator.components.values():
            comp_type = comp.component_type.value
            if comp_type not in component_stats["by_type"]:
                component_stats["by_type"][comp_type] = 0
            component_stats["by_type"][comp_type] += 1
        
        return {
            "status": "success",
            "stats": {
                "components": component_stats
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# 辅助函数
# ============================================================================

def _estimate_workflow_duration(workflow_type: str) -> str:
    """估算工作流执行时间"""
    durations = {
        "full_fine_tuning": "2-4 hours",
        "model_optimization": "30-60 minutes",
        "evaluation_only": "10-20 minutes",
        "data_processing": "5-15 minutes"
    }
    return durations.get(workflow_type, "Unknown")

async def _generate_documentation_background(doc_generator: DocumentationGenerator):
    """后台生成文档"""
    redis_client = get_platform_integrator().redis_client
    started_at = __import__("time").time()
    redis_client.setex(
        "platform:documentation:status",
        86400,
        json.dumps(
            {
                "status": "running",
                "started_at": from_timestamp(started_at).isoformat(),
            },
            ensure_ascii=False,
        ),
    )
    try:
        result = await doc_generator.generate_complete_documentation()
        redis_client.setex(
            "platform:documentation:status",
            86400,
            json.dumps(
                {
                    "status": "completed",
                    "started_at": from_timestamp(started_at).isoformat(),
                    "completed_at": utc_now().isoformat(),
                    "result": result,
                },
                ensure_ascii=False,
            ),
        )
        logger.info(f"Documentation generation completed: {result['status']}")
    except Exception as e:
        redis_client.setex(
            "platform:documentation:status",
            86400,
            json.dumps(
                {
                    "status": "failed",
                    "started_at": from_timestamp(started_at).isoformat(),
                    "failed_at": utc_now().isoformat(),
                    "error": str(e),
                },
                ensure_ascii=False,
            ),
        )
        logger.error(f"Background documentation generation failed: {e}")

async def _generate_training_materials_background(doc_generator: DocumentationGenerator):
    """后台生成培训材料"""
    try:
        result = await doc_generator.generate_training_materials()
        logger.info(f"Training materials generation completed: {result['status']}")
    except Exception as e:
        logger.error(f"Background training materials generation failed: {e}")

# 导入Response类型
from src.core.logging import get_logger
