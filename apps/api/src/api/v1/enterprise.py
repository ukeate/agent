"""
企业级架构管理API
提供企业级架构的监控、配置和管理接口
"""

import json
import time
import uuid
from io import BytesIO
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response
from pydantic import Field
from src.ai.autogen.security.trism import AITRiSMFramework
from src.ai.autogen.security.attack_detection import AttackDetectionManager
from src.ai.autogen.security.auto_response import SecurityResponseManager
from src.ai.autogen.performance_optimization import PerformanceOptimizer, PerformanceProfile
from src.ai.autogen.compliance_testing import ComplianceFramework, ComplianceStandard
from src.core.config import get_settings
from src.core.database import get_db_session, test_database_connection
from src.core.monitoring import monitoring_service
from src.core.qdrant import qdrant_manager
from src.core.redis import get_redis, test_redis_connection
from src.core.security.audit import audit_logger
from sqlalchemy import text
from src.api.base_model import ApiBaseModel

settings = get_settings()
router = APIRouter(prefix="/enterprise", tags=["enterprise"])

_trism_framework: Optional[AITRiSMFramework] = None
_attack_detector: Optional[AttackDetectionManager] = None
_response_manager: Optional[SecurityResponseManager] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None
_compliance_framework: Optional[ComplianceFramework] = None
_performance_optimizer_started = False

_CFG_KEY = "enterprise:configuration"
_ASSESSMENT_KEY_PREFIX = "enterprise:compliance:assessment:"
_SECURITY_SCAN_KEY_PREFIX = "enterprise:security:scan:"
_LAST_SECURITY_SCAN_KEY = "enterprise:security:last_scan"
_ALERT_KEY_PREFIX = "enterprise:alert:"
_ALERT_INDEX_KEY = "enterprise:alerts"
_ALERT_DEDUP_KEY_PREFIX = "enterprise:alert:dedup:"
_TREND_KEY_PREFIX = "enterprise:trend:"

# Pydantic模型
class SystemHealthResponse(ApiBaseModel):
    overall_status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_agents: int
    active_tasks: int
    error_rate: float
    response_time: float
    timestamp: datetime

class SecurityMetricsResponse(ApiBaseModel):
    threat_level: str
    detected_attacks: int
    blocked_requests: int
    security_events: int
    compliance_score: float
    last_security_scan: datetime
    active_threats: List[Dict[str, Any]]

class PerformanceMetricsResponse(ApiBaseModel):
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    concurrent_users: int
    cache_hit_rate: float
    optimization_level: str
    resource_utilization: Dict[str, float]

class ComplianceDataResponse(ApiBaseModel):
    overall_score: float
    status: str
    standards: List[str]
    last_assessment: datetime
    issues_count: int
    requirements_total: int
    requirements_passed: int
    detailed_results: List[Dict[str, Any]]

class EnterpriseConfigurationResponse(ApiBaseModel):
    security: Dict[str, Any]
    performance: Dict[str, Any]
    monitoring: Dict[str, Any]
    compliance: Dict[str, Any]

class EnterpriseOverviewResponse(ApiBaseModel):
    system_health: SystemHealthResponse
    security: SecurityMetricsResponse
    performance: PerformanceMetricsResponse
    compliance: ComplianceDataResponse
    configuration: EnterpriseConfigurationResponse
    last_updated: datetime

class ConfigurationUpdateRequest(ApiBaseModel):
    security: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    compliance: Optional[Dict[str, Any]] = None

class ComplianceAssessmentRequest(ApiBaseModel):
    standards: Optional[List[str]] = Field(default_factory=list)

class SecurityScanRequest(ApiBaseModel):
    deep_scan: bool = False
    target_components: Optional[List[str]] = None

class PerformanceOptimizationRequest(ApiBaseModel):
    optimizations: List[str]

def get_trism_framework():
    """获取TRiSM框架实例"""
    global _trism_framework
    if _trism_framework is None:
        _trism_framework = AITRiSMFramework()
    return _trism_framework

def get_attack_detector():
    """获取攻击检测器实例"""
    global _attack_detector
    if _attack_detector is None:
        _attack_detector = AttackDetectionManager()
    return _attack_detector

def get_response_manager():
    """获取安全响应管理器实例"""
    global _response_manager
    if _response_manager is None:
        _response_manager = SecurityResponseManager()
    return _response_manager

def get_performance_optimizer():
    """获取性能优化器实例"""
    global _performance_optimizer
    if _performance_optimizer is None:
        profile = PerformanceProfile()
        _performance_optimizer = PerformanceOptimizer(profile)
    return _performance_optimizer

def get_compliance_framework():
    """获取合规框架实例"""
    global _compliance_framework
    if _compliance_framework is None:
        standards = [
            ComplianceStandard.ISO27001,
            ComplianceStandard.SOC2,
            ComplianceStandard.GDPR,
            ComplianceStandard.AI_GOVERNANCE
        ]
        _compliance_framework = ComplianceFramework(
            standards=standards,
            trism_framework=get_trism_framework(),
            attack_detector=get_attack_detector(),
            response_manager=get_response_manager()
        )
    return _compliance_framework

def _get_redis_client():
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化")
    return redis_client

def _parse_time_range(value: str) -> int:
    if value.endswith("h") and value[:-1].isdigit():
        return int(value[:-1]) * 3600
    if value.endswith("d") and value[:-1].isdigit():
        return int(value[:-1]) * 86400
    raise HTTPException(status_code=400, detail="time_range格式错误，仅支持如24h/30d")

def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(len(values) * q)
    if idx >= len(values):
        idx = len(values) - 1
    return float(values[idx])

async def _record_trend(metric: str, value: float) -> None:
    redis_client = _get_redis_client()
    key = f"{_TREND_KEY_PREFIX}{metric}"
    payload = json.dumps({"timestamp": utc_now().isoformat(), "value": float(value)}, ensure_ascii=False)
    await redis_client.zadd(key, {payload: time.time()})
    card = await redis_client.zcard(key)
    if card > 1000:
        await redis_client.zremrangebyrank(key, 0, card - 1001)

async def _ensure_performance_optimizer_started() -> PerformanceOptimizer:
    global _performance_optimizer_started
    optimizer = get_performance_optimizer()
    if not _performance_optimizer_started:
        await optimizer.start()
        _performance_optimizer_started = True
    return optimizer

def _default_configuration() -> Dict[str, Any]:
    profile = PerformanceProfile()
    return {
        "security": {
            "trism_enabled": True,
            "attack_detection_enabled": True,
            "auto_response_enabled": True,
            "security_level": "standard",
        },
        "performance": {
            "optimization_level": profile.optimization_level.value,
            "max_concurrent_tasks": profile.max_concurrent_tasks,
            "cache_size": profile.cache_size,
            "load_balancing_strategy": "least_loaded",
        },
        "monitoring": {
            "otel_enabled": False,
            "audit_logging_enabled": True,
            "metrics_retention_days": 7,
            "alert_thresholds": {
                "cpu_usage_percent": 80.0,
                "memory_usage_percent": 85.0,
                "error_rate": 0.05,
                "average_response_time_ms": 1000.0,
            },
        },
        "compliance": {
            "enabled_standards": [
                ComplianceStandard.ISO27001.value,
                ComplianceStandard.SOC2.value,
                ComplianceStandard.GDPR.value,
                ComplianceStandard.AI_GOVERNANCE.value,
            ],
            "auto_assessment": False,
            "notification_channels": [],
        },
    }

async def _load_configuration() -> Dict[str, Any]:
    redis_client = _get_redis_client()
    raw = await redis_client.get(_CFG_KEY)
    if raw:
        return json.loads(raw)
    cfg = _default_configuration()
    await redis_client.set(_CFG_KEY, json.dumps(cfg, ensure_ascii=False))
    return cfg

async def _save_configuration(cfg: Dict[str, Any]) -> None:
    redis_client = _get_redis_client()
    await redis_client.set(_CFG_KEY, json.dumps(cfg, ensure_ascii=False))

def _parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)

async def _run_security_scan_checks(
    deep_scan: bool, target_components: Optional[List[str]]
) -> List[Dict[str, Any]]:
    enabled = set(target_components or [])
    findings: List[Dict[str, Any]] = []
    now = utc_now().isoformat()

    def add(finding_type: str, severity: str, description: str):
        if enabled and finding_type not in enabled:
            return
        findings.append(
            {
                "id": f"finding-{uuid.uuid4().hex[:8]}",
                "type": finding_type,
                "severity": severity,
                "description": description,
                "timestamp": now,
            }
        )

    if settings.DEBUG:
        add("debug", "medium", "DEBUG模式已启用")
    if not settings.FORCE_HTTPS:
        add("https", "high", "未启用FORCE_HTTPS")
    if settings.SECRET_KEY and len(settings.SECRET_KEY) < 32:
        add("secret_key", "high", "SECRET_KEY长度不足32")

    db_ok = await test_database_connection()
    if not db_ok:
        add("database", "critical", "数据库连接失败")

    redis_ok = await test_redis_connection()
    if not redis_ok:
        add("redis", "critical", "Redis连接失败")

    qdrant_ok = await qdrant_manager.health_check()
    if not qdrant_ok:
        add("qdrant", "critical", "Qdrant连接失败")

    if deep_scan:
        add("docs", "low", "API文档已启用(/docs)")

    return findings

async def _ensure_last_security_scan() -> Dict[str, Any]:
    redis_client = _get_redis_client()
    last_scan = await redis_client.get(_LAST_SECURITY_SCAN_KEY)
    if last_scan:
        return {"last_scan_at": _parse_datetime(last_scan)}
    scan_id = f"scan-{uuid.uuid4().hex[:8]}"
    started_at = utc_now().isoformat()
    findings = await _run_security_scan_checks(False, None)
    completed_at = utc_now().isoformat()
    record = {
        "scan_id": scan_id,
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "options": {"deep_scan": False, "target_components": None},
        "findings": findings,
    }
    await redis_client.set(f"{_SECURITY_SCAN_KEY_PREFIX}{scan_id}", json.dumps(record, ensure_ascii=False))
    await redis_client.set(_LAST_SECURITY_SCAN_KEY, completed_at)
    return {"last_scan_at": _parse_datetime(completed_at), "findings": findings}

@router.get("/overview", response_model=EnterpriseOverviewResponse)
async def get_enterprise_overview():
    """获取企业架构总览"""
    try:
        # 获取系统健康状态
        system_health = await get_system_health_data()
        
        # 获取安全指标
        security_metrics = await get_security_metrics_data()
        
        # 获取性能指标
        performance_metrics = await get_performance_metrics_data()
        
        # 获取合规数据
        compliance_data = await get_compliance_data()
        
        # 获取配置信息
        configuration = await get_configuration_data()
        
        return EnterpriseOverviewResponse(
            system_health=system_health,
            security=security_metrics,
            performance=performance_metrics,
            compliance=compliance_data,
            configuration=configuration,
            last_updated=utc_now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取企业总览失败: {str(e)}")

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """获取系统健康指标"""
    return await get_system_health_data()

async def get_system_health_data() -> SystemHealthResponse:
    """获取系统健康数据"""
    try:
        import psutil
    except Exception:
        raise HTTPException(status_code=503, detail="监控依赖不可用")
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
    except Exception:
        raise HTTPException(status_code=500, detail="获取系统指标失败")
    disk_percent = float(disk.percent)
    if cpu_usage > 90 or memory.percent > 95 or disk_percent > 95:
        overall_status = "unhealthy"
    elif cpu_usage > 80 or memory.percent > 85 or disk_percent > 90:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    perf_stats = await monitoring_service.performance_monitor.get_stats()
    error_rate = float(perf_stats.get("error_rate", 0.0))
    response_time_ms = float(perf_stats.get("average_response_time_ms", 0.0))
    await _record_trend("cpu_usage", float(cpu_usage))
    await _record_trend("memory_usage", float(memory.percent))
    await _record_trend("disk_usage", disk_percent)
    await _record_trend("error_rate", error_rate)
    await _record_trend("response_time_ms", response_time_ms)
    return SystemHealthResponse(
        overall_status=overall_status,
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        disk_usage=disk_percent,
        active_agents=0,
        active_tasks=int(perf_stats.get("active_requests", 0)),
        error_rate=error_rate,
        response_time=response_time_ms,
        timestamp=utc_now()
    )

@router.get("/security", response_model=SecurityMetricsResponse)
async def get_security_metrics():
    """获取安全指标"""
    return await get_security_metrics_data()

async def get_security_metrics_data() -> SecurityMetricsResponse:
    """获取安全指标数据"""
    try:
        response_manager = get_response_manager()
        
        # 获取安全统计
        stats = response_manager.get_statistics()
        redis_client = _get_redis_client()
        acl_blocked = await redis_client.hget("acl:metrics", "blocked_requests")
        blocked_requests = int(acl_blocked or 0)
        scan_info = await _ensure_last_security_scan()
        last_scan_at = scan_info["last_scan_at"]
        findings = scan_info.get("findings") or []
        compliance_score = 0.0
        framework = get_compliance_framework()
        latest_report = framework.get_latest_report() if framework else None
        if latest_report:
            compliance_score = float(latest_report.overall_score)
        
        # 计算威胁等级
        critical_events = stats.get("events", {}).get("by_threat_level", {}).get("critical", 0)
        high_events = stats.get("events", {}).get("by_threat_level", {}).get("high", 0)
        critical_findings = sum(1 for f in findings if f.get("severity") == "critical")
        high_findings = sum(1 for f in findings if f.get("severity") == "high")
        
        if critical_events > 0 or critical_findings > 0:
            threat_level = "critical"
        elif high_events > 0 or high_findings > 0:
            threat_level = "high"
        elif stats.get("events", {}).get("by_threat_level", {}).get("medium", 0) > 0:
            threat_level = "medium"
        else:
            threat_level = "low"
        await _record_trend("blocked_requests", float(blocked_requests))
        
        return SecurityMetricsResponse(
            threat_level=threat_level,
            detected_attacks=int(stats.get("events", {}).get("total", 0)),
            blocked_requests=blocked_requests,
            security_events=int(stats.get("events", {}).get("recent_24h", 0)),
            compliance_score=compliance_score,
            last_security_scan=last_scan_at,
            active_threats=[f for f in findings if f.get("severity") in ["high", "critical"]],
        )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"获取安全指标失败: {e}")

@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """获取性能指标"""
    return await get_performance_metrics_data()

async def get_performance_metrics_data() -> PerformanceMetricsResponse:
    """获取性能指标数据"""
    optimizer = await _ensure_performance_optimizer_started()
    try:
        overview = optimizer.get_performance_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取性能指标失败: {e}")
    perf_stats = await monitoring_service.performance_monitor.get_stats()
    recent = [
        r for r in monitoring_service.performance_monitor.request_times
        if r.get("timestamp", 0) > time.time() - 60
    ]
    durations_ms = [float(r.get("duration", 0.0)) * 1000 for r in recent]
    latest_metrics = optimizer.resource_monitor.get_latest_metrics()
    io_per_s = 0.0
    net_per_s = 0.0
    if len(optimizer.resource_monitor.metrics_history) >= 2:
        a = optimizer.resource_monitor.metrics_history[-2]
        b = optimizer.resource_monitor.metrics_history[-1]
        dt = (b.timestamp - a.timestamp).total_seconds() or 0.0
        if dt > 0:
            io_per_s = float((b.io_read_bytes - a.io_read_bytes) + (b.io_write_bytes - a.io_write_bytes)) / dt
            net_per_s = float((b.network_sent - a.network_sent) + (b.network_recv - a.network_recv)) / dt
    return PerformanceMetricsResponse(
        throughput=float(perf_stats.get("requests_per_minute", 0)) / 60.0,
        latency_p50=_percentile(durations_ms, 0.5),
        latency_p95=_percentile(durations_ms, 0.95),
        latency_p99=_percentile(durations_ms, 0.99),
        concurrent_users=int(perf_stats.get("active_requests", 0)),
        cache_hit_rate=float(overview.get("cache", {}).get("hit_rate", 0.0)),
        optimization_level=str(overview.get("profile", {}).get("optimization_level", "unknown")),
        resource_utilization={
            "cpu": float(getattr(latest_metrics, "cpu_usage", 0.0) if latest_metrics else 0.0),
            "memory": float(getattr(latest_metrics, "memory_usage", 0.0) if latest_metrics else 0.0),
            "io": io_per_s,
            "network": net_per_s,
        },
    )

@router.get("/compliance", response_model=ComplianceDataResponse)
async def get_compliance_data_endpoint():
    """获取合规数据"""
    return await get_compliance_data()

async def get_compliance_data() -> ComplianceDataResponse:
    """获取合规数据"""
    try:
        framework = get_compliance_framework()
        
        if framework:
            latest_report = framework.get_latest_report()
            if not latest_report:
                latest_report = await framework.run_compliance_assessment()
            await _record_trend("compliance_score", float(latest_report.overall_score))
            return ComplianceDataResponse(
                overall_score=latest_report.overall_score,
                status=latest_report.status.value,
                standards=[s.value for s in latest_report.standards],
                last_assessment=latest_report.generated_at,
                issues_count=len([r for r in latest_report.test_results if r.status.value == "non_compliant"]),
                requirements_total=len(latest_report.test_results),
                requirements_passed=len([r for r in latest_report.test_results if r.status.value == "compliant"]),
                detailed_results=[r.to_dict() for r in latest_report.test_results],
            )
        raise HTTPException(status_code=503, detail="合规框架不可用")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取合规数据失败: {e}")

@router.get("/configuration", response_model=EnterpriseConfigurationResponse)
async def get_configuration():
    """获取企业配置"""
    return await get_configuration_data()

async def get_configuration_data() -> EnterpriseConfigurationResponse:
    """获取配置数据"""
    cfg = await _load_configuration()
    return EnterpriseConfigurationResponse(**cfg)

@router.put("/configuration", response_model=EnterpriseConfigurationResponse)
async def update_configuration(request: ConfigurationUpdateRequest):
    """更新企业配置"""
    cfg = await _load_configuration()
    if request.security is not None:
        cfg["security"].update(request.security)
    if request.performance is not None:
        cfg["performance"].update(request.performance)
    if request.monitoring is not None:
        cfg["monitoring"].update(request.monitoring)
    if request.compliance is not None:
        cfg["compliance"].update(request.compliance)
    await _save_configuration(cfg)
    await audit_logger.log_security_event(
        event_type="enterprise_config_updated",
        details={"updated_keys": [k for k in ["security", "performance", "monitoring", "compliance"] if getattr(request, k) is not None]},
        level="info",
    )
    return EnterpriseConfigurationResponse(**cfg)

@router.post("/compliance/assess")
async def run_compliance_assessment(
    request: ComplianceAssessmentRequest,
    background_tasks: BackgroundTasks
):
    """运行合规评估"""
    framework = get_compliance_framework()
    if not framework:
        raise HTTPException(status_code=503, detail="合规框架未初始化")
    assessment_id = f"assessment-{uuid.uuid4().hex[:8]}"
    redis_client = _get_redis_client()
    started_at = utc_now().isoformat()
    standards_raw = request.standards or []
    record = {
        "assessment_id": assessment_id,
        "status": "running",
        "progress": 0,
        "started_at": started_at,
        "standards": standards_raw,
    }
    await redis_client.set(f"{_ASSESSMENT_KEY_PREFIX}{assessment_id}", json.dumps(record, ensure_ascii=False))

    async def run_assessment_task():
        key = f"{_ASSESSMENT_KEY_PREFIX}{assessment_id}"
        try:
            await redis_client.set(
                key,
                json.dumps({**record, "progress": 10}, ensure_ascii=False),
            )
            standards = [ComplianceStandard(s) for s in standards_raw] if standards_raw else None
            report = await framework.run_compliance_assessment(standards)
            data = ComplianceDataResponse(
                overall_score=report.overall_score,
                status=report.status.value,
                standards=[s.value for s in report.standards],
                last_assessment=report.generated_at,
                issues_count=len([r for r in report.test_results if r.status.value == "non_compliant"]),
                requirements_total=len(report.test_results),
                requirements_passed=len([r for r in report.test_results if r.status.value == "compliant"]),
                detailed_results=[r.to_dict() for r in report.test_results],
            )
            await redis_client.set(
                key,
                json.dumps(
                    {
                        **record,
                        "status": "completed",
                        "progress": 100,
                        "completed_at": utc_now().isoformat(),
                        "result": json.loads(data.model_dump_json()),
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as e:
            await redis_client.set(
                key,
                json.dumps(
                    {
                        **record,
                        "status": "failed",
                        "progress": 100,
                        "completed_at": utc_now().isoformat(),
                        "error": str(e),
                    },
                    ensure_ascii=False,
                ),
            )

    background_tasks.add_task(run_assessment_task)
    return {"assessment_id": assessment_id, "status": "started"}

@router.get("/compliance/assess/{assessment_id}")
async def get_assessment_status(assessment_id: str):
    """获取合规评估状态"""
    redis_client = _get_redis_client()
    raw = await redis_client.get(f"{_ASSESSMENT_KEY_PREFIX}{assessment_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="assessment_id不存在")
    record = json.loads(raw)
    return {
        "status": record.get("status", "unknown"),
        "progress": int(record.get("progress", 0)),
        "result": record.get("result"),
    }

@router.post("/security/scan")
async def trigger_security_scan(
    request: SecurityScanRequest,
    background_tasks: BackgroundTasks
):
    """触发安全扫描"""
    redis_client = _get_redis_client()
    scan_id = f"scan-{uuid.uuid4().hex[:8]}"
    started_at = utc_now().isoformat()
    record = {
        "scan_id": scan_id,
        "status": "running",
        "started_at": started_at,
        "options": {"deep_scan": bool(request.deep_scan), "target_components": request.target_components},
        "findings": [],
    }
    await redis_client.set(f"{_SECURITY_SCAN_KEY_PREFIX}{scan_id}", json.dumps(record, ensure_ascii=False))

    async def run_scan_task():
        key = f"{_SECURITY_SCAN_KEY_PREFIX}{scan_id}"
        try:
            findings = await _run_security_scan_checks(bool(request.deep_scan), request.target_components)
            completed_at = utc_now().isoformat()
            await redis_client.set(
                key,
                json.dumps({**record, "status": "completed", "completed_at": completed_at, "findings": findings}, ensure_ascii=False),
            )
            await redis_client.set(_LAST_SECURITY_SCAN_KEY, completed_at)
            await _record_trend("security_findings", float(len(findings)))
        except Exception as e:
            await redis_client.set(
                key,
                json.dumps({**record, "status": "failed", "completed_at": utc_now().isoformat(), "error": str(e)}, ensure_ascii=False),
            )

    background_tasks.add_task(run_scan_task)
    return {"scan_id": scan_id}

@router.get("/security/scan/{scan_id}")
async def get_security_scan_result(scan_id: str):
    """获取安全扫描结果"""
    redis_client = _get_redis_client()
    raw = await redis_client.get(f"{_SECURITY_SCAN_KEY_PREFIX}{scan_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="scan_id不存在")
    record = json.loads(raw)
    return {"status": record.get("status", "unknown"), "findings": record.get("findings", [])}

@router.get("/alerts")
async def get_alerts(
    level: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取监控告警"""
    try:
        import psutil
    except Exception:
        raise HTTPException(status_code=503, detail="监控依赖不可用")
    cfg = await _load_configuration()
    thresholds = cfg.get("monitoring", {}).get("alert_thresholds", {})
    perf_stats = await monitoring_service.performance_monitor.get_stats()
    candidates = [
        ("cpu_usage_percent", float(psutil.cpu_percent()), thresholds.get("cpu_usage_percent"), "CPU使用率过高", "cpu"),
        ("memory_usage_percent", float(psutil.virtual_memory().percent), thresholds.get("memory_usage_percent"), "内存使用率过高", "memory"),
        ("disk_usage_percent", float(psutil.disk_usage('/').percent), thresholds.get("disk_usage_percent"), "磁盘使用率过高", "disk"),
        ("error_rate", float(perf_stats.get("error_rate", 0.0)) * 100.0, float(thresholds.get("error_rate", 0.0)) * 100.0, "错误率过高", "error"),
        ("average_response_time_ms", float(perf_stats.get("average_response_time_ms", 0.0)), thresholds.get("average_response_time_ms"), "平均响应时间过高", "latency"),
    ]
    redis_client = _get_redis_client()
    now_score = time.time()
    for key, value, threshold, title, kind in candidates:
        if threshold is None:
            continue
        if value <= float(threshold):
            continue
        dedup_key = f"{_ALERT_DEDUP_KEY_PREFIX}{kind}"
        existing = await redis_client.get(dedup_key)
        if existing:
            raw_alert = await redis_client.get(f"{_ALERT_KEY_PREFIX}{existing}")
            if raw_alert:
                alert = json.loads(raw_alert)
                if not alert.get("resolved", False):
                    continue
        alert_id = f"alert-{uuid.uuid4().hex[:8]}"
        alert = {
            "id": alert_id,
            "level": "high" if kind in ["cpu", "memory", "disk"] else "medium",
            "title": title,
            "description": f"{key}={value:.2f} 超过阈值 {float(threshold):.2f}",
            "timestamp": utc_now().isoformat(),
            "resolved": False,
        }
        await redis_client.set(f"{_ALERT_KEY_PREFIX}{alert_id}", json.dumps(alert, ensure_ascii=False))
        await redis_client.zadd(_ALERT_INDEX_KEY, {alert_id: now_score})
        await redis_client.setex(dedup_key, 3600, alert_id)

    ids: List[str] = []
    total = 0
    if level:
        all_ids = await redis_client.zrevrange(_ALERT_INDEX_KEY, 0, -1)
        for alert_id in all_ids:
            raw_alert = await redis_client.get(f"{_ALERT_KEY_PREFIX}{alert_id}")
            if not raw_alert:
                continue
            alert = json.loads(raw_alert)
            if alert.get("level") == level:
                ids.append(alert_id)
        total = len(ids)
        ids = ids[offset: offset + limit]
    else:
        total = await redis_client.zcard(_ALERT_INDEX_KEY)
        ids = await redis_client.zrevrange(_ALERT_INDEX_KEY, offset, offset + limit - 1)

    alerts: List[Dict[str, Any]] = []
    for alert_id in ids:
        raw_alert = await redis_client.get(f"{_ALERT_KEY_PREFIX}{alert_id}")
        if raw_alert:
            alerts.append(json.loads(raw_alert))
    return {"alerts": alerts, "total": int(total)}

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, payload: Dict[str, Any]):
    """处理告警"""
    redis_client = _get_redis_client()
    raw = await redis_client.get(f"{_ALERT_KEY_PREFIX}{alert_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="alert_id不存在")
    alert = json.loads(raw)
    if alert.get("resolved", False):
        return {"success": True}
    alert["resolved"] = True
    alert["resolved_at"] = utc_now().isoformat()
    resolution = payload.get("resolution")
    if resolution:
        alert["resolution"] = str(resolution)
    await redis_client.set(f"{_ALERT_KEY_PREFIX}{alert_id}", json.dumps(alert, ensure_ascii=False))
    await audit_logger.log_security_event(
        event_type="enterprise_alert_resolved",
        details={"alert_id": alert_id},
        level="info",
    )
    return {"success": True}

@router.get("/audit/logs")
async def get_audit_logs(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取审计日志"""
    start_dt = _parse_datetime(start_time) if start_time else None
    end_dt = _parse_datetime(end_time) if end_time else None
    async with get_db_session() as session:
        table = await session.execute(text("SELECT to_regclass('public.audit_logs')"))
        if not table.scalar():
            return {"logs": [], "total": 0}
        where = []
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_dt:
            where.append("timestamp >= :start_time")
            params["start_time"] = start_dt
        if end_dt:
            where.append("timestamp <= :end_time")
            params["end_time"] = end_dt
        if user_id:
            where.append("user_id = :user_id")
            params["user_id"] = user_id
        if event_type:
            where.append("event_type LIKE :event_type")
            params["event_type"] = f"{event_type}%"
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        total_row = await session.execute(text("SELECT COUNT(*) FROM audit_logs " + where_sql), params)
        total = int(total_row.scalar() or 0)
        query_sql = (
            "SELECT id, event_type, timestamp, user_id, action, result, details\n"
            "FROM audit_logs\n"
            + where_sql
            + "\nORDER BY timestamp DESC\n"
            "LIMIT :limit OFFSET :offset"
        )
        rows = await session.execute(
            text(query_sql),
            params,
        )
        logs: List[Dict[str, Any]] = []
        for row in rows.mappings():
            details = row.get("details")
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {}
            logs.append(
                {
                    "event_id": row.get("id"),
                    "event_type": row.get("event_type"),
                    "timestamp": row.get("timestamp").isoformat() if row.get("timestamp") else utc_now().isoformat(),
                    "user_id": row.get("user_id"),
                    "action": row.get("action") or "",
                    "result": row.get("result") or "",
                    "details": details or {},
                }
            )
        return {"logs": logs, "total": total}

@router.get("/compliance/export")
async def export_compliance_report(format: str = Query("pdf")):
    """导出合规报告"""
    framework = get_compliance_framework()
    if not framework:
        raise HTTPException(status_code=503, detail="合规框架未初始化")
    report = framework.get_latest_report()
    if not report:
        report = await framework.run_compliance_assessment()
    filename = f"compliance_report_{utc_now().strftime('%Y%m%d_%H%M%S')}.{format}"
    if format == "json":
        payload = {
            "overall_score": report.overall_score,
            "status": report.status.value,
            "standards": [s.value for s in report.standards],
            "generated_at": report.generated_at.isoformat(),
            "test_results": [r.to_dict() for r in report.test_results],
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    if format == "excel":
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Compliance"
        ws.append(["overall_score", report.overall_score])
        ws.append(["status", report.status.value])
        ws.append(["generated_at", report.generated_at.isoformat()])
        ws.append([])
        ws.append(["requirement_id", "title", "status", "score", "last_tested"])
        for r in report.test_results:
            requirement = framework.get_requirement(r.requirement_id)
            title = requirement.title if requirement else r.test_name
            ws.append([
                r.requirement_id,
                title,
                r.status.value,
                r.score,
                r.timestamp.isoformat() if r.timestamp else "",
            ])
        buf = BytesIO()
        wb.save(buf)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    if format == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        y = height - 40
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "Compliance Report")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Status: {report.status.value}")
        y -= 16
        c.drawString(40, y, f"Overall score: {report.overall_score}")
        y -= 16
        c.drawString(40, y, f"Generated at: {report.generated_at.isoformat()}")
        y -= 24
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "Results")
        y -= 16
        c.setFont("Helvetica", 8)
        for r in report.test_results[:50]:
            requirement = framework.get_requirement(r.requirement_id)
            title = requirement.title if requirement else r.test_name
            line = f"{r.requirement_id} | {r.status.value} | {r.score:.2f} | {title}"
            c.drawString(40, y, line[:120])
            y -= 12
            if y < 40:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 8)
        c.save()
        return Response(
            content=buf.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    raise HTTPException(status_code=400, detail="format仅支持pdf/excel/json")

@router.get("/topology")
async def get_architecture_topology():
    """获取架构拓扑"""
    try:
        import psutil
    except Exception:
        raise HTTPException(status_code=503, detail="监控依赖不可用")
    db_ok = await test_database_connection()
    redis_ok = await test_redis_connection()
    qdrant_ok = await qdrant_manager.health_check()
    perf_stats = await monitoring_service.performance_monitor.get_stats()
    nodes = [
        {
            "id": "api",
            "type": "service",
            "name": "API服务",
            "status": "healthy",
            "metrics": {
                "cpu": float(psutil.cpu_percent()),
                "memory": float(psutil.virtual_memory().percent),
                "error_rate": float(perf_stats.get("error_rate", 0.0)) * 100.0,
            },
        },
        {"id": "postgres", "type": "database", "name": "Postgres", "status": "healthy" if db_ok else "unhealthy", "metrics": {}},
        {"id": "redis", "type": "cache", "name": "Redis", "status": "healthy" if redis_ok else "unhealthy", "metrics": {}},
        {"id": "qdrant", "type": "vector_db", "name": "Qdrant", "status": "healthy" if qdrant_ok else "unhealthy", "metrics": {}},
    ]
    edges = [
        {"source": "api", "target": "postgres", "type": "data_flow", "metrics": {}},
        {"source": "api", "target": "redis", "type": "data_flow", "metrics": {}},
        {"source": "api", "target": "qdrant", "type": "data_flow", "metrics": {}},
    ]
    return {"nodes": nodes, "edges": edges}

@router.post("/test/connections")
async def test_connections():
    """测试企业组件连接"""
    results: List[Dict[str, Any]] = []

    async def check(name: str, func):
        start = time.perf_counter()
        try:
            await func()
            results.append({"name": name, "status": "healthy", "response_time": (time.perf_counter() - start) * 1000})
        except Exception as e:
            results.append(
                {
                    "name": name,
                    "status": "unhealthy",
                    "response_time": (time.perf_counter() - start) * 1000,
                    "error": str(e),
                }
            )

    async def check_db():
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))

    async def check_redis():
        redis_client = _get_redis_client()
        await redis_client.ping()

    async def check_qdrant():
        ok = await qdrant_manager.health_check()
        if not ok:
            raise RuntimeError("Qdrant连接失败")

    async def check_trism():
        get_trism_framework()

    async def check_attack_detector():
        get_attack_detector()

    async def check_response_manager():
        get_response_manager().get_statistics()

    async def check_performance_optimizer():
        optimizer = await _ensure_performance_optimizer_started()
        optimizer.get_performance_overview()

    async def check_compliance_framework():
        framework = get_compliance_framework()
        if not framework:
            raise RuntimeError("合规框架未初始化")
        framework.get_latest_report()

    await check("Postgres", check_db)
    await check("Redis", check_redis)
    await check("Qdrant", check_qdrant)
    await check("TRiSM框架", check_trism)
    await check("攻击检测器", check_attack_detector)
    await check("安全响应管理器", check_response_manager)
    await check("性能优化器", check_performance_optimizer)
    await check("合规框架", check_compliance_framework)

    overall_status = "healthy"
    if any(c["status"] == "unhealthy" for c in results):
        overall_status = "unhealthy"
    elif any(c["status"] == "degraded" for c in results):
        overall_status = "degraded"
    return {"components": results, "overall_status": overall_status}

@router.get("/performance/recommendations")
async def get_performance_recommendations():
    """获取性能优化建议"""
    try:
        import psutil
    except Exception:
        raise HTTPException(status_code=503, detail="监控依赖不可用")
    optimizer = await _ensure_performance_optimizer_started()
    perf_stats = await monitoring_service.performance_monitor.get_stats()
    cpu = float(psutil.cpu_percent())
    mem = float(psutil.virtual_memory().percent)
    avg_rt = float(perf_stats.get("average_response_time_ms", 0.0))
    err = float(perf_stats.get("error_rate", 0.0))
    cache_hit = float(optimizer.cache.get_stats().get("hit_rate", 0.0))

    recs: List[Dict[str, str]] = []
    if cpu > 80:
        recs.append({"type": "cpu", "description": "CPU使用率偏高，建议降低并发或扩容", "impact": "high", "effort": "low"})
    if mem > 85:
        recs.append({"type": "memory", "description": "内存使用率偏高，建议优化缓存或限制任务并发", "impact": "high", "effort": "medium"})
    if avg_rt > 1000:
        recs.append({"type": "latency", "description": "平均响应时间偏高，建议检查慢请求并优化缓存", "impact": "medium", "effort": "medium"})
    if err > 0.05:
        recs.append({"type": "errors", "description": "错误率偏高，建议检查错误日志与依赖健康状态", "impact": "high", "effort": "medium"})
    if cache_hit < 0.5:
        recs.append({"type": "cache", "description": "缓存命中率偏低，建议调整缓存大小或预热关键数据", "impact": "medium", "effort": "low"})
    return {"recommendations": recs}

@router.post("/performance/optimize")
async def apply_performance_optimization(request: PerformanceOptimizationRequest):
    """应用性能优化"""
    optimizer = await _ensure_performance_optimizer_started()
    applied: List[str] = []
    failed: List[str] = []
    for opt in request.optimizations:
        try:
            if opt == "cpu":
                await optimizer._optimize_cpu_usage()
            elif opt == "memory":
                await optimizer._optimize_memory_usage()
            elif opt == "cache":
                await optimizer._optimize_cache()
            elif opt == "load_balancing":
                await optimizer._optimize_load_balancing()
            else:
                raise ValueError("未知优化类型")
            applied.append(opt)
        except Exception:
            failed.append(opt)
    await audit_logger.log_security_event(
        event_type="enterprise_performance_optimized",
        details={"applied": applied, "failed": failed},
        level="info",
    )
    return {"applied": applied, "failed": failed}

@router.get("/capacity/planning")
async def get_capacity_planning(time_range: str = Query("30d")):
    """获取容量规划建议"""
    seconds = _parse_time_range(time_range)
    redis_client = _get_redis_client()
    now = time.time()

    async def fetch(metric: str) -> List[Dict[str, Any]]:
        key = f"{_TREND_KEY_PREFIX}{metric}"
        raw_items = await redis_client.zrangebyscore(key, now - seconds, now)
        items: List[Dict[str, Any]] = []
        for raw in raw_items:
            try:
                items.append(json.loads(raw))
            except Exception:
                continue
        return items

    cpu_points = await fetch("cpu_usage")
    mem_points = await fetch("memory_usage")
    disk_points = await fetch("disk_usage")

    def current(points: List[Dict[str, Any]]) -> float:
        if not points:
            return 0.0
        return float(points[-1].get("value", 0.0))

    def projected(points: List[Dict[str, Any]]) -> float:
        if len(points) < 2:
            return current(points)
        a = float(points[0].get("value", 0.0))
        b = float(points[-1].get("value", 0.0))
        return max(0.0, min(100.0, b + (b - a)))

    current_usage = {"cpu": current(cpu_points), "memory": current(mem_points), "disk": current(disk_points)}
    projected_usage = {"cpu": projected(cpu_points), "memory": projected(mem_points), "disk": projected(disk_points)}
    recs: List[Dict[str, str]] = []
    if projected_usage["cpu"] > 80:
        recs.append({"resource": "cpu", "action": "scale_up", "timeline": time_range, "impact": "high"})
    if projected_usage["memory"] > 85:
        recs.append({"resource": "memory", "action": "scale_up", "timeline": time_range, "impact": "high"})
    if projected_usage["disk"] > 90:
        recs.append({"resource": "disk", "action": "cleanup_or_expand", "timeline": time_range, "impact": "high"})
    return {"current_usage": current_usage, "projected_usage": projected_usage, "recommendations": recs}

@router.get("/trends/{metric}")
async def get_trends(metric: str, time_range: str = Query("24h")):
    """获取历史趋势数据"""
    seconds = _parse_time_range(time_range)
    redis_client = _get_redis_client()
    now = time.time()
    raw_items = await redis_client.zrangebyscore(f"{_TREND_KEY_PREFIX}{metric}", now - seconds, now)
    points: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
            points.append({"timestamp": item.get("timestamp"), "value": float(item.get("value", 0.0))})
        except Exception:
            continue
    values = [p["value"] for p in points if p.get("timestamp")]
    if values:
        avg = sum(values) / len(values)
        delta = values[-1] - values[0]
        if abs(delta) < max(0.01, (max(values) - min(values)) * 0.05):
            trend = "stable"
        else:
            trend = "increasing" if delta > 0 else "decreasing"
        summary = {"min": min(values), "max": max(values), "avg": avg, "trend": trend}
    else:
        summary = {"min": 0.0, "max": 0.0, "avg": 0.0, "trend": "stable"}
    return {"metric": metric, "time_range": time_range, "data_points": points, "summary": summary}
