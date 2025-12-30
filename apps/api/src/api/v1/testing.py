"""测试和验证API端点"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import Field
from fastapi.encoders import jsonable_encoder
from src.api.base_model import ApiBaseModel
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.testing.integration import (
    Epic5IntegrationTestManager,
    PerformanceBenchmarkSuite,
    LoadTestRunner,
    StressTestRunner,
    SystemHealthMonitor,
    SecurityValidator,
    PenetrationTester
)

router = APIRouter(prefix="/testing", tags=["testing"])

_INTEGRATION_RUN_KEY_PREFIX = "testing:integration:run:"
_BENCHMARK_RUN_KEY_PREFIX = "testing:benchmarks:run:"
_SECURITY_RUN_KEY_PREFIX = "testing:security:run:"
_APP_STARTED_AT = time.time()

def _integration_run_key(test_id: str) -> str:
    return f"{_INTEGRATION_RUN_KEY_PREFIX}{test_id}"

def _benchmark_run_key(run_id: str) -> str:
    return f"{_BENCHMARK_RUN_KEY_PREFIX}{run_id}"

def _security_run_key(run_id: str) -> str:
    return f"{_SECURITY_RUN_KEY_PREFIX}{run_id}"

async def _store_json(key: str, value: Any) -> None:
    redis_client = get_redis()
    if not redis_client:
        return
    await redis_client.set(key, json.dumps(jsonable_encoder(value), ensure_ascii=False))

async def _load_json(key: str) -> Any:
    redis_client = get_redis()
    if not redis_client:
        return None
    raw = await redis_client.get(key)
    if not raw:
        return None
    return json.loads(raw)

# 全局实例
test_manager = Epic5IntegrationTestManager()
benchmark_suite = PerformanceBenchmarkSuite()
load_tester = LoadTestRunner()
stress_tester = StressTestRunner()
health_monitor = SystemHealthMonitor()
security_validator = SecurityValidator()
penetration_tester = PenetrationTester()

# 请求和响应模型
class TestSuiteRequest(ApiBaseModel):
    """测试套件请求"""
    suite_name: str = Field(..., description="测试套件名称")
    test_types: List[str] = Field(default_factory=lambda: ["integration"], description="测试类型列表")
    async_execution: bool = Field(default=False, description="是否异步执行")

class BenchmarkRequest(ApiBaseModel):
    """性能基准测试请求"""
    benchmark_types: List[str] = Field(
        default_factory=lambda: ["cpu", "memory", "io", "network", "database"],
        description="基准测试类型"
    )
    compare_with_baseline: bool = Field(default=True, description="是否与基线对比")

class LoadTestRequest(ApiBaseModel):
    """负载测试请求"""
    target_qps: int = Field(default=1000, description="目标QPS")
    duration_seconds: int = Field(default=300, description="测试持续时间(秒)")
    max_concurrent_users: int = Field(default=1000, description="最大并发用户数")

class HealthCheckRequest(ApiBaseModel):
    """健康检查请求"""
    components: Optional[List[str]] = Field(default=None, description="要检查的组件列表")
    include_dependencies: bool = Field(default=True, description="是否包含依赖检查")
    production_readiness: bool = Field(default=False, description="是否检查生产就绪度")

class SecurityAuditRequest(ApiBaseModel):
    """安全审计请求"""
    audit_types: List[str] = Field(
        default_factory=lambda: ["owasp", "mcp", "api", "data"],
        description="审计类型"
    )
    run_penetration_test: bool = Field(default=False, description="是否运行渗透测试")

def _enum_value(v: Any) -> Any:
    return getattr(v, "value", v)

# 集成测试端点
@router.get("/integration/suites", summary="获取集成测试套件列表")
async def get_integration_test_suites() -> Dict[str, Any]:
    """获取可用的集成测试套件列表"""
    suites = test_manager.test_suites.list_suites()
    
    return {
        "status": "success",
        "suites": suites,
        "total": len(suites)
    }

@router.post("/integration/run", summary="执行集成测试")
async def run_integration_tests(
    request: TestSuiteRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行集成测试套件"""
    try:
        test_id = f"test-{uuid.uuid4().hex}"
        start_time = utc_now()

        if request.async_execution:
            # 异步执行
            await _store_json(
                _integration_run_key(test_id),
                {
                    "test_id": test_id,
                    "test_type": "integration",
                    "suite_name": request.suite_name,
                    "test_types": request.test_types,
                    "status": "running",
                    "start_time": start_time.isoformat(),
                },
            )

            async def _run_and_store():
                try:
                    results = await test_manager.run_comprehensive_validation()
                    await _store_json(
                        _integration_run_key(test_id),
                        {
                            "test_id": test_id,
                            "test_type": "integration",
                            "suite_name": request.suite_name,
                            "test_types": request.test_types,
                            "status": "completed",
                            "start_time": start_time.isoformat(),
                            "end_time": utc_now().isoformat(),
                            "results": results,
                        },
                    )
                except Exception as e:
                    await _store_json(
                        _integration_run_key(test_id),
                        {
                            "test_id": test_id,
                            "test_type": "integration",
                            "suite_name": request.suite_name,
                            "test_types": request.test_types,
                            "status": "failed",
                            "start_time": start_time.isoformat(),
                            "end_time": utc_now().isoformat(),
                            "error": str(e),
                        },
                    )

            background_tasks.add_task(_run_and_store)
            return {
                "status": "started",
                "message": "集成测试已在后台启动",
                "test_id": test_id,
                "suite_name": request.suite_name
            }
        else:
            # 同步执行
            results = await test_manager.run_comprehensive_validation()
            await _store_json(
                _integration_run_key(test_id),
                {
                    "test_id": test_id,
                    "test_type": "integration",
                    "suite_name": request.suite_name,
                    "test_types": request.test_types,
                    "status": "completed",
                    "start_time": start_time.isoformat(),
                    "end_time": utc_now().isoformat(),
                    "results": results,
                },
            )
            return {
                "status": "completed",
                "test_id": test_id,
                "results": results
            }
            
    except Exception as e:
        monitor.log_error(f"集成测试执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/integration/results/{test_id}", summary="获取测试结果")
async def get_integration_test_results(test_id: str) -> Dict[str, Any]:
    """获取特定测试的结果"""
    run = await _load_json(_integration_run_key(test_id))
    if not run:
        raise HTTPException(status_code=404, detail="测试结果不存在")
    return run

@router.get("/integration/reports", summary="获取测试报告")
async def get_integration_test_reports(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(10, ge=1, le=100)
) -> Dict[str, Any]:
    """获取集成测试报告"""
    redis_client = get_redis()
    if not redis_client:
        return {"status": "success", "reports": [], "total": 0}

    ids = await redis_client.keys(f"{_INTEGRATION_RUN_KEY_PREFIX}*")
    reports: List[Dict[str, Any]] = []
    for key in ids:
        run = await _load_json(key)
        if not run:
            continue
        try:
            ts = datetime.fromisoformat(run.get("start_time"))
        except Exception:
            ts = None
        if start_date and ts and ts < start_date:
            continue
        if end_date and ts and ts > end_date:
            continue
        reports.append(run)

    reports.sort(key=lambda r: r.get("start_time") or "", reverse=True)
    return {"status": "success", "reports": reports[:limit], "total": len(reports)}

@router.get("/status/running", summary="获取运行中的测试")
async def get_running_tests() -> List[Dict[str, Any]]:
    """获取所有运行中的测试（来源：Redis中的运行记录）"""
    redis_client = get_redis()
    if not redis_client:
        return []

    keys = await redis_client.keys(f"{_INTEGRATION_RUN_KEY_PREFIX}*")
    runs = []
    for key in keys:
        run = await _load_json(key)
        if not run or run.get("status") != "running":
            continue
        if "test_type" not in run:
            run["test_type"] = "integration"
        runs.append(run)

    runs.sort(key=lambda r: r.get("start_time") or "", reverse=True)
    return runs

@router.get("/health/status", summary="获取系统健康状态")
async def get_system_health_status() -> Dict[str, Any]:
    """面向前端的系统健康状态（无静态假数据）"""
    core = await get_core_health_status(detailed=False)
    core_components = core.get("components") or {}
    now = utc_now()

    components: List[Dict[str, Any]] = []
    for name, item in core_components.items():
        status_raw = _enum_value((item or {}).get("status"))
        status = "down" if status_raw == CoreHealthStatus.UNHEALTHY.value else status_raw
        response_time = (item or {}).get("response_time_ms")
        components.append(
            {
                "name": name,
                "status": status,
                "response_time": float(response_time) if response_time is not None else 0.0,
                "error_rate": 0.0,
                "last_check": now.isoformat(),
            }
        )

    qdrant_start = time.time()
    qdrant_status = "healthy"
    try:
        client = qdrant_manager.get_client()
        client.get_collections()
    except Exception:
        qdrant_status = "down"

    components.append(
        {
            "name": "qdrant",
            "status": qdrant_status,
            "response_time": round((time.time() - qdrant_start) * 1000, 2),
            "error_rate": 0.0,
            "last_check": now.isoformat(),
        }
    )

    overall = "healthy"
    if any(c["status"] == "down" for c in components):
        overall = "critical"
    elif any(c["status"] == CoreHealthStatus.DEGRADED.value for c in components):
        overall = "warning"

    uptime = None
    api_component = core_components.get("api") or {}
    api_uptime = api_component.get("uptime_seconds")
    if api_uptime is not None:
        uptime = int(api_uptime)
    else:
        uptime = int(time.time() - _APP_STARTED_AT)

    return {
        "overall_status": overall,
        "components": components,
        "last_updated": now.isoformat(),
        "uptime": uptime,
    }

@router.get("/coverage", summary="获取覆盖率报告")
async def get_coverage_report() -> Dict[str, Any]:
    """读取最新的coverage数据并返回汇总（无静态假数据）"""
    try:
        api_dir = Path(__file__).resolve().parents[3]
        project_root = (api_dir / "src").resolve()
        candidates = [api_dir / ".coverage", api_dir / "src" / ".coverage"]

        def _map_measured_file(raw: str) -> Path | None:
            s = str(raw)
            s_norm = s.replace("\\", "/")

            if s:
                p = Path(s)
                if not p.is_absolute():
                    p = api_dir / s
                if p.exists():
                    rp = p.resolve()
                    if rp == project_root or project_root in rp.parents:
                        return rp

            idx = s_norm.rfind("/src/")
            if idx != -1:
                rel = s_norm[idx + 1 :]
                p2 = (api_dir / rel).resolve()
                if p2.exists() and (p2 == project_root or project_root in p2.parents):
                    return p2

            return None

        from coverage import Coverage

        best_coverage_file = None
        best_cov = None
        best_files: list[Path] = []
        for p in candidates:
            if not p.exists():
                continue
            try:
                cov = Coverage(data_file=str(p))
                cov.load()
                data = cov.get_data()
                mapped = []
                for f in data.measured_files():
                    mp = _map_measured_file(str(f))
                    if mp:
                        mapped.append(mp)
                if best_coverage_file is None or len(mapped) > len(best_files) or (
                    len(mapped) == len(best_files) and p.stat().st_mtime > best_coverage_file.stat().st_mtime
                ):
                    best_coverage_file = p
                    best_cov = cov
                    best_files = mapped
            except Exception:
                continue

        if best_coverage_file is None or best_cov is None:
            return {"overall_percentage": 0.0, "total_files": 0, "files": []}

        total_statements = 0
        total_missing = 0
        files: List[Dict[str, Any]] = []
        for p in sorted(set(best_files)):
            try:
                _, statements, excluded, missing, _ = best_cov.analysis2(str(p))
            except Exception:
                continue

            statements_count = len(statements) - len(excluded)
            if statements_count <= 0:
                continue
            missing_count = len(missing)

            total_statements += statements_count
            total_missing += missing_count

            try:
                rel_path = str(p.relative_to(api_dir))
            except Exception:
                rel_path = str(p)

            percent = (statements_count - missing_count) / statements_count * 100
            files.append(
                {
                    "path": rel_path,
                    "coverage_percentage": round(percent, 2),
                    "statements": statements_count,
                    "missing": missing_count,
                }
            )

        overall = (total_statements - total_missing) / total_statements * 100 if total_statements else 0.0
        return {
            "coverage_file": str(best_coverage_file),
            "generated_at": datetime.fromtimestamp(best_coverage_file.stat().st_mtime, tz=timezone.utc).isoformat(),
            "overall_percentage": round(overall, 2),
            "total_files": len(files),
            "files": files,
        }
    except Exception as e:
        monitor.log_error(f"读取覆盖率失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 性能基准测试端点
@router.get("/benchmark/baseline", summary="获取性能基准数据")
async def get_performance_benchmarks() -> Dict[str, Any]:
    """获取最近一次基准测试结果（无静态假数据）"""
    redis_client = get_redis()
    if not redis_client:
        return {"status": "success", "baseline": None}

    keys = await redis_client.keys(f"{_BENCHMARK_RUN_KEY_PREFIX}*")
    latest = None
    for key in keys:
        run = await _load_json(key)
        if not run:
            continue
        ts = run.get("timestamp") or ""
        if latest is None or ts > (latest.get("timestamp") or ""):
            latest = run

    return {"status": "success", "baseline": latest}

@router.post("/benchmark/run", summary="执行性能基准测试")
async def run_performance_benchmarks(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行性能基准测试"""
    try:
        run_id = f"benchmark-{uuid.uuid4().hex}"
        results = await benchmark_suite.run_comprehensive_benchmark(request.benchmark_types)

        comparisons: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        if request.compare_with_baseline:
            redis_client = get_redis()
            baseline_benchmarks = None
            if redis_client:
                keys = await redis_client.keys(f"{_BENCHMARK_RUN_KEY_PREFIX}*")
                latest = None
                for key in keys:
                    run = await _load_json(key)
                    if not run or run.get("run_id") == run_id:
                        continue
                    ts = run.get("timestamp") or ""
                    if latest is None or ts > (latest.get("timestamp") or ""):
                        latest = run
                if latest:
                    baseline_benchmarks = (latest.get("results") or {}).get("benchmarks") or {}

            if baseline_benchmarks:
                from src.testing.integration.performance_benchmarks import BenchmarkMetrics
                raw_comparisons = {}
                for name, cur in (results.get("benchmarks") or {}).items():
                    base = baseline_benchmarks.get(name)
                    if not isinstance(cur, dict) or not isinstance(base, dict):
                        continue
                    try:
                        base_m = BenchmarkMetrics(**base)
                        cur_m = BenchmarkMetrics(**cur)
                    except Exception:
                        continue
                    raw_comparisons[name] = benchmark_suite.compare_metrics(base_m, cur_m)

                if raw_comparisons:
                    comparisons = {k: [jsonable_encoder(c) for c in v] for k, v in raw_comparisons.items()}
                    summary = benchmark_suite.generate_summary(raw_comparisons)

        if comparisons:
            results["comparisons"] = comparisons
        if summary:
            results["summary"] = summary

        await _store_json(
            _benchmark_run_key(run_id),
            {
                "run_id": run_id,
                "status": "completed",
                "timestamp": utc_now().isoformat(),
                "results": results,
            },
        )
        
        return {
            "status": "completed",
            "run_id": run_id,
            "results": results
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        monitor.log_error(f"性能基准测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/history", summary="获取基准测试历史")
async def get_benchmark_history(limit: int = Query(20, ge=1, le=200)) -> List[Dict[str, Any]]:
    """获取基准测试历史（来源：Redis）"""
    redis_client = get_redis()
    if not redis_client:
        return []

    keys = await redis_client.keys(f"{_BENCHMARK_RUN_KEY_PREFIX}*")
    runs = []
    for key in keys:
        run = await _load_json(key)
        if run:
            runs.append(run)

    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return runs[:limit]

@router.get("/benchmark/compare", summary="比较Epic 5前后性能")
async def compare_performance_benchmarks() -> Dict[str, Any]:
    """比较Epic 5升级前后的性能"""
    try:
        # 运行当前基准测试
        current_results = await benchmark_suite.run_comprehensive_benchmark()
        
        return {
            "status": "success",
            "comparison": current_results.get("comparisons", {}),
            "summary": current_results.get("summary", {}),
            "epic5_objectives_met": current_results.get("summary", {}).get("epic5_objectives_met", {})
        }
        
    except Exception as e:
        monitor.log_error(f"性能对比失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmark/trends", summary="获取性能趋势数据")
async def get_performance_trends(
    metric: str = Query("latency_p95", description="指标名称"),
    days: int = Query(7, ge=1, le=30, description="天数")
) -> Dict[str, Any]:
    """获取性能趋势数据"""
    redis_client = get_redis()
    if not redis_client:
        return {"status": "success", "metric": metric, "days": days, "points": []}

    keys = await redis_client.keys(f"{_BENCHMARK_RUN_KEY_PREFIX}*")
    points = []
    cutoff = utc_now().timestamp() - days * 86400
    for key in keys:
        run = await _load_json(key)
        if not run:
            continue
        ts_str = run.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            continue
        if ts < cutoff:
            continue
        results = (run.get("results") or {}).get("benchmarks") or {}
        values = []
        for m in results.values():
            if isinstance(m, dict) and metric in m:
                values.append(m[metric])
        if not values:
            continue
        points.append({"timestamp": ts_str, "value": sum(values) / len(values)})

    points.sort(key=lambda p: p["timestamp"])
    return {"status": "success", "metric": metric, "days": days, "points": points}

# 负载和压力测试端点
@router.post("/load/run", summary="运行负载测试")
async def run_load_test(
    request: LoadTestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """运行负载测试"""
    try:
        results = await load_tester.run_load_test(
            target_qps=request.target_qps,
            duration_seconds=request.duration_seconds,
            concurrency=request.max_concurrent_users,
        )
        
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        monitor.log_error(f"负载测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stress/run", summary="运行压力测试")
async def run_stress_test() -> Dict[str, Any]:
    """运行压力测试"""
    try:
        results = await stress_tester.run_stress_test()
        
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        monitor.log_error(f"压力测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 系统健康监控端点
@router.get("/health/comprehensive", summary="获取全面系统健康状态")
async def get_comprehensive_health() -> Dict[str, Any]:
    """获取系统的全面健康状态"""
    try:
        health_report = await health_monitor.run_comprehensive_health_check()
        
        return health_report
        
    except Exception as e:
        monitor.log_error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/components", summary="获取各组件健康状态")
async def get_component_health(
    components: Optional[List[str]] = Query(None, description="组件名称列表")
) -> Dict[str, Any]:
    """获取指定组件的健康状态"""
    try:
        health_report = await health_monitor.run_comprehensive_health_check()
        
        if components:
            filtered_components = {
                name: status
                for name, status in health_report["components"].items()
                if name in components
            }
            return {
                "status": "success",
                "components": filtered_components
            }
        else:
            return {
                "status": "success",
                "components": health_report["components"]
            }
            
    except Exception as e:
        monitor.log_error(f"组件健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health/check", summary="执行健康检查")
async def perform_health_check(request: HealthCheckRequest) -> Dict[str, Any]:
    """执行指定的健康检查"""
    try:
        if request.production_readiness:
            results = await health_monitor.run_production_readiness_check()
        else:
            results = await health_monitor.run_comprehensive_health_check()
            
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        monitor.log_error(f"健康检查执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/dependencies", summary="获取依赖服务状态")
async def get_dependency_status() -> Dict[str, Any]:
    """获取所有依赖服务的状态"""
    try:
        dependencies = await health_monitor.check_dependencies()
        
        return {
            "status": "success",
            "dependencies": dependencies
        }
        
    except Exception as e:
        monitor.log_error(f"依赖检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 生产就绪度评估端点
@router.get("/validation/readiness", summary="获取生产就绪度评估")
async def get_production_readiness() -> Dict[str, Any]:
    """获取系统的生产就绪度评估"""
    try:
        readiness = await health_monitor.run_production_readiness_check()
        
        return readiness
        
    except Exception as e:
        monitor.log_error(f"就绪度评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validation/epic5-verification", summary="执行Epic 5验证")
async def verify_epic5_objectives() -> Dict[str, Any]:
    """验证Epic 5的目标达成情况"""
    try:
        verification_results = await test_manager.verify_epic5_objectives()
        
        return {
            "status": "completed",
            "objectives": verification_results,
            "all_objectives_met": all(
                obj["achieved"] for obj in verification_results.values()
            )
        }
        
    except Exception as e:
        monitor.log_error(f"Epic 5验证失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validation/regression-tests", summary="获取回归测试结果")
async def get_regression_test_results() -> Dict[str, Any]:
    """获取回归测试结果"""
    redis_client = get_redis()
    if not redis_client:
        return {"status": "success", "latest": None}
    keys = await redis_client.keys(f"{_INTEGRATION_RUN_KEY_PREFIX}*")
    if not keys:
        return {"status": "success", "latest": None}
    runs = []
    for key in keys:
        run = await _load_json(key)
        if run:
            runs.append(run)
    runs.sort(key=lambda r: r.get("start_time") or "", reverse=True)
    return {"status": "success", "latest": runs[0]}

# 安全审计端点
@router.get("/validation/security-audit", summary="获取安全审计结果")
async def get_security_audit_results() -> Dict[str, Any]:
    """获取最新的安全审计结果"""
    try:
        audit_results = await security_validator.validate_security_compliance()
        
        return audit_results
        
    except Exception as e:
        monitor.log_error(f"安全审计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/security/scan", summary="执行安全扫描")
async def run_security_scan(request: SecurityAuditRequest) -> Dict[str, Any]:
    """执行安全扫描并保存结果（来源：真实检查流程）"""
    try:
        run_id = f"security-{uuid.uuid4().hex}"
        started_at = utc_now()
        results = await security_validator.validate_security_compliance()
        ended_at = utc_now()

        record = {
            "scan_id": run_id,
            "status": "completed",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_ms": int((ended_at - started_at).total_seconds() * 1000),
            "results": results,
        }
        await _store_json(_security_run_key(run_id), record)
        return record
    except Exception as e:
        monitor.log_error(f"安全扫描失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security/results/{scan_id}", summary="获取安全扫描结果")
async def get_security_scan_result(scan_id: str) -> Dict[str, Any]:
    record = await _load_json(_security_run_key(scan_id))
    if not record:
        raise HTTPException(status_code=404, detail="扫描结果不存在")
    return record

@router.get("/security/history", summary="获取安全扫描历史")
async def get_security_scan_history(limit: int = Query(20, ge=1, le=200)) -> List[Dict[str, Any]]:
    redis_client = get_redis()
    if not redis_client:
        return []

    keys = await redis_client.keys(f"{_SECURITY_RUN_KEY_PREFIX}*")
    runs = []
    for key in keys:
        run = await _load_json(key)
        if run:
            runs.append(run)
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return runs[:limit]

@router.post("/security/penetration", summary="执行渗透测试")
async def run_penetration_test() -> Dict[str, Any]:
    """执行渗透测试"""
    try:
        results = await penetration_tester.run_penetration_test()
        
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        monitor.log_error(f"渗透测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
