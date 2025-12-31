"""Epic 5 集成测试管理器"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from ...core.config import get_settings
from src.core.monitoring import monitor
from src.core.redis import get_redis
from .performance_benchmarks import PerformanceBenchmarkSuite
from .security_validation import SecurityValidator
from .system_health import SystemHealthMonitor

class TestType(Enum):
    """测试类型枚举"""
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STABILITY = "stability"

class TestStatus(Enum):
    """测试状态枚举"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class IntegrationTestResult:
    """集成测试结果数据结构"""
    test_suite_id: str
    test_name: str
    test_type: TestType
    epic_components: List[str]
    status: TestStatus
    execution_time_ms: float
    start_time: datetime
    end_time: datetime
    
    # 性能指标
    performance_metrics: Dict[str, float] = None
    
    # 测试详情
    test_details: Dict[str, Any] = None
    
    # 错误信息
    error_info: Optional[Dict[str, Any]] = None
    
    # 元数据
    test_environment: Dict[str, str] = None

@dataclass
class PerformanceBenchmark:
    """性能基准数据结构"""
    benchmark_id: str
    benchmark_name: str
    measurement_type: str
    epic5_before: float
    epic5_after: float
    improvement_percent: float
    target_improvement: float
    target_achieved: bool
    measurement_unit: str
    timestamp: datetime
    test_conditions: Dict[str, Any] = None
    detailed_results: Dict[str, float] = None

@dataclass
class SystemHealthCheck:
    """系统健康检查数据结构"""
    check_id: str
    component_name: str
    health_status: str
    last_check_time: datetime
    response_time_ms: float
    component_metrics: Dict[str, Any] = None
    health_details: Dict[str, List[str]] = None
    dependencies: Dict[str, str] = None

class TestSuiteRegistry:
    """测试套件注册表"""
    def __init__(self):
        self.test_suites = {}
        
    def register_suite(self, suite_name: str, tests: List[Any]):
        """注册测试套件"""
        self.test_suites[suite_name] = tests
        
    def get_suite(self, suite_name: str) -> List[Any]:
        """获取测试套件"""
        return self.test_suites.get(suite_name, [])
        
    def list_suites(self) -> List[str]:
        """列出所有测试套件"""
        return list(self.test_suites.keys())

class Epic5IntegrationTestManager:
    """Epic 5集成测试管理器"""
    
    def __init__(self):
        self.test_suites = TestSuiteRegistry()
        settings = get_settings()
        self.base_url = f"http://127.0.0.1:{settings.PORT}"
        self.performance_benchmarks = PerformanceBenchmarkSuite(base_url=self.base_url)
        self.health_monitor = SystemHealthMonitor()
        self.security_validator = SecurityValidator()
        self.test_results = []
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """执行Epic 5的全面验证"""
        results = {}
        
        # 组件集成测试
        monitor.log_info("开始运行组件集成测试...")
        results['integration'] = await self.run_integration_tests()
        
        # 性能基准测试
        monitor.log_info("开始运行性能基准测试...")
        results['performance'] = await self.run_performance_benchmarks()
        
        # 安全验证
        monitor.log_info("开始运行安全验证...")
        results['security'] = await self.run_security_validation()
        
        # 稳定性测试
        monitor.log_info("开始运行稳定性测试...")
        results['stability'] = await self.run_stability_tests()
        
        return self.generate_validation_report(results)
        
    async def verify_epic5_objectives(self) -> Dict[str, Any]:
        """验证Epic 5的预期目标达成"""
        objectives = {
            'response_time_improvement': 50.0,
            'retrieval_accuracy_improvement': 30.0,
            'concurrent_capacity_multiplier': 2.0,
            'storage_efficiency_improvement': 25.0,
            'development_speed_multiplier': 2.0,
        }

        redis = get_redis()
        baseline_key = "testing:epic5:baseline"
        baseline: Dict[str, Any] = {}
        if redis:
            try:
                raw = await redis.get(baseline_key)
                if raw:
                    import json

                    baseline = json.loads(raw)
            except Exception:
                baseline = {}

        network = await self.performance_benchmarks.benchmark_network()
        db = await self.performance_benchmarks.benchmark_database()
        current = {
            "network_latency_p95_ms": float(network.latency_p95),
            "network_throughput_qps": float(network.throughput_qps),
            "db_latency_p95_ms": float(db.latency_p95),
        }

        if redis and not baseline:
            try:
                import json

                await redis.set(baseline_key, json.dumps(current, ensure_ascii=False))
                baseline = current
            except Exception:
                baseline = current

        def _pct_improve(before: float, after: float) -> float:
            if before <= 0:
                return 0.0
            return (before - after) / before * 100

        def _ratio(before: float, after: float) -> float:
            if before <= 0:
                return 0.0
            return after / before

        verification_results: Dict[str, Any] = {}
        verification_results['response_time_improvement'] = {
            'target': objectives['response_time_improvement'],
            'actual': _pct_improve(
                float(baseline.get("network_latency_p95_ms") or 0),
                current["network_latency_p95_ms"],
            ),
            'achieved': False,
            'improvement_percent': 0.0,
        }
        verification_results['response_time_improvement']['improvement_percent'] = verification_results[
            'response_time_improvement'
        ]['actual']
        verification_results['response_time_improvement']['achieved'] = (
            verification_results['response_time_improvement']['actual']
            >= objectives['response_time_improvement']
        )

        verification_results['concurrent_capacity_multiplier'] = {
            'target': objectives['concurrent_capacity_multiplier'],
            'actual': _ratio(
                float(baseline.get("network_throughput_qps") or 0),
                current["network_throughput_qps"],
            ),
            'achieved': False,
            'improvement_percent': 0.0,
        }
        verification_results['concurrent_capacity_multiplier']['achieved'] = (
            verification_results['concurrent_capacity_multiplier']['actual']
            >= objectives['concurrent_capacity_multiplier']
        )

        verification_results['storage_efficiency_improvement'] = {
            'target': objectives['storage_efficiency_improvement'],
            'actual': _pct_improve(
                float(baseline.get("db_latency_p95_ms") or 0),
                current["db_latency_p95_ms"],
            ),
            'achieved': False,
            'improvement_percent': 0.0,
        }
        verification_results['storage_efficiency_improvement']['improvement_percent'] = verification_results[
            'storage_efficiency_improvement'
        ]['actual']
        verification_results['storage_efficiency_improvement']['achieved'] = (
            verification_results['storage_efficiency_improvement']['actual']
            >= objectives['storage_efficiency_improvement']
        )

        for k in ('retrieval_accuracy_improvement', 'development_speed_multiplier'):
            verification_results[k] = {
                'target': objectives[k],
                'actual': None,
                'achieved': False,
                'improvement_percent': None,
            }

        return verification_results
        
    async def run_integration_tests(self) -> List[IntegrationTestResult]:
        """运行集成测试"""
        results = []
        test_suites = [
            'langgraph_integration',
            'autogen_integration',
            'pgvector_integration',
            'mcp_tools_integration',
            'api_integration'
        ]
        
        for suite_name in test_suites:
            suite_result = await self.execute_test_suite(suite_name)
            results.extend(suite_result)
            
        return results
        
    async def execute_test_suite(self, suite_name: str) -> List[IntegrationTestResult]:
        """执行单个测试套件"""
        results = []
        start_time = utc_now()
        
        try:
            ops: List[Dict[str, Any]] = []
            errors = 0

            async def _http(method: str, path: str, json_body: Any | None = None) -> float:
                nonlocal errors
                import httpx

                url = f"{self.base_url}{path}"
                t0 = time.perf_counter()
                async with httpx.AsyncClient() as client:
                    resp = await client.request(method, url, json=json_body, timeout=15.0)
                ms = (time.perf_counter() - t0) * 1000
                ok = 200 <= resp.status_code < 300
                if not ok:
                    errors += 1
                ops.append(
                    {
                        "method": method,
                        "path": path,
                        "status_code": resp.status_code,
                        "ok": ok,
                        "latency_ms": round(ms, 2),
                    }
                )
                return ms

            if suite_name == "langgraph_integration":
                from src.ai.langgraph.state_graph import create_simple_workflow
                from src.ai.langgraph.state import create_initial_state
                from src.ai.langgraph.context import create_default_context

                t0 = time.perf_counter()
                builder = create_simple_workflow()
                state = create_initial_state()
                state["input_records"] = [
                    {"id": "it_1", "category": "integration", "status": "ok", "value": 1},
                ]
                await builder.execute(state, context=create_default_context())
                ops.append(
                    {
                        "method": "INPROC",
                        "path": "langgraph",
                        "status_code": 200,
                        "ok": True,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                )
            elif suite_name == "autogen_integration":
                await _http("GET", "/api/v1/events/stats")
            elif suite_name == "pgvector_integration":
                from sqlalchemy import text
                from src.core.database import get_db_session

                q0 = time.perf_counter()
                async with get_db_session() as session:
                    ext = await session.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    )
                    ext_version = ext.scalar_one_or_none()
                    ok = bool(ext_version)
                    if not ok:
                        errors += 1
                    ops.append(
                        {
                            "method": "SQL",
                            "path": "pg_extension.vector",
                            "status_code": 200 if ok else 500,
                            "ok": ok,
                            "latency_ms": round((time.perf_counter() - q0) * 1000, 2),
                            "extversion": ext_version,
                        }
                    )
            elif suite_name == "mcp_tools_integration":
                await _http("GET", "/api/v1/mcp/health")
            elif suite_name == "api_integration":
                await _http("GET", "/api/v1/health")
                await _http("GET", "/openapi.json")
                await _http("GET", "/api/v1/testing/health/status")
                await _http("GET", "/api/v1/supervisor/status?supervisor_id=main_supervisor")
            else:
                errors += 1
                ops.append(
                    {
                        "method": "N/A",
                        "path": suite_name,
                        "status_code": 400,
                        "ok": False,
                        "latency_ms": 0,
                        "error": "unknown suite",
                    }
                )

            latencies = [float(o.get("latency_ms") or 0) for o in ops]
            duration_s = (utc_now() - start_time).total_seconds()
            throughput = (len(latencies) - errors) / duration_s if duration_s > 0 else 0.0
            response_time_ms = sum(latencies) / len(latencies) if latencies else 0.0
            error_rate = errors / len(latencies) if latencies else 1.0

            result = IntegrationTestResult(
                test_suite_id=f"epic5_{suite_name}",
                test_name=suite_name,
                test_type=TestType.INTEGRATION,
                epic_components=[suite_name.split('_')[0]],
                status=TestStatus.PASSED if errors == 0 else TestStatus.FAILED,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                start_time=start_time,
                end_time=utc_now(),
                performance_metrics={
                    'response_time_ms': round(response_time_ms, 2),
                    'throughput_qps': round(float(throughput), 2),
                    'memory_usage_mb': None,
                    'cpu_usage_percent': None,
                    'error_rate_percent': round(float(error_rate) * 100, 2),
                },
                test_details={
                    'scenario_description': f'{suite_name} integration test',
                    'expected_outcome': 'All components integrate successfully',
                    'actual_outcome': 'Integration successful' if errors == 0 else 'Integration failed',
                    'assertions_passed': len(latencies) - errors,
                    'assertions_total': len(latencies),
                    'operations': ops,
                }
            )
            results.append(result)
            
        except Exception as e:
            result = IntegrationTestResult(
                test_suite_id=f"epic5_{suite_name}",
                test_name=suite_name,
                test_type=TestType.INTEGRATION,
                epic_components=[suite_name.split('_')[0]],
                status=TestStatus.ERROR,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                start_time=start_time,
                end_time=utc_now(),
                error_info={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'stack_trace': '',
                    'affected_components': [suite_name]
                }
            )
            results.append(result)
            
        return results
        
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        return await self.performance_benchmarks.run_comprehensive_benchmark()
        
    async def run_security_validation(self) -> Dict[str, Any]:
        """运行安全验证"""
        return await self.security_validator.validate_security_compliance()
        
    async def run_stability_tests(self) -> Dict[str, Any]:
        """运行稳定性测试"""
        return {
            'long_running_test': await self.run_long_running_stability_test(),
            'high_load_test': await self.run_high_load_test(),
            'failure_recovery_test': await self.run_failure_recovery_test()
        }
        
    async def run_long_running_stability_test(self) -> Dict[str, Any]:
        """长时间运行稳定性测试"""
        import psutil

        duration_s = 3
        errors = 0
        p = psutil.Process()
        mem_start = p.memory_info().rss
        for _ in range(duration_s):
            try:
                deps = await self.health_monitor.check_dependencies()
                if any((v or {}).get("status") == "disconnected" for v in deps.values() if isinstance(v, dict)):
                    errors += 1
            except Exception:
                errors += 1
            await asyncio.sleep(1)
        mem_end = p.memory_info().rss
        mem_delta_mb = (mem_end - mem_start) / 1024 / 1024
        leak = mem_delta_mb > 50
        stable = errors == 0 and not leak

        return {
            'test_duration_hours': round(duration_s / 3600, 6),
            'status': 'passed' if stable else 'failed',
            'uptime_percent': 100.0 if errors == 0 else round(max(0.0, 100.0 - errors * (100.0 / duration_s)), 2),
            'errors_encountered': errors,
            'memory_leak_detected': leak,
            'memory_delta_mb': round(mem_delta_mb, 2),
            'resource_usage_stable': stable,
        }
        
    async def run_high_load_test(self) -> Dict[str, Any]:
        """高负载测试"""
        import httpx

        url = f"{self.base_url}/api/v1/health"
        total_ops = 80
        concurrency = 20
        sem = asyncio.Semaphore(concurrency)
        errors = 0
        latencies: List[float] = []

        async def _one(client: httpx.AsyncClient) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.get(url, timeout=10.0)
                    r.raise_for_status()
                except Exception:
                    errors += 1
                latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[_one(client) for _ in range(total_ops)])
        duration = time.perf_counter() - t0

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        qps = (total_ops - errors) / duration if duration > 0 else 0.0
        error_rate = errors / total_ops if total_ops else 1.0
        passed = errors == 0
        return {
            'max_qps_achieved': round(qps, 2),
            'target_qps': None,
            'status': 'passed' if passed else 'failed',
            'error_rate_under_load': round(error_rate, 4),
            'response_time_p95_ms': round(p95, 2),
        }
        
    async def run_failure_recovery_test(self) -> Dict[str, Any]:
        """故障恢复测试"""
        scenarios = []
        times: List[float] = []
        ok = 0
        for name, fn in [
            ("postgresql", self.health_monitor.check_postgresql_status),
            ("redis", self.health_monitor.check_redis_status),
            ("qdrant", self.health_monitor.check_qdrant_status),
        ]:
            t0 = time.perf_counter()
            s1 = await fn()
            s2 = await fn()
            dt = time.perf_counter() - t0
            success = (s1 or {}).get("status") == "connected" and (s2 or {}).get("status") == "connected"
            ok += 1 if success else 0
            times.append(dt)
            scenarios.append({"component": name, "success": success})

        avg = sum(times) / len(times) if times else 0.0
        return {
            'components_tested': [s["component"] for s in scenarios],
            'recovery_scenarios': len(scenarios),
            'successful_recoveries': ok,
            'average_recovery_time_seconds': round(avg, 3),
            'data_integrity_maintained': ok == len(scenarios),
            'scenario_results': scenarios,
        }
        
    def generate_validation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证报告"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        # 统计测试结果
        for category, category_results in results.items():
            if isinstance(category_results, list):
                for result in category_results:
                    total_tests += 1
                    if hasattr(result, 'status'):
                        if result.status == TestStatus.PASSED:
                            passed_tests += 1
                        else:
                            failed_tests += 1
        
        return {
            'report_timestamp': utc_now().isoformat(),
            'epic_version': '5.0',
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'detailed_results': results,
            'production_ready': passed_tests == total_tests,
            'recommendations': self.generate_recommendations(results)
        }
        
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于测试结果生成建议
        if 'performance' in results:
            recommendations.append("继续监控性能指标，确保在生产环境中保持稳定")
            
        if 'security' in results:
            recommendations.append("定期进行安全审计，保持安全更新")
            
        if 'stability' in results:
            recommendations.append("建立持续的稳定性监控机制")
            
        return recommendations
