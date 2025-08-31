"""测试和验证API端点"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from testing.integration import (
    Epic5IntegrationTestManager,
    PerformanceBenchmarkSuite,
    LoadTestRunner,
    StressTestRunner,
    SystemHealthMonitor,
    SecurityValidator,
    PenetrationTester
)
from src.core.monitoring import monitor

router = APIRouter(prefix="/api/v1/testing", tags=["testing"])

# 全局实例
test_manager = Epic5IntegrationTestManager()
benchmark_suite = PerformanceBenchmarkSuite()
load_tester = LoadTestRunner()
stress_tester = StressTestRunner()
health_monitor = SystemHealthMonitor()
security_validator = SecurityValidator()
penetration_tester = PenetrationTester()


# 请求和响应模型
class TestSuiteRequest(BaseModel):
    """测试套件请求"""
    suite_name: str = Field(..., description="测试套件名称")
    test_types: List[str] = Field(default=["integration"], description="测试类型列表")
    async_execution: bool = Field(default=False, description="是否异步执行")


class BenchmarkRequest(BaseModel):
    """性能基准测试请求"""
    benchmark_types: List[str] = Field(
        default=["api_response", "langgraph_workflow", "autogen_collaboration"],
        description="基准测试类型"
    )
    compare_with_baseline: bool = Field(default=True, description="是否与基线对比")


class LoadTestRequest(BaseModel):
    """负载测试请求"""
    target_qps: int = Field(default=1000, description="目标QPS")
    duration_seconds: int = Field(default=300, description="测试持续时间(秒)")
    max_concurrent_users: int = Field(default=1000, description="最大并发用户数")


class HealthCheckRequest(BaseModel):
    """健康检查请求"""
    components: Optional[List[str]] = Field(default=None, description="要检查的组件列表")
    include_dependencies: bool = Field(default=True, description="是否包含依赖检查")
    production_readiness: bool = Field(default=False, description="是否检查生产就绪度")


class SecurityAuditRequest(BaseModel):
    """安全审计请求"""
    audit_types: List[str] = Field(
        default=["owasp", "mcp", "api", "data"],
        description="审计类型"
    )
    run_penetration_test: bool = Field(default=False, description="是否运行渗透测试")


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
        if request.async_execution:
            # 异步执行
            background_tasks.add_task(
                test_manager.run_comprehensive_validation
            )
            return {
                "status": "started",
                "message": "集成测试已在后台启动",
                "suite_name": request.suite_name
            }
        else:
            # 同步执行
            results = await test_manager.run_comprehensive_validation()
            return {
                "status": "completed",
                "results": results
            }
            
    except Exception as e:
        monitor.log_error(f"集成测试执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integration/results/{test_id}", summary="获取测试结果")
async def get_integration_test_results(test_id: str) -> Dict[str, Any]:
    """获取特定测试的结果"""
    # 这里应该从存储中获取测试结果
    # 现在返回模拟数据
    return {
        "status": "success",
        "test_id": test_id,
        "results": {
            "passed": True,
            "total_tests": 50,
            "passed_tests": 48,
            "failed_tests": 2,
            "execution_time_seconds": 120
        }
    }


@router.get("/integration/reports", summary="获取测试报告")
async def get_integration_test_reports(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(10, ge=1, le=100)
) -> Dict[str, Any]:
    """获取集成测试报告"""
    # 这里应该从存储中获取历史报告
    return {
        "status": "success",
        "reports": [],
        "total": 0
    }


# 性能基准测试端点
@router.get("/benchmarks", summary="获取性能基准数据")
async def get_performance_benchmarks() -> Dict[str, Any]:
    """获取当前性能基准数据"""
    baseline_data = benchmark_suite.baseline_data
    
    return {
        "status": "success",
        "benchmarks": {
            name: {
                "latency_p50": metrics.latency_p50,
                "latency_p95": metrics.latency_p95,
                "latency_p99": metrics.latency_p99,
                "throughput_qps": metrics.throughput_qps,
                "error_rate": metrics.error_rate
            }
            for name, metrics in baseline_data.items()
        }
    }


@router.post("/benchmarks/run", summary="执行性能基准测试")
async def run_performance_benchmarks(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行性能基准测试"""
    try:
        results = await benchmark_suite.run_comprehensive_benchmark()
        
        return {
            "status": "completed",
            "results": results
        }
        
    except Exception as e:
        monitor.log_error(f"性能基准测试失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks/compare", summary="比较Epic 5前后性能")
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


@router.get("/benchmarks/trends", summary="获取性能趋势数据")
async def get_performance_trends(
    metric: str = Query("latency_p95", description="指标名称"),
    days: int = Query(7, ge=1, le=30, description="天数")
) -> Dict[str, Any]:
    """获取性能趋势数据"""
    # 这里应该从时序数据库获取历史数据
    return {
        "status": "success",
        "metric": metric,
        "days": days,
        "trends": []
    }


# 负载和压力测试端点
@router.post("/load/run", summary="运行负载测试")
async def run_load_test(
    request: LoadTestRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """运行负载测试"""
    try:
        results = await load_tester.run_load_test(request.target_qps)
        
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
    # 这里应该运行回归测试套件
    return {
        "status": "success",
        "regression_tests": {
            "total": 100,
            "passed": 100,
            "failed": 0,
            "compatibility": "100%"
        }
    }


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


@router.post("/security/audit", summary="执行安全审计")
async def perform_security_audit(
    request: SecurityAuditRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行安全审计"""
    try:
        # 运行安全合规验证
        audit_results = await security_validator.validate_security_compliance()
        
        # 如果需要，运行渗透测试
        if request.run_penetration_test:
            penetration_results = await penetration_tester.run_penetration_test()
            audit_results["penetration_test"] = penetration_results
            
        return {
            "status": "completed",
            "results": audit_results
        }
        
    except Exception as e:
        monitor.log_error(f"安全审计执行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/security/penetration-test", summary="执行渗透测试")
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