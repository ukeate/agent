"""平台集成API端点"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Any, Optional
import logging

from ai.platform_integration.integrator import PlatformIntegrator
from ai.platform_integration.optimizer import PerformanceOptimizer
from ai.platform_integration.monitoring import MonitoringSystem
from ai.platform_integration.documentation import DocumentationGenerator
from ai.platform_integration.models import (
    ComponentRegistration,
    WorkflowRequest,
    PlatformHealthStatus,
    PerformanceMetrics,
    MonitoringConfig
)

logger = logging.getLogger(__name__)

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
        # 这里应该实现工作流取消逻辑
        # 目前返回模拟响应
        
        logger.info(f"Workflow {workflow_id} cancellation requested")
        
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Workflow cancellation requested"
        }
        
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflows")
async def list_workflows(
    status: Optional[str] = Query(None, description="按状态过滤"),
    workflow_type: Optional[str] = Query(None, description="按类型过滤"),
    limit: int = Query(10, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
):
    """列出工作流"""
    try:
        # 这里应该从数据库或Redis中查询工作流列表
        # 目前返回模拟数据
        
        workflows = [
            {
                "workflow_id": "workflow_1234567890",
                "workflow_type": "full_fine_tuning",
                "status": "completed",
                "started_at": "2025-01-15T10:00:00Z",
                "completed_at": "2025-01-15T12:30:00Z"
            },
            {
                "workflow_id": "workflow_1234567891",
                "workflow_type": "model_optimization", 
                "status": "running",
                "started_at": "2025-01-15T14:00:00Z",
                "completed_at": None
            }
        ]
        
        # 应用过滤条件
        if status:
            workflows = [w for w in workflows if w["status"] == status]
        if workflow_type:
            workflows = [w for w in workflows if w["workflow_type"] == workflow_type]
        
        # 应用分页
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
        # 这里应该检查实际的生成状态
        # 目前返回模拟状态
        
        return {
            "status": "success",
            "generation_status": "completed",
            "last_generated": "2025-01-15T10:00:00Z",
            "available_documents": [
                "user_guide.md",
                "api_documentation.json", 
                "developer_guide.md",
                "deployment_guide.md",
                "troubleshooting_guide.md",
                "architecture_documentation.md"
            ]
        }
        
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
        
        # 工作流统计（模拟数据）
        workflow_stats = {
            "total_executed": 156,
            "currently_running": 3,
            "success_rate": 0.94,
            "avg_duration_minutes": 45
        }
        
        return {
            "status": "success",
            "stats": {
                "components": component_stats,
                "workflows": workflow_stats,
                "uptime_hours": 72.5,
                "last_restart": "2025-01-12T14:30:00Z"
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
    try:
        result = await doc_generator.generate_complete_documentation()
        logger.info(f"Documentation generation completed: {result['status']}")
    except Exception as e:
        logger.error(f"Background documentation generation failed: {e}")


async def _generate_training_materials_background(doc_generator: DocumentationGenerator):
    """后台生成培训材料"""
    try:
        result = await doc_generator.generate_training_materials()
        logger.info(f"Training materials generation completed: {result['status']}")
    except Exception as e:
        logger.error(f"Background training materials generation failed: {e}")


# 导入Response类型
from fastapi import Response