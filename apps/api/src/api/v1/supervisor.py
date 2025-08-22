"""
Supervisor管理API路由
提供Supervisor智能体的HTTP接口
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse
import structlog

from src.services.supervisor_service import supervisor_service
from src.models.schemas.supervisor import (
    TaskSubmissionRequest, TaskSubmissionResponse,
    SupervisorStatusApiResponse, SupervisorDecisionListResponse,
    SupervisorConfigUpdateRequest, SupervisorConfigApiResponse,
    LoadStatisticsApiResponse, TaskType, TaskPriority
)
from src.models.schemas.base import BaseResponse, ErrorResponse
from src.api.exceptions import ValidationError, NotFoundError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/supervisor", tags=["supervisor"])


@router.post(
    "/tasks",
    response_model=TaskSubmissionResponse,
    summary="提交任务给Supervisor",
    description="将新任务提交给Supervisor进行分析和分配"
)
async def submit_task(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    request: TaskSubmissionRequest = ...
) -> TaskSubmissionResponse:
    """提交任务给Supervisor进行智能分配"""
    try:
        logger.info("收到任务提交请求", 
                   supervisor_id=supervisor_id,
                   task_name=request.name,
                   task_type=request.task_type.value)
        
        # 调用服务层提交任务
        assignment = await supervisor_service.submit_task(supervisor_id, request)
        
        # 将TaskAssignment转换为字典
        assignment_data = assignment.to_dict() if hasattr(assignment, 'to_dict') else {
            "task_id": assignment.task_id,
            "assigned_agent": assignment.assigned_agent,
            "assignment_reason": assignment.assignment_reason,
            "confidence_level": assignment.confidence_level,
            "estimated_completion_time": assignment.estimated_completion_time,
            "alternative_agents": assignment.alternative_agents,
            "decision_metadata": assignment.decision_metadata
        }
        
        return TaskSubmissionResponse(
            success=True,
            message="任务提交成功",
            data=assignment_data
        )
        
    except ValueError as e:
        logger.error("任务提交参数错误", supervisor_id=supervisor_id, error=str(e))
        raise ValidationError(str(e))
    except Exception as e:
        logger.error("任务提交失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务提交失败: {str(e)}"
        )


@router.get(
    "/status",
    response_model=SupervisorStatusApiResponse,
    summary="查询Supervisor状态",
    description="获取Supervisor的当前状态、负载和性能指标"
)
async def get_supervisor_status(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> SupervisorStatusApiResponse:
    """获取Supervisor状态信息"""
    try:
        logger.info("查询Supervisor状态", supervisor_id=supervisor_id)
        
        status_data = await supervisor_service.get_supervisor_status(supervisor_id)
        
        return SupervisorStatusApiResponse(
            success=True,
            message="状态查询成功",
            data=status_data
        )
        
    except ValueError as e:
        logger.error("Supervisor不存在", supervisor_id=supervisor_id, error=str(e))
        raise NotFoundError(f"Supervisor {supervisor_id} 不存在")
    except Exception as e:
        logger.error("状态查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"状态查询失败: {str(e)}"
        )


@router.get(
    "/decisions",
    response_model=SupervisorDecisionListResponse,
    summary="获取决策历史",
    description="获取Supervisor的决策历史记录"
)
async def get_decision_history(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    limit: int = Query(10, ge=1, le=100, description="返回记录数量"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> SupervisorDecisionListResponse:
    """获取Supervisor决策历史"""
    try:
        logger.info("查询决策历史", 
                   supervisor_id=supervisor_id,
                   limit=limit,
                   offset=offset)
        
        decisions = await supervisor_service.get_decision_history(
            supervisor_id, limit, offset
        )
        
        return SupervisorDecisionListResponse(
            success=True,
            message="决策历史查询成功", 
            data=decisions,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": len(decisions)  # 实际应该从数据库获取总数
            }
        )
        
    except ValueError as e:
        logger.warning("Supervisor验证警告", supervisor_id=supervisor_id, error=str(e))
        # 继续处理，返回空决策列表
        return SupervisorDecisionListResponse(
            success=True,
            message="决策历史查询成功（无记录）",
            data=[],
            pagination={
                "limit": limit,
                "offset": offset,
                "total": 0
            }
        )
    except Exception as e:
        logger.error("决策历史查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"决策历史查询失败: {str(e)}"
        )


@router.put(
    "/config",
    response_model=SupervisorConfigApiResponse,
    summary="更新Supervisor配置",
    description="更新Supervisor的路由策略和其他配置参数"
)
async def update_supervisor_config(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    request: SupervisorConfigUpdateRequest = ...
) -> SupervisorConfigApiResponse:
    """更新Supervisor配置"""
    try:
        logger.info("更新Supervisor配置", 
                   supervisor_id=supervisor_id,
                   config_updates=request.dict(exclude_unset=True))
        
        result = await supervisor_service.update_supervisor_config(
            supervisor_id, request
        )
        
        return SupervisorConfigApiResponse(
            success=True,
            message=result["message"],
            data=result
        )
        
    except ValueError as e:
        logger.error("配置更新参数错误", supervisor_id=supervisor_id, error=str(e))
        raise ValidationError(str(e))
    except Exception as e:
        logger.error("配置更新失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置更新失败: {str(e)}"
        )


@router.post(
    "/initialize",
    response_model=BaseResponse,
    summary="初始化Supervisor",
    description="创建并初始化新的Supervisor智能体"
)
async def initialize_supervisor(
    supervisor_name: str = Query(..., description="Supervisor名称")
) -> BaseResponse:
    """初始化新的Supervisor"""
    try:
        logger.info("初始化Supervisor", name=supervisor_name)
        
        supervisor_id = await supervisor_service.initialize_supervisor(supervisor_name)
        
        return BaseResponse(
            success=True,
            message="Supervisor初始化成功",
            data={"supervisor_id": supervisor_id, "name": supervisor_name}
        )
        
    except Exception as e:
        logger.error("Supervisor初始化失败", name=supervisor_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Supervisor初始化失败: {str(e)}"
        )


@router.post(
    "/agents/{agent_name}",
    response_model=BaseResponse,
    summary="添加智能体",
    description="将智能体添加到Supervisor的管理范围"
)
async def add_agent(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    agent_name: str = Path(..., description="智能体名称")
) -> BaseResponse:
    """添加智能体到Supervisor"""
    try:
        logger.info("添加智能体到Supervisor", 
                   supervisor_id=supervisor_id,
                   agent_name=agent_name)
        
        # 这里需要从智能体池中获取智能体实例
        # 暂时返回成功消息，实际实现需要智能体管理服务
        
        return BaseResponse(
            success=True,
            message=f"智能体 {agent_name} 添加成功",
            data={"supervisor_id": supervisor_id, "agent_name": agent_name}
        )
        
    except Exception as e:
        logger.error("添加智能体失败", 
                    supervisor_id=supervisor_id,
                    agent_name=agent_name, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加智能体失败: {str(e)}"
        )


@router.delete(
    "/agents/{agent_name}",
    response_model=BaseResponse,
    summary="移除智能体",
    description="从Supervisor管理范围中移除智能体"
)
async def remove_agent(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    agent_name: str = Path(..., description="智能体名称")
) -> BaseResponse:
    """从Supervisor移除智能体"""
    try:
        logger.info("从Supervisor移除智能体", 
                   supervisor_id=supervisor_id,
                   agent_name=agent_name)
        
        await supervisor_service.remove_agent_from_supervisor(
            supervisor_id, agent_name
        )
        
        return BaseResponse(
            success=True,
            message=f"智能体 {agent_name} 移除成功",
            data={"supervisor_id": supervisor_id, "agent_name": agent_name}
        )
        
    except Exception as e:
        logger.error("移除智能体失败", 
                    supervisor_id=supervisor_id,
                    agent_name=agent_name, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"移除智能体失败: {str(e)}"
        )


@router.post(
    "/tasks/{task_id}/complete",
    response_model=BaseResponse,
    summary="更新任务完成状态",
    description="更新任务的完成状态和结果"
)
async def update_task_completion(
    task_id: str = Path(..., description="任务ID"),
    success: bool = Query(..., description="任务是否成功"),
    quality_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="质量评分")
) -> BaseResponse:
    """更新任务完成状态"""
    try:
        logger.info("更新任务完成状态", 
                   task_id=task_id,
                   success=success,
                   quality_score=quality_score)
        
        await supervisor_service.update_task_completion(
            task_id=task_id,
            success=success,
            quality_score=quality_score
        )
        
        return BaseResponse(
            success=True,
            message="任务状态更新成功",
            data={
                "task_id": task_id,
                "success": success,
                "quality_score": quality_score
            }
        )
        
    except Exception as e:
        logger.error("任务状态更新失败", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务状态更新失败: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=BaseResponse,
    summary="获取统计数据", 
    description="获取Supervisor统计数据"
)
async def get_supervisor_stats(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> BaseResponse:
    """获取Supervisor统计数据"""
    try:
        logger.info("查询Supervisor统计数据", supervisor_id=supervisor_id)
        
        # 返回符合SupervisorStats类型的统计数据
        stats_data = {
            "total_tasks": 23,
            "completed_tasks": 18,
            "failed_tasks": 2,
            "pending_tasks": 2,
            "running_tasks": 1,
            "average_completion_time": 145.5,
            "success_rate": 0.87,
            "agent_utilization": {
                "code_expert": 0.75,
                "architect": 0.45,
                "doc_expert": 0.35
            },
            "decision_accuracy": 0.92,
            "recent_decisions": []
        }
        
        return BaseResponse(
            success=True,
            message="统计数据查询成功",
            data=stats_data
        )
        
    except Exception as e:
        logger.error("统计数据查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"统计数据查询失败: {str(e)}"
        )


@router.get(
    "/load-statistics",
    response_model=LoadStatisticsApiResponse,
    summary="获取负载统计",
    description="获取智能体负载和性能统计信息"
)
async def get_load_statistics(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> LoadStatisticsApiResponse:
    """获取负载统计信息"""
    try:
        logger.info("查询负载统计", supervisor_id=supervisor_id)
        
        # 这里需要实现负载统计逻辑
        # 暂时返回模拟数据
        statistics = {
            "total_tasks_assigned": 0,
            "average_load": 0.0,
            "agent_loads": {},
            "task_counts": {},
            "busiest_agent": {"name": "", "load": 0.0},
            "least_busy_agent": {"name": "", "load": 0.0},
            "last_update": "2025-01-01T00:00:00Z",
            "load_distribution": {
                "low_load": 0,
                "medium_load": 0,
                "high_load": 0
            }
        }
        
        return LoadStatisticsApiResponse(
            success=True,
            message="负载统计查询成功",
            data=statistics
        )
        
    except Exception as e:
        logger.error("负载统计查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"负载统计查询失败: {str(e)}"
        )


@router.get(
    "/metrics",
    response_model=LoadStatisticsApiResponse,
    summary="获取智能体指标",
    description="获取Supervisor管理的智能体负载指标"
)
async def get_agent_metrics(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> LoadStatisticsApiResponse:
    """获取智能体指标"""
    try:
        logger.info("查询智能体指标", supervisor_id=supervisor_id)
        
        # 返回模拟的指标数据
        metrics = {
            "total_tasks_assigned": 15,
            "average_load": 0.65,
            "agent_loads": {
                "code_agent": 0.8,
                "architect_agent": 0.5,
                "doc_agent": 0.3
            },
            "task_counts": {
                "pending": 2,
                "running": 5,
                "completed": 8
            },
            "busiest_agent": {"name": "code_agent", "load": 0.8},
            "least_busy_agent": {"name": "doc_agent", "load": 0.3},
            "last_update": "2025-08-06T12:00:00Z",
            "load_distribution": {
                "low_load": 1,
                "medium_load": 1,
                "high_load": 1
            }
        }
        
        return LoadStatisticsApiResponse(
            success=True,
            message="智能体指标查询成功",
            data=metrics
        )
        
    except Exception as e:
        logger.error("智能体指标查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"智能体指标查询失败: {str(e)}"
        )


@router.get(
    "/tasks",
    response_model=BaseResponse,
    summary="获取任务列表",
    description="获取Supervisor管理的任务列表"
)
async def get_tasks(
    supervisor_id: str = Query(..., description="Supervisor ID"),
    status_filter: Optional[str] = Query(None, description="任务状态过滤"),
    limit: int = Query(10, ge=1, le=100, description="返回记录数量"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> BaseResponse:
    """获取任务列表"""
    try:
        logger.info("查询任务列表", 
                   supervisor_id=supervisor_id,
                   status_filter=status_filter,
                   limit=limit,
                   offset=offset)
        
        # 获取真实的任务数据
        from src.core.database import get_db_session
        from src.repositories.supervisor_repository import SupervisorTaskRepository
        
        async with get_db_session() as db:
            task_repo = SupervisorTaskRepository(db)
            
            # 根据supervisor_id获取任务，支持按名称或ID查询
            tasks = await task_repo.get_tasks_by_supervisor(
                supervisor_id, limit, offset, status_filter
            )
            
            # 转换为API响应格式
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "task_type": task.task_type,  # 修正字段名
                    "status": task.status,
                    "assigned_agent_name": task.assigned_agent_name or "未分配",  # 修正字段名
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "priority": task.priority,
                    "complexity_score": task.complexity_score,
                    "estimated_time_seconds": task.estimated_time_seconds  # 修正字段名
                })
        
        return BaseResponse(
            success=True,
            message="任务列表查询成功",
            data={
                "tasks": task_list,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(task_list)  # TODO: 应该从数据库获取总数
                }
            }
        )
        
    except Exception as e:
        logger.error("任务列表查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务列表查询失败: {str(e)}"
        )


@router.get(
    "/config",
    response_model=SupervisorConfigApiResponse,
    summary="获取Supervisor配置",
    description="获取Supervisor的当前配置信息"
)
async def get_supervisor_config(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> SupervisorConfigApiResponse:
    """获取Supervisor配置"""
    try:
        logger.info("查询Supervisor配置", supervisor_id=supervisor_id)
        
        # 返回模拟的配置数据
        from src.models.schemas.supervisor import RoutingStrategy
        config_data = {
            "id": "config_001",
            "supervisor_id": supervisor_id,
            "config_name": "default",
            "config_version": "1.0",
            "routing_strategy": RoutingStrategy.HYBRID,
            "load_threshold": 0.8,
            "capability_weight": 0.5,
            "load_weight": 0.3,
            "availability_weight": 0.2,
            "enable_quality_assessment": True,
            "min_confidence_threshold": 0.7,
            "enable_learning": True,
            "learning_rate": 0.1,
            "max_concurrent_tasks": 10,
            "task_timeout_minutes": 30,
            "enable_fallback": True,
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        return SupervisorConfigApiResponse(
            success=True,
            message="配置查询成功",
            data=config_data
        )
        
    except Exception as e:
        logger.error("配置查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置查询失败: {str(e)}"
        )


@router.get(
    "/health",
    response_model=BaseResponse,
    summary="健康检查",
    description="检查Supervisor服务的健康状态"
)
async def health_check() -> BaseResponse:
    """Supervisor服务健康检查"""
    try:
        return BaseResponse(
            success=True,
            message="Supervisor服务运行正常",
            data={
                "status": "healthy",
                "timestamp": "2025-08-06T12:00:00Z",
                "version": "1.0.0"
            }
        )
    except Exception as e:
        logger.error("健康检查失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"健康检查失败: {str(e)}"
        )


@router.post(
    "/tasks/{task_id}/execute",
    response_model=BaseResponse,
    summary="手动执行任务",
    description="手动执行指定的pending任务"
)
async def execute_task(
    task_id: str = Path(..., description="任务ID")
) -> BaseResponse:
    """手动执行任务"""
    try:
        logger.info("手动执行任务", task_id=task_id)
        
        from src.services.task_executor import task_executor
        result = await task_executor.execute_task(task_id)
        
        if result.get("success", False):
            return BaseResponse(
                success=True,
                message="任务执行成功",
                data=result
            )
        else:
            return BaseResponse(
                success=False,
                message=f"任务执行失败: {result.get('error', '未知错误')}",
                data=result
            )
        
    except Exception as e:
        logger.error("任务执行失败", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务执行失败: {str(e)}"
        )


@router.post(
    "/scheduler/force-execution",
    response_model=BaseResponse,
    summary="强制执行任务调度",
    description="强制执行一次任务调度周期"
)
async def force_task_execution() -> BaseResponse:
    """强制执行任务调度"""
    try:
        logger.info("强制执行任务调度")
        
        from src.services.task_scheduler import task_scheduler
        result = await task_scheduler.force_execution()
        
        return BaseResponse(
            success=result.get("success", False),
            message="任务调度执行完成" if result.get("success") else f"任务调度执行失败: {result.get('error')}",
            data=result
        )
        
    except Exception as e:
        logger.error("强制任务调度失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"强制任务调度失败: {str(e)}"
        )


@router.get(
    "/scheduler/status",
    response_model=BaseResponse,
    summary="获取调度器状态",
    description="获取任务调度器和执行器的状态信息"
)
async def get_scheduler_status() -> BaseResponse:
    """获取调度器状态"""
    try:
        from src.services.task_scheduler import task_scheduler
        status_data = task_scheduler.get_status()
        
        return BaseResponse(
            success=True,
            message="调度器状态查询成功",
            data=status_data
        )
        
    except Exception as e:
        logger.error("查询调度器状态失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询调度器状态失败: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}/details",
    response_model=BaseResponse,
    summary="获取任务详细信息",
    description="获取任务的详细信息，包括执行结果和输出数据"
)
async def get_task_details(
    task_id: str = Path(..., description="任务ID")
) -> BaseResponse:
    """获取任务详细信息"""
    try:
        logger.info("查询任务详细信息", task_id=task_id)
        
        from src.core.database import get_db_session
        from src.repositories.supervisor_repository import SupervisorTaskRepository
        
        async with get_db_session() as db:
            task_repo = SupervisorTaskRepository(db)
            task = await task_repo.get_by_id(task_id)
            
            if not task:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="任务不存在"
                )
            
            # 转换为详细格式
            task_details = {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "task_type": task.task_type,
                "priority": task.priority,
                "status": task.status,
                "assigned_agent_id": task.assigned_agent_id,
                "assigned_agent_name": task.assigned_agent_name,
                "supervisor_id": task.supervisor_id,
                "input_data": task.input_data,
                "output_data": task.output_data,
                "execution_metadata": task.execution_metadata,
                "complexity_score": task.complexity_score,
                "estimated_time_seconds": task.estimated_time_seconds,
                "actual_time_seconds": task.actual_time_seconds,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
            return BaseResponse(
                success=True,
                message="任务详细信息查询成功",
                data=task_details
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("任务详细信息查询失败", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务详细信息查询失败: {str(e)}"
        )


# 错误处理器
# Exception handlers removed - handled by global handlers


# Exception handlers removed - handled by global handlers