"""
Supervisor管理API路由
提供Supervisor智能体的HTTP接口
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse
from src.services.supervisor_service import supervisor_service
from src.models.schemas.supervisor import (
    TaskSubmissionRequest, TaskSubmissionResponse,
    SupervisorStatusApiResponse, SupervisorDecisionListResponse,
    SupervisorConfigUpdateRequest, SupervisorConfigApiResponse,
    LoadStatisticsApiResponse, TaskType, TaskPriority
)
from src.models.schemas.base import BaseResponse, ErrorResponse
from src.api.exceptions import ValidationError, NotFoundError

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/supervisor", tags=["supervisor"])

def _serialize_supervisor_config(config: Any) -> Dict[str, Any]:
    return {
        "id": config.id,
        "supervisor_id": config.supervisor_id,
        "config_name": config.config_name,
        "config_version": config.config_version,
        "routing_strategy": config.routing_strategy,
        "load_threshold": config.load_threshold,
        "capability_weight": config.capability_weight,
        "load_weight": config.load_weight,
        "availability_weight": config.availability_weight,
        "enable_quality_assessment": config.enable_quality_assessment,
        "min_confidence_threshold": config.min_confidence_threshold,
        "enable_learning": config.enable_learning,
        "learning_rate": config.learning_rate,
        "optimization_interval_hours": config.optimization_interval_hours,
        "max_concurrent_tasks": config.max_concurrent_tasks,
        "task_timeout_minutes": config.task_timeout_minutes,
        "enable_fallback": config.enable_fallback,
        "config_metadata": config.config_metadata,
        "is_active": config.is_active,
        "created_at": config.created_at,
        "updated_at": config.updated_at,
    }

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
        logger.warning("Supervisor不存在，尝试自动初始化", supervisor_id=supervisor_id, error=str(e))
        await supervisor_service.initialize_supervisor(supervisor_id)
        status_data = await supervisor_service.get_supervisor_status(supervisor_id)
        return SupervisorStatusApiResponse(
            success=True,
            message="状态查询成功（自动初始化）",
            data=status_data
        )
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
        total = await supervisor_service.get_decision_history_total(supervisor_id)
        
        return SupervisorDecisionListResponse(
            success=True,
            message="决策历史查询成功", 
            data=decisions,
            pagination={
                "limit": limit,
                "offset": offset,
                "total": total
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
                   config_updates=request.model_dump(exclude_unset=True))
        
        config = await supervisor_service.update_supervisor_config(
            supervisor_id, request
        )
        
        return SupervisorConfigApiResponse(
            success=True,
            message="配置更新成功",
            data=_serialize_supervisor_config(config)
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
        agent = supervisor_service._available_agents.get(agent_name)
        if not agent:
            agent_pool = supervisor_service._create_default_agent_pool()
            agent = agent_pool.get(agent_name)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="智能体不存在",
                )
            supervisor_service._available_agents.update(agent_pool)

        await supervisor_service.add_agent_to_supervisor(supervisor_id, agent_name, agent)
        return BaseResponse(
            success=True,
            message="智能体添加成功",
            data={"supervisor_id": supervisor_id, "agent_name": agent_name},
        )
    except HTTPException:
        raise
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
        from src.core.database import get_db_session
        from src.repositories.supervisor_repository import (
            SupervisorRepository,
            SupervisorTaskRepository,
            SupervisorDecisionRepository,
            AgentLoadMetricsRepository,
        )

        async with get_db_session() as db:
            supervisor_repo = SupervisorRepository(db)
            supervisor = await supervisor_repo.get_by_id(supervisor_id)
            if not supervisor:
                supervisor = await supervisor_repo.get_by_name(supervisor_id)
            if not supervisor:
                created_id = await supervisor_service.initialize_supervisor(supervisor_id)
                supervisor = await supervisor_repo.get_by_id(created_id)
            if not supervisor:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Supervisor不存在")

            task_repo = SupervisorTaskRepository(db)
            decision_repo = SupervisorDecisionRepository(db)
            load_repo = AgentLoadMetricsRepository(db)

            task_stats = await task_repo.get_task_statistics(supervisor.id)
            decision_stats = await decision_repo.get_decision_statistics(supervisor.id)
            current_loads = await load_repo.get_current_loads(supervisor.id)

        return BaseResponse(
            success=True,
            message="统计数据查询成功",
            data={
                "supervisor_id": supervisor.id,
                "supervisor_name": supervisor.name,
                "task_statistics": task_stats,
                "decision_statistics": decision_stats,
                "agent_loads": current_loads,
                "timestamp": utc_now().isoformat(),
            },
        )
    except HTTPException:
        raise
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
        from sqlalchemy import select, func

        from src.core.database import get_db_session
        from src.models.database.supervisor import SupervisorTask
        from src.repositories.supervisor_repository import (
            SupervisorRepository,
            AgentLoadMetricsRepository,
        )

        async with get_db_session() as db:
            supervisor_repo = SupervisorRepository(db)
            supervisor = await supervisor_repo.get_by_id(supervisor_id)
            if not supervisor:
                supervisor = await supervisor_repo.get_by_name(supervisor_id)
            if not supervisor:
                created_id = await supervisor_service.initialize_supervisor(supervisor_id)
                supervisor = await supervisor_repo.get_by_id(created_id)
            if not supervisor:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Supervisor不存在")

            load_repo = AgentLoadMetricsRepository(db)
            agent_loads = await load_repo.get_current_loads(supervisor.id)

            stmt = (
                select(SupervisorTask.assigned_agent_name, func.count(SupervisorTask.id))
                .where(SupervisorTask.supervisor_id == supervisor.id)
                .group_by(SupervisorTask.assigned_agent_name)
            )
            rows = await db.execute(stmt)
            task_counts = {name or "未分配": int(count) for name, count in rows.all()}

        loads = list(agent_loads.values())
        average_load = float(sum(loads) / len(loads)) if loads else 0.0
        busiest_name, busiest_load = max(agent_loads.items(), key=lambda x: x[1], default=("", 0.0))
        least_name, least_load = min(agent_loads.items(), key=lambda x: x[1], default=("", 0.0))

        buckets = {"0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0, "0.75-1.0": 0}
        for load in loads:
            if load < 0.25:
                buckets["0-0.25"] += 1
            elif load < 0.5:
                buckets["0.25-0.5"] += 1
            elif load < 0.75:
                buckets["0.5-0.75"] += 1
            else:
                buckets["0.75-1.0"] += 1

        total_tasks_assigned = int(sum(v for k, v in task_counts.items() if k != "未分配"))

        return LoadStatisticsApiResponse(
            success=True,
            message="负载统计查询成功",
            data={
                "total_tasks_assigned": total_tasks_assigned,
                "average_load": average_load,
                "agent_loads": agent_loads,
                "task_counts": task_counts,
                "busiest_agent": {"agent_name": busiest_name, "current_load": busiest_load},
                "least_busy_agent": {"agent_name": least_name, "current_load": least_load},
                "last_update": utc_now(),
                "load_distribution": buckets,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("负载统计查询失败", supervisor_id=supervisor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"负载统计查询失败: {str(e)}"
        )

@router.get(
    "/metrics",
    response_model=BaseResponse,
    summary="获取智能体指标",
    description="获取Supervisor管理的智能体负载指标"
)
async def get_agent_metrics(
    supervisor_id: str = Query(..., description="Supervisor ID")
) -> BaseResponse:
    """获取智能体指标"""
    try:
        logger.info("查询智能体指标", supervisor_id=supervisor_id)
        from sqlalchemy import select
        from sqlalchemy import desc as sql_desc

        from src.core.database import get_db_session
        from src.models.database.supervisor import AgentLoadMetrics
        from src.repositories.supervisor_repository import SupervisorRepository

        async with get_db_session() as db:
            supervisor_repo = SupervisorRepository(db)
            supervisor = await supervisor_repo.get_by_id(supervisor_id)
            if not supervisor:
                supervisor = await supervisor_repo.get_by_name(supervisor_id)
            if not supervisor:
                created_id = await supervisor_service.initialize_supervisor(supervisor_id)
                supervisor = await supervisor_repo.get_by_id(created_id)
            if not supervisor:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Supervisor不存在")

            stmt = (
                select(AgentLoadMetrics)
                .where(AgentLoadMetrics.supervisor_id == supervisor.id)
                .order_by(sql_desc(AgentLoadMetrics.updated_at))
            )
            result = await db.execute(stmt)
            rows = result.scalars().all()

        metrics = []
        seen = set()
        for m in rows:
            if m.agent_name in seen:
                continue
            seen.add(m.agent_name)
            metrics.append(
                {
                    "id": m.id,
                    "agent_name": m.agent_name,
                    "supervisor_id": m.supervisor_id,
                    "current_load": m.current_load,
                    "task_count": m.task_count,
                    "average_task_time": m.average_task_time,
                    "success_rate": m.success_rate,
                    "response_time_avg": m.response_time_avg,
                    "error_rate": m.error_rate,
                    "availability_score": m.availability_score,
                    "window_start": m.window_start.isoformat(),
                    "window_end": m.window_end.isoformat(),
                    "created_at": m.created_at.isoformat(),
                    "updated_at": m.updated_at.isoformat(),
                }
            )

        return BaseResponse(success=True, message="智能体指标查询成功", data=metrics)
    except HTTPException:
        raise
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
            total = await task_repo.count_tasks_by_supervisor(supervisor_id, status_filter)
            
            # 转换为API响应格式
            task_list = []
            for task in tasks:
                task_list.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "task_type": task.task_type,  # 修正字段名
                    "priority": task.priority,
                    "status": task.status,
                    "complexity_score": task.complexity_score,
                    "estimated_time_seconds": task.estimated_time_seconds,  # 修正字段名
                    "actual_time_seconds": task.actual_time_seconds,
                    "assigned_agent_id": task.assigned_agent_id,
                    "assigned_agent_name": task.assigned_agent_name,
                    "supervisor_id": task.supervisor_id,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                })
        
        return BaseResponse(
            success=True,
            message="任务列表查询成功",
            data={
                "tasks": task_list,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total
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
        from src.core.database import get_db_session
        from src.repositories.supervisor_repository import SupervisorConfigRepository, SupervisorRepository

        async with get_db_session() as db:
            supervisor_repo = SupervisorRepository(db)
            supervisor = await supervisor_repo.get_by_id(supervisor_id)
            if not supervisor:
                supervisor = await supervisor_repo.get_by_name(supervisor_id)
            if not supervisor:
                created_id = await supervisor_service.initialize_supervisor(supervisor_id)
                supervisor = await supervisor_repo.get_by_id(created_id)
            if not supervisor:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Supervisor不存在")

            config_repo = SupervisorConfigRepository(db)
            config = await config_repo.get_active_config(supervisor.id)
            if not config:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到活跃配置")

        return SupervisorConfigApiResponse(
            success=True,
            message="配置查询成功",
            data=_serialize_supervisor_config(config),
        )
    except HTTPException:
        raise
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
                "timestamp": utc_now().isoformat(),
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
