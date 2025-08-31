"""
A/B测试实验平台API端点
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session

from core.database import get_db
from models.schemas.experiment import (
    ExperimentConfig, CreateExperimentRequest, UpdateExperimentRequest,
    ExperimentSummary, ExperimentStatus, ExperimentAssignmentRequest,
    ExperimentAssignmentResponse, RecordEventRequest, BatchEventRequest,
    ExperimentResultsResponse, MetricResult
)
from repositories.experiment_repository import (
    ExperimentRepository, ExperimentAssignmentRepository, 
    ExperimentEventRepository, ExperimentMetricResultRepository
)
from services.ab_testing_service import ABTestingService
from services.experiment_manager import ExperimentManager
from services.multi_variant_allocator import (
    MultiVariantAllocator,
    MultiVariantAllocationRequest,
    AllocationStrategy
)
from services.advanced_traffic_allocator import AllocationRule, StageConfig
from core.logging import logger

router = APIRouter(prefix="/experiments", tags=["experiments"])


# 依赖注入
def get_experiment_repository(db: Session = Depends(get_db)) -> ExperimentRepository:
    return ExperimentRepository(db)

def get_assignment_repository(db: Session = Depends(get_db)) -> ExperimentAssignmentRepository:
    return ExperimentAssignmentRepository(db)

def get_event_repository(db: Session = Depends(get_db)) -> ExperimentEventRepository:
    return ExperimentEventRepository(db)

def get_metric_repository(db: Session = Depends(get_db)) -> ExperimentMetricResultRepository:
    return ExperimentMetricResultRepository(db)

def get_ab_testing_service(
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository),
    assignment_repo: ExperimentAssignmentRepository = Depends(get_assignment_repository),
    event_repo: ExperimentEventRepository = Depends(get_event_repository),
    metric_repo: ExperimentMetricResultRepository = Depends(get_metric_repository)
) -> ABTestingService:
    return ABTestingService(experiment_repo, assignment_repo, event_repo, metric_repo)

def get_experiment_manager(
    ab_service: ABTestingService = Depends(get_ab_testing_service)
) -> ExperimentManager:
    return ExperimentManager(ab_service)

def get_multi_variant_allocator(
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
) -> MultiVariantAllocator:
    return MultiVariantAllocator(experiment_repo)


@router.post("/", response_model=ExperimentConfig, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_request: CreateExperimentRequest,
    experiment_manager: ExperimentManager = Depends(get_experiment_manager)
):
    """创建新实验"""
    try:
        experiment = await experiment_manager.create_experiment(experiment_request)
        logger.info(f"Created experiment {experiment.id} by {experiment_request.owner}")
        return experiment
        
    except ValueError as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating experiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create experiment"
        )


@router.get("/", response_model=List[ExperimentSummary])
async def list_experiments(
    owner: Optional[str] = Query(None, description="实验负责人"),
    status: Optional[ExperimentStatus] = Query(None, description="实验状态"),
    limit: int = Query(50, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
):
    """获取实验列表"""
    try:
        if owner:
            experiments = experiment_repo.get_experiments_by_owner(owner, status)
        else:
            # 获取所有实验（可能需要权限控制）
            query = experiment_repo.db.query(experiment_repo.model)
            if status:
                query = query.filter(experiment_repo.model.status == status)
            experiments = query.offset(offset).limit(limit).all()
        
        # 转换为摘要格式
        summaries = []
        for exp in experiments:
            summary = ExperimentSummary(
                experiment_id=exp.id,
                name=exp.name,
                status=exp.status,
                start_date=exp.start_date,
                end_date=exp.end_date,
                created_at=exp.created_at,
                # TODO: 添加统计数据
                total_users=0,
                total_events=0,
                variants_performance={},
                significant_metrics=[]
            )
            summaries.append(summary)
        
        return summaries
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiments"
        )


@router.get("/{experiment_id}", response_model=ExperimentConfig)
async def get_experiment(
    experiment_id: str,
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
):
    """获取实验详情"""
    try:
        experiment = experiment_repo.get_experiment_with_variants(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found"
            )
        
        # 构建流量分配信息
        from models.schemas.experiment import TrafficAllocation, ExperimentVariant
        
        variants = []
        traffic_allocations = []
        
        for variant in experiment.variants:
            variants.append(ExperimentVariant(
                variant_id=variant.variant_id,
                name=variant.name,
                description=variant.description,
                config=variant.config,
                is_control=variant.is_control
            ))
            
            traffic_allocations.append(TrafficAllocation(
                variant_id=variant.variant_id,
                percentage=variant.traffic_percentage
            ))
        
        # 构建完整的实验配置
        config = ExperimentConfig(
            experiment_id=experiment.id,
            name=experiment.name,
            description=experiment.description,
            hypothesis=experiment.hypothesis,
            owner=experiment.owner,
            status=experiment.status,
            variants=variants,
            traffic_allocation=traffic_allocations,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
            success_metrics=experiment.success_metrics,
            guardrail_metrics=experiment.guardrail_metrics,
            minimum_sample_size=experiment.minimum_sample_size,
            significance_level=experiment.significance_level,
            power=experiment.power,
            layers=experiment.layers,
            targeting_rules=[],  # TODO: 反序列化规则
            metadata=experiment.metadata,
            created_at=experiment.created_at,
            updated_at=experiment.updated_at
        )
        
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiment"
        )


@router.put("/{experiment_id}", response_model=ExperimentConfig)
async def update_experiment(
    experiment_id: str,
    update_request: UpdateExperimentRequest,
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
):
    """更新实验配置"""
    try:
        experiment = experiment_repo.update_experiment(experiment_id, update_request)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be updated"
            )
        
        logger.info(f"Updated experiment {experiment_id}")
        
        # 返回更新后的配置（需要重构为完整的配置对象）
        # TODO: 实现完整的转换逻辑
        return await get_experiment(experiment_id, experiment_repo)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update experiment"
        )


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: str,
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
):
    """删除实验（仅草稿状态）"""
    try:
        success = experiment_repo.delete_experiment(experiment_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be deleted"
            )
        
        logger.info(f"Deleted experiment {experiment_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete experiment"
        )


@router.post("/{experiment_id}/start", response_model=ExperimentConfig)
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    experiment_manager: ExperimentManager = Depends(get_experiment_manager)
):
    """启动实验"""
    try:
        experiment = await experiment_manager.start_experiment(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be started"
            )
        
        logger.info(f"Started experiment {experiment_id}")
        
        # 异步执行启动后的任务
        background_tasks.add_task(experiment_manager.on_experiment_started, experiment_id)
        
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start experiment"
        )


@router.post("/{experiment_id}/pause", response_model=ExperimentConfig)
async def pause_experiment(
    experiment_id: str,
    experiment_manager: ExperimentManager = Depends(get_experiment_manager)
):
    """暂停实验"""
    try:
        experiment = await experiment_manager.pause_experiment(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be paused"
            )
        
        logger.info(f"Paused experiment {experiment_id}")
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause experiment"
        )


@router.post("/{experiment_id}/stop", response_model=ExperimentConfig)
async def stop_experiment(
    experiment_id: str,
    experiment_manager: ExperimentManager = Depends(get_experiment_manager)
):
    """停止实验"""
    try:
        experiment = await experiment_manager.stop_experiment(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or cannot be stopped"
            )
        
        logger.info(f"Stopped experiment {experiment_id}")
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop experiment"
        )


@router.post("/{experiment_id}/assign", response_model=ExperimentAssignmentResponse)
async def assign_user_to_experiment(
    experiment_id: str,
    assignment_request: ExperimentAssignmentRequest,
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """为用户分配实验变体"""
    try:
        assignment = await ab_service.assign_user(
            experiment_id, 
            assignment_request.user_id, 
            assignment_request.context
        )
        
        if not assignment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or user not eligible"
            )
        
        return assignment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign user to experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign user to experiment"
        )


@router.get("/user/{user_id}", response_model=List[ExperimentAssignmentResponse])
async def get_user_experiments(
    user_id: str,
    active_only: bool = Query(True, description="仅返回活跃实验"),
    assignment_repo: ExperimentAssignmentRepository = Depends(get_assignment_repository)
):
    """获取用户的所有实验分配"""
    try:
        # TODO: 实现获取用户所有实验分配的逻辑
        # 目前先返回空列表
        return []
        
    except Exception as e:
        logger.error(f"Failed to get user experiments for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user experiments"
        )


@router.post("/events", status_code=status.HTTP_201_CREATED)
async def record_event(
    event_request: RecordEventRequest,
    experiment_id: str = Query(..., description="实验ID"),
    assignment_id: str = Query(..., description="分配ID"),
    event_repo: ExperimentEventRepository = Depends(get_event_repository)
):
    """记录实验事件"""
    try:
        event = event_repo.record_event(assignment_id, event_request)
        logger.debug(f"Recorded event {event.id} for assignment {assignment_id}")
        return {"event_id": event.id, "status": "recorded"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to record event: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record event"
        )


@router.post("/events/batch", status_code=status.HTTP_201_CREATED)
async def record_batch_events(
    batch_request: BatchEventRequest,
    assignment_id: str = Query(..., description="分配ID"),
    event_repo: ExperimentEventRepository = Depends(get_event_repository)
):
    """批量记录实验事件"""
    try:
        events = event_repo.batch_record_events(assignment_id, batch_request.events)
        logger.info(f"Recorded {len(events)} events for assignment {assignment_id}")
        return {
            "recorded_count": len(events),
            "event_ids": [event.id for event in events]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to record batch events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record events"
        )


@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(
    experiment_id: str,
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """获取实验结果"""
    try:
        results = await ab_service.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or no results available"
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment results for {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve experiment results"
        )


@router.get("/{experiment_id}/report")
async def generate_experiment_report(
    experiment_id: str,
    format: str = Query("html", description="报告格式 (html, pdf, csv)"),
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """生成实验报告"""
    try:
        # TODO: 实现报告生成逻辑
        if format not in ["html", "pdf", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported report format"
            )
        
        # 暂时返回占位响应
        return {"message": f"Report generation for format {format} is not implemented yet"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate report for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate experiment report"
        )


@router.get("/{experiment_id}/metrics/{metric_name}", response_model=MetricResult)
async def get_metric_analysis(
    experiment_id: str,
    metric_name: str,
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """获取特定指标分析"""
    try:
        metric_result = await ab_service.analyze_metric(experiment_id, metric_name)
        if not metric_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metric analysis not found"
            )
        
        return metric_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metric analysis for {metric_name} in experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metric analysis"
        )


@router.get("/{experiment_id}/monitor")
async def get_experiment_monitor_data(
    experiment_id: str,
    hours: int = Query(24, ge=1, le=168, description="监控时间范围（小时）"),
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """获取实验实时监控数据"""
    try:
        # TODO: 实现实时监控数据获取
        return {
            "experiment_id": experiment_id,
            "monitoring_window_hours": hours,
            "message": "Real-time monitoring is not implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring data for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monitoring data"
        )


@router.post("/{experiment_id}/alerts")
async def configure_alerts(
    experiment_id: str,
    alert_config: Dict[str, Any],
    ab_service: ABTestingService = Depends(get_ab_testing_service)
):
    """配置实验告警规则"""
    try:
        # TODO: 实现告警配置逻辑
        return {
            "experiment_id": experiment_id,
            "alert_config": alert_config,
            "message": "Alert configuration is not implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure alerts for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure alerts"
        )


# 多变体分配API端点

@router.post("/{experiment_id}/allocate")
async def allocate_user_advanced(
    experiment_id: str,
    user_id: str = Query(..., description="用户ID"),
    strategy: AllocationStrategy = Query(AllocationStrategy.WEIGHTED, description="分配策略"),
    user_attributes: Dict[str, Any] = Query({}, description="用户属性"),
    force_assignment: bool = Query(False, description="强制重新分配"),
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """使用高级策略为用户分配实验变体"""
    try:
        request = MultiVariantAllocationRequest(
            user_id=user_id,
            experiment_id=experiment_id,
            user_attributes=user_attributes,
            allocation_strategy=strategy,
            force_assignment=force_assignment
        )
        
        result = await allocator.allocate_user_to_variants(request)
        
        return {
            "user_id": result.user_id,
            "experiment_id": result.experiment_id,
            "variant_id": result.variant_id,
            "allocation_strategy": result.allocation_strategy.value,
            "assignment_timestamp": result.assignment_timestamp,
            "is_cached": result.is_cached,
            "allocation_reason": result.allocation_reason
        }
        
    except Exception as e:
        logger.error(f"Failed to allocate user {user_id} to experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to allocate user to experiment"
        )


@router.post("/{experiment_id}/allocate/batch")
async def batch_allocate_users(
    experiment_id: str,
    user_requests: List[Dict[str, Any]],
    strategy: AllocationStrategy = Query(AllocationStrategy.WEIGHTED, description="默认分配策略"),
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """批量分配用户到实验变体"""
    try:
        requests = []
        for user_req in user_requests:
            request = MultiVariantAllocationRequest(
                user_id=user_req.get("user_id"),
                experiment_id=experiment_id,
                user_attributes=user_req.get("user_attributes", {}),
                allocation_strategy=AllocationStrategy(user_req.get("strategy", strategy.value)),
                force_assignment=user_req.get("force_assignment", False)
            )
            requests.append(request)
        
        results = await allocator.batch_allocate_users(requests)
        
        return {
            "experiment_id": experiment_id,
            "total_requests": len(requests),
            "successful_allocations": len([r for r in results if r.variant_id]),
            "failed_allocations": len([r for r in results if not r.variant_id]),
            "results": [{
                "user_id": r.user_id,
                "variant_id": r.variant_id,
                "allocation_strategy": r.allocation_strategy.value,
                "is_cached": r.is_cached,
                "allocation_reason": r.allocation_reason
            } for r in results]
        }
        
    except Exception as e:
        logger.error(f"Failed to batch allocate users to experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch allocate users"
        )


@router.get("/{experiment_id}/allocation/distribution")
async def get_allocation_distribution(
    experiment_id: str,
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """获取实验的分配分布统计"""
    try:
        distribution = await allocator.get_allocation_distribution(experiment_id)
        return distribution
        
    except Exception as e:
        logger.error(f"Failed to get allocation distribution for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve allocation distribution"
        )


@router.post("/{experiment_id}/allocation/simulate")
async def simulate_allocation_distribution(
    experiment_id: str,
    num_users: int = Query(10000, ge=100, le=100000, description="模拟用户数量"),
    user_attributes_samples: Optional[List[Dict[str, Any]]] = None,
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """模拟实验的分配分布"""
    try:
        simulation = await allocator.simulate_allocation_distribution(
            experiment_id, 
            num_users, 
            user_attributes_samples
        )
        return simulation
        
    except Exception as e:
        logger.error(f"Failed to simulate allocation distribution for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simulate allocation distribution"
        )


@router.post("/{experiment_id}/allocation/rules")
async def configure_allocation_rules(
    experiment_id: str,
    rules: List[Dict[str, Any]],
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """配置实验的分配规则"""
    try:
        allocation_rules = []
        for rule_data in rules:
            rule = AllocationRule(
                condition=rule_data.get("condition", ""),
                target_variants=rule_data.get("target_variants", []),
                allocation_percentages=rule_data.get("allocation_percentages", []),
                priority=rule_data.get("priority", 0),
                is_active=rule_data.get("is_active", True)
            )
            allocation_rules.append(rule)
        
        await allocator.configure_experiment_allocation_rules(experiment_id, allocation_rules)
        
        return {
            "experiment_id": experiment_id,
            "rules_configured": len(allocation_rules),
            "message": "Allocation rules configured successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure allocation rules for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure allocation rules"
        )


@router.post("/{experiment_id}/allocation/stages")
async def configure_allocation_stages(
    experiment_id: str,
    stages: List[Dict[str, Any]],
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """配置实验的阶段设置"""
    try:
        from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
        
        stage_configs = []
        for stage_data in stages:
            stage = StageConfig(
                stage_id=stage_data.get("stage_id"),
                start_time=datetime.fromisoformat(stage_data.get("start_time")),
                end_time=datetime.fromisoformat(stage_data.get("end_time")),
                allocation_percentages=stage_data.get("allocation_percentages", {}),
                max_users_per_variant=stage_data.get("max_users_per_variant")
            )
            stage_configs.append(stage)
        
        await allocator.configure_experiment_stages(experiment_id, stage_configs)
        
        return {
            "experiment_id": experiment_id,
            "stages_configured": len(stage_configs),
            "message": "Allocation stages configured successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure allocation stages for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure allocation stages"
        )


@router.delete("/{experiment_id}/allocation/cache")
async def clear_allocation_cache(
    experiment_id: str,
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """清理实验的分配缓存"""
    try:
        allocator.clear_allocation_cache(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "message": "Allocation cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear allocation cache for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear allocation cache"
        )


@router.get("/allocation/stats")
async def get_allocation_stats(
    allocator: MultiVariantAllocator = Depends(get_multi_variant_allocator)
):
    """获取全局分配统计信息"""
    try:
        stats = allocator.get_allocation_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get allocation stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve allocation statistics"
        )