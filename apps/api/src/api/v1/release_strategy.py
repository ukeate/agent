"""
发布策略API端点
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ...services.release_strategy_service import (
    ReleaseStrategyService,
    ReleaseType,
    ApprovalLevel,
    Environment,
    ReleaseStage
)


router = APIRouter(prefix="/release-strategy", tags=["Release Strategy"])

# 服务实例
strategy_service = ReleaseStrategyService()


class CreateStageRequest(BaseModel):
    """创建阶段请求"""
    name: str = Field(..., description="阶段名称")
    environment: Environment = Field(..., description="环境")
    traffic_percentage: float = Field(..., ge=0, le=100, description="流量百分比")
    duration_hours: float = Field(..., ge=0, description="持续时间(小时)")
    success_criteria: Dict[str, Any] = Field({}, description="成功条件")
    rollback_criteria: Dict[str, Any] = Field({}, description="回滚条件")
    approval_required: bool = Field(False, description="是否需要审批")
    approvers: List[str] = Field([], description="审批人列表")


class CreateStrategyRequest(BaseModel):
    """创建策略请求"""
    experiment_id: str = Field(..., description="实验ID")
    name: str = Field(..., description="策略名称")
    description: str = Field("", description="策略描述")
    release_type: ReleaseType = Field(..., description="发布类型")
    stages: List[CreateStageRequest] = Field(..., description="发布阶段")
    approval_level: ApprovalLevel = Field(ApprovalLevel.SINGLE, description="审批级别")
    auto_promote: bool = Field(False, description="自动晋级")
    auto_rollback: bool = Field(True, description="自动回滚")
    monitoring_config: Dict[str, Any] = Field({}, description="监控配置")
    notification_config: Dict[str, Any] = Field({}, description="通知配置")


class CreateFromTemplateRequest(BaseModel):
    """从模板创建请求"""
    experiment_id: str = Field(..., description="实验ID")
    template_name: str = Field(..., description="模板名称")
    customizations: Optional[Dict[str, Any]] = Field(None, description="自定义配置")


class ApproveStageRequest(BaseModel):
    """审批阶段请求"""
    exec_id: str = Field(..., description="执行ID")
    stage_index: int = Field(..., ge=0, description="阶段索引")
    approver: str = Field(..., description="审批人")
    approved: bool = Field(..., description="是否批准")
    comments: Optional[str] = Field(None, description="审批意见")


@router.post("/strategies")
async def create_strategy(request: CreateStrategyRequest) -> Dict[str, Any]:
    """
    创建发布策略
    
    定义实验的发布流程和阶段
    """
    try:
        # 转换阶段数据
        stages = []
        for stage_req in request.stages:
            stage = ReleaseStage(
                name=stage_req.name,
                environment=stage_req.environment,
                traffic_percentage=stage_req.traffic_percentage,
                duration_hours=stage_req.duration_hours,
                success_criteria=stage_req.success_criteria,
                rollback_criteria=stage_req.rollback_criteria,
                approval_required=stage_req.approval_required,
                approvers=stage_req.approvers
            )
            stages.append(stage)
            
        strategy = await strategy_service.create_strategy(
            experiment_id=request.experiment_id,
            name=request.name,
            release_type=request.release_type,
            stages=stages,
            description=request.description,
            approval_level=request.approval_level,
            auto_promote=request.auto_promote,
            auto_rollback=request.auto_rollback,
            monitoring_config=request.monitoring_config,
            notification_config=request.notification_config
        )
        
        # 验证策略
        errors = await strategy_service.validate_strategy(strategy)
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "experiment_id": strategy.experiment_id,
                "release_type": strategy.release_type,
                "num_stages": len(strategy.stages),
                "approval_level": strategy.approval_level,
                "auto_promote": strategy.auto_promote,
                "auto_rollback": strategy.auto_rollback
            },
            "validation_errors": errors
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/from-template")
async def create_from_template(request: CreateFromTemplateRequest) -> Dict[str, Any]:
    """
    从模板创建策略
    
    使用预定义模板快速创建策略
    """
    try:
        strategy = await strategy_service.create_from_template(
            experiment_id=request.experiment_id,
            template_name=request.template_name,
            customizations=request.customizations
        )
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "experiment_id": strategy.experiment_id,
                "release_type": strategy.release_type,
                "template_used": request.template_name,
                "num_stages": len(strategy.stages)
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies(
    experiment_id: Optional[str] = Query(None, description="实验ID"),
    release_type: Optional[ReleaseType] = Query(None, description="发布类型")
) -> Dict[str, Any]:
    """
    列出发布策略
    """
    try:
        strategies = []
        
        for strategy_id, strategy in strategy_service.strategies.items():
            if experiment_id and strategy.experiment_id != experiment_id:
                continue
            if release_type and strategy.release_type != release_type:
                continue
                
            strategies.append({
                "id": strategy.id,
                "name": strategy.name,
                "experiment_id": strategy.experiment_id,
                "release_type": strategy.release_type,
                "num_stages": len(strategy.stages),
                "approval_level": strategy.approval_level,
                "created_at": strategy.created_at.isoformat()
            })
            
        return {
            "success": True,
            "strategies": strategies,
            "total": len(strategies)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_id}")
async def get_strategy(strategy_id: str) -> Dict[str, Any]:
    """
    获取策略详情
    """
    try:
        if strategy_id not in strategy_service.strategies:
            raise HTTPException(status_code=404, detail="策略不存在")
            
        strategy = strategy_service.strategies[strategy_id]
        
        return {
            "success": True,
            "strategy": {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "experiment_id": strategy.experiment_id,
                "release_type": strategy.release_type,
                "stages": [
                    {
                        "name": stage.name,
                        "environment": stage.environment,
                        "traffic_percentage": stage.traffic_percentage,
                        "duration_hours": stage.duration_hours,
                        "success_criteria": stage.success_criteria,
                        "rollback_criteria": stage.rollback_criteria,
                        "approval_required": stage.approval_required,
                        "approvers": stage.approvers
                    }
                    for stage in strategy.stages
                ],
                "approval_level": strategy.approval_level,
                "auto_promote": strategy.auto_promote,
                "auto_rollback": strategy.auto_rollback,
                "monitoring_config": strategy.monitoring_config,
                "notification_config": strategy.notification_config,
                "created_at": strategy.created_at.isoformat(),
                "updated_at": strategy.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute/{strategy_id}")
async def execute_strategy(strategy_id: str) -> Dict[str, Any]:
    """
    执行发布策略
    """
    try:
        execution = await strategy_service.execute_strategy(strategy_id)
        
        # 获取执行ID
        exec_id = None
        for eid, e in strategy_service.executions.items():
            if e == execution:
                exec_id = eid
                break
                
        return {
            "success": True,
            "exec_id": exec_id,
            "execution": {
                "strategy_id": execution.strategy_id,
                "experiment_id": execution.experiment_id,
                "status": execution.status,
                "current_stage": execution.current_stage,
                "started_at": execution.started_at.isoformat()
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve")
async def approve_stage(request: ApproveStageRequest) -> Dict[str, Any]:
    """
    审批发布阶段
    """
    try:
        approved = await strategy_service.approve_stage(
            exec_id=request.exec_id,
            stage_index=request.stage_index,
            approver=request.approver,
            approved=request.approved,
            comments=request.comments
        )
        
        return {
            "success": True,
            "stage_approved": approved,
            "message": "审批已记录" if approved else "需要更多审批"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions/{exec_id}")
async def get_execution_status(exec_id: str) -> Dict[str, Any]:
    """
    获取执行状态
    """
    try:
        status = await strategy_service.get_execution_status(exec_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="执行不存在")
            
        return {
            "success": True,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """
    列出策略模板
    """
    templates = []
    
    for name, template in strategy_service.templates.items():
        templates.append({
            "name": name,
            "display_name": template.name,
            "description": template.description,
            "release_type": template.release_type,
            "num_stages": len(template.stages),
            "approval_level": template.approval_level,
            "auto_promote": template.auto_promote,
            "auto_rollback": template.auto_rollback
        })
        
    return {
        "success": True,
        "templates": templates
    }


@router.get("/release-types")
async def list_release_types() -> Dict[str, Any]:
    """
    列出发布类型
    """
    types = [
        {
            "value": ReleaseType.CANARY,
            "name": "金丝雀发布",
            "description": "小流量验证，逐步扩大",
            "use_case": "新功能验证"
        },
        {
            "value": ReleaseType.BLUE_GREEN,
            "name": "蓝绿发布",
            "description": "完整切换，快速回滚",
            "use_case": "大版本更新"
        },
        {
            "value": ReleaseType.ROLLING,
            "name": "滚动发布",
            "description": "逐步替换实例",
            "use_case": "无缝升级"
        },
        {
            "value": ReleaseType.FEATURE_FLAG,
            "name": "功能开关",
            "description": "动态控制功能开关",
            "use_case": "功能灰度"
        },
        {
            "value": ReleaseType.GRADUAL,
            "name": "渐进发布",
            "description": "基于指标自动调整",
            "use_case": "数据驱动发布"
        },
        {
            "value": ReleaseType.SHADOW,
            "name": "影子发布",
            "description": "镜像流量测试",
            "use_case": "性能验证"
        }
    ]
    
    return {
        "success": True,
        "release_types": types
    }


@router.get("/approval-levels")
async def list_approval_levels() -> Dict[str, Any]:
    """
    列出审批级别
    """
    levels = [
        {
            "value": ApprovalLevel.NONE,
            "name": "无需审批",
            "description": "自动执行所有阶段"
        },
        {
            "value": ApprovalLevel.SINGLE,
            "name": "单人审批",
            "description": "每个阶段需要一人审批"
        },
        {
            "value": ApprovalLevel.MULTIPLE,
            "name": "多人审批",
            "description": "需要所有指定人员审批"
        },
        {
            "value": ApprovalLevel.TIERED,
            "name": "分级审批",
            "description": "不同阶段不同审批要求"
        }
    ]
    
    return {
        "success": True,
        "approval_levels": levels
    }


@router.get("/environments")
async def list_environments() -> Dict[str, Any]:
    """
    列出环境
    """
    environments = [
        {
            "value": Environment.DEVELOPMENT,
            "name": "开发环境",
            "description": "开发测试使用"
        },
        {
            "value": Environment.TESTING,
            "name": "测试环境",
            "description": "QA测试使用"
        },
        {
            "value": Environment.STAGING,
            "name": "预发环境",
            "description": "生产前验证"
        },
        {
            "value": Environment.PRODUCTION,
            "name": "生产环境",
            "description": "正式服务环境"
        }
    ]
    
    return {
        "success": True,
        "environments": environments
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "release_strategy",
        "status": "healthy",
        "total_strategies": len(strategy_service.strategies),
        "active_executions": len([
            e for e in strategy_service.executions.values()
            if e.status == "in_progress"
        ])
    }