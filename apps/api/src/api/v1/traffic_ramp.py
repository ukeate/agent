"""
流量渐进调整API端点
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ...services.traffic_ramp_service import (
    TrafficRampService,
    RampStrategy,
    RampStatus,
    RolloutPhase
)


router = APIRouter(prefix="/traffic-ramp", tags=["Traffic Ramp"])

# 服务实例
ramp_service = TrafficRampService()


class CreateRampPlanRequest(BaseModel):
    """创建爬坡计划请求"""
    experiment_id: str = Field(..., description="实验ID")
    variant: str = Field("treatment", description="变体名称")
    strategy: RampStrategy = Field(RampStrategy.LINEAR, description="爬坡策略")
    start_percentage: float = Field(0.0, ge=0, le=100, description="起始流量百分比")
    target_percentage: float = Field(100.0, ge=0, le=100, description="目标流量百分比")
    duration_hours: float = Field(24.0, gt=0, description="总持续时间(小时)")
    num_steps: int = Field(10, ge=1, le=100, description="步骤数量")
    health_checks: Optional[Dict[str, Any]] = Field(None, description="健康检查配置")
    rollback_conditions: Optional[Dict[str, Any]] = Field(None, description="回滚条件")


class StartRampRequest(BaseModel):
    """开始爬坡请求"""
    plan_id: str = Field(..., description="计划ID")


class ControlRampRequest(BaseModel):
    """控制爬坡请求"""
    exec_id: str = Field(..., description="执行ID")


class GetRecommendedPlanRequest(BaseModel):
    """获取推荐计划请求"""
    experiment_id: str = Field(..., description="实验ID")
    risk_level: str = Field("medium", description="风险等级: low, medium, high")


class QuickRampRequest(BaseModel):
    """快速爬坡请求"""
    experiment_id: str = Field(..., description="实验ID")
    phase: RolloutPhase = Field(..., description="发布阶段")
    duration_hours: Optional[float] = Field(None, description="持续时间")


@router.post("/plans")
async def create_ramp_plan(request: CreateRampPlanRequest) -> Dict[str, Any]:
    """
    创建流量爬坡计划
    
    定义实验流量的渐进调整策略
    """
    try:
        plan = await ramp_service.create_ramp_plan(
            experiment_id=request.experiment_id,
            variant=request.variant,
            strategy=request.strategy,
            start_percentage=request.start_percentage,
            target_percentage=request.target_percentage,
            duration_hours=request.duration_hours,
            num_steps=request.num_steps,
            health_checks=request.health_checks,
            rollback_conditions=request.rollback_conditions
        )
        
        # 获取计划ID
        plan_id = None
        for pid, p in ramp_service.ramp_plans.items():
            if p == plan:
                plan_id = pid
                break
                
        return {
            "success": True,
            "plan_id": plan_id,
            "plan": {
                "experiment_id": plan.experiment_id,
                "variant": plan.variant,
                "strategy": plan.strategy,
                "start_percentage": plan.start_percentage,
                "target_percentage": plan.target_percentage,
                "duration_hours": plan.total_duration_hours,
                "num_steps": len(plan.steps),
                "steps": [
                    {
                        "step": s.step_number,
                        "target": s.target_percentage,
                        "duration_minutes": s.duration_minutes
                    }
                    for s in plan.steps[:5]  # 显示前5步
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_ramp(
    request: StartRampRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    开始执行爬坡计划
    
    启动流量渐进调整过程
    """
    try:
        execution = await ramp_service.start_ramp(request.plan_id)
        
        # 获取执行ID
        exec_id = None
        for eid, e in ramp_service.executions.items():
            if e == execution:
                exec_id = eid
                break
                
        return {
            "success": True,
            "exec_id": exec_id,
            "execution": {
                "plan_id": execution.plan_id,
                "experiment_id": execution.experiment_id,
                "status": execution.status,
                "current_step": execution.current_step,
                "current_percentage": execution.current_percentage,
                "started_at": execution.started_at.isoformat() if execution.started_at else None
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause")
async def pause_ramp(request: ControlRampRequest) -> Dict[str, Any]:
    """
    暂停爬坡
    
    暂停流量调整，保持当前流量比例
    """
    try:
        success = await ramp_service.pause_ramp(request.exec_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法暂停该执行")
            
        return {
            "success": True,
            "message": "爬坡已暂停"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume")
async def resume_ramp(request: ControlRampRequest) -> Dict[str, Any]:
    """
    恢复爬坡
    
    从暂停点继续流量调整
    """
    try:
        success = await ramp_service.resume_ramp(request.exec_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法恢复该执行")
            
        return {
            "success": True,
            "message": "爬坡已恢复"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback")
async def rollback_ramp(
    exec_id: str,
    reason: str = Query(..., description="回滚原因")
) -> Dict[str, Any]:
    """
    回滚流量
    
    立即回滚到初始流量配置
    """
    try:
        await ramp_service._rollback(exec_id, reason)
        
        return {
            "success": True,
            "message": "流量已回滚",
            "reason": reason
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{exec_id}")
async def get_ramp_status(exec_id: str) -> Dict[str, Any]:
    """
    获取爬坡状态
    
    查询当前执行状态和进度
    """
    try:
        status = await ramp_service.get_ramp_status(exec_id)
        
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


@router.get("/plans")
async def list_plans(
    experiment_id: Optional[str] = Query(None, description="实验ID")
) -> Dict[str, Any]:
    """
    列出爬坡计划
    
    获取所有或指定实验的爬坡计划
    """
    try:
        plans = []
        
        for plan_id, plan in ramp_service.ramp_plans.items():
            if experiment_id and plan.experiment_id != experiment_id:
                continue
                
            plans.append({
                "plan_id": plan_id,
                "experiment_id": plan.experiment_id,
                "variant": plan.variant,
                "strategy": plan.strategy,
                "start_percentage": plan.start_percentage,
                "target_percentage": plan.target_percentage,
                "duration_hours": plan.total_duration_hours,
                "num_steps": len(plan.steps),
                "created_at": plan.created_at.isoformat()
            })
            
        return {
            "success": True,
            "plans": plans,
            "total": len(plans)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/executions")
async def list_executions(
    experiment_id: Optional[str] = Query(None, description="实验ID"),
    status: Optional[RampStatus] = Query(None, description="状态筛选")
) -> Dict[str, Any]:
    """
    列出执行记录
    
    获取爬坡执行历史
    """
    try:
        executions = []
        
        for exec_id, execution in ramp_service.executions.items():
            if experiment_id and execution.experiment_id != experiment_id:
                continue
            if status and execution.status != status:
                continue
                
            executions.append({
                "exec_id": exec_id,
                "plan_id": execution.plan_id,
                "experiment_id": execution.experiment_id,
                "status": execution.status,
                "current_step": execution.current_step,
                "current_percentage": execution.current_percentage,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "rollback_reason": execution.rollback_reason
            })
            
        return {
            "success": True,
            "executions": executions,
            "total": len(executions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommended-plan")
async def get_recommended_plan(request: GetRecommendedPlanRequest) -> Dict[str, Any]:
    """
    获取推荐的爬坡计划
    
    基于风险等级自动生成合适的计划
    """
    try:
        plan = await ramp_service.get_recommended_plan(
            request.experiment_id,
            request.risk_level
        )
        
        # 获取计划ID
        plan_id = None
        for pid, p in ramp_service.ramp_plans.items():
            if p == plan:
                plan_id = pid
                break
                
        return {
            "success": True,
            "plan_id": plan_id,
            "risk_level": request.risk_level,
            "recommendation": {
                "strategy": plan.strategy,
                "duration_hours": plan.total_duration_hours,
                "num_steps": len(plan.steps),
                "description": _get_risk_description(request.risk_level)
            },
            "plan": {
                "experiment_id": plan.experiment_id,
                "variant": plan.variant,
                "start_percentage": plan.start_percentage,
                "target_percentage": plan.target_percentage,
                "steps_preview": [
                    {
                        "step": s.step_number,
                        "target": s.target_percentage,
                        "duration_minutes": s.duration_minutes
                    }
                    for s in plan.steps[:3]
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-ramp")
async def quick_ramp(request: QuickRampRequest) -> Dict[str, Any]:
    """
    快速爬坡配置
    
    基于发布阶段快速创建并启动爬坡
    """
    try:
        # 根据阶段确定参数
        phase_configs = {
            RolloutPhase.CANARY: {
                "start": 0, "target": 5, "duration": 2, "steps": 3
            },
            RolloutPhase.PILOT: {
                "start": 5, "target": 20, "duration": 6, "steps": 5
            },
            RolloutPhase.BETA: {
                "start": 20, "target": 50, "duration": 12, "steps": 8
            },
            RolloutPhase.GRADUAL: {
                "start": 50, "target": 95, "duration": 24, "steps": 10
            },
            RolloutPhase.FULL: {
                "start": 95, "target": 100, "duration": 1, "steps": 1
            }
        }
        
        config = phase_configs[request.phase]
        duration = request.duration_hours or config["duration"]
        
        # 创建计划
        plan = await ramp_service.create_ramp_plan(
            experiment_id=request.experiment_id,
            strategy=RampStrategy.LINEAR,
            start_percentage=config["start"],
            target_percentage=config["target"],
            duration_hours=duration,
            num_steps=config["steps"]
        )
        
        # 获取计划ID
        plan_id = None
        for pid, p in ramp_service.ramp_plans.items():
            if p == plan:
                plan_id = pid
                break
                
        # 自动启动
        execution = await ramp_service.start_ramp(plan_id)
        
        # 获取执行ID
        exec_id = None
        for eid, e in ramp_service.executions.items():
            if e == execution:
                exec_id = eid
                break
                
        return {
            "success": True,
            "phase": request.phase,
            "plan_id": plan_id,
            "exec_id": exec_id,
            "config": {
                "start_percentage": config["start"],
                "target_percentage": config["target"],
                "duration_hours": duration,
                "num_steps": config["steps"]
            },
            "message": f"已启动{request.phase}阶段流量爬坡"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies() -> Dict[str, Any]:
    """
    列出可用的爬坡策略
    """
    strategies = [
        {
            "value": RampStrategy.LINEAR,
            "name": "线性增长",
            "description": "流量均匀线性增长",
            "use_case": "标准场景，风险可控"
        },
        {
            "value": RampStrategy.EXPONENTIAL,
            "name": "指数增长",
            "description": "开始缓慢，后期加速",
            "use_case": "低风险功能，需要快速全量"
        },
        {
            "value": RampStrategy.LOGARITHMIC,
            "name": "对数增长",
            "description": "开始快速，后期缓慢",
            "use_case": "高风险功能，需要充分验证"
        },
        {
            "value": RampStrategy.STEP,
            "name": "阶梯增长",
            "description": "分阶段跳跃式增长",
            "use_case": "需要明确阶段验证"
        },
        {
            "value": RampStrategy.CUSTOM,
            "name": "自定义曲线",
            "description": "S型曲线或其他自定义",
            "use_case": "特殊需求场景"
        }
    ]
    
    return {
        "success": True,
        "strategies": strategies
    }


@router.get("/phases")
async def list_phases() -> Dict[str, Any]:
    """
    列出发布阶段
    """
    phases = [
        {
            "value": RolloutPhase.CANARY,
            "name": "金丝雀发布",
            "range": "1-5%",
            "description": "小流量验证，快速发现问题"
        },
        {
            "value": RolloutPhase.PILOT,
            "name": "试点发布",
            "range": "5-20%",
            "description": "扩大验证范围，收集更多数据"
        },
        {
            "value": RolloutPhase.BETA,
            "name": "Beta发布",
            "range": "20-50%",
            "description": "大规模测试，验证稳定性"
        },
        {
            "value": RolloutPhase.GRADUAL,
            "name": "渐进发布",
            "range": "50-95%",
            "description": "逐步推广，监控指标"
        },
        {
            "value": RolloutPhase.FULL,
            "name": "全量发布",
            "range": "95-100%",
            "description": "完成发布，全流量切换"
        }
    ]
    
    return {
        "success": True,
        "phases": phases
    }


@router.get("/current-phase/{exec_id}")
async def get_current_phase(exec_id: str) -> Dict[str, Any]:
    """
    获取当前发布阶段
    """
    try:
        if exec_id not in ramp_service.executions:
            raise HTTPException(status_code=404, detail="执行不存在")
            
        execution = ramp_service.executions[exec_id]
        phase = ramp_service.get_phase_recommendation(execution.current_percentage)
        
        return {
            "success": True,
            "current_percentage": execution.current_percentage,
            "phase": phase,
            "phase_name": _get_phase_name(phase)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_risk_description(risk_level: str) -> str:
    """获取风险等级描述"""
    descriptions = {
        "low": "低风险配置：快速爬坡，适合稳定功能",
        "medium": "中等风险配置：平衡速度和安全性",
        "high": "高风险配置：缓慢谨慎，充分验证每个阶段"
    }
    return descriptions.get(risk_level, "未知风险等级")


def _get_phase_name(phase: RolloutPhase) -> str:
    """获取阶段名称"""
    names = {
        RolloutPhase.CANARY: "金丝雀发布",
        RolloutPhase.PILOT: "试点发布",
        RolloutPhase.BETA: "Beta发布",
        RolloutPhase.GRADUAL: "渐进发布",
        RolloutPhase.FULL: "全量发布"
    }
    return names.get(phase, "未知阶段")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "traffic_ramp",
        "status": "healthy",
        "active_ramps": len(ramp_service.active_ramps),
        "total_plans": len(ramp_service.ramp_plans),
        "total_executions": len(ramp_service.executions)
    }