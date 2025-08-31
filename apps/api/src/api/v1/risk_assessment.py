"""
风险评估和回滚API端点
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ...services.risk_assessment_service import (
    RiskAssessmentService,
    RiskLevel,
    RiskCategory,
    RollbackStrategy
)


router = APIRouter(prefix="/risk-assessment", tags=["Risk Assessment"])

# 服务实例
risk_service = RiskAssessmentService()


class AssessRiskRequest(BaseModel):
    """评估风险请求"""
    experiment_id: str = Field(..., description="实验ID")
    include_predictions: bool = Field(True, description="是否包含预测性分析")


class CreateRollbackPlanRequest(BaseModel):
    """创建回滚计划请求"""
    experiment_id: str = Field(..., description="实验ID")
    strategy: Optional[RollbackStrategy] = Field(None, description="回滚策略")
    auto_execute: bool = Field(False, description="是否自动执行")


class ExecuteRollbackRequest(BaseModel):
    """执行回滚请求"""
    plan_id: str = Field(..., description="回滚计划ID")
    force: bool = Field(False, description="是否强制执行")


class MonitorRiskRequest(BaseModel):
    """监控风险请求"""
    experiment_id: str = Field(..., description="实验ID")
    check_interval_minutes: int = Field(5, ge=1, description="检查间隔(分钟)")


@router.post("/assess")
async def assess_risk(request: AssessRiskRequest) -> Dict[str, Any]:
    """
    评估实验风险
    
    分析实验的各个风险维度并给出综合评估
    """
    try:
        assessment = await risk_service.assess_risk(
            experiment_id=request.experiment_id,
            include_predictions=request.include_predictions
        )
        
        return {
            "success": True,
            "assessment": {
                "experiment_id": assessment.experiment_id,
                "assessment_time": assessment.assessment_time.isoformat(),
                "overall_risk_level": assessment.overall_risk_level,
                "overall_risk_score": assessment.overall_risk_score,
                "requires_rollback": assessment.requires_rollback,
                "rollback_strategy": assessment.rollback_strategy,
                "confidence": assessment.confidence,
                "risk_factors": [
                    {
                        "category": f.category,
                        "name": f.name,
                        "description": f.description,
                        "risk_score": f.risk_score,
                        "severity": f.severity,
                        "likelihood": f.likelihood,
                        "impact": f.impact,
                        "mitigation": f.mitigation
                    }
                    for f in assessment.risk_factors[:10]  # 返回前10个风险因素
                ],
                "recommendations": assessment.recommendations
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{experiment_id}")
async def get_risk_history(
    experiment_id: str,
    limit: int = Query(10, ge=1, le=100, description="返回记录数")
) -> Dict[str, Any]:
    """
    获取风险评估历史
    """
    try:
        assessments = risk_service.assessments.get(experiment_id, [])
        
        # 获取最近的评估记录
        recent_assessments = assessments[-limit:]
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "assessments": [
                {
                    "assessment_time": a.assessment_time.isoformat(),
                    "risk_level": a.overall_risk_level,
                    "risk_score": a.overall_risk_score,
                    "requires_rollback": a.requires_rollback,
                    "num_risk_factors": len(a.risk_factors)
                }
                for a in recent_assessments
            ],
            "total_assessments": len(assessments)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback-plan")
async def create_rollback_plan(request: CreateRollbackPlanRequest) -> Dict[str, Any]:
    """
    创建回滚计划
    
    基于风险评估创建回滚计划
    """
    try:
        # 先进行风险评估
        assessment = await risk_service.assess_risk(request.experiment_id)
        
        # 如果指定了策略，覆盖评估结果
        if request.strategy:
            assessment.rollback_strategy = request.strategy
            
        # 创建回滚计划
        plan = await risk_service.create_rollback_plan(
            request.experiment_id,
            assessment
        )
        
        # 获取计划ID
        plan_id = None
        for pid, p in risk_service.rollback_plans.items():
            if p == plan:
                plan_id = pid
                break
                
        return {
            "success": True,
            "plan_id": plan_id,
            "plan": {
                "experiment_id": plan.experiment_id,
                "trigger_reason": plan.trigger_reason,
                "strategy": plan.strategy,
                "steps": plan.steps,
                "estimated_duration_minutes": plan.estimated_duration_minutes,
                "auto_execute": plan.auto_execute,
                "approval_required": plan.approval_required,
                "created_at": plan.created_at.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rollback/execute")
async def execute_rollback(request: ExecuteRollbackRequest) -> Dict[str, Any]:
    """
    执行回滚
    
    执行指定的回滚计划
    """
    try:
        execution = await risk_service.execute_rollback(
            request.plan_id,
            request.force
        )
        
        return {
            "success": True,
            "execution": {
                "plan_id": execution.plan_id,
                "experiment_id": execution.experiment_id,
                "started_at": execution.started_at.isoformat(),
                "status": execution.status,
                "steps_completed": execution.steps_completed,
                "total_steps": execution.total_steps,
                "errors": execution.errors
            }
        }
        
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rollback/status/{exec_id}")
async def get_rollback_status(exec_id: str) -> Dict[str, Any]:
    """
    获取回滚执行状态
    """
    try:
        if exec_id not in risk_service.rollback_executions:
            raise HTTPException(status_code=404, detail="执行记录不存在")
            
        execution = risk_service.rollback_executions[exec_id]
        
        return {
            "success": True,
            "status": {
                "exec_id": exec_id,
                "experiment_id": execution.experiment_id,
                "status": execution.status,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "progress": f"{execution.steps_completed}/{execution.total_steps}",
                "errors": execution.errors,
                "metrics_comparison": {
                    "before": execution.metrics_before,
                    "after": execution.metrics_after
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor")
async def start_monitoring(
    request: MonitorRiskRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    启动风险持续监控
    
    后台持续监控实验风险
    """
    try:
        # 启动后台监控任务
        background_tasks.add_task(
            risk_service.monitor_risk_continuously,
            request.experiment_id,
            request.check_interval_minutes
        )
        
        return {
            "success": True,
            "message": "风险监控已启动",
            "experiment_id": request.experiment_id,
            "check_interval_minutes": request.check_interval_minutes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/thresholds")
async def get_risk_thresholds() -> Dict[str, Any]:
    """
    获取风险阈值配置
    """
    return {
        "success": True,
        "thresholds": risk_service.risk_thresholds
    }


@router.put("/thresholds")
async def update_risk_thresholds(
    category: str = Query(..., description="风险类别"),
    metric: str = Query(..., description="指标名称"),
    value: float = Query(..., description="阈值")
) -> Dict[str, Any]:
    """
    更新风险阈值
    """
    try:
        if category not in risk_service.risk_thresholds:
            raise HTTPException(status_code=400, detail="无效的风险类别")
            
        risk_service.risk_thresholds[category][metric] = value
        
        return {
            "success": True,
            "message": "阈值已更新",
            "category": category,
            "metric": metric,
            "new_value": value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-levels")
async def list_risk_levels() -> Dict[str, Any]:
    """
    列出风险等级
    """
    levels = [
        {
            "value": RiskLevel.MINIMAL,
            "name": "极低风险",
            "score_range": "0-0.2",
            "color": "green",
            "action": "继续监控"
        },
        {
            "value": RiskLevel.LOW,
            "name": "低风险",
            "score_range": "0.2-0.4",
            "color": "blue",
            "action": "正常运行"
        },
        {
            "value": RiskLevel.MEDIUM,
            "name": "中等风险",
            "score_range": "0.4-0.6",
            "color": "yellow",
            "action": "增加监控"
        },
        {
            "value": RiskLevel.HIGH,
            "name": "高风险",
            "score_range": "0.6-0.8",
            "color": "orange",
            "action": "考虑回滚"
        },
        {
            "value": RiskLevel.CRITICAL,
            "name": "严重风险",
            "score_range": "0.8-1.0",
            "color": "red",
            "action": "立即回滚"
        }
    ]
    
    return {
        "success": True,
        "levels": levels
    }


@router.get("/categories")
async def list_risk_categories() -> Dict[str, Any]:
    """
    列出风险类别
    """
    categories = [
        {
            "value": RiskCategory.PERFORMANCE,
            "name": "性能风险",
            "description": "系统性能相关的风险",
            "metrics": ["latency", "error_rate", "timeout_rate"]
        },
        {
            "value": RiskCategory.BUSINESS,
            "name": "业务风险",
            "description": "业务指标相关的风险",
            "metrics": ["conversion_rate", "revenue", "user_retention"]
        },
        {
            "value": RiskCategory.TECHNICAL,
            "name": "技术风险",
            "description": "技术实现相关的风险",
            "metrics": ["cpu_usage", "memory_usage", "db_connections"]
        },
        {
            "value": RiskCategory.USER_EXPERIENCE,
            "name": "用户体验风险",
            "description": "用户体验相关的风险",
            "metrics": ["bounce_rate", "page_load_time", "crash_rate"]
        },
        {
            "value": RiskCategory.DATA_QUALITY,
            "name": "数据质量风险",
            "description": "数据准确性相关的风险",
            "metrics": ["missing_rate", "duplicate_rate", "srm"]
        }
    ]
    
    return {
        "success": True,
        "categories": categories
    }


@router.get("/strategies")
async def list_rollback_strategies() -> Dict[str, Any]:
    """
    列出回滚策略
    """
    strategies = [
        {
            "value": RollbackStrategy.IMMEDIATE,
            "name": "立即回滚",
            "description": "立即停止实验并回滚",
            "duration": "< 5分钟",
            "use_case": "严重故障或数据问题"
        },
        {
            "value": RollbackStrategy.GRADUAL,
            "name": "渐进回滚",
            "description": "逐步减少流量直至停止",
            "duration": "30-60分钟",
            "use_case": "性能问题或指标恶化"
        },
        {
            "value": RollbackStrategy.PARTIAL,
            "name": "部分回滚",
            "description": "只回滚部分功能或用户",
            "duration": "根据情况",
            "use_case": "特定功能问题"
        },
        {
            "value": RollbackStrategy.MANUAL,
            "name": "手动确认",
            "description": "需要人工确认后执行",
            "duration": "人工决定",
            "use_case": "非紧急情况"
        }
    ]
    
    return {
        "success": True,
        "strategies": strategies
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "risk_assessment",
        "status": "healthy",
        "total_assessments": sum(len(a) for a in risk_service.assessments.values()),
        "active_rollback_plans": len(risk_service.rollback_plans),
        "rollback_executions": len(risk_service.rollback_executions)
    }