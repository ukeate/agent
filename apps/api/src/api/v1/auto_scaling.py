"""
自动扩量API端点
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ...services.auto_scaling_service import (
    AutoScalingService,
    ScalingMode,
    ScalingDirection,
    ScalingTrigger,
    ScalingCondition,
    ScalingTemplates
)


router = APIRouter(prefix="/auto-scaling", tags=["Auto Scaling"])

# 服务实例
scaling_service = AutoScalingService()


class CreateScalingRuleRequest(BaseModel):
    """创建扩量规则请求"""
    experiment_id: str = Field(..., description="实验ID")
    name: str = Field(..., description="规则名称")
    mode: ScalingMode = Field(ScalingMode.BALANCED, description="扩量模式")
    variant: str = Field("treatment", description="变体名称")
    description: str = Field("", description="规则描述")
    scale_increment: float = Field(10.0, ge=1, le=50, description="扩量增量(%)")
    scale_decrement: float = Field(5.0, ge=1, le=50, description="缩量减量(%)")
    min_percentage: float = Field(1.0, ge=0, le=100, description="最小流量(%)")
    max_percentage: float = Field(100.0, ge=0, le=100, description="最大流量(%)")
    cooldown_minutes: int = Field(30, ge=5, description="冷却时间(分钟)")
    enabled: bool = Field(True, description="是否启用")


class CreateConditionRequest(BaseModel):
    """创建条件请求"""
    trigger: ScalingTrigger = Field(..., description="触发器类型")
    metric_name: Optional[str] = Field(None, description="指标名称")
    operator: str = Field(..., description="操作符")
    threshold: float = Field(..., description="阈值")
    confidence_level: float = Field(0.95, ge=0.5, le=0.999, description="置信水平")
    min_sample_size: int = Field(1000, ge=100, description="最小样本量")


class StartScalingRequest(BaseModel):
    """启动扩量请求"""
    rule_id: str = Field(..., description="规则ID")


class SimulateScalingRequest(BaseModel):
    """模拟扩量请求"""
    experiment_id: str = Field(..., description="实验ID")
    days: int = Field(7, ge=1, le=30, description="模拟天数")


@router.post("/rules")
async def create_scaling_rule(request: CreateScalingRuleRequest) -> Dict[str, Any]:
    """
    创建自动扩量规则
    
    定义基于指标的自动流量调整规则
    """
    try:
        rule = await scaling_service.create_scaling_rule(
            experiment_id=request.experiment_id,
            name=request.name,
            mode=request.mode,
            variant=request.variant,
            description=request.description,
            scale_increment=request.scale_increment,
            scale_decrement=request.scale_decrement,
            min_percentage=request.min_percentage,
            max_percentage=request.max_percentage,
            cooldown_minutes=request.cooldown_minutes,
            enabled=request.enabled
        )
        
        return {
            "success": True,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "experiment_id": rule.experiment_id,
                "mode": rule.mode,
                "enabled": rule.enabled,
                "scale_increment": rule.scale_increment,
                "scale_decrement": rule.scale_decrement,
                "min_percentage": rule.min_percentage,
                "max_percentage": rule.max_percentage
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rules/{rule_id}/conditions")
async def add_condition(
    rule_id: str,
    condition_type: str = Query(..., description="Condition type: scale_up or scale_down"),
    request: CreateConditionRequest = ...
) -> Dict[str, Any]:
    """
    添加扩量条件
    
    为规则添加触发条件
    """
    try:
        if rule_id not in scaling_service.rules:
            raise HTTPException(status_code=404, detail="规则不存在")
            
        rule = scaling_service.rules[rule_id]
        
        condition = ScalingCondition(
            trigger=request.trigger,
            metric_name=request.metric_name,
            operator=request.operator,
            threshold=request.threshold,
            confidence_level=request.confidence_level,
            min_sample_size=request.min_sample_size
        )
        
        if condition_type == "scale_up":
            rule.scale_up_conditions.append(condition)
        elif condition_type == "scale_down":
            rule.scale_down_conditions.append(condition)
        else:
            raise HTTPException(status_code=400, detail="无效的条件类型")
            
        return {
            "success": True,
            "message": f"条件已添加到{condition_type}列表"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_auto_scaling(
    request: StartScalingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    启动自动扩量
    
    开始监控并自动调整流量
    """
    try:
        success = await scaling_service.start_auto_scaling(request.rule_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法启动自动扩量")
            
        return {
            "success": True,
            "message": "自动扩量已启动",
            "rule_id": request.rule_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{rule_id}")
async def stop_auto_scaling(rule_id: str) -> Dict[str, Any]:
    """
    停止自动扩量
    """
    try:
        success = await scaling_service.stop_auto_scaling(rule_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="无法停止自动扩量")
            
        return {
            "success": True,
            "message": "自动扩量已停止"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def list_rules(
    experiment_id: Optional[str] = Query(None, description="实验ID")
) -> Dict[str, Any]:
    """
    列出扩量规则
    """
    try:
        rules = []
        
        for rule_id, rule in scaling_service.rules.items():
            if experiment_id and rule.experiment_id != experiment_id:
                continue
                
            rules.append({
                "id": rule.id,
                "name": rule.name,
                "experiment_id": rule.experiment_id,
                "mode": rule.mode,
                "enabled": rule.enabled,
                "scale_up_conditions_count": len(rule.scale_up_conditions),
                "scale_down_conditions_count": len(rule.scale_down_conditions),
                "created_at": rule.created_at.isoformat()
            })
            
        return {
            "success": True,
            "rules": rules,
            "total": len(rules)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{experiment_id}")
async def get_scaling_history(experiment_id: str) -> Dict[str, Any]:
    """
    获取扩量历史
    """
    try:
        history = scaling_service.histories.get(experiment_id)
        
        if not history:
            return {
                "success": True,
                "history": None,
                "message": "无扩量历史"
            }
            
        return {
            "success": True,
            "history": {
                "experiment_id": history.experiment_id,
                "current_percentage": history.current_percentage,
                "last_scaled_at": history.last_scaled_at.isoformat() if history.last_scaled_at else None,
                "total_scale_ups": history.total_scale_ups,
                "total_scale_downs": history.total_scale_downs,
                "recent_decisions": [
                    {
                        "timestamp": d.timestamp.isoformat(),
                        "direction": d.direction,
                        "from": d.current_percentage,
                        "to": d.target_percentage,
                        "reason": d.reason,
                        "confidence": d.confidence
                    }
                    for d in history.decisions[-10:]  # 最近10条
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{experiment_id}")
async def get_recommendations(experiment_id: str) -> Dict[str, Any]:
    """
    获取扩量建议
    
    基于当前指标提供扩量建议
    """
    try:
        recommendations = await scaling_service.get_scaling_recommendations(
            experiment_id
        )
        
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate")
async def simulate_scaling(request: SimulateScalingRequest) -> Dict[str, Any]:
    """
    模拟扩量过程
    
    预测未来几天的扩量情况
    """
    try:
        simulations = await scaling_service.simulate_scaling(
            request.experiment_id,
            request.days
        )
        
        return {
            "success": True,
            "simulations": simulations,
            "summary": {
                "start_percentage": simulations[0]["current_percentage"] if simulations else 0,
                "end_percentage": simulations[-1]["new_percentage"] if simulations else 0,
                "total_days": len(simulations),
                "scale_ups": sum(1 for s in simulations if s["action"] == "scale_up")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/safe")
async def create_safe_template(experiment_id: str) -> Dict[str, Any]:
    """
    创建安全扩量模板
    
    适合高风险实验的保守策略
    """
    try:
        rule = await ScalingTemplates.create_safe_scaling_rule(
            scaling_service,
            experiment_id
        )
        
        return {
            "success": True,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "mode": rule.mode,
                "description": "安全扩量模板：小步快跑，充分验证"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/aggressive")
async def create_aggressive_template(experiment_id: str) -> Dict[str, Any]:
    """
    创建激进扩量模板
    
    适合低风险实验的快速扩量
    """
    try:
        rule = await ScalingTemplates.create_aggressive_scaling_rule(
            scaling_service,
            experiment_id
        )
        
        return {
            "success": True,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "mode": rule.mode,
                "description": "激进扩量模板：快速扩量，迅速验证"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modes")
async def list_scaling_modes() -> Dict[str, Any]:
    """
    列出扩量模式
    """
    modes = [
        {
            "value": ScalingMode.AGGRESSIVE,
            "name": "激进模式",
            "description": "快速扩量，适合低风险功能",
            "scale_increment": 20,
            "cooldown_minutes": 15
        },
        {
            "value": ScalingMode.CONSERVATIVE,
            "name": "保守模式",
            "description": "缓慢扩量，适合高风险功能",
            "scale_increment": 5,
            "cooldown_minutes": 60
        },
        {
            "value": ScalingMode.BALANCED,
            "name": "平衡模式",
            "description": "平衡速度和安全性",
            "scale_increment": 10,
            "cooldown_minutes": 30
        },
        {
            "value": ScalingMode.ADAPTIVE,
            "name": "自适应模式",
            "description": "根据指标动态调整策略",
            "scale_increment": "dynamic",
            "cooldown_minutes": "dynamic"
        }
    ]
    
    return {
        "success": True,
        "modes": modes
    }


@router.get("/triggers")
async def list_triggers() -> Dict[str, Any]:
    """
    列出触发器类型
    """
    triggers = [
        {
            "value": ScalingTrigger.METRIC_THRESHOLD,
            "name": "指标阈值",
            "description": "当指标达到特定阈值时触发",
            "required_params": ["metric_name", "operator", "threshold"]
        },
        {
            "value": ScalingTrigger.STATISTICAL_SIGNIFICANCE,
            "name": "统计显著性",
            "description": "当达到统计显著时触发",
            "required_params": ["confidence_level"]
        },
        {
            "value": ScalingTrigger.SAMPLE_SIZE,
            "name": "样本量",
            "description": "当样本量达标时触发",
            "required_params": ["min_sample_size"]
        },
        {
            "value": ScalingTrigger.CONFIDENCE_INTERVAL,
            "name": "置信区间",
            "description": "基于置信区间判断",
            "required_params": ["confidence_level"]
        },
        {
            "value": ScalingTrigger.TIME_BASED,
            "name": "基于时间",
            "description": "定时触发扩量",
            "required_params": ["evaluation_window_minutes"]
        }
    ]
    
    return {
        "success": True,
        "triggers": triggers
    }


@router.get("/status")
async def get_scaling_status() -> Dict[str, Any]:
    """
    获取扩量服务状态
    """
    active_rules = len(scaling_service.active_monitors)
    total_rules = len(scaling_service.rules)
    
    active_experiments = []
    for rule_id in scaling_service.active_monitors:
        if rule_id in scaling_service.rules:
            rule = scaling_service.rules[rule_id]
            active_experiments.append({
                "experiment_id": rule.experiment_id,
                "rule_name": rule.name,
                "mode": rule.mode
            })
            
    return {
        "success": True,
        "status": {
            "active_rules": active_rules,
            "total_rules": total_rules,
            "active_experiments": active_experiments
        }
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "auto_scaling",
        "status": "healthy",
        "active_monitors": len(scaling_service.active_monitors),
        "total_rules": len(scaling_service.rules)
    }