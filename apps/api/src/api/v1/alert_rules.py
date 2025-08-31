"""
告警规则API端点
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ...services.alert_rules_service import (
    AlertRulesEngine,
    AlertRule,
    AlertCondition,
    Alert,
    AlertSeverity,
    AlertChannel,
    RuleOperator,
    RuleAggregation,
    AlertRuleBuilder,
    AlertRuleTemplates
)


router = APIRouter(prefix="/alert-rules", tags=["Alert Rules"])

# 服务实例
alert_engine = AlertRulesEngine()


class CreateRuleRequest(BaseModel):
    """创建规则请求"""
    id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: str = Field(..., description="规则描述")
    experiment_id: Optional[str] = Field(None, description="实验ID")
    metric_name: Optional[str] = Field(None, description="指标名称")
    conditions: List[Dict[str, Any]] = Field(..., description="告警条件")
    aggregation: RuleAggregation = Field(RuleAggregation.ALL, description="聚合方式")
    severity: AlertSeverity = Field(AlertSeverity.WARNING, description="严重级别")
    channels: List[AlertChannel] = Field([AlertChannel.DASHBOARD], description="通知渠道")
    enabled: bool = Field(True, description="是否启用")
    cooldown_minutes: int = Field(5, ge=1, description="冷却时间(分钟)")
    max_alerts_per_hour: int = Field(10, ge=1, description="每小时最大告警数")


class UpdateRuleRequest(BaseModel):
    """更新规则请求"""
    name: Optional[str] = Field(None, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    conditions: Optional[List[Dict[str, Any]]] = Field(None, description="告警条件")
    aggregation: Optional[RuleAggregation] = Field(None, description="聚合方式")
    severity: Optional[AlertSeverity] = Field(None, description="严重级别")
    channels: Optional[List[AlertChannel]] = Field(None, description="通知渠道")
    enabled: Optional[bool] = Field(None, description="是否启用")
    cooldown_minutes: Optional[int] = Field(None, ge=1, description="冷却时间")
    max_alerts_per_hour: Optional[int] = Field(None, ge=1, description="每小时最大告警数")


class EvaluateRequest(BaseModel):
    """评估请求"""
    data: Dict[str, Any] = Field(..., description="要评估的数据")
    rule_ids: Optional[List[str]] = Field(None, description="指定规则ID列表")


class CreateTemplateRuleRequest(BaseModel):
    """从模板创建规则请求"""
    template_type: str = Field(..., description="模板类型")
    metric_name: Optional[str] = Field(None, description="指标名称")
    experiment_id: Optional[str] = Field(None, description="实验ID")
    threshold: Optional[float] = Field(None, description="阈值")


class NotificationConfigRequest(BaseModel):
    """通知配置请求"""
    channel: AlertChannel = Field(..., description="通知渠道")
    config: Dict[str, Any] = Field(..., description="渠道配置")


@router.post("/rules")
async def create_rule(request: CreateRuleRequest) -> Dict[str, Any]:
    """
    创建告警规则
    
    定义新的告警规则和触发条件
    """
    try:
        # 构建条件列表
        conditions = []
        for cond_dict in request.conditions:
            condition = AlertCondition(
                field=cond_dict["field"],
                operator=RuleOperator(cond_dict["operator"]),
                value=cond_dict["value"],
                description=cond_dict.get("description")
            )
            conditions.append(condition)
            
        # 创建规则
        rule = AlertRule(
            id=request.id,
            name=request.name,
            description=request.description,
            experiment_id=request.experiment_id,
            metric_name=request.metric_name,
            conditions=conditions,
            aggregation=request.aggregation,
            severity=request.severity,
            channels=request.channels,
            enabled=request.enabled,
            cooldown_minutes=request.cooldown_minutes,
            max_alerts_per_hour=request.max_alerts_per_hour
        )
        
        success = await alert_engine.add_rule(rule)
        
        return {
            "success": success,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity,
                "conditions_count": len(rule.conditions)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}")
async def update_rule(rule_id: str, request: UpdateRuleRequest) -> Dict[str, Any]:
    """
    更新告警规则
    
    修改现有规则的配置
    """
    try:
        updates = request.dict(exclude_unset=True)
        
        # 如果更新条件，需要转换格式
        if "conditions" in updates:
            conditions = []
            for cond_dict in updates["conditions"]:
                condition = AlertCondition(
                    field=cond_dict["field"],
                    operator=RuleOperator(cond_dict["operator"]),
                    value=cond_dict["value"],
                    description=cond_dict.get("description")
                )
                conditions.append(condition)
            updates["conditions"] = conditions
            
        success = await alert_engine.update_rule(rule_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="规则不存在")
            
        return {
            "success": True,
            "message": "规则已更新"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str) -> Dict[str, Any]:
    """
    删除告警规则
    """
    try:
        success = await alert_engine.delete_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="规则不存在")
            
        return {
            "success": True,
            "message": "规则已删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def list_rules(
    experiment_id: Optional[str] = Query(None, description="实验ID"),
    enabled: Optional[bool] = Query(None, description="是否启用")
) -> Dict[str, Any]:
    """
    列出告警规则
    
    获取所有或筛选的告警规则
    """
    try:
        rules = list(alert_engine.rules.values())
        
        # 筛选
        if experiment_id is not None:
            rules = [r for r in rules if r.experiment_id == experiment_id]
        if enabled is not None:
            rules = [r for r in rules if r.enabled == enabled]
            
        return {
            "success": True,
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "experiment_id": r.experiment_id,
                    "metric_name": r.metric_name,
                    "severity": r.severity,
                    "enabled": r.enabled,
                    "conditions_count": len(r.conditions),
                    "channels": r.channels,
                    "created_at": r.created_at.isoformat(),
                    "updated_at": r.updated_at.isoformat()
                }
                for r in rules
            ],
            "total": len(rules)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str) -> Dict[str, Any]:
    """
    获取规则详情
    """
    try:
        if rule_id not in alert_engine.rules:
            raise HTTPException(status_code=404, detail="规则不存在")
            
        rule = alert_engine.rules[rule_id]
        
        return {
            "success": True,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "experiment_id": rule.experiment_id,
                "metric_name": rule.metric_name,
                "conditions": [
                    {
                        "field": c.field,
                        "operator": c.operator.value,
                        "value": c.value,
                        "description": c.description
                    }
                    for c in rule.conditions
                ],
                "aggregation": rule.aggregation,
                "severity": rule.severity,
                "channels": rule.channels,
                "enabled": rule.enabled,
                "cooldown_minutes": rule.cooldown_minutes,
                "max_alerts_per_hour": rule.max_alerts_per_hour,
                "metadata": rule.metadata,
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_rules(request: EvaluateRequest) -> Dict[str, Any]:
    """
    评估规则
    
    根据数据触发告警
    """
    try:
        alerts = await alert_engine.evaluate_rules(request.data)
        
        return {
            "success": True,
            "alerts_triggered": len(alerts),
            "alerts": [
                {
                    "id": a.id,
                    "rule_name": a.rule_name,
                    "severity": a.severity,
                    "title": a.title,
                    "description": a.description,
                    "triggered_at": a.triggered_at.isoformat(),
                    "notifications_sent": a.notifications_sent
                }
                for a in alerts
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates")
async def create_from_template(request: CreateTemplateRuleRequest) -> Dict[str, Any]:
    """
    从模板创建规则
    
    使用预定义模板快速创建规则
    """
    try:
        rule = None
        
        if request.template_type == "metric_spike":
            if not request.metric_name or request.threshold is None:
                raise HTTPException(status_code=400, detail="需要metric_name和threshold参数")
            rule = AlertRuleTemplates.metric_spike_rule(request.metric_name, request.threshold)
            
        elif request.template_type == "metric_drop":
            if not request.metric_name or request.threshold is None:
                raise HTTPException(status_code=400, detail="需要metric_name和threshold参数")
            rule = AlertRuleTemplates.metric_drop_rule(request.metric_name, request.threshold)
            
        elif request.template_type == "srm":
            if not request.experiment_id:
                raise HTTPException(status_code=400, detail="需要experiment_id参数")
            rule = AlertRuleTemplates.srm_rule(request.experiment_id)
            
        elif request.template_type == "data_quality":
            rule = AlertRuleTemplates.data_quality_rule()
            
        else:
            raise HTTPException(status_code=400, detail="未知的模板类型")
            
        await alert_engine.add_rule(rule)
        
        return {
            "success": True,
            "rule": {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "template_type": request.template_type
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates() -> Dict[str, Any]:
    """
    列出可用模板
    """
    templates = [
        {
            "type": "metric_spike",
            "name": "指标突增",
            "description": "检测指标值突然增加",
            "required_params": ["metric_name", "threshold"]
        },
        {
            "type": "metric_drop",
            "name": "指标突降",
            "description": "检测指标值突然下降",
            "required_params": ["metric_name", "threshold"]
        },
        {
            "type": "srm",
            "name": "样本比例不匹配",
            "description": "检测实验分组比例异常",
            "required_params": ["experiment_id"]
        },
        {
            "type": "data_quality",
            "name": "数据质量",
            "description": "检测数据质量问题",
            "required_params": []
        }
    ]
    
    return {
        "success": True,
        "templates": templates
    }


@router.get("/alerts")
async def list_alerts(
    experiment_id: Optional[str] = Query(None, description="实验ID"),
    severity: Optional[AlertSeverity] = Query(None, description="严重级别"),
    active_only: bool = Query(True, description="仅显示活跃告警")
) -> Dict[str, Any]:
    """
    列出告警
    
    获取当前告警列表
    """
    try:
        if active_only:
            alerts = await alert_engine.get_active_alerts(experiment_id, severity)
        else:
            alerts = list(alert_engine.alerts.values())
            if experiment_id:
                alerts = [a for a in alerts if a.data.get("experiment_id") == experiment_id]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
                
        return {
            "success": True,
            "alerts": [
                {
                    "id": a.id,
                    "rule_id": a.rule_id,
                    "rule_name": a.rule_name,
                    "severity": a.severity,
                    "title": a.title,
                    "description": a.description,
                    "triggered_at": a.triggered_at.isoformat(),
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                    "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
                    "acknowledged_by": a.acknowledged_by,
                    "is_active": a.resolved_at is None
                }
                for a in alerts
            ],
            "total": len(alerts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, user_id: str = Query(..., description="用户ID")) -> Dict[str, Any]:
    """
    确认告警
    """
    try:
        success = await alert_engine.acknowledge_alert(alert_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="告警不存在")
            
        return {
            "success": True,
            "message": "告警已确认"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """
    解决告警
    """
    try:
        success = await alert_engine.resolve_alert(alert_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="告警不存在")
            
        return {
            "success": True,
            "message": "告警已解决"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间")
) -> Dict[str, Any]:
    """
    获取告警统计
    """
    try:
        stats = await alert_engine.get_alert_statistics(start_time, end_time)
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def test_rule(rule_id: str, test_data: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    测试规则
    
    使用测试数据验证规则是否会触发
    """
    try:
        if rule_id not in alert_engine.rules:
            raise HTTPException(status_code=404, detail="规则不存在")
            
        rule = alert_engine.rules[rule_id]
        
        # 评估规则
        would_trigger = rule.evaluate(test_data)
        
        # 详细评估每个条件
        condition_results = []
        for condition in rule.conditions:
            result = condition.evaluate(test_data)
            condition_results.append({
                "field": condition.field,
                "operator": condition.operator.value,
                "expected": condition.value,
                "actual": condition._get_nested_value(test_data, condition.field),
                "result": result
            })
            
        return {
            "success": True,
            "would_trigger": would_trigger,
            "rule_name": rule.name,
            "aggregation": rule.aggregation,
            "condition_results": condition_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    return {
        "success": True,
        "service": "alert_rules",
        "status": "healthy",
        "active_rules": len([r for r in alert_engine.rules.values() if r.enabled]),
        "total_rules": len(alert_engine.rules),
        "active_alerts": len(await alert_engine.get_active_alerts())
    }