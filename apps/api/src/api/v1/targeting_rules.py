"""
定向规则管理API端点
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import Field
from src.api.base_model import ApiBaseModel
from src.services.targeting_rules_engine import (
    TargetingRulesEngine,
    TargetingRule,
    RuleCondition,
    CompositeCondition,
    RuleType,
    TargetingRule,
    RuleCondition,
    CompositeCondition,
    RuleType,
    RuleOperator,
    LogicalOperator
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/targeting", tags=["targeting-rules"])

# 请求模型
class RuleConditionRequest(ApiBaseModel):
    field: str
    operator: str
    value: Any
    case_sensitive: bool = True

class CompositeConditionRequest(ApiBaseModel):
    logical_operator: str
    conditions: List[Any]  # 可以是RuleConditionRequest或CompositeConditionRequest

class CreateRuleRequest(ApiBaseModel):
    rule_id: str
    name: str
    description: str
    rule_type: str
    condition: Dict[str, Any]
    priority: int = 0
    is_active: bool = True
    experiment_ids: List[str] = Field(default_factory=list)
    variant_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UpdateRuleRequest(ApiBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    experiment_ids: Optional[List[str]] = None
    variant_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class EvaluateUserRequest(ApiBaseModel):
    user_id: str
    user_context: Dict[str, Any]
    experiment_id: Optional[str] = None

class BatchEvaluateRequest(ApiBaseModel):
    user_contexts: List[Dict[str, Any]]
    experiment_id: Optional[str] = None

# 全局规则引擎实例
_rules_engine: Optional[TargetingRulesEngine] = None

def get_rules_engine() -> TargetingRulesEngine:
    """获取规则引擎实例"""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = TargetingRulesEngine()
    return _rules_engine

def _parse_condition(condition_data: Dict[str, Any]) -> Any:
    """解析条件数据"""
    try:
        if "logical_operator" in condition_data:
            # 复合条件
            logical_op = LogicalOperator(condition_data["logical_operator"])
            conditions = []
            
            for cond_data in condition_data["conditions"]:
                conditions.append(_parse_condition(cond_data))
            
            return CompositeCondition(
                logical_operator=logical_op,
                conditions=conditions
            )
        else:
            # 单个条件
            return RuleCondition(
                field=condition_data["field"],
                operator=RuleOperator(condition_data["operator"]),
                value=condition_data["value"],
                case_sensitive=condition_data.get("case_sensitive", True)
            )
    except Exception as e:
        raise ValueError(f"Invalid condition format: {str(e)}")

@router.post("/rules", status_code=status.HTTP_201_CREATED)
async def create_targeting_rule(
    request: CreateRuleRequest,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """创建定向规则"""
    try:
        # 解析条件
        condition = _parse_condition(request.condition)
        
        # 创建规则对象
        rule = TargetingRule(
            rule_id=request.rule_id,
            name=request.name,
            description=request.description,
            rule_type=RuleType(request.rule_type),
            condition=condition,
            priority=request.priority,
            is_active=request.is_active,
            experiment_ids=request.experiment_ids,
            variant_ids=request.variant_ids,
            metadata=request.metadata
        )
        
        # 添加规则
        success = engine.add_rule(rule)
        
        if success:
            return {
                "message": "Targeting rule created successfully",
                "rule_id": rule.rule_id,
                "rule_type": rule.rule_type.value,
                "is_active": rule.is_active
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create targeting rule"
            )
            
    except ValueError as e:
        logger.error(f"Invalid rule data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating targeting rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create targeting rule"
        )

@router.get("/rules")
async def list_targeting_rules(
    rule_type: Optional[str] = Query(None, description="规则类型过滤"),
    experiment_id: Optional[str] = Query(None, description="实验ID过滤"),
    active_only: bool = Query(True, description="只返回活跃规则"),
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """获取定向规则列表"""
    try:
        rule_type_enum = RuleType(rule_type) if rule_type else None
        rules = engine.get_rules(rule_type_enum, experiment_id)
        
        if active_only:
            rules = [r for r in rules if r.is_active]
        
        return {
            "total_rules": len(rules),
            "rules": [rule.to_dict() for rule in rules]
        }
        
    except Exception as e:
        logger.error(f"Error listing targeting rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve targeting rules"
        )

@router.get("/rules/{rule_id}")
async def get_targeting_rule(
    rule_id: str,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """获取特定定向规则"""
    try:
        rules = engine.get_rules()
        rule = next((r for r in rules if r.rule_id == rule_id), None)
        
        if rule:
            return rule.to_dict()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Targeting rule not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting targeting rule {rule_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve targeting rule"
        )

@router.put("/rules/{rule_id}")
async def update_targeting_rule(
    rule_id: str,
    request: UpdateRuleRequest,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """更新定向规则"""
    try:
        # 获取现有规则
        rules = engine.get_rules()
        existing_rule = next((r for r in rules if r.rule_id == rule_id), None)
        
        if not existing_rule:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Targeting rule not found"
            )
        
        # 更新字段
        if request.name is not None:
            existing_rule.name = request.name
        if request.description is not None:
            existing_rule.description = request.description
        if request.condition is not None:
            existing_rule.condition = _parse_condition(request.condition)
        if request.priority is not None:
            existing_rule.priority = request.priority
        if request.is_active is not None:
            existing_rule.is_active = request.is_active
        if request.experiment_ids is not None:
            existing_rule.experiment_ids = request.experiment_ids
        if request.variant_ids is not None:
            existing_rule.variant_ids = request.variant_ids
        if request.metadata is not None:
            existing_rule.metadata = request.metadata
        
        # 更新规则
        success = engine.update_rule(existing_rule)
        
        if success:
            return {
                "message": "Targeting rule updated successfully",
                "rule_id": rule_id,
                "updated_at": existing_rule.updated_at
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update targeting rule"
            )
            
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid rule update data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating targeting rule {rule_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update targeting rule"
        )

@router.delete("/rules/{rule_id}")
async def delete_targeting_rule(
    rule_id: str,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """删除定向规则"""
    try:
        success = engine.remove_rule(rule_id)
        
        if success:
            return {
                "message": "Targeting rule deleted successfully",
                "rule_id": rule_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Targeting rule not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting targeting rule {rule_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete targeting rule"
        )

@router.post("/evaluate")
async def evaluate_user_targeting(
    request: EvaluateUserRequest,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """评估用户定向规则"""
    try:
        evaluation_results = engine.evaluate_user(
            request.user_id,
            request.user_context,
            request.experiment_id
        )
        
        return {
            "user_id": request.user_id,
            "experiment_id": request.experiment_id,
            "total_rules_evaluated": len(evaluation_results),
            "matched_rules": len([r for r in evaluation_results if r.matched]),
            "results": [
                {
                    "rule_id": result.rule_id,
                    "rule_type": result.rule_type.value,
                    "matched": result.matched,
                    "evaluation_reason": result.evaluation_reason,
                    "forced_variant_id": result.forced_variant_id,
                    "experiment_ids": result.experiment_ids,
                    "evaluation_time": result.evaluation_time,
                    "metadata": result.metadata
                }
                for result in evaluation_results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error evaluating user targeting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate user targeting"
        )

@router.post("/evaluate/batch")
async def batch_evaluate_user_targeting(
    request: BatchEvaluateRequest,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """批量评估用户定向规则"""
    try:
        batch_results = engine.batch_evaluate_users(
            request.user_contexts,
            request.experiment_id
        )
        
        summary_results = []
        total_users = len(request.user_contexts)
        total_matched_users = 0
        
        for user_id, evaluation_results in batch_results.items():
            matched_rules = [r for r in evaluation_results if r.matched]
            if matched_rules:
                total_matched_users += 1
            
            summary_results.append({
                "user_id": user_id,
                "total_rules_evaluated": len(evaluation_results),
                "matched_rules_count": len(matched_rules),
                "has_forced_variant": any(r.forced_variant_id for r in matched_rules),
                "matched_rule_types": list(set(r.rule_type.value for r in matched_rules))
            })
        
        return {
            "experiment_id": request.experiment_id,
            "total_users": total_users,
            "matched_users": total_matched_users,
            "match_rate_percentage": (total_matched_users / total_users * 100) if total_users > 0 else 0,
            "detailed_results": summary_results
        }
        
    except Exception as e:
        logger.error(f"Error in batch user targeting evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate batch user targeting"
        )

@router.post("/check-eligibility")
async def check_user_eligibility(
    user_id: str = Query(..., description="用户ID"),
    experiment_id: str = Query(..., description="实验ID"),
    user_context: Optional[Dict[str, Any]] = None,
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """检查用户实验参与资格"""
    try:
        user_context = user_context or {}
        eligibility_result = engine.check_user_eligibility(
            user_id,
            user_context,
            experiment_id
        )
        
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            **eligibility_result
        }
        
    except Exception as e:
        logger.error(f"Error checking user eligibility: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check user eligibility"
        )

@router.get("/statistics")
async def get_targeting_statistics(
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """获取定向规则统计信息"""
    try:
        stats = engine.get_rule_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting targeting statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve targeting statistics"
        )

@router.delete("/rules")
async def clear_targeting_rules(
    rule_type: Optional[str] = Query(None, description="特定规则类型"),
    confirm: bool = Query(False, description="确认清除规则"),
    engine: TargetingRulesEngine = Depends(get_rules_engine)
):
    """清除定向规则"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please set confirm=true to clear rules"
            )
        
        rule_type_enum = RuleType(rule_type) if rule_type else None
        cleared_count = engine.clear_rules(rule_type_enum)
        
        return {
            "message": "Targeting rules cleared successfully",
            "cleared_count": cleared_count,
            "rule_type": rule_type or "all"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing targeting rules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear targeting rules"
        )

@router.post("/rules/templates")
async def create_rule_templates():
    """创建常用规则模板"""
    try:
        templates = {
            "blacklist_by_country": {
                "name": "按国家黑名单",
                "description": "根据用户国家进行黑名单过滤",
                "rule_type": "blacklist",
                "condition": {
                    "field": "country",
                    "operator": "in",
                    "value": ["CN", "RU"],
                    "case_sensitive": False
                }
            },
            "whitelist_premium_users": {
                "name": "付费用户白名单",
                "description": "只允许付费用户参与实验",
                "rule_type": "whitelist",
                "condition": {
                    "field": "user_type",
                    "operator": "eq",
                    "value": "premium",
                    "case_sensitive": False
                }
            },
            "targeting_mobile_users": {
                "name": "移动端用户定向",
                "description": "针对移动端用户的定向规则",
                "rule_type": "targeting",
                "condition": {
                    "field": "device_type",
                    "operator": "eq",
                    "value": "mobile",
                    "case_sensitive": False
                }
            },
            "composite_age_and_location": {
                "name": "年龄和地区复合条件",
                "description": "年龄在18-35岁且位于特定地区的用户",
                "rule_type": "targeting",
                "condition": {
                    "logical_operator": "and",
                    "conditions": [
                        {
                            "field": "age",
                            "operator": "between",
                            "value": [18, 35]
                        },
                        {
                            "field": "region",
                            "operator": "in",
                            "value": ["US", "CA", "GB"],
                            "case_sensitive": False
                        }
                    ]
                }
            }
        }
        
        return {
            "message": "Rule templates available",
            "templates": templates
        }
        
    except Exception as e:
        logger.error(f"Error creating rule templates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create rule templates"
        )

@router.get("/operators")
async def list_available_operators():
    """列出所有可用的操作符"""
    try:
        operators = [
            {
                "operator": op.value,
                "name": op.name,
                "description": _get_operator_description(op)
            }
            for op in RuleOperator
        ]
        
        logical_operators = [
            {
                "operator": op.value,
                "name": op.name,
                "description": _get_logical_operator_description(op)
            }
            for op in LogicalOperator
        ]
        
        return {
            "rule_operators": operators,
            "logical_operators": logical_operators,
            "rule_types": [rt.value for rt in RuleType]
        }
        
    except Exception as e:
        logger.error(f"Error listing operators: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list operators"
        )

def _get_operator_description(operator: RuleOperator) -> str:
    """获取操作符描述"""
    descriptions = {
        RuleOperator.EQUALS: "字段值等于指定值",
        RuleOperator.NOT_EQUALS: "字段值不等于指定值",
        RuleOperator.IN: "字段值包含在指定列表中",
        RuleOperator.NOT_IN: "字段值不包含在指定列表中",
        RuleOperator.CONTAINS: "字段值包含指定子字符串",
        RuleOperator.NOT_CONTAINS: "字段值不包含指定子字符串",
        RuleOperator.STARTS_WITH: "字段值以指定字符串开头",
        RuleOperator.ENDS_WITH: "字段值以指定字符串结尾",
        RuleOperator.REGEX: "字段值匹配正则表达式",
        RuleOperator.GREATER_THAN: "字段值大于指定值",
        RuleOperator.GREATER_EQUAL: "字段值大于等于指定值",
        RuleOperator.LESS_THAN: "字段值小于指定值",
        RuleOperator.LESS_EQUAL: "字段值小于等于指定值",
        RuleOperator.BETWEEN: "字段值在指定范围内",
        RuleOperator.EXISTS: "字段存在",
        RuleOperator.NOT_EXISTS: "字段不存在"
    }
    return descriptions.get(operator, "未知操作符")

def _get_logical_operator_description(operator: LogicalOperator) -> str:
    """获取逻辑操作符描述"""
    descriptions = {
        LogicalOperator.AND: "所有条件都必须为真",
        LogicalOperator.OR: "至少一个条件为真",
        LogicalOperator.NOT: "条件为假"
    }
    return descriptions.get(operator, "未知逻辑操作符")
