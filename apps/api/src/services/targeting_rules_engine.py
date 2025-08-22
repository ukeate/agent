"""
定向规则引擎 - 实现黑白名单管理和复杂定向规则系统
"""
import re
import json
from typing import Dict, List, Any, Optional, Set, Union, Callable
from enum import Enum
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
import operator
from functools import reduce

from core.logging import logger


class RuleType(Enum):
    """规则类型"""
    WHITELIST = "whitelist"  # 白名单
    BLACKLIST = "blacklist"  # 黑名单
    TARGETING = "targeting"  # 定向规则
    EXCLUSION = "exclusion"  # 排除规则


class RuleOperator(Enum):
    """规则操作符"""
    EQUALS = "eq"           # 等于
    NOT_EQUALS = "ne"       # 不等于
    IN = "in"              # 包含于
    NOT_IN = "not_in"      # 不包含于
    CONTAINS = "contains"   # 字符串包含
    NOT_CONTAINS = "not_contains"  # 字符串不包含
    STARTS_WITH = "starts_with"    # 以...开头
    ENDS_WITH = "ends_with"        # 以...结尾
    REGEX = "regex"         # 正则匹配
    GREATER_THAN = "gt"     # 大于
    GREATER_EQUAL = "gte"   # 大于等于
    LESS_THAN = "lt"        # 小于
    LESS_EQUAL = "lte"      # 小于等于
    BETWEEN = "between"     # 介于之间
    EXISTS = "exists"       # 字段存在
    NOT_EXISTS = "not_exists"  # 字段不存在


class LogicalOperator(Enum):
    """逻辑操作符"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class RuleCondition:
    """规则条件"""
    field: str
    operator: RuleOperator
    value: Union[str, int, float, List[Any], Dict[str, Any]]
    case_sensitive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "case_sensitive": self.case_sensitive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleCondition':
        return cls(
            field=data["field"],
            operator=RuleOperator(data["operator"]),
            value=data["value"],
            case_sensitive=data.get("case_sensitive", True)
        )


@dataclass
class CompositeCondition:
    """复合条件"""
    logical_operator: LogicalOperator
    conditions: List[Union[RuleCondition, 'CompositeCondition']]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_operator": self.logical_operator.value,
            "conditions": [
                cond.to_dict() for cond in self.conditions
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompositeCondition':
        conditions = []
        for cond_data in data["conditions"]:
            if "logical_operator" in cond_data:
                conditions.append(CompositeCondition.from_dict(cond_data))
            else:
                conditions.append(RuleCondition.from_dict(cond_data))
        
        return cls(
            logical_operator=LogicalOperator(data["logical_operator"]),
            conditions=conditions
        )


@dataclass
class TargetingRule:
    """定向规则"""
    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    condition: Union[RuleCondition, CompositeCondition]
    priority: int = 0
    is_active: bool = True
    experiment_ids: List[str] = field(default_factory=list)  # 适用的实验ID
    variant_ids: List[str] = field(default_factory=list)     # 强制分配的变体ID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "condition": self.condition.to_dict(),
            "priority": self.priority,
            "is_active": self.is_active,
            "experiment_ids": self.experiment_ids,
            "variant_ids": self.variant_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetingRule':
        cond_data = data["condition"]
        if "logical_operator" in cond_data:
            condition = CompositeCondition.from_dict(cond_data)
        else:
            condition = RuleCondition.from_dict(cond_data)
        
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            rule_type=RuleType(data["rule_type"]),
            condition=condition,
            priority=data.get("priority", 0),
            is_active=data.get("is_active", True),
            experiment_ids=data.get("experiment_ids", []),
            variant_ids=data.get("variant_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"],
            metadata=data.get("metadata", {})
        )


@dataclass
class EvaluationResult:
    """规则评估结果"""
    user_id: str
    rule_id: str
    matched: bool
    rule_type: RuleType
    evaluation_reason: str
    forced_variant_id: Optional[str] = None
    experiment_ids: List[str] = field(default_factory=list)
    evaluation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class TargetingRulesEngine:
    """定向规则引擎"""
    
    def __init__(self):
        """初始化规则引擎"""
        self._rules: Dict[str, TargetingRule] = {}
        self._compiled_regex_cache: Dict[str, re.Pattern] = {}
        
        # 操作符映射
        self._operators: Dict[RuleOperator, Callable] = {
            RuleOperator.EQUALS: self._op_equals,
            RuleOperator.NOT_EQUALS: self._op_not_equals,
            RuleOperator.IN: self._op_in,
            RuleOperator.NOT_IN: self._op_not_in,
            RuleOperator.CONTAINS: self._op_contains,
            RuleOperator.NOT_CONTAINS: self._op_not_contains,
            RuleOperator.STARTS_WITH: self._op_starts_with,
            RuleOperator.ENDS_WITH: self._op_ends_with,
            RuleOperator.REGEX: self._op_regex,
            RuleOperator.GREATER_THAN: self._op_gt,
            RuleOperator.GREATER_EQUAL: self._op_gte,
            RuleOperator.LESS_THAN: self._op_lt,
            RuleOperator.LESS_EQUAL: self._op_lte,
            RuleOperator.BETWEEN: self._op_between,
            RuleOperator.EXISTS: self._op_exists,
            RuleOperator.NOT_EXISTS: self._op_not_exists,
        }
        
        # 评估统计
        self._evaluation_stats: Dict[str, int] = {
            "total_evaluations": 0,
            "matched_rules": 0,
            "blacklist_blocks": 0,
            "whitelist_allows": 0,
            "targeting_matches": 0
        }
    
    def add_rule(self, rule: TargetingRule) -> bool:
        """
        添加规则
        
        Args:
            rule: 定向规则
            
        Returns:
            是否成功
        """
        try:
            # 验证规则
            validation_result = self._validate_rule(rule)
            if not validation_result["is_valid"]:
                logger.error(f"Invalid rule {rule.rule_id}: {validation_result['errors']}")
                return False
            
            self._rules[rule.rule_id] = rule
            logger.info(f"Added targeting rule {rule.rule_id} ({rule.rule_type.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule {rule.rule_id}: {str(e)}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        移除规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功
        """
        try:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Removed targeting rule {rule_id}")
                return True
            else:
                logger.warning(f"Rule {rule_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error removing rule {rule_id}: {str(e)}")
            return False
    
    def update_rule(self, rule: TargetingRule) -> bool:
        """
        更新规则
        
        Args:
            rule: 更新的规则
            
        Returns:
            是否成功
        """
        try:
            if rule.rule_id not in self._rules:
                logger.error(f"Rule {rule.rule_id} not found for update")
                return False
            
            # 验证规则
            validation_result = self._validate_rule(rule)
            if not validation_result["is_valid"]:
                logger.error(f"Invalid rule update {rule.rule_id}: {validation_result['errors']}")
                return False
            
            rule.updated_at = datetime.now(timezone.utc)
            self._rules[rule.rule_id] = rule
            logger.info(f"Updated targeting rule {rule.rule_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating rule {rule.rule_id}: {str(e)}")
            return False
    
    def evaluate_user(self, user_id: str, user_context: Dict[str, Any], 
                     experiment_id: Optional[str] = None) -> List[EvaluationResult]:
        """
        评估用户是否匹配规则
        
        Args:
            user_id: 用户ID
            user_context: 用户上下文信息
            experiment_id: 特定实验ID（可选）
            
        Returns:
            评估结果列表
        """
        try:
            self._evaluation_stats["total_evaluations"] += 1
            results = []
            
            # 获取适用的规则（按优先级排序）
            applicable_rules = self._get_applicable_rules(experiment_id)
            
            for rule in applicable_rules:
                try:
                    matched = self._evaluate_condition(rule.condition, user_context)
                    
                    if matched:
                        self._evaluation_stats["matched_rules"] += 1
                        
                        if rule.rule_type == RuleType.BLACKLIST:
                            self._evaluation_stats["blacklist_blocks"] += 1
                        elif rule.rule_type == RuleType.WHITELIST:
                            self._evaluation_stats["whitelist_allows"] += 1
                        elif rule.rule_type == RuleType.TARGETING:
                            self._evaluation_stats["targeting_matches"] += 1
                    
                    result = EvaluationResult(
                        user_id=user_id,
                        rule_id=rule.rule_id,
                        matched=matched,
                        rule_type=rule.rule_type,
                        evaluation_reason=f"Rule {rule.rule_id} evaluated: {matched}",
                        forced_variant_id=rule.variant_ids[0] if matched and rule.variant_ids else None,
                        experiment_ids=rule.experiment_ids,
                        metadata={
                            "rule_name": rule.name,
                            "rule_priority": rule.priority
                        }
                    )
                    
                    results.append(result)
                    
                    # 如果是黑名单匹配，立即返回（优先级最高）
                    if matched and rule.rule_type == RuleType.BLACKLIST:
                        logger.debug(f"User {user_id} blocked by blacklist rule {rule.rule_id}")
                        return [result]
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.rule_id} for user {user_id}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating user {user_id}: {str(e)}")
            return []
    
    def check_user_eligibility(self, user_id: str, user_context: Dict[str, Any], 
                              experiment_id: str) -> Dict[str, Any]:
        """
        检查用户是否有资格参与实验
        
        Args:
            user_id: 用户ID
            user_context: 用户上下文
            experiment_id: 实验ID
            
        Returns:
            资格检查结果
        """
        try:
            evaluation_results = self.evaluate_user(user_id, user_context, experiment_id)
            
            # 检查黑名单
            blacklist_matches = [r for r in evaluation_results if r.matched and r.rule_type == RuleType.BLACKLIST]
            if blacklist_matches:
                return {
                    "eligible": False,
                    "reason": "blocked_by_blacklist",
                    "blocking_rule": blacklist_matches[0].rule_id,
                    "forced_variant_id": None
                }
            
            # 检查白名单
            whitelist_matches = [r for r in evaluation_results if r.matched and r.rule_type == RuleType.WHITELIST]
            whitelist_rules = [r for r in evaluation_results if r.rule_type == RuleType.WHITELIST]
            
            # 如果有白名单规则但用户不匹配，则排除
            if whitelist_rules and not whitelist_matches:
                return {
                    "eligible": False,
                    "reason": "not_in_whitelist",
                    "blocking_rule": None,
                    "forced_variant_id": None
                }
            
            # 检查定向规则
            targeting_matches = [r for r in evaluation_results if r.matched and r.rule_type == RuleType.TARGETING]
            forced_variant_id = None
            
            if targeting_matches:
                # 使用优先级最高的定向规则
                highest_priority_match = max(targeting_matches, key=lambda x: self._rules[x.rule_id].priority)
                forced_variant_id = highest_priority_match.forced_variant_id
            
            return {
                "eligible": True,
                "reason": "eligible",
                "matching_rules": [r.rule_id for r in evaluation_results if r.matched],
                "forced_variant_id": forced_variant_id,
                "evaluation_results": evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Error checking eligibility for user {user_id}: {str(e)}")
            return {
                "eligible": False,
                "reason": "evaluation_error",
                "error": str(e)
            }
    
    def batch_evaluate_users(self, user_contexts: List[Dict[str, Any]], 
                           experiment_id: Optional[str] = None) -> Dict[str, List[EvaluationResult]]:
        """
        批量评估用户
        
        Args:
            user_contexts: 用户上下文列表，每个包含user_id和其他属性
            experiment_id: 特定实验ID（可选）
            
        Returns:
            用户ID到评估结果的映射
        """
        try:
            results = {}
            
            for user_context in user_contexts:
                user_id = user_context.get("user_id")
                if not user_id:
                    logger.warning("Missing user_id in context, skipping")
                    continue
                
                evaluation_results = self.evaluate_user(user_id, user_context, experiment_id)
                results[user_id] = evaluation_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            return {}
    
    def _get_applicable_rules(self, experiment_id: Optional[str] = None) -> List[TargetingRule]:
        """获取适用的规则（按优先级排序）"""
        try:
            applicable_rules = []
            
            for rule in self._rules.values():
                if not rule.is_active:
                    continue
                
                # 如果指定了实验ID，只返回适用于该实验的规则
                if experiment_id and rule.experiment_ids and experiment_id not in rule.experiment_ids:
                    continue
                
                applicable_rules.append(rule)
            
            # 按优先级排序（黑名单 > 白名单 > 定向，同类型按priority字段排序）
            def sort_key(rule):
                type_priority = {
                    RuleType.BLACKLIST: 3,
                    RuleType.WHITELIST: 2,
                    RuleType.TARGETING: 1,
                    RuleType.EXCLUSION: 1
                }
                return (type_priority.get(rule.rule_type, 0), rule.priority)
            
            applicable_rules.sort(key=sort_key, reverse=True)
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error getting applicable rules: {str(e)}")
            return []
    
    def _evaluate_condition(self, condition: Union[RuleCondition, CompositeCondition], 
                          user_context: Dict[str, Any]) -> bool:
        """评估条件"""
        try:
            if isinstance(condition, RuleCondition):
                return self._evaluate_single_condition(condition, user_context)
            elif isinstance(condition, CompositeCondition):
                return self._evaluate_composite_condition(condition, user_context)
            else:
                logger.error(f"Unknown condition type: {type(condition)}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {str(e)}")
            return False
    
    def _evaluate_single_condition(self, condition: RuleCondition, 
                                  user_context: Dict[str, Any]) -> bool:
        """评估单个条件"""
        try:
            field_value = self._get_field_value(condition.field, user_context)
            operator_func = self._operators.get(condition.operator)
            
            if not operator_func:
                logger.error(f"Unknown operator: {condition.operator}")
                return False
            
            return operator_func(field_value, condition.value, condition.case_sensitive)
            
        except Exception as e:
            logger.error(f"Error evaluating single condition: {str(e)}")
            return False
    
    def _evaluate_composite_condition(self, condition: CompositeCondition, 
                                    user_context: Dict[str, Any]) -> bool:
        """评估复合条件"""
        try:
            if not condition.conditions:
                return True
            
            results = []
            for sub_condition in condition.conditions:
                result = self._evaluate_condition(sub_condition, user_context)
                results.append(result)
            
            if condition.logical_operator == LogicalOperator.AND:
                return all(results)
            elif condition.logical_operator == LogicalOperator.OR:
                return any(results)
            elif condition.logical_operator == LogicalOperator.NOT:
                # NOT操作符应该只有一个条件
                return not results[0] if results else True
            else:
                logger.error(f"Unknown logical operator: {condition.logical_operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating composite condition: {str(e)}")
            return False
    
    def _get_field_value(self, field_path: str, context: Dict[str, Any]) -> Any:
        """从上下文中获取字段值（支持嵌套路径）"""
        try:
            # 支持点号分隔的嵌套路径，如 "user.profile.age"
            keys = field_path.split('.')
            value = context
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting field value for {field_path}: {str(e)}")
            return None
    
    # 操作符实现
    def _op_equals(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """等于操作"""
        if field_value is None:
            return condition_value is None
        
        if isinstance(field_value, str) and isinstance(condition_value, str) and not case_sensitive:
            return field_value.lower() == condition_value.lower()
        
        return field_value == condition_value
    
    def _op_not_equals(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """不等于操作"""
        return not self._op_equals(field_value, condition_value, case_sensitive)
    
    def _op_in(self, field_value: Any, condition_value: List[Any], case_sensitive: bool) -> bool:
        """包含于操作"""
        if field_value is None or not isinstance(condition_value, list):
            return False
        
        if isinstance(field_value, str) and not case_sensitive:
            return any(field_value.lower() == str(v).lower() for v in condition_value)
        
        return field_value in condition_value
    
    def _op_not_in(self, field_value: Any, condition_value: List[Any], case_sensitive: bool) -> bool:
        """不包含于操作"""
        return not self._op_in(field_value, condition_value, case_sensitive)
    
    def _op_contains(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """包含操作"""
        if field_value is None or not isinstance(condition_value, str):
            return False
        
        field_str = str(field_value)
        condition_str = str(condition_value)
        
        if not case_sensitive:
            field_str = field_str.lower()
            condition_str = condition_str.lower()
        
        return condition_str in field_str
    
    def _op_not_contains(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """不包含操作"""
        return not self._op_contains(field_value, condition_value, case_sensitive)
    
    def _op_starts_with(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """以...开头"""
        if field_value is None or not isinstance(condition_value, str):
            return False
        
        field_str = str(field_value)
        condition_str = str(condition_value)
        
        if not case_sensitive:
            field_str = field_str.lower()
            condition_str = condition_str.lower()
        
        return field_str.startswith(condition_str)
    
    def _op_ends_with(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """以...结尾"""
        if field_value is None or not isinstance(condition_value, str):
            return False
        
        field_str = str(field_value)
        condition_str = str(condition_value)
        
        if not case_sensitive:
            field_str = field_str.lower()
            condition_str = condition_str.lower()
        
        return field_str.endswith(condition_str)
    
    def _op_regex(self, field_value: Any, condition_value: str, case_sensitive: bool) -> bool:
        """正则匹配"""
        if field_value is None or not isinstance(condition_value, str):
            return False
        
        try:
            # 缓存编译的正则表达式
            cache_key = f"{condition_value}_{case_sensitive}"
            if cache_key not in self._compiled_regex_cache:
                flags = 0 if case_sensitive else re.IGNORECASE
                self._compiled_regex_cache[cache_key] = re.compile(condition_value, flags)
            
            pattern = self._compiled_regex_cache[cache_key]
            return bool(pattern.search(str(field_value)))
            
        except re.error as e:
            logger.error(f"Invalid regex pattern '{condition_value}': {str(e)}")
            return False
    
    def _op_gt(self, field_value: Any, condition_value: Union[int, float], case_sensitive: bool) -> bool:
        """大于"""
        try:
            return float(field_value) > float(condition_value)
        except (TypeError, ValueError):
            return False
    
    def _op_gte(self, field_value: Any, condition_value: Union[int, float], case_sensitive: bool) -> bool:
        """大于等于"""
        try:
            return float(field_value) >= float(condition_value)
        except (TypeError, ValueError):
            return False
    
    def _op_lt(self, field_value: Any, condition_value: Union[int, float], case_sensitive: bool) -> bool:
        """小于"""
        try:
            return float(field_value) < float(condition_value)
        except (TypeError, ValueError):
            return False
    
    def _op_lte(self, field_value: Any, condition_value: Union[int, float], case_sensitive: bool) -> bool:
        """小于等于"""
        try:
            return float(field_value) <= float(condition_value)
        except (TypeError, ValueError):
            return False
    
    def _op_between(self, field_value: Any, condition_value: List[Union[int, float]], case_sensitive: bool) -> bool:
        """介于之间"""
        try:
            if not isinstance(condition_value, list) or len(condition_value) != 2:
                return False
            
            value = float(field_value)
            min_val, max_val = float(condition_value[0]), float(condition_value[1])
            return min_val <= value <= max_val
            
        except (TypeError, ValueError):
            return False
    
    def _op_exists(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """字段存在"""
        return field_value is not None
    
    def _op_not_exists(self, field_value: Any, condition_value: Any, case_sensitive: bool) -> bool:
        """字段不存在"""
        return field_value is None
    
    def _validate_rule(self, rule: TargetingRule) -> Dict[str, Any]:
        """验证规则"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 检查必填字段
            if not rule.rule_id:
                validation_result["is_valid"] = False
                validation_result["errors"].append("rule_id is required")
            
            if not rule.name:
                validation_result["is_valid"] = False
                validation_result["errors"].append("name is required")
            
            # 检查规则ID唯一性
            if rule.rule_id in self._rules and self._rules[rule.rule_id] != rule:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"rule_id {rule.rule_id} already exists")
            
            # 验证条件
            condition_validation = self._validate_condition(rule.condition)
            if not condition_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["errors"].extend(condition_validation["errors"])
            
            # 检查强制分配的变体ID
            if rule.rule_type == RuleType.TARGETING and rule.variant_ids:
                if len(rule.variant_ids) > 1:
                    validation_result["warnings"].append("Multiple variant_ids specified, only first one will be used")
            
            return validation_result
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result
    
    def _validate_condition(self, condition: Union[RuleCondition, CompositeCondition]) -> Dict[str, Any]:
        """验证条件"""
        validation_result = {
            "is_valid": True,
            "errors": []
        }
        
        try:
            if isinstance(condition, RuleCondition):
                if not condition.field:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("field is required in condition")
                
                if condition.operator not in self._operators:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Unknown operator: {condition.operator}")
                
                # 验证特定操作符的值格式
                if condition.operator in [RuleOperator.IN, RuleOperator.NOT_IN]:
                    if not isinstance(condition.value, list):
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(f"Operator {condition.operator.value} requires list value")
                
                elif condition.operator == RuleOperator.BETWEEN:
                    if not isinstance(condition.value, list) or len(condition.value) != 2:
                        validation_result["is_valid"] = False
                        validation_result["errors"].append("BETWEEN operator requires list with exactly 2 values")
                
                elif condition.operator == RuleOperator.REGEX:
                    try:
                        re.compile(str(condition.value))
                    except re.error as e:
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(f"Invalid regex pattern: {str(e)}")
            
            elif isinstance(condition, CompositeCondition):
                if not condition.conditions:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Composite condition must have at least one sub-condition")
                
                # 递归验证子条件
                for sub_condition in condition.conditions:
                    sub_validation = self._validate_condition(sub_condition)
                    if not sub_validation["is_valid"]:
                        validation_result["is_valid"] = False
                        validation_result["errors"].extend(sub_validation["errors"])
            
            return validation_result
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Condition validation error: {str(e)}")
            return validation_result
    
    def get_rules(self, rule_type: Optional[RuleType] = None, 
                  experiment_id: Optional[str] = None) -> List[TargetingRule]:
        """
        获取规则列表
        
        Args:
            rule_type: 规则类型过滤
            experiment_id: 实验ID过滤
            
        Returns:
            规则列表
        """
        try:
            rules = list(self._rules.values())
            
            if rule_type:
                rules = [r for r in rules if r.rule_type == rule_type]
            
            if experiment_id:
                rules = [r for r in rules if not r.experiment_ids or experiment_id in r.experiment_ids]
            
            return rules
            
        except Exception as e:
            logger.error(f"Error getting rules: {str(e)}")
            return []
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        try:
            total_rules = len(self._rules)
            active_rules = len([r for r in self._rules.values() if r.is_active])
            
            rule_type_counts = {}
            for rule_type in RuleType:
                rule_type_counts[rule_type.value] = len([r for r in self._rules.values() if r.rule_type == rule_type])
            
            return {
                "total_rules": total_rules,
                "active_rules": active_rules,
                "inactive_rules": total_rules - active_rules,
                "rule_type_distribution": rule_type_counts,
                "evaluation_stats": self._evaluation_stats.copy(),
                "compiled_regex_cache_size": len(self._compiled_regex_cache)
            }
            
        except Exception as e:
            logger.error(f"Error getting rule statistics: {str(e)}")
            return {"error": str(e)}
    
    def clear_rules(self, rule_type: Optional[RuleType] = None) -> int:
        """
        清除规则
        
        Args:
            rule_type: 特定类型（可选）
            
        Returns:
            清除的数量
        """
        try:
            if rule_type:
                to_remove = [rule_id for rule_id, rule in self._rules.items() if rule.rule_type == rule_type]
                for rule_id in to_remove:
                    del self._rules[rule_id]
                logger.info(f"Cleared {len(to_remove)} rules of type {rule_type.value}")
                return len(to_remove)
            else:
                count = len(self._rules)
                self._rules.clear()
                self._compiled_regex_cache.clear()
                logger.info(f"Cleared all {count} rules")
                return count
                
        except Exception as e:
            logger.error(f"Error clearing rules: {str(e)}")
            return 0