"""
工作流结果处理器
负责结果验证、聚合、格式化和缓存
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import uuid4
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from dataclasses import dataclass, asdict
import hashlib

import redis.asyncio as redis
from pydantic import BaseModel, ValidationError

from models.schemas.workflow import (
    WorkflowExecution, WorkflowStepExecution, WorkflowResult,
    ResultAggregationStrategy, WorkflowStepType, WorkflowStepStatus
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class ValidationLevel(str, Enum):
    """验证级别"""
    STRICT = "strict"      # 严格验证
    NORMAL = "normal"      # 标准验证
    PERMISSIVE = "permissive"  # 宽松验证


class ResultFormat(str, Enum):
    """结果格式"""
    JSON = "json"
    XML = "xml"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


@dataclass
class ValidationRule:
    """验证规则"""
    id: str
    name: str
    description: str
    rule_type: str  # schema, range, format, custom
    parameters: Dict[str, Any]
    error_message: str
    severity: str = "error"  # error, warning, info
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    score: float  # 0-100 验证评分
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedResult:
    """处理后的结果"""
    execution_id: str
    original_results: Dict[str, Any]
    validated_results: Dict[str, Any]
    aggregated_result: Any
    formatted_outputs: Dict[str, Any]  # format -> content
    validation_report: ValidationResult
    processing_metadata: Dict[str, Any]
    created_at: datetime
    cache_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = data['created_at'].isoformat()
        return data


class ResultValidator:
    """结果验证器"""
    
    def __init__(self):
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
        # 内置验证规则
        self._register_builtin_rules()
    
    def _register_builtin_rules(self):
        """注册内置验证规则"""
        # 数据类型验证
        self.register_rule(ValidationRule(
            id="type_check",
            name="数据类型检查",
            description="验证结果的数据类型",
            rule_type="type",
            parameters={"allowed_types": ["dict", "list", "str", "int", "float", "bool"]},
            error_message="结果数据类型不符合要求"
        ))
        
        # 必填字段验证
        self.register_rule(ValidationRule(
            id="required_fields",
            name="必填字段检查",
            description="验证必填字段是否存在",
            rule_type="schema",
            parameters={"required_fields": []},
            error_message="缺少必填字段"
        ))
        
        # 数值范围验证
        self.register_rule(ValidationRule(
            id="numeric_range",
            name="数值范围检查",
            description="验证数值是否在指定范围内",
            rule_type="range",
            parameters={"min_value": None, "max_value": None},
            error_message="数值超出允许范围"
        ))
        
        # 字符串格式验证
        self.register_rule(ValidationRule(
            id="string_format",
            name="字符串格式检查", 
            description="验证字符串格式",
            rule_type="format",
            parameters={"pattern": None, "min_length": 0, "max_length": None},
            error_message="字符串格式不正确"
        ))
        
        # 置信度检查
        self.register_rule(ValidationRule(
            id="confidence_check",
            name="置信度检查",
            description="验证结果置信度",
            rule_type="range",
            parameters={"min_value": 0.0, "max_value": 1.0, "threshold": 0.5},
            error_message="置信度过低",
            severity="warning"
        ))
    
    def register_rule(self, rule: ValidationRule):
        """注册验证规则"""
        self.validation_rules[rule.id] = rule
        logger.debug(f"注册验证规则: {rule.id}")
    
    def register_custom_validator(self, name: str, validator: Callable):
        """注册自定义验证器"""
        self.custom_validators[name] = validator
        logger.debug(f"注册自定义验证器: {name}")
    
    async def validate_results(
        self, 
        results: Dict[str, Any], 
        rules: List[str] = None,
        level: ValidationLevel = ValidationLevel.NORMAL
    ) -> ValidationResult:
        """
        验证结果
        
        Args:
            results: 要验证的结果
            rules: 验证规则列表，None表示使用所有规则
            level: 验证级别
            
        Returns:
            验证结果
        """
        try:
            errors = []
            warnings = []
            score = 100.0
            
            # 选择验证规则
            if rules is None:
                active_rules = list(self.validation_rules.values())
            else:
                active_rules = [self.validation_rules[rule_id] for rule_id in rules if rule_id in self.validation_rules]
            
            # 执行验证
            for rule in active_rules:
                try:
                    rule_result = await self._apply_validation_rule(rule, results, level)
                    
                    if not rule_result["passed"]:
                        issue = {
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "message": rule_result.get("message", rule.error_message),
                            "details": rule_result.get("details", {}),
                            "severity": rule.severity
                        }
                        
                        if rule.severity == "error":
                            errors.append(issue)
                            score -= rule_result.get("penalty", 10)
                        elif rule.severity == "warning":
                            warnings.append(issue)
                            score -= rule_result.get("penalty", 5)
                
                except Exception as e:
                    logger.error(f"验证规则执行失败: {rule.id}, 错误: {e}")
                    if level == ValidationLevel.STRICT:
                        errors.append({
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "message": f"验证规则执行失败: {e}",
                            "severity": "error"
                        })
            
            # 计算最终评分
            score = max(0.0, min(100.0, score))
            is_valid = len(errors) == 0 and (level != ValidationLevel.STRICT or len(warnings) == 0)
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                score=score,
                metadata={
                    "validation_level": level,
                    "rules_applied": len(active_rules),
                    "validation_time": utc_now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"结果验证失败: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{"message": f"验证过程异常: {e}", "severity": "error"}],
                warnings=[],
                score=0.0,
                metadata={"error": str(e)}
            )
    
    async def _apply_validation_rule(self, rule: ValidationRule, results: Any, level: ValidationLevel) -> Dict[str, Any]:
        """应用验证规则"""
        rule_result = {"passed": True, "penalty": 0}
        
        try:
            if rule.rule_type == "type":
                rule_result = await self._validate_type(rule, results)
            elif rule.rule_type == "schema":
                rule_result = await self._validate_schema(rule, results)
            elif rule.rule_type == "range":
                rule_result = await self._validate_range(rule, results)
            elif rule.rule_type == "format":
                rule_result = await self._validate_format(rule, results)
            elif rule.rule_type == "custom":
                rule_result = await self._validate_custom(rule, results)
            else:
                rule_result = {"passed": False, "message": f"未知验证规则类型: {rule.rule_type}"}
        
        except Exception as e:
            rule_result = {"passed": False, "message": f"规则执行异常: {e}"}
        
        return rule_result
    
    async def _validate_type(self, rule: ValidationRule, results: Any) -> Dict[str, Any]:
        """验证数据类型"""
        allowed_types = rule.parameters.get("allowed_types", [])
        
        if not allowed_types:
            return {"passed": True}
        
        result_type = type(results).__name__
        
        if result_type not in allowed_types:
            return {
                "passed": False,
                "message": f"数据类型 {result_type} 不在允许范围内: {allowed_types}",
                "penalty": 15
            }
        
        return {"passed": True}
    
    async def _validate_schema(self, rule: ValidationRule, results: Any) -> Dict[str, Any]:
        """验证数据结构"""
        if not isinstance(results, dict):
            return {"passed": True}  # 非字典类型跳过结构验证
        
        required_fields = rule.parameters.get("required_fields", [])
        missing_fields = []
        
        for field in required_fields:
            if field not in results:
                missing_fields.append(field)
        
        if missing_fields:
            return {
                "passed": False,
                "message": f"缺少必填字段: {missing_fields}",
                "details": {"missing_fields": missing_fields},
                "penalty": len(missing_fields) * 10
            }
        
        return {"passed": True}
    
    async def _validate_range(self, rule: ValidationRule, results: Any) -> Dict[str, Any]:
        """验证数值范围"""
        min_value = rule.parameters.get("min_value")
        max_value = rule.parameters.get("max_value")
        threshold = rule.parameters.get("threshold")
        
        # 处理嵌套结构中的数值
        numeric_values = self._extract_numeric_values(results)
        
        for value in numeric_values:
            if min_value is not None and value < min_value:
                return {
                    "passed": False,
                    "message": f"数值 {value} 小于最小值 {min_value}",
                    "penalty": 10
                }
            
            if max_value is not None and value > max_value:
                return {
                    "passed": False,
                    "message": f"数值 {value} 大于最大值 {max_value}",
                    "penalty": 10
                }
            
            if threshold is not None and value < threshold:
                return {
                    "passed": False,
                    "message": f"数值 {value} 低于阈值 {threshold}",
                    "penalty": 5
                }
        
        return {"passed": True}
    
    async def _validate_format(self, rule: ValidationRule, results: Any) -> Dict[str, Any]:
        """验证格式"""
        import re
        
        pattern = rule.parameters.get("pattern")
        min_length = rule.parameters.get("min_length", 0)
        max_length = rule.parameters.get("max_length")
        
        # 处理字符串字段
        string_values = self._extract_string_values(results)
        
        for value in string_values:
            if len(value) < min_length:
                return {
                    "passed": False,
                    "message": f"字符串长度 {len(value)} 小于最小长度 {min_length}",
                    "penalty": 5
                }
            
            if max_length is not None and len(value) > max_length:
                return {
                    "passed": False,
                    "message": f"字符串长度 {len(value)} 大于最大长度 {max_length}",
                    "penalty": 5
                }
            
            if pattern and not re.match(pattern, value):
                return {
                    "passed": False,
                    "message": f"字符串 '{value}' 不匹配模式 '{pattern}'",
                    "penalty": 8
                }
        
        return {"passed": True}
    
    async def _validate_custom(self, rule: ValidationRule, results: Any) -> Dict[str, Any]:
        """验证自定义规则"""
        validator_name = rule.parameters.get("validator")
        
        if validator_name not in self.custom_validators:
            return {"passed": False, "message": f"自定义验证器 {validator_name} 不存在"}
        
        validator = self.custom_validators[validator_name]
        
        try:
            if asyncio.iscoroutinefunction(validator):
                result = await validator(results, rule.parameters)
            else:
                result = validator(results, rule.parameters)
            
            if isinstance(result, bool):
                return {"passed": result}
            elif isinstance(result, dict):
                return result
            else:
                return {"passed": bool(result)}
        
        except Exception as e:
            return {"passed": False, "message": f"自定义验证器执行失败: {e}"}
    
    def _extract_numeric_values(self, data: Any) -> List[float]:
        """提取数值"""
        values = []
        
        if isinstance(data, (int, float)):
            values.append(float(data))
        elif isinstance(data, dict):
            for value in data.values():
                values.extend(self._extract_numeric_values(value))
        elif isinstance(data, list):
            for item in data:
                values.extend(self._extract_numeric_values(item))
        
        return values
    
    def _extract_string_values(self, data: Any) -> List[str]:
        """提取字符串值"""
        values = []
        
        if isinstance(data, str):
            values.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                values.extend(self._extract_string_values(value))
        elif isinstance(data, list):
            for item in data:
                values.extend(self._extract_string_values(item))
        
        return values


class ResultAggregator:
    """结果聚合器"""
    
    def __init__(self):
        self.aggregation_strategies = {
            ResultAggregationStrategy.MERGE: self._merge_aggregation,
            ResultAggregationStrategy.CONSENSUS: self._consensus_aggregation,
            ResultAggregationStrategy.WEIGHTED_AVERAGE: self._weighted_average_aggregation,
            ResultAggregationStrategy.MAJORITY_VOTE: self._majority_vote_aggregation,
            ResultAggregationStrategy.BEST_RESULT: self._best_result_aggregation
        }
    
    async def aggregate_results(
        self, 
        step_results: Dict[str, Any], 
        strategy: ResultAggregationStrategy = ResultAggregationStrategy.MERGE,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        聚合步骤结果
        
        Args:
            step_results: 步骤结果字典 {step_id: result}
            strategy: 聚合策略
            config: 聚合配置
            
        Returns:
            聚合后的结果
        """
        try:
            config = config or {}
            
            if not step_results:
                return {}
            
            # 选择聚合策略
            aggregator = self.aggregation_strategies.get(strategy, self._merge_aggregation)
            
            # 执行聚合
            aggregated_result = await aggregator(step_results, config)
            
            # 添加聚合元数据
            aggregated_result["_aggregation_metadata"] = {
                "strategy": strategy.value,
                "source_count": len(step_results),
                "source_steps": list(step_results.keys()),
                "aggregated_at": utc_now().isoformat()
            }
            
            logger.info(f"结果聚合完成: 策略={strategy.value}, 源数量={len(step_results)}")
            return aggregated_result
            
        except Exception as e:
            logger.error(f"结果聚合失败: {e}")
            return {
                "error": f"聚合失败: {e}",
                "original_results": step_results
            }
    
    async def _merge_aggregation(self, step_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """合并聚合"""
        merged = {}
        
        for step_id, result in step_results.items():
            if isinstance(result, dict):
                # 嵌套合并字典
                for key, value in result.items():
                    if key.startswith("_"):  # 跳过元数据字段
                        continue
                    
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        # 冲突处理：创建列表
                        if not isinstance(merged[key], list):
                            merged[key] = [merged[key]]
                        merged[key].append(value)
            else:
                merged[step_id] = result
        
        return merged
    
    async def _consensus_aggregation(self, step_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """共识聚合"""
        # 寻找置信度最高的结果作为共识
        best_result = None
        best_confidence = 0.0
        best_step = None
        
        for step_id, result in step_results.items():
            confidence = 0.5  # 默认置信度
            
            if isinstance(result, dict):
                confidence = result.get("confidence", confidence)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = result
                best_step = step_id
        
        return {
            "consensus_result": best_result,
            "consensus_confidence": best_confidence,
            "consensus_source": best_step,
            "alternative_results": {k: v for k, v in step_results.items() if k != best_step}
        }
    
    async def _weighted_average_aggregation(self, step_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """加权平均聚合"""
        weights = config.get("weights", {})
        
        # 收集数值字段
        numeric_fields = set()
        for result in step_results.values():
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        numeric_fields.add(key)
        
        # 计算加权平均
        averaged_result = {}
        for field in numeric_fields:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for step_id, result in step_results.items():
                if isinstance(result, dict) and field in result:
                    value = result[field]
                    weight = weights.get(step_id, 1.0)
                    
                    if isinstance(value, (int, float)):
                        weighted_sum += value * weight
                        total_weight += weight
            
            if total_weight > 0:
                averaged_result[field] = weighted_sum / total_weight
        
        # 保留非数值字段（从置信度最高的结果）
        for step_id, result in step_results.items():
            if isinstance(result, dict):
                confidence = result.get("confidence", 0.5)
                if "best_confidence" not in averaged_result or confidence > averaged_result["best_confidence"]:
                    for key, value in result.items():
                        if key not in numeric_fields and not key.startswith("_"):
                            averaged_result[key] = value
                    averaged_result["best_confidence"] = confidence
        
        return averaged_result
    
    async def _majority_vote_aggregation(self, step_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """多数投票聚合"""
        # 统计各个结果的出现次数
        result_counts = {}
        
        for step_id, result in step_results.items():
            # 将结果转换为可哈希的形式进行计数
            result_key = self._serialize_for_comparison(result)
            
            if result_key not in result_counts:
                result_counts[result_key] = {
                    "count": 0,
                    "result": result,
                    "sources": []
                }
            
            result_counts[result_key]["count"] += 1
            result_counts[result_key]["sources"].append(step_id)
        
        # 找到得票最多的结果
        majority_result = max(result_counts.values(), key=lambda x: x["count"])
        
        return {
            "majority_result": majority_result["result"],
            "vote_count": majority_result["count"],
            "total_votes": len(step_results),
            "vote_percentage": majority_result["count"] / len(step_results),
            "supporting_sources": majority_result["sources"],
            "vote_distribution": {k: v["count"] for k, v in result_counts.items()}
        }
    
    async def _best_result_aggregation(self, step_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """最佳结果聚合"""
        ranking_criteria = config.get("ranking_criteria", ["confidence", "score", "quality"])
        
        best_result = None
        best_score = -1
        best_step = None
        
        for step_id, result in step_results.items():
            score = self._calculate_result_score(result, ranking_criteria)
            
            if score > best_score:
                best_score = score
                best_result = result
                best_step = step_id
        
        return {
            "best_result": best_result,
            "best_score": best_score,
            "best_source": best_step,
            "ranking_criteria": ranking_criteria,
            "all_scores": {
                step_id: self._calculate_result_score(result, ranking_criteria)
                for step_id, result in step_results.items()
            }
        }
    
    def _serialize_for_comparison(self, obj: Any) -> str:
        """序列化对象用于比较"""
        try:
            if isinstance(obj, dict):
                # 排除元数据字段
                filtered = {k: v for k, v in obj.items() if not k.startswith("_")}
                return json.dumps(filtered, sort_keys=True, default=str)
            else:
                return json.dumps(obj, sort_keys=True, default=str)
        except Exception:
            return str(obj)
    
    def _calculate_result_score(self, result: Any, criteria: List[str]) -> float:
        """计算结果评分"""
        score = 0.0
        
        if not isinstance(result, dict):
            return score
        
        for criterion in criteria:
            if criterion in result:
                value = result[criterion]
                if isinstance(value, (int, float)):
                    score += value
                elif isinstance(value, bool):
                    score += 1.0 if value else 0.0
        
        return score


class ResultFormatter:
    """结果格式化器"""
    
    def __init__(self):
        self.formatters = {
            ResultFormat.JSON: self._format_json,
            ResultFormat.XML: self._format_xml,
            ResultFormat.TEXT: self._format_text,
            ResultFormat.MARKDOWN: self._format_markdown,
            ResultFormat.HTML: self._format_html,
            ResultFormat.CSV: self._format_csv
        }
    
    async def format_result(
        self, 
        result: Any, 
        format_type: ResultFormat,
        config: Dict[str, Any] = None
    ) -> str:
        """
        格式化结果
        
        Args:
            result: 要格式化的结果
            format_type: 输出格式
            config: 格式化配置
            
        Returns:
            格式化后的字符串
        """
        try:
            config = config or {}
            formatter = self.formatters.get(format_type, self._format_json)
            
            formatted_content = await formatter(result, config)
            logger.debug(f"结果格式化完成: {format_type.value}")
            
            return formatted_content
            
        except Exception as e:
            logger.error(f"结果格式化失败: {format_type.value}, 错误: {e}")
            return f"格式化失败: {e}\n\n原始结果:\n{result}"
    
    async def _format_json(self, result: Any, config: Dict[str, Any]) -> str:
        """JSON格式化"""
        indent = config.get("indent", 2)
        ensure_ascii = config.get("ensure_ascii", False)
        
        return json.dumps(result, indent=indent, ensure_ascii=ensure_ascii, default=str)
    
    async def _format_xml(self, result: Any, config: Dict[str, Any]) -> str:
        """XML格式化"""
        root_tag = config.get("root_tag", "result")
        
        def dict_to_xml(obj, parent_tag="item"):
            if isinstance(obj, dict):
                items = []
                for key, value in obj.items():
                    items.append(f"<{key}>{dict_to_xml(value, key)}</{key}>")
                return "".join(items)
            elif isinstance(obj, list):
                items = []
                for item in obj:
                    items.append(f"<{parent_tag}>{dict_to_xml(item, parent_tag)}</{parent_tag}>")
                return "".join(items)
            else:
                return str(obj)
        
        xml_content = dict_to_xml(result, root_tag)
        return f"<?xml version='1.0' encoding='UTF-8'?>\n<{root_tag}>{xml_content}</{root_tag}>"
    
    async def _format_text(self, result: Any, config: Dict[str, Any]) -> str:
        """文本格式化"""
        def obj_to_text(obj, indent=0):
            spaces = "  " * indent
            
            if isinstance(obj, dict):
                lines = []
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"{spaces}{key}:")
                        lines.append(obj_to_text(value, indent + 1))
                    else:
                        lines.append(f"{spaces}{key}: {value}")
                return "\n".join(lines)
            elif isinstance(obj, list):
                lines = []
                for i, item in enumerate(obj):
                    lines.append(f"{spaces}[{i}] {obj_to_text(item, indent + 1)}")
                return "\n".join(lines)
            else:
                return f"{spaces}{obj}"
        
        return obj_to_text(result)
    
    async def _format_markdown(self, result: Any, config: Dict[str, Any]) -> str:
        """Markdown格式化"""
        title = config.get("title", "工作流执行结果")
        
        def obj_to_markdown(obj, level=1):
            if isinstance(obj, dict):
                lines = []
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"{'#' * level} {key}\n")
                        lines.append(obj_to_markdown(value, level + 1))
                    else:
                        lines.append(f"**{key}**: {value}\n")
                return "\n".join(lines)
            elif isinstance(obj, list):
                lines = []
                for item in obj:
                    if isinstance(item, dict):
                        lines.append(obj_to_markdown(item, level))
                    else:
                        lines.append(f"- {item}")
                return "\n".join(lines)
            else:
                return str(obj)
        
        content = obj_to_markdown(result)
        return f"# {title}\n\n{content}"
    
    async def _format_html(self, result: Any, config: Dict[str, Any]) -> str:
        """HTML格式化"""
        title = config.get("title", "工作流执行结果")
        
        def obj_to_html(obj):
            if isinstance(obj, dict):
                items = []
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        items.append(f"<li><strong>{key}:</strong><ul>{obj_to_html(value)}</ul></li>")
                    else:
                        items.append(f"<li><strong>{key}:</strong> {value}</li>")
                return f"<ul>{''.join(items)}</ul>"
            elif isinstance(obj, list):
                items = []
                for item in obj:
                    items.append(f"<li>{obj_to_html(item)}</li>")
                return f"<ul>{''.join(items)}</ul>"
            else:
                return str(obj)
        
        content = obj_to_html(result)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                ul {{ margin-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {content}
        </body>
        </html>
        """
    
    async def _format_csv(self, result: Any, config: Dict[str, Any]) -> str:
        """CSV格式化"""
        if not isinstance(result, (list, dict)):
            return f"value\n{result}"
        
        # 处理字典列表
        if isinstance(result, list) and result and isinstance(result[0], dict):
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            for row in result:
                writer.writerow(row)
            return output.getvalue()
        
        # 处理单个字典
        elif isinstance(result, dict):
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            for key, value in result.items():
                writer.writerow([key, value])
            return output.getvalue()
        
        # 处理其他类型
        else:
            return f"value\n{result}"


class ResultCache:
    """结果缓存器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_prefix = "workflow:result_cache:"
        self.default_ttl = 3600  # 1小时
    
    def _generate_cache_key(self, execution_id: str, result_hash: str = None) -> str:
        """生成缓存键"""
        if result_hash:
            return f"{self.cache_prefix}{execution_id}:{result_hash}"
        else:
            return f"{self.cache_prefix}{execution_id}"
    
    def _hash_result(self, result: Any) -> str:
        """计算结果哈希"""
        content = json.dumps(result, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def cache_result(
        self, 
        execution_id: str, 
        processed_result: ProcessedResult,
        ttl: int = None
    ) -> str:
        """
        缓存处理结果
        
        Args:
            execution_id: 执行ID
            processed_result: 处理后的结果
            ttl: 过期时间(秒)
            
        Returns:
            缓存键
        """
        try:
            ttl = ttl or self.default_ttl
            
            # 生成缓存键
            result_hash = self._hash_result(processed_result.aggregated_result)
            cache_key = self._generate_cache_key(execution_id, result_hash)
            
            # 序列化结果
            cached_data = {
                "processed_result": json.dumps(processed_result.to_dict(), default=str),
                "cached_at": utc_now().isoformat(),
                "execution_id": execution_id,
                "result_hash": result_hash
            }
            
            # 存储到Redis
            await self.redis.hset(cache_key, mapping=cached_data)
            await self.redis.expire(cache_key, ttl)
            
            # 同时维护执行ID到缓存键的映射
            index_key = self._generate_cache_key(execution_id)
            await self.redis.sadd(index_key, cache_key)
            await self.redis.expire(index_key, ttl)
            
            logger.info(f"结果已缓存: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"缓存结果失败: {execution_id}, 错误: {e}")
            return None
    
    async def get_cached_result(self, execution_id: str, result_hash: str = None) -> Optional[ProcessedResult]:
        """
        获取缓存的结果
        
        Args:
            execution_id: 执行ID
            result_hash: 结果哈希，None表示获取最新缓存
            
        Returns:
            缓存的处理结果
        """
        try:
            if result_hash:
                # 直接获取指定哈希的结果
                cache_key = self._generate_cache_key(execution_id, result_hash)
                cached_data = await self.redis.hgetall(cache_key)
                
                if cached_data:
                    result_data = json.loads(cached_data[b"processed_result"].decode())
                    return self._deserialize_processed_result(result_data)
            else:
                # 获取最新的缓存结果
                index_key = self._generate_cache_key(execution_id)
                cache_keys = await self.redis.smembers(index_key)
                
                if cache_keys:
                    # 获取最新的缓存项
                    latest_time = None
                    latest_result = None
                    
                    for cache_key in cache_keys:
                        cached_data = await self.redis.hgetall(cache_key)
                        
                        if cached_data:
                            cached_at = datetime.fromisoformat(
                                cached_data[b"cached_at"].decode()
                            )
                            
                            if latest_time is None or cached_at > latest_time:
                                latest_time = cached_at
                                result_data = json.loads(
                                    cached_data[b"processed_result"].decode()
                                )
                                latest_result = self._deserialize_processed_result(result_data)
                    
                    return latest_result
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存结果失败: {execution_id}, 错误: {e}")
            return None
    
    async def invalidate_cache(self, execution_id: str) -> bool:
        """
        失效缓存
        
        Args:
            execution_id: 执行ID
            
        Returns:
            是否成功
        """
        try:
            # 获取所有相关的缓存键
            index_key = self._generate_cache_key(execution_id)
            cache_keys = await self.redis.smembers(index_key)
            
            # 删除所有缓存项
            if cache_keys:
                await self.redis.delete(*cache_keys)
            
            # 删除索引
            await self.redis.delete(index_key)
            
            logger.info(f"缓存已失效: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"失效缓存失败: {execution_id}, 错误: {e}")
            return False
    
    def _deserialize_processed_result(self, data: Dict[str, Any]) -> ProcessedResult:
        """反序列化处理结果"""
        # 处理日期字段
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return ProcessedResult(**data)


class WorkflowResultProcessor:
    """工作流结果处理器主类"""
    
    def __init__(self, redis_client: redis.Redis):
        self.validator = ResultValidator()
        self.aggregator = ResultAggregator()
        self.formatter = ResultFormatter()
        self.cache = ResultCache(redis_client)
        
        # 处理配置
        self.processing_config = {
            "validation_level": ValidationLevel.NORMAL,
            "aggregation_strategy": ResultAggregationStrategy.MERGE,
            "cache_ttl": 3600,
            "enable_caching": True,
            "default_formats": [ResultFormat.JSON, ResultFormat.TEXT]
        }
    
    async def process_execution_results(
        self, 
        execution: WorkflowExecution,
        config: Dict[str, Any] = None
    ) -> ProcessedResult:
        """
        处理工作流执行结果
        
        Args:
            execution: 工作流执行实例
            config: 处理配置
            
        Returns:
            处理后的结果
        """
        try:
            config = {**self.processing_config, **(config or {})}
            
            logger.info(f"开始处理执行结果: {execution.id}")
            
            # 1. 收集原始结果
            original_results = self._collect_step_results(execution)
            
            # 2. 验证结果
            validation_result = await self.validator.validate_results(
                original_results,
                level=config["validation_level"]
            )
            
            # 3. 聚合结果
            aggregated_result = await self.aggregator.aggregate_results(
                original_results,
                strategy=config["aggregation_strategy"],
                config=config.get("aggregation_config", {})
            )
            
            # 4. 格式化输出
            formatted_outputs = {}
            output_formats = config.get("output_formats", config["default_formats"])
            
            for format_type in output_formats:
                try:
                    formatted_content = await self.formatter.format_result(
                        aggregated_result,
                        format_type,
                        config.get("format_config", {})
                    )
                    formatted_outputs[format_type.value] = formatted_content
                except Exception as e:
                    logger.warning(f"格式化失败: {format_type.value}, 错误: {e}")
            
            # 5. 创建处理结果
            processed_result = ProcessedResult(
                execution_id=execution.id,
                original_results=original_results,
                validated_results=original_results if validation_result.is_valid else {},
                aggregated_result=aggregated_result,
                formatted_outputs=formatted_outputs,
                validation_report=validation_result,
                processing_metadata={
                    "processing_config": config,
                    "step_count": len(original_results),
                    "validation_score": validation_result.score,
                    "aggregation_strategy": config["aggregation_strategy"].value,
                    "output_formats": [f.value for f in output_formats]
                },
                created_at=utc_now()
            )
            
            # 6. 缓存结果
            if config["enable_caching"]:
                cache_key = await self.cache.cache_result(
                    execution.id,
                    processed_result,
                    ttl=config["cache_ttl"]
                )
                processed_result.cache_key = cache_key
            
            logger.info(f"执行结果处理完成: {execution.id}")
            return processed_result
            
        except Exception as e:
            logger.error(f"处理执行结果失败: {execution.id}, 错误: {e}")
            
            # 返回错误结果
            return ProcessedResult(
                execution_id=execution.id,
                original_results={},
                validated_results={},
                aggregated_result={"error": f"处理失败: {e}"},
                formatted_outputs={"text": f"结果处理失败: {e}"},
                validation_report=ValidationResult(
                    is_valid=False,
                    errors=[{"message": f"处理异常: {e}"}],
                    warnings=[],
                    score=0.0,
                    metadata={}
                ),
                processing_metadata={"error": str(e)},
                created_at=utc_now()
            )
    
    def _collect_step_results(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """收集步骤结果"""
        results = {}
        
        for step_execution in execution.step_executions:
            if step_execution.status == WorkflowStepStatus.COMPLETED and step_execution.output_data:
                results[step_execution.step_id] = step_execution.output_data
        
        return results
    
    async def get_cached_result(self, execution_id: str) -> Optional[ProcessedResult]:
        """获取缓存的处理结果"""
        return await self.cache.get_cached_result(execution_id)
    
    async def invalidate_result_cache(self, execution_id: str) -> bool:
        """失效结果缓存"""
        return await self.cache.invalidate_cache(execution_id)
    
    def configure_validation_rules(self, rules: List[ValidationRule]):
        """配置验证规则"""
        for rule in rules:
            self.validator.register_rule(rule)
    
    def configure_custom_validator(self, name: str, validator: Callable):
        """配置自定义验证器"""
        self.validator.register_custom_validator(name, validator)
    
    def update_processing_config(self, config: Dict[str, Any]):
        """更新处理配置"""
        self.processing_config.update(config)