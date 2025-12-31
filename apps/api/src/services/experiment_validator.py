"""
A/B测试实验配置验证服务 - 提供全面的实验配置验证逻辑
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import re
from dataclasses import dataclass
from src.models.schemas.experiment import (
    CreateExperimentRequest, ExperimentVariant, TrafficAllocation,
    TargetingRule, ExperimentConfig, ExperimentStatus
)

@dataclass
class ValidationError:
    """验证错误"""
    field: str
    code: str
    message: str
    severity: str = "error"  # error, warning, info

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class ExperimentConfigValidator:
    """实验配置验证器"""
    
    def __init__(self):
        self.min_experiment_duration = timedelta(days=1)
        self.max_experiment_duration = timedelta(days=90)
        self.min_variants = 2
        self.max_variants = 10
        self.min_sample_size = 100
        self.max_sample_size = 1000000
        self.valid_significance_levels = [0.01, 0.05, 0.1]
        self.valid_power_levels = [0.7, 0.8, 0.9, 0.95]
    
    def validate_create_experiment(self, experiment_request: CreateExperimentRequest) -> ValidationResult:
        """验证创建实验请求"""
        errors = []
        warnings = []
        
        # 基本字段验证
        errors.extend(self._validate_basic_fields(experiment_request))
        
        # 变体配置验证
        errors.extend(self._validate_variants(experiment_request.variants))
        
        # 流量分配验证
        errors.extend(self._validate_traffic_allocation(
            experiment_request.variants, 
            experiment_request.traffic_allocation
        ))
        
        # 时间配置验证
        time_errors, time_warnings = self._validate_time_config(
            experiment_request.start_date, 
            experiment_request.end_date
        )
        errors.extend(time_errors)
        warnings.extend(time_warnings)
        
        # 指标配置验证
        metric_errors, metric_warnings = self._validate_metrics(
            experiment_request.success_metrics,
            experiment_request.guardrail_metrics
        )
        errors.extend(metric_errors)
        warnings.extend(metric_warnings)
        
        # 统计配置验证
        stat_errors, stat_warnings = self._validate_statistical_config(
            experiment_request.minimum_sample_size,
            experiment_request.significance_level,
            experiment_request.power
        )
        errors.extend(stat_errors)
        warnings.extend(stat_warnings)
        
        # 定向规则验证
        targeting_errors = self._validate_targeting_rules(experiment_request.targeting_rules)
        errors.extend(targeting_errors)
        
        # 层配置验证
        layer_errors = self._validate_layers(experiment_request.layers)
        errors.extend(layer_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_experiment_update(self, current_experiment: ExperimentConfig, 
                                 update_fields: Dict[str, Any]) -> ValidationResult:
        """验证实验更新请求"""
        errors = []
        warnings = []
        
        # 检查实验状态是否允许更新
        if current_experiment.status == ExperimentStatus.COMPLETED:
            errors.append(ValidationError(
                field="status",
                code="EXPERIMENT_COMPLETED",
                message="已完成的实验不能修改"
            ))
        
        if current_experiment.status == ExperimentStatus.TERMINATED:
            errors.append(ValidationError(
                field="status",
                code="EXPERIMENT_TERMINATED",
                message="已终止的实验不能修改"
            ))
        
        # 检查运行中实验的限制
        if current_experiment.status == ExperimentStatus.RUNNING:
            restricted_fields = ['variants', 'traffic_allocation', 'layers', 'start_date']
            for field in restricted_fields:
                if field in update_fields:
                    errors.append(ValidationError(
                        field=field,
                        code="FIELD_IMMUTABLE_WHILE_RUNNING",
                        message=f"运行中的实验不能修改{field}字段"
                    ))
        
        # 验证更新字段的合法性
        if 'end_date' in update_fields:
            end_date = update_fields['end_date']
            if end_date and end_date <= current_experiment.start_date:
                errors.append(ValidationError(
                    field="end_date",
                    code="INVALID_END_DATE",
                    message="结束时间必须晚于开始时间"
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_experiment_start(self, experiment: ExperimentConfig) -> ValidationResult:
        """验证实验启动条件"""
        errors = []
        warnings = []
        
        # 检查实验状态
        if experiment.status != ExperimentStatus.DRAFT:
            errors.append(ValidationError(
                field="status",
                code="INVALID_STATUS_FOR_START",
                message=f"只有草稿状态的实验可以启动，当前状态：{experiment.status}"
            ))
        
        # 检查开始时间
        now = utc_now()
        if experiment.start_date > now + timedelta(hours=1):
            warnings.append(ValidationError(
                field="start_date",
                code="FUTURE_START_DATE",
                message=f"实验预计在{experiment.start_date}开始",
                severity="warning"
            ))
        
        # 检查配置完整性
        config_errors = self._validate_experiment_readiness(experiment)
        errors.extend(config_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_statistical_power(self, experiment: ExperimentConfig, 
                                 current_sample_size: int, effect_size: float = 0.05) -> ValidationResult:
        """验证统计功效"""
        errors = []
        warnings = []
        
        try:
            # 计算当前配置下的统计功效
            required_sample_size = self._calculate_required_sample_size(
                effect_size, experiment.significance_level, experiment.power
            )
            
            if current_sample_size < required_sample_size:
                warnings.append(ValidationError(
                    field="sample_size",
                    code="INSUFFICIENT_SAMPLE_SIZE",
                    message=f"当前样本量({current_sample_size})低于统计功效要求({required_sample_size})",
                    severity="warning"
                ))
            
            # 检查最小样本量
            if current_sample_size < experiment.minimum_sample_size:
                errors.append(ValidationError(
                    field="sample_size",
                    code="BELOW_MINIMUM_SAMPLE_SIZE",
                    message=f"样本量({current_sample_size})低于设定的最小值({experiment.minimum_sample_size})"
                ))
            
        except Exception as e:
            warnings.append(ValidationError(
                field="statistical_power",
                code="POWER_CALCULATION_ERROR",
                message=f"统计功效计算失败: {str(e)}",
                severity="warning"
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    # 私有验证方法
    def _validate_basic_fields(self, experiment_request: CreateExperimentRequest) -> List[ValidationError]:
        """验证基本字段"""
        errors = []
        
        # 实验名称
        if not experiment_request.name or len(experiment_request.name.strip()) == 0:
            errors.append(ValidationError("name", "EMPTY_NAME", "实验名称不能为空"))
        elif len(experiment_request.name) > 255:
            errors.append(ValidationError("name", "NAME_TOO_LONG", "实验名称长度不能超过255字符"))
        
        # 实验描述
        if not experiment_request.description or len(experiment_request.description.strip()) == 0:
            errors.append(ValidationError("description", "EMPTY_DESCRIPTION", "实验描述不能为空"))
        elif len(experiment_request.description) > 2000:
            errors.append(ValidationError("description", "DESCRIPTION_TOO_LONG", "实验描述长度不能超过2000字符"))
        
        # 实验假设
        if not experiment_request.hypothesis or len(experiment_request.hypothesis.strip()) == 0:
            errors.append(ValidationError("hypothesis", "EMPTY_HYPOTHESIS", "实验假设不能为空"))
        
        # 负责人
        if not experiment_request.owner or len(experiment_request.owner.strip()) == 0:
            errors.append(ValidationError("owner", "EMPTY_OWNER", "实验负责人不能为空"))
        
        return errors
    
    def _validate_variants(self, variants: List[ExperimentVariant]) -> List[ValidationError]:
        """验证实验变体"""
        errors = []
        
        # 变体数量检查
        if len(variants) < self.min_variants:
            errors.append(ValidationError(
                "variants", "TOO_FEW_VARIANTS", 
                f"实验至少需要{self.min_variants}个变体"
            ))
        elif len(variants) > self.max_variants:
            errors.append(ValidationError(
                "variants", "TOO_MANY_VARIANTS", 
                f"实验最多支持{self.max_variants}个变体"
            ))
        
        # 变体ID唯一性检查
        variant_ids = [v.variant_id for v in variants]
        if len(variant_ids) != len(set(variant_ids)):
            errors.append(ValidationError(
                "variants", "DUPLICATE_VARIANT_IDS", 
                "变体ID必须唯一"
            ))
        
        # 对照组检查
        control_groups = [v for v in variants if v.is_control]
        if len(control_groups) == 0:
            errors.append(ValidationError(
                "variants", "NO_CONTROL_GROUP", 
                "实验必须有一个对照组"
            ))
        elif len(control_groups) > 1:
            errors.append(ValidationError(
                "variants", "MULTIPLE_CONTROL_GROUPS", 
                "实验只能有一个对照组"
            ))
        
        # 验证每个变体
        for i, variant in enumerate(variants):
            # 变体ID格式检查
            if not re.match(r'^[a-zA-Z0-9_-]+$', variant.variant_id):
                errors.append(ValidationError(
                    f"variants[{i}].variant_id", "INVALID_VARIANT_ID_FORMAT", 
                    f"变体ID '{variant.variant_id}' 格式不正确，只能包含字母、数字、下划线和短横线"
                ))
            
            # 变体名称检查
            if not variant.name or len(variant.name.strip()) == 0:
                errors.append(ValidationError(
                    f"variants[{i}].name", "EMPTY_VARIANT_NAME", 
                    f"变体 '{variant.variant_id}' 的名称不能为空"
                ))
        
        return errors
    
    def _validate_traffic_allocation(self, variants: List[ExperimentVariant], 
                                   allocations: List[TrafficAllocation]) -> List[ValidationError]:
        """验证流量分配"""
        errors = []
        
        # 检查分配数量是否与变体数量匹配
        if len(allocations) != len(variants):
            errors.append(ValidationError(
                "traffic_allocation", "ALLOCATION_VARIANT_MISMATCH", 
                "流量分配数量与变体数量不匹配"
            ))
        
        # 检查分配ID是否都有对应的变体
        variant_ids = {v.variant_id for v in variants}
        allocation_ids = {a.variant_id for a in allocations}
        
        missing_variants = variant_ids - allocation_ids
        if missing_variants:
            errors.append(ValidationError(
                "traffic_allocation", "MISSING_VARIANT_ALLOCATION", 
                f"缺少变体的流量分配: {missing_variants}"
            ))
        
        extra_allocations = allocation_ids - variant_ids
        if extra_allocations:
            errors.append(ValidationError(
                "traffic_allocation", "EXTRA_VARIANT_ALLOCATION", 
                f"多余的流量分配: {extra_allocations}"
            ))
        
        # 检查分配百分比
        total_percentage = sum(a.percentage for a in allocations)
        if abs(total_percentage - 100.0) > 0.01:
            errors.append(ValidationError(
                "traffic_allocation", "INVALID_TOTAL_PERCENTAGE", 
                f"流量分配总和必须为100%，当前为{total_percentage}%"
            ))
        
        # 检查每个分配的百分比范围
        for allocation in allocations:
            if allocation.percentage < 0 or allocation.percentage > 100:
                errors.append(ValidationError(
                    f"traffic_allocation.{allocation.variant_id}", "INVALID_PERCENTAGE_RANGE", 
                    f"变体 '{allocation.variant_id}' 的流量分配必须在0-100%之间"
                ))
            elif allocation.percentage == 0:
                errors.append(ValidationError(
                    f"traffic_allocation.{allocation.variant_id}", "ZERO_PERCENTAGE", 
                    f"变体 '{allocation.variant_id}' 的流量分配不能为0%"
                ))
        
        return errors
    
    def _validate_time_config(self, start_date: datetime, 
                            end_date: Optional[datetime]) -> Tuple[List[ValidationError], List[ValidationError]]:
        """验证时间配置"""
        errors = []
        warnings = []
        
        now = utc_now()
        
        # 开始时间验证
        if start_date < now - timedelta(hours=1):
            warnings.append(ValidationError(
                "start_date", "PAST_START_DATE", 
                "实验开始时间已过期",
                severity="warning"
            ))
        elif start_date > now + timedelta(days=30):
            warnings.append(ValidationError(
                "start_date", "FAR_FUTURE_START_DATE", 
                "实验开始时间距离现在太远",
                severity="warning"
            ))
        
        # 结束时间验证
        if end_date:
            if end_date <= start_date:
                errors.append(ValidationError(
                    "end_date", "INVALID_END_DATE", 
                    "结束时间必须晚于开始时间"
                ))
            else:
                duration = end_date - start_date
                if duration < self.min_experiment_duration:
                    warnings.append(ValidationError(
                        "end_date", "SHORT_DURATION", 
                        f"实验持续时间过短，建议至少{self.min_experiment_duration.days}天",
                        severity="warning"
                    ))
                elif duration > self.max_experiment_duration:
                    warnings.append(ValidationError(
                        "end_date", "LONG_DURATION", 
                        f"实验持续时间过长，建议不超过{self.max_experiment_duration.days}天",
                        severity="warning"
                    ))
        
        return errors, warnings
    
    def _validate_metrics(self, success_metrics: List[str], 
                        guardrail_metrics: List[str]) -> Tuple[List[ValidationError], List[ValidationError]]:
        """验证指标配置"""
        errors = []
        warnings = []
        
        # 成功指标验证
        if not success_metrics:
            errors.append(ValidationError(
                "success_metrics", "NO_SUCCESS_METRICS", 
                "实验至少需要一个成功指标"
            ))
        elif len(success_metrics) > 10:
            warnings.append(ValidationError(
                "success_metrics", "TOO_MANY_SUCCESS_METRICS", 
                "成功指标过多可能导致多重检验问题",
                severity="warning"
            ))
        
        # 检查指标名称格式
        metric_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')
        for metric in success_metrics:
            if not metric_pattern.match(metric):
                errors.append(ValidationError(
                    "success_metrics", "INVALID_METRIC_NAME", 
                    f"指标名称 '{metric}' 格式不正确"
                ))
        
        for metric in guardrail_metrics:
            if not metric_pattern.match(metric):
                errors.append(ValidationError(
                    "guardrail_metrics", "INVALID_METRIC_NAME", 
                    f"护栏指标名称 '{metric}' 格式不正确"
                ))
        
        # 检查重复指标
        all_metrics = success_metrics + guardrail_metrics
        if len(all_metrics) != len(set(all_metrics)):
            errors.append(ValidationError(
                "metrics", "DUPLICATE_METRICS", 
                "成功指标和护栏指标中存在重复"
            ))
        
        return errors, warnings
    
    def _validate_statistical_config(self, minimum_sample_size: int, 
                                   significance_level: float, 
                                   power: float) -> Tuple[List[ValidationError], List[ValidationError]]:
        """验证统计配置"""
        errors = []
        warnings = []
        
        # 最小样本量验证
        if minimum_sample_size < self.min_sample_size:
            errors.append(ValidationError(
                "minimum_sample_size", "TOO_SMALL_SAMPLE_SIZE", 
                f"最小样本量不能小于{self.min_sample_size}"
            ))
        elif minimum_sample_size > self.max_sample_size:
            warnings.append(ValidationError(
                "minimum_sample_size", "VERY_LARGE_SAMPLE_SIZE", 
                f"最小样本量过大({minimum_sample_size})，可能需要很长时间收集",
                severity="warning"
            ))
        
        # 显著性水平验证
        if significance_level not in self.valid_significance_levels:
            warnings.append(ValidationError(
                "significance_level", "UNUSUAL_SIGNIFICANCE_LEVEL", 
                f"不常见的显著性水平({significance_level})，建议使用{self.valid_significance_levels}",
                severity="warning"
            ))
        
        # 统计功效验证
        if power not in self.valid_power_levels:
            warnings.append(ValidationError(
                "power", "UNUSUAL_POWER_LEVEL", 
                f"不常见的统计功效({power})，建议使用{self.valid_power_levels}",
                severity="warning"
            ))
        
        return errors, warnings
    
    def _validate_targeting_rules(self, targeting_rules: List[TargetingRule]) -> List[ValidationError]:
        """验证定向规则"""
        errors = []
        
        for i, rule in enumerate(targeting_rules):
            # 验证操作符
            valid_operators = ['eq', 'ne', 'in', 'not_in', 'gt', 'lt', 'gte', 'lte']
            if rule.operator not in valid_operators:
                errors.append(ValidationError(
                    f"targeting_rules[{i}].operator", "INVALID_OPERATOR", 
                    f"无效的操作符 '{rule.operator}'"
                ))
            
            # 验证属性名称
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', rule.attribute):
                errors.append(ValidationError(
                    f"targeting_rules[{i}].attribute", "INVALID_ATTRIBUTE_NAME", 
                    f"无效的属性名称 '{rule.attribute}'"
                ))
        
        return errors
    
    def _validate_layers(self, layers: List[str]) -> List[ValidationError]:
        """验证层配置"""
        errors = []
        
        # 检查层名称格式
        for layer in layers:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', layer):
                errors.append(ValidationError(
                    "layers", "INVALID_LAYER_NAME", 
                    f"无效的层名称 '{layer}'"
                ))
        
        # 检查重复层
        if len(layers) != len(set(layers)):
            errors.append(ValidationError(
                "layers", "DUPLICATE_LAYERS", 
                "层名称不能重复"
            ))
        
        return errors
    
    def _validate_experiment_readiness(self, experiment: ExperimentConfig) -> List[ValidationError]:
        """验证实验是否准备就绪"""
        errors = []
        
        # 检查是否有足够的配置信息
        if not experiment.variants:
            errors.append(ValidationError(
                "variants", "NO_VARIANTS", 
                "实验没有配置变体"
            ))
        
        if not experiment.traffic_allocation:
            errors.append(ValidationError(
                "traffic_allocation", "NO_TRAFFIC_ALLOCATION", 
                "实验没有配置流量分配"
            ))
        
        if not experiment.success_metrics:
            errors.append(ValidationError(
                "success_metrics", "NO_SUCCESS_METRICS", 
                "实验没有配置成功指标"
            ))
        
        return errors
    
    def _calculate_required_sample_size(self, effect_size: float, alpha: float, power: float) -> int:
        """计算所需样本量（简化实现）"""
        # 这里使用简化的样本量计算公式
        # 实际应用中应该使用更精确的统计方法
        import math
        
        # 标准正态分布的分位数（近似值）
        z_alpha = 1.96 if alpha == 0.05 else (2.58 if alpha == 0.01 else 1.64)
        z_beta = 0.84 if power == 0.8 else (1.04 if power == 0.85 else 1.28)
        
        # Cohen's d 效果量转换
        if effect_size == 0:
            effect_size = 0.05  # 默认小效果量
        
        # 样本量计算（每组）
        n_per_group = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        
        # 总样本量（假设两组）
        total_n = int(math.ceil(n_per_group * 2))
        
        return total_n
