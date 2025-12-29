"""
合并策略

实现三路合并、支持语义合并和创建合并算法
"""

import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum

from src.core.logging import get_logger
logger = get_logger(__name__)

class MergeOperation(str, Enum):
    """合并操作"""
    KEEP_LOCAL = "keep_local"
    KEEP_REMOTE = "keep_remote"
    MERGE_VALUES = "merge_values"
    CONFLICT = "conflict"
    DELETE_FIELD = "delete_field"
    ADD_FIELD = "add_field"

class FieldType(str, Enum):
    """字段类型"""
    PRIMITIVE = "primitive"  # 基本类型（字符串、数字、布尔）
    LIST = "list"           # 列表
    DICT = "dict"           # 字典
    TIMESTAMP = "timestamp"  # 时间戳
    REFERENCE = "reference"  # 引用类型

@dataclass
class MergeConflict:
    """合并冲突"""
    field_path: str
    local_value: Any
    remote_value: Any
    base_value: Any = None
    conflict_type: str = "value_conflict"
    suggested_resolution: Optional[Any] = None

@dataclass
class MergeResult:
    """合并结果"""
    merged_data: Dict[str, Any]
    conflicts: List[MergeConflict]
    confidence_score: float
    merge_operations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class MergeStrategies:
    """合并策略集合"""
    
    def __init__(self):
        # 字段优先级映射
        self.field_priorities = {
            "id": 10,
            "created_at": 9,
            "user_id": 8,
            "updated_at": 7,
            "status": 6,
            "name": 5,
            "title": 5,
            "email": 4,
            "description": 3,
            "metadata": 2,
            "cache": 1
        }
        
        # 可自动合并的字段类型
        self.auto_mergeable_types = {
            "tags", "categories", "permissions", "features"
        }
        
        # 时间戳字段
        self.timestamp_fields = {
            "created_at", "updated_at", "deleted_at", "last_seen",
            "expires_at", "start_time", "end_time"
        }
        
        # 累积字段（数值可以累加）
        self.accumulative_fields = {
            "count", "total", "sum", "score", "points", "views", "likes"
        }
    
    def three_way_merge(
        self,
        base_data: Dict[str, Any],
        local_data: Dict[str, Any],
        remote_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """三路合并算法"""
        conflicts = []
        merged_data = {}
        merge_operations = []
        
        # 获取所有字段
        all_fields = set(base_data.keys()) | set(local_data.keys()) | set(remote_data.keys())
        
        for field in all_fields:
            base_value = base_data.get(field)
            local_value = local_data.get(field)
            remote_value = remote_data.get(field)
            
            # 执行字段级合并
            merge_result = self._merge_field(
                field, base_value, local_value, remote_value
            )
            
            if merge_result["operation"] == MergeOperation.CONFLICT:
                conflicts.append(MergeConflict(
                    field_path=field,
                    local_value=local_value,
                    remote_value=remote_value,
                    base_value=base_value,
                    suggested_resolution=merge_result.get("suggested_value")
                ))
                # 在冲突情况下，使用建议值或本地值
                merged_data[field] = merge_result.get("suggested_value", local_value)
            else:
                merged_data[field] = merge_result["value"]
            
            merge_operations.append({
                "field": field,
                "operation": merge_result["operation"].value,
                "confidence": merge_result.get("confidence", 1.0)
            })
        
        # 计算整体置信度
        confidence_score = self._calculate_merge_confidence(merge_operations, conflicts)
        
        return merged_data, confidence_score
    
    def _merge_field(
        self,
        field_name: str,
        base_value: Any,
        local_value: Any,
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并单个字段"""
        # 如果值相同，无需合并
        if local_value == remote_value:
            return {
                "operation": MergeOperation.KEEP_LOCAL,
                "value": local_value,
                "confidence": 1.0
            }
        
        # 处理None值
        if local_value is None and remote_value is not None:
            return {
                "operation": MergeOperation.KEEP_REMOTE,
                "value": remote_value,
                "confidence": 0.9
            }
        elif local_value is not None and remote_value is None:
            return {
                "operation": MergeOperation.KEEP_LOCAL,
                "value": local_value,
                "confidence": 0.9
            }
        elif local_value is None and remote_value is None:
            return {
                "operation": MergeOperation.DELETE_FIELD,
                "value": None,
                "confidence": 1.0
            }
        
        # 确定字段类型
        field_type = self._determine_field_type(field_name, local_value, remote_value)
        
        # 根据字段类型选择合并策略
        if field_type == FieldType.TIMESTAMP:
            return self._merge_timestamp_field(field_name, base_value, local_value, remote_value)
        elif field_type == FieldType.LIST:
            return self._merge_list_field(field_name, base_value, local_value, remote_value)
        elif field_type == FieldType.DICT:
            return self._merge_dict_field(field_name, base_value, local_value, remote_value)
        elif field_name in self.accumulative_fields:
            return self._merge_accumulative_field(field_name, base_value, local_value, remote_value)
        else:
            return self._merge_primitive_field(field_name, base_value, local_value, remote_value)
    
    def _determine_field_type(self, field_name: str, local_value: Any, remote_value: Any) -> FieldType:
        """确定字段类型"""
        if field_name in self.timestamp_fields:
            return FieldType.TIMESTAMP
        elif isinstance(local_value, list) or isinstance(remote_value, list):
            return FieldType.LIST
        elif isinstance(local_value, dict) or isinstance(remote_value, dict):
            return FieldType.DICT
        else:
            return FieldType.PRIMITIVE
    
    def _merge_timestamp_field(
        self, 
        field_name: str, 
        base_value: Any, 
        local_value: Any, 
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并时间戳字段"""
        # 对于时间戳，通常取最新的值
        try:
            if field_name == "updated_at":
                # updated_at 取最新的
                if isinstance(local_value, str) and isinstance(remote_value, str):
                    local_dt = datetime.fromisoformat(local_value.replace('Z', '+00:00'))
                    remote_dt = datetime.fromisoformat(remote_value.replace('Z', '+00:00'))
                    
                    if local_dt >= remote_dt:
                        return {
                            "operation": MergeOperation.KEEP_LOCAL,
                            "value": local_value,
                            "confidence": 0.9
                        }
                    else:
                        return {
                            "operation": MergeOperation.KEEP_REMOTE,
                            "value": remote_value,
                            "confidence": 0.9
                        }
            elif field_name == "created_at":
                # created_at 取最早的
                if isinstance(local_value, str) and isinstance(remote_value, str):
                    local_dt = datetime.fromisoformat(local_value.replace('Z', '+00:00'))
                    remote_dt = datetime.fromisoformat(remote_value.replace('Z', '+00:00'))
                    
                    if local_dt <= remote_dt:
                        return {
                            "operation": MergeOperation.KEEP_LOCAL,
                            "value": local_value,
                            "confidence": 0.9
                        }
                    else:
                        return {
                            "operation": MergeOperation.KEEP_REMOTE,
                            "value": remote_value,
                            "confidence": 0.9
                        }
        except (ValueError, TypeError):
            logger.debug("时间戳解析失败，返回冲突", exc_info=True)
        
        # 如果无法解析时间戳，标记为冲突
        return {
            "operation": MergeOperation.CONFLICT,
            "suggested_value": local_value,
            "confidence": 0.3
        }
    
    def _merge_list_field(
        self, 
        field_name: str, 
        base_value: Any, 
        local_value: Any, 
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并列表字段"""
        if not isinstance(local_value, list) or not isinstance(remote_value, list):
            return {
                "operation": MergeOperation.CONFLICT,
                "suggested_value": local_value,
                "confidence": 0.2
            }
        
        # 对于某些字段类型，可以自动合并
        if field_name in self.auto_mergeable_types:
            # 合并去重
            merged_list = list(set(local_value + remote_value))
            return {
                "operation": MergeOperation.MERGE_VALUES,
                "value": merged_list,
                "confidence": 0.8
            }
        
        # 对于普通列表，检查是否只是添加操作
        base_list = base_value if isinstance(base_value, list) else []
        local_additions = [item for item in local_value if item not in base_list]
        remote_additions = [item for item in remote_value if item not in base_list]
        
        if not local_additions and remote_additions:
            # 只有远程添加
            return {
                "operation": MergeOperation.KEEP_REMOTE,
                "value": remote_value,
                "confidence": 0.9
            }
        elif local_additions and not remote_additions:
            # 只有本地添加
            return {
                "operation": MergeOperation.KEEP_LOCAL,
                "value": local_value,
                "confidence": 0.9
            }
        elif local_additions and remote_additions:
            # 两边都有添加，尝试合并
            merged_list = base_list + local_additions + remote_additions
            return {
                "operation": MergeOperation.MERGE_VALUES,
                "value": merged_list,
                "confidence": 0.7
            }
        
        # 默认冲突
        return {
            "operation": MergeOperation.CONFLICT,
            "suggested_value": local_value,
            "confidence": 0.3
        }
    
    def _merge_dict_field(
        self, 
        field_name: str, 
        base_value: Any, 
        local_value: Any, 
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并字典字段"""
        if not isinstance(local_value, dict) or not isinstance(remote_value, dict):
            return {
                "operation": MergeOperation.CONFLICT,
                "suggested_value": local_value,
                "confidence": 0.2
            }
        
        # 递归合并字典
        base_dict = base_value if isinstance(base_value, dict) else {}
        merged_dict = {}
        has_conflicts = False
        
        all_keys = set(local_value.keys()) | set(remote_value.keys())
        
        for key in all_keys:
            local_sub_value = local_value.get(key)
            remote_sub_value = remote_value.get(key)
            base_sub_value = base_dict.get(key)
            
            sub_result = self._merge_field(
                f"{field_name}.{key}",
                base_sub_value,
                local_sub_value,
                remote_sub_value
            )
            
            if sub_result["operation"] == MergeOperation.CONFLICT:
                has_conflicts = True
                merged_dict[key] = sub_result.get("suggested_value", local_sub_value)
            else:
                merged_dict[key] = sub_result["value"]
        
        confidence = 0.5 if has_conflicts else 0.8
        operation = MergeOperation.CONFLICT if has_conflicts else MergeOperation.MERGE_VALUES
        
        return {
            "operation": operation,
            "value": merged_dict,
            "confidence": confidence
        }
    
    def _merge_accumulative_field(
        self, 
        field_name: str, 
        base_value: Any, 
        local_value: Any, 
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并累积字段"""
        try:
            # 尝试数值累加
            if isinstance(local_value, (int, float)) and isinstance(remote_value, (int, float)):
                base_num = base_value if isinstance(base_value, (int, float)) else 0
                local_delta = local_value - base_num
                remote_delta = remote_value - base_num
                merged_value = base_num + local_delta + remote_delta
                
                return {
                    "operation": MergeOperation.MERGE_VALUES,
                    "value": merged_value,
                    "confidence": 0.8
                }
        except (TypeError, ValueError):
            logger.debug("累积字段数值合并失败", exc_info=True)
        
        # 如果不能累加，取最大值
        try:
            if isinstance(local_value, (int, float)) and isinstance(remote_value, (int, float)):
                return {
                    "operation": MergeOperation.MERGE_VALUES,
                    "value": max(local_value, remote_value),
                    "confidence": 0.6
                }
        except (TypeError, ValueError):
            logger.debug("累积字段最大值合并失败", exc_info=True)
        
        # 默认冲突
        return {
            "operation": MergeOperation.CONFLICT,
            "suggested_value": local_value,
            "confidence": 0.3
        }
    
    def _merge_primitive_field(
        self, 
        field_name: str, 
        base_value: Any, 
        local_value: Any, 
        remote_value: Any
    ) -> Dict[str, Any]:
        """合并基本类型字段"""
        # 检查字段优先级
        field_priority = self.field_priorities.get(field_name, 5)
        
        # 高优先级字段倾向于保持稳定
        if field_priority >= 8:
            # 优先保留本地值
            return {
                "operation": MergeOperation.KEEP_LOCAL,
                "value": local_value,
                "confidence": 0.7
            }
        
        # 检查是否只有一边修改
        if base_value is not None:
            if local_value == base_value and remote_value != base_value:
                # 只有远程修改
                return {
                    "operation": MergeOperation.KEEP_REMOTE,
                    "value": remote_value,
                    "confidence": 0.9
                }
            elif local_value != base_value and remote_value == base_value:
                # 只有本地修改
                return {
                    "operation": MergeOperation.KEEP_LOCAL,
                    "value": local_value,
                    "confidence": 0.9
                }
        
        # 尝试智能合并字符串
        if isinstance(local_value, str) and isinstance(remote_value, str):
            string_merge_result = self._merge_string_intelligently(local_value, remote_value)
            if string_merge_result:
                return {
                    "operation": MergeOperation.MERGE_VALUES,
                    "value": string_merge_result,
                    "confidence": 0.6
                }
        
        # 默认标记为冲突
        return {
            "operation": MergeOperation.CONFLICT,
            "suggested_value": local_value,
            "confidence": 0.3
        }
    
    def _merge_string_intelligently(self, local_str: str, remote_str: str) -> Optional[str]:
        """智能合并字符串"""
        # 检查是否为简单的添加操作
        if local_str in remote_str:
            return remote_str
        elif remote_str in local_str:
            return local_str
        
        # 检查是否为空格或标点的差异
        local_normalized = ''.join(local_str.split())
        remote_normalized = ''.join(remote_str.split())
        
        if local_normalized == remote_normalized:
            # 保留较长的版本（可能有更好的格式）
            return local_str if len(local_str) >= len(remote_str) else remote_str
        
        # 检查是否为大小写差异
        if local_str.lower() == remote_str.lower():
            # 保留有更多大写字母的版本（可能是标题格式）
            local_upper_count = sum(1 for c in local_str if c.isupper())
            remote_upper_count = sum(1 for c in remote_str if c.isupper())
            return local_str if local_upper_count >= remote_upper_count else remote_str
        
        return None
    
    def _calculate_merge_confidence(
        self, 
        merge_operations: List[Dict[str, Any]], 
        conflicts: List[MergeConflict]
    ) -> float:
        """计算合并置信度"""
        if not merge_operations:
            return 0.0
        
        # 基础置信度是所有操作置信度的平均值
        total_confidence = sum(op.get("confidence", 0.5) for op in merge_operations)
        base_confidence = total_confidence / len(merge_operations)
        
        # 冲突会降低置信度
        conflict_penalty = len(conflicts) * 0.1
        
        # 确保置信度在0-1之间
        final_confidence = max(0.0, min(1.0, base_confidence - conflict_penalty))
        
        return final_confidence
    
    def semantic_merge(
        self,
        local_data: Dict[str, Any],
        remote_data: Dict[str, Any],
        schema_info: Optional[Dict[str, Any]] = None
    ) -> MergeResult:
        """语义合并"""
        conflicts = []
        merged_data = {}
        merge_operations = []
        
        # 如果有schema信息，使用schema指导合并
        if schema_info:
            for field_name, field_schema in schema_info.items():
                local_value = local_data.get(field_name)
                remote_value = remote_data.get(field_name)
                
                merge_result = self._semantic_merge_field(
                    field_name, local_value, remote_value, field_schema
                )
                
                if merge_result["has_conflict"]:
                    conflicts.append(MergeConflict(
                        field_path=field_name,
                        local_value=local_value,
                        remote_value=remote_value,
                        conflict_type="semantic_conflict",
                        suggested_resolution=merge_result["value"]
                    ))
                
                merged_data[field_name] = merge_result["value"]
                merge_operations.append({
                    "field": field_name,
                    "operation": merge_result["operation"],
                    "confidence": merge_result["confidence"]
                })
        else:
            # 无schema信息时，回退到普通三路合并
            merged_data, confidence = self.three_way_merge({}, local_data, remote_data)
            return MergeResult(
                merged_data=merged_data,
                conflicts=[],
                confidence_score=confidence
            )
        
        # 计算整体置信度
        confidence = self._calculate_merge_confidence(merge_operations, conflicts)
        
        return MergeResult(
            merged_data=merged_data,
            conflicts=conflicts,
            confidence_score=confidence,
            merge_operations=merge_operations,
            metadata={"merge_type": "semantic", "schema_guided": True}
        )
    
    def _semantic_merge_field(
        self,
        field_name: str,
        local_value: Any,
        remote_value: Any,
        field_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """基于语义的字段合并"""
        field_type = field_schema.get("type", "string")
        constraints = field_schema.get("constraints", {})
        
        # 基于类型的语义合并
        if field_type == "enum":
            # 枚举类型：检查是否都是有效值
            valid_values = constraints.get("values", [])
            if local_value in valid_values and remote_value in valid_values:
                # 如果都有效，使用优先级或默认选择
                priority_order = constraints.get("priority_order", valid_values)
                if local_value in priority_order and remote_value in priority_order:
                    local_priority = priority_order.index(local_value)
                    remote_priority = priority_order.index(remote_value)
                    chosen_value = local_value if local_priority <= remote_priority else remote_value
                    return {
                        "value": chosen_value,
                        "operation": "semantic_select",
                        "confidence": 0.8,
                        "has_conflict": False
                    }
        
        elif field_type == "reference":
            # 引用类型：检查引用的有效性
            ref_table = constraints.get("ref_table")
            if ref_table:
                # 这里应该检查引用的有效性，简化处理
                return {
                    "value": local_value,  # 优先保留本地引用
                    "operation": "keep_local_reference",
                    "confidence": 0.7,
                    "has_conflict": local_value != remote_value
                }
        
        # 默认处理
        if local_value == remote_value:
            return {
                "value": local_value,
                "operation": "no_change",
                "confidence": 1.0,
                "has_conflict": False
            }
        else:
            return {
                "value": local_value,
                "operation": "conflict",
                "confidence": 0.3,
                "has_conflict": True
            }
    
    def custom_merge(
        self,
        local_data: Dict[str, Any],
        remote_data: Dict[str, Any],
        merge_rules: Dict[str, Any]
    ) -> MergeResult:
        """自定义合并策略"""
        conflicts = []
        merged_data = {}
        merge_operations = []
        
        # 应用自定义规则
        for field_name, rule in merge_rules.items():
            local_value = local_data.get(field_name)
            remote_value = remote_data.get(field_name)
            
            rule_type = rule.get("type", "default")
            
            if rule_type == "always_local":
                merged_data[field_name] = local_value
                merge_operations.append({
                    "field": field_name,
                    "operation": "always_local",
                    "confidence": 1.0
                })
            elif rule_type == "always_remote":
                merged_data[field_name] = remote_value
                merge_operations.append({
                    "field": field_name,
                    "operation": "always_remote",
                    "confidence": 1.0
                })
            elif rule_type == "custom_function":
                # 执行自定义函数
                func_name = rule.get("function")
                if func_name and hasattr(self, f"_merge_{func_name}"):
                    func = getattr(self, f"_merge_{func_name}")
                    result = func(local_value, remote_value, rule.get("params", {}))
                    merged_data[field_name] = result["value"]
                    merge_operations.append({
                        "field": field_name,
                        "operation": f"custom_{func_name}",
                        "confidence": result.get("confidence", 0.5)
                    })
            else:
                # 使用默认合并
                merge_result = self._merge_field(field_name, None, local_value, remote_value)
                merged_data[field_name] = merge_result["value"]
                merge_operations.append({
                    "field": field_name,
                    "operation": merge_result["operation"].value,
                    "confidence": merge_result.get("confidence", 0.5)
                })
        
        # 处理规则中未覆盖的字段
        all_fields = set(local_data.keys()) | set(remote_data.keys())
        uncovered_fields = all_fields - set(merge_rules.keys())
        
        for field_name in uncovered_fields:
            local_value = local_data.get(field_name)
            remote_value = remote_data.get(field_name)
            
            merge_result = self._merge_field(field_name, None, local_value, remote_value)
            merged_data[field_name] = merge_result["value"]
            merge_operations.append({
                "field": field_name,
                "operation": merge_result["operation"].value,
                "confidence": merge_result.get("confidence", 0.5)
            })
        
        # 计算置信度
        confidence = self._calculate_merge_confidence(merge_operations, conflicts)
        
        return MergeResult(
            merged_data=merged_data,
            conflicts=conflicts,
            confidence_score=confidence,
            merge_operations=merge_operations,
            metadata={"merge_type": "custom", "rules_applied": len(merge_rules)}
        )
