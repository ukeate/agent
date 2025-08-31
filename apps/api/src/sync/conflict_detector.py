"""
冲突检测器

实现冲突检测算法、支持多种冲突类型和创建冲突分类机制
"""

import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from ..models.schemas.offline import (
    SyncOperation, SyncOperationType, VectorClock, 
    ConflictRecord, ConflictType, ConflictResolutionStrategy
)
from .vector_clock import VectorClockManager, CausalRelation


class ConflictSeverity(str, Enum):
    """冲突严重程度"""
    LOW = "low"           # 低严重程度：不影响功能的冲突
    MEDIUM = "medium"     # 中等严重程度：可能影响部分功能
    HIGH = "high"         # 高严重程度：严重影响功能
    CRITICAL = "critical" # 关键严重程度：可能导致数据丢失


class ConflictCategory(str, Enum):
    """冲突分类"""
    DATA_CONFLICT = "data_conflict"         # 数据冲突
    SCHEMA_CONFLICT = "schema_conflict"     # 模式冲突
    PERMISSION_CONFLICT = "permission_conflict"  # 权限冲突
    TEMPORAL_CONFLICT = "temporal_conflict" # 时间冲突
    SEMANTIC_CONFLICT = "semantic_conflict" # 语义冲突


@dataclass
class ConflictContext:
    """冲突上下文"""
    local_operation: SyncOperation
    remote_operation: SyncOperation
    conflict_type: ConflictType
    conflict_category: ConflictCategory
    severity: ConflictSeverity
    auto_resolvable: bool
    confidence_score: float  # 0.0-1.0，冲突检测的置信度
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictDetectionResult:
    """冲突检测结果"""
    has_conflicts: bool
    conflicts: List[ConflictContext]
    total_operations_checked: int
    detection_duration_ms: float
    summary: Dict[str, Any] = field(default_factory=dict)


class ConflictDetector:
    """冲突检测器"""
    
    def __init__(self):
        self.vector_clock_manager = VectorClockManager()
        
        # 检测配置
        self.detection_strategies = {
            ConflictType.UPDATE_UPDATE: self._detect_update_update_conflict,
            ConflictType.UPDATE_DELETE: self._detect_update_delete_conflict,
            ConflictType.DELETE_UPDATE: self._detect_delete_update_conflict,
            ConflictType.CREATE_CREATE: self._detect_create_create_conflict,
            ConflictType.SCHEMA_MISMATCH: self._detect_schema_conflict,
            ConflictType.PERMISSION_DENIED: self._detect_permission_conflict
        }
        
        # 冲突阈值配置
        self.conflict_thresholds = {
            "time_window_seconds": 300,    # 5分钟内的操作认为可能冲突
            "field_similarity_threshold": 0.8,  # 字段相似度阈值
            "auto_resolve_confidence": 0.9,     # 自动解决的置信度阈值
        }
        
        # 统计信息
        self.detection_stats = {
            "total_detections": 0,
            "conflicts_found": 0,
            "auto_resolvable_conflicts": 0,
            "detection_time_total_ms": 0.0
        }
    
    def detect_conflicts(
        self,
        local_operations: List[SyncOperation],
        remote_operations: List[SyncOperation],
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictDetectionResult:
        """检测操作间的冲突"""
        start_time = utc_now()
        conflicts = []
        total_operations = len(local_operations) + len(remote_operations)
        
        # 按对象分组操作
        local_by_object = self._group_operations_by_object(local_operations)
        remote_by_object = self._group_operations_by_object(remote_operations)
        
        # 找出共同操作的对象
        common_objects = set(local_by_object.keys()) & set(remote_by_object.keys())
        
        for object_key in common_objects:
            table_name, object_id = object_key
            local_ops = local_by_object[object_key]
            remote_ops = remote_by_object[object_key]
            
            # 检测该对象的冲突
            object_conflicts = self._detect_object_conflicts(
                local_ops, remote_ops, table_name, object_id, context
            )
            conflicts.extend(object_conflicts)
        
        # 检测跨对象冲突
        cross_object_conflicts = self._detect_cross_object_conflicts(
            local_operations, remote_operations, context
        )
        conflicts.extend(cross_object_conflicts)
        
        # 计算检测时间
        end_time = utc_now()
        detection_duration = (end_time - start_time).total_seconds() * 1000
        
        # 更新统计
        self.detection_stats["total_detections"] += 1
        self.detection_stats["conflicts_found"] += len(conflicts)
        self.detection_stats["auto_resolvable_conflicts"] += sum(
            1 for c in conflicts if c.auto_resolvable
        )
        self.detection_stats["detection_time_total_ms"] += detection_duration
        
        # 生成摘要
        summary = self._generate_conflict_summary(conflicts)
        
        return ConflictDetectionResult(
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts,
            total_operations_checked=total_operations,
            detection_duration_ms=detection_duration,
            summary=summary
        )
    
    def _group_operations_by_object(
        self, 
        operations: List[SyncOperation]
    ) -> Dict[Tuple[str, str], List[SyncOperation]]:
        """按对象分组操作"""
        grouped = {}
        
        for operation in operations:
            key = (operation.table_name, operation.object_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(operation)
        
        # 按时间戳排序
        for ops in grouped.values():
            ops.sort(key=lambda x: x.client_timestamp)
        
        return grouped
    
    def _detect_object_conflicts(
        self,
        local_ops: List[SyncOperation],
        remote_ops: List[SyncOperation],
        table_name: str,
        object_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ConflictContext]:
        """检测单个对象的冲突"""
        conflicts = []
        
        # 获取最新的本地和远程操作
        latest_local = local_ops[-1] if local_ops else None
        latest_remote = remote_ops[-1] if remote_ops else None
        
        if not latest_local or not latest_remote:
            return conflicts
        
        # 使用向量时钟检测因果关系
        causal_relation = self.vector_clock_manager.compare_clocks(
            latest_local.vector_clock,
            latest_remote.vector_clock
        )
        
        # 并发操作可能产生冲突
        if causal_relation == CausalRelation.CONCURRENT:
            conflict_type = self._determine_conflict_type(latest_local, latest_remote)
            
            if conflict_type:
                conflict_context = self._analyze_conflict_context(
                    latest_local, latest_remote, conflict_type, context
                )
                conflicts.append(conflict_context)
        
        # 检测时间窗口内的操作冲突
        time_window_conflicts = self._detect_time_window_conflicts(
            local_ops, remote_ops, context
        )
        conflicts.extend(time_window_conflicts)
        
        return conflicts
    
    def _determine_conflict_type(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictType]:
        """确定冲突类型"""
        local_type = local_op.operation_type
        remote_type = remote_op.operation_type
        
        # 检查操作类型组合
        if local_type == SyncOperationType.PUT and remote_type == SyncOperationType.PUT:
            return ConflictType.CREATE_CREATE
        elif local_type in [SyncOperationType.PUT, SyncOperationType.PATCH] and remote_type in [SyncOperationType.PUT, SyncOperationType.PATCH]:
            return ConflictType.UPDATE_UPDATE
        elif local_type in [SyncOperationType.PUT, SyncOperationType.PATCH] and remote_type == SyncOperationType.DELETE:
            return ConflictType.UPDATE_DELETE
        elif local_type == SyncOperationType.DELETE and remote_type in [SyncOperationType.PUT, SyncOperationType.PATCH]:
            return ConflictType.DELETE_UPDATE
        
        return None
    
    def _analyze_conflict_context(
        self,
        local_op: SyncOperation,
        remote_op: SyncOperation,
        conflict_type: ConflictType,
        context: Optional[Dict[str, Any]] = None
    ) -> ConflictContext:
        """分析冲突上下文"""
        # 确定冲突分类
        category = self._categorize_conflict(local_op, remote_op, conflict_type)
        
        # 评估严重程度
        severity = self._assess_conflict_severity(local_op, remote_op, conflict_type)
        
        # 检查是否可以自动解决
        auto_resolvable, confidence = self._assess_auto_resolvability(
            local_op, remote_op, conflict_type
        )
        
        return ConflictContext(
            local_operation=local_op,
            remote_operation=remote_op,
            conflict_type=conflict_type,
            conflict_category=category,
            severity=severity,
            auto_resolvable=auto_resolvable,
            confidence_score=confidence,
            metadata={
                "detection_time": utc_now().isoformat(),
                "context": context or {}
            }
        )
    
    def _categorize_conflict(
        self,
        local_op: SyncOperation,
        remote_op: SyncOperation,
        conflict_type: ConflictType
    ) -> ConflictCategory:
        """分类冲突"""
        # 基于操作数据的简单分类逻辑
        if conflict_type == ConflictType.SCHEMA_MISMATCH:
            return ConflictCategory.SCHEMA_CONFLICT
        elif conflict_type == ConflictType.PERMISSION_DENIED:
            return ConflictCategory.PERMISSION_CONFLICT
        else:
            # 检查是否为语义冲突
            if self._is_semantic_conflict(local_op, remote_op):
                return ConflictCategory.SEMANTIC_CONFLICT
            # 检查是否为时间冲突
            elif self._is_temporal_conflict(local_op, remote_op):
                return ConflictCategory.TEMPORAL_CONFLICT
            else:
                return ConflictCategory.DATA_CONFLICT
    
    def _assess_conflict_severity(
        self,
        local_op: SyncOperation,
        remote_op: SyncOperation,
        conflict_type: ConflictType
    ) -> ConflictSeverity:
        """评估冲突严重程度"""
        # 基于冲突类型的基础严重程度
        base_severity = {
            ConflictType.UPDATE_UPDATE: ConflictSeverity.MEDIUM,
            ConflictType.UPDATE_DELETE: ConflictSeverity.HIGH,
            ConflictType.DELETE_UPDATE: ConflictSeverity.HIGH,
            ConflictType.CREATE_CREATE: ConflictSeverity.LOW,
            ConflictType.SCHEMA_MISMATCH: ConflictSeverity.CRITICAL,
            ConflictType.PERMISSION_DENIED: ConflictSeverity.HIGH
        }.get(conflict_type, ConflictSeverity.MEDIUM)
        
        # 基于数据内容调整严重程度
        if self._contains_critical_fields(local_op, remote_op):
            if base_severity == ConflictSeverity.LOW:
                base_severity = ConflictSeverity.MEDIUM
            elif base_severity == ConflictSeverity.MEDIUM:
                base_severity = ConflictSeverity.HIGH
        
        return base_severity
    
    def _assess_auto_resolvability(
        self,
        local_op: SyncOperation,
        remote_op: SyncOperation,
        conflict_type: ConflictType
    ) -> Tuple[bool, float]:
        """评估自动解决能力"""
        # 基于冲突类型的基础可解决性
        base_resolvability = {
            ConflictType.UPDATE_UPDATE: (True, 0.7),
            ConflictType.CREATE_CREATE: (True, 0.8),
            ConflictType.UPDATE_DELETE: (False, 0.3),
            ConflictType.DELETE_UPDATE: (False, 0.3),
            ConflictType.SCHEMA_MISMATCH: (False, 0.1),
            ConflictType.PERMISSION_DENIED: (False, 0.0)
        }.get(conflict_type, (False, 0.5))
        
        resolvable, confidence = base_resolvability
        
        # 基于数据相似性调整置信度
        if conflict_type == ConflictType.UPDATE_UPDATE:
            similarity = self._calculate_data_similarity(local_op, remote_op)
            if similarity > 0.8:
                confidence = min(0.9, confidence + 0.2)
            elif similarity < 0.3:
                confidence = max(0.2, confidence - 0.3)
        
        # 检查置信度阈值
        final_resolvable = resolvable and confidence >= self.conflict_thresholds["auto_resolve_confidence"]
        
        return final_resolvable, confidence
    
    def _detect_time_window_conflicts(
        self,
        local_ops: List[SyncOperation],
        remote_ops: List[SyncOperation],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ConflictContext]:
        """检测时间窗口内的冲突"""
        conflicts = []
        time_window = timedelta(seconds=self.conflict_thresholds["time_window_seconds"])
        
        for local_op in local_ops:
            for remote_op in remote_ops:
                # 检查时间窗口
                time_diff = abs((local_op.client_timestamp - remote_op.client_timestamp).total_seconds())
                
                if time_diff <= time_window.total_seconds():
                    # 检查是否为真正的冲突
                    if self._are_operations_conflicting(local_op, remote_op):
                        conflict_type = self._determine_conflict_type(local_op, remote_op)
                        if conflict_type:
                            conflict_context = self._analyze_conflict_context(
                                local_op, remote_op, conflict_type, context
                            )
                            conflicts.append(conflict_context)
        
        return conflicts
    
    def _detect_cross_object_conflicts(
        self,
        local_operations: List[SyncOperation],
        remote_operations: List[SyncOperation],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ConflictContext]:
        """检测跨对象冲突"""
        conflicts = []
        
        # 检测外键约束冲突
        fk_conflicts = self._detect_foreign_key_conflicts(local_operations, remote_operations)
        conflicts.extend(fk_conflicts)
        
        # 检测唯一性约束冲突
        unique_conflicts = self._detect_uniqueness_conflicts(local_operations, remote_operations)
        conflicts.extend(unique_conflicts)
        
        return conflicts
    
    def _detect_foreign_key_conflicts(
        self,
        local_operations: List[SyncOperation],
        remote_operations: List[SyncOperation]
    ) -> List[ConflictContext]:
        """检测外键约束冲突"""
        # 简化实现，实际需要根据具体的数据模型来检测
        return []
    
    def _detect_uniqueness_conflicts(
        self,
        local_operations: List[SyncOperation],
        remote_operations: List[SyncOperation]
    ) -> List[ConflictContext]:
        """检测唯一性约束冲突"""
        conflicts = []
        
        # 检查同一表的创建操作是否有重复的唯一字段
        local_creates = [op for op in local_operations if op.operation_type == SyncOperationType.PUT]
        remote_creates = [op for op in remote_operations if op.operation_type == SyncOperationType.PUT]
        
        for local_op in local_creates:
            for remote_op in remote_creates:
                if (local_op.table_name == remote_op.table_name and
                    local_op.object_id != remote_op.object_id and
                    self._have_duplicate_unique_fields(local_op, remote_op)):
                    
                    conflict_context = ConflictContext(
                        local_operation=local_op,
                        remote_operation=remote_op,
                        conflict_type=ConflictType.CREATE_CREATE,
                        conflict_category=ConflictCategory.DATA_CONFLICT,
                        severity=ConflictSeverity.HIGH,
                        auto_resolvable=False,
                        confidence_score=0.9,
                        metadata={"conflict_reason": "unique_constraint_violation"}
                    )
                    conflicts.append(conflict_context)
        
        return conflicts
    
    def _is_semantic_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> bool:
        """检查是否为语义冲突"""
        # 简化实现：检查是否修改了相关的语义字段
        semantic_fields = {"status", "state", "type", "category", "role"}
        
        local_data = local_op.data or {}
        remote_data = remote_op.data or {}
        
        local_semantic_fields = set(local_data.keys()) & semantic_fields
        remote_semantic_fields = set(remote_data.keys()) & semantic_fields
        
        # 如果都修改了语义字段，认为是语义冲突
        return len(local_semantic_fields) > 0 and len(remote_semantic_fields) > 0
    
    def _is_temporal_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> bool:
        """检查是否为时间冲突"""
        # 简化实现：检查是否修改了时间相关字段
        temporal_fields = {"created_at", "updated_at", "deleted_at", "expires_at", "start_time", "end_time"}
        
        local_data = local_op.data or {}
        remote_data = remote_op.data or {}
        
        local_temporal_fields = set(local_data.keys()) & temporal_fields
        remote_temporal_fields = set(remote_data.keys()) & temporal_fields
        
        return len(local_temporal_fields) > 0 and len(remote_temporal_fields) > 0
    
    def _contains_critical_fields(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> bool:
        """检查是否包含关键字段"""
        critical_fields = {"id", "user_id", "password", "email", "permissions", "balance", "amount"}
        
        local_data = local_op.data or {}
        remote_data = remote_op.data or {}
        
        local_critical = set(local_data.keys()) & critical_fields
        remote_critical = set(remote_data.keys()) & critical_fields
        
        return len(local_critical) > 0 or len(remote_critical) > 0
    
    def _calculate_data_similarity(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> float:
        """计算数据相似性"""
        local_data = local_op.data or {}
        remote_data = remote_op.data or {}
        
        # 简化的相似性计算
        all_keys = set(local_data.keys()) | set(remote_data.keys())
        if not all_keys:
            return 1.0
        
        matching_keys = 0
        for key in all_keys:
            if local_data.get(key) == remote_data.get(key):
                matching_keys += 1
        
        return matching_keys / len(all_keys)
    
    def _are_operations_conflicting(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> bool:
        """检查两个操作是否冲突"""
        # 基本冲突检查
        if local_op.table_name != remote_op.table_name or local_op.object_id != remote_op.object_id:
            return False
        
        # 检查向量时钟
        relation = self.vector_clock_manager.compare_clocks(
            local_op.vector_clock,
            remote_op.vector_clock
        )
        
        return relation == CausalRelation.CONCURRENT
    
    def _have_duplicate_unique_fields(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> bool:
        """检查是否有重复的唯一字段"""
        # 简化实现：假设email和username是唯一字段
        unique_fields = {"email", "username", "phone"}
        
        local_data = local_op.data or {}
        remote_data = remote_op.data or {}
        
        for field in unique_fields:
            if (field in local_data and field in remote_data and
                local_data[field] == remote_data[field] and
                local_data[field] is not None):
                return True
        
        return False
    
    def _generate_conflict_summary(self, conflicts: List[ConflictContext]) -> Dict[str, Any]:
        """生成冲突摘要"""
        if not conflicts:
            return {
                "total_conflicts": 0,
                "auto_resolvable": 0,
                "manual_resolution_required": 0,
                "severity_distribution": {},
                "category_distribution": {},
                "type_distribution": {}
            }
        
        # 统计各种分布
        severity_counts = {}
        category_counts = {}
        type_counts = {}
        auto_resolvable = 0
        
        for conflict in conflicts:
            # 严重程度分布
            severity = conflict.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # 分类分布
            category = conflict.conflict_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # 类型分布
            conflict_type = conflict.conflict_type.value
            type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
            
            # 自动解决统计
            if conflict.auto_resolvable:
                auto_resolvable += 1
        
        return {
            "total_conflicts": len(conflicts),
            "auto_resolvable": auto_resolvable,
            "manual_resolution_required": len(conflicts) - auto_resolvable,
            "severity_distribution": severity_counts,
            "category_distribution": category_counts,
            "type_distribution": type_counts,
            "average_confidence": sum(c.confidence_score for c in conflicts) / len(conflicts)
        }
    
    def _detect_update_update_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测更新-更新冲突"""
        # 实现具体的检测逻辑
        return None
    
    def _detect_update_delete_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测更新-删除冲突"""
        return None
    
    def _detect_delete_update_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测删除-更新冲突"""
        return None
    
    def _detect_create_create_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测创建-创建冲突"""
        return None
    
    def _detect_schema_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测模式冲突"""
        return None
    
    def _detect_permission_conflict(
        self, 
        local_op: SyncOperation, 
        remote_op: SyncOperation
    ) -> Optional[ConflictContext]:
        """检测权限冲突"""
        return None
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        stats = self.detection_stats.copy()
        
        if stats["total_detections"] > 0:
            stats["average_conflicts_per_detection"] = stats["conflicts_found"] / stats["total_detections"]
            stats["average_detection_time_ms"] = stats["detection_time_total_ms"] / stats["total_detections"]
            stats["auto_resolvable_rate"] = stats["auto_resolvable_conflicts"] / stats["conflicts_found"] if stats["conflicts_found"] > 0 else 0
        else:
            stats["average_conflicts_per_detection"] = 0
            stats["average_detection_time_ms"] = 0
            stats["auto_resolvable_rate"] = 0
        
        return stats
    
    def configure_thresholds(self, new_thresholds: Dict[str, Any]):
        """配置检测阈值"""
        self.conflict_thresholds.update(new_thresholds)
    
    def reset_statistics(self):
        """重置统计信息"""
        self.detection_stats = {
            "total_detections": 0,
            "conflicts_found": 0,
            "auto_resolvable_conflicts": 0,
            "detection_time_total_ms": 0.0
        }