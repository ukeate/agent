"""
增量数据计算器

计算数据差异、支持压缩传输和实现智能同步策略
"""

import json
import gzip
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from ..models.schemas.offline import SyncOperation, SyncOperationType, VectorClock

from src.core.logging import get_logger
logger = get_logger(__name__)

class DeltaType(str, Enum):
    """差异类型"""
    FIELD_ADDITION = "field_addition"       # 字段添加
    FIELD_DELETION = "field_deletion"       # 字段删除
    FIELD_MODIFICATION = "field_modification"  # 字段修改
    OBJECT_CREATION = "object_creation"     # 对象创建
    OBJECT_DELETION = "object_deletion"     # 对象删除
    LIST_INSERTION = "list_insertion"       # 列表插入
    LIST_DELETION = "list_deletion"         # 列表删除
    LIST_MODIFICATION = "list_modification" # 列表修改

class CompressionAlgorithm(str, Enum):
    """压缩算法"""
    NONE = "none"
    GZIP = "gzip"
    JSON_DIFF = "json_diff"
    BINARY_DIFF = "binary_diff"

@dataclass
class DeltaOperation:
    """差异操作"""
    operation_type: DeltaType
    path: str  # JSON路径，如 "user.name" 或 "items[0].value"
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ObjectDelta:
    """对象差异"""
    object_id: str
    table_name: str
    operations: List[DeltaOperation]
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.NONE
    original_size: int = 0
    compressed_size: int = 0
    checksum: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_now)

@dataclass
class SyncDelta:
    """同步差异包"""
    session_id: str
    delta_id: str
    object_deltas: List[ObjectDelta]
    vector_clock: VectorClock
    compression_ratio: float = 0.0
    total_operations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class DeltaCalculator:
    """增量数据计算器"""
    
    def __init__(self):
        # 压缩配置
        self.compression_threshold = 1024  # 1KB以上数据进行压缩
        self.default_compression = CompressionAlgorithm.GZIP
        
        # 差异计算配置
        self.max_diff_depth = 10  # 最大递归深度
        self.enable_list_diff = True  # 是否启用列表差异
        
        # 缓存
        self.delta_cache: Dict[str, ObjectDelta] = {}
        self.checksum_cache: Dict[str, str] = {}
    
    def calculate_object_delta(
        self,
        object_id: str,
        table_name: str,
        old_data: Optional[Dict[str, Any]],
        new_data: Optional[Dict[str, Any]]
    ) -> ObjectDelta:
        """计算对象差异"""
        operations = []
        
        if old_data is None and new_data is not None:
            # 对象创建
            operations.append(DeltaOperation(
                operation_type=DeltaType.OBJECT_CREATION,
                path="",
                new_value=new_data
            ))
        elif old_data is not None and new_data is None:
            # 对象删除
            operations.append(DeltaOperation(
                operation_type=DeltaType.OBJECT_DELETION,
                path="",
                old_value=old_data
            ))
        elif old_data is not None and new_data is not None:
            # 对象修改 - 计算字段差异
            operations.extend(self._calculate_field_deltas(old_data, new_data, ""))
        
        # 计算原始大小
        original_size = len(json.dumps({
            "old_data": old_data,
            "new_data": new_data
        }, ensure_ascii=False).encode('utf-8'))
        
        # 创建对象差异
        delta = ObjectDelta(
            object_id=object_id,
            table_name=table_name,
            operations=operations,
            original_size=original_size
        )
        
        # 计算校验和
        delta.checksum = self._calculate_delta_checksum(delta)
        
        return delta
    
    def _calculate_field_deltas(
        self,
        old_obj: Dict[str, Any],
        new_obj: Dict[str, Any],
        path_prefix: str = "",
        depth: int = 0
    ) -> List[DeltaOperation]:
        """计算字段差异"""
        if depth > self.max_diff_depth:
            return []
        
        operations = []
        all_keys = set(old_obj.keys()) | set(new_obj.keys())
        
        for key in all_keys:
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            old_value = old_obj.get(key)
            new_value = new_obj.get(key)
            
            if old_value is None and new_value is not None:
                # 字段添加
                operations.append(DeltaOperation(
                    operation_type=DeltaType.FIELD_ADDITION,
                    path=current_path,
                    new_value=new_value
                ))
            elif old_value is not None and new_value is None:
                # 字段删除
                operations.append(DeltaOperation(
                    operation_type=DeltaType.FIELD_DELETION,
                    path=current_path,
                    old_value=old_value
                ))
            elif old_value != new_value:
                # 字段修改
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    # 递归处理嵌套对象
                    operations.extend(self._calculate_field_deltas(
                        old_value, new_value, current_path, depth + 1
                    ))
                elif isinstance(old_value, list) and isinstance(new_value, list) and self.enable_list_diff:
                    # 处理列表差异
                    operations.extend(self._calculate_list_deltas(
                        old_value, new_value, current_path
                    ))
                else:
                    # 简单值修改
                    operations.append(DeltaOperation(
                        operation_type=DeltaType.FIELD_MODIFICATION,
                        path=current_path,
                        old_value=old_value,
                        new_value=new_value
                    ))
        
        return operations
    
    def _calculate_list_deltas(
        self,
        old_list: List[Any],
        new_list: List[Any],
        path_prefix: str
    ) -> List[DeltaOperation]:
        """计算列表差异"""
        operations = []
        
        # 简化的列表差异算法
        # 更复杂的实现可以使用Myers算法或者其他差异算法
        
        old_len = len(old_list)
        new_len = len(new_list)
        min_len = min(old_len, new_len)
        
        # 处理共同索引的元素
        for i in range(min_len):
            if old_list[i] != new_list[i]:
                operations.append(DeltaOperation(
                    operation_type=DeltaType.LIST_MODIFICATION,
                    path=f"{path_prefix}[{i}]",
                    old_value=old_list[i],
                    new_value=new_list[i]
                ))
        
        # 处理新增元素
        if new_len > old_len:
            for i in range(old_len, new_len):
                operations.append(DeltaOperation(
                    operation_type=DeltaType.LIST_INSERTION,
                    path=f"{path_prefix}[{i}]",
                    new_value=new_list[i]
                ))
        
        # 处理删除元素
        elif old_len > new_len:
            for i in range(new_len, old_len):
                operations.append(DeltaOperation(
                    operation_type=DeltaType.LIST_DELETION,
                    path=f"{path_prefix}[{i}]",
                    old_value=old_list[i]
                ))
        
        return operations
    
    def calculate_sync_delta(
        self,
        session_id: str,
        operations: List[SyncOperation],
        vector_clock: VectorClock
    ) -> SyncDelta:
        """计算同步差异包"""
        object_deltas = []
        total_original_size = 0
        total_compressed_size = 0
        
        # 按对象分组操作
        grouped_operations = self._group_operations_by_object(operations)
        
        for (table_name, object_id), ops in grouped_operations.items():
            # 计算对象的累积状态
            old_state, new_state = self._calculate_object_states(ops)
            
            # 计算对象差异
            object_delta = self.calculate_object_delta(
                object_id, table_name, old_state, new_state
            )
            
            # 压缩对象差异
            if object_delta.original_size > self.compression_threshold:
                compressed_delta = self.compress_object_delta(object_delta)
                object_deltas.append(compressed_delta)
                total_compressed_size += compressed_delta.compressed_size
            else:
                object_deltas.append(object_delta)
                total_compressed_size += object_delta.original_size
            
            total_original_size += object_delta.original_size
        
        # 计算压缩比
        compression_ratio = (
            1.0 - (total_compressed_size / total_original_size)
            if total_original_size > 0 else 0.0
        )
        
        return SyncDelta(
            session_id=session_id,
            delta_id=str(hashlib.md5(f"{session_id}_{utc_now().isoformat()}".encode()).hexdigest()),
            object_deltas=object_deltas,
            vector_clock=vector_clock,
            compression_ratio=compression_ratio,
            total_operations=len(operations)
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
    
    def _calculate_object_states(
        self,
        operations: List[SyncOperation]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """计算对象的初始和最终状态"""
        if not operations:
            return None, None
        
        # 假设第一个操作前的状态（简化）
        first_op = operations[0]
        if first_op.operation_type == SyncOperationType.PUT:
            old_state = None  # 新创建的对象
        else:
            old_state = {}  # 假设空的初始状态
        
        # 应用所有操作得到最终状态
        current_state = old_state.copy() if old_state else {}
        
        for operation in operations:
            if operation.operation_type == SyncOperationType.PUT:
                current_state = operation.data or {}
            elif operation.operation_type == SyncOperationType.PATCH:
                if operation.patch_data:
                    current_state.update(operation.patch_data)
            elif operation.operation_type == SyncOperationType.DELETE:
                current_state = None
                break
        
        return old_state, current_state
    
    def compress_object_delta(
        self,
        delta: ObjectDelta,
        algorithm: Optional[CompressionAlgorithm] = None
    ) -> ObjectDelta:
        """压缩对象差异"""
        algorithm = algorithm or self.default_compression
        
        if algorithm == CompressionAlgorithm.NONE:
            return delta
        
        # 序列化操作
        operations_data = [
            {
                "operation_type": op.operation_type.value,
                "path": op.path,
                "old_value": op.old_value,
                "new_value": op.new_value,
                "metadata": op.metadata
            }
            for op in delta.operations
        ]
        
        # 应用压缩
        if algorithm == CompressionAlgorithm.GZIP:
            original_data = json.dumps(operations_data, ensure_ascii=False)
            compressed_data = gzip.compress(original_data.encode('utf-8'))
            compressed_size = len(compressed_data)
        else:
            # 其他压缩算法的实现
            compressed_size = delta.original_size
        
        # 更新差异对象
        delta.compression_algorithm = algorithm
        delta.compressed_size = compressed_size
        
        return delta
    
    def decompress_object_delta(self, delta: ObjectDelta) -> List[DeltaOperation]:
        """解压缩对象差异"""
        if delta.compression_algorithm == CompressionAlgorithm.NONE:
            return delta.operations
        
        # 这里应该实现实际的解压缩逻辑
        # 简化实现，直接返回操作列表
        return delta.operations
    
    def apply_delta_to_object(
        self,
        original_data: Optional[Dict[str, Any]],
        delta: ObjectDelta
    ) -> Optional[Dict[str, Any]]:
        """将差异应用到对象"""
        if not delta.operations:
            return original_data
        
        # 处理对象级操作
        first_op = delta.operations[0]
        if first_op.operation_type == DeltaType.OBJECT_CREATION:
            return first_op.new_value
        elif first_op.operation_type == DeltaType.OBJECT_DELETION:
            return None
        
        # 处理字段级操作
        result_data = original_data.copy() if original_data else {}
        
        for operation in delta.operations:
            if operation.operation_type == DeltaType.FIELD_ADDITION:
                self._set_nested_value(result_data, operation.path, operation.new_value)
            elif operation.operation_type == DeltaType.FIELD_DELETION:
                self._delete_nested_value(result_data, operation.path)
            elif operation.operation_type == DeltaType.FIELD_MODIFICATION:
                self._set_nested_value(result_data, operation.path, operation.new_value)
            elif operation.operation_type in [DeltaType.LIST_INSERTION, DeltaType.LIST_MODIFICATION]:
                self._set_nested_value(result_data, operation.path, operation.new_value)
            elif operation.operation_type == DeltaType.LIST_DELETION:
                self._delete_nested_value(result_data, operation.path)
        
        return result_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """设置嵌套值"""
        keys = self._parse_path(path)
        current = data
        
        for key in keys[:-1]:
            if isinstance(key, int):
                # 列表索引
                while len(current) <= key:
                    current.append(None)
                if current[key] is None:
                    current[key] = {}
                current = current[key]
            else:
                # 字典键
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        # 设置最终值
        final_key = keys[-1]
        if isinstance(final_key, int):
            while len(current) <= final_key:
                current.append(None)
            current[final_key] = value
        else:
            current[final_key] = value
    
    def _delete_nested_value(self, data: Dict[str, Any], path: str):
        """删除嵌套值"""
        keys = self._parse_path(path)
        current = data
        
        try:
            for key in keys[:-1]:
                if isinstance(key, int):
                    current = current[key]
                else:
                    current = current[key]
            
            # 删除最终值
            final_key = keys[-1]
            if isinstance(final_key, int):
                if 0 <= final_key < len(current):
                    current.pop(final_key)
            else:
                current.pop(final_key, None)
        except (KeyError, IndexError, TypeError):
            # 路径不存在，忽略
            logger.debug("删除路径失败，路径不存在或类型不匹配", exc_info=True)
    
    def _parse_path(self, path: str) -> List[Union[str, int]]:
        """解析路径"""
        if not path:
            return []
        
        keys = []
        parts = path.split('.')
        
        for part in parts:
            # 处理数组索引 like "items[0]"
            if '[' in part and ']' in part:
                field_name = part[:part.index('[')]
                if field_name:
                    keys.append(field_name)
                
                # 提取所有索引
                remaining = part[part.index('['):]
                while '[' in remaining and ']' in remaining:
                    start = remaining.index('[') + 1
                    end = remaining.index(']')
                    index_str = remaining[start:end]
                    try:
                        keys.append(int(index_str))
                    except ValueError:
                        keys.append(index_str)  # 非数字索引
                    remaining = remaining[end + 1:]
            else:
                keys.append(part)
        
        return keys
    
    def _calculate_delta_checksum(self, delta: ObjectDelta) -> str:
        """计算差异校验和"""
        delta_data = {
            "object_id": delta.object_id,
            "table_name": delta.table_name,
            "operations": [
                {
                    "type": op.operation_type.value,
                    "path": op.path,
                    "old_value": op.old_value,
                    "new_value": op.new_value
                }
                for op in delta.operations
            ]
        }
        
        serialized = json.dumps(delta_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    def validate_delta_integrity(self, delta: ObjectDelta) -> bool:
        """验证差异完整性"""
        calculated_checksum = self._calculate_delta_checksum(delta)
        return calculated_checksum == delta.checksum
    
    def estimate_sync_size(self, operations: List[SyncOperation]) -> Dict[str, int]:
        """估算同步数据大小"""
        total_size = 0
        operation_sizes = []
        
        for operation in operations:
            # 计算操作大小
            op_data = {
                "id": operation.id,
                "operation_type": operation.operation_type.value,
                "table_name": operation.table_name,
                "object_id": operation.object_id,
                "data": operation.data,
                "patch_data": operation.patch_data
            }
            
            op_size = len(json.dumps(op_data, ensure_ascii=False).encode('utf-8'))
            operation_sizes.append(op_size)
            total_size += op_size
        
        # 估算压缩后大小（假设30%压缩率）
        estimated_compressed_size = int(total_size * 0.7)
        
        return {
            "total_operations": len(operations),
            "total_size_bytes": total_size,
            "estimated_compressed_size": estimated_compressed_size,
            "average_operation_size": total_size // len(operations) if operations else 0,
            "max_operation_size": max(operation_sizes) if operation_sizes else 0,
            "min_operation_size": min(operation_sizes) if operation_sizes else 0
        }
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        if not self.delta_cache:
            return {
                "total_deltas": 0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "average_compression_ratio": 0.0
            }
        
        total_original = sum(delta.original_size for delta in self.delta_cache.values())
        total_compressed = sum(delta.compressed_size for delta in self.delta_cache.values())
        
        avg_compression_ratio = (
            1.0 - (total_compressed / total_original)
            if total_original > 0 else 0.0
        )
        
        return {
            "total_deltas": len(self.delta_cache),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "average_compression_ratio": avg_compression_ratio,
            "compression_algorithms_used": list(set(
                delta.compression_algorithm.value for delta in self.delta_cache.values()
            ))
        }
