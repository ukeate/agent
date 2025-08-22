"""
状态管理器

实现状态快照、状态回放和版本管理功能
"""

import json
import gzip
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

from ..models.schemas.offline import VectorClock, SyncOperation, SyncOperationType, StateSnapshot
from .models import OfflineDatabase


class SnapshotType(str, Enum):
    """快照类型"""
    MANUAL = "manual"
    AUTO_PERIODIC = "auto_periodic"
    AUTO_CHECKPOINT = "auto_checkpoint"
    AUTO_SYNC = "auto_sync"


class CompressionType(str, Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
    JSON_GZIP = "json_gzip"


@dataclass
class StateUpdate:
    """状态更新记录"""
    id: str
    session_id: str
    entity_type: str
    entity_id: str
    operation_type: str
    old_value: Dict[str, Any]
    new_value: Dict[str, Any]
    timestamp: datetime
    vector_clock: VectorClock
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "operation_type": self.operation_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "vector_clock": self.vector_clock.model_dump()
        }


class StateManager:
    """状态管理器"""
    
    def __init__(self, database: OfflineDatabase):
        self.database = database
        self.current_states: Dict[str, Dict[str, Any]] = {}
        self.state_updates: Dict[str, List[StateUpdate]] = {}
        
        # 配置
        self.max_states_in_memory = 1000
        self.auto_snapshot_interval = timedelta(minutes=30)
        self.max_snapshot_history = 50
        
        # 压缩配置
        self.default_compression = CompressionType.JSON_GZIP
        self.compression_threshold = 1024  # 1KB以上的状态进行压缩
        
    def create_snapshot(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
        compression_type: Optional[CompressionType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建状态快照"""
        compression_type = compression_type or self.default_compression
        
        # 计算状态大小
        serialized_state = json.dumps(state_data, ensure_ascii=False)
        state_size = len(serialized_state.encode('utf-8'))
        
        # 决定是否压缩
        should_compress = state_size > self.compression_threshold
        
        # 压缩状态数据
        if should_compress and compression_type != CompressionType.NONE:
            compressed_data, actual_compression = self._compress_state(
                state_data, compression_type
            )
            final_state_data = compressed_data
            is_compressed = True
        else:
            final_state_data = state_data
            is_compressed = False
            actual_compression = CompressionType.NONE
        
        # 创建快照对象
        snapshot = StateSnapshot(
            id=str(uuid4()),
            session_id=session_id,
            snapshot_type=snapshot_type.value,
            version=self._get_next_version(session_id, snapshot_type.value),
            checkpoint=str(uuid4()),
            state_data=final_state_data,
            schema_version="1.0",
            is_compressed=is_compressed,
            compression_type=actual_compression.value if is_compressed else None,
            vector_clock=VectorClock(node_id=session_id),
            size_bytes=state_size,
            metadata=metadata or {}
        )
        
        # 保存快照
        snapshot_id = self.database.create_snapshot(snapshot)
        
        # 清理旧快照
        self._cleanup_old_snapshots(session_id, snapshot_type)
        
        return snapshot_id
    
    def _compress_state(
        self, 
        state_data: Dict[str, Any], 
        compression_type: CompressionType
    ) -> tuple[Any, CompressionType]:
        """压缩状态数据"""
        if compression_type == CompressionType.GZIP:
            json_data = json.dumps(state_data, ensure_ascii=False)
            compressed = gzip.compress(json_data.encode('utf-8'))
            return compressed, CompressionType.GZIP
        
        elif compression_type == CompressionType.PICKLE:
            compressed = pickle.dumps(state_data)
            return compressed, CompressionType.PICKLE
        
        elif compression_type == CompressionType.JSON_GZIP:
            json_data = json.dumps(state_data, ensure_ascii=False)
            compressed = gzip.compress(json_data.encode('utf-8'))
            return compressed, CompressionType.JSON_GZIP
        
        else:
            return state_data, CompressionType.NONE
    
    def _decompress_state(
        self, 
        compressed_data: Any, 
        compression_type: str
    ) -> Dict[str, Any]:
        """解压缩状态数据"""
        if compression_type == CompressionType.GZIP.value:
            json_data = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_data)
        
        elif compression_type == CompressionType.PICKLE.value:
            return pickle.loads(compressed_data)
        
        elif compression_type == CompressionType.JSON_GZIP.value:
            json_data = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_data)
        
        else:
            return compressed_data
    
    def _get_next_version(self, session_id: str, snapshot_type: str) -> int:
        """获取下一个版本号"""
        latest = self.database.get_latest_snapshot(session_id, snapshot_type)
        return (latest.version + 1) if latest else 1
    
    def restore_from_snapshot(
        self, 
        session_id: str, 
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
        version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """从快照恢复状态"""
        if version:
            snapshot = self.database.get_snapshot_by_version(
                session_id, snapshot_type.value, version
            )
        else:
            snapshot = self.database.get_latest_snapshot(session_id, snapshot_type.value)
        
        if not snapshot:
            return None
        
        # 解压缩状态数据
        if snapshot.is_compressed:
            state_data = self._decompress_state(
                snapshot.state_data, 
                snapshot.compression_type
            )
        else:
            state_data = snapshot.state_data
        
        # 更新内存状态
        self.current_states[session_id] = state_data
        
        return state_data
    
    def get_current_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取当前状态"""
        return self.current_states.get(session_id)
    
    def update_state(
        self, 
        session_id: str, 
        entity_type: str,
        entity_id: str,
        updates: Dict[str, Any],
        operation_type: str = "update"
    ) -> str:
        """更新状态并记录变更"""
        # 确保会话状态存在
        if session_id not in self.current_states:
            self.current_states[session_id] = {}
        
        # 构建实体键
        entity_key = f"{entity_type}.{entity_id}"
        
        # 获取旧值
        old_value = self.current_states[session_id].get(entity_key, {}).copy()
        
        # 应用更新
        if entity_key not in self.current_states[session_id]:
            self.current_states[session_id][entity_key] = {}
        
        if operation_type == "delete":
            new_value = {}
            self.current_states[session_id].pop(entity_key, None)
        else:
            new_value = {**self.current_states[session_id][entity_key], **updates}
            self.current_states[session_id][entity_key] = new_value
        
        # 记录状态变更
        update_record = StateUpdate(
            id=str(uuid4()),
            session_id=session_id,
            entity_type=entity_type,
            entity_id=entity_id,
            operation_type=operation_type,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.utcnow(),
            vector_clock=VectorClock(node_id=session_id)
        )
        
        # 添加到更新历史
        if session_id not in self.state_updates:
            self.state_updates[session_id] = []
        self.state_updates[session_id].append(update_record)
        
        # 清理旧的更新记录
        self._cleanup_old_updates(session_id)
        
        return update_record.id
    
    def replay_operations(
        self, 
        session_id: str, 
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """重放操作以重建状态"""
        # 获取操作记录
        operations = self.database.get_pending_operations(session_id, limit=10000)
        
        # 过滤时间范围
        if from_timestamp:
            operations = [op for op in operations if op.client_timestamp >= from_timestamp]
        if to_timestamp:
            operations = [op for op in operations if op.client_timestamp <= to_timestamp]
        
        # 按时间和向量时钟排序
        operations.sort(key=lambda x: (x.client_timestamp, x.vector_clock.clock.get(x.session_id, 0)))
        
        # 重建状态
        replayed_state = {}
        applied_operations = []
        
        for operation in operations:
            try:
                self._apply_operation_to_state(replayed_state, operation)
                applied_operations.append(operation)
            except Exception as e:
                print(f"重放操作失败: {operation.id}, 错误: {e}")
        
        # 更新当前状态
        self.current_states[session_id] = replayed_state
        
        return {
            "replayed_state": replayed_state,
            "applied_operations_count": len(applied_operations),
            "total_operations_count": len(operations),
            "replay_timestamp": datetime.utcnow().isoformat()
        }
    
    def _apply_operation_to_state(self, state: Dict[str, Any], operation: SyncOperation):
        """将操作应用到状态"""
        entity_key = f"{operation.table_name}.{operation.object_id}"
        
        if operation.operation_type == SyncOperationType.PUT:
            state[entity_key] = operation.data or {}
        elif operation.operation_type == SyncOperationType.PATCH:
            if entity_key in state:
                state[entity_key].update(operation.patch_data or {})
            else:
                state[entity_key] = operation.patch_data or {}
        elif operation.operation_type == SyncOperationType.DELETE:
            state.pop(entity_key, None)
    
    def get_state_diff(
        self, 
        session_id: str,
        from_snapshot: Optional[str] = None,
        to_snapshot: Optional[str] = None
    ) -> Dict[str, Any]:
        """计算状态差异"""
        # 获取快照
        if from_snapshot:
            from_snapshot_obj = self.database.get_snapshot(from_snapshot)
            if from_snapshot_obj and from_snapshot_obj.is_compressed:
                from_state = self._decompress_state(
                    from_snapshot_obj.state_data,
                    from_snapshot_obj.compression_type
                )
            else:
                from_state = from_snapshot_obj.state_data if from_snapshot_obj else {}
        else:
            from_state = {}
        
        if to_snapshot:
            to_snapshot_obj = self.database.get_snapshot(to_snapshot)
            if to_snapshot_obj and to_snapshot_obj.is_compressed:
                to_state = self._decompress_state(
                    to_snapshot_obj.state_data,
                    to_snapshot_obj.compression_type
                )
            else:
                to_state = to_snapshot_obj.state_data if to_snapshot_obj else {}
        else:
            to_state = self.current_states.get(session_id, {})
        
        # 计算差异
        added = {}
        modified = {}
        deleted = {}
        
        # 查找新增和修改
        for key, value in to_state.items():
            if key not in from_state:
                added[key] = value
            elif from_state[key] != value:
                modified[key] = {
                    "old": from_state[key],
                    "new": value
                }
        
        # 查找删除
        for key in from_state:
            if key not in to_state:
                deleted[key] = from_state[key]
        
        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "total_changes": len(added) + len(modified) + len(deleted)
        }
    
    def create_auto_snapshot(self, session_id: str) -> Optional[str]:
        """创建自动快照"""
        current_state = self.current_states.get(session_id)
        if not current_state:
            return None
        
        # 检查是否需要创建快照
        latest_snapshot = self.database.get_latest_snapshot(
            session_id, 
            SnapshotType.AUTO_PERIODIC.value
        )
        
        should_create = False
        if not latest_snapshot:
            should_create = True
        else:
            time_since_last = datetime.utcnow() - latest_snapshot.created_at
            should_create = time_since_last >= self.auto_snapshot_interval
        
        if should_create:
            return self.create_snapshot(
                session_id=session_id,
                state_data=current_state,
                snapshot_type=SnapshotType.AUTO_PERIODIC,
                metadata={"auto_created": True}
            )
        
        return None
    
    def _cleanup_old_updates(self, session_id: str, max_updates: int = 1000):
        """清理旧的状态更新记录"""
        if session_id in self.state_updates:
            updates = self.state_updates[session_id]
            if len(updates) > max_updates:
                self.state_updates[session_id] = updates[-max_updates:]
    
    def _cleanup_old_snapshots(self, session_id: str, snapshot_type: SnapshotType):
        """清理旧快照"""
        # 这里应该调用数据库方法清理旧快照
        # 简化实现，实际需要在数据库层面实现
        pass
    
    def get_state_statistics(self, session_id: str) -> Dict[str, Any]:
        """获取状态统计信息"""
        current_state = self.current_states.get(session_id, {})
        state_updates = self.state_updates.get(session_id, [])
        
        # 计算实体数量
        entity_counts = {}
        for key in current_state.keys():
            entity_type = key.split('.')[0]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # 计算更新频率
        recent_updates = [
            u for u in state_updates
            if (datetime.utcnow() - u.timestamp).total_seconds() < 3600  # 最近1小时
        ]
        
        return {
            "total_entities": len(current_state),
            "entity_types": entity_counts,
            "total_updates": len(state_updates),
            "recent_updates": len(recent_updates),
            "memory_usage_entities": len(current_state),
            "last_update": state_updates[-1].timestamp.isoformat() if state_updates else None
        }
    
    def export_state(
        self, 
        session_id: str, 
        format_type: str = "json",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """导出状态数据"""
        current_state = self.current_states.get(session_id, {})
        
        export_data = {
            "state": current_state,
            "export_timestamp": datetime.utcnow().isoformat(),
            "format": format_type
        }
        
        if include_metadata:
            export_data["metadata"] = {
                "session_id": session_id,
                "total_entities": len(current_state),
                "statistics": self.get_state_statistics(session_id)
            }
        
        return export_data
    
    def import_state(
        self, 
        session_id: str, 
        state_data: Dict[str, Any],
        merge_strategy: str = "replace"
    ) -> bool:
        """导入状态数据"""
        try:
            if merge_strategy == "replace":
                self.current_states[session_id] = state_data
            elif merge_strategy == "merge":
                if session_id not in self.current_states:
                    self.current_states[session_id] = {}
                self.current_states[session_id].update(state_data)
            else:
                return False
            
            return True
        except Exception as e:
            print(f"导入状态失败: {e}")
            return False