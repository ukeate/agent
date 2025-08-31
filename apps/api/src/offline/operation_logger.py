"""
操作日志系统

记录所有用户操作，支持：
- 操作序列化
- 日志压缩
- 状态快照
- 状态回放
"""

import json
import gzip
import pickle
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

from ..models.schemas.offline import VectorClock, SyncOperation, SyncOperationType
from .models import OfflineDatabase


class OperationType(str, Enum):
    """操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    SYNC = "sync"
    LOGIN = "login"
    LOGOUT = "logout"


@dataclass
class Operation:
    """操作记录"""
    id: str
    session_id: str
    operation_type: OperationType
    entity_type: str
    entity_id: str
    data: Dict[str, Any]
    timestamp: datetime
    vector_clock: VectorClock
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class OperationLogger:
    """操作日志记录器"""
    
    def __init__(self, database: OfflineDatabase):
        self.database = database
        self.session_operations: Dict[str, List[Operation]] = {}
        
    def log_operation(
        self,
        session_id: str,
        operation_type: OperationType,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        vector_clock: VectorClock,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """记录操作"""
        operation = Operation(
            id=str(uuid4()),
            session_id=session_id,
            operation_type=operation_type,
            entity_type=entity_type,
            entity_id=entity_id,
            data=data,
            timestamp=utc_now(),
            vector_clock=vector_clock,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # 转换为同步操作并存储
        sync_op = self._operation_to_sync_operation(operation)
        self.database.add_operation(sync_op)
        
        # 添加到内存缓存
        if session_id not in self.session_operations:
            self.session_operations[session_id] = []
        self.session_operations[session_id].append(operation)
        
        return operation.id
    
    def _operation_to_sync_operation(self, operation: Operation) -> SyncOperation:
        """转换操作为同步操作"""
        sync_type_mapping = {
            OperationType.CREATE: SyncOperationType.PUT,
            OperationType.UPDATE: SyncOperationType.PATCH,
            OperationType.DELETE: SyncOperationType.DELETE,
            OperationType.READ: SyncOperationType.PUT,  # 读操作记录为PUT
            OperationType.SYNC: SyncOperationType.PUT,
            OperationType.LOGIN: SyncOperationType.PUT,
            OperationType.LOGOUT: SyncOperationType.PUT
        }
        
        return SyncOperation(
            id=operation.id,
            session_id=operation.session_id,
            operation_type=sync_type_mapping.get(operation.operation_type, SyncOperationType.PUT),
            table_name=operation.entity_type,
            object_id=operation.entity_id,
            object_type=operation.entity_type,
            data=operation.data,
            client_timestamp=operation.timestamp,
            vector_clock=operation.vector_clock,
            metadata=operation.metadata
        )
    
    def get_session_operations(self, session_id: str, limit: int = 100) -> List[Operation]:
        """获取会话操作记录"""
        # 先从内存获取
        if session_id in self.session_operations:
            return self.session_operations[session_id][-limit:]
        
        # 从数据库获取
        sync_ops = self.database.get_pending_operations(session_id, limit)
        operations = [self._sync_operation_to_operation(op) for op in sync_ops]
        
        # 缓存到内存
        self.session_operations[session_id] = operations
        
        return operations
    
    def _sync_operation_to_operation(self, sync_op: SyncOperation) -> Operation:
        """转换同步操作为操作"""
        type_mapping = {
            SyncOperationType.PUT: OperationType.CREATE,
            SyncOperationType.PATCH: OperationType.UPDATE,
            SyncOperationType.DELETE: OperationType.DELETE
        }
        
        return Operation(
            id=sync_op.id,
            session_id=sync_op.session_id,
            operation_type=type_mapping.get(sync_op.operation_type, OperationType.UPDATE),
            entity_type=sync_op.table_name,
            entity_id=sync_op.object_id,
            data=sync_op.data or {},
            timestamp=sync_op.client_timestamp,
            vector_clock=sync_op.vector_clock,
            user_id=sync_op.metadata.get('user_id'),
            metadata=sync_op.metadata
        )


# StateManager 已移动到独立的 state_manager.py 文件