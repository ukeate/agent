"""
离线数据存储实现

基于SQLite实现离线数据持久化，支持：
- 操作日志记录
- 状态快照管理
- 向量时钟存储
- 事务一致性
"""

import sqlite3
import json
import gzip
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager
from uuid import UUID

from ..models.schemas.offline import (
    OfflineSession, SyncOperation, ConflictRecord, 
    SyncBatch, StateSnapshot, VectorClock,
    SyncOperationType, ConflictType
)
from ..core.config import get_settings


class OfflineDatabase:
    """离线数据库管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or Path(settings.OFFLINE_STORAGE_PATH) / "offline.db"
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            # 离线会话表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offline_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    last_sync_at TEXT,
                    last_heartbeat TEXT NOT NULL,
                    network_status TEXT NOT NULL,
                    connection_quality REAL NOT NULL,
                    bandwidth_kbps REAL,
                    pending_operations INTEGER DEFAULT 0,
                    has_conflicts BOOLEAN DEFAULT FALSE,
                    sync_in_progress BOOLEAN DEFAULT FALSE,
                    vector_clock TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 同步操作表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_operations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    data TEXT,
                    old_data TEXT,
                    patch_data TEXT,
                    client_timestamp TEXT NOT NULL,
                    server_timestamp TEXT,
                    vector_clock TEXT NOT NULL,
                    is_applied BOOLEAN DEFAULT FALSE,
                    is_synced BOOLEAN DEFAULT FALSE,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    checksum TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES offline_sessions(session_id)
                )
            """)
            
            # 冲突记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conflict_records (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    local_operation_id TEXT NOT NULL,
                    remote_operation_id TEXT NOT NULL,
                    local_data TEXT NOT NULL,
                    remote_data TEXT NOT NULL,
                    base_data TEXT,
                    local_vector_clock TEXT NOT NULL,
                    remote_vector_clock TEXT NOT NULL,
                    resolution_strategy TEXT,
                    resolved_data TEXT,
                    is_resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES offline_sessions(session_id)
                )
            """)
            
            # 同步批次表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_batches (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    operation_ids TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    conflict_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES offline_sessions(session_id)
                )
            """)
            
            # 状态快照表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    checkpoint TEXT NOT NULL,
                    state_data BLOB NOT NULL,
                    schema_version TEXT NOT NULL,
                    is_compressed BOOLEAN DEFAULT FALSE,
                    compression_ratio REAL,
                    vector_clock TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES offline_sessions(session_id)
                )
            """)
            
            # 创建索引
            self._create_indexes(conn)
    
    def _create_indexes(self, conn: sqlite3.Connection):
        """创建数据库索引"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON offline_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON offline_sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_operations_session_id ON sync_operations(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_operations_object ON sync_operations(table_name, object_id)",
            "CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON sync_operations(client_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_operations_sync_status ON sync_operations(is_synced)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_session_id ON conflict_records(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_object ON conflict_records(table_name, object_id)",
            "CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflict_records(is_resolved)",
            "CREATE INDEX IF NOT EXISTS idx_batches_session_id ON sync_batches(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_batches_status ON sync_batches(status)",
            "CREATE INDEX IF NOT EXISTS idx_snapshots_session_id ON state_snapshots(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_snapshots_type_version ON state_snapshots(snapshot_type, version)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    # 离线会话管理
    
    def create_session(self, session: OfflineSession) -> str:
        """创建离线会话"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO offline_sessions (
                    id, user_id, session_id, mode, started_at, last_sync_at,
                    last_heartbeat, network_status, connection_quality, bandwidth_kbps,
                    pending_operations, has_conflicts, sync_in_progress,
                    vector_clock, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(session.id), str(session.user_id), session.session_id,
                session.mode.value, session.started_at.isoformat(),
                session.last_sync_at.isoformat() if session.last_sync_at else None,
                session.last_heartbeat.isoformat(), session.network_status.value,
                session.connection_quality, session.bandwidth_kbps,
                session.pending_operations, session.has_conflicts,
                session.sync_in_progress, json.dumps(session.vector_clock.dict()),
                json.dumps(session.metadata), session.created_at.isoformat(),
                session.updated_at.isoformat()
            ))
        return str(session.id)
    
    def get_session(self, session_id: str) -> Optional[OfflineSession]:
        """获取离线会话"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM offline_sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            
            if not row:
                return None
            
            vector_clock_data = json.loads(row['vector_clock'])
            return OfflineSession(
                id=UUID(row['id']),
                user_id=UUID(row['user_id']),
                session_id=row['session_id'],
                mode=row['mode'],
                started_at=datetime.fromisoformat(row['started_at']),
                last_sync_at=datetime.fromisoformat(row['last_sync_at']) if row['last_sync_at'] else None,
                last_heartbeat=datetime.fromisoformat(row['last_heartbeat']),
                network_status=row['network_status'],
                connection_quality=row['connection_quality'],
                bandwidth_kbps=row['bandwidth_kbps'],
                pending_operations=row['pending_operations'],
                has_conflicts=row['has_conflicts'],
                sync_in_progress=row['sync_in_progress'],
                vector_clock=VectorClock(**vector_clock_data),
                metadata=json.loads(row['metadata']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
    
    def update_session(self, session: OfflineSession) -> bool:
        """更新离线会话"""
        session.updated_at = datetime.utcnow()
        with self.get_connection() as conn:
            cursor = conn.execute("""
                UPDATE offline_sessions SET
                    mode = ?, last_sync_at = ?, last_heartbeat = ?,
                    network_status = ?, connection_quality = ?, bandwidth_kbps = ?,
                    pending_operations = ?, has_conflicts = ?, sync_in_progress = ?,
                    vector_clock = ?, metadata = ?, updated_at = ?
                WHERE session_id = ?
            """, (
                session.mode.value,
                session.last_sync_at.isoformat() if session.last_sync_at else None,
                session.last_heartbeat.isoformat(), session.network_status.value,
                session.connection_quality, session.bandwidth_kbps,
                session.pending_operations, session.has_conflicts,
                session.sync_in_progress, json.dumps(session.vector_clock.dict()),
                json.dumps(session.metadata), session.updated_at.isoformat(),
                session.session_id
            ))
            return cursor.rowcount > 0
    
    # 同步操作管理
    
    def add_operation(self, operation: SyncOperation) -> str:
        """添加同步操作"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO sync_operations (
                    id, session_id, operation_type, table_name, object_id, object_type,
                    data, old_data, patch_data, client_timestamp, server_timestamp,
                    vector_clock, is_applied, is_synced, retry_count, error_message,
                    checksum, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(operation.id), operation.session_id, operation.operation_type.value,
                operation.table_name, operation.object_id, operation.object_type,
                json.dumps(operation.data) if operation.data else None,
                json.dumps(operation.old_data) if operation.old_data else None,
                json.dumps(operation.patch_data) if operation.patch_data else None,
                operation.client_timestamp.isoformat(),
                operation.server_timestamp.isoformat() if operation.server_timestamp else None,
                json.dumps(operation.vector_clock.dict()), operation.is_applied,
                operation.is_synced, operation.retry_count, operation.error_message,
                operation.checksum, json.dumps(operation.metadata),
                operation.created_at.isoformat(), operation.updated_at.isoformat()
            ))
        return str(operation.id)
    
    def get_pending_operations(self, session_id: str, limit: int = 100) -> List[SyncOperation]:
        """获取待同步操作"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM sync_operations 
                WHERE session_id = ? AND is_synced = FALSE
                ORDER BY client_timestamp ASC
                LIMIT ?
            """, (session_id, limit)).fetchall()
            
            operations = []
            for row in rows:
                vector_clock_data = json.loads(row['vector_clock'])
                operation = SyncOperation(
                    id=UUID(row['id']),
                    session_id=row['session_id'],
                    operation_type=SyncOperationType(row['operation_type']),
                    table_name=row['table_name'],
                    object_id=row['object_id'],
                    object_type=row['object_type'],
                    data=json.loads(row['data']) if row['data'] else None,
                    old_data=json.loads(row['old_data']) if row['old_data'] else None,
                    patch_data=json.loads(row['patch_data']) if row['patch_data'] else None,
                    client_timestamp=datetime.fromisoformat(row['client_timestamp']),
                    server_timestamp=datetime.fromisoformat(row['server_timestamp']) if row['server_timestamp'] else None,
                    vector_clock=VectorClock(**vector_clock_data),
                    is_applied=row['is_applied'],
                    is_synced=row['is_synced'],
                    retry_count=row['retry_count'],
                    error_message=row['error_message'],
                    checksum=row['checksum'],
                    metadata=json.loads(row['metadata']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                operations.append(operation)
            
            return operations
    
    def mark_operation_synced(self, operation_id: UUID, server_timestamp: Optional[datetime] = None) -> bool:
        """标记操作已同步"""
        now = datetime.utcnow()
        with self.get_connection() as conn:
            cursor = conn.execute("""
                UPDATE sync_operations SET
                    is_synced = TRUE,
                    server_timestamp = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                server_timestamp.isoformat() if server_timestamp else now.isoformat(),
                now.isoformat(),
                str(operation_id)
            ))
            return cursor.rowcount > 0
    
    # 冲突记录管理
    
    def add_conflict(self, conflict: ConflictRecord) -> str:
        """添加冲突记录"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO conflict_records (
                    id, session_id, conflict_type, table_name, object_id,
                    local_operation_id, remote_operation_id, local_data, remote_data,
                    base_data, local_vector_clock, remote_vector_clock,
                    resolution_strategy, resolved_data, is_resolved,
                    resolved_at, resolved_by, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(conflict.id), conflict.session_id, conflict.conflict_type.value,
                conflict.table_name, conflict.object_id, str(conflict.local_operation_id),
                str(conflict.remote_operation_id), json.dumps(conflict.local_data),
                json.dumps(conflict.remote_data),
                json.dumps(conflict.base_data) if conflict.base_data else None,
                json.dumps(conflict.local_vector_clock.dict()),
                json.dumps(conflict.remote_vector_clock.dict()),
                conflict.resolution_strategy.value if conflict.resolution_strategy else None,
                json.dumps(conflict.resolved_data) if conflict.resolved_data else None,
                conflict.is_resolved,
                conflict.resolved_at.isoformat() if conflict.resolved_at else None,
                conflict.resolved_by, json.dumps(conflict.metadata),
                conflict.created_at.isoformat(), conflict.updated_at.isoformat()
            ))
        return str(conflict.id)
    
    def get_unresolved_conflicts(self, session_id: str) -> List[ConflictRecord]:
        """获取未解决的冲突"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM conflict_records 
                WHERE session_id = ? AND is_resolved = FALSE
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()
            
            conflicts = []
            for row in rows:
                local_vector_clock_data = json.loads(row['local_vector_clock'])
                remote_vector_clock_data = json.loads(row['remote_vector_clock'])
                conflict = ConflictRecord(
                    id=UUID(row['id']),
                    session_id=row['session_id'],
                    conflict_type=ConflictType(row['conflict_type']),
                    table_name=row['table_name'],
                    object_id=row['object_id'],
                    local_operation_id=UUID(row['local_operation_id']),
                    remote_operation_id=UUID(row['remote_operation_id']),
                    local_data=json.loads(row['local_data']),
                    remote_data=json.loads(row['remote_data']),
                    base_data=json.loads(row['base_data']) if row['base_data'] else None,
                    local_vector_clock=VectorClock(**local_vector_clock_data),
                    remote_vector_clock=VectorClock(**remote_vector_clock_data),
                    resolution_strategy=row['resolution_strategy'],
                    resolved_data=json.loads(row['resolved_data']) if row['resolved_data'] else None,
                    is_resolved=row['is_resolved'],
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                    resolved_by=row['resolved_by'],
                    metadata=json.loads(row['metadata']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                conflicts.append(conflict)
            
            return conflicts
    
    # 状态快照管理
    
    def create_snapshot(self, snapshot: StateSnapshot) -> str:
        """创建状态快照"""
        # 压缩状态数据
        state_data_json = json.dumps(snapshot.state_data)
        
        if snapshot.is_compressed:
            compressed_data = gzip.compress(state_data_json.encode('utf-8'))
            state_data_blob = compressed_data
            compression_ratio = len(compressed_data) / len(state_data_json.encode('utf-8'))
        else:
            state_data_blob = state_data_json.encode('utf-8')
            compression_ratio = None
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO state_snapshots (
                    id, session_id, snapshot_type, version, checkpoint,
                    state_data, schema_version, is_compressed, compression_ratio,
                    vector_clock, size_bytes, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(snapshot.id), snapshot.session_id, snapshot.snapshot_type,
                snapshot.version, snapshot.checkpoint, state_data_blob,
                snapshot.schema_version, snapshot.is_compressed, compression_ratio,
                json.dumps(snapshot.vector_clock.dict()), snapshot.size_bytes,
                json.dumps(snapshot.metadata), snapshot.created_at.isoformat()
            ))
        
        return str(snapshot.id)
    
    def get_latest_snapshot(self, session_id: str, snapshot_type: str) -> Optional[StateSnapshot]:
        """获取最新状态快照"""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM state_snapshots
                WHERE session_id = ? AND snapshot_type = ?
                ORDER BY version DESC
                LIMIT 1
            """, (session_id, snapshot_type)).fetchone()
            
            if not row:
                return None
            
            # 解压状态数据
            if row['is_compressed']:
                state_data_json = gzip.decompress(row['state_data']).decode('utf-8')
            else:
                state_data_json = row['state_data'].decode('utf-8')
            
            state_data = json.loads(state_data_json)
            vector_clock_data = json.loads(row['vector_clock'])
            
            return StateSnapshot(
                id=UUID(row['id']),
                session_id=row['session_id'],
                snapshot_type=row['snapshot_type'],
                version=row['version'],
                checkpoint=row['checkpoint'],
                state_data=state_data,
                schema_version=row['schema_version'],
                is_compressed=row['is_compressed'],
                compression_ratio=row['compression_ratio'],
                vector_clock=VectorClock(**vector_clock_data),
                size_bytes=row['size_bytes'],
                metadata=json.loads(row['metadata']),
                created_at=datetime.fromisoformat(row['created_at'])
            )
    
    # 数据统计和清理
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计信息"""
        with self.get_connection() as conn:
            # 操作统计
            ops_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN is_synced THEN 1 ELSE 0 END) as synced_operations,
                    SUM(CASE WHEN is_applied THEN 1 ELSE 0 END) as applied_operations,
                    AVG(retry_count) as avg_retry_count
                FROM sync_operations WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            # 冲突统计
            conflicts_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_conflicts,
                    SUM(CASE WHEN is_resolved THEN 1 ELSE 0 END) as resolved_conflicts
                FROM conflict_records WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            # 快照统计
            snapshots_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_snapshots,
                    SUM(size_bytes) as total_size_bytes,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM state_snapshots WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            return {
                'operations': dict(ops_stats) if ops_stats else {},
                'conflicts': dict(conflicts_stats) if conflicts_stats else {},
                'snapshots': dict(snapshots_stats) if snapshots_stats else {}
            }
    
    def cleanup_old_data(self, session_id: str, keep_days: int = 7) -> Dict[str, int]:
        """清理旧数据"""
        cutoff_date = (datetime.utcnow() - datetime.timedelta(days=keep_days)).isoformat()
        
        with self.get_connection() as conn:
            # 清理已同步的操作
            synced_ops_deleted = conn.execute("""
                DELETE FROM sync_operations 
                WHERE session_id = ? AND is_synced = TRUE AND created_at < ?
            """, (session_id, cutoff_date)).rowcount
            
            # 清理已解决的冲突
            resolved_conflicts_deleted = conn.execute("""
                DELETE FROM conflict_records 
                WHERE session_id = ? AND is_resolved = TRUE AND created_at < ?
            """, (session_id, cutoff_date)).rowcount
            
            # 保留最新的快照，清理旧版本
            old_snapshots_deleted = conn.execute("""
                DELETE FROM state_snapshots 
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM state_snapshots 
                    WHERE session_id = ? 
                    GROUP BY snapshot_type 
                    HAVING version = MAX(version)
                ) AND created_at < ?
            """, (session_id, session_id, cutoff_date)).rowcount
            
            return {
                'synced_operations_deleted': synced_ops_deleted,
                'resolved_conflicts_deleted': resolved_conflicts_deleted,
                'old_snapshots_deleted': old_snapshots_deleted
            }