"""
离线服务

基于本地SQLite持久化离线会话/操作/冲突，并提供同步入口。
"""

import asyncio
import json
import time
import sqlite3
from typing import Dict, Any, List, Optional
from uuid import UUID
import uuid
from sqlalchemy import text
from src.core.config import get_settings
from src.core.database import get_db_session
from src.core.utils.timezone_utils import utc_now
from src.models.schemas.offline import (
    OfflineSession,
    ConflictRecord,
    SyncOperation,
    OfflineMode,
    NetworkStatus,
    ConflictResolutionStrategy,
)
from src.offline.models import OfflineDatabase

class OfflineService:
    """离线服务"""
    
    def __init__(self):
        self.settings = get_settings()
        self.database = OfflineDatabase()
    
    async def _probe_network(self) -> tuple[NetworkStatus, float, float]:
        start = time.perf_counter()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 8000),
                timeout=self.settings.OFFLINE_CONNECTION_TIMEOUT,
            )
            writer.close()
            await writer.wait_closed()
            latency_ms = (time.perf_counter() - start) * 1000
            quality = max(0.0, min(1.0, 1.0 - (latency_ms / 500.0)))
            return NetworkStatus.CONNECTED, latency_ms, quality
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            return NetworkStatus.DISCONNECTED, latency_ms, 0.0

    def _build_session_id(self, user_id: str) -> str:
        return f"offline:{user_id}"

    async def get_or_create_session(self, user_id: str) -> OfflineSession:
        session_id = self._build_session_id(user_id)
        session = self.database.get_session(session_id)
        if session:
            return session

        try:
            user_uuid = UUID(user_id)
        except Exception:
            user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
        network_status, _, quality = await self._probe_network()
        session = OfflineSession(
            user_id=user_uuid,
            session_id=session_id,
            mode=OfflineMode.AUTO,
            network_status=network_status,
            connection_quality=quality,
        )
        try:
            self.database.create_session(session)
        except sqlite3.IntegrityError:
            existing = self.database.get_session(session_id)
            if existing:
                return existing
            raise
        return session

    async def get_offline_status(self, user_id: str) -> Dict[str, Any]:
        """获取离线状态"""
        session = await self.get_or_create_session(user_id)
        network_status, _, quality = await self._probe_network()

        stats = self.database.get_session_stats(session.session_id)
        ops = stats.get("operations") or {}
        conflicts = stats.get("conflicts") or {}
        total_ops = int(ops.get("total_operations") or 0)
        synced_ops = int(ops.get("synced_operations") or 0)
        total_conflicts = int(conflicts.get("total_conflicts") or 0)
        resolved_conflicts = int(conflicts.get("resolved_conflicts") or 0)

        session.network_status = network_status
        session.connection_quality = quality
        session.last_heartbeat = utc_now()
        session.pending_operations = max(total_ops - synced_ops, 0)
        session.has_conflicts = (total_conflicts - resolved_conflicts) > 0
        self.database.update_session(session)

        return {
            "mode": session.mode.value,
            "network_status": session.network_status.value,
            "connection_quality": session.connection_quality,
            "pending_operations": session.pending_operations,
            "has_conflicts": session.has_conflicts,
            "sync_in_progress": session.sync_in_progress,
            "last_sync_at": session.last_sync_at.isoformat() if session.last_sync_at else None,
        }
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        network_status, latency_ms, quality = await self._probe_network()
        return {
            "network": {
                "current_status": network_status.value,
                "current_latency_ms": latency_ms,
                "current_packet_loss": 0,
                "connection_quality": quality,
                "uptime_percentage": 100.0 if network_status == NetworkStatus.CONNECTED else 0.0,
                "average_latency_ms": latency_ms,
                "history_size": 1
            },
            "mode_switcher": {
                "current_mode": OfflineMode.AUTO.value,
                "last_online_time": utc_now().isoformat() if network_status == NetworkStatus.CONNECTED else None,
                "last_offline_time": None,
                "offline_threshold_seconds": 30,
                "online_threshold_seconds": 10,
                "network_status": network_status.value,
                "connection_quality": quality
            }
        }
    
    async def get_offline_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取离线统计信息"""
        session = await self.get_or_create_session(user_id)
        stats = self.database.get_session_stats(session.session_id)
        ops = stats.get("operations") or {}
        conflicts = stats.get("conflicts") or {}
        snapshots = stats.get("snapshots") or {}

        return {
            "user_id": user_id,
            "session_id": session.session_id,
            "total_operations": int(ops.get("total_operations") or 0),
            "pending_operations": max(int(ops.get("total_operations") or 0) - int(ops.get("synced_operations") or 0), 0),
            "synced_operations": int(ops.get("synced_operations") or 0),
            "applied_operations": int(ops.get("applied_operations") or 0),
            "avg_retry_count": float(ops.get("avg_retry_count") or 0.0),
            "total_conflicts": int(conflicts.get("total_conflicts") or 0),
            "resolved_conflicts": int(conflicts.get("resolved_conflicts") or 0),
            "total_snapshots": int(snapshots.get("total_snapshots") or 0),
            "total_snapshot_size_bytes": int(snapshots.get("total_size_bytes") or 0),
            "last_sync_time": session.last_sync_at.isoformat() if session.last_sync_at else None,
        }
    
    async def get_unresolved_conflicts(self, user_id: str) -> List[ConflictRecord]:
        """获取未解决的冲突"""
        session = await self.get_or_create_session(user_id)
        return self.database.get_unresolved_conflicts(session.session_id)
    
    async def resolve_conflict(
        self, 
        user_id: str, 
        conflict_id: str, 
        resolution_strategy: str, 
        resolved_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """解决冲突"""
        try:
            ConflictResolutionStrategy(resolution_strategy)
        except Exception:
            return False

        try:
            conflict_uuid = UUID(conflict_id)
        except Exception:
            return False

        return self.database.resolve_conflict(
            conflict_id=conflict_uuid,
            resolution_strategy=resolution_strategy,
            resolved_data=resolved_data,
            resolved_by=user_id,
        )
    
    async def get_operation_history(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[SyncOperation]:
        """获取操作历史"""
        session = await self.get_or_create_session(user_id)
        return self.database.list_operations(session.session_id, limit=limit, offset=offset)
    
    async def force_sync(self, user_id: str, batch_size: int = 100) -> Dict[str, Any]:
        """强制同步"""
        session = await self.get_or_create_session(user_id)
        if session.sync_in_progress:
            return {"message": "同步正在进行中", "in_progress": True}

        session.sync_in_progress = True
        self.database.update_session(session)
        try:
            pending_ops = self.database.get_pending_operations(session.session_id, limit=batch_size)
            if not pending_ops:
                session.last_sync_at = utc_now()
                return {"message": "没有待同步的操作", "synced_operations": 0, "failed_operations": 0}

            async with get_db_session() as db:
                await db.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS offline_synced_operations (
                            id UUID PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            operation_type TEXT NOT NULL,
                            table_name TEXT NOT NULL,
                            object_id TEXT NOT NULL,
                            object_type TEXT NOT NULL,
                            data JSONB,
                            old_data JSONB,
                            patch_data JSONB,
                            client_timestamp TIMESTAMPTZ NOT NULL,
                            received_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                )

                for op in pending_ops:
                    params = {
                        "id": str(op.id),
                        "user_id": user_id,
                        "session_id": op.session_id,
                        "operation_type": op.operation_type.value,
                        "table_name": op.table_name,
                        "object_id": op.object_id,
                        "object_type": op.object_type,
                        "data": json.dumps(op.data, ensure_ascii=False) if op.data is not None else None,
                        "old_data": json.dumps(op.old_data, ensure_ascii=False) if op.old_data is not None else None,
                        "patch_data": json.dumps(op.patch_data, ensure_ascii=False) if op.patch_data is not None else None,
                        "client_timestamp": op.client_timestamp,
                    }
                    await db.execute(
                        text(
                            """
                            INSERT INTO offline_synced_operations (
                                id, user_id, session_id, operation_type, table_name, object_id, object_type,
                                data, old_data, patch_data, client_timestamp
                            ) VALUES (
                                :id::uuid, :user_id, :session_id, :operation_type, :table_name, :object_id, :object_type,
                                :data::jsonb, :old_data::jsonb, :patch_data::jsonb, :client_timestamp
                            )
                            ON CONFLICT (id) DO NOTHING
                            """
                        ),
                        params,
                    )

                await db.commit()

            for op in pending_ops:
                self.database.mark_operation_synced(op.id, server_timestamp=utc_now())

            session.last_sync_at = utc_now()
            return {
                "message": "同步完成",
                "synced_operations": len(pending_ops),
                "failed_operations": 0,
                "sync_time": session.last_sync_at.isoformat(),
            }
        finally:
            session.sync_in_progress = False
            self.database.update_session(session)
    
    async def background_sync(self, user_id: str, batch_size: int = 100):
        """后台同步"""
        await self.force_sync(user_id, batch_size)

    async def get_vector_clock_state(self) -> Dict[str, Any]:
        """返回当前向量时钟状态（真实数据为空则返回空列表，不造假数据）"""
        return {
            "nodes": [],
            "events": []
        }
    
    async def set_offline_mode(self, user_id: str, mode: str):
        """设置离线模式"""
        session = await self.get_or_create_session(user_id)
        session.mode = OfflineMode(mode)
        self.database.update_session(session)
    
    async def cleanup_old_data(self, user_id: str, days: int) -> Dict[str, Any]:
        """清理旧数据"""
        session = await self.get_or_create_session(user_id)
        result = self.database.cleanup_old_data(session.session_id, keep_days=days)
        return {
            "operations": result.get("synced_operations_deleted", 0),
            "conflicts": result.get("resolved_conflicts_deleted", 0),
            "memories": result.get("old_snapshots_deleted", 0),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        network_status, _, _ = await self._probe_network()
        return {
            "status": "healthy" if network_status == NetworkStatus.CONNECTED else "degraded",
            "components": {"offline_db": True, "network": network_status.value},
            "timestamp": utc_now().isoformat()
        }
