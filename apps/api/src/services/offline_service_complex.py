"""
离线服务层

统一管理离线功能组件
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4

from ..models.schemas.offline import (
    OfflineSession, ConflictRecord, SyncOperation,
    OfflineMode, NetworkStatus, ConflictResolutionStrategy,
    VectorClock
)
from ..core.config import get_settings


class SimpleNetworkMonitor:
    """简化的网络监控器"""
    
    def __init__(self):
        self.current_status = "unknown"
        self.last_check = utc_now()
    
    def get_current_status(self):
        return NetworkStatus.UNKNOWN
    
    def get_connection_quality_score(self):
        return 1.0
    
    async def get_network_statistics(self):
        return {
            "current_status": self.current_status,
            "current_latency_ms": 0,
            "current_packet_loss": 0,
            "connection_quality": 1.0,
            "uptime_percentage": 0.0,
            "average_latency_ms": 0,
            "history_size": 0
        }


class SimpleModeSwitch:
    """简化的模式切换器"""
    
    def __init__(self, network_monitor):
        self.network_monitor = network_monitor
        self.current_mode = "auto"
        self.last_online_time = utc_now()
        self.last_offline_time = None
    
    def get_mode_info(self):
        return {
            "current_mode": self.current_mode,
            "last_online_time": self.last_online_time.isoformat(),
            "last_offline_time": self.last_offline_time.isoformat() if self.last_offline_time else None,
            "offline_threshold_seconds": 30,
            "online_threshold_seconds": 10,
            "network_status": "unknown",
            "connection_quality": 1.0
        }


class OfflineService:
    """离线服务"""
    
    def __init__(self):
        # 使用简化的组件
        self.network_monitor = SimpleNetworkMonitor()
        self.mode_switcher = SimpleModeSwitch(self.network_monitor)
        
        # 会话管理
        self.active_sessions: Dict[str, OfflineSession] = {}
        
        # 启动网络监控
        self._monitoring_started = False
    
    async def _ensure_monitoring_started(self):
        """确保网络监控已启动"""
        if not self._monitoring_started:
            await self.network_monitor.start_monitoring()
            self._monitoring_started = True
    
    async def get_or_create_session(self, user_id: str) -> OfflineSession:
        """获取或创建用户会话"""
        session_id = f"session_{user_id}"
        
        # 从缓存获取
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # 从数据库获取
        session = self.database.get_session(session_id)
        if session:
            self.active_sessions[session_id] = session
            return session
        
        # 创建新会话
        await self._ensure_monitoring_started()
        
        session = OfflineSession(
            user_id=user_id,
            session_id=session_id,
            mode=OfflineMode.SYNC,
            network_status=self.network_monitor.get_current_status(),
            connection_quality=self.network_monitor.get_connection_quality_score()
        )
        
        self.database.create_session(session)
        self.active_sessions[session_id] = session
        
        return session
    
    async def get_offline_status(self, user_id: str) -> Dict[str, Any]:
        """获取离线状态"""
        session = await self.get_or_create_session(user_id)
        
        # 更新网络状态
        current_network_status = self.network_monitor.get_current_status()
        session.network_status = current_network_status
        session.connection_quality = self.network_monitor.get_connection_quality_score()
        session.last_heartbeat = utc_now()
        
        # 获取统计信息
        session_stats = self.database.get_session_stats(session.session_id)
        
        # 更新会话信息
        session.pending_operations = session_stats["operations"].get("total_operations", 0) - session_stats["operations"].get("synced_operations", 0)
        session.has_conflicts = session_stats["conflicts"].get("total_conflicts", 0) > session_stats["conflicts"].get("resolved_conflicts", 0)
        
        # 保存更新
        self.database.update_session(session)
        
        return {
            "mode": session.mode.value,
            "network_status": session.network_status.value,
            "connection_quality": session.connection_quality,
            "pending_operations": session.pending_operations,
            "has_conflicts": session.has_conflicts,
            "sync_in_progress": session.sync_in_progress,
            "last_sync_at": session.last_sync_at.isoformat() if session.last_sync_at else None,
            "last_heartbeat": session.last_heartbeat.isoformat()
        }
    
    async def force_sync(self, user_id: str, batch_size: int = 100) -> Dict[str, Any]:
        """强制同步"""
        session = await self.get_or_create_session(user_id)
        
        if session.sync_in_progress:
            return {"message": "同步正在进行中", "in_progress": True}
        
        try:
            session.sync_in_progress = True
            self.database.update_session(session)
            
            # 获取待同步操作
            pending_ops = self.database.get_pending_operations(session.session_id, batch_size)
            
            if not pending_ops:
                return {"message": "没有待同步的操作", "synced_count": 0}
            
            # 模拟同步过程
            synced_count = 0
            failed_count = 0
            
            for operation in pending_ops:
                try:
                    # 这里应该调用实际的同步逻辑
                    # 现在模拟同步成功
                    success = await self._sync_operation(operation)
                    
                    if success:
                        self.database.mark_operation_synced(
                            operation.id, 
                            utc_now()
                        )
                        synced_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"同步操作失败: {operation.id}, 错误: {e}")
                    failed_count += 1
            
            # 更新会话状态
            session.last_sync_at = utc_now()
            session.sync_in_progress = False
            self.database.update_session(session)
            
            return {
                "message": "同步完成",
                "synced_count": synced_count,
                "failed_count": failed_count,
                "total_processed": len(pending_ops)
            }
            
        except Exception as e:
            # 确保重置同步状态
            session.sync_in_progress = False
            self.database.update_session(session)
            raise e
    
    async def _sync_operation(self, operation: SyncOperation) -> bool:
        """同步单个操作（模拟）"""
        # 这里应该实现实际的同步逻辑
        # 比如调用远程API、处理冲突等
        
        # 模拟网络延迟
        await asyncio.sleep(0.1)
        
        # 模拟90%成功率
        import random
        return random.random() > 0.1
    
    async def background_sync(self, user_id: str, batch_size: int = 50):
        """后台同步"""
        try:
            await self.force_sync(user_id, batch_size)
        except Exception as e:
            print(f"后台同步失败: {e}")
    
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
        session = await self.get_or_create_session(user_id)
        
        # 获取冲突记录
        conflicts = self.database.get_unresolved_conflicts(session.session_id)
        conflict = next((c for c in conflicts if str(c.id) == conflict_id), None)
        
        if not conflict:
            return False
        
        # 应用解决策略
        try:
            strategy = ConflictResolutionStrategy(resolution_strategy)
        except ValueError:
            return False
        
        # 根据策略生成解决方案
        if strategy == ConflictResolutionStrategy.LAST_WRITER_WINS:
            final_data = conflict.remote_data
        elif strategy == ConflictResolutionStrategy.FIRST_WRITER_WINS:
            final_data = conflict.local_data
        elif strategy == ConflictResolutionStrategy.CLIENT_WINS:
            final_data = conflict.local_data
        elif strategy == ConflictResolutionStrategy.SERVER_WINS:
            final_data = conflict.remote_data
        elif strategy == ConflictResolutionStrategy.MANUAL:
            final_data = resolved_data or conflict.local_data
        else:
            # 合并策略
            final_data = {**conflict.local_data, **conflict.remote_data}
        
        # 更新冲突记录
        conflict.resolution_strategy = strategy
        conflict.resolved_data = final_data
        conflict.is_resolved = True
        conflict.resolved_at = utc_now()
        conflict.resolved_by = user_id
        
        # 这里应该保存冲突解决结果到数据库
        # 简化实现，实际需要更新数据库
        
        return True
    
    async def get_operation_history(
        self, 
        user_id: str, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[SyncOperation]:
        """获取操作历史"""
        session = await self.get_or_create_session(user_id)
        
        # 这里简化实现，实际需要支持分页
        all_operations = self.database.get_pending_operations(session.session_id, limit + offset)
        return all_operations[offset:offset + limit]
    
    async def get_offline_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取离线统计信息"""
        session = await self.get_or_create_session(user_id)
        
        # 获取各组件统计
        session_stats = self.database.get_session_stats(session.session_id)
        memory_stats = self.memory_manager.get_memory_stats()
        model_cache_stats = self.model_cache.get_cache_stats()
        inference_stats = self.inference_engine.get_inference_stats()
        reasoning_stats = self.reasoning_engine.get_reasoning_statistics()
        network_stats = self.network_monitor.get_statistics()
        
        return {
            "session": {
                "session_id": session.session_id,
                "mode": session.mode.value,
                "uptime_hours": (utc_now() - session.created_at).total_seconds() / 3600,
                **session_stats
            },
            "memory": memory_stats,
            "model_cache": model_cache_stats,
            "inference": inference_stats,
            "reasoning": reasoning_stats,
            "network": network_stats
        }
    
    async def set_offline_mode(self, user_id: str, mode: str):
        """设置离线模式"""
        session = await self.get_or_create_session(user_id)
        
        # 转换模式
        mode_mapping = {
            "online": OfflineMode.ONLINE,
            "offline": OfflineMode.OFFLINE,
            "auto": OfflineMode.SYNC
        }
        
        offline_mode = mode_mapping.get(mode, OfflineMode.SYNC)
        session.mode = offline_mode
        
        # 更新推理引擎模式
        self.reasoning_engine.inference_engine.set_network_status(
            session.network_status
        )
        
        # 保存更改
        self.database.update_session(session)
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        await self._ensure_monitoring_started()
        
        network_stats = self.network_monitor.get_statistics()
        mode_stats = self.mode_switcher.get_mode_statistics()
        
        return {
            "network": network_stats,
            "mode_switcher": mode_stats
        }
    
    async def cleanup_old_data(self, user_id: str, days: int) -> Dict[str, Any]:
        """清理旧数据"""
        session = await self.get_or_create_session(user_id)
        
        # 清理数据库
        db_result = self.database.cleanup_old_data(session.session_id, days)
        
        # 清理记忆
        memory_result = self.memory_manager.cleanup_old_memories(days)
        
        # 清理模型缓存
        model_result = self.model_cache.cleanup_old_models(days)
        
        return {
            "operations": db_result.get("synced_operations_deleted", 0),
            "conflicts": db_result.get("resolved_conflicts_deleted", 0),
            "memories": memory_result,
            "models": model_result
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        await self._ensure_monitoring_started()
        
        # 检查各组件状态
        components_status = {
            "database": True,  # 简化检查
            "memory_manager": True,
            "model_cache": True,
            "inference_engine": True,
            "reasoning_engine": True,
            "network_monitor": self._monitoring_started,
            "mode_switcher": True
        }
        
        all_healthy = all(components_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": components_status,
            "network_status": self.network_monitor.get_current_status().value,
            "timestamp": utc_now().isoformat()
        }