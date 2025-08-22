"""
离线能力API端点

提供离线状态管理、同步控制等功能
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from ...services.offline_service import OfflineService
from ...models.schemas.offline import OfflineMode, NetworkStatus
from ...core.auth import get_current_user


router = APIRouter(prefix="/offline", tags=["offline"])


class OfflineStatusResponse(BaseModel):
    """离线状态响应"""
    mode: str
    network_status: str
    connection_quality: float
    pending_operations: int
    has_conflicts: bool
    sync_in_progress: bool
    last_sync_at: Optional[str] = None


class SyncRequest(BaseModel):
    """同步请求"""
    force: bool = False
    batch_size: int = 100


class ConflictResolutionRequest(BaseModel):
    """冲突解决请求"""
    conflict_id: str
    resolution_strategy: str
    resolved_data: Optional[Dict[str, Any]] = None


@router.get("/status", response_model=OfflineStatusResponse)
async def get_offline_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> OfflineStatusResponse:
    """获取离线状态"""
    try:
        service = OfflineService()
        status = await service.get_offline_status(current_user["id"])
        
        return OfflineStatusResponse(
            mode=status["mode"],
            network_status=status["network_status"],
            connection_quality=status["connection_quality"],
            pending_operations=status["pending_operations"],
            has_conflicts=status["has_conflicts"],
            sync_in_progress=status["sync_in_progress"],
            last_sync_at=status.get("last_sync_at")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取离线状态失败: {str(e)}")


@router.post("/sync")
async def manual_sync(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """手动同步"""
    try:
        service = OfflineService()
        
        if request.force:
            # 强制同步
            result = await service.force_sync(
                current_user["id"], 
                batch_size=request.batch_size
            )
        else:
            # 添加到后台任务
            background_tasks.add_task(
                service.background_sync, 
                current_user["id"],
                request.batch_size
            )
            result = {"message": "同步已启动", "background": True}
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同步失败: {str(e)}")


@router.get("/conflicts")
async def get_conflicts(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """获取冲突列表"""
    try:
        service = OfflineService()
        conflicts = await service.get_unresolved_conflicts(current_user["id"])
        
        return [
            {
                "id": conflict.id,
                "table_name": conflict.table_name,
                "object_id": conflict.object_id,
                "conflict_type": conflict.conflict_type.value,
                "local_data": conflict.local_data,
                "remote_data": conflict.remote_data,
                "created_at": conflict.created_at.isoformat()
            }
            for conflict in conflicts
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取冲突失败: {str(e)}")


@router.post("/resolve")
async def resolve_conflict(
    request: ConflictResolutionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """解决冲突"""
    try:
        service = OfflineService()
        success = await service.resolve_conflict(
            current_user["id"],
            request.conflict_id,
            request.resolution_strategy,
            request.resolved_data
        )
        
        if success:
            return {"message": "冲突已解决", "conflict_id": request.conflict_id}
        else:
            raise HTTPException(status_code=404, detail="冲突不存在或已解决")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解决冲突失败: {str(e)}")


@router.get("/operations")
async def get_operations(
    limit: int = 100,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """获取操作历史"""
    try:
        service = OfflineService()
        operations = await service.get_operation_history(
            current_user["id"], 
            limit=limit, 
            offset=offset
        )
        
        return [
            {
                "id": op.id,
                "operation_type": op.operation_type.value,
                "table_name": op.table_name,
                "object_id": op.object_id,
                "timestamp": op.client_timestamp.isoformat(),
                "is_synced": op.is_synced,
                "retry_count": op.retry_count
            }
            for op in operations
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取操作历史失败: {str(e)}")


@router.get("/statistics")
async def get_statistics(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取离线统计信息"""
    try:
        service = OfflineService()
        stats = await service.get_offline_statistics(current_user["id"])
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post("/mode/{mode}")
async def set_offline_mode(
    mode: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """设置离线模式"""
    try:
        # 验证模式
        if mode not in ["online", "offline", "auto"]:
            raise HTTPException(status_code=400, detail="无效的离线模式")
        
        service = OfflineService()
        await service.set_offline_mode(current_user["id"], mode)
        
        return {"message": f"离线模式已设置为: {mode}", "mode": mode}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置离线模式失败: {str(e)}")


@router.get("/network")
async def get_network_status() -> Dict[str, Any]:
    """获取网络状态"""
    try:
        service = OfflineService()
        network_stats = await service.get_network_statistics()
        
        return network_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取网络状态失败: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_data(
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """清理旧数据"""
    try:
        if days < 1:
            raise HTTPException(status_code=400, detail="天数必须大于0")
        
        service = OfflineService()
        result = await service.cleanup_old_data(current_user["id"], days)
        
        return {
            "message": f"已清理{days}天前的数据",
            "cleaned_operations": result.get("operations", 0),
            "cleaned_conflicts": result.get("conflicts", 0),
            "cleaned_memories": result.get("memories", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理数据失败: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """离线系统健康检查"""
    try:
        service = OfflineService()
        health = await service.health_check()
        
        return health
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")