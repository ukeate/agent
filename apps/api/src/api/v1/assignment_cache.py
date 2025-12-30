"""
用户分配缓存管理API端点
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from src.services.user_assignment_cache import (
    UserAssignmentCache, 
    CachedAssignment, 
    CacheStrategy
)
from src.api.base_model import ApiBaseModel
from src.core.database import get_db
from src.models.database.experiment import Experiment, ExperimentVariant

from src.core.logging import get_logger
logger = get_logger(__name__)

# 请求模型
class CreateAssignmentRequest(ApiBaseModel):
    user_id: str
    experiment_id: str
    variant_id: str
    assignment_context: Optional[Dict[str, Any]] = None
    ttl: Optional[int] = None

class BatchAssignmentRequest(ApiBaseModel):
    assignments: List[CreateAssignmentRequest]

# 全局缓存实例（在生产环境中应该使用依赖注入）
_cache_instance: Optional[UserAssignmentCache] = None

async def get_cache() -> UserAssignmentCache:
    """获取缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = UserAssignmentCache()
        await _cache_instance.initialize()
    return _cache_instance

@asynccontextmanager
async def lifespan(_: APIRouter) -> AsyncGenerator[None, None]:
    """缓存路由生命周期管理"""
    await get_cache()
    yield
    global _cache_instance
    if _cache_instance:
        await _cache_instance.close()

router = APIRouter(prefix="/assignment-cache", tags=["assignment-cache"], lifespan=lifespan)

@router.get("/assignments/{user_id}/{experiment_id}")
async def get_user_assignment(
    user_id: str,
    experiment_id: str,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """获取用户在特定实验中的分配"""
    try:
        assignment, cache_status = await cache.get_assignment(user_id, experiment_id)
        
        if assignment:
            return {
                "user_id": assignment.user_id,
                "experiment_id": assignment.experiment_id,
                "variant_id": assignment.variant_id,
                "assigned_at": assignment.assigned_at,
                "cache_status": cache_status.value,
                "assignment_context": assignment.assignment_context
            }
        else:
            return {
                "user_id": user_id,
                "experiment_id": experiment_id,
                "variant_id": None,
                "cache_status": cache_status.value,
                "message": "No assignment found"
            }
            
    except Exception as e:
        logger.error(f"Error getting assignment for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get assignment"
        )

@router.post("/assignments", status_code=status.HTTP_201_CREATED)
async def create_assignment(
    request: CreateAssignmentRequest,
    db: AsyncSession = Depends(get_db),
    cache: UserAssignmentCache = Depends(get_cache)
):
    """创建用户分配"""
    try:
        experiment_exists = await db.get(Experiment, request.experiment_id)
        if not experiment_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Experiment not found"
            )
        variant_query = await db.execute(
            select(ExperimentVariant).where(
                and_(
                    ExperimentVariant.experiment_id == request.experiment_id,
                    ExperimentVariant.variant_id == request.variant_id
                )
            )
        )
        if not variant_query.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Experiment variant not found"
            )

        assignment = CachedAssignment(
            user_id=request.user_id,
            experiment_id=request.experiment_id,
            variant_id=request.variant_id,
            assigned_at=utc_now(),
            assignment_context=request.assignment_context or {}
        )
        
        success = await cache.set_assignment(assignment, request.ttl)
        
        if success:
            return {
                "message": "Assignment created successfully",
                "user_id": assignment.user_id,
                "experiment_id": assignment.experiment_id,
                "variant_id": assignment.variant_id,
                "assigned_at": assignment.assigned_at
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create assignment"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating assignment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create assignment"
        )

@router.post("/assignments/batch", status_code=status.HTTP_201_CREATED)
async def create_batch_assignments(
    request: BatchAssignmentRequest,
    background_tasks: BackgroundTasks,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """批量创建用户分配"""
    try:
        results = []
        successful_count = 0
        failed_count = 0
        
        for assignment_req in request.assignments:
            try:
                assignment = CachedAssignment(
                    user_id=assignment_req.user_id,
                    experiment_id=assignment_req.experiment_id,
                    variant_id=assignment_req.variant_id,
                    assigned_at=utc_now(),
                    assignment_context=assignment_req.assignment_context or {}
                )
                
                success = await cache.set_assignment(assignment, assignment_req.ttl)
                
                if success:
                    successful_count += 1
                    results.append({
                        "user_id": assignment.user_id,
                        "experiment_id": assignment.experiment_id,
                        "success": True
                    })
                else:
                    failed_count += 1
                    results.append({
                        "user_id": assignment_req.user_id,
                        "experiment_id": assignment_req.experiment_id,
                        "success": False,
                        "error": "Failed to set assignment"
                    })
                    
            except Exception as e:
                failed_count += 1
                results.append({
                    "user_id": assignment_req.user_id,
                    "experiment_id": assignment_req.experiment_id,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total_requests": len(request.assignments),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch assignment creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create batch assignments"
        )

@router.get("/users/{user_id}/assignments")
async def get_user_all_assignments(
    user_id: str,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """获取用户的所有分配"""
    try:
        assignments = await cache.get_user_assignments(user_id)
        
        return {
            "user_id": user_id,
            "total_assignments": len(assignments),
            "assignments": [
                {
                    "experiment_id": assignment.experiment_id,
                    "variant_id": assignment.variant_id,
                    "assigned_at": assignment.assigned_at,
                    "assignment_context": assignment.assignment_context
                }
                for assignment in assignments
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting all assignments for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user assignments"
        )

@router.post("/assignments/batch-get")
async def batch_get_assignments(
    user_experiment_pairs: List[Dict[str, str]],
    cache: UserAssignmentCache = Depends(get_cache)
):
    """批量获取分配"""
    try:
        # 转换为元组列表
        pairs = [(pair["user_id"], pair["experiment_id"]) for pair in user_experiment_pairs]
        
        assignments = await cache.batch_get_assignments(pairs)
        
        results = []
        for pair, assignment in assignments.items():
            user_id, experiment_id = pair
            if assignment:
                results.append({
                    "user_id": user_id,
                    "experiment_id": experiment_id,
                    "variant_id": assignment.variant_id,
                    "assigned_at": assignment.assigned_at,
                    "found": True
                })
            else:
                results.append({
                    "user_id": user_id,
                    "experiment_id": experiment_id,
                    "variant_id": None,
                    "found": False
                })
        
        return {
            "total_requests": len(pairs),
            "found_count": len([r for r in results if r["found"]]),
            "not_found_count": len([r for r in results if not r["found"]]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch get assignments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch get assignments"
        )

@router.delete("/assignments/{user_id}/{experiment_id}")
async def delete_assignment(
    user_id: str,
    experiment_id: str,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """删除用户分配"""
    try:
        success = await cache.delete_assignment(user_id, experiment_id)
        
        if success:
            return {
                "message": "Assignment deleted successfully",
                "user_id": user_id,
                "experiment_id": experiment_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assignment not found or failed to delete"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting assignment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete assignment"
        )

@router.delete("/users/{user_id}/assignments")
async def clear_user_assignments(
    user_id: str,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """清除用户的所有分配"""
    try:
        deleted_count = await cache.clear_user_assignments(user_id)
        
        return {
            "message": "User assignments cleared successfully",
            "user_id": user_id,
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing user assignments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear user assignments"
        )

@router.get("/metrics")
async def get_cache_metrics(
    cache: UserAssignmentCache = Depends(get_cache)
):
    """获取缓存指标"""
    try:
        metrics = await cache.get_cache_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting cache metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache metrics"
        )

@router.get("/health")
async def cache_health_check(
    cache: UserAssignmentCache = Depends(get_cache)
):
    """缓存健康检查"""
    try:
        health_status = await cache.health_check()
        
        if health_status.get("status") == "healthy":
            return health_status
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=health_status
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.post("/clear")
async def clear_all_cache(
    confirm: bool = Query(False, description="确认清空所有缓存"),
    cache: UserAssignmentCache = Depends(get_cache)
):
    """清空所有缓存"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please set confirm=true to clear all cache"
            )
        
        success = await cache.clear_cache()
        
        if success:
            return {
                "message": "All cache cleared successfully",
                "timestamp": utc_now()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )

@router.get("/info")
async def get_cache_info(
    cache: UserAssignmentCache = Depends(get_cache)
):
    """获取缓存配置信息"""
    try:
        return {
            "cache_strategy": cache.cache_strategy.value,
            "default_ttl_seconds": cache.default_ttl,
            "max_cache_size": cache.max_cache_size,
            "batch_size": cache.batch_size,
            "batch_timeout_seconds": cache.batch_timeout,
            "key_prefix": cache.key_prefix,
            "redis_url": cache.redis_url.replace("redis://", "redis://***").split("@")[-1] if "@" in cache.redis_url else cache.redis_url
        }
        
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache info"
        )

@router.post("/warmup")
async def warmup_cache(
    user_ids: List[str],
    background_tasks: BackgroundTasks,
    cache: UserAssignmentCache = Depends(get_cache)
):
    """预热缓存"""
    try:
        # 启动后台任务进行缓存预热
        background_tasks.add_task(_warmup_cache_task, cache, user_ids)
        
        return {
            "message": "Cache warmup initiated",
            "user_count": len(user_ids),
            "status": "background_task_started"
        }
        
    except Exception as e:
        logger.error(f"Error initiating cache warmup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate cache warmup"
        )

async def _warmup_cache_task(cache: UserAssignmentCache, user_ids: List[str]):
    """缓存预热后台任务"""
    try:
        logger.info(f"Starting cache warmup for {len(user_ids)} users")
        warmed_count = 0
        
        for user_id in user_ids:
            try:
                # 获取用户的所有分配，这会触发数据库查询并缓存结果
                assignments = await cache.get_user_assignments(user_id)
                warmed_count += len(assignments)
                
                # 添加小延迟防止过载
                if warmed_count % 100 == 0:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error warming cache for user {user_id}: {str(e)}")
                continue
        
        logger.info(f"Cache warmup completed: {warmed_count} assignments cached for {len(user_ids)} users")
        
    except Exception as e:
        logger.error(f"Error in cache warmup task: {str(e)}")
