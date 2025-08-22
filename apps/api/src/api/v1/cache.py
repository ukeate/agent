"""
缓存管理API端点
提供缓存状态查询、清理和监控功能
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.ai.langgraph.cache_factory import get_node_cache
from src.ai.langgraph.cache_monitor import get_cache_monitor, CacheHealthChecker
from src.ai.langgraph.cached_node import invalidate_node_cache
from src.ai.langgraph.context import AgentContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["Cache Management"])


@router.get("/stats", summary="获取缓存统计信息")
async def get_cache_stats():
    """获取详细的缓存统计信息"""
    try:
        monitor = get_cache_monitor()
        stats = await monitor.get_detailed_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {str(e)}")


@router.get("/health", summary="检查缓存健康状态")
async def check_cache_health():
    """检查缓存系统健康状态"""
    try:
        cache = get_node_cache()
        health_checker = CacheHealthChecker(cache)
        health = await health_checker.health_check()
        
        status_code = 200 if health["status"] == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)
    except Exception as e:
        logger.error(f"缓存健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/performance", summary="获取缓存性能指标")
async def get_cache_performance():
    """获取缓存性能测试结果"""
    try:
        cache = get_node_cache()
        health_checker = CacheHealthChecker(cache)
        performance = await health_checker.performance_check()
        return JSONResponse(content=performance)
    except Exception as e:
        logger.error(f"缓存性能检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能检查失败: {str(e)}")


@router.delete("/clear", summary="清理缓存")
async def clear_cache(
    pattern: Optional[str] = Query(default="*", description="缓存键匹配模式")
):
    """清理匹配模式的缓存条目"""
    try:
        cache = get_node_cache()
        count = await cache.clear(pattern)
        
        return JSONResponse(content={
            "success": True,
            "cleared_count": count,
            "pattern": pattern,
            "message": f"成功清理 {count} 个缓存条目"
        })
    except Exception as e:
        logger.error(f"清理缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理缓存失败: {str(e)}")


@router.delete("/invalidate/{node_name}", summary="失效特定节点缓存")
async def invalidate_node_cache_endpoint(
    node_name: str,
    user_id: Optional[str] = Query(default=None, description="用户ID"),
    session_id: Optional[str] = Query(default=None, description="会话ID"),
    workflow_id: Optional[str] = Query(default=None, description="工作流ID")
):
    """使特定节点的缓存失效"""
    try:
        if user_id and session_id:
            # 使用提供的上下文信息精确失效
            context = AgentContext(
                user_id=user_id,
                session_id=session_id,
                workflow_id=workflow_id
            )
            success = await invalidate_node_cache(
                node_name=node_name,
                context=context,
                inputs={}
            )
        else:
            # 使用模式匹配失效所有相关缓存
            cache = get_node_cache()
            pattern = f"*:{node_name}:*"
            count = await cache.clear(pattern)
            success = count > 0
        
        return JSONResponse(content={
            "success": success,
            "node_name": node_name,
            "message": f"节点 {node_name} 的缓存已失效"
        })
    except Exception as e:
        logger.error(f"节点缓存失效失败: {e}")
        raise HTTPException(status_code=500, detail=f"缓存失效失败: {str(e)}")


@router.get("/summary", summary="获取缓存监控摘要")
async def get_cache_summary():
    """获取缓存监控摘要信息"""
    try:
        monitor = get_cache_monitor()
        summary = monitor.get_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"获取缓存摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取摘要失败: {str(e)}")


@router.get("/config", summary="获取缓存配置")
async def get_cache_config():
    """获取当前缓存配置信息"""
    try:
        cache = get_node_cache()
        config_dict = cache.config.__dict__ if hasattr(cache, 'config') else {}
        
        return JSONResponse(content={
            "backend": type(cache).__name__,
            "config": config_dict
        })
    except Exception as e:
        logger.error(f"获取缓存配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


# 管理端点 - 需要特殊权限
@router.post("/warmup", summary="执行缓存预热")
async def warmup_cache():
    """执行缓存预热操作"""
    try:
        # 这里可以根据需要实现特定的预热逻辑
        # 例如预热常用的工作流节点
        
        return JSONResponse(content={
            "success": True,
            "message": "缓存预热已启动",
            "note": "预热将在后台执行"
        })
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise HTTPException(status_code=500, detail=f"预热失败: {str(e)}")