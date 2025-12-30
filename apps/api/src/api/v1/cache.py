"""
缓存管理API端点
提供缓存状态查询、清理和监控功能
"""

import asyncio
import fnmatch
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from src.ai.langgraph.cache_factory import get_node_cache
from src.ai.langgraph.cache_monitor import get_cache_monitor, CacheHealthChecker
from src.ai.langgraph.cached_node import invalidate_node_cache
from src.ai.langgraph.context import AgentContext
from src.ai.langgraph.caching import RedisNodeCache, MemoryNodeCache

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/cache", tags=["Cache Management"])

def _decode_redis_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value

def _decode_redis_dict(data: Dict[Any, Any]) -> Dict[str, Any]:
    decoded: Dict[str, Any] = {}
    for key, value in data.items():
        decoded[str(_decode_redis_value(key))] = _decode_redis_value(value)
    return decoded

def _normalize_pattern(cache: Any, pattern: str) -> str:
    prefix = getattr(getattr(cache, "config", None), "key_prefix", "")
    if not prefix:
        return pattern
    if pattern.startswith(prefix):
        return pattern
    if pattern == "*" or not pattern:
        return f"{prefix}:*"
    return f"{prefix}:{pattern}"

def _resolve_cache_key(cache: Any, key: str) -> str:
    prefix = getattr(getattr(cache, "config", None), "key_prefix", "")
    if not prefix:
        return key
    if key.startswith(prefix):
        return key
    return f"{prefix}:{key}"

def _to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

async def _list_cache_keys(cache: Any, pattern: str, limit: int) -> List[str]:
    normalized_pattern = _normalize_pattern(cache, pattern)
    if isinstance(cache, RedisNodeCache):
        redis = await cache._get_redis()
        cursor = 0
        keys: List[str] = []
        scan_count = min(max(10, limit), 1000)
        while True:
            cursor, batch = await redis.scan(
                cursor=cursor,
                match=normalized_pattern,
                count=scan_count
            )
            for item in batch:
                keys.append(_decode_redis_value(item))
                if len(keys) >= limit:
                    break
            if len(keys) >= limit or cursor == 0:
                break
        return keys[:limit]
    if isinstance(cache, MemoryNodeCache):
        keys = [
            key for key in cache._cache.keys()
            if fnmatch.fnmatch(key, normalized_pattern)
        ]
        return keys[:limit]
    return []

def _map_redis_policy_to_strategy(policy: str) -> str:
    policy_lower = policy.lower()
    if "lru" in policy_lower:
        return "LRU"
    if "lfu" in policy_lower:
        return "LFU"
    if "ttl" in policy_lower:
        return "TTL"
    if policy_lower == "noeviction":
        return "NOEVICTION"
    return "LRU"

def _map_strategy_to_redis_policy(strategy: str) -> str:
    if strategy == "LRU":
        return "allkeys-lru"
    if strategy == "LFU":
        return "allkeys-lfu"
    if strategy == "TTL":
        return "volatile-ttl"
    if strategy == "NOEVICTION":
        return "noeviction"
    raise HTTPException(status_code=400, detail="Redis不支持该驱逐策略")

async def _get_cache_strategy(cache: Any) -> Dict[str, Any]:
    if isinstance(cache, RedisNodeCache):
        redis = await cache._get_redis()
        raw_maxmemory = _decode_redis_dict(await redis.config_get("maxmemory"))
        raw_policy = _decode_redis_dict(await redis.config_get("maxmemory-policy"))
        maxmemory = int(raw_maxmemory.get("maxmemory", 0))
        policy = str(raw_policy.get("maxmemory-policy", "noeviction"))
        return {
            "eviction_policy": _map_redis_policy_to_strategy(policy),
            "max_size_mb": round(maxmemory / 1024 / 1024) if maxmemory > 0 else 0,
            "default_ttl_seconds": cache.config.ttl_default,
            "compression_enabled": cache.config.compression,
            "warming_enabled": cache.config.warming_enabled,
            "redis_policy": policy
        }
    if isinstance(cache, MemoryNodeCache):
        return {
            "eviction_policy": "LRU",
            "max_size_mb": cache.config.max_size_mb,
            "default_ttl_seconds": cache.config.ttl_default,
            "compression_enabled": cache.config.compression,
            "warming_enabled": cache.config.warming_enabled
        }
    return {
        "eviction_policy": "LRU",
        "max_size_mb": 0,
        "default_ttl_seconds": 0,
        "compression_enabled": False,
        "warming_enabled": False
    }

async def _apply_strategy_update(cache: Any, payload: Dict[str, Any]) -> None:
    if "default_ttl_seconds" in payload and payload["default_ttl_seconds"] is not None:
        cache.config.ttl_default = int(payload["default_ttl_seconds"])
    if "compression_enabled" in payload and payload["compression_enabled"] is not None:
        cache.config.compression = bool(payload["compression_enabled"])
    if "warming_enabled" in payload and payload["warming_enabled"] is not None:
        cache.config.warming_enabled = bool(payload["warming_enabled"])
    if "max_size_mb" in payload and payload["max_size_mb"] is not None:
        cache.config.max_size_mb = int(payload["max_size_mb"])
    if "eviction_policy" in payload and payload["eviction_policy"] is not None:
        policy = str(payload["eviction_policy"]).upper()
        if isinstance(cache, RedisNodeCache):
            redis = await cache._get_redis()
            redis_policy = _map_strategy_to_redis_policy(policy)
            await redis.config_set("maxmemory-policy", redis_policy)
        elif isinstance(cache, MemoryNodeCache):
            if policy != "LRU":
                raise HTTPException(status_code=400, detail="内存缓存仅支持LRU策略")
        else:
            raise HTTPException(status_code=400, detail="不支持的缓存后端")
    if isinstance(cache, RedisNodeCache) and "max_size_mb" in payload and payload["max_size_mb"] is not None:
        redis = await cache._get_redis()
        max_size_mb = int(payload["max_size_mb"])
        max_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else 0
        await redis.config_set("maxmemory", max_bytes)
    if isinstance(cache, MemoryNodeCache):
        cache._enforce_max_entries()

async def _warm_cache_keys(
    cache: Any,
    keys: List[str],
    ttl_seconds: Optional[int] = None
) -> Dict[str, Any]:
    if not keys:
        return {"success": False, "warmed_count": 0, "failed_keys": []}
    if not cache.config.warming_enabled:
        raise HTTPException(status_code=403, detail="缓存预热已禁用")
    ttl = ttl_seconds if ttl_seconds is not None else cache.config.ttl_default
    warmed = 0
    failed: List[str] = []
    unique_keys = list(dict.fromkeys([k for k in keys if k]))
    if isinstance(cache, RedisNodeCache):
        redis = await cache._get_redis()
        for key in unique_keys:
            resolved = _resolve_cache_key(cache, key)
            exists = await redis.exists(resolved)
            if not exists:
                failed.append(key)
                continue
            if ttl:
                await redis.expire(resolved, ttl)
            warmed += 1
        return {"success": True, "warmed_count": warmed, "failed_keys": failed}
    if isinstance(cache, MemoryNodeCache):
        for key in unique_keys:
            resolved = _resolve_cache_key(cache, key)
            if resolved not in cache._cache:
                failed.append(key)
                continue
            value, _ = cache._cache[resolved]
            await cache.set(resolved, value, ttl=ttl)
            warmed += 1
        return {"success": True, "warmed_count": warmed, "failed_keys": failed}
    return {"success": False, "warmed_count": 0, "failed_keys": unique_keys}

@router.get("/stats", summary="获取缓存统计信息")
async def get_cache_stats():
    """获取详细的缓存统计信息"""
    try:
        cache = get_node_cache()
        cache_stats = cache.stats.to_dict()
        storage_stats = await cache.get_stats() if hasattr(cache, "get_stats") else {}
        used_bytes = storage_stats.get("redis_used_memory") or storage_stats.get("memory_usage_bytes", 0)
        total_items = storage_stats.get("cache_entries", 0)
        
        max_memory_mb = 0
        evictions = 0
        expired_items = 0
        if isinstance(cache, RedisNodeCache):
            redis = await cache._get_redis()
            info_stats = await redis.info("stats")
            info_memory = await redis.info("memory")
            evictions = info_stats.get("evicted_keys", 0)
            expired_items = info_stats.get("expired_keys", 0)
            max_memory = info_memory.get("maxmemory", 0)
            max_memory_mb = round(max_memory / 1024 / 1024) if max_memory > 0 else 0
        elif isinstance(cache, MemoryNodeCache):
            max_memory_mb = cache.config.max_size_mb
        
        hit_rate_percent = round(cache_stats.get("hit_rate", 0) * 100, 2)
        miss_rate_percent = round(cache_stats.get("miss_rate", 0) * 100, 2)
        
        stats = {
            "total_size": used_bytes,
            "total_items": total_items,
            "hit_rate": hit_rate_percent,
            "miss_rate": miss_rate_percent,
            "total_hits": cache_stats.get("hit_count", 0),
            "total_misses": cache_stats.get("miss_count", 0),
            "memory_usage_mb": round(used_bytes / 1024 / 1024, 2),
            "max_memory_mb": max_memory_mb,
            "evictions": evictions,
            "expired_items": expired_items,
            "cache_efficiency": cache_stats.get("hit_rate", 0),
            "nodes": {}
        }
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
        raw_health = await health_checker.health_check()
        cache_stats = cache.stats.to_dict()
        
        hit_rate_percent = round(cache_stats.get("hit_rate", 0) * 100, 2)
        avg_response_time_ms = round(cache_stats.get("avg_get_latency_ms", 0), 2)
        
        memory_usage_percent = 0.0
        if isinstance(cache, RedisNodeCache):
            redis = await cache._get_redis()
            info = await redis.info("memory")
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            if max_memory > 0:
                memory_usage_percent = round(used_memory / max_memory * 100, 2)
        elif isinstance(cache, MemoryNodeCache):
            stats = await cache.get_stats()
            used_mb = stats.get("memory_usage_bytes", 0) / 1024 / 1024
            max_mb = cache.config.max_size_mb
            if max_mb > 0:
                memory_usage_percent = round(used_mb / max_mb * 100, 2)
        
        connectivity_ok = all(
            check.get("status") == "pass"
            for check in raw_health.get("checks", {}).values()
        )
        memory_ok = memory_usage_percent == 0 or memory_usage_percent < 85
        hit_rate_ok = hit_rate_percent == 0 or hit_rate_percent >= 60
        response_ok = avg_response_time_ms == 0 or avg_response_time_ms <= 50
        
        issues = []
        recommendations = []
        if not connectivity_ok:
            issues.append("缓存连接失败")
            recommendations.append("检查缓存服务连接与配置")
        if not memory_ok:
            issues.append("内存使用率过高")
            recommendations.append("考虑增大缓存容量或优化缓存策略")
        if not hit_rate_ok:
            issues.append("缓存命中率偏低")
            recommendations.append("检查缓存键设计或提高热点数据缓存比例")
        if not response_ok:
            issues.append("缓存响应时间偏高")
            recommendations.append("检查缓存后端性能或网络延迟")
        
        status = "healthy"
        if not connectivity_ok:
            status = "unhealthy"
        elif not (memory_ok and hit_rate_ok and response_ok):
            status = "degraded"
        
        health = {
            "status": status,
            "checks": {
                "connectivity": connectivity_ok,
                "memory_usage": memory_ok,
                "hit_rate": hit_rate_ok,
                "response_time": response_ok
            },
            "metrics": {
                "memory_usage_percent": memory_usage_percent,
                "hit_rate_percent": hit_rate_percent,
                "avg_response_time_ms": avg_response_time_ms
            },
            "issues": issues,
            "recommendations": recommendations
        }
        
        status_code = 200 if status == "healthy" else 503
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
        invalidated_count = 0
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
            invalidated_count = 1 if success else 0
        else:
            # 使用模式匹配失效所有相关缓存
            cache = get_node_cache()
            pattern = f"*:{node_name}:*"
            count = await cache.clear(pattern)
            success = count > 0
            invalidated_count = count
        
        return JSONResponse(content={
            "success": success,
            "node_name": node_name,
            "invalidated_count": invalidated_count,
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

@router.get("/strategy", summary="获取缓存策略")
async def get_cache_strategy():
    """获取缓存策略配置"""
    try:
        cache = get_node_cache()
        strategy = await _get_cache_strategy(cache)
        return JSONResponse(content=strategy)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"获取缓存策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存策略失败: {str(e)}")

@router.put("/strategy", summary="更新缓存策略")
async def update_cache_strategy(payload: Dict[str, Any] = Body(...)):
    """更新缓存策略配置"""
    try:
        cache = get_node_cache()
        await _apply_strategy_update(cache, payload)
        strategy = await _get_cache_strategy(cache)
        return JSONResponse(content={"success": True, "strategy": strategy})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"更新缓存策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新缓存策略失败: {str(e)}")

@router.get("/keys", summary="获取缓存键列表")
async def list_cache_keys(
    pattern: Optional[str] = Query(default="*", description="缓存键匹配模式"),
    limit: Optional[int] = Query(default=100, ge=1, le=1000, description="返回数量限制")
):
    """获取缓存键列表"""
    try:
        cache = get_node_cache()
        keys = await _list_cache_keys(cache, pattern, limit)
        return JSONResponse(content={"keys": keys, "count": len(keys)})
    except Exception as e:
        logger.error(f"获取缓存键列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存键列表失败: {str(e)}")

@router.get("/entry/{key}", summary="获取缓存条目")
async def get_cache_entry(key: str):
    """获取单个缓存条目"""
    try:
        cache = get_node_cache()
        resolved_key = _resolve_cache_key(cache, key)
        if isinstance(cache, RedisNodeCache):
            redis = await cache._get_redis()
            data = await redis.get(resolved_key)
            if data is None:
                raise HTTPException(status_code=404, detail="缓存条目不存在")
            ttl = await redis.ttl(resolved_key)
            size = await redis.strlen(resolved_key)
            accessed_at = None
            try:
                idle = await redis.object("idletime", resolved_key)
                if idle is not None:
                    accessed_at = _to_iso(datetime.now(tz=timezone.utc).timestamp() - idle)
            except Exception:
                accessed_at = None
            return JSONResponse(content={
                "key": resolved_key,
                "value": cache._deserialize_value(data),
                "size_bytes": size,
                "ttl_seconds": ttl if ttl >= 0 else None,
                "created_at": None,
                "accessed_at": accessed_at,
                "access_count": None
            })
        if isinstance(cache, MemoryNodeCache):
            if resolved_key not in cache._cache:
                raise HTTPException(status_code=404, detail="缓存条目不存在")
            value, expire_time = cache._cache[resolved_key]
            ttl = None
            if expire_time > 0:
                ttl = max(0, int(expire_time - datetime.now(tz=timezone.utc).timestamp()))
            return JSONResponse(content={
                "key": resolved_key,
                "value": value,
                "size_bytes": cache._entry_sizes.get(resolved_key, 0),
                "ttl_seconds": ttl,
                "created_at": None,
                "accessed_at": _to_iso(cache._access_order.get(resolved_key)),
                "access_count": None
            })
        raise HTTPException(status_code=400, detail="不支持的缓存后端")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"获取缓存条目失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存条目失败: {str(e)}")

@router.post("/entry", summary="设置缓存条目")
async def set_cache_entry(payload: Dict[str, Any] = Body(...)):
    """设置单个缓存条目"""
    try:
        key = payload.get("key")
        value = payload.get("value")
        ttl = payload.get("ttl")
        if not key:
            raise HTTPException(status_code=400, detail="缓存键不能为空")
        cache = get_node_cache()
        resolved_key = _resolve_cache_key(cache, key)
        success = await cache.set(resolved_key, value, ttl=ttl)
        return JSONResponse(content={"success": bool(success), "key": resolved_key})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"设置缓存条目失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置缓存条目失败: {str(e)}")

@router.delete("/entry/{key}", summary="删除缓存条目")
async def delete_cache_entry(key: str):
    """删除单个缓存条目"""
    try:
        cache = get_node_cache()
        resolved_key = _resolve_cache_key(cache, key)
        success = await cache.delete(resolved_key)
        return JSONResponse(content={"success": bool(success), "key": resolved_key})
    except Exception as e:
        logger.error(f"删除缓存条目失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除缓存条目失败: {str(e)}")

@router.post("/mget", summary="批量获取缓存条目")
async def get_multiple_entries(payload: Dict[str, Any] = Body(...)):
    """批量获取缓存条目"""
    try:
        keys = payload.get("keys", [])
        if not isinstance(keys, list):
            raise HTTPException(status_code=400, detail="keys必须是数组")
        cache = get_node_cache()
        resolved = [_resolve_cache_key(cache, key) for key in keys]
        values = await asyncio.gather(*[cache.get(key) for key in resolved])
        entries = {key: value for key, value in zip(keys, values) if value is not None}
        return JSONResponse(content={"entries": entries})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"批量获取缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量获取缓存失败: {str(e)}")

@router.post("/mset", summary="批量设置缓存条目")
async def set_multiple_entries(payload: Dict[str, Any] = Body(...)):
    """批量设置缓存条目"""
    try:
        entries = payload.get("entries", {})
        ttl = payload.get("ttl")
        if not isinstance(entries, dict):
            raise HTTPException(status_code=400, detail="entries必须是对象")
        cache = get_node_cache()
        tasks = []
        for key, value in entries.items():
            resolved = _resolve_cache_key(cache, key)
            tasks.append(cache.set(resolved, value, ttl=ttl))
        results = await asyncio.gather(*tasks)
        success = all(bool(result) for result in results) if results else False
        return JSONResponse(content={"success": success, "count": len(results)})
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"批量设置缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量设置缓存失败: {str(e)}")

@router.post("/warm", summary="执行缓存预热")
async def warm_cache(payload: Dict[str, Any] = Body(...)):
    """执行缓存预热"""
    try:
        keys = payload.get("keys", [])
        ttl = payload.get("ttl_seconds")
        cache = get_node_cache()
        result = await _warm_cache_keys(cache, keys, ttl_seconds=ttl)
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise HTTPException(status_code=500, detail=f"预热失败: {str(e)}")

# 管理端点 - 需要特殊权限
@router.post("/warmup", summary="执行缓存预热")
async def warmup_cache(payload: Optional[Dict[str, Any]] = Body(default=None)):
    """执行缓存预热操作"""
    try:
        payload = payload or {}
        cache = get_node_cache()
        keys = payload.get("keys", [])
        ttl = payload.get("ttl_seconds")
        if not keys:
            keys = await _list_cache_keys(cache, "*", 100)
        result = await _warm_cache_keys(cache, keys, ttl_seconds=ttl)
        return JSONResponse(content={
            "success": result["success"],
            "warmed_count": result["warmed_count"],
            "failed_keys": result["failed_keys"],
            "total": len(keys)
        })
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise HTTPException(status_code=500, detail=f"预热失败: {str(e)}")
