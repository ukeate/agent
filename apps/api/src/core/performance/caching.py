"""
API缓存管理系统
"""

import hashlib
import json
import zlib
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import format_iso_string
from typing import Any, Callable, Dict, Optional
from fastapi import Request, Response
from redis.asyncio import Redis
from src.core.config import get_settings
from src.core.redis import get_redis

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class CacheKey:
    """缓存键生成器"""
    
    @staticmethod
    def generate(
        endpoint: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """生成缓存键"""
        key_parts = [
            settings.CACHE_KEY_PREFIX,
            "api",
            endpoint.replace("/", ":"),
            method.lower()
        ]
        
        if user_id:
            key_parts.append(f"user:{user_id}")
        
        if params:
            # 对参数进行排序和哈希
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.sha256(params_str.encode("utf-8")).hexdigest()[:16]
            key_parts.append(f"params:{params_hash}")
        
        return ":".join(key_parts)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.enabled = settings.CACHE_ENABLED
        self.default_ttl = settings.CACHE_TTL_DEFAULT
        self.compression = settings.CACHE_COMPRESSION
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """初始化缓存管理器"""
        if self.enabled:
            try:
                # get_redis() 是同步函数，不需要await
                self.redis = get_redis()
                logger.info("Cache manager initialized")
            except Exception as e:
                logger.error("Failed to initialize cache manager", error=str(e))
                self.enabled = False
    
    def _json_default(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return format_iso_string(value)
        return value

    def _serialize(self, data: Any) -> bytes:
        """序列化数据"""
        serialized = json.dumps(
            data,
            default=self._json_default,
            ensure_ascii=False,
        ).encode()
        
        if self.compression:
            serialized = zlib.compress(serialized)
        
        return serialized
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据"""
        if self.compression:
            data = zlib.decompress(data)
        
        return json.loads(data.decode())
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.enabled or not self.redis:
            return None
        
        try:
            data = await self.redis.get(key)
            if data:
                self.cache_stats["hits"] += 1
                logger.debug("Cache hit", key=key)
                return self._deserialize(data)
            else:
                self.cache_stats["misses"] += 1
                logger.debug("Cache miss", key=key)
                return None
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置缓存"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            serialized = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            await self.redis.setex(key, ttl, serialized)
            self.cache_stats["sets"] += 1
            
            logger.debug("Cache set", key=key, ttl=ttl)
            return True
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            self.cache_stats["deletes"] += 1
            logger.debug("Cache delete", key=key)
            return bool(result)
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """删除匹配模式的缓存"""
        if not self.enabled or not self.redis:
            return 0
        
        try:
            keys = []
            async for key in self.redis.scan_iter(pattern):
                keys.append(key)
            
            if keys:
                result = await self.redis.delete(*keys)
                self.cache_stats["deletes"] += result
                logger.debug("Cache pattern delete", pattern=pattern, count=result)
                return result
            return 0
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error("Cache pattern delete error", pattern=pattern, error=str(e))
            return 0
    
    async def clear(self) -> bool:
        """清空所有缓存"""
        if not self.enabled or not self.redis:
            return False
        
        try:
            pattern = f"{settings.CACHE_KEY_PREFIX}:*"
            count = await self.delete_pattern(pattern)
            logger.info("Cache cleared", count=count)
            return True
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = 0
        if total > 0:
            hit_rate = self.cache_stats["hits"] / total
        
        return {
            **self.cache_stats,
            "total_requests": total,
            "hit_rate": hit_rate,
            "enabled": self.enabled
        }

class CachedResponse:
    """缓存响应装饰器"""
    
    def __init__(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
        condition: Optional[Callable] = None,
        invalidate_on: Optional[List[str]] = None
    ):
        """
        初始化缓存响应装饰器
        
        Args:
            ttl: 缓存过期时间（秒）
            key_func: 自定义缓存键生成函数
            condition: 缓存条件函数
            invalidate_on: 触发缓存失效的事件列表
        """
        self.ttl = ttl
        self.key_func = key_func
        self.condition = condition
        self.invalidate_on = invalidate_on or []
        self.cache_manager = cache_manager
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器实现"""
        async def wrapper(request: Request, *args, **kwargs):
            # 检查缓存条件
            if self.condition and not self.condition(request):
                return await func(request, *args, **kwargs)
            
            # 生成缓存键
            if self.key_func:
                cache_key = self.key_func(request, *args, **kwargs)
            else:
                params = dict(request.query_params)
                user_id = None
                if hasattr(request.state, "user"):
                    user_id = getattr(request.state.user, "id", None)
                
                cache_key = CacheKey.generate(
                    endpoint=request.url.path,
                    method=request.method,
                    params=params,
                    user_id=user_id
                )
            
            # 尝试从缓存获取
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                # 构建响应
                response = Response(
                    content=cached_data.get("content"),
                    status_code=cached_data.get("status_code", 200),
                    headers=cached_data.get("headers", {}),
                    media_type=cached_data.get("media_type", "application/json")
                )
                response.headers["X-Cache"] = "HIT"
                return response
            
            # 执行原函数
            result = await func(request, *args, **kwargs)
            
            # 缓存响应
            if isinstance(result, Response):
                cache_data = {
                    "content": result.body,
                    "status_code": result.status_code,
                    "headers": dict(result.headers),
                    "media_type": result.media_type
                }
                
                # 只缓存成功响应
                if result.status_code < 400:
                    await self.cache_manager.set(cache_key, cache_data, self.ttl)
                
                result.headers["X-Cache"] = "MISS"
            
            return result
        
        return wrapper

# 全局缓存管理器实例
cache_manager = CacheManager()

# 便捷函数
def cache_response(ttl: int = None):
    """缓存响应装饰器便捷函数"""
    return CachedResponse(ttl=ttl)

def cache_invalidate(patterns: List[str]):
    """缓存失效函数"""
    async def invalidate():
        for pattern in patterns:
            await cache_manager.delete_pattern(pattern)
    return invalidate
