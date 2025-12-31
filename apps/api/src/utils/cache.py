"""
缓存工具函数
提供Redis缓存封装和装饰器
"""

import json
from typing import Any, Optional, Union, Callable, Dict
from datetime import timedelta
from functools import wraps
import redis.asyncio as redis
from src.core.utils.timezone_utils import format_iso_string

from src.core.logging import get_logger
logger = get_logger(__name__)

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis_pool: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        """获取Redis连接"""
        if self._redis_pool is None:
            self._redis_pool = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis_pool
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存值"""
        try:
            r = await self.get_redis()
            
            # 序列化值
            serialized_value = json.dumps(
                value,
                ensure_ascii=False,
                default=format_iso_string,
            )
            
            # 设置过期时间
            if isinstance(expire, timedelta):
                expire_seconds = int(expire.total_seconds())
            else:
                expire_seconds = expire
            
            await r.set(key, serialized_value, ex=expire_seconds)
            logger.debug(f"设置缓存: {key}, expire: {expire_seconds}秒")
            return True
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {str(e)}")
            return False
    
    async def get(
        self, 
        key: str, 
        default: Any = None
    ) -> Any:
        """获取缓存值"""
        try:
            r = await self.get_redis()
            value = await r.get(key)
            
            if value is None:
                logger.debug(f"缓存未命中: {key}")
                return default
            
            # 反序列化值
            return json.loads(value)
        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {str(e)}")
            return default
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            r = await self.get_redis()
            result = await r.delete(key)
            logger.debug(f"删除缓存: {key}, 结果: {bool(result)}")
            return bool(result)
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            r = await self.get_redis()
            result = await r.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {key}: {str(e)}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置缓存过期时间"""
        try:
            r = await self.get_redis()
            result = await r.expire(key, seconds)
            logger.debug(f"设置缓存过期时间: {key}, {seconds}秒")
            return bool(result)
        except Exception as e:
            logger.error(f"设置缓存过期时间失败 {key}: {str(e)}")
            return False
    
    async def keys(self, pattern: str = "*") -> list[str]:
        """获取匹配的键列表"""
        try:
            r = await self.get_redis()
            keys = await r.keys(pattern)
            logger.debug(f"查找键: {pattern}, 找到 {len(keys)} 个")
            return keys
        except Exception as e:
            logger.error(f"查找键失败 {pattern}: {str(e)}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的所有键"""
        try:
            keys = await self.keys(pattern)
            if keys:
                r = await self.get_redis()
                result = await r.delete(*keys)
                logger.info(f"清除缓存模式 {pattern}: {result} 个键")
                return result
            return 0
        except Exception as e:
            logger.error(f"清除缓存模式失败 {pattern}: {str(e)}")
            return 0
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        try:
            r = await self.get_redis()
            result = await r.incr(key, amount)
            logger.debug(f"递增计数器: {key}, 增量: {amount}, 结果: {result}")
            return result
        except Exception as e:
            logger.error(f"递增计数器失败 {key}: {str(e)}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        try:
            r = await self.get_redis()
            result = await r.decr(key, amount)
            logger.debug(f"递减计数器: {key}, 减量: {amount}, 结果: {result}")
            return result
        except Exception as e:
            logger.error(f"递减计数器失败 {key}: {str(e)}")
            return 0
    
    async def sadd(self, key: str, *values) -> int:
        """向集合添加元素"""
        try:
            r = await self.get_redis()
            result = await r.sadd(key, *values)
            logger.debug(f"向集合添加元素: {key}, 添加: {len(values)} 个")
            return result
        except Exception as e:
            logger.error(f"向集合添加元素失败 {key}: {str(e)}")
            return 0
    
    async def srem(self, key: str, *values) -> int:
        """从集合移除元素"""
        try:
            r = await self.get_redis()
            result = await r.srem(key, *values)
            logger.debug(f"从集合移除元素: {key}, 移除: {len(values)} 个")
            return result
        except Exception as e:
            logger.error(f"从集合移除元素失败 {key}: {str(e)}")
            return 0
    
    async def smembers(self, key: str) -> set:
        """获取集合所有成员"""
        try:
            r = await self.get_redis()
            result = await r.smembers(key)
            logger.debug(f"获取集合成员: {key}, 成员数: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"获取集合成员失败 {key}: {str(e)}")
            return set()
    
    async def close(self):
        """关闭连接"""
        if self._redis_pool:
            await self._redis_pool.aclose()

# 全局缓存管理器实例
cache_manager = CacheManager()

def cached(
    expire: Union[int, timedelta] = 300,
    key_prefix: str = ""
):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(
                cache_key
            )
            
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存储到缓存
            await cache_manager.set(
                cache_key,
                result,
                expire=expire,
            )
            
            logger.debug(f"缓存存储: {cache_key}")
            return result
        
        return wrapper
    return decorator

def cache_key_generator(*args, **kwargs) -> str:
    """生成缓存键"""
    import hashlib
    
    def _stable_repr(value: Any) -> str:
        if isinstance(value, dict):
            items = ",".join(
                f"{k}:{_stable_repr(v)}" for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            )
            return f"{{{items}}}"
        if isinstance(value, (list, tuple)):
            items = ",".join(_stable_repr(item) for item in value)
            return f"[{items}]"
        if hasattr(value, "__dict__"):
            return _stable_repr(value.__dict__)
        return str(value)

    key_parts = []
    
    # 添加位置参数
    for arg in args:
        key_parts.append(_stable_repr(arg))
    
    # 添加关键字参数
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{_stable_repr(v)}")
    
    # 生成SHA256哈希
    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int = 60
    ) -> tuple[bool, Dict[str, Any]]:
        """检查是否允许请求"""
        try:
            current_count = await self.cache.incr(f"rate_limit:{key}")
            
            if current_count == 1:
                # 首次请求，设置过期时间
                await self.cache.expire(f"rate_limit:{key}", window)
            
            remaining = max(0, limit - current_count)
            allowed = current_count <= limit
            
            info = {
                "allowed": allowed,
                "limit": limit,
                "remaining": remaining,
                "current": current_count,
                "window": window
            }
            
            logger.debug(f"速率限制检查: {key}, 允许: {allowed}, 剩余: {remaining}")
            return allowed, info
        except Exception as e:
            logger.error(f"速率限制检查失败 {key}: {str(e)}")
            # 出错时允许请求
            return True, {"allowed": True, "error": str(e)}

def rate_limit(limit: int, window: int = 60, key_func: Optional[Callable] = None):
    """速率限制装饰器"""
    def decorator(func: Callable) -> Callable:
        limiter = RateLimiter(cache_manager)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成限制键
            if key_func:
                rate_key = key_func(*args, **kwargs)
            else:
                rate_key = f"{func.__name__}:default"
            
            # 检查速率限制
            allowed, info = await limiter.is_allowed(rate_key, limit, window)
            
            if not allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "请求过于频繁",
                        "rate_limit_info": info
                    }
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
