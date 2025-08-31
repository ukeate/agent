"""
LangGraph Node级缓存实现
基于LangGraph v0.6 Node Cache API，集成Redis后端存储
"""

import hashlib
import json
import pickle
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Union
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory

from .context import AgentContext

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True
    backend: str = "redis"  # redis, memory, hybrid
    ttl_default: int = 3600  # 1小时
    max_entries: int = 10000
    key_prefix: str = "langgraph:cache"
    redis_url: str = "redis://localhost:6379/1"
    compression: bool = True
    monitoring: bool = True
    # LRU配置
    lru_enabled: bool = True
    # 失效策略
    cleanup_interval: int = 300  # 5分钟清理一次
    # 序列化配置
    serialize_method: str = "pickle"  # json, pickle


@dataclass
class CacheStats:
    """缓存统计信息"""
    hit_count: int = 0
    miss_count: int = 0
    set_count: int = 0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)
    # 延迟统计
    get_latency_total: float = 0.0
    set_latency_total: float = 0.0
    get_operations: int = 0
    set_operations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        return 1.0 - self.hit_rate
    
    @property
    def avg_get_latency_ms(self) -> float:
        """平均获取延迟（毫秒）"""
        return (self.get_latency_total * 1000 / self.get_operations) if self.get_operations > 0 else 0.0
    
    @property
    def avg_set_latency_ms(self) -> float:
        """平均设置延迟（毫秒）"""
        return (self.set_latency_total * 1000 / self.set_operations) if self.set_operations > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "set_count": self.set_count,
            "error_count": self.error_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "avg_get_latency_ms": self.avg_get_latency_ms,
            "avg_set_latency_ms": self.avg_set_latency_ms,
            "uptime_seconds": time.time() - self.start_time
        }


class NodeCache(ABC):
    """Node缓存抽象基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """从缓存获取值"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass
    
    @abstractmethod
    async def clear(self, pattern: str = "*") -> int:
        """清理缓存"""
        pass
    
    def generate_cache_key(
        self, 
        node_name: str, 
        context: AgentContext, 
        inputs: Dict[str, Any]
    ) -> str:
        """生成缓存键"""
        try:
            # 基础键组件
            base_parts = [
                self.config.key_prefix,
                node_name,
                f"user:{context.user_id}",
                f"session:{context.session_id}"
            ]
            
            # 可选的上下文组件
            if context.workflow_id:
                base_parts.append(f"workflow:{context.workflow_id}")
            if context.agent_id:
                base_parts.append(f"agent:{context.agent_id}")
            
            # 输入参数哈希 - 使用SHA256提供更好的安全性
            inputs_str = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
            inputs_hash = hashlib.sha256(inputs_str.encode()).hexdigest()[:16]
            base_parts.append(f"inputs:{inputs_hash}")
            
            # 上下文相关状态哈希（排除动态字段）
            context_data = {
                "max_iterations": context.max_iterations,
                "timeout_seconds": context.timeout_seconds,
                "enable_checkpoints": context.enable_checkpoints,
                "metadata": context.metadata
            }
            context_str = json.dumps(context_data, sort_keys=True, ensure_ascii=False)
            context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:8]
            base_parts.append(f"ctx:{context_hash}")
            
            # 组合最终键
            cache_key = ":".join(base_parts)
            
            logger.debug(f"生成缓存键: {cache_key}")
            return cache_key
            
        except Exception as e:
            logger.error(f"缓存键生成失败: {e}")
            # 生成fallback键
            fallback_key = f"{self.config.key_prefix}:error:{node_name}:{hash(str(inputs))}"
            return fallback_key
    
    def _serialize_value(self, value: Any) -> bytes:
        """序列化值"""
        try:
            if self.config.serialize_method == "json":
                return json.dumps(value, ensure_ascii=False).encode()
            elif self.config.serialize_method == "pickle":
                return pickle.dumps(value)
            else:
                return str(value).encode()
        except Exception as e:
            logger.error(f"序列化失败: {e}")
            return pickle.dumps(value)  # 降级到pickle
    
    def _deserialize_value(self, data: bytes) -> Any:
        """反序列化值"""
        try:
            if self.config.serialize_method == "json":
                return json.loads(data.decode())
            elif self.config.serialize_method == "pickle":
                return pickle.loads(data)
            else:
                return data.decode()
        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            try:
                return pickle.loads(data)  # 降级到pickle
            except:
                return None


class RedisNodeCache(NodeCache):
    """Redis实现的Node缓存"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._redis_pool: Optional[Any] = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3
    
    async def _get_redis(self):
        """获取Redis连接，支持连接重试"""
        if self._redis_pool is None:
            try:
                import redis.asyncio as redis
                self._redis_pool = redis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # 处理二进制数据
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                    max_connections=20
                )
                # 测试连接
                await self._redis_pool.ping()
                self._connection_attempts = 0
                logger.info("Redis缓存连接成功")
                
            except Exception as e:
                self._connection_attempts += 1
                logger.error(f"Redis连接失败 (尝试 {self._connection_attempts}): {e}")
                
                if self._connection_attempts >= self._max_connection_attempts:
                    logger.error("Redis连接达到最大重试次数，禁用缓存")
                    self.config.enabled = False
                
                raise e
        
        return self._redis_pool
    
    async def get(self, key: str) -> Optional[Any]:
        """从Redis获取缓存值"""
        if not self.config.enabled:
            return None
        
        start_time = time.time()
        try:
            redis = await self._get_redis()
            data = await redis.get(key)
            
            # 记录延迟
            latency = time.time() - start_time
            self.stats.get_latency_total += latency
            self.stats.get_operations += 1
            
            if data is None:
                self.stats.miss_count += 1
                logger.debug(f"缓存未命中: {key}")
                return None
            
            # 反序列化
            value = self._deserialize_value(data)
            if value is not None:
                self.stats.hit_count += 1
                logger.debug(f"缓存命中: {key}")
                return value
            else:
                self.stats.miss_count += 1
                return None
                
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"获取缓存失败 {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置Redis缓存值"""
        if not self.config.enabled:
            return False
        
        start_time = time.time()
        try:
            redis = await self._get_redis()
            
            # 序列化值
            serialized_data = self._serialize_value(value)
            
            # 使用配置的TTL或默认TTL
            expire_time = ttl or self.config.ttl_default
            
            # 存储到Redis
            result = await redis.set(key, serialized_data, ex=expire_time)
            
            # 记录延迟
            latency = time.time() - start_time
            self.stats.set_latency_total += latency
            self.stats.set_operations += 1
            
            if result:
                self.stats.set_count += 1
                logger.debug(f"缓存设置成功: {key}, TTL: {expire_time}秒")
                return True
            
            return False
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"设置缓存失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除Redis缓存"""
        if not self.config.enabled:
            return False
        
        try:
            redis = await self._get_redis()
            result = await redis.delete(key)
            success = bool(result)
            logger.debug(f"删除缓存: {key}, 结果: {success}")
            return success
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"删除缓存失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查Redis缓存是否存在"""
        if not self.config.enabled:
            return False
        
        try:
            redis = await self._get_redis()
            result = await redis.exists(key)
            return bool(result)
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"检查缓存存在性失败 {key}: {e}")
            return False
    
    async def clear(self, pattern: str = "*") -> int:
        """清理匹配模式的缓存"""
        if not self.config.enabled:
            return 0
        
        try:
            redis = await self._get_redis()
            
            # 构建完整匹配模式
            full_pattern = f"{self.config.key_prefix}:{pattern}" if not pattern.startswith(self.config.key_prefix) else pattern
            
            # 获取匹配的键
            keys = await redis.keys(full_pattern)
            
            if keys:
                result = await redis.delete(*keys)
                logger.info(f"清理缓存: 模式={full_pattern}, 删除={result}个键")
                return result
            
            return 0
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"清理缓存失败 {pattern}: {e}")
            return 0
    
    async def cleanup_expired(self) -> int:
        """清理过期缓存（Redis会自动清理，这里主要用于统计）"""
        try:
            redis = await self._get_redis()
            # 获取所有缓存键的信息
            pattern = f"{self.config.key_prefix}:*"
            keys = await redis.keys(pattern)
            
            expired_count = 0
            for key in keys:
                ttl = await redis.ttl(key)
                if ttl == -2:  # 键不存在（已过期）
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(f"检测到 {expired_count} 个过期缓存键")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        basic_stats = self.stats.to_dict()
        
        try:
            redis = await self._get_redis()
            
            # Redis特定统计
            info = await redis.info('memory')
            used_memory = info.get('used_memory', 0)
            used_memory_human = info.get('used_memory_human', '0B')
            
            # 获取缓存键数量
            pattern = f"{self.config.key_prefix}:*"
            keys = await redis.keys(pattern)
            cache_entries = len(keys)
            
            basic_stats.update({
                "redis_used_memory": used_memory,
                "redis_used_memory_human": used_memory_human,
                "cache_entries": cache_entries,
                "max_entries": self.config.max_entries,
                "backend": "redis"
            })
            
        except Exception as e:
            logger.error(f"获取Redis统计信息失败: {e}")
            basic_stats.update({
                "backend": "redis",
                "stats_error": str(e)
            })
        
        return basic_stats
    
    async def close(self):
        """关闭Redis连接"""
        if self._redis_pool:
            await self._redis_pool.close()
            self._redis_pool = None
            logger.info("Redis缓存连接已关闭")


class MemoryNodeCache(NodeCache):
    """内存实现的Node缓存（用于降级或测试）"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (value, expire_time)
        self._access_order: Dict[str, float] = {}  # LRU跟踪
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, (value, expire_time) in self._cache.items():
            if expire_time > 0 and current_time > expire_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)
    
    def _enforce_max_entries(self):
        """强制执行最大条目限制（LRU淘汰）"""
        if len(self._cache) <= self.config.max_entries:
            return
        
        # 按访问时间排序，移除最久未使用的条目
        sorted_keys = sorted(self._access_order.items(), key=lambda x: x[1])
        keys_to_remove = len(self._cache) - self.config.max_entries
        
        for i in range(keys_to_remove):
            key = sorted_keys[i][0]
            self._cache.pop(key, None)
            self._access_order.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """从内存获取缓存值"""
        if not self.config.enabled:
            return None
        
        self._cleanup_expired()
        
        if key not in self._cache:
            self.stats.miss_count += 1
            return None
        
        value, expire_time = self._cache[key]
        current_time = time.time()
        
        # 检查是否过期
        if expire_time > 0 and current_time > expire_time:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)
            self.stats.miss_count += 1
            return None
        
        # 更新访问时间（LRU）
        self._access_order[key] = current_time
        self.stats.hit_count += 1
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置内存缓存值"""
        if not self.config.enabled:
            return False
        
        try:
            current_time = time.time()
            expire_time = 0
            
            if ttl:
                expire_time = current_time + ttl
            elif self.config.ttl_default > 0:
                expire_time = current_time + self.config.ttl_default
            
            self._cache[key] = (value, expire_time)
            self._access_order[key] = current_time
            
            # 清理过期条目和强制执行大小限制
            self._cleanup_expired()
            self._enforce_max_entries()
            
            self.stats.set_count += 1
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error(f"设置内存缓存失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除内存缓存"""
        if key in self._cache:
            self._cache.pop(key)
            self._access_order.pop(key, None)
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """检查内存缓存是否存在"""
        return key in self._cache
    
    async def clear(self, pattern: str = "*") -> int:
        """清理内存缓存"""
        if pattern == "*":
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count
        
        # 模式匹配清理
        import fnmatch
        keys_to_remove = [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]
        
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)
        
        return len(keys_to_remove)
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取内存缓存统计信息"""
        basic_stats = self.stats.to_dict()
        
        basic_stats.update({
            "cache_entries": len(self._cache),
            "max_entries": self.config.max_entries,
            "backend": "memory"
        })
        
        return basic_stats


def create_node_cache(config: CacheConfig) -> NodeCache:
    """创建Node缓存实例"""
    if config.backend == "redis":
        return RedisNodeCache(config)
    elif config.backend == "memory":
        return MemoryNodeCache(config)
    else:
        logger.warning(f"不支持的缓存后端: {config.backend}，降级到内存缓存")
        return MemoryNodeCache(config)