"""
LangGraph 0.6.5 Node Caching Implementation
基于LangGraph 0.6.5的节点级缓存功能实现
兼容官方CachePolicy和InMemoryCache API
"""
from typing import Any, Dict, Optional, Callable, Literal, Union
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import hashlib
import json
import asyncio
from abc import ABC, abstractmethod

try:
    # 尝试导入LangGraph官方缓存类型
    from langgraph.types import CachePolicy as LangGraphCachePolicy
    from langgraph.cache.memory import InMemoryCache as LangGraphInMemoryCache
    LANGGRAPH_CACHE_AVAILABLE = True
except ImportError:
    LANGGRAPH_CACHE_AVAILABLE = False

from .state import MessagesState


@dataclass
class CachePolicy:
    """缓存策略配置 - 兼容LangGraph官方CachePolicy"""
    ttl: int = 300  # 缓存存活时间（秒）
    max_size: int = 1000  # 最大缓存条目数
    cache_key_fields: Optional[list] = None  # 用于生成缓存键的字段
    enabled: bool = True  # 是否启用缓存
    
    @classmethod
    def from_langgraph_policy(cls, policy) -> 'CachePolicy':
        """从LangGraph CachePolicy转换"""
        if LANGGRAPH_CACHE_AVAILABLE and isinstance(policy, LangGraphCachePolicy):
            return cls(
                ttl=getattr(policy, 'ttl', 300),
                enabled=True
            )
        return cls()


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除缓存值"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """清空所有缓存"""
        pass


class InMemoryCache(CacheBackend):
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        expires_at = entry.get("expires_at")
        
        # 检查是否过期
        if expires_at and utc_now() > expires_at:
            await self.delete(key)
            return None
        
        # 更新访问时间
        self.access_times[key] = utc_now()
        return entry["value"]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        # 检查缓存大小限制
        if len(self.cache) >= self.max_size and key not in self.cache:
            await self._evict_lru()
        
        expires_at = None
        if ttl is not None:
            expires_at = utc_now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": utc_now()
        }
        self.access_times[key] = utc_now()
    
    async def delete(self, key: str) -> None:
        """删除缓存值"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    async def clear(self) -> None:
        """清空所有缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    async def _evict_lru(self) -> None:
        """淘汰最近最少使用的缓存项"""
        if not self.access_times:
            return
        
        # 找到访问时间最早的键
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        await self.delete(lru_key)


class RedisCache(CacheBackend):
    """Redis缓存实现 - 兼容LangGraph Redis Store"""
    
    def __init__(self, redis_client, key_prefix: str = "langgraph_cache:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self._connection_pool = None
    
    @classmethod
    async def from_conn_string(cls, conn_string: str, key_prefix: str = "langgraph_cache:") -> 'RedisCache':
        """从连接字符串创建Redis缓存实例"""
        import redis.asyncio as redis
        pool = redis.ConnectionPool.from_url(conn_string)
        redis_client = redis.Redis(connection_pool=pool)
        instance = cls(redis_client, key_prefix)
        instance._connection_pool = pool
        return instance
    
    async def setup(self):
        """设置Redis缓存"""
        try:
            # 测试连接
            await self.redis.ping()
            print("Redis缓存连接成功")
        except Exception as e:
            print(f"Redis缓存连接失败: {e}")
            raise
    
    async def close(self):
        """关闭Redis连接"""
        try:
            if self._connection_pool:
                await self._connection_pool.disconnect()
        except Exception as e:
            print(f"关闭Redis连接失败: {e}")
    
    def _make_key(self, key: str) -> str:
        """生成Redis键名"""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            redis_key = self._make_key(key)
            value = await self.redis.get(redis_key)
            if value is None:
                return None
            
            # 尝试JSON反序列化
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # 如果JSON解码失败，返回原始字符串
                return value.decode('utf-8') if isinstance(value, bytes) else value
        except Exception as e:
            print(f"Redis缓存获取失败: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        try:
            redis_key = self._make_key(key)
            
            # 序列化值
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str, ensure_ascii=False)
            else:
                serialized_value = str(value)
            
            if ttl is not None:
                await self.redis.setex(redis_key, ttl, serialized_value)
            else:
                await self.redis.set(redis_key, serialized_value)
                
            print(f"Redis缓存设置成功: {redis_key}")
        except Exception as e:
            print(f"Redis缓存设置失败: {e}")
    
    async def delete(self, key: str) -> None:
        """删除缓存值"""
        try:
            redis_key = self._make_key(key)
            deleted_count = await self.redis.delete(redis_key)
            if deleted_count > 0:
                print(f"Redis缓存删除成功: {redis_key}")
        except Exception as e:
            print(f"Redis缓存删除失败: {e}")
    
    async def clear(self) -> None:
        """清空所有缓存"""
        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0
            total_deleted = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted_count = await self.redis.delete(*keys)
                    total_deleted += deleted_count
                if cursor == 0:
                    break
                    
            print(f"Redis缓存清空完成，删除 {total_deleted} 个键")
        except Exception as e:
            print(f"Redis缓存清空失败: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取Redis缓存统计信息"""
        try:
            info = await self.redis.info()
            pattern = f"{self.key_prefix}*"
            
            # 计算缓存键数量
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=1000)
                key_count += len(keys)
                if cursor == 0:
                    break
            
            return {
                "key_count": key_count,
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": f"{(info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)) * 100:.2f}%"
            }
        except Exception as e:
            print(f"获取Redis统计信息失败: {e}")
            return {"error": str(e)}


class NodeCacheManager:
    """节点缓存管理器"""
    
    def __init__(self, backend: CacheBackend, default_policy: Optional[CachePolicy] = None):
        self.backend = backend
        self.default_policy = default_policy or CachePolicy()
        self.node_policies: Dict[str, CachePolicy] = {}
    
    def set_node_policy(self, node_name: str, policy: CachePolicy) -> None:
        """为特定节点设置缓存策略"""
        self.node_policies[node_name] = policy
    
    def get_node_policy(self, node_name: str) -> CachePolicy:
        """获取节点缓存策略"""
        return self.node_policies.get(node_name, self.default_policy)
    
    def _generate_cache_key(self, node_name: str, state: MessagesState, policy: CachePolicy) -> str:
        """生成缓存键"""
        key_data = {
            "node": node_name
        }
        
        # 根据策略添加状态字段到缓存键
        if policy.cache_key_fields:
            for field in policy.cache_key_fields:
                if field in state and field == "messages":
                    # 对于消息字段，只使用用户消息内容，忽略timestamp等变化字段
                    messages = state.get("messages", [])
                    user_messages = []
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_messages.append({
                                "role": msg.get("role"),
                                "content": msg.get("content")
                            })
                    key_data[field] = json.dumps(user_messages, sort_keys=True)
                elif field in state:
                    # 对复杂对象进行序列化
                    try:
                        key_data[field] = json.dumps(state[field], sort_keys=True, default=str)
                    except:
                        key_data[field] = str(state[field])
        else:
            # 默认使用消息内容生成键（只包含用户消息内容）
            messages = state.get("messages", [])
            user_messages = []
            for msg in messages:
                if msg.get("role") == "user":
                    user_messages.append({
                        "role": msg.get("role"), 
                        "content": msg.get("content")
                    })
            key_data["user_messages"] = json.dumps(user_messages, sort_keys=True)
        
        # 生成哈希键
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        print(f"生成缓存键 {cache_key} for node {node_name}, key_data: {key_data}")
        return cache_key
    
    async def get_cached_result(self, node_name: str, state: MessagesState) -> Optional[MessagesState]:
        """获取缓存的节点执行结果"""
        policy = self.get_node_policy(node_name)
        
        if not policy.enabled:
            return None
        
        cache_key = self._generate_cache_key(node_name, state, policy)
        
        try:
            cached_result = await self.backend.get(cache_key)
            if cached_result is not None:
                print(f"节点 {node_name} 缓存命中: {cache_key}")
                return cached_result
        except Exception as e:
            print(f"获取缓存失败: {e}")
        
        return None
    
    async def cache_result(self, node_name: str, state: MessagesState, result: MessagesState) -> None:
        """缓存节点执行结果"""
        policy = self.get_node_policy(node_name)
        
        if not policy.enabled:
            return
        
        cache_key = self._generate_cache_key(node_name, state, policy)
        
        try:
            await self.backend.set(cache_key, result, policy.ttl)
            print(f"节点 {node_name} 结果已缓存: {cache_key}")
        except Exception as e:
            print(f"缓存结果失败: {e}")
    
    async def invalidate_node_cache(self, node_name: str) -> None:
        """使特定节点的所有缓存失效"""
        # 注意：这是一个简化实现，实际生产环境可能需要更复杂的缓存键管理
        try:
            # 清空所有缓存（简化实现）
            await self.backend.clear()
            print(f"节点 {node_name} 缓存已清空")
        except Exception as e:
            print(f"缓存清空失败: {e}")


# 全局缓存管理器实例
_cache_manager: Optional[NodeCacheManager] = None


async def create_redis_cache_manager(redis_uri: str = "redis://localhost:6379") -> NodeCacheManager:
    """创建Redis缓存管理器"""
    try:
        redis_cache = await RedisCache.from_conn_string(redis_uri)
        await redis_cache.setup()
        
        # 创建缓存管理器
        manager = NodeCacheManager(redis_cache)
        print(f"Redis缓存管理器创建成功: {redis_uri}")
        return manager
    except Exception as e:
        print(f"创建Redis缓存管理器失败: {e}，回退到内存缓存")
        # 回退到内存缓存
        return NodeCacheManager(InMemoryCache())


def get_cache_manager() -> NodeCacheManager:
    """获取全局缓存管理器"""
    global _cache_manager
    if _cache_manager is None:
        # 默认使用内存缓存
        backend = InMemoryCache()
        _cache_manager = NodeCacheManager(backend)
    return _cache_manager


def set_cache_manager(manager: NodeCacheManager) -> None:
    """设置全局缓存管理器"""
    global _cache_manager
    _cache_manager = manager


async def initialize_redis_caching(redis_uri: Optional[str] = None) -> None:
    """初始化Redis缓存（如果可用）"""
    if redis_uri:
        try:
            redis_manager = await create_redis_cache_manager(redis_uri)
            set_cache_manager(redis_manager)
            print("Redis缓存初始化成功")
        except Exception as e:
            print(f"Redis缓存初始化失败，使用内存缓存: {e}")
    else:
        print("未提供Redis URI，使用内存缓存")


class LangGraphCompatibleInMemoryCache:
    """兼容LangGraph InMemoryCache接口的包装器"""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or InMemoryCache()
        
    def compile(self, builder):
        """编译图时被调用，兼容LangGraph InMemoryCache接口"""
        return builder
        
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值 - LangGraph兼容接口"""
        return await self.backend.get(key)
        
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值 - LangGraph兼容接口"""
        return await self.backend.set(key, value, ttl)


def create_langgraph_compatible_cache(redis_uri: Optional[str] = None) -> LangGraphCompatibleInMemoryCache:
    """创建兼容LangGraph的缓存实例"""
    if redis_uri:
        try:
            import redis.asyncio as redis
            pool = redis.ConnectionPool.from_url(redis_uri)
            redis_client = redis.Redis(connection_pool=pool)
            redis_backend = RedisCache(redis_client)
            return LangGraphCompatibleInMemoryCache(redis_backend)
        except Exception as e:
            print(f"Redis缓存创建失败，使用内存缓存: {e}")
            
    return LangGraphCompatibleInMemoryCache(InMemoryCache())


def create_cached_node(node_name: str, handler: Callable, cache_policy: Optional[CachePolicy] = None):
    """创建具有缓存功能的节点包装器
    
    Args:
        node_name: 节点名称
        handler: 原始节点处理函数
        cache_policy: 缓存策略，兼容LangGraph CachePolicy
    
    Returns:
        包装后的缓存节点处理函数
    """
    cache_manager = get_cache_manager()
    
    # 处理LangGraph官方CachePolicy
    if cache_policy and LANGGRAPH_CACHE_AVAILABLE:
        cache_policy = CachePolicy.from_langgraph_policy(cache_policy)
    
    if cache_policy:
        cache_manager.set_node_policy(node_name, cache_policy)
    
    async def cached_handler(state: MessagesState, *args, **kwargs) -> MessagesState:
        """缓存包装的节点处理器"""
        # 尝试从缓存获取结果
        cached_result = await cache_manager.get_cached_result(node_name, state)
        if cached_result is not None:
            # 添加缓存命中元数据
            if "__metadata__" not in cached_result:
                cached_result["__metadata__"] = {}
            cached_result["__metadata__"]["cached"] = True
            return cached_result
        
        # 缓存未命中，执行原始处理器
        if asyncio.iscoroutinefunction(handler):
            result = await handler(state, *args, **kwargs)
        else:
            result = handler(state, *args, **kwargs)
        
        # 缓存结果
        await cache_manager.cache_result(node_name, state, result)
        
        return result
    
    return cached_handler