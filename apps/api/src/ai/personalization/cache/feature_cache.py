import json
import asyncio
from typing import Dict, Optional, Any, List, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from redis.asyncio import Redis
import hashlib
import logging
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    default_ttl: int = 300  # 默认过期时间（秒）
    max_cache_size: int = 10000  # 最大缓存项数
    enable_compression: bool = True  # 启用压缩
    cache_hit_tracking: bool = True  # 跟踪缓存命中率
    preload_popular: bool = True  # 预加载热门项
    eviction_policy: str = "LRU"  # 驱逐策略: LRU, LFU, FIFO


class FeatureCacheManager:
    """特征缓存管理器"""
    
    def __init__(self, redis_client: Redis, config: CacheConfig = None):
        self.redis = redis_client
        self.config = config or CacheConfig()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0
        }
        self._lock = asyncio.Lock()
        self._popular_keys: Set[str] = set()
        self._access_counts: Dict[str, int] = {}
        
    async def get(
        self,
        key: str,
        namespace: str = "features"
    ) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
            namespace: 命名空间
            
        Returns:
            Optional[Any]: 缓存值，如果不存在则返回None
        """
        try:
            full_key = self._make_key(namespace, key)
            
            # 获取缓存数据
            cached_data = await self.redis.get(full_key)
            
            if cached_data:
                # 更新统计
                self.cache_stats["hits"] += 1
                
                # 更新访问计数（用于LFU）
                if self.config.cache_hit_tracking:
                    await self._update_access_count(full_key)
                
                # 反序列化数据
                data = self._deserialize(cached_data)
                
                # 更新TTL（可选的滑动过期）
                await self._refresh_ttl(full_key)
                
                return data
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"缓存获取失败 key={key}: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "features"
    ) -> bool:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            namespace: 命名空间
            
        Returns:
            bool: 是否设置成功
        """
        try:
            full_key = self._make_key(namespace, key)
            ttl = ttl or self.config.default_ttl
            
            # 序列化数据
            serialized_data = self._serialize(value)
            
            # 检查缓存大小限制
            await self._check_cache_size(namespace)
            
            # 设置缓存
            await self.redis.setex(full_key, ttl, serialized_data)
            
            # 记录访问（用于预加载）
            if self.config.preload_popular:
                await self._track_popular_key(full_key)
            
            return True
            
        except Exception as e:
            logger.error(f"缓存设置失败 key={key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    async def delete(
        self,
        key: str,
        namespace: str = "features"
    ) -> bool:
        """删除缓存值
        
        Args:
            key: 缓存键
            namespace: 命名空间
            
        Returns:
            bool: 是否删除成功
        """
        try:
            full_key = self._make_key(namespace, key)
            result = await self.redis.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"缓存删除失败 key={key}: {e}")
            return False
    
    async def exists(
        self,
        key: str,
        namespace: str = "features"
    ) -> bool:
        """检查缓存是否存在
        
        Args:
            key: 缓存键
            namespace: 命名空间
            
        Returns:
            bool: 是否存在
        """
        try:
            full_key = self._make_key(namespace, key)
            return await self.redis.exists(full_key) > 0
            
        except Exception as e:
            logger.error(f"缓存检查失败 key={key}: {e}")
            return False
    
    async def get_batch(
        self,
        keys: List[str],
        namespace: str = "features"
    ) -> Dict[str, Any]:
        """批量获取缓存值
        
        Args:
            keys: 缓存键列表
            namespace: 命名空间
            
        Returns:
            Dict[str, Any]: 键值对字典
        """
        result = {}
        
        try:
            # 构建完整键列表
            full_keys = [self._make_key(namespace, key) for key in keys]
            
            # 批量获取
            values = await self.redis.mget(full_keys)
            
            # 处理结果
            for key, full_key, value in zip(keys, full_keys, values):
                if value:
                    self.cache_stats["hits"] += 1
                    result[key] = self._deserialize(value)
                    
                    # 更新访问计数
                    if self.config.cache_hit_tracking:
                        await self._update_access_count(full_key)
                else:
                    self.cache_stats["misses"] += 1
                    
        except Exception as e:
            logger.error(f"批量缓存获取失败: {e}")
            self.cache_stats["errors"] += 1
            
        return result
    
    async def set_batch(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
        namespace: str = "features"
    ) -> int:
        """批量设置缓存值
        
        Args:
            items: 键值对字典
            ttl: 过期时间（秒）
            namespace: 命名空间
            
        Returns:
            int: 成功设置的数量
        """
        success_count = 0
        ttl = ttl or self.config.default_ttl
        
        try:
            # 使用管道批量设置
            pipe = self.redis.pipeline()
            
            for key, value in items.items():
                full_key = self._make_key(namespace, key)
                serialized_data = self._serialize(value)
                pipe.setex(full_key, ttl, serialized_data)
            
            results = await pipe.execute()
            success_count = sum(1 for r in results if r)
            
            # 跟踪热门键
            if self.config.preload_popular:
                for key in items.keys():
                    full_key = self._make_key(namespace, key)
                    await self._track_popular_key(full_key)
                    
        except Exception as e:
            logger.error(f"批量缓存设置失败: {e}")
            self.cache_stats["errors"] += 1
            
        return success_count
    
    async def clear_namespace(self, namespace: str = "features") -> int:
        """清空命名空间下的所有缓存
        
        Args:
            namespace: 命名空间
            
        Returns:
            int: 删除的键数量
        """
        try:
            pattern = f"{namespace}:*"
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    deleted_count += await self.redis.delete(*keys)
                
                if cursor == 0:
                    break
                    
            return deleted_count
            
        except Exception as e:
            logger.error(f"清空命名空间失败 namespace={namespace}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = self.cache_stats.copy()
        
        # 计算命中率
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0.0
        
        # 添加配置信息
        stats["config"] = asdict(self.config)
        
        # 添加热门键信息
        if self.config.preload_popular:
            stats["popular_keys_count"] = len(self._popular_keys)
            stats["top_popular_keys"] = list(self._popular_keys)[:10]
        
        return stats
    
    async def warm_up(
        self,
        keys: List[str],
        loader_func: callable,
        namespace: str = "features"
    ) -> int:
        """预热缓存
        
        Args:
            keys: 需要预热的键列表
            loader_func: 加载数据的函数
            namespace: 命名空间
            
        Returns:
            int: 成功预热的数量
        """
        success_count = 0
        
        for key in keys:
            try:
                # 检查是否已存在
                if not await self.exists(key, namespace):
                    # 加载数据
                    value = await loader_func(key)
                    if value is not None:
                        # 设置缓存
                        if await self.set(key, value, namespace=namespace):
                            success_count += 1
                            
            except Exception as e:
                logger.error(f"预热缓存失败 key={key}: {e}")
                
        logger.info(f"预热缓存完成: {success_count}/{len(keys)} 成功")
        return success_count
    
    def _make_key(self, namespace: str, key: str) -> str:
        """构建完整的缓存键
        
        Args:
            namespace: 命名空间
            key: 原始键
            
        Returns:
            str: 完整键
        """
        return f"{namespace}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """序列化数据
        
        Args:
            value: 要序列化的值
            
        Returns:
            bytes: 序列化后的数据
        """
        try:
            # 尝试JSON序列化（更快）
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(value, default=str).encode()
            else:
                # 使用pickle作为后备
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"序列化失败: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """反序列化数据
        
        Args:
            data: 序列化的数据
            
        Returns:
            Any: 反序列化后的值
        """
        try:
            # 尝试JSON反序列化
            try:
                return json.loads(data.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 使用pickle作为后备
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            raise
    
    async def _check_cache_size(self, namespace: str):
        """检查并控制缓存大小
        
        Args:
            namespace: 命名空间
        """
        try:
            # 获取当前缓存数量
            pattern = f"{namespace}:*"
            cursor, keys = await self.redis.scan(
                cursor=0,
                match=pattern,
                count=1000
            )
            
            if len(keys) >= self.config.max_cache_size:
                # 执行驱逐策略
                await self._evict_keys(namespace, keys)
                
        except Exception as e:
            logger.error(f"检查缓存大小失败: {e}")
    
    async def _evict_keys(self, namespace: str, keys: List[str]):
        """驱逐缓存键
        
        Args:
            namespace: 命名空间
            keys: 键列表
        """
        try:
            evict_count = max(1, len(keys) // 10)  # 驱逐10%
            
            if self.config.eviction_policy == "LRU":
                # LRU: 删除最近最少使用的
                # 获取所有键的访问时间
                access_times = []
                for key in keys[:100]:  # 采样
                    ttl = await self.redis.ttl(key)
                    access_times.append((key, ttl))
                
                # 排序并删除TTL最小的
                access_times.sort(key=lambda x: x[1])
                keys_to_evict = [k for k, _ in access_times[:evict_count]]
                
            elif self.config.eviction_policy == "LFU":
                # LFU: 删除访问频率最低的
                sorted_keys = sorted(
                    keys[:100],
                    key=lambda k: self._access_counts.get(k, 0)
                )
                keys_to_evict = sorted_keys[:evict_count]
                
            else:  # FIFO
                # FIFO: 删除最早的
                keys_to_evict = keys[:evict_count]
            
            # 执行删除
            if keys_to_evict:
                await self.redis.delete(*keys_to_evict)
                self.cache_stats["evictions"] += len(keys_to_evict)
                logger.debug(f"驱逐了 {len(keys_to_evict)} 个缓存键")
                
        except Exception as e:
            logger.error(f"驱逐缓存失败: {e}")
    
    async def _update_access_count(self, key: str):
        """更新访问计数
        
        Args:
            key: 缓存键
        """
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        
        # 定期清理访问计数
        if len(self._access_counts) > self.config.max_cache_size * 2:
            # 保留访问次数最多的一半
            sorted_items = sorted(
                self._access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._access_counts = dict(sorted_items[:self.config.max_cache_size])
    
    async def _refresh_ttl(self, key: str):
        """刷新TTL（滑动过期）
        
        Args:
            key: 缓存键
        """
        try:
            # 延长过期时间
            await self.redis.expire(key, self.config.default_ttl)
        except Exception as e:
            logger.debug(f"刷新TTL失败 key={key}: {e}")
    
    async def _track_popular_key(self, key: str):
        """跟踪热门键
        
        Args:
            key: 缓存键
        """
        self._popular_keys.add(key)
        
        # 限制热门键集合大小
        if len(self._popular_keys) > 100:
            # 随机删除一些
            self._popular_keys = set(list(self._popular_keys)[-50:])


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, redis_client: Redis, local_cache_size: int = 100):
        self.redis_cache = FeatureCacheManager(redis_client)
        self.local_cache = {}  # L1缓存（内存）
        self.local_cache_size = local_cache_size
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """获取值（先L1，后L2）"""
        # 先查L1缓存
        if key in self.local_cache:
            return self.local_cache[key]
        
        # 查L2缓存（Redis）
        value = await self.redis_cache.get(key)
        
        if value is not None:
            # 更新L1缓存
            await self._update_local_cache(key, value)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置值（同时更新L1和L2）"""
        # 更新L1缓存
        await self._update_local_cache(key, value)
        
        # 更新L2缓存
        return await self.redis_cache.set(key, value, ttl)
    
    async def _update_local_cache(self, key: str, value: Any):
        """更新本地缓存"""
        async with self._lock:
            # LRU驱逐
            if len(self.local_cache) >= self.local_cache_size:
                # 删除最早的项
                first_key = next(iter(self.local_cache))
                del self.local_cache[first_key]
            
            self.local_cache[key] = value