"""
向量缓存管理器

提供向量数据的高效缓存和LRU策略
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import asyncio

from src.core.logging import get_logger
logger = get_logger(__name__)

class VectorCacheManager:
    """向量缓存管理器"""
    
    def __init__(self, redis_client, cache_size: int = 10000):
        self.redis = redis_client
        self.cache_size = cache_size
        self.hit_rate_threshold = 0.7
        self.access_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
    async def get_cached_vector(
        self, 
        vector_id: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """获取缓存的向量"""
        cache_key = f"vector:{vector_id}"
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                vector = np.array(data["vector"])
                metadata = data["metadata"]
                
                # 更新访问统计
                await self._update_access_stats(vector_id, hit=True)
                
                logger.debug(f"Cache hit for vector {vector_id}")
                return vector, metadata
            else:
                await self._update_access_stats(vector_id, hit=False)
                logger.debug(f"Cache miss for vector {vector_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached vector {vector_id}: {e}")
            await self._update_access_stats(vector_id, hit=False)
            return None
    
    async def cache_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """缓存向量"""
        cache_key = f"vector:{vector_id}"
        
        try:
            # 检查缓存空间
            current_size = await self._get_cache_size()
            if current_size >= self.cache_size:
                await self._evict_least_used()
            
            # 序列化向量数据
            cache_data = {
                "vector": vector.tolist(),
                "metadata": metadata,
                "timestamp": utc_now().isoformat(),
                "access_count": 1
            }
            
            await self.redis.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data)
            )
            
            # 更新访问记录
            access_key = f"access:{vector_id}"
            await self.redis.setex(access_key, ttl, "1")
            
            logger.debug(f"Cached vector {vector_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching vector {vector_id}: {e}")
            return False
    
    async def _update_access_stats(self, vector_id: str, hit: bool) -> None:
        """更新访问统计"""
        self.access_stats["total_requests"] += 1
        
        if hit:
            self.access_stats["hits"] += 1
            # 更新访问时间戳
            access_key = f"access:{vector_id}"
            try:
                await self.redis.setex(access_key, 3600, str(int(utc_now().timestamp())))
            except Exception as e:
                logger.warning(f"Failed to update access timestamp: {e}")
        else:
            self.access_stats["misses"] += 1
    
    async def _get_cache_size(self) -> int:
        """获取当前缓存大小"""
        try:
            # 计算vector:*键的数量
            keys = await self.redis.keys("vector:*")
            return len(keys)
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return 0
    
    async def _evict_least_used(self) -> None:
        """驱逐最少使用的向量"""
        try:
            # 驱逐最少使用的10%
            evict_count = max(1, self.cache_size // 10)
            
            # 获取访问时间最旧的键
            keys_to_evict = await self._get_lru_keys(evict_count)
            
            if keys_to_evict:
                # 删除向量和访问记录
                vector_keys = [f"vector:{key}" for key in keys_to_evict]
                access_keys = [f"access:{key}" for key in keys_to_evict]
                all_keys = vector_keys + access_keys
                
                await self.redis.delete(*all_keys)
                self.access_stats["evictions"] += len(keys_to_evict)
                
                logger.debug(f"Evicted {len(keys_to_evict)} vectors from cache")
                
        except Exception as e:
            logger.error(f"Error during cache eviction: {e}")
    
    async def _get_lru_keys(self, count: int) -> List[str]:
        """获取最少使用的键"""
        try:
            # 获取所有访问记录
            access_keys = await self.redis.keys("access:*")
            key_timestamps = []
            
            for access_key in access_keys:
                try:
                    timestamp = await self.redis.get(access_key)
                    if timestamp:
                        vector_id = access_key.decode() if isinstance(access_key, bytes) else access_key
                        vector_id = vector_id.replace("access:", "")
                        key_timestamps.append((vector_id, int(timestamp)))
                except Exception:
                    continue
            
            # 按时间戳排序，最旧的在前
            key_timestamps.sort(key=lambda x: x[1])
            
            # 返回最旧的count个键
            return [key for key, _ in key_timestamps[:count]]
            
        except Exception as e:
            logger.error(f"Error getting LRU keys: {e}")
            return []
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.access_stats["total_requests"]
        hit_rate = 0.0
        
        if total_requests > 0:
            hit_rate = self.access_stats["hits"] / total_requests
        
        current_size = await self._get_cache_size()
        
        return {
            "hit_rate": hit_rate,
            "hits": self.access_stats["hits"],
            "misses": self.access_stats["misses"],
            "evictions": self.access_stats["evictions"],
            "total_requests": total_requests,
            "current_size": current_size,
            "max_size": self.cache_size,
            "utilization": current_size / self.cache_size if self.cache_size > 0 else 0.0,
            "timestamp": utc_now().isoformat()
        }
    
    async def clear_cache(self) -> bool:
        """清空缓存"""
        try:
            # 删除所有向量和访问记录
            vector_keys = await self.redis.keys("vector:*")
            access_keys = await self.redis.keys("access:*")
            
            all_keys = vector_keys + access_keys
            if all_keys:
                await self.redis.delete(*all_keys)
            
            # 重置统计信息
            self.access_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0
            }
            
            logger.info("Vector cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def preload_vectors(
        self, 
        vectors_data: List[Tuple[str, np.ndarray, Dict[str, Any]]],
        ttl: int = 3600
    ) -> int:
        """预加载向量到缓存"""
        successful_loads = 0
        
        for vector_id, vector, metadata in vectors_data:
            try:
                success = await self.cache_vector(vector_id, vector, metadata, ttl)
                if success:
                    successful_loads += 1
            except Exception as e:
                logger.error(f"Error preloading vector {vector_id}: {e}")
                continue
        
        logger.info(f"Preloaded {successful_loads}/{len(vectors_data)} vectors")
        return successful_loads
    
    async def get_cache_health(self) -> Dict[str, Any]:
        """获取缓存健康状态"""
        stats = await self.get_cache_stats()
        
        health_status = "healthy"
        issues = []
        
        # 检查命中率
        if stats["hit_rate"] < self.hit_rate_threshold and stats["total_requests"] > 100:
            health_status = "warning"
            issues.append(f"Low hit rate: {stats['hit_rate']:.2%} (threshold: {self.hit_rate_threshold:.2%})")
        
        # 检查缓存利用率
        if stats["utilization"] > 0.9:
            health_status = "warning"
            issues.append(f"High cache utilization: {stats['utilization']:.2%}")
        
        # 检查驱逐频率
        eviction_rate = stats["evictions"] / max(stats["total_requests"], 1)
        if eviction_rate > 0.1:  # 超过10%的请求导致驱逐
            health_status = "warning"
            issues.append(f"High eviction rate: {eviction_rate:.2%}")
        
        return {
            "status": health_status,
            "issues": issues,
            "stats": stats,
            "recommendations": self._get_recommendations(stats)
        }
    
    def _get_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        if stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache TTL or cache size")
        
        if stats["utilization"] > 0.8:
            recommendations.append("Consider increasing cache size to reduce evictions")
        
        if stats["evictions"] > stats["hits"]:
            recommendations.append("Cache size may be too small for current workload")
        
        return recommendations
