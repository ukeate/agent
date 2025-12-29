"""
GraphRAG缓存管理器

提供多层缓存机制，包括：
- 查询结果缓存
- 图谱片段缓存
- 推理路径缓存
- 智能预加载和缓存失效
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import asdict
from collections import OrderedDict
import redis.asyncio as redis
from .data_models import GraphRAGRequest, GraphRAGResponse, GraphContext, ReasoningPath
from ...core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

class CacheManager:
    """GraphRAG缓存管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        
        # 内存缓存 (LRU)
        self.memory_cache: OrderedDict = OrderedDict()
        self.max_memory_cache_size = 1000
        
        # 缓存统计
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "redis_hits": 0,
            "evictions": 0,
            "errors": 0
        }
        
        # 缓存配置
        self.cache_config = {
            "query_result_ttl": 3600,  # 查询结果缓存1小时
            "graph_context_ttl": 7200,  # 图谱上下文缓存2小时
            "reasoning_path_ttl": 1800,  # 推理路径缓存30分钟
            "subgraph_ttl": 14400,     # 子图缓存4小时
            "enable_compression": True,
            "max_cache_key_length": 250,
            "cache_key_prefix": "graphrag:"
        }

    async def initialize(self):
        """初始化Redis连接"""
        try:
            if hasattr(self.settings, 'REDIS_URL') and self.settings.REDIS_URL:
                self.redis_client = redis.from_url(
                    self.settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # 测试连接
                await self.redis_client.ping()
                logger.info("GraphRAG缓存管理器Redis连接成功")
            else:
                logger.warning("Redis未配置，仅使用内存缓存")
        except Exception as e:
            logger.error(f"Redis连接失败，仅使用内存缓存: {e}")
            self.redis_client = None

    async def close(self):
        """关闭连接"""
        if self.redis_client:
            await self.redis_client.aclose()

    def _generate_cache_key(self, key_data: Dict[str, Any], key_type: str) -> str:
        """生成缓存键"""
        # 创建确定性的键
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        
        cache_key = f"{self.cache_config['cache_key_prefix']}{key_type}:{key_hash}"
        
        # 确保键长度不超过限制
        if len(cache_key) > self.cache_config['max_cache_key_length']:
            cache_key = cache_key[:self.cache_config['max_cache_key_length']]
        
        return cache_key

    def _query_id_cache_key(self, query_id: str) -> str:
        key = f"{self.cache_config['cache_key_prefix']}query_id:{query_id}"
        if len(key) > self.cache_config["max_cache_key_length"]:
            return key[: self.cache_config["max_cache_key_length"]]
        return key

    def _compress_data(self, data: Any) -> str:
        """压缩数据"""
        if not self.cache_config['enable_compression']:
            return json.dumps(data, ensure_ascii=False, default=str)
        
        try:
            import gzip
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            compressed = gzip.compress(json_str.encode('utf-8'))
            return compressed.hex()  # 转换为十六进制字符串存储
        except Exception as e:
            logger.warning(f"数据压缩失败，使用原始JSON: {e}")
            return json.dumps(data, ensure_ascii=False, default=str)

    def _decompress_data(self, compressed_data: str) -> Any:
        """解压缩数据"""
        if not self.cache_config['enable_compression']:
            return json.loads(compressed_data)
        
        try:
            import gzip
            # 尝试解压缩
            compressed_bytes = bytes.fromhex(compressed_data)
            decompressed = gzip.decompress(compressed_bytes)
            return json.loads(decompressed.decode('utf-8'))
        except Exception:
            # 如果解压缩失败，尝试直接解析JSON
            try:
                return json.loads(compressed_data)
            except Exception as e:
                logger.error(f"数据解压缩失败: {e}")
                return None

    async def _get_from_memory(self, cache_key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        if cache_key in self.memory_cache:
            # 移动到最近使用的位置
            data = self.memory_cache.pop(cache_key)
            self.memory_cache[cache_key] = data
            self.cache_stats["memory_hits"] += 1
            return data
        return None

    async def _set_to_memory(self, cache_key: str, data: Any):
        """设置到内存缓存"""
        # 如果缓存已满，移除最老的项
        if len(self.memory_cache) >= self.max_memory_cache_size:
            oldest_key = next(iter(self.memory_cache))
            self.memory_cache.pop(oldest_key)
            self.cache_stats["evictions"] += 1
        
        self.memory_cache[cache_key] = data

    async def _get_from_redis(self, cache_key: str) -> Optional[Any]:
        """从Redis获取数据"""
        if not self.redis_client:
            return None
        
        try:
            compressed_data = await self.redis_client.get(cache_key)
            if compressed_data:
                data = self._decompress_data(compressed_data)
                if data:
                    # 同时存储到内存缓存
                    await self._set_to_memory(cache_key, data)
                    self.cache_stats["redis_hits"] += 1
                    return data
        except Exception as e:
            logger.warning(f"Redis获取数据失败: {e}")
            self.cache_stats["errors"] += 1
        
        return None

    async def _set_to_redis(self, cache_key: str, data: Any, ttl: int):
        """设置到Redis"""
        if not self.redis_client:
            return
        
        try:
            compressed_data = self._compress_data(data)
            await self.redis_client.setex(cache_key, ttl, compressed_data)
        except Exception as e:
            logger.warning(f"Redis设置数据失败: {e}")
            self.cache_stats["errors"] += 1

    async def get_cached_result(
        self, 
        query: str, 
        retrieval_mode: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Optional[GraphRAGResponse]:
        """获取缓存的查询结果"""
        try:
            key_data = {
                "query": query,
                "retrieval_mode": retrieval_mode,
                **(additional_params or {})
            }
            cache_key = self._generate_cache_key(key_data, "result")
            
            # 先检查内存缓存
            cached_data = await self._get_from_memory(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return cached_data
            
            # 检查Redis缓存
            cached_data = await self._get_from_redis(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return cached_data
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def cache_result(
        self, 
        query: str, 
        response: GraphRAGResponse,
        additional_params: Optional[Dict[str, Any]] = None
    ):
        """缓存查询结果"""
        try:
            key_data = {
                "query": query,
                "retrieval_mode": response.get("retrieval_mode", "hybrid"),
                **(additional_params or {})
            }
            cache_key = self._generate_cache_key(key_data, "result")
            
            # 同时存储到内存和Redis
            await self._set_to_memory(cache_key, response)
            await self._set_to_redis(
                cache_key, 
                response, 
                self.cache_config["query_result_ttl"]
            )

            query_id = response.get("query_id")
            if query_id:
                query_id_key = self._query_id_cache_key(str(query_id))
                await self._set_to_memory(query_id_key, response)
                await self._set_to_redis(
                    query_id_key,
                    response,
                    self.cache_config["query_result_ttl"],
                )
            
        except Exception as e:
            logger.error(f"缓存结果失败: {e}")
            self.cache_stats["errors"] += 1

    async def get_cached_result_by_query_id(self, query_id: str) -> Optional[GraphRAGResponse]:
        """通过查询ID获取缓存结果"""
        try:
            cache_key = self._query_id_cache_key(query_id)
            cached_data = await self._get_from_memory(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return cached_data

            cached_data = await self._get_from_redis(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return cached_data

            self.cache_stats["misses"] += 1
            return None
        except Exception as e:
            logger.error(f"通过查询ID获取缓存结果失败: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def get_cached_graph_context(
        self, 
        entity_ids: List[str], 
        expansion_depth: int
    ) -> Optional[GraphContext]:
        """获取缓存的图谱上下文"""
        try:
            key_data = {
                "entity_ids": sorted(entity_ids),  # 排序确保一致性
                "expansion_depth": expansion_depth
            }
            cache_key = self._generate_cache_key(key_data, "graph_context")
            
            cached_data = await self._get_from_memory(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return GraphContext(**cached_data)
            
            cached_data = await self._get_from_redis(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                return GraphContext(**cached_data)
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存图谱上下文失败: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def cache_graph_context(
        self, 
        entity_ids: List[str], 
        expansion_depth: int, 
        graph_context: GraphContext
    ):
        """缓存图谱上下文"""
        try:
            key_data = {
                "entity_ids": sorted(entity_ids),
                "expansion_depth": expansion_depth
            }
            cache_key = self._generate_cache_key(key_data, "graph_context")
            
            context_data = graph_context.to_dict()
            await self._set_to_memory(cache_key, context_data)
            await self._set_to_redis(
                cache_key, 
                context_data, 
                self.cache_config["graph_context_ttl"]
            )
            
        except Exception as e:
            logger.error(f"缓存图谱上下文失败: {e}")
            self.cache_stats["errors"] += 1

    async def get_cached_reasoning_paths(
        self, 
        source_entity: str, 
        target_entity: str, 
        max_hops: int
    ) -> Optional[List[ReasoningPath]]:
        """获取缓存的推理路径"""
        try:
            key_data = {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "max_hops": max_hops
            }
            cache_key = self._generate_cache_key(key_data, "reasoning_paths")
            
            cached_data = await self._get_from_memory(cache_key)
            if not cached_data:
                cached_data = await self._get_from_redis(cache_key)
            
            if cached_data:
                self.cache_stats["hits"] += 1
                return [ReasoningPath(**path_data) for path_data in cached_data]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存推理路径失败: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def cache_reasoning_paths(
        self, 
        source_entity: str, 
        target_entity: str, 
        max_hops: int, 
        reasoning_paths: List[ReasoningPath]
    ):
        """缓存推理路径"""
        try:
            key_data = {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "max_hops": max_hops
            }
            cache_key = self._generate_cache_key(key_data, "reasoning_paths")
            
            paths_data = [path.to_dict() for path in reasoning_paths]
            await self._set_to_memory(cache_key, paths_data)
            await self._set_to_redis(
                cache_key, 
                paths_data, 
                self.cache_config["reasoning_path_ttl"]
            )
            
        except Exception as e:
            logger.error(f"缓存推理路径失败: {e}")
            self.cache_stats["errors"] += 1

    async def get_cached_subgraph(
        self, 
        center_entity: str, 
        depth: int, 
        max_nodes: int
    ) -> Optional[Dict[str, Any]]:
        """获取缓存的子图"""
        try:
            key_data = {
                "center_entity": center_entity,
                "depth": depth,
                "max_nodes": max_nodes
            }
            cache_key = self._generate_cache_key(key_data, "subgraph")
            
            cached_data = await self._get_from_memory(cache_key)
            if not cached_data:
                cached_data = await self._get_from_redis(cache_key)
            
            if cached_data:
                self.cache_stats["hits"] += 1
                return cached_data
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存子图失败: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def cache_subgraph(
        self, 
        center_entity: str, 
        depth: int, 
        max_nodes: int, 
        subgraph: Dict[str, Any]
    ):
        """缓存子图"""
        try:
            key_data = {
                "center_entity": center_entity,
                "depth": depth,
                "max_nodes": max_nodes
            }
            cache_key = self._generate_cache_key(key_data, "subgraph")
            
            await self._set_to_memory(cache_key, subgraph)
            await self._set_to_redis(
                cache_key, 
                subgraph, 
                self.cache_config["subgraph_ttl"]
            )
            
        except Exception as e:
            logger.error(f"缓存子图失败: {e}")
            self.cache_stats["errors"] += 1

    async def invalidate_cache(self, pattern: str = None):
        """清除缓存"""
        try:
            if pattern:
                # 清除特定模式的缓存
                if self.redis_client:
                    keys = await self.redis_client.keys(f"{self.cache_config['cache_key_prefix']}{pattern}")
                    if keys:
                        await self.redis_client.delete(*keys)
                
                # 清除内存缓存中匹配的键
                keys_to_remove = [
                    key for key in self.memory_cache.keys() 
                    if pattern in key
                ]
                for key in keys_to_remove:
                    self.memory_cache.pop(key, None)
            else:
                # 清除所有缓存
                if self.redis_client:
                    await self.redis_client.flushdb()
                self.memory_cache.clear()
            
            logger.info(f"缓存清除完成: {pattern or '所有缓存'}")
            
        except Exception as e:
            logger.error(f"缓存清除失败: {e}")
            self.cache_stats["errors"] += 1

    async def preload_cache(self, frequent_queries: List[Dict[str, Any]]):
        """预加载常用查询的缓存"""
        try:
            logger.info(f"开始预加载缓存，共{len(frequent_queries)}个查询")
            
            # 这里可以实现预加载逻辑
            # 例如异步执行常用查询并缓存结果
            
        except Exception as e:
            logger.error(f"预加载缓存失败: {e}")
            self.cache_stats["errors"] += 1

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
            "cache_config": self.cache_config
        }

    async def warm_up_cache(self, entities: List[str], max_depth: int = 2):
        """预热缓存 - 预先计算和缓存常用实体的图谱上下文"""
        try:
            logger.info(f"开始预热缓存，实体数量: {len(entities)}")
            
            # 可以在这里实现缓存预热逻辑
            # 例如预先计算实体的邻居关系、子图等
            
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
            self.cache_stats["errors"] += 1

# 全局缓存管理器实例
_cache_manager_instance: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例（单例模式）"""
    global _cache_manager_instance
    
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
        await _cache_manager_instance.initialize()
    
    return _cache_manager_instance

async def close_cache_manager():
    """关闭缓存管理器"""
    global _cache_manager_instance
    
    if _cache_manager_instance:
        await _cache_manager_instance.close()
        _cache_manager_instance = None
