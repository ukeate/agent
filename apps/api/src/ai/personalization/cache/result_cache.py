import json
import hashlib
from typing import Dict, Optional, Any, List
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from redis.asyncio import Redis
import asyncio
from src.models.schemas.personalization import RecommendationResponse, RecommendationRequest

from src.core.logging import get_logger
logger = get_logger(__name__)

class ResultCacheManager:
    """推荐结果缓存管理器"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 300  # 5分钟
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0
        }
        
    def _generate_cache_key(
        self,
        request: RecommendationRequest,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成缓存键
        
        Args:
            request: 推荐请求
            extra_params: 额外参数
            
        Returns:
            str: 缓存键
        """
        # 构建键的组成部分
        key_parts = {
            "user_id": request.user_id,
            "scenario": request.scenario.value,
            "n_recommendations": request.n_recommendations,
            "filters": json.dumps(request.filters or {}, sort_keys=True),
            "diversity_weight": request.diversity_weight
        }
        
        # 添加上下文的哈希（避免键过长）
        if request.context:
            context_str = json.dumps(request.context, sort_keys=True, default=str)
            context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]
            key_parts["context_hash"] = context_hash
        
        # 添加额外参数
        if extra_params:
            for k, v in extra_params.items():
                key_parts[k] = str(v)
        
        # 生成最终的键
        key_str = json.dumps(key_parts, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        
        return f"recommendation:result:{request.user_id}:{key_hash}"
    
    async def get_cached_result(
        self,
        request: RecommendationRequest
    ) -> Optional[RecommendationResponse]:
        """获取缓存的推荐结果
        
        Args:
            request: 推荐请求
            
        Returns:
            Optional[RecommendationResponse]: 缓存的推荐结果
        """
        if not request.use_cache:
            return None
            
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                # 解析缓存数据
                data = json.loads(cached_data)
                
                # 转换为响应对象
                response = RecommendationResponse(**data)
                
                # 更新缓存命中标记
                response.cache_hit = True
                
                # 更新统计
                self.cache_stats["hits"] += 1
                
                logger.debug(f"推荐结果缓存命中: {cache_key}")
                return response
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return None
    
    async def cache_result(
        self,
        request: RecommendationRequest,
        response: RecommendationResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """缓存推荐结果
        
        Args:
            request: 推荐请求
            response: 推荐响应
            ttl: 缓存过期时间
            
        Returns:
            bool: 是否缓存成功
        """
        if not request.use_cache:
            return False
            
        try:
            cache_key = self._generate_cache_key(request)
            ttl = ttl or self.default_ttl
            
            # 序列化响应
            response_dict = response.model_dump()
            # 确保时间戳可序列化
            response_dict["timestamp"] = response.timestamp.isoformat()
            
            # 设置缓存
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(response_dict, default=str)
            )
            
            # 添加到用户的缓存索引（用于批量失效）
            index_key = f"recommendation:index:{request.user_id}"
            await self.redis.sadd(index_key, cache_key)
            await self.redis.expire(index_key, ttl)
            
            logger.debug(f"推荐结果已缓存: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"缓存结果失败: {e}")
            return False
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """失效用户的所有缓存
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 失效的缓存数量
        """
        try:
            # 获取用户的所有缓存键
            index_key = f"recommendation:index:{user_id}"
            cache_keys = await self.redis.smembers(index_key)
            
            if cache_keys:
                # 删除所有缓存
                cache_keys_list = list(cache_keys)
                deleted_count = await self.redis.delete(*cache_keys_list)
                
                # 删除索引
                await self.redis.delete(index_key)
                
                # 更新统计
                self.cache_stats["invalidations"] += deleted_count
                
                logger.info(f"失效用户 {user_id} 的 {deleted_count} 个缓存项")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"失效用户缓存失败 user_id={user_id}: {e}")
            return 0
    
    async def invalidate_scenario_cache(
        self,
        scenario: str,
        user_ids: Optional[List[str]] = None
    ) -> int:
        """失效特定场景的缓存
        
        Args:
            scenario: 推荐场景
            user_ids: 用户ID列表（可选，如果不提供则失效所有用户）
            
        Returns:
            int: 失效的缓存数量
        """
        try:
            deleted_count = 0
            
            if user_ids:
                # 失效特定用户的场景缓存
                for user_id in user_ids:
                    pattern = f"recommendation:result:{user_id}:*"
                    cursor = 0
                    
                    while True:
                        cursor, keys = await self.redis.scan(
                            cursor=cursor,
                            match=pattern,
                            count=100
                        )
                        
                        # 筛选包含特定场景的键
                        for key in keys:
                            key_str = key.decode() if isinstance(key, bytes) else key
                            if scenario in key_str:
                                if await self.redis.delete(key_str):
                                    deleted_count += 1
                        
                        if cursor == 0:
                            break
            else:
                # 失效所有用户的场景缓存
                pattern = f"recommendation:result:*"
                cursor = 0
                
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100
                    )
                    
                    for key in keys:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        if scenario in key_str:
                            if await self.redis.delete(key_str):
                                deleted_count += 1
                    
                    if cursor == 0:
                        break
            
            # 更新统计
            self.cache_stats["invalidations"] += deleted_count
            
            logger.info(f"失效场景 {scenario} 的 {deleted_count} 个缓存项")
            return deleted_count
            
        except Exception as e:
            logger.error(f"失效场景缓存失败 scenario={scenario}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
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
        
        # 获取缓存大小
        try:
            pattern = "recommendation:result:*"
            cursor, keys = await self.redis.scan(
                cursor=0,
                match=pattern,
                count=1000
            )
            stats["cache_size"] = len(keys)
        except Exception as e:
            logger.error(f"获取缓存大小失败: {e}")
            stats["cache_size"] = -1
        
        return stats
    
    async def warm_up_cache(
        self,
        user_ids: List[str],
        scenarios: List[str],
        recommendation_func: callable
    ) -> int:
        """预热缓存
        
        Args:
            user_ids: 用户ID列表
            scenarios: 场景列表
            recommendation_func: 获取推荐的函数
            
        Returns:
            int: 预热的缓存数量
        """
        warmed_count = 0
        
        for user_id in user_ids:
            for scenario in scenarios:
                try:
                    # 构建请求
                    request = RecommendationRequest(
                        user_id=user_id,
                        scenario=scenario,
                        use_cache=False  # 强制生成新的推荐
                    )
                    
                    # 检查是否已有缓存
                    cache_key = self._generate_cache_key(request)
                    if await self.redis.exists(cache_key):
                        continue
                    
                    # 生成推荐
                    response = await recommendation_func(request)
                    
                    if response:
                        # 缓存结果
                        request.use_cache = True  # 启用缓存以保存结果
                        if await self.cache_result(request, response):
                            warmed_count += 1
                        
                        # 避免过载
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"预热缓存失败 user_id={user_id}, scenario={scenario}: {e}")
        
        logger.info(f"预热了 {warmed_count} 个缓存项")
        return warmed_count

class CacheInvalidator:
    """缓存失效器"""
    
    def __init__(self, result_cache: ResultCacheManager):
        self.result_cache = result_cache
        self.invalidation_rules = []
        
    def add_rule(self, rule_func: callable):
        """添加失效规则
        
        Args:
            rule_func: 规则函数，接收事件并返回需要失效的用户ID列表
        """
        self.invalidation_rules.append(rule_func)
    
    async def process_event(self, event: Dict[str, Any]):
        """处理事件并执行缓存失效
        
        Args:
            event: 事件数据
        """
        try:
            # 应用所有规则
            user_ids_to_invalidate = set()
            
            for rule_func in self.invalidation_rules:
                user_ids = await rule_func(event)
                if user_ids:
                    user_ids_to_invalidate.update(user_ids)
            
            # 执行失效
            for user_id in user_ids_to_invalidate:
                await self.result_cache.invalidate_user_cache(user_id)
                
        except Exception as e:
            logger.error(f"处理失效事件失败: {e}")
    
    @staticmethod
    async def user_update_rule(event: Dict[str, Any]) -> List[str]:
        """用户更新规则"""
        if event.get("type") == "user_update":
            return [event.get("user_id")]
        return []
    
    @staticmethod
    async def item_update_rule(event: Dict[str, Any]) -> List[str]:
        """物品更新规则"""
        if event.get("type") == "item_update":
            # 获取与该物品相关的用户
            # 这里简化处理，实际应该查询数据库
            return event.get("affected_users", [])
        return []
    
    @staticmethod
    async def model_update_rule(event: Dict[str, Any]) -> List[str]:
        """模型更新规则"""
        if event.get("type") == "model_update":
            # 模型更新时失效所有缓存
            return ["*"]  # 特殊标记，表示所有用户
        return []
