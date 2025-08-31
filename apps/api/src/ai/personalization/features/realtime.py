import asyncio
import json
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
from redis.asyncio import Redis
import numpy as np
from collections import defaultdict, deque
import logging

from models.schemas.personalization import RealTimeFeatures
from .extractors import FeatureExtractor
from .aggregators import FeatureAggregator

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特征计算配置"""
    window_sizes: List[int] = None  # 滑动窗口大小（秒）
    update_interval: int = 60  # 更新间隔（秒）
    cache_ttl: int = 300  # 缓存过期时间（秒）
    max_history_size: int = 1000  # 最大历史记录数
    feature_weights: Dict[str, float] = None  # 特征权重
    enable_precompute: bool = True  # 启用预计算
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [60, 300, 1800, 3600]  # 1分钟，5分钟，30分钟，1小时
        if self.feature_weights is None:
            self.feature_weights = {
                "temporal": 0.3,
                "behavioral": 0.4,
                "contextual": 0.2,
                "aggregated": 0.1
            }


class RealTimeFeatureEngine:
    """实时特征计算引擎"""
    
    def __init__(self, redis_client: Redis, config: FeatureConfig = None):
        self.redis = redis_client
        self.config = config or FeatureConfig()
        self.feature_extractors: Dict[str, FeatureExtractor] = {}
        self.aggregators: Dict[str, FeatureAggregator] = {}
        self.feature_cache: Dict[str, Any] = {}
        self.sliding_windows: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(deque))
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # 初始化特征提取器和聚合器
        self._initialize_processors()
        
    def _initialize_processors(self):
        """初始化特征处理器"""
        # 默认特征提取器
        self.feature_extractors = {
            "temporal": FeatureExtractor("temporal"),
            "behavioral": FeatureExtractor("behavioral"),
            "contextual": FeatureExtractor("contextual")
        }
        
        # 默认聚合器
        self.aggregators = {
            "count": FeatureAggregator("count"),
            "avg": FeatureAggregator("average"),
            "sum": FeatureAggregator("sum"),
            "max": FeatureAggregator("max"),
            "std": FeatureAggregator("standard_deviation")
        }
    
    async def start(self):
        """启动特征引擎"""
        if self._running:
            return
            
        self._running = True
        logger.info("启动实时特征计算引擎")
        
        # 启动后台任务
        if self.config.enable_precompute:
            task = asyncio.create_task(self._precompute_features())
            self._background_tasks.append(task)
            
        # 启动缓存清理任务
        cleanup_task = asyncio.create_task(self._cleanup_expired_features())
        self._background_tasks.append(cleanup_task)
        
    async def stop(self):
        """停止特征引擎"""
        if not self._running:
            return
            
        self._running = False
        logger.info("停止实时特征计算引擎")
        
        # 取消所有后台任务
        for task in self._background_tasks:
            task.cancel()
            
        # 等待任务完成
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
    
    async def compute_features(
        self,
        user_id: str,
        context: Dict[str, Any],
        use_cache: bool = True
    ) -> RealTimeFeatures:
        """计算实时特征
        
        Args:
            user_id: 用户ID
            context: 上下文信息
            use_cache: 是否使用缓存
            
        Returns:
            RealTimeFeatures: 实时特征对象
        """
        start_time = time.time()
        
        # 尝试从缓存获取
        if use_cache:
            cached_features = await self.get_cached_features(user_id)
            if cached_features:
                logger.debug(f"用户 {user_id} 特征缓存命中")
                return cached_features
        
        try:
            # 并行计算各种特征
            tasks = [
                self._compute_temporal_features(user_id, context),
                self._compute_behavioral_features(user_id, context),
                self._compute_contextual_features(user_id, context)
            ]
            
            temporal_features, behavioral_features, contextual_features = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # 处理异常
            if isinstance(temporal_features, Exception):
                logger.error(f"时间特征计算失败: {temporal_features}")
                temporal_features = {}
            if isinstance(behavioral_features, Exception):
                logger.error(f"行为特征计算失败: {behavioral_features}")
                behavioral_features = {}
            if isinstance(contextual_features, Exception):
                logger.error(f"上下文特征计算失败: {contextual_features}")
                contextual_features = {}
            
            # 计算聚合特征
            aggregated_features = await self._compute_aggregated_features(
                user_id, temporal_features, behavioral_features, contextual_features
            )
            
            # 构建特征对象
            features = RealTimeFeatures(
                temporal=temporal_features or {},
                behavioral=behavioral_features or {},
                contextual=contextual_features or {},
                aggregated=aggregated_features,
                timestamp=utc_now()
            )
            
            # 缓存特征
            if use_cache:
                await self._cache_features(user_id, features)
            
            # 更新滑动窗口
            await self._update_sliding_windows(user_id, features)
            
            computation_time = (time.time() - start_time) * 1000
            logger.debug(f"用户 {user_id} 特征计算完成，耗时 {computation_time:.2f}ms")
            
            return features
            
        except Exception as e:
            logger.error(f"特征计算失败 user_id={user_id}: {e}", exc_info=True)
            # 返回空特征而不是抛出异常
            return RealTimeFeatures(
                timestamp=utc_now()
            )
    
    async def get_cached_features(self, user_id: str) -> Optional[RealTimeFeatures]:
        """获取缓存特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[RealTimeFeatures]: 缓存的特征对象，如果不存在则返回None
        """
        try:
            cache_key = f"features:{user_id}"
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                feature_dict = json.loads(cached_data)
                # 检查特征是否过期
                timestamp = datetime.fromisoformat(feature_dict["timestamp"])
                if utc_now() - timestamp < timedelta(seconds=self.config.cache_ttl):
                    return RealTimeFeatures(**feature_dict)
                else:
                    # 删除过期特征
                    await self.redis.delete(cache_key)
                    
            return None
            
        except Exception as e:
            logger.error(f"获取缓存特征失败 user_id={user_id}: {e}")
            return None
    
    async def _cache_features(self, user_id: str, features: RealTimeFeatures):
        """缓存特征"""
        try:
            cache_key = f"features:{user_id}"
            feature_dict = features.model_dump()
            feature_dict["timestamp"] = features.timestamp.isoformat()
            
            await self.redis.setex(
                cache_key,
                self.config.cache_ttl,
                json.dumps(feature_dict, default=str)
            )
            
        except Exception as e:
            logger.error(f"缓存特征失败 user_id={user_id}: {e}")
    
    async def _compute_temporal_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算时间特征"""
        try:
            extractor = self.feature_extractors["temporal"]
            return await extractor.extract_features(user_id, context)
        except Exception as e:
            logger.error(f"时间特征计算失败: {e}")
            return {}
    
    async def _compute_behavioral_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算行为特征"""
        try:
            extractor = self.feature_extractors["behavioral"]
            return await extractor.extract_features(user_id, context)
        except Exception as e:
            logger.error(f"行为特征计算失败: {e}")
            return {}
    
    async def _compute_contextual_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算上下文特征"""
        try:
            extractor = self.feature_extractors["contextual"]
            return await extractor.extract_features(user_id, context)
        except Exception as e:
            logger.error(f"上下文特征计算失败: {e}")
            return {}
    
    async def _compute_aggregated_features(
        self,
        user_id: str,
        temporal: Dict[str, float],
        behavioral: Dict[str, float],
        contextual: Dict[str, float]
    ) -> Dict[str, float]:
        """计算聚合特征"""
        aggregated = {}
        
        try:
            # 特征权重聚合
            weights = self.config.feature_weights
            all_features = {
                **{f"temporal_{k}": v * weights["temporal"] for k, v in temporal.items()},
                **{f"behavioral_{k}": v * weights["behavioral"] for k, v in behavioral.items()},
                **{f"contextual_{k}": v * weights["contextual"] for k, v in contextual.items()}
            }
            
            # 计算统计聚合特征
            if all_features:
                feature_values = list(all_features.values())
                aggregated.update({
                    "feature_count": len(feature_values),
                    "feature_mean": np.mean(feature_values),
                    "feature_std": np.std(feature_values),
                    "feature_max": np.max(feature_values),
                    "feature_min": np.min(feature_values)
                })
            
            # 滑动窗口聚合特征
            window_features = await self._compute_window_aggregations(user_id)
            aggregated.update(window_features)
            
        except Exception as e:
            logger.error(f"聚合特征计算失败: {e}")
            
        return aggregated
    
    async def _compute_window_aggregations(self, user_id: str) -> Dict[str, float]:
        """计算滑动窗口聚合特征"""
        window_features = {}
        
        try:
            user_windows = self.sliding_windows.get(user_id, {})
            
            for window_size, window_data in user_windows.items():
                if len(window_data) == 0:
                    continue
                    
                # 计算窗口内特征的统计值
                window_values = list(window_data)
                prefix = f"window_{window_size}s"
                
                if window_values:
                    window_features.update({
                        f"{prefix}_count": len(window_values),
                        f"{prefix}_avg": np.mean(window_values),
                        f"{prefix}_sum": np.sum(window_values),
                        f"{prefix}_max": np.max(window_values),
                        f"{prefix}_std": np.std(window_values) if len(window_values) > 1 else 0.0
                    })
                    
        except Exception as e:
            logger.error(f"窗口聚合计算失败: {e}")
            
        return window_features
    
    async def _update_sliding_windows(self, user_id: str, features: RealTimeFeatures):
        """更新滑动窗口数据"""
        try:
            current_time = time.time()
            
            # 计算特征综合得分
            all_features = {**features.temporal, **features.behavioral, **features.contextual}
            if all_features:
                feature_score = np.mean(list(all_features.values()))
            else:
                feature_score = 0.0
            
            # 更新各个窗口
            for window_size in self.config.window_sizes:
                window = self.sliding_windows[user_id][window_size]
                window.append((current_time, feature_score))
                
                # 清理过期数据
                cutoff_time = current_time - window_size
                while window and window[0][0] < cutoff_time:
                    window.popleft()
                    
                # 限制窗口大小
                max_size = min(self.config.max_history_size, window_size)
                while len(window) > max_size:
                    window.popleft()
                    
        except Exception as e:
            logger.error(f"更新滑动窗口失败: {e}")
    
    async def _precompute_features(self):
        """预计算特征后台任务"""
        while self._running:
            try:
                await asyncio.sleep(self.config.update_interval)
                
                # 获取活跃用户列表（这里可以从数据库或缓存中获取）
                active_users = await self._get_active_users()
                
                # 为活跃用户预计算特征
                for user_id in active_users:
                    try:
                        await self.compute_features(
                            user_id=user_id,
                            context={"precompute": True},
                            use_cache=False
                        )
                        await asyncio.sleep(0.1)  # 避免过载
                    except Exception as e:
                        logger.error(f"用户 {user_id} 预计算特征失败: {e}")
                        
            except Exception as e:
                logger.error(f"预计算特征任务失败: {e}")
    
    async def _cleanup_expired_features(self):
        """清理过期特征后台任务"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次
                
                # 清理过期的滑动窗口数据
                current_time = time.time()
                for user_id, windows in list(self.sliding_windows.items()):
                    for window_size, window_data in list(windows.items()):
                        cutoff_time = current_time - window_size - 3600  # 额外保留1小时
                        while window_data and window_data[0][0] < cutoff_time:
                            window_data.popleft()
                        
                        # 如果窗口为空且用户不活跃，删除窗口
                        if not window_data:
                            del windows[window_size]
                    
                    # 如果用户没有任何窗口，删除用户记录
                    if not windows:
                        del self.sliding_windows[user_id]
                        
            except Exception as e:
                logger.error(f"清理过期特征失败: {e}")
    
    async def _get_active_users(self) -> List[str]:
        """获取活跃用户列表"""
        try:
            # 从Redis获取活跃用户（可以基于最近的活动记录）
            active_users_key = "active_users"
            users = await self.redis.smembers(active_users_key)
            return [user.decode() if isinstance(user, bytes) else user for user in users]
        except Exception as e:
            logger.error(f"获取活跃用户失败: {e}")
            return []
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """获取特征统计信息"""
        stats = {
            "total_users": len(self.sliding_windows),
            "window_sizes": self.config.window_sizes,
            "cache_ttl": self.config.cache_ttl,
            "is_running": self._running,
            "extractors": list(self.feature_extractors.keys()),
            "aggregators": list(self.aggregators.keys())
        }
        
        # 计算每个窗口的数据量
        window_stats = {}
        for user_id, windows in self.sliding_windows.items():
            for window_size, window_data in windows.items():
                key = f"window_{window_size}s"
                if key not in window_stats:
                    window_stats[key] = {"total_entries": 0, "users": 0}
                window_stats[key]["total_entries"] += len(window_data)
                window_stats[key]["users"] += 1
        
        stats["window_statistics"] = window_stats
        return stats