import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from redis.asyncio import Redis

from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    UserFeedback,
    UserProfile,
    RealTimeFeatures,
    ModelConfig
)
from ai.personalization.engine import PersonalizationEngine
from ai.personalization.features.realtime import FeatureConfig
from ai.personalization.cache.feature_cache import CacheConfig
from core.database import get_redis_client

logger = logging.getLogger(__name__)


class PersonalizationService:
    """个性化推荐服务"""
    
    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.engine: Optional[PersonalizationEngine] = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """确保服务已初始化"""
        if self._initialized:
            return
            
        self.redis_client = await get_redis_client()
        
        # 初始化个性化引擎
        feature_config = FeatureConfig()
        cache_config = CacheConfig()
        
        self.engine = PersonalizationEngine(
            redis_client=self.redis_client,
            feature_config=feature_config,
            cache_config=cache_config
        )
        
        # 启动引擎
        await self.engine.start()
        self._initialized = True
        logger.info("个性化服务初始化完成")
    
    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> RecommendationResponse:
        """获取个性化推荐
        
        Args:
            request: 推荐请求
            
        Returns:
            RecommendationResponse: 推荐响应
        """
        await self._ensure_initialized()
        return await self.engine.get_recommendations(request)
    
    async def process_feedback(self, feedback: UserFeedback):
        """处理用户反馈
        
        Args:
            feedback: 用户反馈
        """
        await self._ensure_initialized()
        await self.engine.process_feedback(feedback)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[UserProfile]: 用户画像
        """
        await self._ensure_initialized()
        return await self.engine._get_user_profile(user_id)
    
    async def update_user_profile(self, profile: UserProfile) -> bool:
        """更新用户画像
        
        Args:
            profile: 用户画像
            
        Returns:
            bool: 是否更新成功
        """
        try:
            await self._ensure_initialized()
            
            # 创建反馈用于更新画像
            feedback = UserFeedback(
                user_id=profile.user_id,
                item_id="profile_update",
                feedback_type="profile_update",
                feedback_value=profile.model_dump(),
                context={"action": "profile_update"}
            )
            
            await self.engine._update_user_profile(feedback)
            return True
            
        except Exception as e:
            logger.error(f"更新用户画像失败: {e}")
            return False
    
    async def get_realtime_features(self, user_id: str) -> Optional[RealTimeFeatures]:
        """获取实时特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[RealTimeFeatures]: 实时特征
        """
        await self._ensure_initialized()
        return await self.engine.feature_engine.compute_features(
            user_id=user_id,
            context={},
            use_cache=True
        )
    
    async def compute_features(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Optional[RealTimeFeatures]:
        """计算特征
        
        Args:
            user_id: 用户ID
            context: 上下文
            
        Returns:
            Optional[RealTimeFeatures]: 计算的特征
        """
        await self._ensure_initialized()
        return await self.engine.feature_engine.compute_features(
            user_id=user_id,
            context=context,
            use_cache=False
        )
    
    async def get_model_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """获取模型状态
        
        Args:
            model_id: 模型ID
            
        Returns:
            Dict[str, Any]: 模型状态
        """
        await self._ensure_initialized()
        return await self.engine.model_service.get_model_status(model_id)
    
    async def predict(
        self,
        features: Any,
        model_id: str
    ) -> Any:
        """模型预测
        
        Args:
            features: 特征
            model_id: 模型ID
            
        Returns:
            Any: 预测结果
        """
        await self._ensure_initialized()
        return await self.engine.model_service.predict(features, model_id)
    
    async def update_model(self, model_config: ModelConfig):
        """更新模型
        
        Args:
            model_config: 模型配置
        """
        await self._ensure_initialized()
        await self.engine.model_service.update_model(
            model_config.model_id,
            model_config.version,
            rollout_percentage=0.1
        )
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计
        
        Returns:
            Dict[str, Any]: 缓存统计
        """
        await self._ensure_initialized()
        return {
            "feature_cache": self.engine.feature_cache.get_stats() if hasattr(self.engine.feature_cache, 'get_stats') else {},
            "result_cache": self.engine.result_cache.get_stats() if hasattr(self.engine.result_cache, 'get_stats') else {},
            "feature_engine_stats": self.engine.feature_engine.get_feature_statistics()
        }
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """失效用户缓存
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 失效的缓存项数量
        """
        await self._ensure_initialized()
        
        count = 0
        try:
            # 失效特征缓存
            feature_key = f"features:{user_id}"
            if await self.redis_client.exists(feature_key):
                await self.redis_client.delete(feature_key)
                count += 1
            
            # 失效结果缓存
            await self.engine.result_cache.invalidate_user_cache(user_id)
            count += 1
            
        except Exception as e:
            logger.error(f"失效用户缓存失败: {e}")
        
        return count
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标
        
        Returns:
            Dict[str, Any]: 性能指标
        """
        await self._ensure_initialized()
        return self.engine.get_metrics()
    
    async def shutdown(self):
        """关闭服务"""
        if self.engine:
            await self.engine.stop()
        self._initialized = False
        logger.info("个性化服务已关闭")


# 全局服务实例
_personalization_service: Optional[PersonalizationService] = None


async def get_personalization_service() -> PersonalizationService:
    """获取个性化服务实例（依赖注入）
    
    Returns:
        PersonalizationService: 服务实例
    """
    global _personalization_service
    
    if _personalization_service is None:
        _personalization_service = PersonalizationService()
    
    return _personalization_service