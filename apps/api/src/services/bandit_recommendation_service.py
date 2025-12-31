"""
多臂老虎机推荐服务

管理推荐引擎的初始化、配置和生命周期。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.ai.reinforcement_learning.recommendation_engine import (

    BanditRecommendationEngine,
    AlgorithmType,
    RecommendationRequest,
    FeedbackData
)

from src.core.logging import get_logger
logger = get_logger(__name__)

class BanditRecommendationService:
    """多臂老虎机推荐服务"""
    
    def __init__(self):
        """初始化推荐服务"""
        self.engine: Optional[BanditRecommendationEngine] = None
        self.is_initialized = False
        self.default_config = {
            "n_items": 1000,
            "default_algorithm": AlgorithmType.UCB,
            "enable_cold_start": True,
            "enable_evaluation": True,
            "cache_ttl_seconds": 300,
            "max_cache_size": 10000
        }
        self.algorithm_configs = {
            "ucb": {"c": 2.0, "random_state": 42},
            "thompson_sampling": {"alpha_init": 1.0, "beta_init": 1.0, "random_state": 42},
            "epsilon_greedy": {"epsilon": 0.1, "decay_rate": 0.995, "random_state": 42},
            "linear_contextual": {"n_features": 50, "alpha": 1.0, "lambda_reg": 1.0, "random_state": 42}
        }
        
    async def initialize(
        self,
        n_items: int = None,
        algorithm_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        初始化推荐引擎
        
        Args:
            n_items: 物品数量
            algorithm_configs: 算法配置
            **kwargs: 其他配置参数
            
        Returns:
            是否初始化成功
        """
        try:
            if self.is_initialized:
                logger.warning("推荐引擎已经初始化")
                return True
                
            # 合并配置
            config = self.default_config.copy()
            config.update(kwargs)
            
            if n_items:
                config["n_items"] = n_items
                
            # 创建推荐引擎
            self.engine = BanditRecommendationEngine(
                default_algorithm=config["default_algorithm"],
                enable_cold_start=config["enable_cold_start"],
                enable_evaluation=config["enable_evaluation"],
                cache_ttl_seconds=config["cache_ttl_seconds"],
                max_cache_size=config["max_cache_size"]
            )
            
            # 初始化算法
            algorithm_configs = algorithm_configs or self.algorithm_configs
            await self.engine.initialize_algorithms(config["n_items"], algorithm_configs)
            
            self.is_initialized = True
            logger.info(f"推荐引擎初始化成功，支持{config['n_items']}个物品")
            
            return True
            
        except Exception as e:
            logger.error(f"推荐引擎初始化失败: {e}")
            self.engine = None
            self.is_initialized = False
            return False
    
    async def get_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 10,
        context: Optional[Dict[str, Any]] = None,
        exclude_items: Optional[List[str]] = None,
        include_explanations: bool = False,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取推荐
        
        Args:
            user_id: 用户ID
            num_recommendations: 推荐数量
            context: 上下文信息
            exclude_items: 排除的物品
            include_explanations: 是否包含解释
            experiment_id: 实验ID
            
        Returns:
            推荐结果
        """
        if not self.is_initialized or not self.engine:
            raise RuntimeError("推荐引擎未初始化")
            
        request = RecommendationRequest(
            user_id=user_id,
            context=context,
            num_recommendations=num_recommendations,
            exclude_items=exclude_items,
            include_explanations=include_explanations,
            experiment_id=experiment_id
        )
        
        response = await self.engine.get_recommendations(request)
        
        return {
            "request_id": response.request_id,
            "user_id": response.user_id,
            "recommendations": response.recommendations,
            "algorithm_used": response.algorithm_used,
            "confidence_score": response.confidence_score,
            "cold_start_strategy": response.cold_start_strategy,
            "explanations": response.explanations,
            "timestamp": response.timestamp,
            "processing_time_ms": response.processing_time_ms
        }
    
    async def process_feedback(
        self,
        user_id: str,
        item_id: str,
        feedback_type: str,
        feedback_value: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        处理用户反馈
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            feedback_type: 反馈类型
            feedback_value: 反馈值
            context: 上下文信息
            
        Returns:
            是否处理成功
        """
        if not self.is_initialized or not self.engine:
            raise RuntimeError("推荐引擎未初始化")
            
        try:
            feedback = FeedbackData(
                user_id=user_id,
                item_id=item_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                context=context,
                timestamp=utc_now()
            )
            
            await self.engine.process_feedback(feedback)
            logger.info(f"处理用户{user_id}对物品{item_id}的反馈: {feedback_type}")
            return True
            
        except Exception as e:
            logger.error(f"处理反馈失败: {e}")
            return False
    
    async def update_user_context(self, user_id: str, context: Dict[str, Any]) -> bool:
        """
        更新用户上下文
        
        Args:
            user_id: 用户ID
            context: 上下文信息
            
        Returns:
            是否更新成功
        """
        if not self.is_initialized or not self.engine:
            raise RuntimeError("推荐引擎未初始化")
            
        try:
            await self.engine.update_user_context(user_id, context)
            return True
        except Exception as e:
            logger.error(f"更新用户上下文失败: {e}")
            return False
    
    async def update_item_features(self, item_id: str, features: Dict[str, Any]) -> bool:
        """
        更新物品特征
        
        Args:
            item_id: 物品ID
            features: 物品特征
            
        Returns:
            是否更新成功
        """
        if not self.is_initialized or not self.engine:
            raise RuntimeError("推荐引擎未初始化")
            
        try:
            await self.engine.update_item_features(item_id, features)
            return True
        except Exception as e:
            logger.error(f"更新物品特征失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.is_initialized or not self.engine:
            raise RuntimeError("推荐引擎未初始化")
            
        try:
            return self.engine.get_engine_statistics()
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态
        
        Returns:
            健康状态字典
        """
        status = {
            "service": "bandit_recommendation_service",
            "is_initialized": self.is_initialized,
            "timestamp": utc_now().isoformat()
        }
        
        if self.is_initialized and self.engine:
            try:
                stats = self.engine.get_engine_statistics()
                status.update({
                    "status": "healthy",
                    "engine_stats": {
                        "total_requests": stats["engine_stats"]["total_requests"],
                        "cache_hits": stats["engine_stats"]["cache_hits"],
                        "active_users": stats["active_users"],
                        "cache_size": stats["cache_size"],
                        "algorithms": list(stats["algorithm_stats"].keys())
                    }
                })
            except Exception as e:
                status.update({
                    "status": "degraded",
                    "error": str(e)
                })
        else:
            status["status"] = "not_initialized"
            
        return status
    
    async def shutdown(self):
        """关闭推荐服务"""
        if self.engine:
            # 这里可以添加保存状态、清理资源等逻辑
            logger.info("推荐引擎正在关闭")
            self.engine = None
            self.is_initialized = False

# 全局服务实例
bandit_recommendation_service = BanditRecommendationService()
