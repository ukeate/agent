import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from redis.asyncio import Redis
import numpy as np
import logging

from models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    UserProfile,
    UserFeedback
)
from .features.realtime import RealTimeFeatureEngine, FeatureConfig
from .models.service import DistributedModelService
from .cache.feature_cache import FeatureCacheManager, CacheConfig
from .cache.result_cache import ResultCacheManager
from ai.reinforcement_learning.bandits.contextual import ContextualBandit
from ai.reinforcement_learning.qlearning.q_learning import QLearningAgent

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """个性化推荐引擎"""
    
    def __init__(
        self,
        redis_client: Redis,
        feature_config: Optional[FeatureConfig] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        self.redis = redis_client
        
        # 初始化组件
        self.feature_engine = RealTimeFeatureEngine(redis_client, feature_config)
        self.model_service = DistributedModelService()
        self.feature_cache = FeatureCacheManager(redis_client, cache_config)
        self.result_cache = ResultCacheManager(redis_client)
        
        # 初始化强化学习组件（复用Story 6.1和6.2的实现）
        try:
            # 尝试初始化多臂老虎机管理器
            self.bandit_manager = ContextualBandit(
                n_arms=100,  # 支持100个推荐项
                context_dim=feature_config.feature_weights if feature_config else 64
            )
        except Exception as e:
            logger.warning(f"多臂老虎机初始化失败: {e}")
            self.bandit_manager = None
        
        try:
            # 尝试初始化Q-Learning智能体
            self.qlearning_agent = QLearningAgent(
                state_space_size=10,  # 简化状态空间
                action_space_size=100,  # 100个可能的推荐动作
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=0.1
            )
        except Exception as e:
            logger.warning(f"Q-Learning智能体初始化失败: {e}")
            self.qlearning_agent = None
        
        # 性能监控
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "error_rate": 0.0
        }
        
        # 降级策略配置
        self.fallback_enabled = True
        self.circuit_breaker_threshold = 0.5  # 50%错误率触发熔断
        self.circuit_breaker_window = 60  # 60秒窗口
        self.circuit_breaker_errors = []
        self.circuit_breaker_open = False
        
        self._running = False
        
    async def start(self):
        """启动个性化引擎"""
        if self._running:
            return
            
        self._running = True
        logger.info("启动个性化推荐引擎")
        
        # 启动特征引擎
        await self.feature_engine.start()
        
        # 初始化模型服务
        await self.model_service.initialize()
        
        # 预热缓存
        asyncio.create_task(self._warm_up_cache())
        
    async def stop(self):
        """停止个性化引擎"""
        if not self._running:
            return
            
        self._running = False
        logger.info("停止个性化推荐引擎")
        
        # 停止特征引擎
        await self.feature_engine.stop()
        
        # 关闭模型服务
        await self.model_service.shutdown()
    
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
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # 更新请求计数
            self.metrics["total_requests"] += 1
            
            # 检查熔断器状态
            if self.circuit_breaker_open:
                logger.warning("熔断器开启，使用降级策略")
                return await self._get_fallback_recommendations(request, request_id)
            
            # 尝试从缓存获取结果
            if request.use_cache:
                cached_response = await self.result_cache.get_cached_result(request)
                if cached_response:
                    self.metrics["cache_hits"] += 1
                    cached_response.request_id = request_id
                    return cached_response
            
            # 获取用户画像
            user_profile = await self._get_user_profile(request.user_id)
            
            # 计算实时特征
            features = await self.feature_engine.compute_features(
                user_id=request.user_id,
                context={
                    **request.context,
                    "user_profile": user_profile.model_dump() if user_profile else {},
                    "scenario": request.scenario
                }
            )
            
            # 准备特征向量
            feature_vector = self._prepare_feature_vector(features, user_profile)
            
            # 获取候选集
            candidates = await self._get_candidates(request, user_profile)
            
            if not candidates:
                logger.warning(f"用户 {request.user_id} 没有候选推荐项")
                return self._create_empty_response(request, request_id)
            
            # 批量预测分数
            scores = await self._predict_scores(
                feature_vector,
                candidates,
                request.scenario
            )
            
            # 排序和过滤
            recommendations = await self._rank_and_filter(
                candidates,
                scores,
                request
            )
            
            # 应用多样性策略
            if request.diversity_weight > 0:
                recommendations = await self._apply_diversity(
                    recommendations,
                    request.diversity_weight
                )
            
            # 限制返回数量
            recommendations = recommendations[:request.n_recommendations]
            
            # 生成解释
            explanations = await self._generate_explanations(
                recommendations,
                features,
                user_profile
            )
            
            # 构建响应
            response = RecommendationResponse(
                request_id=request_id,
                user_id=request.user_id,
                recommendations=recommendations,
                latency_ms=(time.time() - start_time) * 1000,
                model_version=await self.model_service.get_active_version(),
                scenario=request.scenario,
                explanation=explanations.get("overall", ""),
                debug_info={
                    "feature_count": len(feature_vector) if feature_vector else 0,
                    "candidate_count": len(candidates),
                    "cache_hit": False
                }
            )
            
            # 缓存结果
            if request.use_cache:
                await self.result_cache.cache_result(request, response)
            
            # 更新性能指标
            self._update_metrics(response.latency_ms, success=True)
            
            return response
            
        except Exception as e:
            logger.error(f"获取推荐失败 user_id={request.user_id}: {e}", exc_info=True)
            
            # 记录错误
            self._record_error()
            
            # 使用降级策略
            if self.fallback_enabled:
                return await self._get_fallback_recommendations(request, request_id)
            
            # 返回空响应
            return self._create_empty_response(request, request_id)
    
    async def process_feedback(
        self,
        feedback: UserFeedback
    ):
        """处理用户反馈进行在线学习
        
        Args:
            feedback: 用户反馈
        """
        try:
            # 更新用户画像
            await self._update_user_profile(feedback)
            
            # 更新强化学习模型
            if self.bandit_manager:
                await self.bandit_manager.update(
                    user_id=feedback.user_id,
                    item_id=feedback.item_id,
                    reward=self._calculate_reward(feedback)
                )
            
            if self.qlearning_agent:
                state = await self._get_user_state(feedback.user_id)
                action = feedback.item_id
                reward = self._calculate_reward(feedback)
                next_state = await self._get_user_state(feedback.user_id)
                
                await self.qlearning_agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state
                )
            
            # 失效相关缓存
            await self.result_cache.invalidate_user_cache(feedback.user_id)
            
            logger.debug(f"处理用户反馈: user_id={feedback.user_id}, item_id={feedback.item_id}")
            
        except Exception as e:
            logger.error(f"处理反馈失败: {e}")
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[UserProfile]: 用户画像
        """
        try:
            # 从缓存或数据库获取
            profile_key = f"user_profile:{user_id}"
            profile_data = await self.redis.get(profile_key)
            
            if profile_data:
                import json
                data = json.loads(profile_data)
                # 转换时间戳
                if "last_updated" in data:
                    data["last_updated"] = datetime.fromisoformat(data["last_updated"])
                return UserProfile(**data)
            
            # 创建默认画像
            return UserProfile(
                user_id=user_id,
                features={},
                preferences={},
                behavior_history=[],
                last_updated=utc_now()
            )
            
        except Exception as e:
            logger.error(f"获取用户画像失败 user_id={user_id}: {e}")
            return None
    
    async def _update_user_profile(self, feedback: UserFeedback):
        """更新用户画像
        
        Args:
            feedback: 用户反馈
        """
        try:
            profile = await self._get_user_profile(feedback.user_id) or UserProfile(
                user_id=feedback.user_id
            )
            
            # 更新行为历史
            profile.behavior_history.append({
                "item_id": feedback.item_id,
                "feedback_type": feedback.feedback_type,
                "feedback_value": feedback.feedback_value,
                "timestamp": feedback.timestamp.isoformat()
            })
            
            # 限制历史长度
            if len(profile.behavior_history) > 100:
                profile.behavior_history = profile.behavior_history[-100:]
            
            # 更新偏好
            if feedback.feedback_type in ["like", "purchase", "click"]:
                item_type = feedback.context.get("item_type", "unknown")
                profile.preferences[item_type] = profile.preferences.get(item_type, 0) + 1
            
            profile.last_updated = utc_now()
            
            # 保存到缓存
            import json
            profile_key = f"user_profile:{feedback.user_id}"
            profile_dict = profile.model_dump()
            profile_dict["last_updated"] = profile.last_updated.isoformat()
            
            await self.redis.setex(
                profile_key,
                86400,  # 24小时过期
                json.dumps(profile_dict, default=str)
            )
            
        except Exception as e:
            logger.error(f"更新用户画像失败: {e}")
    
    def _prepare_feature_vector(
        self,
        features: Any,
        user_profile: Optional[UserProfile]
    ) -> np.ndarray:
        """准备特征向量
        
        Args:
            features: 实时特征
            user_profile: 用户画像
            
        Returns:
            np.ndarray: 特征向量
        """
        feature_list = []
        
        # 添加实时特征
        if features:
            for feature_dict in [features.temporal, features.behavioral, 
                               features.contextual, features.aggregated]:
                feature_list.extend(feature_dict.values())
        
        # 添加用户画像特征
        if user_profile:
            feature_list.extend(user_profile.features.values())
            feature_list.extend(user_profile.preferences.values())
        
        # 确保有特征
        if not feature_list:
            feature_list = [0.0]
        
        return np.array(feature_list, dtype=np.float32)
    
    async def _get_candidates(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> List[Dict[str, Any]]:
        """获取候选推荐项
        
        Args:
            request: 推荐请求
            user_profile: 用户画像
            
        Returns:
            List[Dict[str, Any]]: 候选列表
        """
        # 这里简化处理，实际应该从数据库或搜索引擎获取
        candidates = []
        
        try:
            # 根据场景获取候选
            if request.scenario == "content":
                # 内容推荐候选
                candidates = await self._get_content_candidates(request, user_profile)
            elif request.scenario == "product":
                # 产品推荐候选
                candidates = await self._get_product_candidates(request, user_profile)
            else:
                # 默认候选
                candidates = await self._get_default_candidates(request)
            
            # 应用过滤器
            if request.filters:
                candidates = self._apply_filters(candidates, request.filters)
            
        except Exception as e:
            logger.error(f"获取候选失败: {e}")
            
        return candidates
    
    async def _get_content_candidates(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> List[Dict[str, Any]]:
        """获取内容候选"""
        # 模拟从数据库获取
        return [
            {"item_id": f"content_{i}", "type": "article", "score": 0.0}
            for i in range(100)
        ]
    
    async def _get_product_candidates(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> List[Dict[str, Any]]:
        """获取产品候选"""
        # 模拟从数据库获取
        return [
            {"item_id": f"product_{i}", "type": "product", "score": 0.0}
            for i in range(100)
        ]
    
    async def _get_default_candidates(
        self,
        request: RecommendationRequest
    ) -> List[Dict[str, Any]]:
        """获取默认候选"""
        return [
            {"item_id": f"item_{i}", "type": "default", "score": 0.0}
            for i in range(50)
        ]
    
    def _apply_filters(
        self,
        candidates: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """应用过滤器"""
        filtered = []
        
        for candidate in candidates:
            include = True
            
            for key, value in filters.items():
                if key in candidate and candidate[key] != value:
                    include = False
                    break
            
            if include:
                filtered.append(candidate)
        
        return filtered
    
    async def _predict_scores(
        self,
        feature_vector: np.ndarray,
        candidates: List[Dict[str, Any]],
        scenario: str
    ) -> np.ndarray:
        """预测候选分数
        
        Args:
            feature_vector: 特征向量
            candidates: 候选列表
            scenario: 推荐场景
            
        Returns:
            np.ndarray: 分数数组
        """
        try:
            # 批量预测 - 修复：使用正确的接口
            scores = await self.model_service.batch_predict(
                features=feature_vector,
                candidates=candidates,
                model_id=f"model_{scenario}",
                batch_size=32  # 添加批次大小参数
            )
            
            return scores
            
        except Exception as e:
            logger.error(f"预测分数失败: {e}")
            # 返回随机分数作为后备
            return np.random.rand(len(candidates))
    
    async def _rank_and_filter(
        self,
        candidates: List[Dict[str, Any]],
        scores: np.ndarray,
        request: RecommendationRequest
    ) -> List[RecommendationItem]:
        """排序和过滤
        
        Args:
            candidates: 候选列表
            scores: 分数数组
            request: 推荐请求
            
        Returns:
            List[RecommendationItem]: 推荐项列表
        """
        # 组合候选和分数
        scored_candidates = list(zip(candidates, scores))
        
        # 按分数降序排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 转换为推荐项
        recommendations = []
        for candidate, score in scored_candidates[:request.n_recommendations * 2]:
            item = RecommendationItem(
                item_id=candidate["item_id"],
                score=float(score),
                confidence=min(0.9, float(score)),  # 简化的置信度计算
                metadata=candidate
            )
            recommendations.append(item)
        
        return recommendations
    
    async def _apply_diversity(
        self,
        recommendations: List[RecommendationItem],
        diversity_weight: float
    ) -> List[RecommendationItem]:
        """应用多样性策略
        
        Args:
            recommendations: 推荐项列表
            diversity_weight: 多样性权重
            
        Returns:
            List[RecommendationItem]: 多样化后的推荐列表
        """
        if len(recommendations) <= 1:
            return recommendations
        
        # 简单的多样性算法：降低相似项的分数
        diverse_recommendations = [recommendations[0]]
        
        for item in recommendations[1:]:
            # 计算与已选项的相似度
            similarity = 0.0
            for selected in diverse_recommendations:
                # 简化：基于类型的相似度
                if item.metadata.get("type") == selected.metadata.get("type"):
                    similarity += 0.5
            
            # 调整分数
            adjusted_score = item.score * (1 - diversity_weight * similarity)
            item.score = adjusted_score
            
            diverse_recommendations.append(item)
        
        # 重新排序
        diverse_recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return diverse_recommendations
    
    async def _generate_explanations(
        self,
        recommendations: List[RecommendationItem],
        features: Any,
        user_profile: Optional[UserProfile]
    ) -> Dict[str, str]:
        """生成推荐解释
        
        Args:
            recommendations: 推荐项列表
            features: 特征
            user_profile: 用户画像
            
        Returns:
            Dict[str, str]: 解释字典
        """
        explanations = {}
        
        # 整体解释
        explanations["overall"] = "基于您的偏好和行为历史生成的个性化推荐"
        
        # 为每个推荐项生成解释
        for item in recommendations[:3]:  # 只为前3个生成详细解释
            if user_profile and user_profile.preferences:
                # 基于偏好的解释
                top_pref = max(user_profile.preferences, key=user_profile.preferences.get)
                item.explanation = f"基于您对{top_pref}的偏好推荐"
            else:
                # 默认解释
                item.explanation = f"推荐分数: {item.score:.2f}"
        
        return explanations
    
    async def _get_fallback_recommendations(
        self,
        request: RecommendationRequest,
        request_id: str
    ) -> RecommendationResponse:
        """获取降级推荐
        
        Args:
            request: 推荐请求
            request_id: 请求ID
            
        Returns:
            RecommendationResponse: 降级响应
        """
        try:
            # 从热门推荐池获取
            popular_key = f"popular_items:{request.scenario}"
            popular_items = await self.redis.zrevrange(
                popular_key,
                0,
                request.n_recommendations - 1,
                withscores=True
            )
            
            recommendations = []
            for item_id, score in popular_items:
                item_id_str = item_id.decode() if isinstance(item_id, bytes) else item_id
                recommendations.append(
                    RecommendationItem(
                        item_id=item_id_str,
                        score=float(score),
                        confidence=0.5,
                        explanation="热门推荐",
                        metadata={"fallback": True}
                    )
                )
            
            return RecommendationResponse(
                request_id=request_id,
                user_id=request.user_id,
                recommendations=recommendations,
                latency_ms=0.0,
                model_version="fallback",
                scenario=request.scenario,
                explanation="使用热门推荐（降级策略）",
                debug_info={"fallback": True}
            )
            
        except Exception as e:
            logger.error(f"降级推荐失败: {e}")
            return self._create_empty_response(request, request_id)
    
    def _create_empty_response(
        self,
        request: RecommendationRequest,
        request_id: str
    ) -> RecommendationResponse:
        """创建空响应"""
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=[],
            latency_ms=0.0,
            model_version="none",
            scenario=request.scenario,
            explanation="暂无推荐",
            debug_info={"empty": True}
        )
    
    def _calculate_reward(self, feedback: UserFeedback) -> float:
        """计算奖励值
        
        Args:
            feedback: 用户反馈
            
        Returns:
            float: 奖励值
        """
        reward_map = {
            "click": 0.1,
            "view": 0.05,
            "like": 0.3,
            "share": 0.5,
            "comment": 0.4,
            "purchase": 1.0,
            "dislike": -0.5,
            "report": -1.0
        }
        
        return reward_map.get(feedback.feedback_type, 0.0)
    
    async def _get_user_state(self, user_id: str) -> str:
        """获取用户状态（用于Q-Learning）
        
        Args:
            user_id: 用户ID
            
        Returns:
            str: 状态标识
        """
        # 简化：基于用户活跃度返回状态
        profile = await self._get_user_profile(user_id)
        if profile and profile.behavior_history:
            activity_level = len(profile.behavior_history)
            if activity_level > 50:
                return "high_activity"
            elif activity_level > 10:
                return "medium_activity"
        return "low_activity"
    
    def _record_error(self):
        """记录错误（用于熔断器）"""
        current_time = time.time()
        self.circuit_breaker_errors.append(current_time)
        
        # 清理过期错误
        cutoff_time = current_time - self.circuit_breaker_window
        self.circuit_breaker_errors = [
            t for t in self.circuit_breaker_errors if t > cutoff_time
        ]
        
        # 检查是否触发熔断
        error_rate = len(self.circuit_breaker_errors) / max(self.metrics["total_requests"], 1)
        if error_rate > self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            logger.warning(f"熔断器开启: 错误率 {error_rate:.2%}")
            
            # 设置恢复任务
            asyncio.create_task(self._recover_circuit_breaker())
    
    async def _recover_circuit_breaker(self):
        """恢复熔断器"""
        await asyncio.sleep(30)  # 30秒后尝试恢复
        self.circuit_breaker_open = False
        logger.info("熔断器恢复")
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """更新性能指标
        
        Args:
            latency_ms: 延迟（毫秒）
            success: 是否成功
        """
        # 更新平均延迟（简单移动平均）
        self.metrics["avg_latency_ms"] = (
            self.metrics["avg_latency_ms"] * 0.9 + latency_ms * 0.1
        )
        
        # 更新P99（简化：使用最大值的90%）
        if latency_ms > self.metrics["p99_latency_ms"]:
            self.metrics["p99_latency_ms"] = latency_ms
        else:
            self.metrics["p99_latency_ms"] = (
                self.metrics["p99_latency_ms"] * 0.99 + latency_ms * 0.01
            )
        
        # 更新错误率
        if not success:
            error_count = len(self.circuit_breaker_errors)
            self.metrics["error_rate"] = error_count / max(self.metrics["total_requests"], 1)
    
    async def _warm_up_cache(self):
        """预热缓存"""
        try:
            # 获取热门用户
            popular_users_key = "popular_users"
            popular_users = await self.redis.zrevrange(
                popular_users_key,
                0,
                9
            )
            
            scenarios = ["content", "product"]
            
            for user_id in popular_users:
                user_id_str = user_id.decode() if isinstance(user_id, bytes) else user_id
                for scenario in scenarios:
                    request = RecommendationRequest(
                        user_id=user_id_str,
                        scenario=scenario,
                        use_cache=False
                    )
                    
                    # 生成推荐
                    response = await self.get_recommendations(request)
                    
                    # 缓存结果
                    if response.recommendations:
                        request.use_cache = True
                        await self.result_cache.cache_result(request, response)
                    
                    await asyncio.sleep(0.5)  # 避免过载
            
            logger.info("缓存预热完成")
            
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标
        
        Returns:
            Dict[str, Any]: 性能指标
        """
        metrics = self.metrics.copy()
        
        # 添加缓存命中率
        if metrics["total_requests"] > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["total_requests"]
        else:
            metrics["cache_hit_rate"] = 0.0
        
        # 添加熔断器状态
        metrics["circuit_breaker_open"] = self.circuit_breaker_open
        
        return metrics