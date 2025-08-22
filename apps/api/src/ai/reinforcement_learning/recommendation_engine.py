"""
多臂老虎机推荐引擎

整合多种多臂老虎机算法，提供统一的推荐服务接口，支持冷启动、特征处理和性能评估。
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import uuid
from enum import Enum

from .bandits.base import MultiArmedBandit
from .bandits.ucb import UCBBandit
from .bandits.thompson_sampling import ThompsonSamplingBandit
from .bandits.epsilon_greedy import EpsilonGreedyBandit
from .bandits.contextual import LinearContextualBandit
from .cold_start import ColdStartStrategy, ContentBasedColdStart, PopularityBasedColdStart, HybridColdStart
from .feature_processor import ContextFeatureProcessor, UserFeatureProcessor, ItemFeatureProcessor
from .evaluation import OnlineEvaluator, ABTestManager, InteractionEvent, EvaluationMetrics


class AlgorithmType(Enum):
    """算法类型枚举"""
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    EPSILON_GREEDY = "epsilon_greedy"
    LINEAR_CONTEXTUAL = "linear_contextual"


@dataclass
class RecommendationRequest:
    """推荐请求"""
    user_id: str
    context: Optional[Dict[str, Any]] = None
    num_recommendations: int = 10
    exclude_items: Optional[List[str]] = None
    include_explanations: bool = False
    experiment_id: Optional[str] = None


@dataclass
class RecommendationResponse:
    """推荐响应"""
    request_id: str
    user_id: str
    recommendations: List[Dict[str, Any]]
    algorithm_used: str
    confidence_score: float
    cold_start_strategy: Optional[str] = None
    explanations: Optional[List[str]] = None
    timestamp: datetime = None
    processing_time_ms: float = 0.0


@dataclass
class FeedbackData:
    """用户反馈数据"""
    user_id: str
    item_id: str
    feedback_type: str  # 'click', 'like', 'purchase', 'rating', etc.
    feedback_value: float
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None


class BanditRecommendationEngine:
    """多臂老虎机推荐引擎"""
    
    def __init__(
        self,
        default_algorithm: AlgorithmType = AlgorithmType.UCB,
        enable_cold_start: bool = True,
        enable_evaluation: bool = True,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 10000
    ):
        """
        初始化推荐引擎
        
        Args:
            default_algorithm: 默认算法
            enable_cold_start: 是否启用冷启动
            enable_evaluation: 是否启用评估
            cache_ttl_seconds: 缓存TTL(秒)
            max_cache_size: 最大缓存大小
        """
        self.default_algorithm = default_algorithm
        self.enable_cold_start = enable_cold_start
        self.enable_evaluation = enable_evaluation
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        
        # 算法实例
        self.algorithms: Dict[str, MultiArmedBandit] = {}
        self.user_algorithms: Dict[str, str] = {}  # 用户 -> 算法映射
        
        # 冷启动策略
        self.cold_start_strategy: Optional[ColdStartStrategy] = None
        if enable_cold_start:
            content_strategy = ContentBasedColdStart()
            popularity_strategy = PopularityBasedColdStart()
            self.cold_start_strategy = HybridColdStart(content_strategy, popularity_strategy)
        
        # 特征处理器
        self.user_feature_processor = UserFeatureProcessor()
        self.item_feature_processor = ItemFeatureProcessor()
        self.feature_processors_fitted = False
        
        # 评估器
        self.evaluator: Optional[OnlineEvaluator] = None
        self.ab_test_manager: Optional[ABTestManager] = None
        if enable_evaluation:
            self.evaluator = OnlineEvaluator()
            self.ab_test_manager = ABTestManager()
        
        # 缓存
        self.recommendation_cache: Dict[str, Tuple[RecommendationResponse, datetime]] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        self.item_features: Dict[str, Dict[str, Any]] = {}
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cold_start_requests": 0,
            "algorithm_usage": {},
            "average_response_time_ms": 0.0
        }
    
    async def initialize_algorithms(self, n_items: int, algorithm_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        初始化算法
        
        Args:
            n_items: 物品数量
            algorithm_configs: 算法配置
        """
        default_configs = {
            "ucb": {"c": 2.0, "random_state": 42},
            "thompson_sampling": {"alpha_init": 1.0, "beta_init": 1.0, "random_state": 42},
            "epsilon_greedy": {"epsilon": 0.1, "decay_rate": 0.995, "random_state": 42},
            "linear_contextual": {"n_features": 50, "alpha": 1.0, "lambda_reg": 1.0, "random_state": 42}
        }
        
        configs = algorithm_configs or default_configs
        
        # 初始化各种算法
        for algo_name, config in configs.items():
            if algo_name == "ucb":
                self.algorithms[algo_name] = UCBBandit(n_arms=n_items, **config)
            elif algo_name == "thompson_sampling":
                self.algorithms[algo_name] = ThompsonSamplingBandit(n_arms=n_items, **config)
            elif algo_name == "epsilon_greedy":
                self.algorithms[algo_name] = EpsilonGreedyBandit(n_arms=n_items, **config)
            elif algo_name == "linear_contextual":
                self.algorithms[algo_name] = LinearContextualBandit(n_arms=n_items, **config)
        
        # 初始化统计
        for algo_name in self.algorithms.keys():
            self.stats["algorithm_usage"][algo_name] = 0
    
    async def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        获取推荐
        
        Args:
            request: 推荐请求
            
        Returns:
            推荐响应
        """
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_recommendation(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                cached_response.request_id = request_id  # 更新请求ID
                return cached_response
            
            # 检查A/B测试
            algorithm_name = await self._select_algorithm(request)
            
            # 处理冷启动
            if self._is_cold_start_user(request.user_id):
                response = await self._handle_cold_start(request, request_id)
                if response:
                    return response
            
            # 正常推荐流程
            response = await self._generate_recommendations(request, request_id, algorithm_name)
            
            # 缓存结果
            self._cache_recommendation(cache_key, response)
            
            # 更新统计
            self._update_statistics(response, start_time)
            
            return response
            
        except Exception as e:
            # 出错时返回默认推荐
            return await self._generate_fallback_recommendations(request, request_id, str(e))
    
    async def process_feedback(self, feedback: FeedbackData):
        """
        处理用户反馈
        
        Args:
            feedback: 反馈数据
        """
        try:
            # 转换为交互事件
            event = InteractionEvent(
                event_id=str(uuid.uuid4()),
                user_id=feedback.user_id,
                item_id=feedback.item_id,
                timestamp=feedback.timestamp or datetime.now(),
                action_type=feedback.feedback_type,
                reward=self._convert_feedback_to_reward(feedback),
                context=feedback.context
            )
            
            # 更新算法
            await self._update_algorithms(feedback)
            
            # 更新冷启动策略
            if self.cold_start_strategy and hasattr(self.cold_start_strategy, 'content_strategy'):
                self.cold_start_strategy.content_strategy.update_user_item_interaction(
                    feedback.user_id,
                    feedback.item_id,
                    feedback.feedback_value
                )
            
            # 添加到评估器
            if self.evaluator:
                self.evaluator.add_interaction(event)
            
        except Exception as e:
            print(f"处理反馈时出错: {e}")
    
    async def _select_algorithm(self, request: RecommendationRequest) -> str:
        """选择推荐算法"""
        # 检查用户是否已分配算法
        if request.user_id in self.user_algorithms:
            return self.user_algorithms[request.user_id]
        
        # 检查A/B测试
        if self.ab_test_manager and request.experiment_id:
            variant = self.ab_test_manager.assign_user_to_variant(request.experiment_id, request.user_id)
            if variant:
                self.user_algorithms[request.user_id] = variant
                return variant
        
        # 使用默认算法
        algorithm_name = self.default_algorithm.value
        self.user_algorithms[request.user_id] = algorithm_name
        return algorithm_name
    
    async def _generate_recommendations(
        self, 
        request: RecommendationRequest, 
        request_id: str, 
        algorithm_name: str
    ) -> RecommendationResponse:
        """生成推荐"""
        if algorithm_name not in self.algorithms:
            algorithm_name = self.default_algorithm.value
        
        algorithm = self.algorithms[algorithm_name]
        
        # 准备上下文特征
        context = await self._prepare_context(request)
        
        # 生成推荐
        recommendations = []
        exclude_set = set(request.exclude_items or [])
        
        for _ in range(request.num_recommendations):
            try:
                if algorithm_name == "linear_contextual":
                    item_id = algorithm.select_arm(context)
                else:
                    item_id = algorithm.select_arm()
                
                item_id_str = str(item_id)
                
                if item_id_str not in exclude_set:
                    recommendations.append({
                        "item_id": item_id_str,
                        "score": self._calculate_item_score(algorithm, item_id, context),
                        "confidence": self._calculate_confidence(algorithm, item_id)
                    })
                    exclude_set.add(item_id_str)
                
            except Exception as e:
                print(f"生成推荐时出错: {e}")
                continue
        
        # 计算整体置信度
        confidence_score = np.mean([rec["confidence"] for rec in recommendations]) if recommendations else 0.0
        
        # 生成解释（如果需要）
        explanations = None
        if request.include_explanations:
            explanations = self._generate_explanations(recommendations, algorithm_name)
        
        response = RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used=algorithm_name,
            confidence_score=confidence_score,
            explanations=explanations,
            timestamp=datetime.now()
        )
        
        return response
    
    async def _handle_cold_start(
        self, 
        request: RecommendationRequest, 
        request_id: str
    ) -> Optional[RecommendationResponse]:
        """处理冷启动"""
        if not self.cold_start_strategy:
            return None
        
        self.stats["cold_start_requests"] += 1
        
        user_features = self.user_contexts.get(request.user_id, {})
        cold_start_result = self.cold_start_strategy.handle_new_user(request.user_id, user_features)
        
        recommendations = []
        for item_id in cold_start_result["recommendations"][:request.num_recommendations]:
            recommendations.append({
                "item_id": item_id,
                "score": 0.5,  # 冷启动默认分数
                "confidence": cold_start_result["confidence"]
            })
        
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used="cold_start",
            confidence_score=cold_start_result["confidence"],
            cold_start_strategy=cold_start_result["strategy"],
            timestamp=datetime.now()
        )
    
    async def _prepare_context(self, request: RecommendationRequest) -> Optional[Dict[str, Any]]:
        """准备上下文特征"""
        if not request.context:
            return None
        
        try:
            # 如果特征处理器已拟合，使用它们处理特征
            if self.feature_processors_fitted:
                user_features = self.user_feature_processor.transform(request.context)
                return {"features": user_features.tolist()}
            else:
                # 简单的特征转换：将上下文字典转为数值特征向量
                if 'features' not in request.context:
                    # 自动将上下文转换为特征向量
                    features = []
                    for key, value in sorted(request.context.items()):
                        if isinstance(value, (int, float)):
                            features.append(float(value))
                        elif isinstance(value, str):
                            # 字符串特征使用简单哈希
                            features.append(hash(value) % 1000 / 1000.0)
                        else:
                            features.append(0.0)
                    
                    # 补齐或截断到固定长度（基于线性上下文算法的特征维度）
                    target_len = 10  # 默认特征维度
                    if len(features) > target_len:
                        features = features[:target_len]
                    elif len(features) < target_len:
                        features.extend([0.0] * (target_len - len(features)))
                    
                    return {"features": features}
                else:
                    return request.context
        except Exception as e:
            print(f"处理上下文特征时出错: {e}")
            return request.context
    
    def _calculate_item_score(self, algorithm: MultiArmedBandit, item_id: int, context: Optional[Dict[str, Any]]) -> float:
        """计算物品分数"""
        try:
            if hasattr(algorithm, 'get_ucb_values'):
                ucb_values = algorithm.get_ucb_values()
                return float(ucb_values[item_id]) if item_id < len(ucb_values) else 0.0
            elif hasattr(algorithm, 'get_posterior_stats'):
                posterior_stats = algorithm.get_posterior_stats()
                means = posterior_stats['posterior_means']
                return float(means[item_id]) if item_id < len(means) else 0.0
            elif hasattr(algorithm, 'predict_reward') and context:
                features = np.array(context.get('features', [0.0]))
                return float(algorithm.predict_reward(item_id, features))
            else:
                # 使用平均奖励作为分数
                if algorithm.n_pulls[item_id] > 0:
                    return float(algorithm.rewards[item_id] / algorithm.n_pulls[item_id])
                return 0.0
        except Exception as e:
            print(f"计算物品分数时出错: {e}")
            return 0.0
    
    def _calculate_confidence(self, algorithm: MultiArmedBandit, item_id: int) -> float:
        """计算置信度"""
        try:
            confidence_intervals = algorithm._get_confidence_intervals()
            if item_id < len(confidence_intervals):
                # 置信度与置信区间成反比
                ci = confidence_intervals[item_id]
                return float(1.0 / (1.0 + ci)) if ci > 0 else 0.5
            return 0.5
        except Exception as e:
            print(f"计算置信度时出错: {e}")
            return 0.5
    
    def _generate_explanations(self, recommendations: List[Dict[str, Any]], algorithm_name: str) -> List[str]:
        """生成推荐解释"""
        explanations = []
        
        for rec in recommendations:
            if algorithm_name == "ucb":
                explanation = f"基于UCB算法选择，平衡了探索与利用（置信度: {rec['confidence']:.2f}）"
            elif algorithm_name == "thompson_sampling":
                explanation = f"基于贝叶斯Thompson Sampling，从后验分布采样选择（置信度: {rec['confidence']:.2f}）"
            elif algorithm_name == "epsilon_greedy":
                explanation = f"基于Epsilon-Greedy策略，以一定概率探索新选项（置信度: {rec['confidence']:.2f}）"
            elif algorithm_name == "linear_contextual":
                explanation = f"基于上下文特征的线性模型预测（置信度: {rec['confidence']:.2f}）"
            else:
                explanation = f"基于{algorithm_name}算法推荐（置信度: {rec['confidence']:.2f}）"
            
            explanations.append(explanation)
        
        return explanations
    
    async def _update_algorithms(self, feedback: FeedbackData):
        """更新算法"""
        item_id = int(feedback.item_id) if feedback.item_id.isdigit() else hash(feedback.item_id) % 1000
        reward = self._convert_feedback_to_reward(feedback)
        
        # 获取用户使用的算法
        algorithm_name = self.user_algorithms.get(feedback.user_id, self.default_algorithm.value)
        
        if algorithm_name in self.algorithms:
            algorithm = self.algorithms[algorithm_name]
            
            try:
                if algorithm_name == "linear_contextual" and feedback.context:
                    context = await self._prepare_context(RecommendationRequest(user_id=feedback.user_id, context=feedback.context))
                    algorithm.update(item_id, reward, context)
                elif algorithm_name == "thompson_sampling":
                    success = reward > 0.5
                    algorithm.update_with_binary_reward(item_id, success)
                else:
                    algorithm.update(item_id, reward)
                    
            except Exception as e:
                print(f"更新算法{algorithm_name}时出错: {e}")
    
    def _convert_feedback_to_reward(self, feedback: FeedbackData) -> float:
        """将反馈转换为奖励值"""
        feedback_type = feedback.feedback_type.lower()
        
        # 定义反馈类型的奖励映射
        reward_mapping = {
            "view": 0.1,
            "click": 0.3, 
            "like": 0.6,
            "share": 0.8,
            "purchase": 1.0,
            "rating": feedback.feedback_value / 5.0  # 假设评分是1-5
        }
        
        if feedback_type in reward_mapping:
            if feedback_type == "rating":
                return max(0.0, min(1.0, reward_mapping[feedback_type]))
            else:
                return reward_mapping[feedback_type]
        else:
            # 直接使用反馈值
            return max(0.0, min(1.0, feedback.feedback_value))
    
    def _is_cold_start_user(self, user_id: str) -> bool:
        """判断是否为冷启动用户"""
        # 简单策略：检查用户是否有历史记录
        return user_id not in self.user_algorithms and user_id not in self.user_contexts
    
    def _generate_cache_key(self, request: RecommendationRequest) -> str:
        """生成缓存键"""
        context_hash = hash(str(sorted(request.context.items()))) if request.context else 0
        return f"{request.user_id}_{request.num_recommendations}_{context_hash}"
    
    def _get_cached_recommendation(self, cache_key: str) -> Optional[RecommendationResponse]:
        """获取缓存的推荐"""
        if cache_key in self.recommendation_cache:
            response, timestamp = self.recommendation_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                return response
            else:
                # 过期，删除缓存
                del self.recommendation_cache[cache_key]
        return None
    
    def _cache_recommendation(self, cache_key: str, response: RecommendationResponse):
        """缓存推荐结果"""
        # 限制缓存大小
        if len(self.recommendation_cache) >= self.max_cache_size:
            # 删除最老的缓存项
            oldest_key = min(self.recommendation_cache.keys(), 
                           key=lambda k: self.recommendation_cache[k][1])
            del self.recommendation_cache[oldest_key]
        
        self.recommendation_cache[cache_key] = (response, datetime.now())
    
    def _update_statistics(self, response: RecommendationResponse, start_time: datetime):
        """更新统计信息"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        response.processing_time_ms = processing_time
        
        self.stats["total_requests"] += 1
        self.stats["algorithm_usage"][response.algorithm_used] = \
            self.stats["algorithm_usage"].get(response.algorithm_used, 0) + 1
        
        # 更新平均响应时间
        current_avg = self.stats["average_response_time_ms"]
        total_requests = self.stats["total_requests"]
        self.stats["average_response_time_ms"] = \
            (current_avg * (total_requests - 1) + processing_time) / total_requests
    
    async def _generate_fallback_recommendations(
        self, 
        request: RecommendationRequest, 
        request_id: str, 
        error_msg: str
    ) -> RecommendationResponse:
        """生成备用推荐"""
        # 返回随机推荐
        recommendations = []
        for i in range(min(request.num_recommendations, 10)):
            recommendations.append({
                "item_id": str(i),
                "score": 0.1,
                "confidence": 0.1
            })
        
        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm_used="fallback",
            confidence_score=0.1,
            timestamp=datetime.now()
        )
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        algorithm_stats = {}
        for name, algorithm in self.algorithms.items():
            algorithm_stats[name] = algorithm.get_performance_metrics()
        
        evaluation_metrics = None
        if self.evaluator:
            evaluation_metrics = asdict(self.evaluator.calculate_current_metrics())
        
        return {
            "engine_stats": self.stats,
            "algorithm_stats": algorithm_stats,
            "evaluation_metrics": evaluation_metrics,
            "cache_size": len(self.recommendation_cache),
            "active_users": len(self.user_algorithms),
            "cold_start_enabled": self.enable_cold_start,
            "evaluation_enabled": self.enable_evaluation
        }
    
    async def update_user_context(self, user_id: str, context: Dict[str, Any]):
        """更新用户上下文"""
        self.user_contexts[user_id] = context
    
    async def update_item_features(self, item_id: str, features: Dict[str, Any]):
        """更新物品特征"""
        self.item_features[item_id] = features