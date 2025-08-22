"""
Q-Learning与推荐引擎策略协调服务

实现Q-Learning智能体与推荐引擎的集成，提供统一的策略协调和决策服务
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json

from .qlearning_service import QLearningService
from .qlearning_strategy_service import QLearningStrategyService, StrategyInferenceRequest
from .bandit_recommendation_service import BanditRecommendationService
from ..core.logging import get_logger

logger = get_logger(__name__)


class StrategyCombinationMode(Enum):
    """策略组合模式"""
    WEIGHTED_AVERAGE = "weighted_average"
    EPSILON_SWITCHING = "epsilon_switching"
    CONTEXTUAL_SELECTION = "contextual_selection"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE_VOTING = "ensemble_voting"


class DecisionSource(Enum):
    """决策来源"""
    Q_LEARNING = "q_learning"
    BANDIT = "bandit"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass
class RecommendationContext:
    """推荐上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    item_features: Optional[List[float]] = None
    user_features: Optional[List[float]] = None
    context_features: Optional[List[float]] = None
    timestamp: Optional[datetime] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class HybridRecommendationRequest:
    """混合推荐请求"""
    agent_id: str
    bandit_arm_id: Optional[str] = None
    state: List[float] = None
    context: Optional[RecommendationContext] = None
    combination_mode: StrategyCombinationMode = StrategyCombinationMode.WEIGHTED_AVERAGE
    q_learning_weight: float = 0.5
    bandit_weight: float = 0.5
    fallback_to_random: bool = True


@dataclass
class HybridRecommendationResponse:
    """混合推荐响应"""
    recommended_action: int
    action_name: Optional[str] = None
    confidence_score: float = 0.0
    decision_source: DecisionSource = DecisionSource.HYBRID
    q_learning_result: Optional[Dict[str, Any]] = None
    bandit_result: Optional[Dict[str, Any]] = None
    combination_details: Optional[Dict[str, Any]] = None
    inference_time_ms: float = 0.0
    timestamp: datetime = None


@dataclass
class StrategyPerformanceMetrics:
    """策略性能指标"""
    strategy_name: str
    total_decisions: int = 0
    successful_decisions: int = 0
    average_reward: float = 0.0
    average_confidence: float = 0.0
    average_inference_time_ms: float = 0.0
    last_used: Optional[datetime] = None


class QLearningRecommendationService:
    """Q-Learning与推荐引擎协调服务"""
    
    def __init__(
        self, 
        qlearning_service: QLearningService,
        strategy_service: QLearningStrategyService,
        bandit_service: Optional[BanditRecommendationService] = None
    ):
        self.qlearning_service = qlearning_service
        self.strategy_service = strategy_service
        self.bandit_service = bandit_service
        
        # 策略性能跟踪
        self.strategy_metrics: Dict[str, StrategyPerformanceMetrics] = {
            "q_learning": StrategyPerformanceMetrics("q_learning"),
            "bandit": StrategyPerformanceMetrics("bandit"),
            "hybrid": StrategyPerformanceMetrics("hybrid")
        }
        
        # 决策历史
        self.decision_history: List[Dict[str, Any]] = []
        self.max_history_length = 10000
        
        # 自适应权重
        self.adaptive_weights: Dict[str, float] = {
            "q_learning": 0.5,
            "bandit": 0.5
        }
        
        logger.info("Q-Learning推荐协调服务初始化完成")
    
    async def hybrid_recommendation(self, request: HybridRecommendationRequest) -> HybridRecommendationResponse:
        """混合推荐决策"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            q_learning_result = None
            bandit_result = None
            
            # Q-Learning推理
            if request.state is not None:
                try:
                    q_inference_request = StrategyInferenceRequest(
                        agent_id=request.agent_id,
                        state=request.state,
                        evaluation_mode=True,
                        return_q_values=True,
                        return_confidence=True,
                        context=asdict(request.context) if request.context else None
                    )
                    
                    q_response = await self.strategy_service.single_inference(q_inference_request)
                    q_learning_result = {
                        "action": q_response.action,
                        "confidence": q_response.confidence_score,
                        "q_values": q_response.q_values,
                        "source": "q_learning"
                    }
                except Exception as e:
                    logger.warning(f"Q-Learning推理失败: {e}")
            
            # Bandit推理
            if self.bandit_service and request.bandit_arm_id:
                try:
                    # 这里需要根据实际的bandit服务接口调整
                    bandit_action = await self._get_bandit_recommendation(
                        request.bandit_arm_id, 
                        request.context
                    )
                    bandit_result = {
                        "action": bandit_action["action"],
                        "confidence": bandit_action.get("confidence", 0.5),
                        "source": "bandit"
                    }
                except Exception as e:
                    logger.warning(f"Bandit推理失败: {e}")
            
            # 策略组合
            final_action, decision_source, combination_details = self._combine_strategies(
                q_learning_result,
                bandit_result,
                request.combination_mode,
                request.q_learning_weight,
                request.bandit_weight
            )
            
            # 如果没有有效决策且允许fallback，使用随机选择
            if final_action is None and request.fallback_to_random:
                # 需要从智能体获取动作空间大小
                session = self.qlearning_service.active_sessions.get(request.agent_id)
                if session:
                    action_size = session.agent.action_size
                    final_action = np.random.randint(action_size)
                    decision_source = DecisionSource.FALLBACK
                else:
                    final_action = 0  # 默认动作
                    decision_source = DecisionSource.FALLBACK
            
            # 计算综合置信度
            confidence_score = self._calculate_combined_confidence(
                q_learning_result,
                bandit_result,
                combination_details
            )
            
            # 推理时间
            end_time = asyncio.get_event_loop().time()
            inference_time_ms = (end_time - start_time) * 1000
            
            # 创建响应
            response = HybridRecommendationResponse(
                recommended_action=final_action,
                confidence_score=confidence_score,
                decision_source=decision_source,
                q_learning_result=q_learning_result,
                bandit_result=bandit_result,
                combination_details=combination_details,
                inference_time_ms=inference_time_ms,
                timestamp=datetime.now()
            )
            
            # 记录决策历史
            self._record_decision(request, response)
            
            # 更新策略指标
            self._update_strategy_metrics(decision_source.value, inference_time_ms, confidence_score)
            
            logger.debug(f"混合推荐完成: 智能体={request.agent_id}, 动作={final_action}, "
                        f"来源={decision_source.value}, 时间={inference_time_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"混合推荐失败: {e}")
            raise
    
    async def adaptive_recommendation(self, request: HybridRecommendationRequest) -> HybridRecommendationResponse:
        """自适应推荐（根据历史性能自动调整权重）"""
        # 更新自适应权重
        self._update_adaptive_weights()
        
        # 使用自适应权重
        request.q_learning_weight = self.adaptive_weights["q_learning"]
        request.bandit_weight = self.adaptive_weights["bandit"]
        request.combination_mode = StrategyCombinationMode.WEIGHTED_AVERAGE
        
        return await self.hybrid_recommendation(request)
    
    async def contextual_recommendation(
        self, 
        agent_id: str, 
        state: List[float],
        context: RecommendationContext
    ) -> HybridRecommendationResponse:
        """基于上下文的智能推荐"""
        # 根据上下文特征选择最佳策略组合
        combination_mode, q_weight, bandit_weight = self._select_contextual_strategy(context)
        
        request = HybridRecommendationRequest(
            agent_id=agent_id,
            state=state,
            context=context,
            combination_mode=combination_mode,
            q_learning_weight=q_weight,
            bandit_weight=bandit_weight
        )
        
        return await self.hybrid_recommendation(request)
    
    async def update_strategy_feedback(
        self, 
        decision_id: str, 
        reward: float, 
        success: bool = None
    ):
        """更新策略反馈"""
        try:
            # 查找对应的决策记录
            decision_record = None
            for record in reversed(self.decision_history):
                if record.get("decision_id") == decision_id:
                    decision_record = record
                    break
            
            if not decision_record:
                logger.warning(f"未找到决策记录: {decision_id}")
                return
            
            # 更新决策记录
            decision_record["reward"] = reward
            decision_record["success"] = success
            decision_record["feedback_timestamp"] = datetime.now()
            
            # 更新策略性能指标
            decision_source = decision_record.get("decision_source")
            if decision_source in self.strategy_metrics:
                metrics = self.strategy_metrics[decision_source]
                
                # 更新奖励统计
                old_avg = metrics.average_reward
                metrics.total_decisions += 1
                metrics.average_reward = (
                    old_avg * (metrics.total_decisions - 1) + reward
                ) / metrics.total_decisions
                
                # 更新成功率
                if success is not None:
                    if success:
                        metrics.successful_decisions += 1
            
            # 基于反馈更新自适应权重
            self._update_adaptive_weights_with_feedback(decision_source, reward, success)
            
            logger.debug(f"策略反馈更新完成: decision_id={decision_id}, reward={reward}, success={success}")
            
        except Exception as e:
            logger.error(f"更新策略反馈失败: {e}")
            raise
    
    async def get_strategy_performance_report(self) -> Dict[str, Any]:
        """获取策略性能报告"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "strategy_metrics": {},
                "adaptive_weights": self.adaptive_weights.copy(),
                "decision_statistics": {},
                "recommendations": []
            }
            
            # 策略指标
            for name, metrics in self.strategy_metrics.items():
                report["strategy_metrics"][name] = {
                    "total_decisions": metrics.total_decisions,
                    "successful_decisions": metrics.successful_decisions,
                    "success_rate": (
                        metrics.successful_decisions / max(metrics.total_decisions, 1)
                    ),
                    "average_reward": metrics.average_reward,
                    "average_confidence": metrics.average_confidence,
                    "average_inference_time_ms": metrics.average_inference_time_ms,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else None
                }
            
            # 决策统计
            if self.decision_history:
                recent_decisions = self.decision_history[-1000:]  # 最近1000个决策
                
                decision_sources = [d.get("decision_source") for d in recent_decisions]
                source_counts = {}
                for source in decision_sources:
                    source_counts[source] = source_counts.get(source, 0) + 1
                
                report["decision_statistics"] = {
                    "total_recent_decisions": len(recent_decisions),
                    "source_distribution": source_counts,
                    "average_confidence": np.mean([
                        d.get("confidence_score", 0.0) for d in recent_decisions
                    ]),
                    "average_inference_time_ms": np.mean([
                        d.get("inference_time_ms", 0.0) for d in recent_decisions
                    ])
                }
            
            # 策略建议
            recommendations = self._generate_strategy_recommendations()
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"获取策略性能报告失败: {e}")
            raise
    
    async def optimize_strategy_weights(self) -> Dict[str, float]:
        """优化策略权重"""
        try:
            # 基于历史性能数据优化权重
            if len(self.decision_history) < 100:
                logger.warning("决策历史数据不足，无法优化权重")
                return self.adaptive_weights.copy()
            
            # 获取最近的决策数据
            recent_decisions = [
                d for d in self.decision_history[-1000:]
                if d.get("reward") is not None
            ]
            
            if not recent_decisions:
                logger.warning("没有包含奖励信息的决策记录")
                return self.adaptive_weights.copy()
            
            # 按决策来源分组计算性能
            source_performance = {}
            for decision in recent_decisions:
                source = decision.get("decision_source")
                reward = decision.get("reward", 0.0)
                
                if source not in source_performance:
                    source_performance[source] = []
                source_performance[source].append(reward)
            
            # 计算每个策略的平均性能
            avg_performance = {}
            for source, rewards in source_performance.items():
                avg_performance[source] = np.mean(rewards)
            
            # 基于性能重新分配权重
            if "q_learning" in avg_performance and "bandit" in avg_performance:
                q_perf = avg_performance["q_learning"]
                bandit_perf = avg_performance["bandit"]
                
                # 使用softmax函数计算权重
                total_perf = q_perf + bandit_perf
                if total_perf > 0:
                    self.adaptive_weights["q_learning"] = q_perf / total_perf
                    self.adaptive_weights["bandit"] = bandit_perf / total_perf
                else:
                    # 如果性能都为负，使用等权重
                    self.adaptive_weights["q_learning"] = 0.5
                    self.adaptive_weights["bandit"] = 0.5
            
            logger.info(f"策略权重优化完成: {self.adaptive_weights}")
            return self.adaptive_weights.copy()
            
        except Exception as e:
            logger.error(f"优化策略权重失败: {e}")
            raise
    
    def _combine_strategies(
        self,
        q_learning_result: Optional[Dict[str, Any]],
        bandit_result: Optional[Dict[str, Any]], 
        combination_mode: StrategyCombinationMode,
        q_weight: float,
        bandit_weight: float
    ) -> Tuple[Optional[int], DecisionSource, Dict[str, Any]]:
        """组合策略结果"""
        combination_details = {
            "mode": combination_mode.value,
            "q_learning_available": q_learning_result is not None,
            "bandit_available": bandit_result is not None
        }
        
        # 如果只有一个策略可用
        if q_learning_result and not bandit_result:
            return q_learning_result["action"], DecisionSource.Q_LEARNING, combination_details
        elif bandit_result and not q_learning_result:
            return bandit_result["action"], DecisionSource.BANDIT, combination_details
        elif not q_learning_result and not bandit_result:
            return None, DecisionSource.FALLBACK, combination_details
        
        # 两个策略都可用，根据组合模式决定
        if combination_mode == StrategyCombinationMode.WEIGHTED_AVERAGE:
            # 基于置信度加权选择
            q_conf = q_learning_result.get("confidence", 0.0)
            bandit_conf = bandit_result.get("confidence", 0.0)
            
            # 综合权重和置信度
            q_score = q_weight * q_conf
            bandit_score = bandit_weight * bandit_conf
            
            combination_details.update({
                "q_learning_score": q_score,
                "bandit_score": bandit_score,
                "q_learning_confidence": q_conf,
                "bandit_confidence": bandit_conf
            })
            
            if q_score > bandit_score:
                return q_learning_result["action"], DecisionSource.Q_LEARNING, combination_details
            else:
                return bandit_result["action"], DecisionSource.BANDIT, combination_details
        
        elif combination_mode == StrategyCombinationMode.EPSILON_SWITCHING:
            # 随机选择一个策略
            if np.random.random() < q_weight:
                return q_learning_result["action"], DecisionSource.Q_LEARNING, combination_details
            else:
                return bandit_result["action"], DecisionSource.BANDIT, combination_details
        
        elif combination_mode == StrategyCombinationMode.CONTEXTUAL_SELECTION:
            # 基于上下文选择最佳策略（简化实现）
            # 这里可以根据具体的上下文特征实现更复杂的选择逻辑
            q_conf = q_learning_result.get("confidence", 0.0)
            bandit_conf = bandit_result.get("confidence", 0.0)
            
            if q_conf > bandit_conf:
                return q_learning_result["action"], DecisionSource.Q_LEARNING, combination_details
            else:
                return bandit_result["action"], DecisionSource.BANDIT, combination_details
        
        else:
            # 默认使用置信度选择
            q_conf = q_learning_result.get("confidence", 0.0)
            bandit_conf = bandit_result.get("confidence", 0.0)
            
            if q_conf > bandit_conf:
                return q_learning_result["action"], DecisionSource.Q_LEARNING, combination_details
            else:
                return bandit_result["action"], DecisionSource.BANDIT, combination_details
    
    def _calculate_combined_confidence(
        self,
        q_learning_result: Optional[Dict[str, Any]],
        bandit_result: Optional[Dict[str, Any]],
        combination_details: Dict[str, Any]
    ) -> float:
        """计算综合置信度"""
        q_conf = q_learning_result.get("confidence", 0.0) if q_learning_result else 0.0
        bandit_conf = bandit_result.get("confidence", 0.0) if bandit_result else 0.0
        
        if q_learning_result and bandit_result:
            # 两个策略都可用，使用加权平均
            return q_conf * self.adaptive_weights.get("q_learning", 0.5) + \
                   bandit_conf * self.adaptive_weights.get("bandit", 0.5)
        elif q_learning_result:
            return q_conf
        elif bandit_result:
            return bandit_conf
        else:
            return 0.0
    
    def _record_decision(self, request: HybridRecommendationRequest, response: HybridRecommendationResponse):
        """记录决策历史"""
        decision_record = {
            "decision_id": f"{response.timestamp.strftime('%Y%m%d%H%M%S')}_{response.recommended_action}",
            "agent_id": request.agent_id,
            "recommended_action": response.recommended_action,
            "decision_source": response.decision_source.value,
            "confidence_score": response.confidence_score,
            "inference_time_ms": response.inference_time_ms,
            "timestamp": response.timestamp,
            "combination_mode": request.combination_mode.value,
            "q_learning_weight": request.q_learning_weight,
            "bandit_weight": request.bandit_weight,
            "q_learning_result": response.q_learning_result,
            "bandit_result": response.bandit_result,
            "reward": None,  # 待后续更新
            "success": None,  # 待后续更新
        }
        
        self.decision_history.append(decision_record)
        
        # 限制历史记录长度
        if len(self.decision_history) > self.max_history_length:
            self.decision_history = self.decision_history[-self.max_history_length//2:]
    
    def _update_strategy_metrics(self, strategy_name: str, inference_time_ms: float, confidence: float):
        """更新策略指标"""
        if strategy_name in self.strategy_metrics:
            metrics = self.strategy_metrics[strategy_name]
            
            # 更新推理时间
            old_time = metrics.average_inference_time_ms
            old_count = metrics.total_decisions
            metrics.average_inference_time_ms = (
                old_time * old_count + inference_time_ms
            ) / (old_count + 1)
            
            # 更新置信度
            old_conf = metrics.average_confidence
            metrics.average_confidence = (
                old_conf * old_count + confidence
            ) / (old_count + 1)
            
            # 更新使用时间
            metrics.last_used = datetime.now()
    
    def _update_adaptive_weights(self):
        """更新自适应权重"""
        # 基于最近的性能表现调整权重
        if len(self.decision_history) < 50:
            return
        
        recent_decisions = self.decision_history[-100:]
        
        # 计算各策略的成功率
        q_learning_rewards = []
        bandit_rewards = []
        
        for decision in recent_decisions:
            reward = decision.get("reward")
            if reward is not None:
                source = decision.get("decision_source")
                if source == "q_learning":
                    q_learning_rewards.append(reward)
                elif source == "bandit":
                    bandit_rewards.append(reward)
        
        # 根据平均奖励调整权重
        if q_learning_rewards and bandit_rewards:
            q_avg = np.mean(q_learning_rewards)
            bandit_avg = np.mean(bandit_rewards)
            
            # 使用softmax调整权重
            exp_q = np.exp(q_avg)
            exp_bandit = np.exp(bandit_avg)
            total = exp_q + exp_bandit
            
            self.adaptive_weights["q_learning"] = exp_q / total
            self.adaptive_weights["bandit"] = exp_bandit / total
    
    def _update_adaptive_weights_with_feedback(self, decision_source: str, reward: float, success: bool):
        """基于反馈更新自适应权重"""
        # 简单的学习率调整
        learning_rate = 0.01
        
        if decision_source == "q_learning":
            # 根据奖励调整Q-Learning权重
            if reward > 0:
                self.adaptive_weights["q_learning"] = min(
                    0.9, 
                    self.adaptive_weights["q_learning"] + learning_rate
                )
                self.adaptive_weights["bandit"] = 1.0 - self.adaptive_weights["q_learning"]
        elif decision_source == "bandit":
            # 根据奖励调整Bandit权重
            if reward > 0:
                self.adaptive_weights["bandit"] = min(
                    0.9,
                    self.adaptive_weights["bandit"] + learning_rate
                )
                self.adaptive_weights["q_learning"] = 1.0 - self.adaptive_weights["bandit"]
    
    def _select_contextual_strategy(self, context: RecommendationContext) -> Tuple[StrategyCombinationMode, float, float]:
        """基于上下文选择策略组合"""
        # 简化的上下文策略选择逻辑
        # 实际应用中可以根据具体的上下文特征实现更复杂的选择策略
        
        if not context:
            return StrategyCombinationMode.WEIGHTED_AVERAGE, 0.5, 0.5
        
        # 基于用户特征选择策略
        if context.user_features:
            # 如果用户特征维度高，偏向使用Q-Learning
            if len(context.user_features) > 10:
                return StrategyCombinationMode.WEIGHTED_AVERAGE, 0.7, 0.3
            else:
                return StrategyCombinationMode.WEIGHTED_AVERAGE, 0.3, 0.7
        
        # 默认策略
        return StrategyCombinationMode.WEIGHTED_AVERAGE, 0.5, 0.5
    
    def _generate_strategy_recommendations(self) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        # 分析策略性能
        q_learning_metrics = self.strategy_metrics.get("q_learning")
        bandit_metrics = self.strategy_metrics.get("bandit")
        
        if q_learning_metrics and bandit_metrics:
            if q_learning_metrics.average_reward > bandit_metrics.average_reward * 1.2:
                recommendations.append("Q-Learning策略表现显著优于Bandit，建议增加Q-Learning权重")
            elif bandit_metrics.average_reward > q_learning_metrics.average_reward * 1.2:
                recommendations.append("Bandit策略表现显著优于Q-Learning，建议增加Bandit权重")
            
            if q_learning_metrics.average_inference_time_ms > bandit_metrics.average_inference_time_ms * 2:
                recommendations.append("Q-Learning推理时间较长，建议在实时性要求高的场景中降低其权重")
        
        # 检查决策历史
        if len(self.decision_history) > 1000:
            recent_success_rate = np.mean([
                1 if d.get("success") else 0
                for d in self.decision_history[-100:]
                if d.get("success") is not None
            ])
            
            if recent_success_rate < 0.5:
                recommendations.append("最近决策成功率较低，建议重新评估策略权重或模型参数")
        
        if not recommendations:
            recommendations.append("策略表现正常，建议继续监控性能指标")
        
        return recommendations
    
    async def _get_bandit_recommendation(self, arm_id: str, context: RecommendationContext) -> Dict[str, Any]:
        """获取Bandit推荐（模拟实现）"""
        # 这里需要根据实际的bandit服务接口实现
        # 目前提供一个模拟实现
        
        if not self.bandit_service:
            raise ValueError("Bandit服务未配置")
        
        # 模拟bandit推荐
        return {
            "action": np.random.randint(4),  # 假设4个动作
            "confidence": np.random.uniform(0.3, 0.9),
            "arm_id": arm_id,
            "expected_reward": np.random.uniform(0, 1)
        }