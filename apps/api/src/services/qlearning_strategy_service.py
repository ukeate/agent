"""
Q-Learning智能体策略推理服务

提供训练好的Q-Learning智能体的策略推理和决策服务
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
from pathlib import Path

from .qlearning_service import QLearningService, QLearningAgentSession
from ..ai.reinforcement_learning.qlearning import QLearningAgent
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyInferenceRequest:
    """策略推理请求"""
    agent_id: str
    state: List[float]
    context: Optional[Dict[str, Any]] = None
    evaluation_mode: bool = True
    return_q_values: bool = True
    return_confidence: bool = True


@dataclass
class StrategyInferenceResponse:
    """策略推理响应"""
    agent_id: str
    action: int
    action_name: Optional[str] = None
    q_values: Optional[List[float]] = None
    confidence_score: float = 0.0
    exploration_info: Optional[Dict[str, Any]] = None
    inference_time_ms: float = 0.0
    timestamp: datetime = None


@dataclass
class BatchInferenceRequest:
    """批量推理请求"""
    agent_id: str
    states: List[List[float]]
    evaluation_mode: bool = True
    return_details: bool = False


@dataclass
class BatchInferenceResponse:
    """批量推理响应"""
    agent_id: str
    actions: List[int]
    total_inference_time_ms: float
    average_inference_time_ms: float
    details: Optional[List[StrategyInferenceResponse]] = None


@dataclass
class StrategyComparison:
    """策略比较结果"""
    agent_ids: List[str]
    state: List[float]
    recommendations: List[Dict[str, Any]]
    best_agent_id: str
    comparison_metrics: Dict[str, Any]


class QLearningStrategyService:
    """Q-Learning策略推理服务"""
    
    def __init__(self, qlearning_service: QLearningService):
        self.qlearning_service = qlearning_service
        self.inference_cache: Dict[str, Dict] = {}
        self.cache_ttl_seconds = 300  # 5分钟缓存
        self.performance_metrics: Dict[str, List[float]] = {}
        
        logger.info("Q-Learning策略推理服务初始化完成")
    
    async def single_inference(self, request: StrategyInferenceRequest) -> StrategyInferenceResponse:
        """单个状态推理"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 检查智能体是否存在
            if request.agent_id not in self.qlearning_service.active_sessions:
                raise ValueError(f"智能体不存在: {request.agent_id}")
            
            session = self.qlearning_service.active_sessions[request.agent_id]
            agent = session.agent
            
            # 检查缓存
            cache_key = self._generate_cache_key(request.agent_id, request.state, request.evaluation_mode)
            cached_result = self._get_cached_inference(cache_key)
            
            if cached_result:
                logger.debug(f"使用缓存结果: {cache_key}")
                return cached_result
            
            # 执行推理
            state_array = np.array(request.state)
            
            # 获取Q值
            q_values = None
            if hasattr(agent, 'get_q_values'):
                q_values = agent.get_q_values(state_array)
            elif hasattr(agent, 'q_table') and hasattr(agent, '_state_to_key'):
                # 经典Q-Learning智能体
                state_key = agent._state_to_key(state_array)
                if state_key in agent.q_table:
                    q_values = np.array([
                        agent.q_table[state_key].get(a, 0.0) 
                        for a in range(agent.action_size)
                    ])
                else:
                    q_values = np.zeros(agent.action_size)
            
            # 选择动作
            if request.evaluation_mode:
                # 评估模式：贪婪选择
                if q_values is not None:
                    action = int(np.argmax(q_values))
                else:
                    action = agent.act(state_array, evaluation=True)
                exploration_info = {"mode": "greedy", "exploration_rate": 0.0}
            else:
                # 训练模式：使用探索策略
                action = agent.act(state_array)
                exploration_info = {
                    "mode": "exploration",
                    "exploration_rate": session.exploration_strategy.get_exploration_rate(),
                    "strategy_type": type(session.exploration_strategy).__name__
                }
            
            # 计算置信度
            confidence_score = 0.0
            if q_values is not None and len(q_values) > 1:
                sorted_q = np.sort(q_values)
                if sorted_q[-1] > sorted_q[-2]:
                    confidence_score = float((sorted_q[-1] - sorted_q[-2]) / max(abs(sorted_q[-1]), 1.0))
                else:
                    confidence_score = 0.0
            
            # 获取动作名称
            action_name = None
            if hasattr(session.environment, 'get_action_name'):
                action_name = session.environment.get_action_name(action)
            elif hasattr(session.environment, 'action_space') and hasattr(session.environment.action_space, 'action_names'):
                action_names = session.environment.action_space.action_names
                if action_names and action < len(action_names):
                    action_name = action_names[action]
            
            # 计算推理时间
            end_time = asyncio.get_event_loop().time()
            inference_time_ms = (end_time - start_time) * 1000
            
            # 创建响应
            response = StrategyInferenceResponse(
                agent_id=request.agent_id,
                action=action,
                action_name=action_name,
                q_values=q_values.tolist() if q_values is not None and request.return_q_values else None,
                confidence_score=confidence_score if request.return_confidence else 0.0,
                exploration_info=exploration_info,
                inference_time_ms=inference_time_ms,
                timestamp=utc_now()
            )
            
            # 缓存结果
            self._cache_inference(cache_key, response)
            
            # 记录性能指标
            self._record_performance_metrics(request.agent_id, inference_time_ms)
            
            logger.debug(f"策略推理完成: 智能体={request.agent_id}, 动作={action}, 时间={inference_time_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"策略推理失败: {e}")
            raise
    
    async def batch_inference(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """批量状态推理"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            actions = []
            details = []
            
            # 批量处理所有状态
            for state in request.states:
                inference_request = StrategyInferenceRequest(
                    agent_id=request.agent_id,
                    state=state,
                    evaluation_mode=request.evaluation_mode,
                    return_q_values=request.return_details,
                    return_confidence=request.return_details
                )
                
                response = await self.single_inference(inference_request)
                actions.append(response.action)
                
                if request.return_details:
                    details.append(response)
            
            # 计算总时间和平均时间
            end_time = asyncio.get_event_loop().time()
            total_time_ms = (end_time - start_time) * 1000
            average_time_ms = total_time_ms / len(request.states) if request.states else 0.0
            
            return BatchInferenceResponse(
                agent_id=request.agent_id,
                actions=actions,
                total_inference_time_ms=total_time_ms,
                average_inference_time_ms=average_time_ms,
                details=details if request.return_details else None
            )
            
        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            raise
    
    async def compare_strategies(self, agent_ids: List[str], state: List[float]) -> StrategyComparison:
        """比较多个智能体的策略"""
        try:
            recommendations = []
            
            # 为每个智能体获取推理结果
            for agent_id in agent_ids:
                try:
                    request = StrategyInferenceRequest(
                        agent_id=agent_id,
                        state=state,
                        evaluation_mode=True,
                        return_q_values=True,
                        return_confidence=True
                    )
                    
                    response = await self.single_inference(request)
                    
                    # 获取智能体性能统计
                    session_info = await self.qlearning_service.get_session_info(agent_id)
                    training_metrics = session_info.get("training_metrics", {})
                    
                    recommendation = {
                        "agent_id": agent_id,
                        "action": response.action,
                        "action_name": response.action_name,
                        "confidence_score": response.confidence_score,
                        "q_values": response.q_values,
                        "inference_time_ms": response.inference_time_ms,
                        "agent_performance": {
                            "mean_reward": training_metrics.get("mean_reward", 0.0),
                            "best_reward": training_metrics.get("best_reward", 0.0),
                            "episode_count": training_metrics.get("episode", 0)
                        }
                    }
                    
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    logger.warning(f"智能体 {agent_id} 策略比较失败: {e}")
                    recommendations.append({
                        "agent_id": agent_id,
                        "error": str(e),
                        "available": False
                    })
            
            # 选择最佳智能体
            best_agent_id = self._select_best_agent(recommendations)
            
            # 计算比较指标
            comparison_metrics = self._calculate_comparison_metrics(recommendations, state)
            
            return StrategyComparison(
                agent_ids=agent_ids,
                state=state,
                recommendations=recommendations,
                best_agent_id=best_agent_id,
                comparison_metrics=comparison_metrics
            )
            
        except Exception as e:
            logger.error(f"策略比较失败: {e}")
            raise
    
    async def get_strategy_insights(self, agent_id: str, num_states: int = 100) -> Dict[str, Any]:
        """获取策略洞察分析"""
        try:
            if agent_id not in self.qlearning_service.active_sessions:
                raise ValueError(f"智能体不存在: {agent_id}")
            
            session = self.qlearning_service.active_sessions[agent_id]
            agent = session.agent
            environment = session.environment
            
            # 生成随机状态进行分析
            insights = {
                "agent_id": agent_id,
                "analysis_timestamp": utc_now().isoformat(),
                "state_space_analysis": {},
                "action_preferences": {},
                "confidence_distribution": {},
                "performance_summary": {}
            }
            
            # 状态空间分析
            if hasattr(environment, 'state_space'):
                state_space = environment.state_space
                if hasattr(state_space, 'sample_states'):
                    # 采样状态进行分析
                    sample_states = state_space.sample_states(num_states)
                else:
                    # 生成随机状态
                    sample_states = [
                        np.random.uniform(-1, 1, agent.state_size).tolist()
                        for _ in range(num_states)
                    ]
            else:
                sample_states = [
                    np.random.uniform(-1, 1, agent.state_size).tolist()
                    for _ in range(num_states)
                ]
            
            # 批量推理分析
            batch_request = BatchInferenceRequest(
                agent_id=agent_id,
                states=sample_states,
                evaluation_mode=True,
                return_details=True
            )
            
            batch_response = await self.batch_inference(batch_request)
            
            # 分析动作偏好
            action_counts = {}
            confidence_scores = []
            
            if batch_response.details:
                for detail in batch_response.details:
                    action = detail.action
                    action_counts[action] = action_counts.get(action, 0) + 1
                    confidence_scores.append(detail.confidence_score)
                
                # 动作偏好统计
                total_actions = len(batch_response.details)
                insights["action_preferences"] = {
                    "distribution": {
                        str(action): count / total_actions
                        for action, count in action_counts.items()
                    },
                    "most_preferred_action": max(action_counts, key=action_counts.get),
                    "action_diversity": len(action_counts) / agent.action_size
                }
                
                # 置信度分析
                insights["confidence_distribution"] = {
                    "mean_confidence": float(np.mean(confidence_scores)),
                    "std_confidence": float(np.std(confidence_scores)),
                    "min_confidence": float(np.min(confidence_scores)),
                    "max_confidence": float(np.max(confidence_scores))
                }
            
            # 性能摘要
            session_info = await self.qlearning_service.get_session_info(agent_id)
            training_metrics = session_info.get("training_metrics", {})
            
            insights["performance_summary"] = {
                "training_episodes": training_metrics.get("episode", 0),
                "mean_reward": training_metrics.get("mean_reward", 0.0),
                "best_reward": training_metrics.get("best_reward", 0.0),
                "training_time": training_metrics.get("training_time", 0.0),
                "is_training": session_info.get("is_training", False),
                "average_inference_time_ms": batch_response.average_inference_time_ms
            }
            
            # 探索策略分析
            exploration_strategy = session.exploration_strategy
            insights["exploration_analysis"] = {
                "strategy_type": type(exploration_strategy).__name__,
                "current_exploration_rate": exploration_strategy.get_exploration_rate(),
                "step_count": exploration_strategy.step_count,
                "exploration_history_length": len(exploration_strategy.exploration_history)
            }
            
            logger.info(f"策略洞察分析完成: {agent_id}")
            return insights
            
        except Exception as e:
            logger.error(f"策略洞察分析失败: {e}")
            raise
    
    def _generate_cache_key(self, agent_id: str, state: List[float], evaluation_mode: bool) -> str:
        """生成缓存键"""
        state_str = ",".join([f"{x:.3f}" for x in state])
        return f"{agent_id}:{state_str}:{evaluation_mode}"
    
    def _get_cached_inference(self, cache_key: str) -> Optional[StrategyInferenceResponse]:
        """获取缓存推理结果"""
        if cache_key in self.inference_cache:
            cached_data = self.inference_cache[cache_key]
            
            # 检查缓存是否过期
            if utc_now() - cached_data["timestamp"] < timedelta(seconds=self.cache_ttl_seconds):
                return cached_data["response"]
            else:
                # 清理过期缓存
                del self.inference_cache[cache_key]
        
        return None
    
    def _cache_inference(self, cache_key: str, response: StrategyInferenceResponse):
        """缓存推理结果"""
        self.inference_cache[cache_key] = {
            "response": response,
            "timestamp": utc_now()
        }
        
        # 限制缓存大小
        if len(self.inference_cache) > 1000:
            # 清理最旧的缓存项
            oldest_key = min(
                self.inference_cache.keys(), 
                key=lambda k: self.inference_cache[k]["timestamp"]
            )
            del self.inference_cache[oldest_key]
    
    def _record_performance_metrics(self, agent_id: str, inference_time_ms: float):
        """记录性能指标"""
        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = []
        
        self.performance_metrics[agent_id].append(inference_time_ms)
        
        # 限制历史记录长度
        if len(self.performance_metrics[agent_id]) > 1000:
            self.performance_metrics[agent_id] = self.performance_metrics[agent_id][-500:]
    
    def _select_best_agent(self, recommendations: List[Dict[str, Any]]) -> str:
        """选择最佳智能体"""
        valid_recommendations = [r for r in recommendations if not r.get("error")]
        
        if not valid_recommendations:
            return recommendations[0]["agent_id"] if recommendations else ""
        
        # 综合评分：置信度 + 性能指标
        best_agent = None
        best_score = -float('inf')
        
        for rec in valid_recommendations:
            confidence = rec.get("confidence_score", 0.0)
            mean_reward = rec.get("agent_performance", {}).get("mean_reward", 0.0)
            
            # 综合评分
            score = confidence * 0.3 + mean_reward * 0.7
            
            if score > best_score:
                best_score = score
                best_agent = rec["agent_id"]
        
        return best_agent or valid_recommendations[0]["agent_id"]
    
    def _calculate_comparison_metrics(self, recommendations: List[Dict[str, Any]], state: List[float]) -> Dict[str, Any]:
        """计算比较指标"""
        valid_recommendations = [r for r in recommendations if not r.get("error")]
        
        if not valid_recommendations:
            return {"error": "没有有效的推理结果"}
        
        # 动作一致性
        actions = [r["action"] for r in valid_recommendations]
        action_consistency = len(set(actions)) == 1
        
        # 置信度统计
        confidences = [r.get("confidence_score", 0.0) for r in valid_recommendations]
        
        # 性能统计
        mean_rewards = [
            r.get("agent_performance", {}).get("mean_reward", 0.0)
            for r in valid_recommendations
        ]
        
        return {
            "num_agents_compared": len(valid_recommendations),
            "action_consistency": action_consistency,
            "unique_actions": len(set(actions)),
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "max": float(np.max(confidences)),
                "min": float(np.min(confidences))
            },
            "performance_stats": {
                "mean_reward_avg": float(np.mean(mean_rewards)),
                "mean_reward_std": float(np.std(mean_rewards)),
                "best_performer": max(valid_recommendations, key=lambda x: x.get("agent_performance", {}).get("mean_reward", 0.0))["agent_id"]
            },
            "state_complexity": len(state)
        }
    
    async def get_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """获取性能指标"""
        if agent_id not in self.performance_metrics:
            return {"agent_id": agent_id, "metrics": {}, "message": "暂无性能数据"}
        
        metrics = self.performance_metrics[agent_id]
        
        return {
            "agent_id": agent_id,
            "inference_performance": {
                "total_inferences": len(metrics),
                "average_time_ms": float(np.mean(metrics)),
                "median_time_ms": float(np.median(metrics)),
                "std_time_ms": float(np.std(metrics)),
                "min_time_ms": float(np.min(metrics)),
                "max_time_ms": float(np.max(metrics)),
                "percentile_95_ms": float(np.percentile(metrics, 95)),
                "percentile_99_ms": float(np.percentile(metrics, 99))
            },
            "cache_stats": {
                "total_cached_items": len(self.inference_cache),
                "cache_hit_estimation": "需要更详细的统计"
            }
        }
    
    async def clear_cache(self, agent_id: Optional[str] = None):
        """清理缓存"""
        if agent_id:
            # 清理特定智能体的缓存
            keys_to_remove = [k for k in self.inference_cache.keys() if k.startswith(f"{agent_id}:")]
            for key in keys_to_remove:
                del self.inference_cache[key]
            logger.info(f"已清理智能体 {agent_id} 的推理缓存，清理了 {len(keys_to_remove)} 项")
        else:
            # 清理所有缓存
            cache_size = len(self.inference_cache)
            self.inference_cache.clear()
            logger.info(f"已清理所有推理缓存，清理了 {cache_size} 项")
    
    async def warmup_agent(self, agent_id: str, num_warmup_states: int = 10) -> Dict[str, Any]:
        """预热智能体（预先计算一些推理结果）"""
        try:
            if agent_id not in self.qlearning_service.active_sessions:
                raise ValueError(f"智能体不存在: {agent_id}")
            
            session = self.qlearning_service.active_sessions[agent_id]
            agent = session.agent
            
            # 生成预热状态
            warmup_states = [
                np.random.uniform(-1, 1, agent.state_size).tolist()
                for _ in range(num_warmup_states)
            ]
            
            # 批量预热推理
            start_time = asyncio.get_event_loop().time()
            
            batch_request = BatchInferenceRequest(
                agent_id=agent_id,
                states=warmup_states,
                evaluation_mode=True,
                return_details=False
            )
            
            await self.batch_inference(batch_request)
            
            end_time = asyncio.get_event_loop().time()
            warmup_time = (end_time - start_time) * 1000
            
            logger.info(f"智能体预热完成: {agent_id}, 用时: {warmup_time:.2f}ms")
            
            return {
                "agent_id": agent_id,
                "warmup_states": num_warmup_states,
                "warmup_time_ms": warmup_time,
                "average_warmup_time_ms": warmup_time / num_warmup_states,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"智能体预热失败: {e}")
            raise