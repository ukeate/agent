"""
奖励信号生成器

基于多维度用户反馈生成统一的奖励信号，为强化学习算法提供高质量的奖励输入。
支持多种奖励计算策略，包括加权聚合、时间衰减、质量调整等。
"""

import math
import numpy as np
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from src.models.schemas.feedback import FeedbackType
from src.core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class RewardStrategy(str, Enum):
    """奖励计算策略"""
    SIMPLE_AVERAGE = "simple_average"      # 简单平均
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    NORMALIZED_SUM = "normalized_sum"      # 归一化求和
    EXPONENTIAL_DECAY = "exponential_decay" # 指数衰减
    CONTEXTUAL_BOOST = "contextual_boost"  # 上下文增强

@dataclass
class FeedbackSignal:
    """标准化的反馈信号"""
    feedback_type: FeedbackType
    normalized_value: float
    timestamp: datetime
    quality_score: float = 1.0
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    item_id: Optional[str] = None

@dataclass
class RewardConfig:
    """奖励计算配置"""
    strategy: RewardStrategy = RewardStrategy.WEIGHTED_AVERAGE
    
    # 权重配置
    feedback_weights: Dict[FeedbackType, float] = None
    
    # 时间衰减配置
    time_decay_enabled: bool = True
    time_decay_half_life_hours: float = 24.0  # 24小时半衰期
    
    # 质量调整配置
    quality_adjustment_enabled: bool = True
    min_quality_threshold: float = 0.3
    
    # 归一化配置
    normalize_output: bool = True
    output_range: Tuple[float, float] = (-1.0, 1.0)
    
    # 上下文增强配置
    context_boost_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.feedback_weights is None:
            # 默认权重配置
            self.feedback_weights = {
                # 显式反馈权重更高
                FeedbackType.RATING: 1.0,
                FeedbackType.LIKE: 0.8,
                FeedbackType.DISLIKE: -0.8,
                FeedbackType.BOOKMARK: 0.6,
                FeedbackType.SHARE: 0.7,
                FeedbackType.COMMENT: 0.5,
                
                # 隐式反馈权重较低
                FeedbackType.CLICK: 0.3,
                FeedbackType.VIEW: 0.2,
                FeedbackType.DWELL_TIME: 0.4,
                FeedbackType.SCROLL_DEPTH: 0.2,
                FeedbackType.HOVER: 0.1,
            }
        
        if self.context_boost_factors is None:
            self.context_boost_factors = {
                "new_user": 1.2,      # 新用户反馈权重提升
                "premium_user": 1.1,  # 付费用户权重提升
                "organic_traffic": 1.0,
                "search_traffic": 0.9,
                "social_traffic": 0.8,
            }

class FeedbackNormalizer:
    """反馈值归一化器"""
    
    @staticmethod
    def normalize_feedback_value(feedback_type: FeedbackType, raw_value: Any) -> float:
        """将原始反馈值归一化到 [0, 1] 范围"""
        try:
            if feedback_type == FeedbackType.RATING:
                # 评分 1-5 -> 0-1
                return (float(raw_value) - 1.0) / 4.0
                
            elif feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE, 
                                   FeedbackType.BOOKMARK, FeedbackType.SHARE, 
                                   FeedbackType.CLICK, FeedbackType.VIEW]:
                # 布尔值直接转换
                return float(bool(raw_value))
                
            elif feedback_type == FeedbackType.DWELL_TIME:
                # 停留时间，使用对数归一化，最大值假设为1800秒(30分钟)
                max_time = 1800.0
                normalized = min(float(raw_value) / max_time, 1.0)
                return normalized
                
            elif feedback_type == FeedbackType.SCROLL_DEPTH:
                # 滚动深度百分比
                return min(float(raw_value) / 100.0, 1.0)
                
            elif feedback_type == FeedbackType.HOVER:
                # 悬停时间，最大值假设为30秒
                max_hover = 30.0
                return min(float(raw_value) / max_hover, 1.0)
                
            elif feedback_type == FeedbackType.COMMENT:
                # 评论长度，使用对数归一化
                if isinstance(raw_value, str):
                    comment_length = len(raw_value.strip())
                    # 使用对数函数，长度越长权重递减
                    return min(math.log(comment_length + 1) / math.log(500), 1.0)
                else:
                    return 0.0
                    
            else:
                logger.warning(f"Unknown feedback type for normalization: {feedback_type}")
                return 0.0
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error normalizing feedback value {raw_value} for type {feedback_type}: {e}")
            return 0.0

class TimeDecayCalculator:
    """时间衰减计算器"""
    
    @staticmethod
    def calculate_decay_factor(
        feedback_time: datetime, 
        current_time: Optional[datetime] = None,
        half_life_hours: float = 24.0
    ) -> float:
        """计算时间衰减因子"""
        if current_time is None:
            current_time = utc_now()
        
        # 计算时间差（小时）
        time_diff = (current_time - feedback_time).total_seconds() / 3600.0
        
        if time_diff <= 0:
            return 1.0
        
        # 指数衰减: decay_factor = 0.5 ^ (time_diff / half_life)
        decay_factor = math.pow(0.5, time_diff / half_life_hours)
        
        return max(decay_factor, 0.01)  # 最小衰减因子为0.01

class ContextBoostCalculator:
    """上下文增强计算器"""
    
    @staticmethod
    def calculate_context_boost(
        context: Dict[str, Any], 
        boost_factors: Dict[str, float]
    ) -> float:
        """计算上下文增强因子"""
        boost_factor = 1.0
        
        # 用户类型增强
        user_type = context.get("user_type")
        if user_type and user_type in boost_factors:
            boost_factor *= boost_factors[user_type]
        
        # 流量来源增强
        traffic_source = context.get("traffic_source")
        if traffic_source and traffic_source in boost_factors:
            boost_factor *= boost_factors[traffic_source]
        
        # 设备类型增强
        device_type = context.get("device_type")
        if device_type:
            device_boost = boost_factors.get(f"{device_type}_device", 1.0)
            boost_factor *= device_boost
        
        # 地理位置增强
        country = context.get("country")
        if country:
            country_boost = boost_factors.get(f"country_{country.lower()}", 1.0)
            boost_factor *= country_boost
        
        return max(boost_factor, 0.1)  # 最小增强因子为0.1

class RewardSignalGenerator:
    """奖励信号生成器主类"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.normalizer = FeedbackNormalizer()
        self.decay_calculator = TimeDecayCalculator()
        self.context_calculator = ContextBoostCalculator()
        
        # 统计信息
        self.stats = {
            "total_generated": 0,
            "strategy_usage": defaultdict(int),
            "avg_reward": 0.0,
            "reward_distribution": defaultdict(int)
        }
        
    async def generate_reward(
        self, 
        feedbacks: List[FeedbackSignal],
        strategy: Optional[RewardStrategy] = None
    ) -> float:
        """从多维反馈生成统一奖励信号"""
        if not feedbacks:
            return 0.0
        
        strategy = strategy or self.config.strategy
        self.stats["strategy_usage"][strategy] += 1
        self.stats["total_generated"] += 1
        
        try:
            if strategy == RewardStrategy.SIMPLE_AVERAGE:
                reward = await self._simple_average_reward(feedbacks)
                
            elif strategy == RewardStrategy.WEIGHTED_AVERAGE:
                reward = await self._weighted_average_reward(feedbacks)
                
            elif strategy == RewardStrategy.NORMALIZED_SUM:
                reward = await self._normalized_sum_reward(feedbacks)
                
            elif strategy == RewardStrategy.EXPONENTIAL_DECAY:
                reward = await self._exponential_decay_reward(feedbacks)
                
            elif strategy == RewardStrategy.CONTEXTUAL_BOOST:
                reward = await self._contextual_boost_reward(feedbacks)
                
            else:
                logger.warning(f"Unknown reward strategy: {strategy}, using default")
                reward = await self._weighted_average_reward(feedbacks)
            
            # 归一化输出
            if self.config.normalize_output:
                reward = self._normalize_reward_output(reward)
            
            # 更新统计
            self._update_stats(reward)
            
            logger.debug(f"Generated reward {reward:.4f} using strategy {strategy}")
            return reward
            
        except Exception as e:
            logger.error(f"Error generating reward: {e}")
            return 0.0
    
    async def _simple_average_reward(self, feedbacks: List[FeedbackSignal]) -> float:
        """简单平均奖励计算"""
        if not feedbacks:
            return 0.0
        
        total_reward = 0.0
        valid_count = 0
        
        for feedback in feedbacks:
            if feedback.quality_score < self.config.min_quality_threshold:
                continue
                
            normalized_value = self.normalizer.normalize_feedback_value(
                feedback.feedback_type, feedback.raw_value
            )
            
            # 处理负面反馈
            if feedback.feedback_type == FeedbackType.DISLIKE:
                normalized_value = -normalized_value
            
            total_reward += normalized_value
            valid_count += 1
        
        return total_reward / valid_count if valid_count > 0 else 0.0
    
    async def _weighted_average_reward(self, feedbacks: List[FeedbackSignal]) -> float:
        """加权平均奖励计算"""
        if not feedbacks:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for feedback in feedbacks:
            if feedback.quality_score < self.config.min_quality_threshold:
                continue
            
            # 获取基础权重
            base_weight = self.config.feedback_weights.get(feedback.feedback_type, 0.1)
            
            # 质量调整
            if self.config.quality_adjustment_enabled:
                quality_weight = base_weight * feedback.quality_score
            else:
                quality_weight = base_weight
            
            # 时间衰减
            if self.config.time_decay_enabled:
                decay_factor = self.decay_calculator.calculate_decay_factor(
                    feedback.timestamp, 
                    half_life_hours=self.config.time_decay_half_life_hours
                )
                time_adjusted_weight = quality_weight * decay_factor
            else:
                time_adjusted_weight = quality_weight
            
            # 归一化反馈值
            normalized_value = self.normalizer.normalize_feedback_value(
                feedback.feedback_type, feedback.raw_value
            )
            
            # 处理负面反馈
            if feedback.feedback_type == FeedbackType.DISLIKE:
                normalized_value = -normalized_value
            
            weighted_sum += normalized_value * time_adjusted_weight
            weight_sum += time_adjusted_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    async def _normalized_sum_reward(self, feedbacks: List[FeedbackSignal]) -> float:
        """归一化求和奖励计算"""
        if not feedbacks:
            return 0.0
        
        total_reward = 0.0
        
        for feedback in feedbacks:
            if feedback.quality_score < self.config.min_quality_threshold:
                continue
            
            # 获取权重
            weight = self.config.feedback_weights.get(feedback.feedback_type, 0.1)
            
            # 归一化反馈值
            normalized_value = self.normalizer.normalize_feedback_value(
                feedback.feedback_type, feedback.raw_value
            )
            
            # 处理负面反馈
            if feedback.feedback_type == FeedbackType.DISLIKE:
                normalized_value = -normalized_value
            
            # 质量和权重调整
            adjusted_value = normalized_value * weight * feedback.quality_score
            
            total_reward += adjusted_value
        
        # 除以反馈数量进行归一化
        return total_reward / len(feedbacks)
    
    async def _exponential_decay_reward(self, feedbacks: List[FeedbackSignal]) -> float:
        """指数衰减奖励计算"""
        if not feedbacks:
            return 0.0
        
        # 按时间排序，最新的反馈权重最高
        sorted_feedbacks = sorted(feedbacks, key=lambda x: x.timestamp, reverse=True)
        
        total_reward = 0.0
        total_weight = 0.0
        
        for i, feedback in enumerate(sorted_feedbacks):
            if feedback.quality_score < self.config.min_quality_threshold:
                continue
            
            # 指数衰减权重
            position_weight = math.exp(-i / len(sorted_feedbacks))
            
            # 基础权重
            base_weight = self.config.feedback_weights.get(feedback.feedback_type, 0.1)
            
            # 综合权重
            final_weight = position_weight * base_weight * feedback.quality_score
            
            # 归一化反馈值
            normalized_value = self.normalizer.normalize_feedback_value(
                feedback.feedback_type, feedback.raw_value
            )
            
            # 处理负面反馈
            if feedback.feedback_type == FeedbackType.DISLIKE:
                normalized_value = -normalized_value
            
            total_reward += normalized_value * final_weight
            total_weight += final_weight
        
        return total_reward / total_weight if total_weight > 0 else 0.0
    
    async def _contextual_boost_reward(self, feedbacks: List[FeedbackSignal]) -> float:
        """上下文增强奖励计算"""
        if not feedbacks:
            return 0.0
        
        total_reward = 0.0
        total_weight = 0.0
        
        for feedback in feedbacks:
            if feedback.quality_score < self.config.min_quality_threshold:
                continue
            
            # 基础权重
            base_weight = self.config.feedback_weights.get(feedback.feedback_type, 0.1)
            
            # 上下文增强
            context_boost = self.context_calculator.calculate_context_boost(
                feedback.context, self.config.context_boost_factors
            )
            
            # 质量和时间衰减
            quality_weight = base_weight * feedback.quality_score
            
            if self.config.time_decay_enabled:
                decay_factor = self.decay_calculator.calculate_decay_factor(
                    feedback.timestamp,
                    half_life_hours=self.config.time_decay_half_life_hours
                )
                time_weight = quality_weight * decay_factor
            else:
                time_weight = quality_weight
            
            # 最终权重
            final_weight = time_weight * context_boost
            
            # 归一化反馈值
            normalized_value = self.normalizer.normalize_feedback_value(
                feedback.feedback_type, feedback.raw_value
            )
            
            # 处理负面反馈
            if feedback.feedback_type == FeedbackType.DISLIKE:
                normalized_value = -normalized_value
            
            total_reward += normalized_value * final_weight
            total_weight += final_weight
        
        return total_reward / total_weight if total_weight > 0 else 0.0
    
    def _normalize_reward_output(self, reward: float) -> float:
        """归一化奖励输出到指定范围"""
        min_val, max_val = self.config.output_range
        
        # 使用tanh函数进行软归一化
        normalized = math.tanh(reward)
        
        # 映射到目标范围
        scaled = min_val + (normalized + 1.0) * (max_val - min_val) / 2.0
        
        return scaled
    
    def _update_stats(self, reward: float):
        """更新统计信息"""
        # 更新平均奖励
        total = self.stats["total_generated"]
        current_avg = self.stats["avg_reward"]
        self.stats["avg_reward"] = (current_avg * (total - 1) + reward) / total
        
        # 更新奖励分布
        reward_bin = int(reward * 10) / 10.0  # 0.1精度的分箱
        self.stats["reward_distribution"][reward_bin] += 1
    
    async def generate_multi_item_rewards(
        self,
        item_feedbacks: Dict[str, List[FeedbackSignal]]
    ) -> Dict[str, float]:
        """为多个推荐项生成奖励信号"""
        rewards = {}
        
        for item_id, feedbacks in item_feedbacks.items():
            reward = await self.generate_reward(feedbacks)
            rewards[item_id] = reward
        
        return rewards
    
    async def get_reward_stats(self) -> Dict[str, Any]:
        """获取奖励生成统计信息"""
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "time_decay_enabled": self.config.time_decay_enabled,
                "quality_adjustment_enabled": self.config.quality_adjustment_enabled,
                "normalize_output": self.config.normalize_output,
                "output_range": self.config.output_range
            },
            "stats": dict(self.stats),
            "strategy_usage": dict(self.stats["strategy_usage"]),
            "reward_distribution": dict(self.stats["reward_distribution"])
        }

# 全局生成器实例
_reward_generator: Optional[RewardSignalGenerator] = None

def get_reward_generator(config: Optional[RewardConfig] = None) -> RewardSignalGenerator:
    """获取全局奖励生成器实例"""
    global _reward_generator
    if _reward_generator is None or config is not None:
        _reward_generator = RewardSignalGenerator(config)
    return _reward_generator

# 便利函数
async def generate_reward_from_feedbacks(
    feedbacks: List[FeedbackSignal],
    strategy: Optional[RewardStrategy] = None,
    config: Optional[RewardConfig] = None
) -> float:
    """便利函数：从反馈列表生成奖励信号"""
    generator = get_reward_generator(config)
    return await generator.generate_reward(feedbacks, strategy)
