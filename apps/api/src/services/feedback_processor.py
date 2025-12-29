"""
反馈处理管道服务

实现多维度反馈信号标准化、时间衰减、质量评估和奖励信号生成
"""

import math
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .sentiment_analysis_service import sentiment_service, SentimentType

from src.core.logging import get_logger
logger = get_logger(__name__)

class FeedbackType(str, Enum):
    """反馈类型枚举"""
    # 隐式反馈
    CLICK = "click"
    VIEW = "view"
    DWELL_TIME = "dwell_time"
    SCROLL_DEPTH = "scroll_depth"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    
    # 显式反馈
    RATING = "rating"
    LIKE = "like"
    DISLIKE = "dislike"
    BOOKMARK = "bookmark"
    SHARE = "share"
    COMMENT = "comment"

@dataclass
class ProcessedFeedback:
    """处理后的反馈信号"""
    feedback_id: str
    user_id: str
    item_id: str
    feedback_type: FeedbackType
    normalized_value: float  # 标准化后的值 [0, 1]
    raw_value: Any
    quality_score: float  # 质量评分 [0, 1]
    time_weight: float  # 时间权重 [0, 1]
    final_weight: float  # 最终权重
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityFactors:
    """质量评分因子"""
    consistency: float = 1.0  # 一致性
    temporal_validity: float = 1.0  # 时序有效性
    anomaly_score: float = 0.0  # 异常分数 (0=正常, 1=异常)
    trust_score: float = 1.0  # 用户可信度
    sentiment_confidence: float = 1.0  # 情感分析置信度

@dataclass
class RewardSignal:
    """奖励信号"""
    user_id: str
    item_id: str
    reward_value: float  # 统一奖励信号 [-1, 1]
    confidence: float
    components: Dict[str, float]  # 各维度贡献
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class FeedbackProcessor:
    """反馈处理器"""
    
    def __init__(self):
        # 标准化配置
        self.normalization_config = {
            FeedbackType.RATING: {'min': 1, 'max': 5, 'optimal': 5},
            FeedbackType.LIKE: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.DISLIKE: {'min': 0, 'max': 1, 'optimal': 0},
            FeedbackType.BOOKMARK: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.CLICK: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.VIEW: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.DWELL_TIME: {'min': 0, 'max': 300, 'optimal': 60},  # 秒
            FeedbackType.SCROLL_DEPTH: {'min': 0, 'max': 100, 'optimal': 80},  # 百分比
            FeedbackType.HOVER: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.FOCUS: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.BLUR: {'min': 0, 'max': 1, 'optimal': 0},
            FeedbackType.SHARE: {'min': 0, 'max': 1, 'optimal': 1},
            FeedbackType.COMMENT: {'min': -1, 'max': 1, 'optimal': 1}  # 基于情感分析
        }
        
        # 权重配置
        self.weight_config = {
            FeedbackType.RATING: 1.0,
            FeedbackType.LIKE: 0.8,
            FeedbackType.DISLIKE: 0.8,
            FeedbackType.BOOKMARK: 0.9,
            FeedbackType.CLICK: 0.3,
            FeedbackType.VIEW: 0.2,
            FeedbackType.DWELL_TIME: 0.6,
            FeedbackType.SCROLL_DEPTH: 0.4,
            FeedbackType.HOVER: 0.2,
            FeedbackType.FOCUS: 0.2,
            FeedbackType.BLUR: 0.1,
            FeedbackType.SHARE: 0.9,
            FeedbackType.COMMENT: 0.9
        }
        
        # 时间衰减配置
        self.decay_config = {
            'half_life_days': 7.0,  # 半衰期天数
            'min_weight': 0.1  # 最小权重
        }
        
        # 用户历史记录 (实际应该从数据库加载)
        self.user_history: Dict[str, List[ProcessedFeedback]] = {}
        self.user_stats: Dict[str, Dict] = {}
    
    async def process_feedback_batch(self, feedbacks: List[Dict]) -> List[ProcessedFeedback]:
        """处理反馈批次"""
        processed_feedbacks = []
        
        for feedback_data in feedbacks:
            try:
                processed = await self.process_single_feedback(feedback_data)
                if processed:
                    processed_feedbacks.append(processed)
            except Exception as e:
                logger.error(f"处理反馈失败: {feedback_data.get('feedback_id', 'unknown')}, 错误: {e}")
        
        return processed_feedbacks
    
    async def process_single_feedback(self, feedback_data: Dict) -> Optional[ProcessedFeedback]:
        """处理单个反馈"""
        try:
            feedback_type = FeedbackType(feedback_data['feedback_type'])
            
            # 1. 标准化反馈值
            normalized_value = self.normalize_feedback_value(
                feedback_type, 
                feedback_data['value']
            )
            
            # 2. 计算质量分数
            quality_score = await self.calculate_quality_score(feedback_data)
            
            # 3. 计算时间权重
            timestamp = datetime.fromisoformat(feedback_data['timestamp']) if isinstance(feedback_data['timestamp'], str) else feedback_data['timestamp']
            time_weight = self.calculate_time_decay(timestamp)
            
            # 4. 计算最终权重
            base_weight = self.weight_config.get(feedback_type, 1.0)
            final_weight = base_weight * quality_score * time_weight
            
            processed = ProcessedFeedback(
                feedback_id=feedback_data['feedback_id'],
                user_id=feedback_data['user_id'],
                item_id=feedback_data.get('item_id', 'unknown'),
                feedback_type=feedback_type,
                normalized_value=normalized_value,
                raw_value=feedback_data['value'],
                quality_score=quality_score,
                time_weight=time_weight,
                final_weight=final_weight,
                timestamp=timestamp,
                context=feedback_data.get('context', {}),
                metadata=feedback_data.get('metadata', {})
            )
            
            # 更新用户历史
            self.update_user_history(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"处理反馈失败: {e}")
            return None
    
    def normalize_feedback_value(self, feedback_type: FeedbackType, raw_value: Any) -> float:
        """标准化反馈值到[0,1]区间"""
        config = self.normalization_config.get(feedback_type)
        if not config:
            return 0.5  # 默认中性值
        
        try:
            if feedback_type == FeedbackType.COMMENT:
                # 评论使用情感分析
                if isinstance(raw_value, str):
                    sentiment_result = sentiment_service.analyze_sentiment(raw_value)
                    if sentiment_result.sentiment == SentimentType.POSITIVE:
                        return 0.5 + sentiment_result.intensity * 0.5
                    elif sentiment_result.sentiment == SentimentType.NEGATIVE:
                        return 0.5 - sentiment_result.intensity * 0.5
                    else:
                        return 0.5
                return 0.5
            
            elif feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE, 
                                   FeedbackType.BOOKMARK, FeedbackType.CLICK, 
                                   FeedbackType.VIEW, FeedbackType.HOVER, 
                                   FeedbackType.FOCUS, FeedbackType.BLUR, 
                                   FeedbackType.SHARE]:
                # 布尔值类型
                if isinstance(raw_value, bool):
                    return 1.0 if raw_value else 0.0
                elif isinstance(raw_value, (int, float)):
                    return float(raw_value)
                return 0.0
            
            elif feedback_type in [FeedbackType.RATING, FeedbackType.DWELL_TIME, 
                                   FeedbackType.SCROLL_DEPTH]:
                # 数值类型
                value = float(raw_value)
                min_val = config['min']
                max_val = config['max']
                
                # 线性标准化
                normalized = (value - min_val) / (max_val - min_val)
                return max(0.0, min(1.0, normalized))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"标准化反馈值失败: {feedback_type}, {raw_value}, 错误: {e}")
            return 0.5
    
    async def calculate_quality_score(self, feedback_data: Dict) -> float:
        """计算反馈质量分数"""
        try:
            factors = QualityFactors()
            
            user_id = feedback_data['user_id']
            feedback_type = FeedbackType(feedback_data['feedback_type'])
            
            # 1. 一致性检查
            factors.consistency = self.check_consistency(user_id, feedback_data)
            
            # 2. 时序有效性
            factors.temporal_validity = self.check_temporal_validity(feedback_data)
            
            # 3. 异常检测
            factors.anomaly_score = self.detect_anomaly(user_id, feedback_data)
            
            # 4. 用户可信度
            factors.trust_score = self.get_user_trust_score(user_id)
            
            # 5. 情感分析置信度（仅评论）
            if feedback_type == FeedbackType.COMMENT:
                sentiment_result = sentiment_service.analyze_sentiment(str(feedback_data['value']))
                factors.sentiment_confidence = sentiment_result.confidence
            
            # 综合质量分数
            quality_score = (
                factors.consistency * 0.25 +
                factors.temporal_validity * 0.15 +
                (1 - factors.anomaly_score) * 0.25 +
                factors.trust_score * 0.25 +
                factors.sentiment_confidence * 0.1
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"计算质量分数失败: {e}")
            return 0.5
    
    def check_consistency(self, user_id: str, feedback_data: Dict) -> float:
        """检查用户反馈一致性"""
        try:
            user_feedbacks = self.user_history.get(user_id, [])
            if len(user_feedbacks) < 3:
                return 1.0  # 新用户给满分
            
            feedback_type = FeedbackType(feedback_data['feedback_type'])
            item_id = feedback_data.get('item_id')
            
            # 检查同类型反馈的一致性
            similar_feedbacks = [
                f for f in user_feedbacks 
                if f.feedback_type == feedback_type and f.item_id == item_id
            ]
            
            if len(similar_feedbacks) == 0:
                return 1.0
            
            # 计算标准差
            values = [f.normalized_value for f in similar_feedbacks]
            if len(values) > 1:
                std_dev = np.std(values)
                consistency = max(0.0, 1.0 - std_dev * 2)  # 标准差越小一致性越高
                return consistency
            
            return 1.0
            
        except Exception as e:
            logger.error(f"检查一致性失败: {e}")
            return 0.5
    
    def check_temporal_validity(self, feedback_data: Dict) -> float:
        """检查时序有效性"""
        try:
            timestamp = datetime.fromisoformat(feedback_data['timestamp']) if isinstance(feedback_data['timestamp'], str) else feedback_data['timestamp']
            now = utc_now()
            
            # 检查时间是否合理
            if timestamp > now:
                return 0.0  # 未来时间，无效
            
            # 检查是否过旧
            age_hours = (now - timestamp).total_seconds() / 3600
            if age_hours > 24 * 30:  # 超过30天
                return 0.5
            
            return 1.0
            
        except Exception as e:
            logger.error(f"检查时序有效性失败: {e}")
            return 0.5
    
    def detect_anomaly(self, user_id: str, feedback_data: Dict) -> float:
        """异常检测"""
        try:
            anomaly_score = 0.0
            
            # 1. 检查频率异常
            user_feedbacks = self.user_history.get(user_id, [])
            recent_feedbacks = [
                f for f in user_feedbacks
                if (utc_now() - f.timestamp).total_seconds() < 3600  # 1小时内
            ]
            
            if len(recent_feedbacks) > 50:  # 1小时内超过50个反馈
                anomaly_score += 0.5
            
            # 2. 检查值异常
            feedback_type = FeedbackType(feedback_data['feedback_type'])
            value = feedback_data['value']
            
            if feedback_type == FeedbackType.RATING:
                if not (1 <= value <= 5):
                    anomaly_score += 0.3
            
            # 3. 检查模式异常
            if len(user_feedbacks) > 10:
                type_counts = {}
                for f in user_feedbacks[-10:]:  # 最近10个反馈
                    type_counts[f.feedback_type] = type_counts.get(f.feedback_type, 0) + 1
                
                # 如果某种类型占比过高
                max_ratio = max(type_counts.values()) / len(user_feedbacks[-10:])
                if max_ratio > 0.8:
                    anomaly_score += 0.2
            
            return min(1.0, anomaly_score)
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return 0.0
    
    def get_user_trust_score(self, user_id: str) -> float:
        """获取用户可信度分数"""
        try:
            user_feedbacks = self.user_history.get(user_id, [])
            
            if len(user_feedbacks) == 0:
                return 0.8  # 新用户中等信任度
            
            # 基于历史行为计算信任度
            total_feedbacks = len(user_feedbacks)
            avg_quality = sum(f.quality_score for f in user_feedbacks) / total_feedbacks
            
            # 活跃度
            activity_days = len(set(f.timestamp.date() for f in user_feedbacks))
            activity_score = min(1.0, activity_days / 30)  # 30天内的活跃度
            
            # 多样性
            unique_types = len(set(f.feedback_type for f in user_feedbacks))
            diversity_score = min(1.0, unique_types / 5)  # 最多5种类型
            
            trust_score = (avg_quality * 0.5 + activity_score * 0.3 + diversity_score * 0.2)
            
            return max(0.1, min(1.0, trust_score))
            
        except Exception as e:
            logger.error(f"计算用户信任度失败: {e}")
            return 0.5
    
    def calculate_time_decay(self, timestamp: datetime, half_life_days: Optional[float] = None) -> float:
        """计算时间衰减权重"""
        try:
            half_life = half_life_days or self.decay_config['half_life_days']
            min_weight = self.decay_config['min_weight']
            
            now = utc_now()
            age_hours = (now - timestamp).total_seconds() / 3600
            age_days = age_hours / 24
            
            # 指数衰减
            decay_weight = math.exp(-math.log(2) * age_days / half_life)
            
            # 应用最小权重
            return max(min_weight, decay_weight)
            
        except Exception as e:
            logger.error(f"计算时间衰减失败: {e}")
            return 1.0
    
    def update_user_history(self, processed_feedback: ProcessedFeedback):
        """更新用户历史记录"""
        user_id = processed_feedback.user_id
        
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append(processed_feedback)
        
        # 保持历史记录在合理大小
        if len(self.user_history[user_id]) > 1000:
            self.user_history[user_id] = self.user_history[user_id][-1000:]
    
    def generate_reward_signal(self, 
                              user_id: str, 
                              item_id: str, 
                              time_window_hours: int = 24) -> RewardSignal:
        """生成统一奖励信号"""
        try:
            # 获取时间窗口内的反馈
            cutoff_time = utc_now() - timedelta(hours=time_window_hours)
            user_feedbacks = self.user_history.get(user_id, [])
            
            relevant_feedbacks = [
                f for f in user_feedbacks
                if f.item_id == item_id and f.timestamp >= cutoff_time
            ]
            
            if not relevant_feedbacks:
                return RewardSignal(
                    user_id=user_id,
                    item_id=item_id,
                    reward_value=0.0,
                    confidence=0.0,
                    components={},
                    timestamp=utc_now()
                )
            
            # 按类型聚合反馈
            type_contributions = {}
            total_weight = 0.0
            weighted_sum = 0.0
            
            for feedback in relevant_feedbacks:
                feedback_type = feedback.feedback_type
                
                # 计算贡献值 [-1, 1]
                if feedback_type in [FeedbackType.DISLIKE, FeedbackType.BLUR]:
                    contribution = -(feedback.normalized_value * 2 - 1)  # 负面反馈取反
                else:
                    contribution = feedback.normalized_value * 2 - 1  # [0,1] -> [-1,1]
                
                weight = feedback.final_weight
                weighted_contribution = contribution * weight
                
                if feedback_type not in type_contributions:
                    type_contributions[feedback_type] = {'value': 0.0, 'weight': 0.0, 'count': 0}
                
                type_contributions[feedback_type]['value'] += weighted_contribution
                type_contributions[feedback_type]['weight'] += weight
                type_contributions[feedback_type]['count'] += 1
                
                weighted_sum += weighted_contribution
                total_weight += weight
            
            # 计算最终奖励值
            if total_weight > 0:
                reward_value = weighted_sum / total_weight
            else:
                reward_value = 0.0
            
            # 计算置信度
            confidence = self.calculate_reward_confidence(relevant_feedbacks, total_weight)
            
            # 标准化到[-1, 1]
            reward_value = max(-1.0, min(1.0, reward_value))
            
            return RewardSignal(
                user_id=user_id,
                item_id=item_id,
                reward_value=reward_value,
                confidence=confidence,
                components={str(k): v['value'] for k, v in type_contributions.items()},
                timestamp=utc_now(),
                metadata={
                    'total_feedbacks': len(relevant_feedbacks),
                    'total_weight': total_weight,
                    'time_window_hours': time_window_hours,
                    'type_distribution': {str(k): v['count'] for k, v in type_contributions.items()}
                }
            )
            
        except Exception as e:
            logger.error(f"生成奖励信号失败: {e}")
            return RewardSignal(
                user_id=user_id,
                item_id=item_id,
                reward_value=0.0,
                confidence=0.0,
                components={},
                timestamp=utc_now()
            )
    
    def calculate_reward_confidence(self, feedbacks: List[ProcessedFeedback], total_weight: float) -> float:
        """计算奖励信号置信度"""
        try:
            if not feedbacks:
                return 0.0
            
            # 基于反馈数量的置信度
            count_confidence = min(1.0, len(feedbacks) / 10)  # 10个反馈达到满置信度
            
            # 基于权重的置信度
            weight_confidence = min(1.0, total_weight / 5.0)  # 权重总和5达到满置信度
            
            # 基于质量的置信度
            avg_quality = sum(f.quality_score for f in feedbacks) / len(feedbacks)
            quality_confidence = avg_quality
            
            # 基于多样性的置信度
            unique_types = len(set(f.feedback_type for f in feedbacks))
            diversity_confidence = min(1.0, unique_types / 3)  # 3种类型达到满置信度
            
            # 综合置信度
            confidence = (
                count_confidence * 0.3 +
                weight_confidence * 0.3 +
                quality_confidence * 0.2 +
                diversity_confidence * 0.2
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"计算奖励置信度失败: {e}")
            return 0.0

# 全局服务实例
feedback_processor = FeedbackProcessor()
