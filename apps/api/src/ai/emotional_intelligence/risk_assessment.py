"""
情感健康风险评估系统
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from .models import RiskAssessment, RiskFactor, RiskLevel, DecisionContext
from ..emotion_modeling.models import EmotionState, PersonalityProfile

from src.core.logging import get_logger
logger = get_logger(__name__)

class RiskAssessmentEngine:
    """情感健康风险评估引擎"""
    
    def __init__(self):
        # 风险指标权重配置
        self.risk_weights = {
            'depression_indicators': 0.3,
            'anxiety_indicators': 0.25,
            'emotional_volatility': 0.2,
            'social_isolation': 0.15,
            'behavioral_changes': 0.1
        }
        
        # 风险阈值配置
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        }
        
        # 抑郁症状关键词
        self.depression_keywords = [
            '绝望', '无助', '空虚', '麻木', '无意义', '孤独',
            '疲惫', '失眠', '食欲不振', '不想动', '没有希望'
        ]
        
        # 焦虑症状关键词
        self.anxiety_keywords = [
            '担心', '紧张', '不安', '恐惧', '心跳加速', '出汗',
            '颤抖', '窒息感', '头晕', '恶心', '控制不住'
        ]
    
    async def assess_comprehensive_risk(
        self,
        user_id: str,
        emotion_history: List[EmotionState],
        personality_profile: Optional[PersonalityProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        综合风险评估
        
        Args:
            user_id: 用户ID
            emotion_history: 情感历史
            personality_profile: 个性画像
            context: 上下文信息
            
        Returns:
            风险评估结果
        """
        try:
            risk_factors = []
            
            # 1. 抑郁指标分析
            depression_score = await self._analyze_depression_indicators(emotion_history, context)
            if depression_score > 0.1:
                risk_factors.append(RiskFactor(
                    factor_type='depression_indicators',
                    score=depression_score,
                    evidence=await self._extract_depression_evidence(emotion_history),
                    weight=self.risk_weights['depression_indicators'],
                    description=f"抑郁症状评估分数: {depression_score:.2f}"
                ))
            
            # 2. 焦虑指标分析
            anxiety_score = await self._analyze_anxiety_indicators(emotion_history, context)
            if anxiety_score > 0.1:
                risk_factors.append(RiskFactor(
                    factor_type='anxiety_indicators',
                    score=anxiety_score,
                    evidence=await self._extract_anxiety_evidence(emotion_history),
                    weight=self.risk_weights['anxiety_indicators'],
                    description=f"焦虑症状评估分数: {anxiety_score:.2f}"
                ))
            
            # 3. 情感波动性分析
            volatility_score = self._calculate_emotional_volatility(emotion_history)
            if volatility_score > 0.2:
                risk_factors.append(RiskFactor(
                    factor_type='emotional_volatility',
                    score=volatility_score,
                    evidence={'volatility_index': volatility_score},
                    weight=self.risk_weights['emotional_volatility'],
                    description=f"情感波动性指数: {volatility_score:.2f}"
                ))
            
            # 4. 社交孤立分析
            isolation_score = await self._analyze_social_isolation(emotion_history, context)
            if isolation_score > 0.2:
                risk_factors.append(RiskFactor(
                    factor_type='social_isolation',
                    score=isolation_score,
                    evidence={'isolation_indicators': True},
                    weight=self.risk_weights['social_isolation'],
                    description=f"社交孤立风险: {isolation_score:.2f}"
                ))
            
            # 5. 行为变化分析
            behavioral_score = await self._analyze_behavioral_changes(emotion_history, context)
            if behavioral_score > 0.2:
                risk_factors.append(RiskFactor(
                    factor_type='behavioral_changes',
                    score=behavioral_score,
                    evidence={'behavioral_risk_indicators': True},
                    weight=self.risk_weights['behavioral_changes'],
                    description=f"行为变化风险: {behavioral_score:.2f}"
                ))
            
            # 计算综合风险分数
            total_risk = sum(
                factor.score * factor.weight 
                for factor in risk_factors
            )
            total_risk = min(1.0, total_risk)  # 确保不超过1.0
            
            # 确定风险等级
            risk_level = self._determine_risk_level(total_risk)
            
            # 生成推荐行动
            recommended_actions = await self._generate_recommendations(risk_level, risk_factors)
            
            # 计算预测置信度
            prediction_confidence = self._calculate_prediction_confidence(risk_factors, len(emotion_history))
            
            # 构建评估结果
            assessment = RiskAssessment(
                user_id=user_id,
                risk_level=risk_level,
                risk_score=total_risk,
                risk_factors=risk_factors,
                prediction_confidence=prediction_confidence,
                recommended_actions=recommended_actions,
                alert_triggered=risk_level in ['high', 'critical'],
                assessment_details={
                    'assessment_method': 'comprehensive_multi_factor',
                    'data_completeness': min(1.0, len(emotion_history) / 100),
                    'personality_included': personality_profile is not None,
                    'context_factors': context is not None
                }
            )
            
            logger.info(f"风险评估完成 - 用户: {user_id}, 风险等级: {risk_level}, 分数: {total_risk:.3f}")
            return assessment
            
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            raise
    
    async def predict_crisis_probability(
        self,
        emotion_history: List[EmotionState],
        time_horizon: timedelta = timedelta(hours=24)
    ) -> Tuple[float, Dict[str, Any]]:
        """
        预测危机发生概率
        
        Args:
            emotion_history: 情感历史
            time_horizon: 预测时间范围
            
        Returns:
            (危机概率, 分析详情)
        """
        try:
            if len(emotion_history) < 5:
                return 0.1, {'reason': 'insufficient_data'}
            
            # 分析情感趋势
            recent_emotions = emotion_history[-10:] if len(emotion_history) >= 10 else emotion_history
            
            # 计算负面情感趋势
            negative_trend = self._calculate_negative_trend(recent_emotions)
            
            # 计算情感恶化速度
            deterioration_rate = self._calculate_deterioration_rate(recent_emotions)
            
            # 检测危机模式
            crisis_patterns = self._detect_crisis_patterns(recent_emotions)
            
            # 综合计算危机概率
            crisis_probability = min(1.0, (
                negative_trend * 0.4 +
                deterioration_rate * 0.3 +
                crisis_patterns * 0.3
            ))
            
            analysis_details = {
                'negative_trend': negative_trend,
                'deterioration_rate': deterioration_rate,
                'crisis_patterns': crisis_patterns,
                'data_points': len(recent_emotions),
                'prediction_method': 'trend_based_analysis',
                'time_horizon_hours': time_horizon.total_seconds() / 3600
            }
            
            return crisis_probability, analysis_details
            
        except Exception as e:
            logger.error(f"危机概率预测失败: {str(e)}")
            return 0.0, {'error': str(e)}
    
    async def analyze_risk_trends(
        self,
        user_id: str,
        assessments: List[RiskAssessment],
        time_period: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        分析风险趋势
        
        Args:
            user_id: 用户ID
            assessments: 历史评估记录
            time_period: 分析时间段
            
        Returns:
            趋势分析结果
        """
        try:
            if len(assessments) < 3:
                return {
                    'trend': 'insufficient_data',
                    'confidence': 0.0,
                    'details': {'message': '数据不足以分析趋势'}
                }
            
            # 按时间排序评估记录
            sorted_assessments = sorted(assessments, key=lambda x: x.timestamp)
            
            # 提取风险分数序列
            risk_scores = [assessment.risk_score for assessment in sorted_assessments]
            timestamps = [assessment.timestamp for assessment in sorted_assessments]
            
            # 计算趋势斜率
            trend_slope = self._calculate_trend_slope(timestamps, risk_scores)
            
            # 确定趋势方向
            if trend_slope > 0.01:
                trend_direction = 'deteriorating'
            elif trend_slope < -0.01:
                trend_direction = 'improving'
            else:
                trend_direction = 'stable'
            
            # 计算波动性
            volatility = np.std(risk_scores) if len(risk_scores) > 1 else 0.0
            
            # 分析风险等级变化
            level_changes = self._analyze_level_changes(sorted_assessments)
            
            # 计算趋势置信度
            confidence = min(1.0, len(assessments) / 10.0)
            
            return {
                'trend': trend_direction,
                'trend_slope': trend_slope,
                'confidence': confidence,
                'volatility': volatility,
                'level_changes': level_changes,
                'current_risk': risk_scores[-1] if risk_scores else 0.0,
                'risk_range': {
                    'min': min(risk_scores),
                    'max': max(risk_scores),
                    'avg': np.mean(risk_scores)
                },
                'assessment_count': len(assessments),
                'analysis_period': time_period.days
            }
            
        except Exception as e:
            logger.error(f"风险趋势分析失败: {str(e)}")
            return {'error': str(e)}
    
    # 私有方法实现
    async def _analyze_depression_indicators(
        self,
        emotion_history: List[EmotionState],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """分析抑郁症状指标"""
        if len(emotion_history) < 5:
            return 0.0
        
        # 分析负面情感持续性
        recent_emotions = emotion_history[-20:] if len(emotion_history) >= 20 else emotion_history
        
        # 计算低效价情感比例
        low_valence_count = sum(1 for emotion in recent_emotions if emotion.valence < -0.3)
        low_valence_ratio = low_valence_count / len(recent_emotions)
        
        # 计算低唤醒度(活动水平下降)
        low_arousal_count = sum(1 for emotion in recent_emotions if emotion.arousal < 0.3)
        low_arousal_ratio = low_arousal_count / len(recent_emotions)
        
        # 分析情感强度下降
        intensity_decline = self._calculate_intensity_decline(recent_emotions)
        
        # 检测抑郁关键词(如果有文本上下文)
        keyword_score = 0.0
        if context and 'user_inputs' in context:
            keyword_score = self._analyze_depression_keywords(context['user_inputs'])
        
        # 综合评分
        depression_score = (
            low_valence_ratio * 0.4 +
            low_arousal_ratio * 0.3 +
            intensity_decline * 0.2 +
            keyword_score * 0.1
        )
        
        return min(1.0, depression_score)
    
    async def _analyze_anxiety_indicators(
        self,
        emotion_history: List[EmotionState],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """分析焦虑症状指标"""
        if len(emotion_history) < 5:
            return 0.0
        
        recent_emotions = emotion_history[-20:] if len(emotion_history) >= 20 else emotion_history
        
        # 分析高唤醒度情感
        high_arousal_count = sum(1 for emotion in recent_emotions if emotion.arousal > 0.7)
        high_arousal_ratio = high_arousal_count / len(recent_emotions)
        
        # 分析焦虑相关情感
        anxiety_emotions = ['anxiety', 'fear', 'worry', 'nervousness']
        anxiety_count = sum(1 for emotion in recent_emotions if emotion.emotion in anxiety_emotions)
        anxiety_ratio = anxiety_count / len(recent_emotions)
        
        # 分析情感波动频率
        volatility = self._calculate_emotional_volatility(recent_emotions)
        
        # 检测焦虑关键词
        keyword_score = 0.0
        if context and 'user_inputs' in context:
            keyword_score = self._analyze_anxiety_keywords(context['user_inputs'])
        
        # 综合评分
        anxiety_score = (
            high_arousal_ratio * 0.3 +
            anxiety_ratio * 0.4 +
            volatility * 0.2 +
            keyword_score * 0.1
        )
        
        return min(1.0, anxiety_score)
    
    def _calculate_emotional_volatility(self, emotion_history: List[EmotionState]) -> float:
        """计算情感波动性"""
        if len(emotion_history) < 3:
            return 0.0
        
        # 计算效价波动
        valences = [emotion.valence for emotion in emotion_history]
        valence_volatility = np.std(valences) if len(valences) > 1 else 0.0
        
        # 计算唤醒度波动
        arousals = [emotion.arousal for emotion in emotion_history]
        arousal_volatility = np.std(arousals) if len(arousals) > 1 else 0.0
        
        # 计算强度波动
        intensities = [emotion.intensity for emotion in emotion_history]
        intensity_volatility = np.std(intensities) if len(intensities) > 1 else 0.0
        
        # 综合波动性
        total_volatility = (valence_volatility + arousal_volatility + intensity_volatility) / 3.0
        
        return min(1.0, total_volatility * 2.0)  # 放大波动性影响
    
    async def _analyze_social_isolation(
        self,
        emotion_history: List[EmotionState],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """分析社交孤立风险"""
        isolation_score = 0.0
        
        # 分析孤独感相关情感
        loneliness_emotions = ['loneliness', 'isolation', 'emptiness']
        loneliness_count = sum(1 for emotion in emotion_history 
                              if emotion.emotion in loneliness_emotions)
        
        if emotion_history:
            loneliness_ratio = loneliness_count / len(emotion_history)
            isolation_score += loneliness_ratio * 0.6
        
        # 分析上下文中的社交指标
        if context:
            social_interactions = context.get('social_interactions', 0)
            if social_interactions < 2:  # 低社交互动
                isolation_score += 0.4
        
        return min(1.0, isolation_score)
    
    async def _analyze_behavioral_changes(
        self,
        emotion_history: List[EmotionState],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """分析行为变化风险"""
        if not context:
            return 0.0
        
        behavioral_score = 0.0
        
        # 分析睡眠模式变化
        sleep_disruption = context.get('sleep_disruption', 0)
        behavioral_score += sleep_disruption * 0.3
        
        # 分析食欲变化
        appetite_changes = context.get('appetite_changes', 0)
        behavioral_score += appetite_changes * 0.2
        
        # 分析活动水平变化
        activity_level_drop = context.get('activity_level_drop', 0)
        behavioral_score += activity_level_drop * 0.3
        
        # 分析社交回避
        social_withdrawal = context.get('social_withdrawal', 0)
        behavioral_score += social_withdrawal * 0.2
        
        return min(1.0, behavioral_score)
    
    async def _extract_depression_evidence(self, emotion_history: List[EmotionState]) -> Dict[str, Any]:
        """提取抑郁症状证据"""
        if not emotion_history:
            return {}
        
        recent_emotions = emotion_history[-10:]
        
        negative_emotions = [e for e in recent_emotions if e.valence < -0.3]
        low_arousal_emotions = [e for e in recent_emotions if e.arousal < 0.3]
        
        return {
            'negative_emotion_count': len(negative_emotions),
            'low_arousal_count': len(low_arousal_emotions),
            'avg_valence': np.mean([e.valence for e in recent_emotions]),
            'avg_intensity': np.mean([e.intensity for e in recent_emotions]),
            'duration_days': (recent_emotions[-1].timestamp - recent_emotions[0].timestamp).days if len(recent_emotions) > 1 else 0
        }
    
    async def _extract_anxiety_evidence(self, emotion_history: List[EmotionState]) -> Dict[str, Any]:
        """提取焦虑症状证据"""
        if not emotion_history:
            return {}
        
        recent_emotions = emotion_history[-10:]
        
        high_arousal_emotions = [e for e in recent_emotions if e.arousal > 0.7]
        anxiety_emotions = [e for e in recent_emotions if e.emotion in ['anxiety', 'fear', 'worry']]
        
        return {
            'high_arousal_count': len(high_arousal_emotions),
            'anxiety_emotion_count': len(anxiety_emotions),
            'avg_arousal': np.mean([e.arousal for e in recent_emotions]),
            'volatility': np.std([e.valence for e in recent_emotions])
        }
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """确定风险等级"""
        if risk_score >= self.risk_thresholds['critical']:
            return RiskLevel.CRITICAL.value
        elif risk_score >= self.risk_thresholds['high']:
            return RiskLevel.HIGH.value
        elif risk_score >= self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value
    
    async def _generate_recommendations(self, risk_level: str, risk_factors: List[RiskFactor]) -> List[str]:
        """生成推荐行动"""
        recommendations = []
        
        # 基于风险等级的通用建议
        if risk_level == RiskLevel.CRITICAL.value:
            recommendations.extend([
                "立即寻求专业心理健康服务",
                "联系危机干预热线",
                "确保安全支持系统",
                "考虑住院治疗评估"
            ])
        elif risk_level == RiskLevel.HIGH.value:
            recommendations.extend([
                "预约心理健康专业人士",
                "增加社会支持网络",
                "制定应急安全计划",
                "考虑药物治疗评估"
            ])
        elif risk_level == RiskLevel.MEDIUM.value:
            recommendations.extend([
                "寻求心理咨询支持",
                "学习压力管理技巧",
                "保持规律作息",
                "增加身体活动"
            ])
        else:
            recommendations.extend([
                "保持良好心理健康习惯",
                "定期情感自我检查",
                "维护社交关系",
                "继续健康生活方式"
            ])
        
        # 基于具体风险因子的针对性建议
        for factor in risk_factors:
            if factor.factor_type == 'depression_indicators':
                recommendations.append("重点关注抑郁症状管理")
            elif factor.factor_type == 'anxiety_indicators':
                recommendations.append("学习焦虑管理技巧")
            elif factor.factor_type == 'emotional_volatility':
                recommendations.append("练习情绪调节策略")
            elif factor.factor_type == 'social_isolation':
                recommendations.append("增加社交活动参与")
        
        return list(set(recommendations))  # 去重
    
    def _calculate_prediction_confidence(self, risk_factors: List[RiskFactor], data_points: int) -> float:
        """计算预测置信度"""
        # 基于风险因子数量和数据完整性
        factor_confidence = min(1.0, len(risk_factors) / 3.0)
        data_confidence = min(1.0, data_points / 50.0)
        
        # 基于风险因子质量
        if risk_factors:
            avg_factor_score = sum(factor.score for factor in risk_factors) / len(risk_factors)
            quality_confidence = avg_factor_score
        else:
            quality_confidence = 0.1
        
        # 综合置信度
        total_confidence = (factor_confidence + data_confidence + quality_confidence) / 3.0
        
        return min(1.0, total_confidence)
    
    def _analyze_depression_keywords(self, user_inputs: List[str]) -> float:
        """分析抑郁关键词"""
        if not user_inputs:
            return 0.0
        
        combined_text = ' '.join(user_inputs).lower()
        matched_keywords = [kw for kw in self.depression_keywords if kw in combined_text]
        
        return min(1.0, len(matched_keywords) / len(self.depression_keywords))
    
    def _analyze_anxiety_keywords(self, user_inputs: List[str]) -> float:
        """分析焦虑关键词"""
        if not user_inputs:
            return 0.0
        
        combined_text = ' '.join(user_inputs).lower()
        matched_keywords = [kw for kw in self.anxiety_keywords if kw in combined_text]
        
        return min(1.0, len(matched_keywords) / len(self.anxiety_keywords))
    
    def _calculate_intensity_decline(self, emotion_history: List[EmotionState]) -> float:
        """计算情感强度下降趋势"""
        if len(emotion_history) < 5:
            return 0.0
        
        # 比较前半段和后半段的平均强度
        mid_point = len(emotion_history) // 2
        early_intensities = [e.intensity for e in emotion_history[:mid_point]]
        recent_intensities = [e.intensity for e in emotion_history[mid_point:]]
        
        if early_intensities and recent_intensities:
            early_avg = np.mean(early_intensities)
            recent_avg = np.mean(recent_intensities)
            decline = max(0.0, early_avg - recent_avg)
            return min(1.0, decline)
        
        return 0.0
    
    def _calculate_negative_trend(self, emotions: List[EmotionState]) -> float:
        """计算负面情感趋势"""
        if len(emotions) < 3:
            return 0.0
        
        # 计算效价的下降趋势
        valences = [e.valence for e in emotions]
        timestamps = [e.timestamp for e in emotions]
        
        # 简单线性趋势计算
        if len(valences) >= 3:
            slope = self._calculate_trend_slope(timestamps, valences)
            return max(0.0, -slope)  # 负斜率表示下降趋势
        
        return 0.0
    
    def _calculate_deterioration_rate(self, emotions: List[EmotionState]) -> float:
        """计算情感恶化速度"""
        if len(emotions) < 3:
            return 0.0
        
        # 计算连续负面情感的加速度
        valences = [e.valence for e in emotions]
        
        # 计算二阶导数(加速度)
        if len(valences) >= 3:
            accelerations = []
            for i in range(2, len(valences)):
                acc = valences[i] - 2 * valences[i-1] + valences[i-2]
                accelerations.append(acc)
            
            if accelerations:
                avg_acceleration = np.mean(accelerations)
                return max(0.0, -avg_acceleration)  # 负加速度表示恶化
        
        return 0.0
    
    def _detect_crisis_patterns(self, emotions: List[EmotionState]) -> float:
        """检测危机模式"""
        if len(emotions) < 5:
            return 0.0
        
        crisis_patterns = 0.0
        
        # 检测极端负面情感
        extreme_negative = sum(1 for e in emotions if e.valence < -0.8 and e.intensity > 0.8)
        crisis_patterns += extreme_negative / len(emotions)
        
        # 检测情感空白(麻木)
        emotional_numbness = sum(1 for e in emotions if e.intensity < 0.2)
        crisis_patterns += (emotional_numbness / len(emotions)) * 0.5
        
        # 检测危险情感组合
        dangerous_emotions = ['despair', 'hopelessness', 'worthlessness']
        dangerous_count = sum(1 for e in emotions if e.emotion in dangerous_emotions)
        crisis_patterns += (dangerous_count / len(emotions)) * 0.8
        
        return min(1.0, crisis_patterns)
    
    def _calculate_trend_slope(self, timestamps: List[datetime], values: List[float]) -> float:
        """计算趋势斜率"""
        if len(timestamps) != len(values) or len(values) < 2:
            return 0.0
        
        # 转换时间戳为数值
        time_nums = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # 计算简单线性回归斜率
        n = len(values)
        sum_x = sum(time_nums)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(time_nums, values))
        sum_x2 = sum(x * x for x in time_nums)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _analyze_level_changes(self, assessments: List[RiskAssessment]) -> Dict[str, int]:
        """分析风险等级变化"""
        level_changes = {
            'improvements': 0,
            'deteriorations': 0,
            'stable_periods': 0
        }
        
        if len(assessments) < 2:
            return level_changes
        
        level_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        for i in range(1, len(assessments)):
            prev_level = level_map.get(assessments[i-1].risk_level, 1)
            curr_level = level_map.get(assessments[i].risk_level, 1)
            
            if curr_level > prev_level:
                level_changes['deteriorations'] += 1
            elif curr_level < prev_level:
                level_changes['improvements'] += 1
            else:
                level_changes['stable_periods'] += 1
        
        return level_changes
