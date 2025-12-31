"""
情感健康监测和仪表盘系统
"""

from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from .models import (
    HealthDashboardData, RiskAssessment, RiskLevel, InterventionPlan
)
from ..emotion_modeling.models import EmotionState, PersonalityProfile

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class HealthGoal:
    """健康目标"""
    goal_id: str
    user_id: str
    goal_type: str  # emotional_stability, risk_reduction, wellbeing_improvement
    title: str
    description: str
    target_value: float
    current_value: float
    target_date: datetime
    created_at: datetime
    status: str  # active, paused, completed, cancelled
    progress_percentage: float = 0.0
    milestones: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = []

class HealthMonitoringSystem:
    """情感健康监测系统"""
    
    def __init__(self):
        self.health_metrics = {
            'emotional_stability': {
                'weight': 0.3,
                'calculation': 'valence_variance_inverse'
            },
            'resilience_score': {
                'weight': 0.25,
                'calculation': 'recovery_rate_analysis'
            },
            'social_connection': {
                'weight': 0.2,
                'calculation': 'social_interaction_quality'
            },
            'coping_effectiveness': {
                'weight': 0.15,
                'calculation': 'stress_response_analysis'
            },
            'life_satisfaction': {
                'weight': 0.1,
                'calculation': 'positive_emotion_ratio'
            }
        }
        
        self.trend_indicators = [
            'improving',     # 明显改善趋势
            'stable',        # 稳定状态
            'fluctuating',   # 波动状态
            'deteriorating'  # 恶化趋势
        ]
    
    async def generate_health_dashboard(
        self,
        user_id: str,
        emotion_history: List[EmotionState],
        risk_assessments: List[RiskAssessment],
        interventions: List[InterventionPlan],
        personality_profile: Optional[PersonalityProfile] = None,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> HealthDashboardData:
        """
        生成健康仪表盘数据
        
        Args:
            user_id: 用户ID
            emotion_history: 情感历史
            risk_assessments: 风险评估历史
            interventions: 干预计划
            personality_profile: 个性画像
            time_period: 分析时间段
            
        Returns:
            健康仪表盘数据
        """
        try:
            if not time_period:
                end_time = utc_now()
                start_time = end_time - timedelta(days=30)
                time_period = (start_time, end_time)
            
            # 筛选时间段内的数据
            filtered_emotions = self._filter_by_time_period(emotion_history, time_period)
            filtered_assessments = self._filter_assessments_by_time(risk_assessments, time_period)
            
            # 计算整体健康指标
            overall_health_score = await self._calculate_overall_health_score(
                filtered_emotions, filtered_assessments, personality_profile
            )
            
            emotional_stability = await self._calculate_emotional_stability(filtered_emotions)
            resilience_score = await self._calculate_resilience_score(filtered_emotions, filtered_assessments)
            
            # 分析风险趋势
            current_risk_level, risk_trend, risk_history = await self._analyze_risk_trends(
                filtered_assessments
            )
            
            # 分析情感趋势
            emotion_trends, dominant_emotions, emotion_volatility = await self._analyze_emotion_trends(
                filtered_emotions
            )
            
            # 分析干预效果
            intervention_stats = await self._analyze_intervention_effectiveness(interventions)
            
            # 获取健康目标进度
            health_goals, goal_progress = await self._get_health_goals_progress(user_id)
            
            # 生成洞察和建议
            insights = await self._generate_health_insights(
                filtered_emotions, filtered_assessments, interventions
            )
            recommendations = await self._generate_health_recommendations(
                overall_health_score, current_risk_level, emotion_volatility
            )
            
            dashboard_data = HealthDashboardData(
                user_id=user_id,
                time_period=time_period,
                overall_health_score=overall_health_score,
                emotional_stability=emotional_stability,
                resilience_score=resilience_score,
                current_risk_level=current_risk_level,
                risk_trend=risk_trend,
                risk_history=risk_history,
                emotion_trends=emotion_trends,
                dominant_emotions=dominant_emotions,
                emotion_volatility=emotion_volatility,
                active_interventions=intervention_stats['active'],
                completed_interventions=intervention_stats['completed'],
                intervention_success_rate=intervention_stats['success_rate'],
                health_goals=health_goals,
                goal_progress=goal_progress,
                insights=insights,
                recommendations=recommendations
            )
            
            logger.info(f"生成健康仪表盘 - 用户: {user_id}, 健康分数: {overall_health_score:.3f}")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"健康仪表盘生成失败: {str(e)}")
            raise
    
    async def track_emotional_patterns(
        self,
        emotion_history: List[EmotionState],
        analysis_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        跟踪情感模式
        
        Args:
            emotion_history: 情感历史
            analysis_window: 分析窗口
            
        Returns:
            情感模式分析结果
        """
        try:
            if not emotion_history:
                return {'patterns': [], 'trends': {}}
            
            patterns = []
            
            # 分析日内模式
            daily_patterns = await self._analyze_daily_patterns(emotion_history)
            if daily_patterns:
                patterns.append({
                    'type': 'daily_cycle',
                    'description': '日内情感变化模式',
                    'pattern': daily_patterns
                })
            
            # 分析周内模式
            weekly_patterns = await self._analyze_weekly_patterns(emotion_history)
            if weekly_patterns:
                patterns.append({
                    'type': 'weekly_cycle',
                    'description': '周内情感变化模式',
                    'pattern': weekly_patterns
                })
            
            # 分析触发因子模式
            trigger_patterns = await self._analyze_trigger_patterns(emotion_history)
            if trigger_patterns:
                patterns.append({
                    'type': 'trigger_patterns',
                    'description': '情感触发因子模式',
                    'pattern': trigger_patterns
                })
            
            # 分析恢复模式
            recovery_patterns = await self._analyze_recovery_patterns(emotion_history)
            if recovery_patterns:
                patterns.append({
                    'type': 'recovery_patterns',
                    'description': '情感恢复模式',
                    'pattern': recovery_patterns
                })
            
            # 计算趋势指标
            trends = {
                'overall_trend': self._calculate_overall_trend(emotion_history),
                'volatility_trend': self._calculate_volatility_trend(emotion_history),
                'valence_trend': self._calculate_valence_trend(emotion_history),
                'intensity_trend': self._calculate_intensity_trend(emotion_history)
            }
            
            return {
                'patterns': patterns,
                'trends': trends,
                'analysis_period': {
                    'start': min(e.timestamp for e in emotion_history).isoformat(),
                    'end': max(e.timestamp for e in emotion_history).isoformat(),
                    'data_points': len(emotion_history)
                }
            }
            
        except Exception as e:
            logger.error(f"情感模式跟踪失败: {str(e)}")
            return {'error': str(e)}
    
    async def assess_intervention_impact(
        self,
        intervention: InterventionPlan,
        before_emotions: List[EmotionState],
        after_emotions: List[EmotionState],
        before_assessment: RiskAssessment,
        after_assessment: Optional[RiskAssessment] = None
    ) -> Dict[str, Any]:
        """
        评估干预影响
        
        Args:
            intervention: 干预计划
            before_emotions: 干预前情感数据
            after_emotions: 干预后情感数据
            before_assessment: 干预前风险评估
            after_assessment: 干预后风险评估
            
        Returns:
            干预影响分析结果
        """
        try:
            impact_analysis = {}
            
            # 情感状态改善分析
            emotional_impact = await self._analyze_emotional_impact(before_emotions, after_emotions)
            impact_analysis['emotional_impact'] = emotional_impact
            
            # 风险水平变化分析
            risk_impact = await self._analyze_risk_impact(before_assessment, after_assessment)
            impact_analysis['risk_impact'] = risk_impact
            
            # 行为变化分析（如果有相关数据）
            behavioral_impact = await self._analyze_behavioral_impact(intervention, before_emotions, after_emotions)
            impact_analysis['behavioral_impact'] = behavioral_impact
            
            # 计算总体效果分数
            overall_effectiveness = self._calculate_overall_effectiveness(
                emotional_impact, risk_impact, behavioral_impact
            )
            impact_analysis['overall_effectiveness'] = overall_effectiveness
            
            # 生成效果总结
            effectiveness_summary = self._generate_effectiveness_summary(
                overall_effectiveness, emotional_impact, risk_impact
            )
            impact_analysis['summary'] = effectiveness_summary
            
            # 改进建议
            improvement_suggestions = await self._generate_improvement_suggestions(
                intervention, impact_analysis
            )
            impact_analysis['improvement_suggestions'] = improvement_suggestions
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"干预影响评估失败: {str(e)}")
            return {'error': str(e)}
    
    # 私有方法实现
    async def _calculate_overall_health_score(
        self,
        emotions: List[EmotionState],
        assessments: List[RiskAssessment],
        personality_profile: Optional[PersonalityProfile]
    ) -> float:
        """计算整体健康分数"""
        if not emotions:
            return 0.5
        
        scores = {}
        
        # 情感稳定性
        scores['emotional_stability'] = await self._calculate_emotional_stability(emotions)
        
        # 韧性分数
        scores['resilience_score'] = await self._calculate_resilience_score(emotions, assessments)
        
        # 社交连接
        scores['social_connection'] = self._estimate_social_connection(emotions)
        
        # 应对效果
        scores['coping_effectiveness'] = self._estimate_coping_effectiveness(emotions)
        
        # 生活满意度
        scores['life_satisfaction'] = self._estimate_life_satisfaction(emotions)
        
        # 加权平均
        weighted_score = sum(
            scores[metric] * self.health_metrics[metric]['weight']
            for metric in scores
        )
        
        return min(1.0, weighted_score)
    
    async def _calculate_emotional_stability(self, emotions: List[EmotionState]) -> float:
        """计算情感稳定性"""
        if len(emotions) < 3:
            return 0.5
        
        # 计算效价方差的倒数
        valences = [e.valence for e in emotions]
        valence_variance = np.var(valences)
        
        # 方差越小，稳定性越高
        stability = 1.0 / (1.0 + valence_variance * 2.0)
        return min(1.0, stability)
    
    async def _calculate_resilience_score(
        self,
        emotions: List[EmotionState],
        assessments: List[RiskAssessment]
    ) -> float:
        """计算韧性分数"""
        if not emotions:
            return 0.5
        
        # 基于从负面情感恢复的速度
        recovery_episodes = self._identify_recovery_episodes(emotions)
        
        if not recovery_episodes:
            return 0.5
        
        # 计算平均恢复速度
        recovery_speeds = []
        for episode in recovery_episodes:
            recovery_time = episode['recovery_time'].total_seconds() / 3600  # 小时
            recovery_magnitude = episode['recovery_magnitude']
            
            if recovery_time > 0:
                speed = recovery_magnitude / recovery_time
                recovery_speeds.append(speed)
        
        if recovery_speeds:
            avg_recovery_speed = np.mean(recovery_speeds)
            resilience = min(1.0, avg_recovery_speed / 2.0)  # 归一化
        else:
            resilience = 0.5
        
        return resilience
    
    def _filter_by_time_period(
        self,
        emotions: List[EmotionState],
        time_period: Tuple[datetime, datetime]
    ) -> List[EmotionState]:
        """按时间段筛选情感数据"""
        start_time, end_time = time_period
        return [
            emotion for emotion in emotions
            if start_time <= emotion.timestamp <= end_time
        ]
    
    def _filter_assessments_by_time(
        self,
        assessments: List[RiskAssessment],
        time_period: Tuple[datetime, datetime]
    ) -> List[RiskAssessment]:
        """按时间段筛选风险评估"""
        start_time, end_time = time_period
        return [
            assessment for assessment in assessments
            if start_time <= assessment.timestamp <= end_time
        ]
    
    async def _analyze_risk_trends(
        self,
        assessments: List[RiskAssessment]
    ) -> Tuple[str, str, List[Tuple[datetime, float]]]:
        """分析风险趋势"""
        if not assessments:
            return RiskLevel.LOW.value, 'stable', []
        
        # 当前风险等级
        current_risk_level = assessments[-1].risk_level
        
        # 风险历史
        risk_history = [(assessment.timestamp, assessment.risk_score) for assessment in assessments]
        
        # 计算趋势
        if len(assessments) >= 3:
            recent_scores = [a.risk_score for a in assessments[-3:]]
            if recent_scores[-1] > recent_scores[0] + 0.1:
                trend = 'deteriorating'
            elif recent_scores[-1] < recent_scores[0] - 0.1:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return current_risk_level, trend, risk_history
    
    async def _analyze_emotion_trends(
        self,
        emotions: List[EmotionState]
    ) -> Tuple[Dict[str, List[Tuple[datetime, float]]], List[Tuple[str, float]], float]:
        """分析情感趋势"""
        emotion_trends = {}
        dominant_emotions = []
        emotion_volatility = 0.0
        
        if not emotions:
            return emotion_trends, dominant_emotions, emotion_volatility
        
        # 按情感类型分组
        emotion_groups = {}
        for emotion in emotions:
            emotion_type = emotion.emotion
            if emotion_type not in emotion_groups:
                emotion_groups[emotion_type] = []
            emotion_groups[emotion_type].append(emotion)
        
        # 计算每种情感的趋势
        for emotion_type, emotion_list in emotion_groups.items():
            trend_data = [(e.timestamp, e.intensity) for e in emotion_list]
            emotion_trends[emotion_type] = trend_data
        
        # 计算主导情感
        emotion_counts = {emotion_type: len(emotion_list) for emotion_type, emotion_list in emotion_groups.items()}
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [(emotion, count/len(emotions)) for emotion, count in sorted_emotions[:5]]
        
        # 计算情感波动性
        valences = [e.valence for e in emotions]
        emotion_volatility = np.std(valences) if len(valences) > 1 else 0.0
        
        return emotion_trends, dominant_emotions, emotion_volatility
    
    async def _analyze_intervention_effectiveness(self, interventions: List[InterventionPlan]) -> Dict[str, Any]:
        """分析干预效果"""
        if not interventions:
            return {'active': 0, 'completed': 0, 'success_rate': 0.0}
        
        active_count = sum(1 for i in interventions if i.status == 'active')
        completed_count = sum(1 for i in interventions if i.status == 'completed')
        
        # 简化的成功率计算
        successful_interventions = sum(1 for i in interventions 
                                     if i.status == 'completed' and i.progress >= 0.8)
        
        success_rate = successful_interventions / completed_count if completed_count > 0 else 0.0
        
        return {
            'active': active_count,
            'completed': completed_count,
            'success_rate': success_rate
        }
    
    async def _get_health_goals_progress(self, user_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """获取健康目标进度"""
        return [], {}
    
    async def _generate_health_insights(
        self,
        emotions: List[EmotionState],
        assessments: List[RiskAssessment],
        interventions: List[InterventionPlan]
    ) -> List[str]:
        """生成健康洞察"""
        insights = []
        
        if emotions:
            # 情感模式洞察
            negative_emotions = [e for e in emotions if e.valence < -0.2]
            if len(negative_emotions) > len(emotions) * 0.6:
                insights.append("您最近的负面情感较多，建议关注情绪调节")
            
            # 时间模式洞察
            morning_emotions = [e for e in emotions if 6 <= e.timestamp.hour <= 12]
            evening_emotions = [e for e in emotions if 18 <= e.timestamp.hour <= 23]
            
            if morning_emotions and evening_emotions:
                morning_avg = np.mean([e.valence for e in morning_emotions])
                evening_avg = np.mean([e.valence for e in evening_emotions])
                
                if morning_avg > evening_avg + 0.3:
                    insights.append("您在上午的情绪状态通常更好")
                elif evening_avg > morning_avg + 0.3:
                    insights.append("您在傍晚的情绪状态通常更好")
        
        if assessments:
            recent_assessment = assessments[-1]
            if recent_assessment.risk_level in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]:
                insights.append("当前风险水平需要关注，建议寻求支持")
        
        if interventions:
            active_interventions = [i for i in interventions if i.status == 'active']
            if active_interventions:
                insights.append(f"您目前有{len(active_interventions)}项活跃的干预计划")
        
        return insights
    
    async def _generate_health_recommendations(
        self,
        health_score: float,
        risk_level: str,
        volatility: float
    ) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        if health_score < 0.5:
            recommendations.append("考虑寻求专业心理健康支持")
        
        if risk_level in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]:
            recommendations.append("建议增加情感支持活动")
        
        if volatility > 0.5:
            recommendations.append("练习情绪调节技巧，如深呼吸或正念冥想")
        
        # 通用建议
        recommendations.extend([
            "保持规律的作息时间",
            "进行适度的体育锻炼",
            "维护良好的社交关系",
            "定期进行自我情感检查"
        ])
        
        return recommendations
    
    def _estimate_social_connection(self, emotions: List[EmotionState]) -> float:
        """估算社交连接水平"""
        # 基于情感中的社交相关触发因子
        social_emotions = [e for e in emotions if any(
            trigger in ['social_interaction', 'friendship', 'family_time']
            for trigger in e.triggers
        )]
        
        if not emotions:
            return 0.5
        
        social_ratio = len(social_emotions) / len(emotions)
        return min(1.0, social_ratio * 2.0)
    
    def _estimate_coping_effectiveness(self, emotions: List[EmotionState]) -> float:
        """估算应对效果"""
        # 基于恢复速度和频率
        recovery_episodes = self._identify_recovery_episodes(emotions)
        
        if not recovery_episodes:
            return 0.5
        
        avg_recovery_magnitude = np.mean([ep['recovery_magnitude'] for ep in recovery_episodes])
        return min(1.0, avg_recovery_magnitude)
    
    def _estimate_life_satisfaction(self, emotions: List[EmotionState]) -> float:
        """估算生活满意度"""
        if not emotions:
            return 0.5
        
        # 基于正面情感比例
        positive_emotions = [e for e in emotions if e.valence > 0.2]
        positive_ratio = len(positive_emotions) / len(emotions)
        
        return positive_ratio
    
    def _identify_recovery_episodes(self, emotions: List[EmotionState]) -> List[Dict[str, Any]]:
        """识别恢复事件"""
        episodes = []
        
        if len(emotions) < 5:
            return episodes
        
        for i in range(len(emotions) - 2):
            current = emotions[i]
            
            # 寻找负面情感的开始
            if current.valence < -0.3:
                # 寻找后续的恢复
                for j in range(i + 1, min(i + 10, len(emotions))):
                    recovery = emotions[j]
                    
                    if recovery.valence > current.valence + 0.4:
                        episodes.append({
                            'start_time': current.timestamp,
                            'recovery_time': recovery.timestamp - current.timestamp,
                            'recovery_magnitude': recovery.valence - current.valence,
                            'start_valence': current.valence,
                            'recovery_valence': recovery.valence
                        })
                        break
        
        return episodes
    
    async def _analyze_daily_patterns(self, emotions: List[EmotionState]) -> Dict[str, Any]:
        """分析日内模式"""
        if len(emotions) < 10:
            return {}
        
        # 按小时分组
        hourly_emotions = {}
        for emotion in emotions:
            hour = emotion.timestamp.hour
            if hour not in hourly_emotions:
                hourly_emotions[hour] = []
            hourly_emotions[hour].append(emotion)
        
        # 计算每小时平均情感状态
        hourly_averages = {}
        for hour, hour_emotions in hourly_emotions.items():
            if len(hour_emotions) >= 2:  # 至少2个数据点
                avg_valence = np.mean([e.valence for e in hour_emotions])
                avg_intensity = np.mean([e.intensity for e in hour_emotions])
                hourly_averages[hour] = {
                    'valence': avg_valence,
                    'intensity': avg_intensity,
                    'count': len(hour_emotions)
                }
        
        return hourly_averages
    
    async def _analyze_weekly_patterns(self, emotions: List[EmotionState]) -> Dict[str, Any]:
        """分析周内模式"""
        if len(emotions) < 14:  # 至少两周数据
            return {}
        
        # 按星期几分组
        weekday_emotions = {}
        for emotion in emotions:
            weekday = emotion.timestamp.weekday()  # 0=Monday, 6=Sunday
            if weekday not in weekday_emotions:
                weekday_emotions[weekday] = []
            weekday_emotions[weekday].append(emotion)
        
        # 计算每天平均情感状态
        weekday_averages = {}
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for weekday, day_emotions in weekday_emotions.items():
            if len(day_emotions) >= 3:  # 至少3个数据点
                avg_valence = np.mean([e.valence for e in day_emotions])
                avg_intensity = np.mean([e.intensity for e in day_emotions])
                weekday_averages[weekday_names[weekday]] = {
                    'valence': avg_valence,
                    'intensity': avg_intensity,
                    'count': len(day_emotions)
                }
        
        return weekday_averages
    
    async def _analyze_trigger_patterns(self, emotions: List[EmotionState]) -> Dict[str, Any]:
        """分析触发因子模式"""
        trigger_stats = {}
        
        for emotion in emotions:
            for trigger in emotion.triggers:
                if trigger not in trigger_stats:
                    trigger_stats[trigger] = {
                        'count': 0,
                        'valences': [],
                        'intensities': []
                    }
                
                trigger_stats[trigger]['count'] += 1
                trigger_stats[trigger]['valences'].append(emotion.valence)
                trigger_stats[trigger]['intensities'].append(emotion.intensity)
        
        # 计算每个触发因子的统计信息
        trigger_analysis = {}
        for trigger, stats in trigger_stats.items():
            if stats['count'] >= 3:  # 至少3次出现
                trigger_analysis[trigger] = {
                    'frequency': stats['count'],
                    'avg_valence': np.mean(stats['valences']),
                    'avg_intensity': np.mean(stats['intensities']),
                    'valence_std': np.std(stats['valences']),
                    'emotional_impact': 'positive' if np.mean(stats['valences']) > 0.1 else 'negative'
                }
        
        return trigger_analysis
    
    async def _analyze_recovery_patterns(self, emotions: List[EmotionState]) -> Dict[str, Any]:
        """分析恢复模式"""
        recovery_episodes = self._identify_recovery_episodes(emotions)
        
        if not recovery_episodes:
            return {}
        
        # 统计恢复模式
        recovery_times = [ep['recovery_time'].total_seconds() / 3600 for ep in recovery_episodes]  # 小时
        recovery_magnitudes = [ep['recovery_magnitude'] for ep in recovery_episodes]
        
        return {
            'avg_recovery_time_hours': np.mean(recovery_times),
            'avg_recovery_magnitude': np.mean(recovery_magnitudes),
            'recovery_episodes_count': len(recovery_episodes),
            'fastest_recovery_hours': min(recovery_times) if recovery_times else 0,
            'strongest_recovery_magnitude': max(recovery_magnitudes) if recovery_magnitudes else 0
        }
    
    def _calculate_overall_trend(self, emotions: List[EmotionState]) -> str:
        """计算总体趋势"""
        if len(emotions) < 5:
            return 'insufficient_data'
        
        recent_emotions = emotions[-10:] if len(emotions) >= 10 else emotions
        valences = [e.valence for e in recent_emotions]
        
        # 简单线性趋势
        trend_slope = np.polyfit(range(len(valences)), valences, 1)[0]
        
        if trend_slope > 0.05:
            return 'improving'
        elif trend_slope < -0.05:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _calculate_volatility_trend(self, emotions: List[EmotionState]) -> str:
        """计算波动性趋势"""
        if len(emotions) < 10:
            return 'insufficient_data'
        
        # 分析前半段和后半段的波动性
        mid_point = len(emotions) // 2
        early_emotions = emotions[:mid_point]
        recent_emotions = emotions[mid_point:]
        
        early_volatility = np.std([e.valence for e in early_emotions])
        recent_volatility = np.std([e.valence for e in recent_emotions])
        
        volatility_change = recent_volatility - early_volatility
        
        if volatility_change > 0.1:
            return 'increasing'
        elif volatility_change < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_valence_trend(self, emotions: List[EmotionState]) -> str:
        """计算效价趋势"""
        return self._calculate_overall_trend(emotions)
    
    def _calculate_intensity_trend(self, emotions: List[EmotionState]) -> str:
        """计算强度趋势"""
        if len(emotions) < 5:
            return 'insufficient_data'
        
        recent_emotions = emotions[-10:] if len(emotions) >= 10 else emotions
        intensities = [e.intensity for e in recent_emotions]
        
        trend_slope = np.polyfit(range(len(intensities)), intensities, 1)[0]
        
        if trend_slope > 0.05:
            return 'increasing'
        elif trend_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _analyze_emotional_impact(
        self,
        before_emotions: List[EmotionState],
        after_emotions: List[EmotionState]
    ) -> Dict[str, Any]:
        """分析情感影响"""
        if not before_emotions or not after_emotions:
            return {'improvement': 0.0, 'details': 'insufficient_data'}
        
        before_avg_valence = np.mean([e.valence for e in before_emotions])
        after_avg_valence = np.mean([e.valence for e in after_emotions])
        
        valence_improvement = after_avg_valence - before_avg_valence
        
        before_volatility = np.std([e.valence for e in before_emotions])
        after_volatility = np.std([e.valence for e in after_emotions])
        
        stability_improvement = before_volatility - after_volatility
        
        return {
            'valence_improvement': valence_improvement,
            'stability_improvement': stability_improvement,
            'overall_improvement': (valence_improvement + stability_improvement) / 2.0
        }
    
    async def _analyze_risk_impact(
        self,
        before_assessment: RiskAssessment,
        after_assessment: Optional[RiskAssessment]
    ) -> Dict[str, Any]:
        """分析风险影响"""
        if not after_assessment:
            return {'risk_reduction': 0.0, 'details': 'no_follow_up_assessment'}
        
        risk_reduction = before_assessment.risk_score - after_assessment.risk_score
        
        level_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        before_level = level_map.get(before_assessment.risk_level, 1)
        after_level = level_map.get(after_assessment.risk_level, 1)
        
        level_improvement = before_level - after_level
        
        return {
            'risk_score_reduction': risk_reduction,
            'risk_level_improvement': level_improvement,
            'overall_risk_improvement': (risk_reduction + level_improvement / 4.0) / 2.0
        }
    
    async def _analyze_behavioral_impact(
        self,
        intervention: InterventionPlan,
        before_emotions: List[EmotionState],
        after_emotions: List[EmotionState]
    ) -> Dict[str, Any]:
        """分析行为影响"""
        # 简化的行为影响分析
        return {
            'behavioral_changes': 'moderate_improvement',
            'engagement_level': 0.7,
            'compliance_rate': intervention.progress
        }
    
    def _calculate_overall_effectiveness(
        self,
        emotional_impact: Dict[str, Any],
        risk_impact: Dict[str, Any],
        behavioral_impact: Dict[str, Any]
    ) -> float:
        """计算总体效果"""
        emotional_score = emotional_impact.get('overall_improvement', 0.0)
        risk_score = risk_impact.get('overall_risk_improvement', 0.0)
        behavioral_score = behavioral_impact.get('engagement_level', 0.0)
        
        # 加权平均
        overall_score = (emotional_score * 0.4 + risk_score * 0.4 + behavioral_score * 0.2)
        return max(0.0, min(1.0, overall_score))
    
    def _generate_effectiveness_summary(
        self,
        overall_effectiveness: float,
        emotional_impact: Dict[str, Any],
        risk_impact: Dict[str, Any]
    ) -> str:
        """生成效果总结"""
        if overall_effectiveness >= 0.7:
            return "干预效果显著，情感状态和风险水平都有明显改善"
        elif overall_effectiveness >= 0.5:
            return "干预效果良好，有积极的改善趋势"
        elif overall_effectiveness >= 0.3:
            return "干预效果一般，需要调整策略"
        else:
            return "干预效果不佳，建议重新评估和制定计划"
    
    async def _generate_improvement_suggestions(
        self,
        intervention: InterventionPlan,
        impact_analysis: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        overall_effectiveness = impact_analysis.get('overall_effectiveness', 0.0)
        
        if overall_effectiveness < 0.5:
            suggestions.append("考虑更换干预策略")
            suggestions.append("增加干预频率和强度")
        
        if impact_analysis['emotional_impact'].get('overall_improvement', 0.0) < 0.3:
            suggestions.append("加强情感调节技能训练")
        
        if impact_analysis['risk_impact'].get('overall_risk_improvement', 0.0) < 0.3:
            suggestions.append("重点关注风险因子管理")
        
        suggestions.append("定期评估和调整干预计划")
        
        return suggestions
