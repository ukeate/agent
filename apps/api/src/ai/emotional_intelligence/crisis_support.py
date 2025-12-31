"""
紧急情感支持和危机干预系统
"""

from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Optional, Any, Tuple
from datetime import timedelta
import re
from .models import CrisisAssessment, SeverityLevel, DecisionContext
from ..emotion_modeling.models import EmotionState

from src.core.logging import get_logger
logger = get_logger(__name__)

class CrisisDetectionSystem:
    """危机检测系统"""
    
    def __init__(self):
        # 危机关键词和短语
        self.crisis_keywords = {
            'suicide': [
                '想死', '自杀', '结束生命', '活着没意义', '不想活了',
                '想要死去', '生无可恋', '死了算了', '去死'
            ],
            'self_harm': [
                '伤害自己', '自残', '割腕', '自虐', '伤害身体',
                '切割', '烫伤', '撞墙', '自己惩罚'
            ],
            'hopelessness': [
                '绝望', '没有希望', '看不到未来', '永远不会好',
                '毫无意义', '完全没用', '彻底失败'
            ],
            'isolation': [
                '没人理解', '完全孤独', '被遗弃', '没人关心',
                '独自一人', '与世隔绝', '被抛弃'
            ],
            'worthlessness': [
                '一无是处', '毫无价值', '不值得', '废物',
                '没用的人', '失败者', '负担'
            ]
        }
        
        # 危机行为模式
        self.crisis_behavioral_patterns = [
            'sudden_mood_drop',      # 情绪急剧下降
            'emotional_numbness',    # 情感麻木
            'extreme_agitation',     # 极度激动
            'social_withdrawal',     # 社交退缩
            'sleep_disruption',      # 睡眠严重紊乱
            'appetite_loss',         # 食欲完全丧失
            'decision_making_impaired'  # 决策能力受损
        ]
        
        # 保护性因子
        self.protective_factors = [
            'strong_social_support',
            'professional_help_seeking',
            'religious_beliefs',
            'future_plans',
            'responsibility_to_others',
            'fear_of_suicide'
        ]
    
    async def detect_crisis_indicators(
        self,
        user_id: str,
        user_input: str,
        emotion_state: EmotionState,
        context: Optional[Dict[str, Any]] = None,
        emotion_history: Optional[List[EmotionState]] = None
    ) -> CrisisAssessment:
        """
        检测危机指标
        
        Args:
            user_id: 用户ID
            user_input: 用户输入文本
            emotion_state: 当前情感状态
            context: 上下文信息
            emotion_history: 情感历史
            
        Returns:
            危机评估结果
        """
        try:
            indicators = []
            
            # 1. 语言指标检测
            language_indicators = await self._analyze_language_indicators(user_input)
            indicators.extend(language_indicators)
            
            # 2. 情感状态指标检测
            emotional_indicators = await self._analyze_emotional_indicators(emotion_state)
            indicators.extend(emotional_indicators)
            
            # 3. 行为模式指标检测
            if context:
                behavioral_indicators = await self._analyze_behavioral_indicators(context)
                indicators.extend(behavioral_indicators)
            
            # 4. 历史模式分析
            if emotion_history:
                pattern_indicators = await self._analyze_pattern_indicators(emotion_history)
                indicators.extend(pattern_indicators)
            
            # 5. 保护性因子评估
            protective_score = await self._assess_protective_factors(context or {})
            
            # 计算综合危机分数
            crisis_score = self._calculate_crisis_score(indicators, protective_score)
            
            # 确定严重程度
            severity_level = self._determine_severity_level(crisis_score)
            
            # 生成即时行动建议
            immediate_actions = await self._generate_immediate_actions(
                severity_level, indicators
            )
            
            # 确定专业服务需求
            professional_required = crisis_score > 0.7 or severity_level in [
                SeverityLevel.SEVERE.value, SeverityLevel.CRITICAL.value
            ]
            
            # 确定监护级别
            monitoring_level = self._determine_monitoring_level(crisis_score)
            
            assessment = CrisisAssessment(
                user_id=user_id,
                crisis_detected=crisis_score > 0.5,
                severity_level=severity_level,
                crisis_type=self._determine_crisis_type(indicators),
                indicators=indicators,
                risk_score=crisis_score,
                confidence=self._calculate_confidence(indicators),
                immediate_actions=immediate_actions,
                professional_required=professional_required,
                monitoring_level=monitoring_level,
                check_frequency=self._determine_check_frequency(crisis_score)
            )
            
            if assessment.crisis_detected:
                logger.warning(f"检测到危机状况 - 用户: {user_id}, 严重程度: {severity_level}, 分数: {crisis_score:.3f}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"危机检测失败: {str(e)}")
            raise
    
    async def assess_suicide_risk(
        self,
        user_input: str,
        emotion_state: EmotionState,
        context: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        评估自杀风险
        
        Args:
            user_input: 用户输入
            emotion_state: 情感状态
            context: 上下文信息
            
        Returns:
            (自杀风险分数, 风险因素列表)
        """
        try:
            risk_factors = []
            
            # 直接自杀意念表达
            suicide_keywords_score = self._analyze_keyword_category(user_input, 'suicide')
            if suicide_keywords_score > 0.3:
                risk_factors.append({
                    'factor': 'direct_suicidal_ideation',
                    'score': suicide_keywords_score,
                    'severity': 'high',
                    'evidence': '直接表达自杀意念'
                })
            
            # 自伤行为倾向
            self_harm_score = self._analyze_keyword_category(user_input, 'self_harm')
            if self_harm_score > 0.3:
                risk_factors.append({
                    'factor': 'self_harm_ideation',
                    'score': self_harm_score,
                    'severity': 'medium',
                    'evidence': '表达自伤行为倾向'
                })
            
            # 绝望感
            hopelessness_score = self._analyze_keyword_category(user_input, 'hopelessness')
            if hopelessness_score > 0.4:
                risk_factors.append({
                    'factor': 'hopelessness',
                    'score': hopelessness_score,
                    'severity': 'high',
                    'evidence': '表达强烈绝望感'
                })
            
            # 情感状态风险
            if emotion_state.emotion in ['despair', 'hopelessness'] and emotion_state.intensity > 0.8:
                risk_factors.append({
                    'factor': 'extreme_negative_emotion',
                    'score': emotion_state.intensity,
                    'severity': 'high',
                    'evidence': f'极端负面情感: {emotion_state.emotion}'
                })
            
            # 社交孤立
            if context.get('social_isolation_score', 0) > 0.6:
                risk_factors.append({
                    'factor': 'social_isolation',
                    'score': context['social_isolation_score'],
                    'severity': 'medium',
                    'evidence': '社交严重孤立'
                })
            
            # 既往自杀史
            if context.get('previous_suicide_attempts', False):
                risk_factors.append({
                    'factor': 'previous_attempts',
                    'score': 0.9,
                    'severity': 'critical',
                    'evidence': '有自杀未遂史'
                })
            
            # 计算综合自杀风险分数
            if not risk_factors:
                total_risk = 0.0
            else:
                # 使用加权平均，高严重程度的因子权重更大
                severity_weights = {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.4}
                
                weighted_sum = sum(
                    factor['score'] * severity_weights.get(factor['severity'], 0.5)
                    for factor in risk_factors
                )
                total_weight = sum(
                    severity_weights.get(factor['severity'], 0.5)
                    for factor in risk_factors
                )
                
                total_risk = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            return min(1.0, total_risk), risk_factors
            
        except Exception as e:
            logger.error(f"自杀风险评估失败: {str(e)}")
            return 0.0, []
    
    async def trigger_emergency_response(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment
    ) -> Dict[str, Any]:
        """
        触发紧急响应流程
        
        Args:
            user_id: 用户ID
            crisis_assessment: 危机评估结果
            
        Returns:
            应急响应执行结果
        """
        try:
            response_actions = []
            
            if crisis_assessment.severity_level == SeverityLevel.CRITICAL.value:
                # 关键危机响应
                response_actions.extend(await self._handle_critical_crisis(user_id))
                
            elif crisis_assessment.severity_level == SeverityLevel.SEVERE.value:
                # 严重危机响应
                response_actions.extend(await self._handle_severe_crisis(user_id))
                
            elif crisis_assessment.crisis_detected:
                # 一般危机响应
                response_actions.extend(await self._handle_moderate_crisis(user_id))
            
            # 激活监护模式
            monitoring_result = await self._activate_monitoring(user_id, crisis_assessment)
            response_actions.append(monitoring_result)
            
            # 记录响应日志
            await self._log_emergency_response(user_id, crisis_assessment, response_actions)
            
            return {
                'user_id': user_id,
                'response_level': crisis_assessment.severity_level,
                'actions_taken': response_actions,
                'monitoring_activated': True,
                'professional_notified': crisis_assessment.professional_required,
                'timestamp': utc_now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"紧急响应触发失败: {str(e)}")
            return {'error': str(e)}
    
    # 私有方法实现
    async def _analyze_language_indicators(self, user_input: str) -> List[Dict[str, Any]]:
        """分析语言指标"""
        indicators = []
        
        if not user_input:
            return indicators
        
        user_input_lower = user_input.lower()
        
        for category, keywords in self.crisis_keywords.items():
            score = self._analyze_keyword_category(user_input, category)
            if score > 0.2:  # 阈值
                indicators.append({
                    'type': 'language_indicator',
                    'category': category,
                    'score': score,
                    'evidence': f'检测到{category}相关表达',
                    'matched_keywords': [kw for kw in keywords if kw in user_input_lower]
                })
        
        return indicators
    
    async def _analyze_emotional_indicators(self, emotion_state: EmotionState) -> List[Dict[str, Any]]:
        """分析情感指标"""
        indicators = []
        
        # 极端负面情感
        if emotion_state.emotion in ['despair', 'hopelessness', 'severe_depression']:
            if emotion_state.intensity > 0.7:
                indicators.append({
                    'type': 'emotional_indicator',
                    'category': 'extreme_negative_emotion',
                    'score': emotion_state.intensity,
                    'evidence': f'极端负面情感: {emotion_state.emotion}',
                    'emotion_details': {
                        'emotion': emotion_state.emotion,
                        'intensity': emotion_state.intensity,
                        'valence': emotion_state.valence
                    }
                })
        
        # 情感麻木
        if emotion_state.intensity < 0.2 and emotion_state.emotion == 'numb':
            indicators.append({
                'type': 'emotional_indicator',
                'category': 'emotional_numbness',
                'score': 1.0 - emotion_state.intensity,
                'evidence': '情感麻木状态',
                'emotion_details': {
                    'intensity': emotion_state.intensity,
                    'emotion': emotion_state.emotion
                }
            })
        
        # 极度焦虑激动
        if emotion_state.emotion in ['panic', 'severe_anxiety'] and emotion_state.arousal > 0.8:
            indicators.append({
                'type': 'emotional_indicator',
                'category': 'extreme_agitation',
                'score': emotion_state.arousal,
                'evidence': '极度焦虑激动状态',
                'emotion_details': {
                    'arousal': emotion_state.arousal,
                    'emotion': emotion_state.emotion
                }
            })
        
        return indicators
    
    async def _analyze_behavioral_indicators(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析行为指标"""
        indicators = []
        
        for pattern in self.crisis_behavioral_patterns:
            if context.get(pattern, False):
                score = context.get(f'{pattern}_score', 0.7)
                indicators.append({
                    'type': 'behavioral_indicator',
                    'category': pattern,
                    'score': score,
                    'evidence': f'检测到危机行为模式: {pattern}',
                    'behavioral_details': context.get(f'{pattern}_details', {})
                })
        
        return indicators
    
    async def _analyze_pattern_indicators(self, emotion_history: List[EmotionState]) -> List[Dict[str, Any]]:
        """分析历史模式指标"""
        indicators = []
        
        if len(emotion_history) < 5:
            return indicators
        
        recent_emotions = emotion_history[-10:] if len(emotion_history) >= 10 else emotion_history
        
        # 急剧情绪恶化
        deterioration_score = self._calculate_deterioration_rate(recent_emotions)
        if deterioration_score > 0.6:
            indicators.append({
                'type': 'pattern_indicator',
                'category': 'rapid_deterioration',
                'score': deterioration_score,
                'evidence': '情绪状态急剧恶化',
                'pattern_details': {
                    'timespan_hours': (recent_emotions[-1].timestamp - recent_emotions[0].timestamp).total_seconds() / 3600,
                    'deterioration_rate': deterioration_score
                }
            })
        
        # 持续低效价情感
        low_valence_ratio = sum(1 for e in recent_emotions if e.valence < -0.5) / len(recent_emotions)
        if low_valence_ratio > 0.7:
            indicators.append({
                'type': 'pattern_indicator',
                'category': 'sustained_negative_emotion',
                'score': low_valence_ratio,
                'evidence': '持续强烈负面情感状态',
                'pattern_details': {
                    'low_valence_ratio': low_valence_ratio,
                    'duration_hours': (recent_emotions[-1].timestamp - recent_emotions[0].timestamp).total_seconds() / 3600
                }
            })
        
        return indicators
    
    async def _assess_protective_factors(self, context: Dict[str, Any]) -> float:
        """评估保护性因子"""
        protective_score = 0.0
        factor_count = 0
        
        for factor in self.protective_factors:
            if context.get(factor, False):
                protective_score += context.get(f'{factor}_strength', 0.5)
                factor_count += 1
        
        if factor_count > 0:
            return protective_score / factor_count
        else:
            return 0.1  # 默认较低保护性
    
    def _analyze_keyword_category(self, user_input: str, category: str) -> float:
        """分析特定类别关键词匹配度"""
        if not user_input or category not in self.crisis_keywords:
            return 0.0
        
        user_input_lower = user_input.lower()
        keywords = self.crisis_keywords[category]
        
        matched_count = sum(1 for kw in keywords if kw in user_input_lower)
        return min(1.0, matched_count / len(keywords) * 2.0)  # 放大匹配效应
    
    def _calculate_crisis_score(self, indicators: List[Dict[str, Any]], protective_score: float) -> float:
        """计算综合危机分数"""
        if not indicators:
            return 0.0
        
        # 按类型加权
        type_weights = {
            'language_indicator': 0.4,
            'emotional_indicator': 0.3,
            'behavioral_indicator': 0.2,
            'pattern_indicator': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for indicator in indicators:
            indicator_type = indicator['type']
            weight = type_weights.get(indicator_type, 0.1)
            weighted_score += indicator['score'] * weight
            total_weight += weight
        
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = 0.0
        
        # 应用保护性因子折扣
        protection_discount = max(0.1, 1.0 - protective_score)
        final_score = base_score * protection_discount
        
        return min(1.0, final_score)
    
    def _determine_severity_level(self, crisis_score: float) -> str:
        """确定严重程度"""
        if crisis_score >= 0.9:
            return SeverityLevel.CRITICAL.value
        elif crisis_score >= 0.7:
            return SeverityLevel.SEVERE.value
        elif crisis_score >= 0.5:
            return SeverityLevel.MODERATE.value
        else:
            return SeverityLevel.MILD.value
    
    def _determine_crisis_type(self, indicators: List[Dict[str, Any]]) -> str:
        """确定危机类型"""
        # 统计各类指标
        emotional_count = sum(1 for ind in indicators if ind['type'] == 'emotional_indicator')
        behavioral_count = sum(1 for ind in indicators if ind['type'] == 'behavioral_indicator')
        
        if emotional_count > behavioral_count:
            return 'emotional'
        elif behavioral_count > emotional_count:
            return 'behavioral'
        else:
            return 'mixed'
    
    def _determine_monitoring_level(self, crisis_score: float) -> str:
        """确定监护级别"""
        if crisis_score >= 0.8:
            return 'intensive'
        elif crisis_score >= 0.6:
            return 'elevated'
        else:
            return 'standard'
    
    def _determine_check_frequency(self, crisis_score: float) -> timedelta:
        """确定检查频率"""
        if crisis_score >= 0.9:
            return timedelta(minutes=10)
        elif crisis_score >= 0.7:
            return timedelta(minutes=30)
        elif crisis_score >= 0.5:
            return timedelta(hours=1)
        else:
            return timedelta(hours=4)
    
    def _calculate_confidence(self, indicators: List[Dict[str, Any]]) -> float:
        """计算检测置信度"""
        if not indicators:
            return 0.1
        
        # 基于指标数量和质量计算置信度
        indicator_count = len(indicators)
        avg_score = sum(ind['score'] for ind in indicators) / indicator_count
        
        # 多种类型指标提高置信度
        unique_types = len(set(ind['type'] for ind in indicators))
        type_diversity_bonus = min(0.3, unique_types * 0.1)
        
        confidence = min(1.0, (indicator_count / 5.0) * avg_score + type_diversity_bonus)
        return confidence
    
    async def _generate_immediate_actions(self, severity_level: str, indicators: List[Dict[str, Any]]) -> List[str]:
        """生成即时行动建议"""
        actions = []
        
        if severity_level == SeverityLevel.CRITICAL.value:
            actions.extend([
                '立即联系当地紧急服务 (110/120)',
                '连接心理危机干预热线',
                '通知紧急联系人',
                '确保用户不独处',
                '移除危险物品',
                '持续安全陪伴'
            ])
        elif severity_level == SeverityLevel.SEVERE.value:
            actions.extend([
                '联系心理危机干预热线',
                '安排紧急心理评估',
                '通知可信赖的亲友',
                '制定安全计划',
                '增加监护频率'
            ])
        elif severity_level == SeverityLevel.MODERATE.value:
            actions.extend([
                '提供即时情感支持',
                '推荐专业心理咨询',
                '激活社会支持网络',
                '制定应对策略',
                '安排定期检查'
            ])
        else:
            actions.extend([
                '提供情感支持和倾听',
                '推荐心理健康资源',
                '鼓励寻求支持',
                '持续关注状态变化'
            ])
        
        # 根据具体指标添加针对性行动
        for indicator in indicators:
            if indicator['category'] == 'suicide':
                actions.append('立即进行自杀风险评估')
            elif indicator['category'] == 'self_harm':
                actions.append('评估自伤风险并制定安全措施')
        
        return list(set(actions))  # 去重
    
    async def _handle_critical_crisis(self, user_id: str) -> List[Dict[str, Any]]:
        """处理关键危机"""
        actions = []
        
        # 立即连接危机热线
        actions.append({
            'action': 'connect_crisis_hotline',
            'status': 'initiated',
            'timestamp': utc_now().isoformat(),
            'details': '尝试连接全国心理危机干预热线'
        })
        
        # 通知紧急联系人
        actions.append({
            'action': 'notify_emergency_contacts',
            'status': 'initiated', 
            'timestamp': utc_now().isoformat(),
            'details': '通知用户紧急联系人'
        })
        
        # 激活24/7监护
        actions.append({
            'action': 'activate_intensive_monitoring',
            'status': 'activated',
            'timestamp': utc_now().isoformat(),
            'details': '激活24/7密集监护模式'
        })
        
        return actions
    
    async def _handle_severe_crisis(self, user_id: str) -> List[Dict[str, Any]]:
        """处理严重危机"""
        actions = []
        
        # 提供即时支持
        actions.append({
            'action': 'provide_immediate_support',
            'status': 'active',
            'timestamp': utc_now().isoformat(),
            'details': '提供即时情感支持和稳定'
        })
        
        # 推荐专业服务
        actions.append({
            'action': 'recommend_professional_services',
            'status': 'recommended',
            'timestamp': utc_now().isoformat(),
            'details': '推荐专业心理健康服务'
        })
        
        return actions
    
    async def _handle_moderate_crisis(self, user_id: str) -> List[Dict[str, Any]]:
        """处理中等危机"""
        actions = []
        
        # 增强支持
        actions.append({
            'action': 'enhanced_support',
            'status': 'active',
            'timestamp': utc_now().isoformat(),
            'details': '提供增强的情感支持'
        })
        
        return actions
    
    async def _activate_monitoring(self, user_id: str, assessment: CrisisAssessment) -> Dict[str, Any]:
        """激活监护模式"""
        return {
            'action': 'activate_monitoring',
            'status': 'activated',
            'monitoring_level': assessment.monitoring_level,
            'check_frequency': assessment.check_frequency.total_seconds(),
            'timestamp': utc_now().isoformat(),
            'details': f'激活{assessment.monitoring_level}级监护'
        }
    
    async def _log_emergency_response(
        self,
        user_id: str,
        assessment: CrisisAssessment,
        actions: List[Dict[str, Any]]
    ):
        """记录紧急响应日志"""
        log_entry = {
            'user_id': user_id,
            'timestamp': utc_now().isoformat(),
            'crisis_score': assessment.risk_score,
            'severity_level': assessment.severity_level,
            'actions_count': len(actions),
            'professional_required': assessment.professional_required
        }
        
        logger.info(f"紧急响应记录: {log_entry}")
    
    def _calculate_deterioration_rate(self, emotions: List[EmotionState]) -> float:
        """计算情绪恶化速度"""
        if len(emotions) < 3:
            return 0.0
        
        # 计算效价的下降趋势斜率
        valences = [e.valence for e in emotions]
        
        # 简单线性趋势
        n = len(valences)
        sum_x = sum(range(n))
        sum_y = sum(valences)
        sum_xy = sum(i * val for i, val in enumerate(valences))
        sum_x2 = sum(i * i for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # 负斜率表示下降，转换为正的恶化率
        deterioration_rate = max(0.0, -slope)
        return min(1.0, deterioration_rate * 5.0)  # 放大效应
