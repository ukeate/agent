"""
智能干预策略引擎
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from .models import (
    InterventionPlan, InterventionStrategy, RiskAssessment, RiskLevel,
    InterventionType, DecisionContext
)
from ..emotion_modeling.models import PersonalityProfile, EmotionState


logger = logging.getLogger(__name__)


@dataclass
class InterventionResource:
    """干预资源"""
    resource_id: str
    resource_type: str  # professional, digital, community, emergency
    name: str
    description: str
    availability: str  # 24/7, business_hours, appointment_only
    contact_info: Dict[str, str]
    specialization: List[str]
    location: Optional[str] = None
    cost: Optional[str] = None
    rating: Optional[float] = None


class InterventionStrategySelector:
    """干预策略选择器"""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.resources = self._initialize_resources()
        
    def _initialize_strategies(self) -> Dict[str, InterventionStrategy]:
        """初始化干预策略库"""
        strategies = {}
        
        # 低风险支持策略
        strategies['positive_reinforcement'] = InterventionStrategy(
            strategy_name='positive_reinforcement',
            description='积极强化和肯定支持',
            intervention_type=InterventionType.PREVENTIVE.value,
            implementation_steps=[
                '识别用户积极行为和成就',
                '提供具体的肯定反馈',
                '强化正向情感体验',
                '鼓励继续积极行为'
            ],
            expected_effectiveness=0.7,
            target_emotions=['sadness', 'neutral', 'mild_anxiety'],
            target_risk_levels=[RiskLevel.LOW.value, RiskLevel.MEDIUM.value]
        )
        
        strategies['mindfulness_reminder'] = InterventionStrategy(
            strategy_name='mindfulness_reminder',
            description='正念冥想提醒和指导',
            intervention_type=InterventionType.PREVENTIVE.value,
            implementation_steps=[
                '提供正念练习指导',
                '设定定期正念提醒',
                '引导呼吸练习',
                '分享正念资源和技巧'
            ],
            expected_effectiveness=0.6,
            target_emotions=['anxiety', 'stress', 'overwhelm'],
            target_risk_levels=[RiskLevel.LOW.value, RiskLevel.MEDIUM.value]
        )
        
        # 中风险干预策略
        strategies['active_listening_session'] = InterventionStrategy(
            strategy_name='active_listening_session',
            description='主动倾听和情感支持会话',
            intervention_type=InterventionType.SUPPORTIVE.value,
            implementation_steps=[
                '创建安全的倾诉环境',
                '运用共情技巧理解用户',
                '反映用户情感和体验',
                '提供情感验证和支持'
            ],
            expected_effectiveness=0.8,
            target_emotions=['sadness', 'loneliness', 'frustration'],
            target_risk_levels=[RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]
        )
        
        strategies['coping_skill_teaching'] = InterventionStrategy(
            strategy_name='coping_skill_teaching',
            description='应对技能教学和练习',
            intervention_type=InterventionType.SUPPORTIVE.value,
            implementation_steps=[
                '评估用户当前应对方式',
                '教授适合的应对技巧',
                '指导技能练习',
                '跟踪应用效果'
            ],
            expected_effectiveness=0.75,
            target_emotions=['anxiety', 'anger', 'stress'],
            target_risk_levels=[RiskLevel.MEDIUM.value]
        )
        
        # 高风险干预策略
        strategies['intensive_emotional_support'] = InterventionStrategy(
            strategy_name='intensive_emotional_support',
            description='强化情感支持和陪伴',
            intervention_type=InterventionType.CORRECTIVE.value,
            implementation_steps=[
                '提供持续密集的情感支持',
                '增加互动频率和深度',
                '监控情感状态变化',
                '及时调整支持策略'
            ],
            expected_effectiveness=0.85,
            target_emotions=['depression', 'despair', 'hopelessness'],
            target_risk_levels=[RiskLevel.HIGH.value]
        )
        
        strategies['professional_referral'] = InterventionStrategy(
            strategy_name='professional_referral',
            description='专业心理健康服务转介',
            intervention_type=InterventionType.CORRECTIVE.value,
            implementation_steps=[
                '评估专业服务需求',
                '推荐合适的专业机构',
                '协助预约和转介',
                '跟进专业服务进展'
            ],
            expected_effectiveness=0.9,
            target_emotions=['severe_depression', 'severe_anxiety'],
            target_risk_levels=[RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        )
        
        # 危机干预策略
        strategies['immediate_crisis_intervention'] = InterventionStrategy(
            strategy_name='immediate_crisis_intervention',
            description='即时危机干预和安全保障',
            intervention_type=InterventionType.CRISIS.value,
            implementation_steps=[
                '立即评估安全风险',
                '提供危机稳定支持',
                '联系紧急服务',
                '持续监护和跟进'
            ],
            expected_effectiveness=0.95,
            target_emotions=['suicidal_ideation', 'self_harm'],
            target_risk_levels=[RiskLevel.CRITICAL.value]
        )
        
        return strategies
    
    def _initialize_resources(self) -> List[InterventionResource]:
        """初始化干预资源库"""
        resources = []
        
        # 专业服务资源
        resources.append(InterventionResource(
            resource_id='crisis_hotline_national',
            resource_type='emergency',
            name='全国心理危机干预热线',
            description='24小时心理危机干预和自杀预防热线',
            availability='24/7',
            contact_info={'phone': '400-161-9995', 'website': 'www.crisis-hotline.org'},
            specialization=['crisis_intervention', 'suicide_prevention']
        ))
        
        resources.append(InterventionResource(
            resource_id='online_therapy_platform',
            resource_type='professional',
            name='在线心理咨询平台',
            description='专业心理咨询师在线服务',
            availability='appointment_only',
            contact_info={'website': 'www.online-therapy.com', 'app': 'TherapyApp'},
            specialization=['depression', 'anxiety', 'trauma']
        ))
        
        # 数字化资源
        resources.append(InterventionResource(
            resource_id='mindfulness_app',
            resource_type='digital',
            name='正念冥想应用',
            description='引导式正念冥想和放松练习',
            availability='24/7',
            contact_info={'app_store': 'MindfulnessApp', 'website': 'www.mindful.com'},
            specialization=['mindfulness', 'stress_reduction', 'sleep']
        ))
        
        # 社区资源
        resources.append(InterventionResource(
            resource_id='support_group_online',
            resource_type='community',
            name='在线互助支持群组',
            description='同伴互助和经验分享群组',
            availability='business_hours',
            contact_info={'website': 'www.support-groups.org'},
            specialization=['peer_support', 'group_therapy']
        ))
        
        return resources
    
    async def select_intervention_strategies(
        self,
        risk_assessment: RiskAssessment,
        user_preferences: Optional[Dict[str, Any]] = None,
        past_effectiveness: Optional[Dict[str, float]] = None,
        personality_profile: Optional[PersonalityProfile] = None
    ) -> List[InterventionStrategy]:
        """
        选择适合的干预策略
        
        Args:
            risk_assessment: 风险评估结果
            user_preferences: 用户偏好
            past_effectiveness: 历史策略效果
            personality_profile: 个性画像
            
        Returns:
            推荐的干预策略列表
        """
        try:
            # 基于风险等级筛选候选策略
            candidate_strategies = self._filter_strategies_by_risk(risk_assessment.risk_level)
            
            # 基于风险因子进一步筛选
            candidate_strategies = self._filter_strategies_by_factors(
                candidate_strategies, 
                risk_assessment.risk_factors
            )
            
            # 计算策略适用性分数
            strategy_scores = []
            for strategy in candidate_strategies:
                score = await self._calculate_strategy_score(
                    strategy,
                    risk_assessment,
                    user_preferences,
                    past_effectiveness,
                    personality_profile
                )
                strategy_scores.append((strategy, score))
            
            # 按分数排序
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前3个策略
            selected_strategies = [strategy for strategy, score in strategy_scores[:3] if score > 0.5]
            
            logger.info(f"为风险等级 {risk_assessment.risk_level} 选择了 {len(selected_strategies)} 个策略")
            return selected_strategies
            
        except Exception as e:
            logger.error(f"策略选择失败: {str(e)}")
            return []
    
    async def create_intervention_plan(
        self,
        risk_assessment: RiskAssessment,
        strategies: List[InterventionStrategy],
        context: Optional[DecisionContext] = None
    ) -> InterventionPlan:
        """
        创建干预计划
        
        Args:
            risk_assessment: 风险评估
            strategies: 选择的策略
            context: 上下文信息
            
        Returns:
            干预计划
        """
        try:
            # 确定主要策略
            primary_strategy = strategies[0] if strategies else None
            
            # 制定时间安排
            timeline = self._create_timeline(risk_assessment.risk_level, strategies)
            
            # 分配资源
            allocated_resources = await self._allocate_resources(strategies, risk_assessment)
            
            # 定义监控频率
            monitoring_frequency = self._determine_monitoring_frequency(risk_assessment.risk_level)
            
            # 设定成功指标
            success_metrics = self._define_success_metrics(strategies, risk_assessment)
            
            plan = InterventionPlan(
                user_id=risk_assessment.user_id,
                intervention_type=self._determine_intervention_type(risk_assessment.risk_level),
                urgency_level=self._determine_urgency_level(risk_assessment),
                target_risk_factors=[factor.factor_type for factor in risk_assessment.risk_factors],
                strategies=strategies,
                primary_strategy=primary_strategy,
                resources=allocated_resources,
                timeline=timeline,
                success_metrics=success_metrics,
                monitoring_frequency=monitoring_frequency,
                status='draft'
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"干预计划创建失败: {str(e)}")
            raise
    
    async def evaluate_strategy_effectiveness(
        self,
        strategy: InterventionStrategy,
        before_state: EmotionState,
        after_state: EmotionState,
        user_feedback: Optional[str] = None
    ) -> float:
        """
        评估策略效果
        
        Args:
            strategy: 干预策略
            before_state: 干预前情感状态
            after_state: 干预后情感状态
            user_feedback: 用户反馈
            
        Returns:
            效果评分 [0,1]
        """
        try:
            effectiveness_score = 0.0
            
            # 情感改善评估
            emotional_improvement = self._assess_emotional_improvement(before_state, after_state)
            effectiveness_score += emotional_improvement * 0.4
            
            # 风险降低评估
            risk_reduction = self._assess_risk_reduction(before_state, after_state)
            effectiveness_score += risk_reduction * 0.3
            
            # 用户满意度评估
            satisfaction_score = self._assess_user_satisfaction(user_feedback)
            effectiveness_score += satisfaction_score * 0.2
            
            # 策略执行完整度评估
            completion_score = 0.8  # 假设大部分策略能完整执行
            effectiveness_score += completion_score * 0.1
            
            return min(1.0, effectiveness_score)
            
        except Exception as e:
            logger.error(f"策略效果评估失败: {str(e)}")
            return 0.0
    
    # 私有方法实现
    def _filter_strategies_by_risk(self, risk_level: str) -> List[InterventionStrategy]:
        """根据风险等级筛选策略"""
        filtered_strategies = []
        
        for strategy in self.strategies.values():
            if risk_level in strategy.target_risk_levels:
                filtered_strategies.append(strategy)
        
        return filtered_strategies
    
    def _filter_strategies_by_factors(
        self,
        strategies: List[InterventionStrategy],
        risk_factors: List
    ) -> List[InterventionStrategy]:
        """根据风险因子筛选策略"""
        # 简化实现，实际可以根据具体风险因子匹配策略
        return strategies
    
    async def _calculate_strategy_score(
        self,
        strategy: InterventionStrategy,
        risk_assessment: RiskAssessment,
        user_preferences: Optional[Dict[str, Any]],
        past_effectiveness: Optional[Dict[str, float]],
        personality_profile: Optional[PersonalityProfile]
    ) -> float:
        """计算策略适用性分数"""
        score = 0.0
        
        # 基础适用性分数
        base_score = strategy.expected_effectiveness
        score += base_score * 0.4
        
        # 用户偏好分数
        if user_preferences:
            preference_score = user_preferences.get(strategy.strategy_name, 0.5)
            score += preference_score * 0.3
        else:
            score += 0.5 * 0.3
        
        # 历史效果分数
        if past_effectiveness:
            historical_score = past_effectiveness.get(strategy.strategy_name, 0.5)
            score += historical_score * 0.2
        else:
            score += 0.5 * 0.2
        
        # 个性匹配分数
        if personality_profile:
            personality_score = self._assess_personality_match(strategy, personality_profile)
            score += personality_score * 0.1
        else:
            score += 0.5 * 0.1
        
        return min(1.0, score)
    
    def _assess_personality_match(
        self,
        strategy: InterventionStrategy,
        personality_profile: PersonalityProfile
    ) -> float:
        """评估策略与个性的匹配度"""
        # 简化的个性匹配逻辑
        match_score = 0.5  # 默认中等匹配
        
        # 根据策略类型和个性特质调整匹配分数
        if strategy.strategy_name == 'mindfulness_reminder':
            # 正念策略适合开放性高的用户
            openness = personality_profile.emotional_traits.get('openness', 0.5)
            match_score = openness
        
        elif strategy.strategy_name == 'active_listening_session':
            # 倾听策略适合需要社交支持的用户
            extraversion = personality_profile.emotional_traits.get('extraversion', 0.5)
            match_score = 1.0 - extraversion  # 内向用户更需要倾听支持
        
        return match_score
    
    def _create_timeline(
        self,
        risk_level: str,
        strategies: List[InterventionStrategy]
    ) -> Dict[str, datetime]:
        """创建干预时间安排"""
        now = datetime.now()
        timeline = {}
        
        if risk_level == RiskLevel.CRITICAL.value:
            timeline = {
                'immediate_start': now,
                'first_intervention': now + timedelta(minutes=15),
                'safety_check': now + timedelta(hours=1),
                'professional_contact': now + timedelta(hours=2),
                'follow_up_1': now + timedelta(hours=6),
                'follow_up_2': now + timedelta(hours=12)
            }
        elif risk_level == RiskLevel.HIGH.value:
            timeline = {
                'plan_activation': now + timedelta(minutes=30),
                'first_intervention': now + timedelta(hours=1),
                'progress_check': now + timedelta(hours=6),
                'strategy_adjustment': now + timedelta(days=1),
                'weekly_review': now + timedelta(days=7)
            }
        elif risk_level == RiskLevel.MEDIUM.value:
            timeline = {
                'plan_start': now + timedelta(hours=2),
                'first_intervention': now + timedelta(hours=4),
                'progress_check': now + timedelta(days=1),
                'weekly_review': now + timedelta(days=7),
                'plan_review': now + timedelta(days=14)
            }
        else:
            timeline = {
                'plan_start': now + timedelta(hours=6),
                'first_check': now + timedelta(days=1),
                'weekly_review': now + timedelta(days=7),
                'monthly_review': now + timedelta(days=30)
            }
        
        return timeline
    
    async def _allocate_resources(
        self,
        strategies: List[InterventionStrategy],
        risk_assessment: RiskAssessment
    ) -> List[Dict[str, Any]]:
        """分配干预资源"""
        allocated_resources = []
        
        for strategy in strategies:
            # 根据策略需求匹配资源
            matching_resources = self._find_matching_resources(strategy, risk_assessment)
            
            for resource in matching_resources:
                allocated_resources.append({
                    'resource_id': resource.resource_id,
                    'resource_type': resource.resource_type,
                    'name': resource.name,
                    'contact_info': resource.contact_info,
                    'allocation_reason': f"支持策略: {strategy.strategy_name}",
                    'urgency': self._determine_resource_urgency(resource, risk_assessment)
                })
        
        return allocated_resources
    
    def _find_matching_resources(
        self,
        strategy: InterventionStrategy,
        risk_assessment: RiskAssessment
    ) -> List[InterventionResource]:
        """查找匹配的资源"""
        matching_resources = []
        
        for resource in self.resources:
            # 检查资源专业领域是否匹配
            if strategy.intervention_type == InterventionType.CRISIS.value:
                if resource.resource_type == 'emergency':
                    matching_resources.append(resource)
            elif strategy.strategy_name == 'professional_referral':
                if resource.resource_type == 'professional':
                    matching_resources.append(resource)
            elif strategy.strategy_name == 'mindfulness_reminder':
                if 'mindfulness' in resource.specialization:
                    matching_resources.append(resource)
        
        return matching_resources
    
    def _determine_resource_urgency(
        self,
        resource: InterventionResource,
        risk_assessment: RiskAssessment
    ) -> str:
        """确定资源分配紧急程度"""
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            return 'immediate'
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            return 'urgent'
        elif risk_assessment.risk_level == RiskLevel.MEDIUM.value:
            return 'standard'
        else:
            return 'routine'
    
    def _determine_monitoring_frequency(self, risk_level: str) -> timedelta:
        """确定监控频率"""
        frequency_map = {
            RiskLevel.CRITICAL.value: timedelta(minutes=15),
            RiskLevel.HIGH.value: timedelta(hours=1),
            RiskLevel.MEDIUM.value: timedelta(hours=4),
            RiskLevel.LOW.value: timedelta(hours=12)
        }
        return frequency_map.get(risk_level, timedelta(hours=6))
    
    def _define_success_metrics(
        self,
        strategies: List[InterventionStrategy],
        risk_assessment: RiskAssessment
    ) -> List[str]:
        """定义成功指标"""
        metrics = [
            'risk_score_reduction_percentage',
            'emotional_stability_improvement',
            'user_engagement_level',
            'intervention_completion_rate'
        ]
        
        # 根据风险等级添加特定指标
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            metrics.extend([
                'safety_maintained',
                'professional_service_connected',
                'crisis_resolution_time'
            ])
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            metrics.extend([
                'symptom_severity_reduction',
                'coping_skill_improvement',
                'support_system_activation'
            ])
        
        return metrics
    
    def _determine_intervention_type(self, risk_level: str) -> str:
        """确定干预类型"""
        type_map = {
            RiskLevel.CRITICAL.value: InterventionType.CRISIS.value,
            RiskLevel.HIGH.value: InterventionType.CORRECTIVE.value,
            RiskLevel.MEDIUM.value: InterventionType.SUPPORTIVE.value,
            RiskLevel.LOW.value: InterventionType.PREVENTIVE.value
        }
        return type_map.get(risk_level, InterventionType.SUPPORTIVE.value)
    
    def _determine_urgency_level(self, risk_assessment: RiskAssessment) -> str:
        """确定紧急程度"""
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            return 'critical'
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            return 'high'
        elif risk_assessment.risk_level == RiskLevel.MEDIUM.value:
            return 'medium'
        else:
            return 'low'
    
    def _assess_emotional_improvement(
        self,
        before_state: EmotionState,
        after_state: EmotionState
    ) -> float:
        """评估情感改善程度"""
        # 效价改善
        valence_improvement = max(0.0, after_state.valence - before_state.valence)
        
        # 强度变化(负面情感强度降低为好)
        if before_state.valence < 0:  # 负面情感
            intensity_improvement = max(0.0, before_state.intensity - after_state.intensity)
        else:  # 正面情感
            intensity_improvement = max(0.0, after_state.intensity - before_state.intensity)
        
        # 综合改善分数
        total_improvement = (valence_improvement + intensity_improvement) / 2.0
        return min(1.0, total_improvement)
    
    def _assess_risk_reduction(
        self,
        before_state: EmotionState,
        after_state: EmotionState
    ) -> float:
        """评估风险降低程度"""
        # 简化的风险评估，基于效价和强度变化
        before_risk = max(0.0, (1.0 - before_state.valence) * before_state.intensity)
        after_risk = max(0.0, (1.0 - after_state.valence) * after_state.intensity)
        
        risk_reduction = max(0.0, before_risk - after_risk)
        return min(1.0, risk_reduction)
    
    def _assess_user_satisfaction(self, user_feedback: Optional[str]) -> float:
        """评估用户满意度"""
        if not user_feedback:
            return 0.5  # 默认中等满意度
        
        # 简化的情感分析
        positive_keywords = ['好', '有用', '有帮助', '感谢', '满意']
        negative_keywords = ['差', '没用', '无效', '不满意', '失望']
        
        feedback_lower = user_feedback.lower()
        
        positive_count = sum(1 for kw in positive_keywords if kw in feedback_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in feedback_lower)
        
        if positive_count > negative_count:
            return 0.8
        elif negative_count > positive_count:
            return 0.2
        else:
            return 0.5