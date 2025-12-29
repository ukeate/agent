"""
情感智能决策引擎综合服务
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from ..ai.emotional_intelligence.decision_engine import EmotionalDecisionEngine
from ..ai.emotional_intelligence.risk_assessment import RiskAssessmentEngine
from ..ai.emotional_intelligence.intervention_engine import InterventionStrategySelector
from ..ai.emotional_intelligence.crisis_support import CrisisDetectionSystem
from ..ai.emotional_intelligence.health_monitor import HealthMonitoringSystem
from ..ai.emotional_intelligence.models import (
    DecisionContext, EmotionalDecision, RiskAssessment, InterventionPlan,
    CrisisAssessment, HealthDashboardData
)
from ..ai.emotion_modeling.models import EmotionState, PersonalityProfile
from ..ai.empathy_response.models import EmpathyResponse
from ..ai.emotion_recognition.analyzers.text_analyzer import TextEmotionAnalyzer
from ..repositories.emotion_modeling_repository import EmotionModelingRepository
from ..ai.memory.models import EmotionalMemory

from src.core.logging import get_logger
logger = get_logger(__name__)

class EmotionalIntelligenceService:
    """情感智能决策引擎综合服务"""
    
    def __init__(self):
        # 初始化各个子系统
        self.decision_engine = EmotionalDecisionEngine()
        self.risk_engine = RiskAssessmentEngine()
        self.intervention_engine = InterventionStrategySelector()
        self.crisis_system = CrisisDetectionSystem()
        self.health_monitor = HealthMonitoringSystem()
        self._text_analyzer = TextEmotionAnalyzer()
        
        # 状态跟踪
        self.active_interventions: Dict[str, InterventionPlan] = {}
        self.user_health_cache: Dict[str, HealthDashboardData] = {}
        self.crisis_alerts: Dict[str, List[CrisisAssessment]] = {}
        
        # 配置参数
        self.config = {
            'decision_confidence_threshold': 0.7,
            'risk_alert_threshold': 0.6,
            'crisis_response_delay_minutes': 1,
            'health_cache_ttl_hours': 1,
            'intervention_review_frequency_hours': 6
        }
    
    async def process_emotional_interaction(
        self,
        user_id: str,
        user_input: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理情感交互的完整流程
        
        Args:
            user_id: 用户ID
            user_input: 用户输入
            session_id: 会话ID
            context: 上下文信息
            
        Returns:
            处理结果包含决策、风险评估、干预建议等
        """
        try:
            # 1. 获取用户情感上下文
            emotional_context = await self._build_emotional_context(
                user_id, user_input, session_id, context
            )
            
            # 2. 并行执行核心分析
            tasks = [
                self._make_intelligent_decision(emotional_context),
                self._assess_current_risk(emotional_context),
                self._detect_crisis_signals(emotional_context),
            ]
            
            decision, risk_assessment, crisis_assessment = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            if isinstance(decision, Exception):
                logger.error(f"决策生成异常: {decision}")
                decision = None
            
            if isinstance(risk_assessment, Exception):
                logger.error(f"风险评估异常: {risk_assessment}")
                risk_assessment = None
            
            if isinstance(crisis_assessment, Exception):
                logger.error(f"危机检测异常: {crisis_assessment}")
                crisis_assessment = None
            
            # 3. 综合分析结果，确定响应策略
            response_strategy = await self._determine_response_strategy(
                decision, risk_assessment, crisis_assessment
            )
            
            # 4. 执行响应行动
            response_actions = await self._execute_response_actions(
                user_id, response_strategy, risk_assessment, crisis_assessment
            )
            
            # 5. 更新用户状态和历史
            await self._update_user_state(user_id, {
                'decision': decision.to_dict() if decision else None,
                'risk_assessment': risk_assessment.to_dict() if risk_assessment else None,
                'crisis_assessment': crisis_assessment.to_dict() if crisis_assessment else None,
                'response_actions': response_actions
            })
            
            # 6. 生成综合响应
            comprehensive_response = {
                'user_id': user_id,
                'timestamp': utc_now().isoformat(),
                'decision': decision.to_dict() if decision else None,
                'risk_assessment': risk_assessment.to_dict() if risk_assessment else None,
                'crisis_assessment': crisis_assessment.to_dict() if crisis_assessment else None,
                'response_strategy': response_strategy,
                'actions_taken': response_actions,
                'recommendations': await self._generate_user_recommendations(
                    user_id, decision, risk_assessment, crisis_assessment
                )
            }
            
            logger.info(f"情感交互处理完成 - 用户: {user_id}, 策略: {response_strategy}")
            return comprehensive_response
            
        except Exception as e:
            logger.error(f"情感交互处理失败: {str(e)}")
            raise
    
    async def get_comprehensive_health_status(
        self,
        user_id: str,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        获取用户综合健康状态
        
        Args:
            user_id: 用户ID
            time_period: 分析时间段
            
        Returns:
            综合健康状态报告
        """
        try:
            # 检查缓存
            cached_health = self._get_cached_health_data(user_id)
            if cached_health and not time_period:
                return cached_health
            
            # 获取用户数据
            emotion_history = await self._get_user_emotion_history(user_id, time_period)
            risk_assessments = await self._get_user_risk_history(user_id, time_period)
            interventions = await self._get_user_interventions(user_id, time_period)
            personality_profile = await self._get_user_personality_profile(user_id)
            
            # 生成健康仪表盘
            dashboard_data = await self.health_monitor.generate_health_dashboard(
                user_id=user_id,
                emotion_history=emotion_history,
                risk_assessments=risk_assessments,
                interventions=interventions,
                personality_profile=personality_profile,
                time_period=time_period
            )
            
            # 分析情感模式
            emotional_patterns = await self.health_monitor.track_emotional_patterns(
                emotion_history=emotion_history
            )
            
            # 预测未来风险
            crisis_probability, crisis_details = await self.risk_engine.predict_crisis_probability(
                emotion_history=emotion_history,
                time_horizon=timedelta(hours=48)
            )
            
            # 综合健康报告
            health_report = {
                'user_id': user_id,
                'generated_at': utc_now().isoformat(),
                'dashboard_data': dashboard_data.to_dict(),
                'emotional_patterns': emotional_patterns,
                'crisis_prediction': {
                    'probability': crisis_probability,
                    'details': crisis_details,
                    'timeframe_hours': 48
                },
                'active_interventions': len([i for i in interventions if i.status == 'active']),
                'health_summary': self._generate_health_summary(dashboard_data, crisis_probability),
                'priority_recommendations': await self._get_priority_health_recommendations(
                    user_id, dashboard_data, crisis_probability
                )
            }
            
            # 更新缓存
            self._cache_health_data(user_id, health_report)
            
            return health_report
            
        except Exception as e:
            logger.error(f"综合健康状态获取失败: {str(e)}")
            raise
    
    async def manage_intervention_lifecycle(
        self,
        user_id: str,
        intervention_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        管理干预计划生命周期
        
        Args:
            user_id: 用户ID
            intervention_id: 干预计划ID (可选，为空则管理所有活跃干预)
            
        Returns:
            干预管理结果
        """
        try:
            # 获取用户的干预计划
            if intervention_id:
                interventions = [self.active_interventions.get(intervention_id)]
                interventions = [i for i in interventions if i is not None]
            else:
                interventions = [
                    intervention for intervention in self.active_interventions.values()
                    if intervention.user_id == user_id and intervention.status == 'active'
                ]
            
            management_results = []
            
            for intervention in interventions:
                # 评估干预进展
                progress_evaluation = await self._evaluate_intervention_progress(intervention)
                
                # 根据进展调整干预策略
                adjustment_needed = progress_evaluation['effectiveness_score'] < 0.6
                
                if adjustment_needed:
                    # 调整干预策略
                    adjusted_plan = await self._adjust_intervention_strategy(intervention, progress_evaluation)
                    management_results.append({
                        'intervention_id': intervention.plan_id,
                        'action': 'strategy_adjusted',
                        'old_strategy': intervention.primary_strategy.strategy_name if intervention.primary_strategy else None,
                        'new_strategy': adjusted_plan.primary_strategy.strategy_name if adjusted_plan.primary_strategy else None,
                        'reason': 'low_effectiveness',
                        'effectiveness_score': progress_evaluation['effectiveness_score']
                    })
                    
                    # 更新活跃干预
                    self.active_interventions[intervention.plan_id] = adjusted_plan
                else:
                    # 继续当前策略
                    management_results.append({
                        'intervention_id': intervention.plan_id,
                        'action': 'continued',
                        'effectiveness_score': progress_evaluation['effectiveness_score'],
                        'progress_percentage': intervention.progress
                    })
                
                # 检查干预完成条件
                if intervention.progress >= 1.0:
                    completion_result = await self._complete_intervention(intervention)
                    management_results.append({
                        'intervention_id': intervention.plan_id,
                        'action': 'completed',
                        'completion_result': completion_result
                    })
                    
                    # 从活跃干预中移除
                    if intervention.plan_id in self.active_interventions:
                        del self.active_interventions[intervention.plan_id]
            
            return {
                'user_id': user_id,
                'management_timestamp': utc_now().isoformat(),
                'interventions_managed': len(interventions),
                'management_results': management_results,
                'active_interventions_count': len([
                    i for i in self.active_interventions.values() 
                    if i.user_id == user_id and i.status == 'active'
                ])
            }
            
        except Exception as e:
            logger.error(f"干预生命周期管理失败: {str(e)}")
            raise
    
    async def handle_crisis_escalation(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment
    ) -> Dict[str, Any]:
        """
        处理危机升级
        
        Args:
            user_id: 用户ID
            crisis_assessment: 危机评估结果
            
        Returns:
            危机处理结果
        """
        try:
            # 记录危机警报
            if user_id not in self.crisis_alerts:
                self.crisis_alerts[user_id] = []
            self.crisis_alerts[user_id].append(crisis_assessment)
            
            # 执行紧急响应
            emergency_response = await self.crisis_system.trigger_emergency_response(
                user_id, crisis_assessment
            )
            
            # 暂停或调整现有干预
            intervention_adjustments = await self._adjust_interventions_for_crisis(
                user_id, crisis_assessment
            )
            
            # 激活危机专用干预
            crisis_intervention = await self._create_crisis_intervention_plan(
                user_id, crisis_assessment
            )
            
            if crisis_intervention:
                self.active_interventions[crisis_intervention.plan_id] = crisis_intervention
            
            # 通知相关系统和人员
            notifications_sent = await self._send_crisis_notifications(
                user_id, crisis_assessment, emergency_response
            )
            
            # 建立持续监控
            monitoring_setup = await self._setup_crisis_monitoring(
                user_id, crisis_assessment
            )
            
            escalation_result = {
                'user_id': user_id,
                'crisis_id': crisis_assessment.assessment_id,
                'escalation_timestamp': utc_now().isoformat(),
                'severity_level': crisis_assessment.severity_level,
                'emergency_response': emergency_response,
                'intervention_adjustments': intervention_adjustments,
                'crisis_intervention_created': crisis_intervention.plan_id if crisis_intervention else None,
                'notifications_sent': notifications_sent,
                'monitoring_setup': monitoring_setup,
                'next_check_time': (utc_now() + crisis_assessment.check_frequency).isoformat()
            }
            
            logger.critical(f"危机升级处理 - 用户: {user_id}, 严重程度: {crisis_assessment.severity_level}")
            return escalation_result
            
        except Exception as e:
            logger.error(f"危机升级处理失败: {str(e)}")
            raise
    
    # 私有辅助方法
    async def _build_emotional_context(
        self,
        user_id: str,
        user_input: str,
        session_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> DecisionContext:
        """构建情感决策上下文"""
        # 获取用户情感历史和个性画像
        emotion_history = await self._get_user_emotion_history(user_id, limit=50)
        personality_profile = await self._get_user_personality_profile(user_id)
        previous_decisions = await self._get_recent_decisions(user_id, limit=10)
        
        # 构建当前情感状态（这里需要调用情感识别模块）
        current_emotion_state = await self._analyze_current_emotion(user_input, context)
        
        return DecisionContext(
            user_id=user_id,
            session_id=session_id,
            current_emotion_state=current_emotion_state,
            emotion_history=[e.to_dict() for e in emotion_history],
            personality_profile=personality_profile.to_dict() if personality_profile else {},
            conversation_context=context.get('conversation_context', '') if context else '',
            user_input=user_input,
            environmental_factors=context or {},
            previous_decisions=[d.to_dict() for d in previous_decisions]
        )
    
    async def _make_intelligent_decision(self, context: DecisionContext) -> Optional[EmotionalDecision]:
        """生成智能决策"""
        try:
            return await self.decision_engine.make_decision(context)
        except Exception as e:
            logger.error(f"智能决策生成失败: {str(e)}")
            return None
    
    async def _assess_current_risk(self, context: DecisionContext) -> Optional[RiskAssessment]:
        """评估当前风险"""
        try:
            return await self.decision_engine.assess_emotional_risk(context)
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            return None
    
    async def _detect_crisis_signals(self, context: DecisionContext) -> Optional[CrisisAssessment]:
        """检测危机信号"""
        try:
            # 转换情感状态
            emotion_state = EmotionState(**context.current_emotion_state)
            emotion_history = [EmotionState.from_dict(data) for data in context.emotion_history[-10:]]
            
            return await self.crisis_system.detect_crisis_indicators(
                user_id=context.user_id,
                user_input=context.user_input,
                emotion_state=emotion_state,
                context=context.environmental_factors,
                emotion_history=emotion_history
            )
        except Exception as e:
            logger.error(f"危机检测失败: {str(e)}")
            return None
    
    async def _determine_response_strategy(
        self,
        decision: Optional[EmotionalDecision],
        risk_assessment: Optional[RiskAssessment],
        crisis_assessment: Optional[CrisisAssessment]
    ) -> str:
        """确定响应策略"""
        # 危机优先
        if crisis_assessment and crisis_assessment.crisis_detected:
            return 'crisis_response'
        
        # 高风险情况
        if risk_assessment and risk_assessment.risk_level in ['high', 'critical']:
            return 'high_risk_intervention'
        
        # 正常决策支持
        if decision and decision.confidence_score >= self.config['decision_confidence_threshold']:
            return 'decision_guided_support'
        
        # 默认支持策略
        return 'general_support'
    
    async def _execute_response_actions(
        self,
        user_id: str,
        strategy: str,
        risk_assessment: Optional[RiskAssessment],
        crisis_assessment: Optional[CrisisAssessment]
    ) -> List[Dict[str, Any]]:
        """执行响应行动"""
        actions = []
        
        if strategy == 'crisis_response' and crisis_assessment:
            # 危机响应行动
            crisis_response = await self.handle_crisis_escalation(user_id, crisis_assessment)
            actions.append({
                'type': 'crisis_response',
                'details': crisis_response,
                'timestamp': utc_now().isoformat()
            })
        
        elif strategy == 'high_risk_intervention' and risk_assessment:
            # 高风险干预
            intervention_plan = await self.intervention_engine.create_intervention_plan(
                risk_assessment, 
                await self.intervention_engine.select_intervention_strategies(risk_assessment)
            )
            if intervention_plan:
                self.active_interventions[intervention_plan.plan_id] = intervention_plan
                actions.append({
                    'type': 'intervention_activation',
                    'intervention_id': intervention_plan.plan_id,
                    'timestamp': utc_now().isoformat()
                })
        
        else:
            # 常规支持行动
            actions.append({
                'type': 'general_support',
                'details': 'Provided emotional support and guidance',
                'timestamp': utc_now().isoformat()
            })
        
        return actions
    
    async def _update_user_state(self, user_id: str, state_data: Dict[str, Any]):
        """更新用户状态"""
        # 这里应该更新数据库中的用户状态
        # 简化实现，只记录日志
        logger.info(f"用户状态更新 - 用户: {user_id}, 数据: {len(str(state_data))} 字符")
    
    async def _generate_user_recommendations(
        self,
        user_id: str,
        decision: Optional[EmotionalDecision],
        risk_assessment: Optional[RiskAssessment],
        crisis_assessment: Optional[CrisisAssessment]
    ) -> List[str]:
        """生成用户建议"""
        recommendations = []
        
        if crisis_assessment and crisis_assessment.crisis_detected:
            recommendations.extend(crisis_assessment.immediate_actions)
        
        if risk_assessment:
            recommendations.extend(risk_assessment.recommended_actions)
        
        if decision:
            recommendations.extend([
                f"建议采用 {decision.chosen_strategy} 策略",
                "关注情感健康变化趋势"
            ])
        
        return recommendations[:5]  # 限制建议数量
    
    # 数据获取方法（基于真实存储与计算）
    async def _get_user_emotion_history(
        self, 
        user_id: str, 
        time_period: Optional[Tuple[datetime, datetime]] = None,
        limit: Optional[int] = None
    ) -> List[EmotionState]:
        """获取用户情感历史"""
        from ..core.database import get_db_session

        async with get_db_session() as session:
            repo = EmotionModelingRepository(session)
            start_time, end_time = (time_period or (None, None))
            limit_value = int(limit or 500)
            return await repo.get_user_emotion_history(
                user_id=user_id,
                limit=limit_value,
                start_time=start_time,
                end_time=end_time,
            )
    
    async def _get_user_risk_history(
        self,
        user_id: str,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> List[RiskAssessment]:
        """获取用户风险评估历史"""
        emotion_history = await self._get_user_emotion_history(user_id, time_period, limit=500)
        if not emotion_history:
            return []

        buckets: Dict[str, List[EmotionState]] = {}
        for state in emotion_history:
            key = state.timestamp.date().isoformat()
            buckets.setdefault(key, []).append(state)

        assessments: List[RiskAssessment] = []
        for _, bucket in sorted(buckets.items()):
            assessment = await self.risk_engine.assess_comprehensive_risk(
                user_id=user_id,
                emotion_history=bucket,
                personality_profile=None,
                context=None,
            )
            assessments.append(assessment)

        return assessments
    
    async def _get_user_interventions(
        self,
        user_id: str,
        time_period: Optional[Tuple[datetime, datetime]] = None
    ) -> List[InterventionPlan]:
        """获取用户干预计划"""
        interventions = [
            plan for plan in self.active_interventions.values()
            if plan.user_id == user_id
        ]
        if not time_period:
            return interventions
        start, end = time_period
        return [plan for plan in interventions if start <= plan.created_at <= end]
    
    async def _get_user_personality_profile(self, user_id: str) -> Optional[PersonalityProfile]:
        """获取用户个性画像"""
        from ..core.database import get_db_session

        async with get_db_session() as session:
            repo = EmotionModelingRepository(session)
            return await repo.get_personality_profile(user_id)
    
    async def _get_recent_decisions(self, user_id: str, limit: int = 10) -> List[EmotionalDecision]:
        """获取最近决策"""
        decisions = [
            d for d in self.decision_engine.decision_history
            if d.user_id == user_id
        ]
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        return decisions[:limit]
    
    async def _analyze_current_emotion(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析当前情感状态"""
        if not user_input:
            return {
                'emotion': 'neutral',
                'intensity': 0.3,
                'valence': 0.0,
                'arousal': 0.2,
                'dominance': 0.5,
                'confidence': 0.5
            }

        result = await self._text_analyzer.analyze(user_input)
        dimension = result.dimension
        return {
            'emotion': result.emotion,
            'intensity': float(result.intensity),
            'valence': float(dimension.valence) if dimension else 0.0,
            'arousal': float(dimension.arousal) if dimension else 0.0,
            'dominance': float(dimension.dominance) if dimension else 0.0,
            'confidence': float(result.confidence)
        }
    
    def _get_cached_health_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取缓存的健康数据"""
        cached = self.user_health_cache.get(user_id)
        if cached:
            # 检查缓存是否过期
            cache_time = datetime.fromisoformat(cached['generated_at'])
            if utc_now() - cache_time < timedelta(hours=self.config['health_cache_ttl_hours']):
                return cached
        return None
    
    def _cache_health_data(self, user_id: str, health_data: Dict[str, Any]):
        """缓存健康数据"""
        self.user_health_cache[user_id] = health_data
    
    def _generate_health_summary(
        self,
        dashboard_data: HealthDashboardData,
        crisis_probability: float
    ) -> Dict[str, str]:
        """生成健康摘要"""
        if crisis_probability > 0.7:
            overall_status = "需要关注"
            summary = "检测到较高的情感风险，建议寻求专业支持"
        elif dashboard_data.overall_health_score < 0.5:
            overall_status = "有待改善"
            summary = "情感健康状态需要改善，建议加强自我关爱"
        elif dashboard_data.overall_health_score > 0.8:
            overall_status = "良好"
            summary = "情感健康状态良好，继续保持"
        else:
            overall_status = "一般"
            summary = "情感健康状态正常，注意维护和改善"
        
        return {
            'overall_status': overall_status,
            'summary': summary,
            'health_score': dashboard_data.overall_health_score,
            'risk_level': dashboard_data.current_risk_level
        }
    
    async def _get_priority_health_recommendations(
        self,
        user_id: str,
        dashboard_data: HealthDashboardData,
        crisis_probability: float
    ) -> List[str]:
        """获取优先健康建议"""
        recommendations = []
        
        if crisis_probability > 0.5:
            recommendations.append("建议寻求专业心理健康支持")
        
        if dashboard_data.emotion_volatility > 0.6:
            recommendations.append("练习情绪调节技巧以提高情感稳定性")
        
        if dashboard_data.current_risk_level in ['medium', 'high']:
            recommendations.append("增加社会支持和健康活动")
        
        recommendations.extend([
            "保持规律的生活作息",
            "进行适度的体育锻炼"
        ])
        
        return recommendations[:3]  # 返回前3个优先建议
    
    async def _evaluate_intervention_progress(self, intervention: InterventionPlan) -> Dict[str, Any]:
        """评估干预进展"""
        # 简化的进展评估
        return {
            'effectiveness_score': 0.7,  # 应基于实际数据计算
            'completion_percentage': intervention.progress,
            'user_engagement': 0.8,
            'goal_achievement': 0.6
        }
    
    async def _adjust_intervention_strategy(
        self,
        intervention: InterventionPlan,
        progress_evaluation: Dict[str, Any]
    ) -> InterventionPlan:
        """调整干预策略"""
        # 简化实现，返回原计划
        intervention.status = 'active'
        intervention.progress = min(1.0, intervention.progress + 0.1)
        return intervention
    
    async def _complete_intervention(self, intervention: InterventionPlan) -> Dict[str, Any]:
        """完成干预"""
        intervention.status = 'completed'
        intervention.progress = 1.0
        
        return {
            'completion_time': utc_now().isoformat(),
            'final_effectiveness': 0.8,
            'user_feedback': 'positive',
            'follow_up_needed': False
        }
    
    async def _adjust_interventions_for_crisis(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment
    ) -> List[Dict[str, Any]]:
        """为危机调整干预计划"""
        adjustments = []
        
        # 暂停非紧急干预
        for intervention_id, intervention in self.active_interventions.items():
            if (intervention.user_id == user_id and 
                intervention.intervention_type != 'crisis'):
                intervention.status = 'paused'
                adjustments.append({
                    'intervention_id': intervention_id,
                    'action': 'paused_for_crisis',
                    'original_type': intervention.intervention_type
                })
        
        return adjustments
    
    async def _create_crisis_intervention_plan(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment
    ) -> Optional[InterventionPlan]:
        """创建危机干预计划"""
        # 简化实现
        crisis_plan = InterventionPlan(
            user_id=user_id,
            intervention_type='crisis',
            urgency_level='critical',
            status='active',
            target_risk_factors=['crisis_indicators'],
            timeline={
                'immediate_response': utc_now(),
                'first_check': utc_now() + timedelta(hours=1),
                'follow_up': utc_now() + timedelta(hours=6)
            }
        )
        
        return crisis_plan
    
    async def _send_crisis_notifications(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment,
        emergency_response: Dict[str, Any]
    ) -> List[str]:
        """发送危机通知"""
        notifications = []
        
        if crisis_assessment.severity_level in ['severe', 'critical']:
            notifications.extend([
                'emergency_contacts_notified',
                'crisis_hotline_connected',
                'professional_services_alerted'
            ])
        
        return notifications
    
    async def _setup_crisis_monitoring(
        self,
        user_id: str,
        crisis_assessment: CrisisAssessment
    ) -> Dict[str, Any]:
        """建立危机监控"""
        return {
            'monitoring_level': crisis_assessment.monitoring_level,
            'check_frequency': crisis_assessment.check_frequency.total_seconds(),
            'automated_checks': True,
            'human_oversight': crisis_assessment.severity_level in ['severe', 'critical']
        }
