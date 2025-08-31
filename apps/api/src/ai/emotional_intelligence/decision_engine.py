"""
情感智能决策引擎核心实现
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import numpy as np

from .models import (
    DecisionContext, EmotionalDecision, RiskAssessment, InterventionPlan,
    CrisisAssessment, HealthDashboardData, RiskLevel, DecisionType,
    InterventionStrategy, RiskFactor, SeverityLevel
)
from ..emotion_modeling.models import EmotionState, PersonalityProfile


logger = logging.getLogger(__name__)


class DecisionStrategy(ABC):
    """决策策略抽象基类"""
    
    @abstractmethod
    async def evaluate(self, context: DecisionContext) -> float:
        """评估策略适用性"""
        pass
    
    @abstractmethod
    async def execute(self, context: DecisionContext) -> Dict[str, Any]:
        """执行策略"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """策略名称"""
        pass


class EmotionalDecisionEngine:
    """情感智能决策引擎主类"""
    
    def __init__(self):
        self.strategies: Dict[str, DecisionStrategy] = {}
        self.risk_thresholds = {
            RiskLevel.LOW.value: 0.3,
            RiskLevel.MEDIUM.value: 0.5,
            RiskLevel.HIGH.value: 0.7,
            RiskLevel.CRITICAL.value: 0.9
        }
        self.decision_history: List[EmotionalDecision] = []
        
    def register_strategy(self, strategy: DecisionStrategy):
        """注册决策策略"""
        self.strategies[strategy.strategy_name] = strategy
        logger.info(f"注册策略: {strategy.strategy_name}")
    
    async def make_decision(self, context: DecisionContext) -> EmotionalDecision:
        """
        基于情感上下文做出智能决策
        
        Args:
            context: 决策上下文
            
        Returns:
            情感智能决策结果
        """
        try:
            # 1. 分析当前情感状态和风险
            risk_assessment = await self.assess_emotional_risk(context)
            
            # 2. 选择最适合的策略
            selected_strategy, confidence = await self._select_best_strategy(context, risk_assessment)
            
            # 3. 生成决策推理
            reasoning = await self._generate_reasoning(context, risk_assessment, selected_strategy)
            
            # 4. 预测决策效果
            expected_outcome = await self._predict_outcome(context, selected_strategy)
            
            # 5. 构建决策结果
            decision = EmotionalDecision(
                user_id=context.user_id,
                session_id=context.session_id,
                decision_type=self._determine_decision_type(risk_assessment),
                chosen_strategy=selected_strategy,
                confidence_score=confidence,
                reasoning=reasoning,
                evidence={
                    'risk_assessment': risk_assessment.to_dict(),
                    'emotion_state': context.current_emotion_state,
                    'context_factors': context.environmental_factors
                },
                expected_outcome=expected_outcome,
                success_metrics=self._define_success_metrics(selected_strategy)
            )
            
            # 6. 记录决策历史
            self.decision_history.append(decision)
            
            logger.info(f"生成决策 {decision.decision_id}: {selected_strategy} (置信度: {confidence:.2f})")
            return decision
            
        except Exception as e:
            logger.error(f"决策生成失败: {str(e)}")
            raise
    
    async def assess_emotional_risk(self, context: DecisionContext) -> RiskAssessment:
        """
        评估情感健康风险
        
        Args:
            context: 决策上下文
            
        Returns:
            风险评估结果
        """
        try:
            risk_factors = []
            
            # 分析情感状态风险
            emotion_risk = await self._analyze_emotion_risk(context)
            if emotion_risk > 0.1:
                risk_factors.append(RiskFactor(
                    factor_type="emotional_state",
                    score=emotion_risk,
                    evidence=context.current_emotion_state,
                    weight=0.4,
                    description="当前情感状态风险评估"
                ))
            
            # 分析历史模式风险  
            pattern_risk = await self._analyze_pattern_risk(context)
            if pattern_risk > 0.1:
                risk_factors.append(RiskFactor(
                    factor_type="emotional_pattern",
                    score=pattern_risk,
                    evidence={'pattern_analysis': 'historical_trends'},
                    weight=0.3,
                    description="情感模式趋势风险"
                ))
            
            # 分析环境因素风险
            environmental_risk = await self._analyze_environmental_risk(context)
            if environmental_risk > 0.1:
                risk_factors.append(RiskFactor(
                    factor_type="environmental",
                    score=environmental_risk,
                    evidence=context.environmental_factors,
                    weight=0.2,
                    description="环境因素风险"
                ))
            
            # 计算综合风险分数
            total_risk = sum(factor.score * factor.weight for factor in risk_factors)
            total_risk = min(1.0, total_risk)  # 确保不超过1.0
            
            # 确定风险等级
            risk_level = self._determine_risk_level(total_risk)
            
            # 生成建议行动
            recommended_actions = await self._generate_risk_actions(risk_level, risk_factors)
            
            assessment = RiskAssessment(
                user_id=context.user_id,
                risk_level=risk_level,
                risk_score=total_risk,
                risk_factors=risk_factors,
                prediction_confidence=self._calculate_prediction_confidence(risk_factors),
                recommended_actions=recommended_actions,
                alert_triggered=risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value],
                assessment_details={
                    'analysis_method': 'multi_factor_weighted',
                    'factors_count': len(risk_factors),
                    'data_completeness': len(context.emotion_history) / 100 if context.emotion_history else 0.1
                }
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            raise
    
    async def detect_crisis(self, context: DecisionContext) -> CrisisAssessment:
        """
        检测情感危机
        
        Args:
            context: 决策上下文
            
        Returns:
            危机评估结果
        """
        try:
            # 危机关键词检测
            crisis_keywords = [
                '想死', '自杀', '结束生命', '活着没意义', '不想活了',
                '伤害自己', '自残', '割腕', '跳楼', '服药'
            ]
            
            indicators = []
            
            # 检测语言指标
            keyword_score = self._analyze_crisis_keywords(context.user_input, crisis_keywords)
            if keyword_score >= 0.1:  # 使用 >= 而不是 >
                indicators.append({
                    'type': 'language_indicator',
                    'score': keyword_score,
                    'evidence': {'crisis_phrases_detected': True},
                    'description': '检测到危机相关语言表达'
                })
            
            # 检测情感状态指标
            emotion_crisis_score = self._analyze_emotion_crisis(context.current_emotion_state)
            if emotion_crisis_score > 0.7:
                indicators.append({
                    'type': 'emotional_indicator',
                    'score': emotion_crisis_score,
                    'evidence': context.current_emotion_state,
                    'description': '检测到极端负面情感状态'
                })
            
            # 检测行为模式指标
            behavioral_score = await self._analyze_behavioral_crisis(context)
            if behavioral_score > 0.6:
                indicators.append({
                    'type': 'behavioral_indicator',
                    'score': behavioral_score,
                    'evidence': {'behavioral_changes': True},
                    'description': '检测到异常行为模式变化'
                })
            
            # 计算综合危机分数
            crisis_score = max([ind['score'] for ind in indicators] + [0.0])
            crisis_detected = crisis_score > 0.6
            severity_level = self._determine_crisis_severity(crisis_score)
            
            # 生成即时行动建议
            immediate_actions = await self._generate_crisis_actions(severity_level, indicators)
            
            assessment = CrisisAssessment(
                user_id=context.user_id,
                crisis_detected=crisis_detected,
                severity_level=severity_level,
                indicators=indicators,
                risk_score=crisis_score,
                confidence=self._calculate_crisis_confidence(indicators),
                immediate_actions=immediate_actions,
                professional_required=crisis_score > 0.8,
                monitoring_level="intensive" if crisis_score > 0.7 else "elevated"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"危机检测失败: {str(e)}")
            raise
    
    async def create_intervention_plan(self, risk_assessment: RiskAssessment) -> InterventionPlan:
        """
        创建干预计划
        
        Args:
            risk_assessment: 风险评估结果
            
        Returns:
            干预计划
        """
        try:
            # 基于风险等级选择干预策略
            strategies = await self._select_intervention_strategies(risk_assessment)
            
            # 确定主要策略
            primary_strategy = strategies[0] if strategies else None
            
            # 制定时间线
            timeline = self._create_intervention_timeline(risk_assessment.risk_level, strategies)
            
            # 定义监控频率
            monitoring_frequency = self._determine_monitoring_frequency(risk_assessment.risk_level)
            
            plan = InterventionPlan(
                user_id=risk_assessment.user_id,
                intervention_type=self._determine_intervention_type(risk_assessment.risk_level),
                urgency_level=self._determine_urgency_level(risk_assessment),
                target_risk_factors=[factor.factor_type for factor in risk_assessment.risk_factors],
                strategies=strategies,
                primary_strategy=primary_strategy,
                timeline=timeline,
                success_metrics=self._define_intervention_metrics(strategies),
                monitoring_frequency=monitoring_frequency,
                status="draft"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"干预计划创建失败: {str(e)}")
            raise
    
    # 私有辅助方法
    async def _select_best_strategy(self, context: DecisionContext, risk_assessment: RiskAssessment) -> Tuple[str, float]:
        """选择最佳策略"""
        if not self.strategies:
            return "default_supportive", 0.5
        
        strategy_scores = {}
        for name, strategy in self.strategies.items():
            score = await strategy.evaluate(context)
            strategy_scores[name] = score
        
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]
    
    async def _generate_reasoning(self, context: DecisionContext, risk_assessment: RiskAssessment, strategy: str) -> List[str]:
        """生成决策推理"""
        reasoning = []
        
        # 基于风险评估的推理
        reasoning.append(f"风险评估等级: {risk_assessment.risk_level}")
        reasoning.append(f"风险分数: {risk_assessment.risk_score:.2f}")
        
        # 基于情感状态的推理
        if context.current_emotion_state:
            emotion = context.current_emotion_state.get('emotion', 'unknown')
            intensity = context.current_emotion_state.get('intensity', 0.0)
            reasoning.append(f"当前情感状态: {emotion} (强度: {intensity:.2f})")
        
        # 基于策略选择的推理
        reasoning.append(f"选择策略: {strategy}")
        
        return reasoning
    
    async def _predict_outcome(self, context: DecisionContext, strategy: str) -> Dict[str, float]:
        """预测决策效果"""
        # 简化的效果预测模型
        return {
            'emotional_improvement': 0.7,
            'risk_reduction': 0.6,
            'user_satisfaction': 0.8,
            'engagement_increase': 0.5
        }
    
    def _determine_decision_type(self, risk_assessment: RiskAssessment) -> str:
        """确定决策类型"""
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            return DecisionType.CRISIS_RESPONSE.value
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            return DecisionType.INTERVENTION_PLANNING.value
        else:
            return DecisionType.INTERACTION_STRATEGY.value
    
    def _define_success_metrics(self, strategy: str) -> List[str]:
        """定义成功指标"""
        return [
            "emotional_state_improvement",
            "risk_level_reduction", 
            "user_engagement_increase",
            "intervention_compliance"
        ]
    
    async def _analyze_emotion_risk(self, context: DecisionContext) -> float:
        """分析情感状态风险"""
        if not context.current_emotion_state:
            return 0.0
        
        emotion = context.current_emotion_state.get('emotion', 'neutral')
        intensity = context.current_emotion_state.get('intensity', 0.5)
        valence = context.current_emotion_state.get('valence', 0.0)
        
        # 负面情感风险评估
        negative_emotions = ['sadness', 'depression', 'anxiety', 'anger', 'fear', 'despair']
        if emotion in negative_emotions:
            risk_score = intensity * (1.0 - valence) / 2.0  # 结合强度和效价
        else:
            risk_score = max(0.0, (0.5 - intensity) * (1.0 - valence) / 2.0)
        
        return min(1.0, risk_score)
    
    async def _analyze_pattern_risk(self, context: DecisionContext) -> float:
        """分析情感模式风险"""
        if not context.emotion_history or len(context.emotion_history) < 3:
            return 0.0
        
        # 分析最近情感趋势
        recent_emotions = context.emotion_history[-10:] if len(context.emotion_history) >= 10 else context.emotion_history
        
        negative_count = sum(1 for emotion in recent_emotions 
                           if emotion.get('valence', 0.0) < -0.3)
        
        pattern_risk = negative_count / len(recent_emotions)
        return min(1.0, pattern_risk)
    
    async def _analyze_environmental_risk(self, context: DecisionContext) -> float:
        """分析环境因素风险"""
        if not context.environmental_factors:
            return 0.0
        
        # 简化的环境风险评估
        risk_factors = context.environmental_factors.get('risk_factors', [])
        stress_level = context.environmental_factors.get('stress_level', 0.0)
        
        environmental_risk = (len(risk_factors) * 0.1 + stress_level) / 2.0
        return min(1.0, environmental_risk)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """确定风险等级"""
        for level, threshold in sorted(self.risk_thresholds.items(), key=lambda x: x[1], reverse=True):
            if risk_score >= threshold:
                return level
        return RiskLevel.LOW.value
    
    def _calculate_prediction_confidence(self, risk_factors: List[RiskFactor]) -> float:
        """计算预测置信度"""
        if not risk_factors:
            return 0.1
        
        # 基于风险因子数量和质量计算置信度
        factor_count = len(risk_factors)
        avg_score = sum(factor.score for factor in risk_factors) / factor_count
        
        confidence = min(1.0, (factor_count / 5.0) * avg_score)
        return confidence
    
    async def _generate_risk_actions(self, risk_level: str, risk_factors: List[RiskFactor]) -> List[str]:
        """生成风险应对行动"""
        actions = []
        
        if risk_level == RiskLevel.CRITICAL.value:
            actions.extend([
                "立即启动危机干预协议",
                "联系紧急心理健康服务",
                "激活24/7监护模式",
                "通知紧急联系人"
            ])
        elif risk_level == RiskLevel.HIGH.value:
            actions.extend([
                "增加心理支持强度",
                "安排专业咨询评估",
                "提高监测频率",
                "启动积极干预措施"
            ])
        elif risk_level == RiskLevel.MEDIUM.value:
            actions.extend([
                "提供针对性支持建议",
                "推荐压力管理技巧",
                "安排定期情感检查",
                "关注情感变化趋势"
            ])
        else:
            actions.extend([
                "保持正向情感支持",
                "提供情感健康教育",
                "记录情感状态变化"
            ])
        
        return actions
    
    def _analyze_crisis_keywords(self, user_input: str, crisis_keywords: List[str]) -> float:
        """分析危机关键词"""
        if not user_input:
            return 0.0
        
        user_input_lower = user_input.lower()
        matched_keywords = [kw for kw in crisis_keywords if kw in user_input_lower]
        
        return min(1.0, len(matched_keywords) / len(crisis_keywords))
    
    def _analyze_emotion_crisis(self, emotion_state: Dict[str, Any]) -> float:
        """分析情感危机指标"""
        if not emotion_state:
            return 0.0
        
        emotion = emotion_state.get('emotion', 'neutral')
        intensity = emotion_state.get('intensity', 0.5)
        valence = emotion_state.get('valence', 0.0)
        
        # 极端负面情感检测
        crisis_emotions = ['despair', 'hopelessness', 'severe_depression']
        if emotion in crisis_emotions and intensity > 0.8:
            return min(1.0, intensity * (1.0 - valence))
        
        return 0.0
    
    async def _analyze_behavioral_crisis(self, context: DecisionContext) -> float:
        """分析行为危机指标"""
        # 简化的行为危机分析
        behavioral_factors = context.environmental_factors.get('behavioral_changes', {})
        
        if not behavioral_factors:
            return 0.0
        
        crisis_behaviors = behavioral_factors.get('crisis_indicators', [])
        return min(1.0, len(crisis_behaviors) * 0.3)
    
    def _determine_crisis_severity(self, crisis_score: float) -> str:
        """确定危机严重程度"""
        if crisis_score >= 0.9:
            return SeverityLevel.CRITICAL.value
        elif crisis_score >= 0.7:
            return SeverityLevel.SEVERE.value
        elif crisis_score >= 0.5:
            return SeverityLevel.MODERATE.value
        else:
            return SeverityLevel.MILD.value
    
    def _calculate_crisis_confidence(self, indicators: List[Dict[str, Any]]) -> float:
        """计算危机评估置信度"""
        if not indicators:
            return 0.0
        
        avg_score = sum(ind['score'] for ind in indicators) / len(indicators)
        return min(1.0, avg_score * len(indicators) / 3.0)
    
    async def _generate_crisis_actions(self, severity_level: str, indicators: List[Dict[str, Any]]) -> List[str]:
        """生成危机应对行动"""
        actions = []
        
        if severity_level == SeverityLevel.CRITICAL.value:
            actions.extend([
                "立即联系危机干预热线",
                "启动紧急安全协议", 
                "通知紧急联系人和专业机构",
                "进入24/7监护模式"
            ])
        elif severity_level == SeverityLevel.SEVERE.value:
            actions.extend([
                "提供即时情感支持",
                "联系心理健康专业服务",
                "增加监测和检查频率",
                "准备专业转介"
            ])
        else:
            actions.extend([
                "提供安抚和支持",
                "推荐专业咨询资源",
                "持续监测情感状态"
            ])
        
        return actions
    
    async def _select_intervention_strategies(self, risk_assessment: RiskAssessment) -> List[InterventionStrategy]:
        """选择干预策略"""
        strategies = []
        
        # 基于风险等级选择策略
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            strategies.append(InterventionStrategy(
                strategy_name="crisis_intervention",
                description="危机干预策略",
                implementation_steps=["immediate_safety_assessment", "professional_referral", "24_7_monitoring"],
                expected_effectiveness=0.9
            ))
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            strategies.append(InterventionStrategy(
                strategy_name="intensive_support",
                description="强化支持策略", 
                implementation_steps=["frequent_check_ins", "coping_skills_training", "professional_consultation"],
                expected_effectiveness=0.8
            ))
        else:
            strategies.append(InterventionStrategy(
                strategy_name="supportive_guidance",
                description="支持性指导策略",
                implementation_steps=["emotional_support", "resource_recommendation", "progress_monitoring"],
                expected_effectiveness=0.7
            ))
        
        return strategies
    
    def _determine_intervention_type(self, risk_level: str) -> str:
        """确定干预类型"""
        if risk_level == RiskLevel.CRITICAL.value:
            return "crisis"
        elif risk_level == RiskLevel.HIGH.value:
            return "corrective"
        else:
            return "supportive"
    
    def _determine_urgency_level(self, risk_assessment: RiskAssessment) -> str:
        """确定紧急程度"""
        if risk_assessment.risk_level == RiskLevel.CRITICAL.value:
            return "critical"
        elif risk_assessment.risk_level == RiskLevel.HIGH.value:
            return "high"
        elif risk_assessment.risk_level == RiskLevel.MEDIUM.value:
            return "medium"
        else:
            return "low"
    
    def _create_intervention_timeline(self, risk_level: str, strategies: List[InterventionStrategy]) -> Dict[str, datetime]:
        """创建干预时间线"""
        now = datetime.now()
        timeline = {}
        
        if risk_level == RiskLevel.CRITICAL.value:
            timeline = {
                "immediate_start": now,
                "first_assessment": now + timedelta(hours=1),
                "professional_contact": now + timedelta(hours=2),
                "follow_up": now + timedelta(hours=6)
            }
        elif risk_level == RiskLevel.HIGH.value:
            timeline = {
                "start": now + timedelta(hours=1),
                "first_check": now + timedelta(hours=6),
                "professional_review": now + timedelta(days=1),
                "progress_review": now + timedelta(days=3)
            }
        else:
            timeline = {
                "start": now + timedelta(hours=2),
                "first_check": now + timedelta(days=1),
                "progress_review": now + timedelta(days=7)
            }
        
        return timeline
    
    def _determine_monitoring_frequency(self, risk_level: str) -> timedelta:
        """确定监控频率"""
        frequency_map = {
            RiskLevel.CRITICAL.value: timedelta(minutes=30),
            RiskLevel.HIGH.value: timedelta(hours=2),
            RiskLevel.MEDIUM.value: timedelta(hours=6),
            RiskLevel.LOW.value: timedelta(hours=24)
        }
        return frequency_map.get(risk_level, timedelta(hours=12))
    
    def _define_intervention_metrics(self, strategies: List[InterventionStrategy]) -> List[str]:
        """定义干预成功指标"""
        return [
            "risk_score_reduction",
            "emotional_stability_improvement", 
            "user_engagement_level",
            "intervention_compliance_rate",
            "professional_service_utilization"
        ]