"""
社交智能决策引擎 - Story 11.6 Task 5
综合群体情感、关系动态、社交场景和文化背景，做出智能社交决策
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime, timedelta
import json
from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface
from .group_emotion_analyzer import GroupEmotionAnalyzer
from .relationship_analyzer import RelationshipDynamicsAnalyzer
from .social_context_adapter import SocialContextAdapter, SocialEnvironment
from .cultural_context_analyzer import CulturalContextAnalyzer

from src.core.logging import get_logger
logger = get_logger(__name__)

class DecisionType(Enum):
    """决策类型"""
    COMMUNICATION_STYLE = "communication_style"
    CONFLICT_INTERVENTION = "conflict_intervention"
    GROUP_FACILITATION = "group_facilitation"
    RELATIONSHIP_BUILDING = "relationship_building"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    EMOTIONAL_REGULATION = "emotional_regulation"
    LEADERSHIP_GUIDANCE = "leadership_guidance"
    TEAM_OPTIMIZATION = "team_optimization"

class DecisionPriority(Enum):
    """决策优先级"""
    CRITICAL = "critical"  # 立即执行
    HIGH = "high"  # 短期内执行
    MEDIUM = "medium"  # 适当时机执行
    LOW = "low"  # 可延迟执行

@dataclass
class DecisionContext:
    """决策上下文"""
    session_id: str
    timestamp: datetime
    participants: List[Dict[str, Any]]
    current_emotions: Dict[str, EmotionVector]
    group_dynamics: Dict[str, Any]
    social_environment: SocialEnvironment
    cultural_profiles: List[Any]  # CulturalProfile
    historical_context: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SocialDecision:
    """社交决策"""
    decision_id: str
    decision_type: DecisionType
    priority: DecisionPriority
    context: DecisionContext
    recommended_actions: List[str]
    reasoning: str
    confidence_score: float
    expected_outcomes: Dict[str, float]  # 预期结果概率
    alternative_options: List[str]
    execution_timeline: str
    monitoring_metrics: List[str]
    risk_assessment: Dict[str, float]

@dataclass
class DecisionOutcome:
    """决策结果"""
    decision_id: str
    execution_success: bool
    actual_outcomes: Dict[str, Any]
    participant_feedback: Dict[str, float]
    effectiveness_score: float
    lessons_learned: List[str]
    timestamp: datetime

class SocialIntelligenceEngine:
    """社交智能决策引擎"""
    
    def __init__(self):
        # 初始化子系统
        self.group_analyzer = GroupEmotionAnalyzer()
        self.relationship_analyzer = RelationshipDynamicsAnalyzer()
        self.context_adapter = SocialContextAdapter()
        self.cultural_analyzer = CulturalContextAnalyzer()
        
        # 决策历史
        self.decision_history: Dict[str, SocialDecision] = {}
        self.outcome_history: Dict[str, DecisionOutcome] = {}
        
        # 学习系统
        self.decision_patterns: Dict[str, Any] = {}
        self.success_metrics: Dict[DecisionType, float] = {}
        
        # 配置
        self.learning_enabled = True
        self.max_history_size = 1000
        
    async def analyze_and_decide(
        self,
        context: DecisionContext,
        decision_types: Optional[List[DecisionType]] = None
    ) -> List[SocialDecision]:
        """分析情况并生成决策建议"""
        try:
            logger.info(f"Starting decision analysis for session {context.session_id}")
            
            # 如果没有指定决策类型，自动识别
            if decision_types is None:
                decision_types = await self._identify_required_decisions(context)
            
            decisions = []
            
            for decision_type in decision_types:
                decision = await self._generate_decision(context, decision_type)
                if decision:
                    decisions.append(decision)
            
            # 按优先级排序
            decisions.sort(key=lambda x: self._priority_score(x.priority), reverse=True)
            
            # 存储决策历史
            for decision in decisions:
                self.decision_history[decision.decision_id] = decision
            
            # 清理历史记录
            await self._cleanup_history()
            
            logger.info(f"Generated {len(decisions)} decisions for session {context.session_id}")
            return decisions
            
        except Exception as e:
            logger.error(f"Decision analysis failed: {e}")
            return []
    
    async def _identify_required_decisions(self, context: DecisionContext) -> List[DecisionType]:
        """自动识别需要的决策类型"""
        required_decisions = []
        
        # 分析群体情感状态
        group_emotions = context.current_emotions
        if group_emotions:
            emotion_intensity = sum(
                max(emotion.emotions.values()) * emotion.intensity 
                for emotion in group_emotions.values()
            ) / len(group_emotions)
            
            if emotion_intensity > 0.8:
                required_decisions.append(DecisionType.EMOTIONAL_REGULATION)
            
            # 检查冲突情感
            conflict_emotions = ["anger", "frustration", "tension", "disagreement"]
            has_conflict = any(
                any(emotion in emotion_vec.emotions and emotion_vec.emotions[emotion] > 0.6 
                    for emotion in conflict_emotions)
                for emotion_vec in group_emotions.values()
            )
            if has_conflict:
                required_decisions.append(DecisionType.CONFLICT_INTERVENTION)
        
        # 分析社交环境
        if context.social_environment:
            if context.social_environment.participants_count > 5:
                required_decisions.append(DecisionType.GROUP_FACILITATION)
            
            if context.social_environment.formality_level > 0.7:
                required_decisions.append(DecisionType.COMMUNICATION_STYLE)
        
        # 分析文化多样性
        if len(context.cultural_profiles) > 1:
            required_decisions.append(DecisionType.CULTURAL_ADAPTATION)
        
        # 分析关系动态
        if context.group_dynamics.get("relationship_tensions", 0) > 0.6:
            required_decisions.append(DecisionType.RELATIONSHIP_BUILDING)
        
        # 默认决策类型
        if not required_decisions:
            required_decisions = [DecisionType.COMMUNICATION_STYLE, DecisionType.GROUP_FACILITATION]
        
        return required_decisions
    
    async def _generate_decision(
        self, 
        context: DecisionContext, 
        decision_type: DecisionType
    ) -> Optional[SocialDecision]:
        """生成具体决策"""
        try:
            decision_id = f"{context.session_id}_{decision_type.value}_{int(utc_now().timestamp())}"
            
            if decision_type == DecisionType.COMMUNICATION_STYLE:
                return await self._decide_communication_style(decision_id, context)
            elif decision_type == DecisionType.CONFLICT_INTERVENTION:
                return await self._decide_conflict_intervention(decision_id, context)
            elif decision_type == DecisionType.GROUP_FACILITATION:
                return await self._decide_group_facilitation(decision_id, context)
            elif decision_type == DecisionType.RELATIONSHIP_BUILDING:
                return await self._decide_relationship_building(decision_id, context)
            elif decision_type == DecisionType.CULTURAL_ADAPTATION:
                return await self._decide_cultural_adaptation(decision_id, context)
            elif decision_type == DecisionType.EMOTIONAL_REGULATION:
                return await self._decide_emotional_regulation(decision_id, context)
            elif decision_type == DecisionType.LEADERSHIP_GUIDANCE:
                return await self._decide_leadership_guidance(decision_id, context)
            elif decision_type == DecisionType.TEAM_OPTIMIZATION:
                return await self._decide_team_optimization(decision_id, context)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate decision for {decision_type}: {e}")
            return None
    
    async def _decide_communication_style(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定沟通风格"""
        # 分析当前沟通需求
        formality = context.social_environment.formality_level
        participants_count = context.social_environment.participants_count
        cultural_diversity = len(set(profile.culture_id for profile in context.cultural_profiles))
        
        actions = []
        reasoning_parts = []
        
        if formality > 0.7:
            actions.extend([
                "adopt_formal_communication_style",
                "use_professional_language",
                "maintain_structured_dialogue"
            ])
            reasoning_parts.append("High formality level requires professional approach")
        elif formality < 0.3:
            actions.extend([
                "use_casual_friendly_tone",
                "encourage_open_discussion",
                "create_relaxed_atmosphere"
            ])
            reasoning_parts.append("Low formality allows for casual interaction")
        
        if participants_count > 10:
            actions.extend([
                "use_clear_projection_techniques",
                "implement_turn_taking_system",
                "provide_structured_participation"
            ])
            reasoning_parts.append("Large group requires structured communication")
        
        if cultural_diversity > 1:
            actions.extend([
                "use_inclusive_language",
                "avoid_culture_specific_references",
                "check_for_understanding_regularly"
            ])
            reasoning_parts.append("Cultural diversity requires inclusive communication")
        
        confidence = min(0.7 + (formality * 0.2), 0.95)
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.COMMUNICATION_STYLE,
            priority=DecisionPriority.HIGH,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=confidence,
            expected_outcomes={
                "improved_clarity": 0.8,
                "better_engagement": 0.7,
                "reduced_misunderstanding": 0.75
            },
            alternative_options=[
                "adaptive_style_switching",
                "participant_preference_polling",
                "real_time_style_adjustment"
            ],
            execution_timeline="immediate",
            monitoring_metrics=["communication_clarity", "participant_engagement", "misunderstanding_rate"],
            risk_assessment={
                "style_mismatch": 0.2,
                "participant_discomfort": 0.15,
                "reduced_effectiveness": 0.1
            }
        )
    
    async def _decide_conflict_intervention(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定冲突干预策略"""
        # 分析冲突强度和类型
        conflict_emotions = ["anger", "frustration", "tension", "disagreement"]
        conflict_intensity = 0.0
        affected_participants = []
        
        for participant_id, emotion_vec in context.current_emotions.items():
            participant_conflict = max(
                emotion_vec.emotions.get(emotion, 0) for emotion in conflict_emotions
            )
            if participant_conflict > 0.4:
                conflict_intensity = max(conflict_intensity, participant_conflict)
                affected_participants.append(participant_id)
        
        actions = []
        reasoning_parts = []
        priority = DecisionPriority.MEDIUM
        
        if conflict_intensity > 0.8:
            # 高强度冲突
            actions.extend([
                "immediate_de_escalation",
                "separate_conflicting_parties",
                "neutral_mediation",
                "focus_on_common_goals"
            ])
            reasoning_parts.append("High conflict intensity requires immediate intervention")
            priority = DecisionPriority.CRITICAL
        elif conflict_intensity > 0.5:
            # 中等强度冲突
            actions.extend([
                "gentle_redirection",
                "acknowledge_different_perspectives",
                "find_common_ground",
                "structured_discussion"
            ])
            reasoning_parts.append("Moderate conflict requires structured mediation")
            priority = DecisionPriority.HIGH
        else:
            # 低强度紧张
            actions.extend([
                "monitor_situation",
                "preventive_communication",
                "team_building_activities",
                "positive_refocusing"
            ])
            reasoning_parts.append("Low-level tension requires preventive measures")
        
        # 考虑文化因素
        if context.cultural_profiles:
            for profile in context.cultural_profiles:
                if hasattr(profile, 'conflict_resolution_style'):
                    if profile.conflict_resolution_style == "indirect_mediation":
                        actions.append("use_indirect_mediation_approach")
                    elif profile.conflict_resolution_style == "direct_discussion":
                        actions.append("encourage_direct_dialogue")
        
        confidence = 0.6 + (len(affected_participants) * 0.1)
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.CONFLICT_INTERVENTION,
            priority=priority,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=min(confidence, 0.9),
            expected_outcomes={
                "conflict_resolution": 0.7,
                "relationship_preservation": 0.65,
                "group_cohesion": 0.6,
                "productivity_recovery": 0.55
            },
            alternative_options=[
                "external_mediator_involvement",
                "cooling_off_period",
                "individual_conversations",
                "conflict_coaching"
            ],
            execution_timeline="immediate" if priority == DecisionPriority.CRITICAL else "within_15_minutes",
            monitoring_metrics=[
                "conflict_intensity_level",
                "participant_satisfaction",
                "group_harmony_score",
                "resolution_effectiveness"
            ],
            risk_assessment={
                "escalation_risk": conflict_intensity * 0.5,
                "relationship_damage": conflict_intensity * 0.3,
                "group_fragmentation": 0.2
            }
        )
    
    async def _decide_group_facilitation(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定群体引导策略"""
        group_size = context.social_environment.participants_count
        scenario = context.social_environment.scenario
        
        actions = []
        reasoning_parts = []
        
        # 基于群体规模的策略
        if group_size <= 3:
            actions.extend([
                "encourage_intimate_discussion",
                "allow_natural_flow",
                "facilitate_deep_exploration"
            ])
            reasoning_parts.append("Small group allows for intimate discussion style")
        elif group_size <= 8:
            actions.extend([
                "round_robin_participation",
                "structured_brainstorming",
                "subgroup_activities"
            ])
            reasoning_parts.append("Medium group benefits from structured participation")
        else:
            actions.extend([
                "formal_facilitation_techniques",
                "time_management",
                "large_group_dynamics_management",
                "breakout_sessions"
            ])
            reasoning_parts.append("Large group requires formal facilitation")
        
        # 基于社交场景的策略
        scenario_strategies = {
            "team_brainstorming": [
                "encourage_wild_ideas",
                "build_on_others_ideas",
                "defer_judgment",
                "stay_focused_on_topic"
            ],
            "formal_meeting": [
                "follow_agenda_strictly",
                "manage_time_carefully",
                "ensure_equal_participation",
                "document_decisions"
            ],
            "conflict_resolution": [
                "maintain_neutrality",
                "ensure_all_voices_heard",
                "focus_on_interests_not_positions",
                "seek_win_win_solutions"
            ]
        }
        
        if hasattr(scenario, 'value') and scenario.value in scenario_strategies:
            actions.extend(scenario_strategies[scenario.value])
            reasoning_parts.append(f"Scenario-specific strategies for {scenario.value}")
        
        confidence = 0.75 if group_size <= 8 else 0.65
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.GROUP_FACILITATION,
            priority=DecisionPriority.HIGH,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=confidence,
            expected_outcomes={
                "improved_participation": 0.8,
                "better_focus": 0.75,
                "increased_satisfaction": 0.7,
                "goal_achievement": 0.65
            },
            alternative_options=[
                "co_facilitation_approach",
                "participant_led_facilitation",
                "technology_assisted_facilitation"
            ],
            execution_timeline="immediate",
            monitoring_metrics=[
                "participation_balance",
                "discussion_quality",
                "goal_progress",
                "participant_engagement"
            ],
            risk_assessment={
                "domination_by_few": 0.25,
                "off_topic_drift": 0.2,
                "time_overrun": 0.15
            }
        )
    
    async def _decide_relationship_building(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定关系建设策略"""
        # 分析关系紧张度
        relationship_tensions = context.group_dynamics.get("relationship_tensions", 0)
        trust_levels = context.group_dynamics.get("trust_levels", {})
        
        actions = []
        reasoning_parts = []
        
        if relationship_tensions > 0.7:
            actions.extend([
                "address_relationship_issues_directly",
                "facilitate_one_on_one_conversations",
                "create_trust_building_exercises",
                "establish_ground_rules"
            ])
            reasoning_parts.append("High relationship tension requires direct intervention")
            priority = DecisionPriority.HIGH
        elif relationship_tensions > 0.4:
            actions.extend([
                "promote_positive_interactions",
                "highlight_common_interests",
                "encourage_collaboration",
                "create_shared_experiences"
            ])
            reasoning_parts.append("Moderate tension requires relationship strengthening")
            priority = DecisionPriority.MEDIUM
        else:
            actions.extend([
                "maintain_positive_momentum",
                "recognize_good_collaboration",
                "plan_team_building_activities",
                "celebrate_successes_together"
            ])
            reasoning_parts.append("Good relationships require maintenance and strengthening")
            priority = DecisionPriority.LOW
        
        # 分析信任水平
        if trust_levels:
            avg_trust = sum(trust_levels.values()) / len(trust_levels)
            if avg_trust < 0.5:
                actions.extend([
                    "build_credibility_through_consistency",
                    "increase_transparency",
                    "follow_through_on_commitments"
                ])
                reasoning_parts.append("Low trust levels require credibility building")
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.RELATIONSHIP_BUILDING,
            priority=priority,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=0.7,
            expected_outcomes={
                "improved_relationships": 0.75,
                "increased_trust": 0.7,
                "better_collaboration": 0.8,
                "reduced_tensions": 0.65
            },
            alternative_options=[
                "professional_team_coaching",
                "external_team_building_facilitator",
                "gradual_relationship_repair"
            ],
            execution_timeline="within_30_minutes",
            monitoring_metrics=[
                "relationship_quality_score",
                "trust_measurement",
                "collaboration_frequency",
                "conflict_reduction"
            ],
            risk_assessment={
                "resistance_to_change": 0.3,
                "superficial_improvement": 0.25,
                "time_investment_required": 0.4
            }
        )
    
    async def _decide_cultural_adaptation(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定文化适配策略"""
        cultural_profiles = context.cultural_profiles
        
        actions = []
        reasoning_parts = []
        
        if len(cultural_profiles) > 1:
            # 多文化环境
            actions.extend([
                "use_culturally_neutral_language",
                "avoid_culture_specific_assumptions",
                "check_for_cultural_misunderstandings",
                "provide_cultural_context_when_needed"
            ])
            reasoning_parts.append("Multi-cultural environment requires neutral approach")
            
            # 分析主要文化差异
            communication_styles = [profile.communication_style for profile in cultural_profiles if hasattr(profile, 'communication_style')]
            if len(set(communication_styles)) > 1:
                actions.extend([
                    "bridge_communication_style_differences",
                    "translate_between_direct_indirect_styles",
                    "clarify_intended_meanings"
                ])
                reasoning_parts.append("Mixed communication styles require bridging")
            
            # 分析权力距离差异
            power_distances = [
                profile.dimensions.get('power_distance', 0.5) 
                for profile in cultural_profiles 
                if hasattr(profile, 'dimensions')
            ]
            if power_distances and max(power_distances) - min(power_distances) > 0.4:
                actions.extend([
                    "balance_hierarchical_expectations",
                    "respect_different_authority_concepts",
                    "create_inclusive_participation_structure"
                ])
                reasoning_parts.append("Power distance differences require careful balancing")
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.CULTURAL_ADAPTATION,
            priority=DecisionPriority.MEDIUM,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=0.65,
            expected_outcomes={
                "cultural_inclusivity": 0.8,
                "reduced_misunderstandings": 0.75,
                "improved_participation": 0.7,
                "cultural_learning": 0.6
            },
            alternative_options=[
                "cultural_liaison_support",
                "pre_meeting_cultural_briefing",
                "cultural_sensitivity_training"
            ],
            execution_timeline="immediate",
            monitoring_metrics=[
                "cultural_sensitivity_score",
                "inclusive_participation_rate",
                "misunderstanding_frequency",
                "cultural_satisfaction_feedback"
            ],
            risk_assessment={
                "cultural_insensitivity": 0.2,
                "overcomplicated_approach": 0.15,
                "cultural_stereotyping": 0.1
            }
        )
    
    async def _decide_emotional_regulation(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定情感调节策略"""
        # 分析群体情感状态
        emotions = context.current_emotions
        high_intensity_emotions = []
        
        for participant_id, emotion_vec in emotions.items():
            if emotion_vec.intensity > 0.7:
                dominant_emotion = max(emotion_vec.emotions, key=emotion_vec.emotions.get)
                high_intensity_emotions.append((participant_id, dominant_emotion, emotion_vec.intensity))
        
        actions = []
        reasoning_parts = []
        priority = DecisionPriority.MEDIUM
        
        if high_intensity_emotions:
            negative_emotions = ["anger", "frustration", "anxiety", "fear", "sadness"]
            positive_emotions = ["excitement", "joy", "enthusiasm"]
            
            negative_count = sum(1 for _, emotion, _ in high_intensity_emotions if emotion in negative_emotions)
            positive_count = sum(1 for _, emotion, _ in high_intensity_emotions if emotion in positive_emotions)
            
            if negative_count > len(high_intensity_emotions) * 0.6:
                # 主要是负面情感
                actions.extend([
                    "acknowledge_difficult_emotions",
                    "provide_emotional_support",
                    "create_calming_environment",
                    "focus_on_problem_solving"
                ])
                reasoning_parts.append("High negative emotions require supportive intervention")
                priority = DecisionPriority.HIGH
            elif positive_count > len(high_intensity_emotions) * 0.6:
                # 主要是正面情感
                actions.extend([
                    "harness_positive_energy",
                    "channel_enthusiasm_productively",
                    "maintain_focus_despite_excitement",
                    "celebrate_appropriately"
                ])
                reasoning_parts.append("High positive emotions need productive channeling")
            else:
                # 混合情感
                actions.extend([
                    "balance_different_emotional_states",
                    "validate_all_emotional_experiences",
                    "create_emotional_stability",
                    "promote_emotional_awareness"
                ])
                reasoning_parts.append("Mixed emotional states require balancing approach")
        
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.EMOTIONAL_REGULATION,
            priority=priority,
            context=context,
            recommended_actions=actions,
            reasoning="; ".join(reasoning_parts),
            confidence_score=0.7,
            expected_outcomes={
                "emotional_stability": 0.75,
                "improved_wellbeing": 0.7,
                "better_decision_making": 0.65,
                "group_emotional_intelligence": 0.6
            },
            alternative_options=[
                "individual_emotional_coaching",
                "group_emotional_processing",
                "mindfulness_techniques",
                "break_for_emotional_reset"
            ],
            execution_timeline="within_10_minutes",
            monitoring_metrics=[
                "emotional_intensity_levels",
                "emotional_stability_score",
                "participant_wellbeing",
                "group_emotional_cohesion"
            ],
            risk_assessment={
                "emotional_escalation": 0.2,
                "emotional_suppression": 0.15,
                "inappropriate_intervention": 0.1
            }
        )
    
    async def _decide_leadership_guidance(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定领导指导策略"""
        # 基础实现
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.LEADERSHIP_GUIDANCE,
            priority=DecisionPriority.MEDIUM,
            context=context,
            recommended_actions=["provide_clear_direction", "model_desired_behavior", "support_team_growth"],
            reasoning="Leadership guidance needed for group effectiveness",
            confidence_score=0.6,
            expected_outcomes={"team_performance": 0.7, "leadership_effectiveness": 0.65},
            alternative_options=["peer_leadership", "rotating_leadership"],
            execution_timeline="ongoing",
            monitoring_metrics=["leadership_effectiveness", "team_satisfaction"],
            risk_assessment={"micromanagement": 0.2, "leadership_resistance": 0.15}
        )
    
    async def _decide_team_optimization(self, decision_id: str, context: DecisionContext) -> SocialDecision:
        """决定团队优化策略"""
        # 基础实现
        return SocialDecision(
            decision_id=decision_id,
            decision_type=DecisionType.TEAM_OPTIMIZATION,
            priority=DecisionPriority.LOW,
            context=context,
            recommended_actions=["optimize_team_composition", "improve_processes", "enhance_collaboration"],
            reasoning="Team optimization for improved performance",
            confidence_score=0.6,
            expected_outcomes={"team_efficiency": 0.75, "process_improvement": 0.7},
            alternative_options=["gradual_optimization", "comprehensive_restructuring"],
            execution_timeline="long_term",
            monitoring_metrics=["team_performance_metrics", "process_efficiency"],
            risk_assessment={"disruption_risk": 0.25, "resistance_to_change": 0.3}
        )
    
    def _priority_score(self, priority: DecisionPriority) -> int:
        """将优先级转换为数值分数"""
        return {
            DecisionPriority.CRITICAL: 4,
            DecisionPriority.HIGH: 3,
            DecisionPriority.MEDIUM: 2,
            DecisionPriority.LOW: 1
        }.get(priority, 1)
    
    async def record_decision_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome
    ) -> None:
        """记录决策结果"""
        self.outcome_history[decision_id] = outcome
        
        # 学习和改进
        if self.learning_enabled:
            await self._learn_from_outcome(decision_id, outcome)
        
        logger.info(f"Recorded outcome for decision {decision_id}: success={outcome.execution_success}")
    
    async def _learn_from_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome
    ) -> None:
        """从决策结果中学习"""
        if decision_id not in self.decision_history:
            return
        
        decision = self.decision_history[decision_id]
        decision_type = decision.decision_type
        
        # 更新成功率
        if decision_type not in self.success_metrics:
            self.success_metrics[decision_type] = 0.5
        
        # 使用指数移动平均更新
        alpha = 0.1
        self.success_metrics[decision_type] = (
            alpha * outcome.effectiveness_score + 
            (1 - alpha) * self.success_metrics[decision_type]
        )
        
        # 学习模式
        context_key = self._generate_pattern_key(decision.context)
        if context_key not in self.decision_patterns:
            self.decision_patterns[context_key] = {
                "successful_actions": [],
                "failed_actions": [],
                "effectiveness_scores": []
            }
        
        pattern = self.decision_patterns[context_key]
        pattern["effectiveness_scores"].append(outcome.effectiveness_score)
        
        if outcome.effectiveness_score > 0.7:
            pattern["successful_actions"].extend(decision.recommended_actions)
        elif outcome.effectiveness_score < 0.4:
            pattern["failed_actions"].extend(decision.recommended_actions)
    
    def _generate_pattern_key(self, context: DecisionContext) -> str:
        """生成决策模式键"""
        elements = [
            f"size_{min(context.social_environment.participants_count // 5, 3)}",
            f"formality_{int(context.social_environment.formality_level * 10)}",
            f"cultures_{len(context.cultural_profiles)}"
        ]
        
        # 添加主要情感
        if context.current_emotions:
            avg_emotions = {}
            for emotion_vec in context.current_emotions.values():
                for emotion, value in emotion_vec.emotions.items():
                    avg_emotions[emotion] = avg_emotions.get(emotion, 0) + value
            
            if avg_emotions:
                dominant_emotion = max(avg_emotions, key=avg_emotions.get)
                elements.append(f"emotion_{dominant_emotion}")
        
        return "_".join(elements)
    
    async def _cleanup_history(self) -> None:
        """清理历史记录"""
        if len(self.decision_history) > self.max_history_size:
            # 保留最新的记录
            sorted_items = sorted(
                self.decision_history.items(),
                key=lambda x: x[1].context.timestamp,
                reverse=True
            )
            self.decision_history = dict(sorted_items[:self.max_history_size])
        
        if len(self.outcome_history) > self.max_history_size:
            sorted_items = sorted(
                self.outcome_history.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            self.outcome_history = dict(sorted_items[:self.max_history_size])
    
    async def get_decision_analytics(self) -> Dict[str, Any]:
        """获取决策分析数据"""
        return {
            "total_decisions": len(self.decision_history),
            "success_metrics": self.success_metrics.copy(),
            "decision_patterns_count": len(self.decision_patterns),
            "recent_decisions": len([
                d for d in self.decision_history.values()
                if d.context.timestamp > utc_now() - timedelta(hours=24)
            ]),
            "average_confidence": sum(d.confidence_score for d in self.decision_history.values()) / len(self.decision_history) if self.decision_history else 0,
            "decision_type_distribution": {
                dt.value: sum(1 for d in self.decision_history.values() if d.decision_type == dt)
                for dt in DecisionType
            }
        }
    
    def get_decision_history(self, limit: int = 100) -> List[SocialDecision]:
        """获取决策历史"""
        sorted_decisions = sorted(
            self.decision_history.values(),
            key=lambda x: x.context.timestamp,
            reverse=True
        )
        return sorted_decisions[:limit]
