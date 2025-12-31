"""
社交场景适配系统 - Story 11.6 Task 3
负责根据不同的社交环境动态调整情感理解和响应策略
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import json
from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface

from src.core.logging import get_logger
logger = get_logger(__name__)

class SocialScenario(Enum):
    """社交场景类型"""
    FORMAL_MEETING = "formal_meeting"
    CASUAL_CONVERSATION = "casual_conversation"
    TEAM_BRAINSTORMING = "team_brainstorming"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PRESENTATION = "presentation"
    NETWORKING = "networking"
    INTERVIEW = "interview"
    TRAINING_SESSION = "training_session"
    SOCIAL_GATHERING = "social_gathering"
    CRISIS_MANAGEMENT = "crisis_management"

@dataclass
class ContextRule:
    """上下文规则"""
    scenario: SocialScenario
    priority: int  # 1-10, 10为最高优先级
    emotional_weights: Dict[str, float]  # 情感权重调整
    behavioral_expectations: List[str]  # 行为期望
    communication_style: str  # 沟通风格
    sensitivity_level: float  # 敏感度级别 0.0-1.0
    
@dataclass
class ContextualResponse:
    """上下文化响应"""
    original_emotion: EmotionVector
    adapted_emotion: EmotionVector
    scenario: SocialScenario
    adaptation_reason: str
    confidence_score: float
    suggested_actions: List[str]

@dataclass
class SocialEnvironment:
    """社交环境描述"""
    scenario: SocialScenario
    participants_count: int
    formality_level: float  # 0.0-1.0, 正式程度
    emotional_intensity: float  # 0.0-1.0, 情感强度
    time_pressure: float  # 0.0-1.0, 时间压力
    cultural_context: Optional[str] = None
    dominant_emotions: List[str] = None
    power_dynamics: Dict[str, float] = None  # 权力动态

class SocialContextAdapter(EmotionModelingInterface):
    """社交场景适配器"""
    
    def __init__(self):
        self.context_rules = self._initialize_context_rules()
        self.scenario_history: List[Tuple[SocialScenario, datetime]] = []
        self.adaptation_cache: Dict[str, ContextualResponse] = {}
        self.learning_enabled = True
        
    def _initialize_context_rules(self) -> Dict[SocialScenario, ContextRule]:
        """初始化上下文规则"""
        return {
            SocialScenario.FORMAL_MEETING: ContextRule(
                scenario=SocialScenario.FORMAL_MEETING,
                priority=8,
                emotional_weights={
                    "professional": 0.9,
                    "respectful": 0.8,
                    "focused": 0.8,
                    "enthusiastic": 0.6,
                    "casual": 0.2
                },
                behavioral_expectations=[
                    "maintain professional demeanor",
                    "speak clearly and concisely",
                    "show active listening",
                    "respect hierarchy"
                ],
                communication_style="formal_professional",
                sensitivity_level=0.9
            ),
            SocialScenario.CASUAL_CONVERSATION: ContextRule(
                scenario=SocialScenario.CASUAL_CONVERSATION,
                priority=5,
                emotional_weights={
                    "friendly": 0.9,
                    "relaxed": 0.8,
                    "humorous": 0.7,
                    "empathetic": 0.8,
                    "formal": 0.3
                },
                behavioral_expectations=[
                    "be approachable and warm",
                    "use casual language",
                    "show personal interest",
                    "maintain eye contact"
                ],
                communication_style="casual_friendly",
                sensitivity_level=0.6
            ),
            SocialScenario.CONFLICT_RESOLUTION: ContextRule(
                scenario=SocialScenario.CONFLICT_RESOLUTION,
                priority=10,
                emotional_weights={
                    "calm": 0.9,
                    "neutral": 0.8,
                    "empathetic": 0.9,
                    "patient": 0.9,
                    "aggressive": 0.1
                },
                behavioral_expectations=[
                    "remain calm and neutral",
                    "actively listen to all parties",
                    "seek win-win solutions",
                    "manage emotional escalation"
                ],
                communication_style="meditative_neutral",
                sensitivity_level=0.95
            ),
            SocialScenario.TEAM_BRAINSTORMING: ContextRule(
                scenario=SocialScenario.TEAM_BRAINSTORMING,
                priority=7,
                emotional_weights={
                    "creative": 0.9,
                    "open": 0.8,
                    "encouraging": 0.8,
                    "energetic": 0.7,
                    "critical": 0.3
                },
                behavioral_expectations=[
                    "encourage creative thinking",
                    "build on others' ideas",
                    "maintain positive energy",
                    "suspend judgment initially"
                ],
                communication_style="collaborative_creative",
                sensitivity_level=0.4
            ),
            SocialScenario.PRESENTATION: ContextRule(
                scenario=SocialScenario.PRESENTATION,
                priority=8,
                emotional_weights={
                    "confident": 0.9,
                    "clear": 0.9,
                    "engaging": 0.8,
                    "authoritative": 0.7,
                    "nervous": 0.2
                },
                behavioral_expectations=[
                    "project confidence and authority",
                    "engage the audience",
                    "maintain clear delivery",
                    "handle questions professionally"
                ],
                communication_style="presentation_authoritative",
                sensitivity_level=0.7
            )
        }
    
    async def adapt_to_context(
        self,
        emotion_vector: EmotionVector,
        social_environment: SocialEnvironment,
        participant_context: Optional[Dict[str, Any]] = None
    ) -> ContextualResponse:
        """根据社交环境适配情感响应"""
        try:
            # 获取适配规则
            context_rule = self.context_rules.get(social_environment.scenario)
            if not context_rule:
                logger.warning(f"No context rule found for scenario: {social_environment.scenario}")
                return ContextualResponse(
                    original_emotion=emotion_vector,
                    adapted_emotion=emotion_vector,
                    scenario=social_environment.scenario,
                    adaptation_reason="No specific rule available",
                    confidence_score=0.5,
                    suggested_actions=["proceed with default behavior"]
                )
            
            # 应用情感权重调整
            adapted_emotions = self._apply_emotional_weights(
                emotion_vector, context_rule.emotional_weights
            )
            
            # 考虑环境因素
            adapted_emotions = self._adjust_for_environment(
                adapted_emotions, social_environment
            )
            
            # 生成建议行动
            suggested_actions = self._generate_suggested_actions(
                context_rule, social_environment, adapted_emotions
            )
            
            # 计算适配信心分数
            confidence_score = self._calculate_adaptation_confidence(
                emotion_vector, adapted_emotions, context_rule
            )
            
            response = ContextualResponse(
                original_emotion=emotion_vector,
                adapted_emotion=adapted_emotions,
                scenario=social_environment.scenario,
                adaptation_reason=f"Applied {context_rule.scenario.value} context rules",
                confidence_score=confidence_score,
                suggested_actions=suggested_actions
            )
            
            # 缓存响应用于学习
            cache_key = self._generate_cache_key(emotion_vector, social_environment)
            self.adaptation_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Context adaptation failed: {e}")
            return ContextualResponse(
                original_emotion=emotion_vector,
                adapted_emotion=emotion_vector,
                scenario=social_environment.scenario,
                adaptation_reason=f"Adaptation failed: {str(e)}",
                confidence_score=0.3,
                suggested_actions=["use default response"]
            )
    
    def _apply_emotional_weights(
        self, 
        emotion_vector: EmotionVector, 
        weights: Dict[str, float]
    ) -> EmotionVector:
        """应用情感权重调整"""
        adjusted_emotions = emotion_vector.emotions.copy()
        
        for emotion, weight in weights.items():
            if emotion in adjusted_emotions:
                adjusted_emotions[emotion] *= weight
                
        # 重新标准化
        total = sum(adjusted_emotions.values())
        if total > 0:
            adjusted_emotions = {k: v/total for k, v in adjusted_emotions.items()}
        
        return EmotionVector(
            emotions=adjusted_emotions,
            intensity=emotion_vector.intensity * (sum(weights.values()) / len(weights)),
            confidence=emotion_vector.confidence,
            context=emotion_vector.context
        )
    
    def _adjust_for_environment(
        self, 
        emotion_vector: EmotionVector, 
        environment: SocialEnvironment
    ) -> EmotionVector:
        """根据环境因素进行调整"""
        adjusted_emotions = emotion_vector.emotions.copy()
        
        # 根据正式程度调整
        if environment.formality_level > 0.7:
            # 高正式环境，降低随意情感
            casual_emotions = ["playful", "silly", "casual"]
            for emotion in casual_emotions:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= (1 - environment.formality_level * 0.5)
        
        # 根据情感强度调整
        intensity_multiplier = 0.5 + environment.emotional_intensity * 0.5
        
        # 根据时间压力调整
        if environment.time_pressure > 0.7:
            # 高时间压力，增加紧迫性情感
            urgent_emotions = ["focused", "determined", "urgent"]
            for emotion in urgent_emotions:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= 1.2
        
        return EmotionVector(
            emotions=adjusted_emotions,
            intensity=emotion_vector.intensity * intensity_multiplier,
            confidence=emotion_vector.confidence,
            context=emotion_vector.context
        )
    
    def _generate_suggested_actions(
        self,
        context_rule: ContextRule,
        environment: SocialEnvironment,
        adapted_emotion: EmotionVector
    ) -> List[str]:
        """生成建议行动"""
        actions = context_rule.behavioral_expectations.copy()
        
        # 根据主导情感添加特定建议
        dominant_emotion = max(adapted_emotion.emotions, key=adapted_emotion.emotions.get)
        
        emotion_specific_actions = {
            "anxious": ["take deep breaths", "focus on preparation", "visualize success"],
            "confident": ["maintain eye contact", "speak clearly", "take initiative"],
            "frustrated": ["take a pause", "reframe the situation", "focus on solutions"],
            "excited": ["channel energy positively", "maintain focus", "share enthusiasm appropriately"],
            "calm": ["maintain steady demeanor", "provide stabilizing presence", "think before responding"]
        }
        
        if dominant_emotion in emotion_specific_actions:
            actions.extend(emotion_specific_actions[dominant_emotion])
        
        # 根据环境添加特定建议
        if environment.participants_count > 10:
            actions.append("project voice for large group")
            actions.append("use inclusive language")
        
        if environment.time_pressure > 0.8:
            actions.append("prioritize key points")
            actions.append("be concise and direct")
        
        return actions[:8]  # 限制建议数量
    
    def _calculate_adaptation_confidence(
        self,
        original: EmotionVector,
        adapted: EmotionVector,
        context_rule: ContextRule
    ) -> float:
        """计算适配信心分数"""
        base_confidence = original.confidence
        
        # 根据规则优先级调整
        priority_factor = context_rule.priority / 10.0
        
        # 根据适配程度调整
        emotion_similarity = self._calculate_emotion_similarity(original, adapted)
        adaptation_factor = 1.0 - (emotion_similarity * 0.3)  # 适配越大，信心越高
        
        confidence = base_confidence * priority_factor * adaptation_factor
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_emotion_similarity(self, e1: EmotionVector, e2: EmotionVector) -> float:
        """计算情感向量相似度"""
        common_emotions = set(e1.emotions.keys()) & set(e2.emotions.keys())
        if not common_emotions:
            return 0.0
        
        similarity_sum = 0.0
        for emotion in common_emotions:
            diff = abs(e1.emotions[emotion] - e2.emotions[emotion])
            similarity_sum += 1.0 - diff
        
        return similarity_sum / len(common_emotions)
    
    def _generate_cache_key(self, emotion: EmotionVector, environment: SocialEnvironment) -> str:
        """生成缓存键"""
        emotion_key = "_".join(f"{k}:{v:.2f}" for k, v in emotion.emotions.items())
        env_key = f"{environment.scenario.value}_{environment.formality_level:.1f}_{environment.participants_count}"
        return f"{emotion_key}|{env_key}"
    
    async def detect_scenario(self, context_clues: Dict[str, Any]) -> Tuple[SocialScenario, float]:
        """基于上下文线索检测社交场景"""
        scenario_scores = {}
        
        # 分析关键词
        keywords = context_clues.get("keywords", [])
        keyword_mapping = {
            SocialScenario.FORMAL_MEETING: ["agenda", "minutes", "presentation", "quarterly", "budget"],
            SocialScenario.CASUAL_CONVERSATION: ["chat", "coffee", "break", "weekend", "family"],
            SocialScenario.CONFLICT_RESOLUTION: ["dispute", "disagreement", "mediation", "complaint", "issue"],
            SocialScenario.TEAM_BRAINSTORMING: ["ideas", "creative", "brainstorm", "innovation", "solution"],
            SocialScenario.PRESENTATION: ["slides", "audience", "demo", "showcase", "pitch"],
        }
        
        for scenario, scenario_keywords in keyword_mapping.items():
            score = len(set(keywords) & set(scenario_keywords)) / max(len(scenario_keywords), 1)
            scenario_scores[scenario] = score
        
        # 分析环境因素
        participant_count = context_clues.get("participant_count", 2)
        formality = context_clues.get("formality_level", 0.5)
        
        # 调整分数
        if participant_count > 10:
            scenario_scores[SocialScenario.PRESENTATION] += 0.3
        if formality > 0.8:
            scenario_scores[SocialScenario.FORMAL_MEETING] += 0.3
        if formality < 0.3:
            scenario_scores[SocialScenario.CASUAL_CONVERSATION] += 0.3
        
        # 找到最高分场景
        if scenario_scores:
            best_scenario = max(scenario_scores, key=scenario_scores.get)
            confidence = min(scenario_scores[best_scenario], 1.0)
            return best_scenario, confidence
        
        return SocialScenario.CASUAL_CONVERSATION, 0.5
    
    async def learn_from_feedback(
        self,
        original_response: ContextualResponse,
        feedback: Dict[str, Any]
    ) -> None:
        """从反馈中学习优化适配规则"""
        if not self.learning_enabled:
            return
        
        effectiveness_score = feedback.get("effectiveness", 0.5)
        user_satisfaction = feedback.get("satisfaction", 0.5)
        
        scenario = original_response.scenario
        if scenario in self.context_rules:
            context_rule = self.context_rules[scenario]
            
            # 根据反馈调整权重
            if effectiveness_score < 0.3:
                # 降低当前权重的影响
                for emotion, weight in context_rule.emotional_weights.items():
                    context_rule.emotional_weights[emotion] *= 0.9
            elif effectiveness_score > 0.8:
                # 增强成功的权重模式
                for emotion, weight in context_rule.emotional_weights.items():
                    if weight > 0.5:
                        context_rule.emotional_weights[emotion] *= 1.05
        
        logger.info(f"Learning applied for scenario {scenario.value} with effectiveness {effectiveness_score}")
    
    def get_context_rules(self) -> Dict[SocialScenario, ContextRule]:
        """获取当前上下文规则"""
        return self.context_rules.copy()
    
    def update_context_rule(self, scenario: SocialScenario, rule: ContextRule) -> None:
        """更新上下文规则"""
        self.context_rules[scenario] = rule
        logger.info(f"Updated context rule for {scenario.value}")
    
    async def get_adaptation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取适配历史"""
        return [
            {
                "scenario": scenario.value,
                "timestamp": timestamp.isoformat(),
                "cache_size": len(self.adaptation_cache)
            }
            for scenario, timestamp in self.scenario_history[-limit:]
        ]
