"""
文化背景感知模块 - Story 11.6 Task 4
处理跨文化交流中的情感理解和适配
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import json
from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface

from src.core.logging import get_logger
logger = get_logger(__name__)

class CulturalDimension(Enum):
    """文化维度"""
    POWER_DISTANCE = "power_distance"  # 权力距离
    INDIVIDUALISM = "individualism"  # 个人主义vs集体主义
    MASCULINITY = "masculinity"  # 男性化vs女性化
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"  # 不确定性规避
    LONG_TERM_ORIENTATION = "long_term_orientation"  # 长期导向
    INDULGENCE = "indulgence"  # 放纵vs约束

class CommunicationStyle(Enum):
    """沟通风格"""
    HIGH_CONTEXT = "high_context"  # 高语境
    LOW_CONTEXT = "low_context"  # 低语境
    DIRECT = "direct"  # 直接
    INDIRECT = "indirect"  # 间接
    FORMAL = "formal"  # 正式
    INFORMAL = "informal"  # 非正式

@dataclass
class CulturalProfile:
    """文化档案"""
    culture_id: str
    name: str
    dimensions: Dict[CulturalDimension, float]  # 0.0-1.0
    communication_style: CommunicationStyle
    emotional_expression_norms: Dict[str, float]  # 情感表达规范
    social_hierarchies: List[str]  # 社会等级制度
    taboo_topics: List[str]  # 禁忌话题
    greeting_customs: List[str]  # 问候习俗
    conflict_resolution_style: str  # 冲突解决风格
    time_orientation: str  # 时间观念

@dataclass
class CulturalAdaptation:
    """文化适配结果"""
    original_emotion: EmotionVector
    adapted_emotion: EmotionVector
    cultural_context: CulturalProfile
    adaptation_strategies: List[str]
    cultural_sensitivity_score: float
    potential_misunderstandings: List[str]
    recommended_approach: str

class CulturalContextAnalyzer(EmotionModelingInterface):
    """文化背景感知分析器"""
    
    def __init__(self):
        self.cultural_profiles = self._initialize_cultural_profiles()
        self.adaptation_rules = self._initialize_adaptation_rules()
        self.cross_cultural_patterns: Dict[str, Any] = {}
        self.learning_enabled = True
        
    def _initialize_cultural_profiles(self) -> Dict[str, CulturalProfile]:
        """初始化文化档案"""
        return {
            "western_individualistic": CulturalProfile(
                culture_id="western_individualistic",
                name="西方个人主义文化",
                dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.3,
                    CulturalDimension.INDIVIDUALISM: 0.8,
                    CulturalDimension.MASCULINITY: 0.5,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.4,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.4,
                    CulturalDimension.INDULGENCE: 0.7
                },
                communication_style=CommunicationStyle.DIRECT,
                emotional_expression_norms={
                    "assertiveness": 0.8,
                    "emotional_openness": 0.7,
                    "professional_distance": 0.6,
                    "personal_space": 0.8
                },
                social_hierarchies=["merit-based", "professional"],
                taboo_topics=["personal_income", "political_extremes", "personal_relationships"],
                greeting_customs=["handshake", "brief_eye_contact", "smile"],
                conflict_resolution_style="direct_discussion",
                time_orientation="monochronic"
            ),
            "east_asian_collectivistic": CulturalProfile(
                culture_id="east_asian_collectivistic",
                name="东亚集体主义文化",
                dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.7,
                    CulturalDimension.INDIVIDUALISM: 0.2,
                    CulturalDimension.MASCULINITY: 0.6,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.8,
                    CulturalDimension.INDULGENCE: 0.3
                },
                communication_style=CommunicationStyle.HIGH_CONTEXT,
                emotional_expression_norms={
                    "emotional_restraint": 0.8,
                    "harmony_maintenance": 0.9,
                    "face_saving": 0.9,
                    "hierarchy_respect": 0.8
                },
                social_hierarchies=["age-based", "position-based", "educational"],
                taboo_topics=["personal_failures", "criticism_of_elders", "individual_achievements_over_group"],
                greeting_customs=["bow", "formal_titles", "group_acknowledgment"],
                conflict_resolution_style="indirect_mediation",
                time_orientation="polychronic"
            ),
            "latin_expressive": CulturalProfile(
                culture_id="latin_expressive",
                name="拉丁表达文化",
                dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.6,
                    CulturalDimension.INDIVIDUALISM: 0.4,
                    CulturalDimension.MASCULINITY: 0.6,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.7,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.3,
                    CulturalDimension.INDULGENCE: 0.8
                },
                communication_style=CommunicationStyle.HIGH_CONTEXT,
                emotional_expression_norms={
                    "emotional_warmth": 0.9,
                    "physical_closeness": 0.8,
                    "expressive_communication": 0.9,
                    "family_orientation": 0.9
                },
                social_hierarchies=["family-based", "age-based", "gender-based"],
                taboo_topics=["family_criticism", "religious_disputes", "economic_inequality"],
                greeting_customs=["embrace", "cheek_kiss", "warm_smile", "extended_conversation"],
                conflict_resolution_style="emotional_expression",
                time_orientation="flexible"
            ),
            "northern_european_reserved": CulturalProfile(
                culture_id="northern_european_reserved",
                name="北欧保守文化",
                dimensions={
                    CulturalDimension.POWER_DISTANCE: 0.2,
                    CulturalDimension.INDIVIDUALISM: 0.6,
                    CulturalDimension.MASCULINITY: 0.3,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.3,
                    CulturalDimension.LONG_TERM_ORIENTATION: 0.7,
                    CulturalDimension.INDULGENCE: 0.5
                },
                communication_style=CommunicationStyle.DIRECT,
                emotional_expression_norms={
                    "emotional_reserve": 0.8,
                    "practical_focus": 0.9,
                    "equality_emphasis": 0.9,
                    "privacy_respect": 0.9
                },
                social_hierarchies=["egalitarian", "competence-based"],
                taboo_topics=["personal_wealth", "emotional_problems", "social_status"],
                greeting_customs=["brief_handshake", "minimal_small_talk", "respect_for_space"],
                conflict_resolution_style="rational_discussion",
                time_orientation="punctual"
            )
        }
    
    def _initialize_adaptation_rules(self) -> Dict[str, Any]:
        """初始化适配规则"""
        return {
            "high_power_distance": {
                "respect_hierarchy": 0.9,
                "formal_address": 0.8,
                "avoid_direct_contradiction": 0.8
            },
            "low_power_distance": {
                "encourage_participation": 0.8,
                "informal_interaction": 0.7,
                "direct_feedback_ok": 0.8
            },
            "high_context": {
                "read_between_lines": 0.9,
                "indirect_communication": 0.8,
                "relationship_focus": 0.9
            },
            "low_context": {
                "explicit_communication": 0.9,
                "task_focus": 0.8,
                "direct_feedback": 0.8
            },
            "collectivistic": {
                "group_harmony": 0.9,
                "consensus_building": 0.8,
                "face_saving": 0.9
            },
            "individualistic": {
                "personal_achievement": 0.8,
                "individual_responsibility": 0.8,
                "direct_recognition": 0.7
            }
        }
    
    async def analyze_cultural_context(
        self,
        participants: List[Dict[str, Any]],
        interaction_context: Dict[str, Any]
    ) -> Tuple[List[CulturalProfile], float]:
        """分析参与者的文化背景"""
        detected_cultures = []
        confidence_scores = []
        
        for participant in participants:
            cultural_indicators = participant.get("cultural_indicators", {})
            
            # 基于指标匹配文化档案
            culture_scores = {}
            
            for culture_id, profile in self.cultural_profiles.items():
                score = self._calculate_cultural_match_score(
                    cultural_indicators, profile
                )
                culture_scores[culture_id] = score
            
            # 选择最匹配的文化
            best_match = max(culture_scores, key=culture_scores.get)
            confidence = culture_scores[best_match]
            
            detected_cultures.append(self.cultural_profiles[best_match])
            confidence_scores.append(confidence)
        
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        return detected_cultures, overall_confidence
    
    def _calculate_cultural_match_score(
        self,
        indicators: Dict[str, Any],
        profile: CulturalProfile
    ) -> float:
        """计算文化匹配分数"""
        score = 0.0
        total_weights = 0.0
        
        # 检查通信风格指标
        comm_style = indicators.get("communication_style")
        if comm_style == profile.communication_style.value:
            score += 0.3
        total_weights += 0.3
        
        # 检查文化维度指标
        for dimension, value in indicators.get("dimensions", {}).items():
            if dimension in profile.dimensions:
                diff = abs(value - profile.dimensions[dimension])
                match_score = 1.0 - diff  # 差异越小分数越高
                score += match_score * 0.1
                total_weights += 0.1
        
        # 检查语言和地区指标
        language = indicators.get("language")
        region = indicators.get("region")
        
        language_culture_mapping = {
            "en": ["western_individualistic", "northern_european_reserved"],
            "zh": ["east_asian_collectivistic"],
            "es": ["latin_expressive"],
            "sv": ["northern_european_reserved"]
        }
        
        if language in language_culture_mapping:
            if profile.culture_id in language_culture_mapping[language]:
                score += 0.2
            total_weights += 0.2
        
        return score / total_weights if total_weights > 0 else 0.5
    
    async def adapt_for_cultural_context(
        self,
        emotion_vector: EmotionVector,
        cultural_profiles: List[CulturalProfile],
        interaction_context: Dict[str, Any]
    ) -> CulturalAdaptation:
        """为文化背景适配情感表达"""
        try:
            if not cultural_profiles:
                # 默认适配
                return CulturalAdaptation(
                    original_emotion=emotion_vector,
                    adapted_emotion=emotion_vector,
                    cultural_context=self.cultural_profiles["western_individualistic"],
                    adaptation_strategies=["use_default_approach"],
                    cultural_sensitivity_score=0.5,
                    potential_misunderstandings=[],
                    recommended_approach="neutral_professional"
                )
            
            # 选择主导文化（简化处理，取第一个）
            primary_culture = cultural_profiles[0]
            
            # 应用文化适配
            adapted_emotion = self._apply_cultural_adaptation(
                emotion_vector, primary_culture
            )
            
            # 生成适配策略
            strategies = self._generate_adaptation_strategies(
                primary_culture, emotion_vector, interaction_context
            )
            
            # 识别潜在误解
            misunderstandings = self._identify_potential_misunderstandings(
                primary_culture, emotion_vector, cultural_profiles[1:] if len(cultural_profiles) > 1 else []
            )
            
            # 推荐方法
            recommended_approach = self._recommend_approach(
                primary_culture, adapted_emotion, interaction_context
            )
            
            # 计算文化敏感度分数
            sensitivity_score = self._calculate_cultural_sensitivity(
                emotion_vector, adapted_emotion, primary_culture
            )
            
            return CulturalAdaptation(
                original_emotion=emotion_vector,
                adapted_emotion=adapted_emotion,
                cultural_context=primary_culture,
                adaptation_strategies=strategies,
                cultural_sensitivity_score=sensitivity_score,
                potential_misunderstandings=misunderstandings,
                recommended_approach=recommended_approach
            )
            
        except Exception as e:
            logger.error(f"Cultural adaptation failed: {e}")
            return CulturalAdaptation(
                original_emotion=emotion_vector,
                adapted_emotion=emotion_vector,
                cultural_context=cultural_profiles[0] if cultural_profiles else self.cultural_profiles["western_individualistic"],
                adaptation_strategies=[f"adaptation_failed: {str(e)}"],
                cultural_sensitivity_score=0.3,
                potential_misunderstandings=["technical_error"],
                recommended_approach="cautious_neutral"
            )
    
    def _apply_cultural_adaptation(
        self,
        emotion_vector: EmotionVector,
        cultural_profile: CulturalProfile
    ) -> EmotionVector:
        """应用文化适配规则"""
        adapted_emotions = emotion_vector.emotions.copy()
        
        # 根据文化档案调整情感表达
        expression_norms = cultural_profile.emotional_expression_norms
        
        for norm, strength in expression_norms.items():
            if norm == "emotional_restraint" and strength > 0.7:
                # 高情感克制文化：降低强烈情感表达
                intense_emotions = ["anger", "excitement", "frustration"]
                for emotion in intense_emotions:
                    if emotion in adapted_emotions:
                        adapted_emotions[emotion] *= (1 - strength * 0.3)
                        
            elif norm == "emotional_warmth" and strength > 0.7:
                # 高情感温暖文化：增强积极情感
                warm_emotions = ["happiness", "enthusiasm", "friendliness"]
                for emotion in warm_emotions:
                    if emotion in adapted_emotions:
                        adapted_emotions[emotion] *= (1 + strength * 0.2)
                        
            elif norm == "hierarchy_respect" and strength > 0.7:
                # 高等级尊重文化：增强尊重相关情感
                respectful_emotions = ["respectful", "formal", "cautious"]
                for emotion in respectful_emotions:
                    if emotion in adapted_emotions:
                        adapted_emotions[emotion] *= (1 + strength * 0.3)
        
        # 根据权力距离调整
        power_distance = cultural_profile.dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5)
        if power_distance > 0.7:
            # 高权力距离：降低挑战性情感
            challenging_emotions = ["assertive", "confrontational", "challenging"]
            for emotion in challenging_emotions:
                if emotion in adapted_emotions:
                    adapted_emotions[emotion] *= (1 - power_distance * 0.4)
        
        # 重新标准化
        total = sum(adapted_emotions.values())
        if total > 0:
            adapted_emotions = {k: v/total for k, v in adapted_emotions.items()}
        
        return EmotionVector(
            emotions=adapted_emotions,
            intensity=emotion_vector.intensity,
            confidence=emotion_vector.confidence,
            context=emotion_vector.context
        )
    
    def _generate_adaptation_strategies(
        self,
        cultural_profile: CulturalProfile,
        emotion_vector: EmotionVector,
        context: Dict[str, Any]
    ) -> List[str]:
        """生成适配策略"""
        strategies = []
        
        # 基于通信风格
        if cultural_profile.communication_style == CommunicationStyle.HIGH_CONTEXT:
            strategies.extend([
                "pay_attention_to_nonverbal_cues",
                "read_between_the_lines",
                "build_relationship_first",
                "use_indirect_communication"
            ])
        elif cultural_profile.communication_style == CommunicationStyle.LOW_CONTEXT:
            strategies.extend([
                "be_explicit_and_direct",
                "focus_on_facts_and_details",
                "avoid_ambiguity",
                "get_to_the_point"
            ])
        
        # 基于权力距离
        power_distance = cultural_profile.dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5)
        if power_distance > 0.7:
            strategies.extend([
                "respect_hierarchical_structure",
                "use_formal_titles",
                "defer_to_authority",
                "avoid_direct_challenges"
            ])
        else:
            strategies.extend([
                "encourage_open_participation",
                "treat_everyone_equally",
                "welcome_direct_feedback",
                "be_informal_when_appropriate"
            ])
        
        # 基于个人主义程度
        individualism = cultural_profile.dimensions.get(CulturalDimension.INDIVIDUALISM, 0.5)
        if individualism > 0.7:
            strategies.extend([
                "recognize_individual_achievements",
                "respect_personal_space",
                "focus_on_individual_goals",
                "encourage_self_expression"
            ])
        else:
            strategies.extend([
                "emphasize_group_harmony",
                "build_consensus",
                "consider_collective_impact",
                "maintain_face_for_all"
            ])
        
        return strategies[:10]  # 限制策略数量
    
    def _identify_potential_misunderstandings(
        self,
        primary_culture: CulturalProfile,
        emotion_vector: EmotionVector,
        other_cultures: List[CulturalProfile]
    ) -> List[str]:
        """识别潜在文化误解"""
        misunderstandings = []
        
        for other_culture in other_cultures:
            # 比较通信风格
            if primary_culture.communication_style != other_culture.communication_style:
                if primary_culture.communication_style == CommunicationStyle.DIRECT:
                    misunderstandings.append("directness_may_seem_rude")
                else:
                    misunderstandings.append("indirectness_may_be_confusing")
            
            # 比较权力距离
            pd_diff = abs(
                primary_culture.dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5) -
                other_culture.dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5)
            )
            if pd_diff > 0.3:
                misunderstandings.append("hierarchy_expectations_differ")
            
            # 比较个人主义程度
            ind_diff = abs(
                primary_culture.dimensions.get(CulturalDimension.INDIVIDUALISM, 0.5) -
                other_culture.dimensions.get(CulturalDimension.INDIVIDUALISM, 0.5)
            )
            if ind_diff > 0.3:
                misunderstandings.append("individual_vs_group_focus_conflict")
        
        # 检查情感表达冲突
        primary_norms = primary_culture.emotional_expression_norms
        if primary_norms.get("emotional_restraint", 0) > 0.7:
            dominant_emotion = max(emotion_vector.emotions, key=emotion_vector.emotions.get)
            if dominant_emotion in ["anger", "excitement", "frustration"]:
                misunderstandings.append("emotional_expression_may_be_inappropriate")
        
        return misunderstandings
    
    def _recommend_approach(
        self,
        cultural_profile: CulturalProfile,
        adapted_emotion: EmotionVector,
        context: Dict[str, Any]
    ) -> str:
        """推荐交流方法"""
        # 基于文化特征选择方法
        if cultural_profile.communication_style == CommunicationStyle.HIGH_CONTEXT:
            if cultural_profile.dimensions.get(CulturalDimension.POWER_DISTANCE, 0.5) > 0.7:
                return "formal_respectful_indirect"
            else:
                return "warm_relationship_focused"
        elif cultural_profile.communication_style == CommunicationStyle.DIRECT:
            if cultural_profile.dimensions.get(CulturalDimension.INDIVIDUALISM, 0.5) > 0.7:
                return "direct_task_focused"
            else:
                return "direct_but_considerate"
        
        return "balanced_neutral"
    
    def _calculate_cultural_sensitivity(
        self,
        original: EmotionVector,
        adapted: EmotionVector,
        cultural_profile: CulturalProfile
    ) -> float:
        """计算文化敏感度分数"""
        base_score = 0.5
        
        # 根据适配程度调整
        adaptation_degree = 1.0 - self._calculate_emotion_similarity(original, adapted)
        adaptation_bonus = adaptation_degree * 0.3
        
        # 根据文化复杂度调整
        cultural_complexity = len(cultural_profile.taboo_topics) * 0.05
        complexity_bonus = min(cultural_complexity, 0.2)
        
        sensitivity_score = base_score + adaptation_bonus + complexity_bonus
        return min(max(sensitivity_score, 0.0), 1.0)
    
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
    
    async def learn_cultural_patterns(
        self,
        interaction_data: Dict[str, Any],
        outcome_feedback: Dict[str, Any]
    ) -> None:
        """学习文化交流模式"""
        if not self.learning_enabled:
            return
        
        culture_combo = interaction_data.get("culture_combination", "unknown")
        success_score = outcome_feedback.get("success_score", 0.5)
        
        if culture_combo not in self.cross_cultural_patterns:
            self.cross_cultural_patterns[culture_combo] = {
                "success_rates": [],
                "effective_strategies": {},
                "common_issues": []
            }
        
        patterns = self.cross_cultural_patterns[culture_combo]
        patterns["success_rates"].append(success_score)
        
        # 记录有效策略
        if success_score > 0.7:
            strategies = interaction_data.get("strategies_used", [])
            for strategy in strategies:
                patterns["effective_strategies"][strategy] = patterns["effective_strategies"].get(strategy, 0) + 1
        
        logger.info(f"Cultural pattern learning updated for {culture_combo}")
    
    def get_cultural_profiles(self) -> Dict[str, CulturalProfile]:
        """获取文化档案"""
        return self.cultural_profiles.copy()
    
    def add_cultural_profile(self, profile: CulturalProfile) -> None:
        """添加新的文化档案"""
        self.cultural_profiles[profile.culture_id] = profile
        logger.info(f"Added cultural profile: {profile.culture_id}")
    
    async def get_cross_cultural_insights(self) -> Dict[str, Any]:
        """获取跨文化交流洞察"""
        insights = {
            "total_patterns": len(self.cross_cultural_patterns),
            "most_successful_combinations": [],
            "common_challenges": [],
            "effective_strategies": {}
        }
        
        # 分析最成功的文化组合
        for combo, data in self.cross_cultural_patterns.items():
            if data["success_rates"]:
                avg_success = sum(data["success_rates"]) / len(data["success_rates"])
                insights["most_successful_combinations"].append({
                    "combination": combo,
                    "success_rate": avg_success,
                    "sample_size": len(data["success_rates"])
                })
        
        # 排序
        insights["most_successful_combinations"].sort(
            key=lambda x: x["success_rate"], reverse=True
        )
        
        return insights
