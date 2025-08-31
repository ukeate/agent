"""
个性化引擎

基于用户个性特征和历史交互优化共情响应
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random

from .models import (
    EmpathyResponse, EmpathyRequest, DialogueContext,
    ResponseTone, CulturalContext
)
from ..emotion_modeling.models import PersonalityProfile, PersonalityTrait, EmotionState

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """个性化共情响应引擎"""
    
    def __init__(self):
        """初始化个性化引擎"""
        # 个性化调整规则
        self.personality_adjustments = {
            # 外向性调整
            PersonalityTrait.EXTRAVERSION: {
                "high": {  # > 0.7
                    "language_patterns": ["我们一起", "分享一下", "聊聊", "交流"],
                    "interaction_encouragement": True,
                    "social_references": True,
                    "tone_adjustment": 0.1  # 更热情
                },
                "low": {  # < 0.3
                    "language_patterns": ["静静", "独自", "内心", "深度思考"],
                    "interaction_encouragement": False,
                    "social_references": False,
                    "tone_adjustment": -0.1  # 更温和
                }
            },
            
            # 宜人性调整
            PersonalityTrait.AGREEABLENESS: {
                "high": {
                    "warmth_multiplier": 1.3,
                    "supportive_language": ["关怀", "温暖", "贴心", "呵护"],
                    "conflict_avoidance": True,
                    "empathy_intensity": 1.2
                },
                "low": {
                    "warmth_multiplier": 0.8,
                    "direct_language": True,
                    "conflict_avoidance": False,
                    "empathy_intensity": 0.9
                }
            },
            
            # 尽责性调整
            PersonalityTrait.CONSCIENTIOUSNESS: {
                "high": {
                    "structured_advice": True,
                    "goal_oriented": True,
                    "planning_language": ["计划", "步骤", "系统性", "有序"],
                    "action_emphasis": 1.3
                },
                "low": {
                    "flexible_approach": True,
                    "spontaneous_language": ["自然", "随性", "灵活", "适应"],
                    "action_emphasis": 0.8
                }
            },
            
            # 神经质调整
            PersonalityTrait.NEUROTICISM: {
                "high": {  # 高敏感性
                    "gentle_language": ["轻柔", "温和", "缓慢", "小心"],
                    "reassurance_emphasis": 1.5,
                    "avoid_intensity": True,
                    "comfort_priority": True
                },
                "low": {  # 情绪稳定
                    "direct_approach": True,
                    "resilience_focus": True,
                    "strength_language": ["坚强", "稳定", "冷静", "理性"]
                }
            },
            
            # 开放性调整
            PersonalityTrait.OPENNESS: {
                "high": {
                    "creative_expressions": True,
                    "metaphor_usage": True,
                    "novel_approaches": ["创新", "独特", "新颖", "探索"],
                    "curiosity_language": True
                },
                "low": {
                    "conventional_approach": True,
                    "familiar_language": ["传统", "熟悉", "常见", "经典"],
                    "practical_focus": True
                }
            }
        }
        
        # 文化适配规则
        self.cultural_adjustments = {
            CulturalContext.COLLECTIVIST: {
                "group_emphasis": True,
                "family_references": True,
                "social_harmony": True,
                "collective_pronouns": ["我们", "大家", "一起"],
                "responsibility_sharing": True
            },
            
            CulturalContext.INDIVIDUALIST: {
                "personal_autonomy": True,
                "self_reliance": True,
                "individual_choice": True,
                "personal_pronouns": ["你", "你的", "你自己"],
                "independence_focus": True
            },
            
            CulturalContext.HIGH_CONTEXT: {
                "indirect_communication": True,
                "implicit_meaning": True,
                "subtle_language": ["似乎", "可能", "或许"],
                "nonverbal_awareness": True
            },
            
            CulturalContext.LOW_CONTEXT: {
                "direct_communication": True,
                "explicit_meaning": True,
                "clear_language": ["明确", "具体", "清楚"],
                "verbal_focus": True
            }
        }
        
        # 个性化模式学习
        self.learned_patterns = {}  # user_id -> pattern_data
        
        # 统计信息
        self.stats = {
            "personalizations_applied": 0,
            "pattern_updates": 0,
            "cultural_adaptations": 0
        }
    
    def personalize_response(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """
        个性化共情响应
        
        Args:
            response: 原始响应
            request: 请求信息
            context: 对话上下文
            
        Returns:
            EmpathyResponse: 个性化后的响应
        """
        try:
            personalized_response = response
            personalization_score = 0.0
            
            # 1. 基于个性特质的调整
            if request.personality_profile:
                personalized_response, personality_score = self._apply_personality_adjustments(
                    personalized_response, request.personality_profile
                )
                personalization_score += personality_score * 0.4
            
            # 2. 文化适配调整
            if request.cultural_context:
                personalized_response, cultural_score = self._apply_cultural_adjustments(
                    personalized_response, request.cultural_context
                )
                personalization_score += cultural_score * 0.2
                self.stats["cultural_adaptations"] += 1
            
            # 3. 历史模式学习调整
            if context:
                personalized_response, pattern_score = self._apply_learned_patterns(
                    personalized_response, request.user_id, context
                )
                personalization_score += pattern_score * 0.2
            
            # 4. 情感状态适配
            if request.emotion_state:
                personalized_response, emotion_score = self._apply_emotional_adaptation(
                    personalized_response, request.emotion_state
                )
                personalization_score += emotion_score * 0.2
            
            # 更新个性化评分
            personalized_response.personalization_score = min(personalization_score, 1.0)
            
            # 学习用户偏好
            self._update_user_patterns(request.user_id, personalized_response, context)
            
            self.stats["personalizations_applied"] += 1
            
            logger.debug(f"Applied personalization with score: {personalization_score}")
            
            return personalized_response
            
        except Exception as e:
            logger.error(f"Error in response personalization: {e}")
            return response  # 返回原始响应
    
    def _apply_personality_adjustments(
        self,
        response: EmpathyResponse,
        personality: PersonalityProfile
    ) -> Tuple[EmpathyResponse, float]:
        """应用个性特质调整"""
        adjusted_text = response.response_text
        adjustments_applied = 0
        total_adjustments = 0
        
        for trait, score in personality.emotional_traits.items():
            if trait not in [t.value for t in PersonalityTrait]:
                continue
                
            trait_enum = PersonalityTrait(trait)
            trait_rules = self.personality_adjustments.get(trait_enum, {})
            
            # 确定特质水平
            if score > 0.7:
                level = "high"
            elif score < 0.3:
                level = "low"
            else:
                continue  # 中等水平不需要特别调整
            
            level_rules = trait_rules.get(level, {})
            if not level_rules:
                continue
                
            total_adjustments += 1
            
            # 应用语言模式调整
            if "language_patterns" in level_rules:
                patterns = level_rules["language_patterns"]
                if self._should_apply_pattern():
                    pattern = random.choice(patterns)
                    if self._can_integrate_pattern(adjusted_text, pattern):
                        adjusted_text = self._integrate_language_pattern(adjusted_text, pattern)
                        adjustments_applied += 1
            
            # 应用交互鼓励
            if level_rules.get("interaction_encouragement", False) and trait_enum == PersonalityTrait.EXTRAVERSION:
                if not any(word in adjusted_text for word in ["一起", "聊聊", "分享"]):
                    adjusted_text += " 想和我多聊聊这个话题吗？"
                    adjustments_applied += 1
            
            # 应用温暖度调整
            if "warmth_multiplier" in level_rules and trait_enum == PersonalityTrait.AGREEABLENESS:
                multiplier = level_rules["warmth_multiplier"]
                if multiplier > 1.0:
                    adjusted_text = self._increase_warmth(adjusted_text)
                    adjustments_applied += 1
                elif multiplier < 1.0:
                    adjusted_text = self._decrease_warmth(adjusted_text)
                    adjustments_applied += 1
            
            # 应用温和语言（神经质调整）
            if level_rules.get("gentle_language") and trait_enum == PersonalityTrait.NEUROTICISM:
                adjusted_text = self._apply_gentle_language(adjusted_text)
                adjustments_applied += 1
            
            # 应用结构化建议（尽责性调整）
            if level_rules.get("structured_advice") and trait_enum == PersonalityTrait.CONSCIENTIOUSNESS:
                response.suggested_actions = self._structure_actions(response.suggested_actions)
                adjustments_applied += 1
        
        response.response_text = adjusted_text
        personalization_score = adjustments_applied / max(total_adjustments, 1)
        
        return response, personalization_score
    
    def _apply_cultural_adjustments(
        self,
        response: EmpathyResponse,
        cultural_context: CulturalContext
    ) -> Tuple[EmpathyResponse, float]:
        """应用文化适配调整"""
        cultural_rules = self.cultural_adjustments.get(cultural_context, {})
        if not cultural_rules:
            return response, 0.0
        
        adjusted_text = response.response_text
        adjustments_applied = 0
        total_possible = len(cultural_rules)
        
        # 集体主义调整
        if cultural_context == CulturalContext.COLLECTIVIST:
            if cultural_rules.get("group_emphasis"):
                adjusted_text = adjusted_text.replace("我理解", "我们大家都理解")
                adjusted_text = adjusted_text.replace("你的", "这种")
                adjustments_applied += 1
            
            if cultural_rules.get("collective_pronouns"):
                pronouns = cultural_rules["collective_pronouns"]
                if random.random() < 0.3:  # 30%的概率添加集体性表达
                    collective_expr = random.choice(pronouns)
                    if collective_expr not in adjusted_text:
                        adjusted_text = adjusted_text.replace("我", collective_expr, 1)
                        adjustments_applied += 1
        
        # 个人主义调整
        elif cultural_context == CulturalContext.INDIVIDUALIST:
            if cultural_rules.get("personal_autonomy"):
                if "自己的决定" not in adjusted_text:
                    adjusted_text += " 最终的选择权在你手中。"
                    adjustments_applied += 1
        
        # 高语境调整
        elif cultural_context == CulturalContext.HIGH_CONTEXT:
            if cultural_rules.get("indirect_communication"):
                # 使语言更加含蓄
                adjusted_text = adjusted_text.replace("确实", "似乎")
                adjusted_text = adjusted_text.replace("一定", "可能")
                adjustments_applied += 1
        
        # 低语境调整
        elif cultural_context == CulturalContext.LOW_CONTEXT:
            if cultural_rules.get("direct_communication"):
                # 使语言更加直接
                clear_words = cultural_rules.get("clear_language", [])
                if clear_words and random.random() < 0.3:
                    clear_word = random.choice(clear_words)
                    if clear_word not in adjusted_text:
                        adjusted_text = f"{clear_word}地说，{adjusted_text}"
                        adjustments_applied += 1
        
        response.response_text = adjusted_text
        response.cultural_adaptation = cultural_context.value
        
        cultural_score = adjustments_applied / max(total_possible, 1)
        return response, cultural_score
    
    def _apply_learned_patterns(
        self,
        response: EmpathyResponse,
        user_id: str,
        context: DialogueContext
    ) -> Tuple[EmpathyResponse, float]:
        """应用学习到的用户模式"""
        if user_id not in self.learned_patterns:
            return response, 0.0
        
        patterns = self.learned_patterns[user_id]
        adjusted_text = response.response_text
        adjustments_applied = 0
        
        # 应用偏好的语调
        if "preferred_tone" in patterns:
            preferred_tone = patterns["preferred_tone"]
            if preferred_tone != response.tone.value:
                response.tone = ResponseTone(preferred_tone)
                adjustments_applied += 1
        
        # 应用偏好的响应长度
        if "preferred_length" in patterns:
            preferred_length = patterns["preferred_length"]
            current_length = len(response.response_text)
            
            if preferred_length == "short" and current_length > 150:
                # 缩短响应
                sentences = adjusted_text.split('。')
                if len(sentences) > 1:
                    adjusted_text = '。'.join(sentences[:2]) + '。'
                    adjustments_applied += 1
            elif preferred_length == "long" and current_length < 100:
                # 延长响应
                if response.suggested_actions:
                    action_text = f" 具体来说，你可以{response.suggested_actions[0]}。"
                    adjusted_text += action_text
                    adjustments_applied += 1
        
        # 应用偏好的共情策略
        if "strategy_preference" in patterns and adjustments_applied == 0:
            # 如果当前策略不是偏好策略，记录但不强制更改
            preferred_strategy = patterns["strategy_preference"]
            if preferred_strategy != response.empathy_type.value:
                # 可以在后续迭代中考虑策略切换
                pass
        
        response.response_text = adjusted_text
        pattern_score = min(adjustments_applied / 3, 1.0)  # 最多3个调整
        
        return response, pattern_score
    
    def _apply_emotional_adaptation(
        self,
        response: EmpathyResponse,
        emotion_state: EmotionState
    ) -> Tuple[EmpathyResponse, float]:
        """基于情感状态的适配"""
        adjusted_text = response.response_text
        adjustments_applied = 0
        
        # 基于情感强度调整语言强度
        if emotion_state.intensity > 0.8:
            # 高强度情感，使用更强烈的表达
            intensity_words = ["深深地", "强烈地", "完全地"]
            if not any(word in adjusted_text for word in intensity_words):
                adjusted_text = adjusted_text.replace("感受到", "深深感受到", 1)
                adjustments_applied += 1
        elif emotion_state.intensity < 0.3:
            # 低强度情感，使用温和的表达
            gentle_words = ["轻微地", "稍许", "有些"]
            if not any(word in adjusted_text for word in gentle_words):
                adjusted_text = adjusted_text.replace("感受到", "稍许感受到", 1)
                adjustments_applied += 1
        
        # 基于情感效价调整
        if emotion_state.valence < -0.7:
            # 非常负面的情感，增加安慰元素
            comfort_phrases = ["我会一直陪伴你", "你并不孤单", "这会过去的"]
            selected_comfort = random.choice(comfort_phrases)
            if selected_comfort not in adjusted_text:
                adjusted_text += f" {selected_comfort}。"
                adjustments_applied += 1
        elif emotion_state.valence > 0.7:
            # 非常积极的情感，增加庆祝元素
            celebration_phrases = ["太棒了", "真为你高兴", "这值得庆祝"]
            selected_celebration = random.choice(celebration_phrases)
            if not any(phrase in adjusted_text for phrase in celebration_phrases):
                adjusted_text = f"{selected_celebration}！{adjusted_text}"
                adjustments_applied += 1
        
        response.response_text = adjusted_text
        emotion_score = min(adjustments_applied / 2, 1.0)  # 最多2个调整
        
        return response, emotion_score
    
    def _update_user_patterns(
        self,
        user_id: str,
        response: EmpathyResponse,
        context: Optional[DialogueContext]
    ):
        """更新用户模式学习"""
        if user_id not in self.learned_patterns:
            self.learned_patterns[user_id] = {}
        
        patterns = self.learned_patterns[user_id]
        
        # 学习语调偏好（基于响应历史）
        if context and context.response_history:
            tone_history = [r.tone.value for r in context.response_history[-5:]]
            most_common_tone = max(set(tone_history), key=tone_history.count)
            patterns["preferred_tone"] = most_common_tone
        
        # 学习响应长度偏好
        current_length = len(response.response_text)
        if "length_history" not in patterns:
            patterns["length_history"] = []
        
        patterns["length_history"].append(current_length)
        if len(patterns["length_history"]) > 10:
            patterns["length_history"] = patterns["length_history"][-10:]
        
        avg_length = sum(patterns["length_history"]) / len(patterns["length_history"])
        if avg_length < 100:
            patterns["preferred_length"] = "short"
        elif avg_length > 200:
            patterns["preferred_length"] = "long"
        else:
            patterns["preferred_length"] = "medium"
        
        # 学习策略偏好
        if context and context.response_history:
            strategy_history = [r.empathy_type.value for r in context.response_history[-10:]]
            if len(strategy_history) > 5:
                most_preferred = max(set(strategy_history), key=strategy_history.count)
                patterns["strategy_preference"] = most_preferred
        
        self.stats["pattern_updates"] += 1
        
        logger.debug(f"Updated patterns for user {user_id}: {patterns}")
    
    # 辅助方法
    def _should_apply_pattern(self) -> bool:
        """决定是否应用语言模式"""
        return random.random() < 0.4  # 40%概率应用
    
    def _can_integrate_pattern(self, text: str, pattern: str) -> bool:
        """检查是否可以集成模式"""
        return pattern not in text and len(text) + len(pattern) < 300
    
    def _integrate_language_pattern(self, text: str, pattern: str) -> str:
        """集成语言模式"""
        if pattern in ["我们一起", "一起"]:
            return text.replace("你可以", f"{pattern}可以", 1)
        elif pattern in ["分享一下", "聊聊"]:
            return text + f" 想{pattern}吗？"
        else:
            return text.replace("我", pattern, 1)
    
    def _increase_warmth(self, text: str) -> str:
        """增加温暖度"""
        warm_replacements = {
            "理解": "深深理解",
            "感受": "真切感受", 
            "支持": "全心全意支持",
            "陪伴": "温暖陪伴"
        }
        
        for original, warmer in warm_replacements.items():
            if original in text:
                text = text.replace(original, warmer, 1)
                break
        
        return text
    
    def _decrease_warmth(self, text: str) -> str:
        """降低温暖度（更理性）"""
        neutral_replacements = {
            "深深理解": "理解",
            "温暖": "",
            "亲爱的": "",
            "贴心": ""
        }
        
        for emotional, neutral in neutral_replacements.items():
            text = text.replace(emotional, neutral)
        
        return text
    
    def _apply_gentle_language(self, text: str) -> str:
        """应用温和语言"""
        gentle_replacements = {
            "确实": "可能",
            "一定": "或许",
            "必须": "可以考虑",
            "应该": "也许可以"
        }
        
        for strong, gentle in gentle_replacements.items():
            text = text.replace(strong, gentle)
        
        return text
    
    def _structure_actions(self, actions: List[str]) -> List[str]:
        """结构化行动建议"""
        if not actions:
            return actions
        
        structured_actions = []
        for i, action in enumerate(actions):
            structured_action = f"{i+1}. {action}"
            structured_actions.append(structured_action)
        
        return structured_actions
    
    def get_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """获取用户学习模式"""
        return self.learned_patterns.get(user_id, {})
    
    def clear_user_patterns(self, user_id: str) -> bool:
        """清除用户模式"""
        if user_id in self.learned_patterns:
            del self.learned_patterns[user_id]
            logger.info(f"Cleared patterns for user {user_id}")
            return True
        return False
    
    def get_personalization_stats(self) -> Dict[str, Any]:
        """获取个性化统计"""
        return {
            **self.stats,
            "total_learned_users": len(self.learned_patterns),
            "avg_patterns_per_user": (
                sum(len(patterns) for patterns in self.learned_patterns.values()) /
                max(len(self.learned_patterns), 1)
            )
        }