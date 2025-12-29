"""
情感共情策略

专注于分享和镜像情感，提供情感上的共鸣和连接
"""

import time
import random
from typing import Dict, Any, Optional, List
from ..models import (
    EmpathyResponse, EmpathyRequest, DialogueContext, 
    EmpathyType, ResponseTone, CulturalContext
)
from ...emotion_modeling.models import EmotionState, PersonalityProfile
from .base_strategy import EmpathyStrategy

class AffectiveEmpathyStrategy(EmpathyStrategy):
    """情感共情策略实现"""
    
    def __init__(self):
        super().__init__(EmpathyType.AFFECTIVE)
        
        # 策略有效性评分 - 情感共情对情感强度较高的情况更有效
        self.effectiveness_scores = {
            "happiness": 0.9,
            "joy": 0.95,
            "excitement": 0.9,
            "sadness": 0.85,
            "grief": 0.9,
            "anger": 0.75,  # 愤怒需要谨慎共鸣
            "fear": 0.8,
            "anxiety": 0.7,  # 焦虑共鸣需要适度
            "surprise": 0.85,
            "disgust": 0.6,
            "neutral": 0.4   # 中性情感不太适合情感共鸣
        }
        
        # 情感共鸣模板库
        self.templates = {
            # 快乐类情感 - 共同分享喜悦
            "happiness": [
                "你的快乐也感染了我！我也为{context}感到高兴。",
                "看到你这么开心，我的心情也变好了！{context}真是太棒了。",
                "你的喜悦让我也感受到了温暖，我们一起庆祝{context}吧！",
                "感受到你的快乐，我也忍不住微笑起来。"
            ],
            "joy": [
                "你纯粹的喜悦深深感染了我，这种快乐是如此美好！",
                "我也被你的快乐包围了，{context}带来的喜悦真是珍贵。",
                "你的欣喜若狂让我也感到兴奋不已！"
            ],
            "excitement": [
                "我也感受到了你的兴奋劲儿！{context}真的让人期待。",
                "你的激动心情也让我热血沸腾，太有感染力了！",
                "这种兴奋感也传递给了我，我们一起期待{context}吧！"
            ],
            
            # 悲伤类情感 - 情感陪伴和共鸣
            "sadness": [
                "看到你难过，我的心也很沉重。我和你一起承受这份痛苦。",
                "你的伤心让我也感到心痛，我陪着你度过这个难关。",
                "你的悲伤也触动了我的心，{context}确实让人心碎。",
                "我感受到了你内心的痛苦，让我陪伴你一起面对。"
            ],
            "grief": [
                "你深深的悲痛也让我的心颤抖，这种失落感我也能感受到。",
                "我和你一起沉浸在这份悲伤中，{context}的离去让我们都很痛苦。",
                "你的眼泪也让我的心在流血，我们一起哀悼这份失去。"
            ],
            "disappointment": [
                "你的失望也让我感到沮丧，{context}没有如愿确实令人难受。",
                "我也为你感到失落，这种期待落空的感觉真的很糟糕。"
            ],
            
            # 愤怒类情感 - 谨慎的情感共鸣
            "anger": [
                "我能感受到你的愤怒，{context}确实让人火大。",
                "你的怒火也点燃了我内心的不平，这确实不公平。",
                "看到你这样愤怒，我也为你感到愤愤不平。"
            ],
            "frustration": [
                "我也感受到了你的挫败感，这种被困住的感觉真的很难受。",
                "你的沮丧也感染了我，{context}确实让人抓狂。"
            ],
            
            # 恐惧类情感 - 情感支持和共鸣
            "fear": [
                "我也感受到了你的恐惧，{context}确实让人心惊胆战。",
                "你的害怕也传递给了我，这种不安全感我们一起承担。",
                "看到你恐惧的样子，我的心也在颤抖。"
            ],
            "anxiety": [
                "我也感受到了你的紧张不安，{context}确实让人焦虑。",
                "你的焦虑情绪也感染了我，这种忧虑让我们都不安。"
            ],
            
            # 惊讶类情感
            "surprise": [
                "我也被{context}震惊了！这种惊讶感太强烈了。",
                "你的惊讶也感染了我，这真是太意外了！",
                "我和你一样感到不可思议，{context}真是让人始料不及。"
            ],
            
            # 其他情感
            "disgust": [
                "我也感受到了你的反感，{context}确实让人不舒服。",
                "你的厌恶感我也能体会到。"
            ],
            "neutral": [
                "我感受到你现在的平静，这种安详的状态让我也很放松。",
                "你的淡定也感染了我，保持这种内心的平衡真好。"
            ]
        }
        
        # 情感强化表达
        self.intensifiers = {
            "high": ["深深地", "强烈地", "真切地", "完全地"],
            "medium": ["也", "同样", "一起"],
            "low": ["有点", "稍微", "轻微地"]
        }
        
        # 情感连接词语
        self.connection_phrases = [
            "我和你一起",
            "我们共同",
            "让我陪你",
            "我也",
            "我同样",
            "我感同身受"
        ]
    
    def generate_response(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """生成情感共情响应"""
        start_time = time.time()
        
        # 获取情感状态
        emotion_state = request.emotion_state
        if not emotion_state:
            emotion_state = EmotionState(emotion="neutral", intensity=0.5)
        
        # 选择合适的模板
        templates = self.templates.get(emotion_state.emotion, self.templates["neutral"])
        base_template = random.choice(templates)
        
        # 构建上下文信息
        context_info = self._extract_context_info(request.message, context)
        
        # 生成基础回应
        base_response = base_template.format(context=context_info)
        
        # 基于情感强度调整表达强度
        base_response = self._adjust_intensity(base_response, emotion_state.intensity)
        
        # 添加情感连接表达
        if emotion_state.intensity > 0.6:
            base_response = self._add_emotional_connection(base_response, emotion_state)
        
        # 个性化调整
        if request.personality_profile:
            base_response = self.adapt_for_personality(base_response, request.personality_profile)
        
        # 文化适配
        base_response = self._adapt_for_culture(base_response, request.cultural_context)
        
        # 计算各项指标
        comfort_level = self.calculate_comfort_level(emotion_state, request.personality_profile)
        # 情感共情的安慰程度通常较高
        comfort_level = min(comfort_level + 0.1, 1.0)
        
        personalization_score = self._calculate_personalization_score(request)
        confidence = self._calculate_confidence(emotion_state, context)
        
        # 生成建议行动
        suggested_actions = self._generate_suggested_actions(emotion_state)
        
        generation_time = (time.time() - start_time) * 1000
        
        return EmpathyResponse(
            response_text=base_response,
            empathy_type=EmpathyType.AFFECTIVE,
            emotion_addressed=emotion_state.emotion,
            comfort_level=comfort_level,
            personalization_score=personalization_score,
            suggested_actions=suggested_actions,
            tone=self._determine_tone(emotion_state, request.personality_profile),
            confidence=confidence,
            generation_time_ms=generation_time,
            cultural_adaptation=request.cultural_context.value if request.cultural_context else None,
            template_used=f"affective_{emotion_state.emotion}",
            metadata={
                "strategy_type": "affective",
                "emotion_intensity": emotion_state.intensity,
                "emotional_resonance": self._calculate_resonance_strength(emotion_state),
                "context_extracted": context_info
            }
        )
    
    def is_suitable(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> float:
        """判断情感共情策略的适合度"""
        base_score = self.get_effectiveness_score(emotion_state.emotion)
        
        # 情感强度调整 - 情感共情对高强度情感更适合
        intensity_bonus = emotion_state.intensity * 0.3
        base_score += intensity_bonus
        
        # 个性调整
        if personality:
            # 高外向性的人更喜欢情感共鸣
            extraversion = personality.emotional_traits.get("extraversion", 0.5)
            base_score += (extraversion - 0.5) * 0.25
            
            # 高情感敏感性的人适合情感共情
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            if neuroticism > 0.6:  # 适度的情感敏感性有利于共情
                base_score += 0.15
            elif neuroticism > 0.8:  # 过高可能需要更理性的方法
                base_score -= 0.1
                
            # 高宜人性的人喜欢情感连接
            agreeableness = personality.emotional_traits.get("agreeableness", 0.5)
            base_score += (agreeableness - 0.5) * 0.2
        
        # 上下文调整
        if context:
            # 检查情感升级情况
            if context.is_emotional_escalation():
                base_score += 0.2  # 情感升级时更需要共鸣
            
            # 避免过度使用同一策略
            recent_responses = context.get_recent_responses(3)
            affective_count = sum(1 for r in recent_responses if r.empathy_type == EmpathyType.AFFECTIVE)
            if affective_count >= 2:
                base_score *= 0.85
        
        # 特殊情感调整
        if emotion_state.emotion in ["anger", "rage"]:
            # 愤怒情感需要谨慎共鸣，避免放大负面情绪
            base_score *= 0.8
        elif emotion_state.emotion in ["anxiety", "panic"]:
            # 焦虑和恐慌需要适度共鸣，避免加剧不安
            base_score *= 0.85
        
        return min(max(base_score, 0.0), 1.0)
    
    def _extract_context_info(self, message: str, context: Optional[DialogueContext] = None) -> str:
        """提取上下文信息"""
        if not message:
            return "这种感受"
        
        # 提取情感相关的关键词
        emotion_contexts = {
            "成功": ["成功", "胜利", "成就", "达成", "完成", "实现"],
            "失败": ["失败", "失误", "错误", "挫折", "败北"],
            "分离": ["分手", "离别", "分离", "离开", "告别", "失去"],
            "团聚": ["相聚", "团聚", "见面", "重逢", "相遇"],
            "挑战": ["挑战", "困难", "难题", "障碍", "问题"],
            "机会": ["机会", "可能", "希望", "前景", "未来"],
            "变化": ["变化", "改变", "转变", "新的", "不同"]
        }
        
        for context_key, keywords in emotion_contexts.items():
            if any(kw in message for kw in keywords):
                return context_key
        
        return "这种经历"
    
    def _adjust_intensity(self, response: str, intensity: float) -> str:
        """基于情感强度调整表达强度"""
        if intensity >= 0.8:
            # 高强度 - 使用强化词语
            intensifier = random.choice(self.intensifiers["high"])
            response = response.replace("感受到", f"{intensifier}感受到")
            response = response.replace("也", f"也{intensifier}")
        elif intensity >= 0.6:
            # 中等强度 - 使用标准表达
            intensifier = random.choice(self.intensifiers["medium"])
            if "我" in response and not response.startswith(intensifier):
                response = response.replace("我", intensifier + "我", 1)
        else:
            # 低强度 - 使用温和表达
            intensifier = random.choice(self.intensifiers["low"])
            response = response.replace("感受到", f"{intensifier}感受到")
        
        return response
    
    def _add_emotional_connection(self, response: str, emotion_state: EmotionState) -> str:
        """添加情感连接表达"""
        if emotion_state.emotion in ["happiness", "joy", "excitement"]:
            # 积极情感添加共享快乐的表达
            connection_phrases = ["让我们一起", "我们共同", "一起"]
            addition = f" {random.choice(connection_phrases)}感受这份美好！"
            response += addition
        elif emotion_state.emotion in ["sadness", "grief", "fear"]:
            # 负面情感添加陪伴表达
            connection_phrases = ["我会陪伴你", "让我陪你一起", "我们一起面对"]
            addition = f" {random.choice(connection_phrases)}。"
            response += addition
        
        return response
    
    def _adapt_for_culture(self, response: str, cultural_context: Optional[CulturalContext]) -> str:
        """文化适配"""
        if not cultural_context:
            return response
        
        if cultural_context == CulturalContext.COLLECTIVIST:
            # 集体主义文化强调群体情感共鸣
            response = response.replace("我", "我们大家")
            response = response.replace("你的", "大家的")
            
        elif cultural_context == CulturalContext.INDIVIDUALIST:
            # 个人主义文化强调个体情感体验
            response = response.replace("我们", "我")
            
        elif cultural_context == CulturalContext.HIGH_CONTEXT:
            # 高语境文化更含蓄表达情感
            response = response.replace("深深地", "")
            response = response.replace("强烈地", "")
            
        return response
    
    def _calculate_personalization_score(self, request: EmpathyRequest) -> float:
        """计算个性化评分"""
        score = 0.0
        
        # 基础分
        if request.personality_profile:
            score += 0.4
        
        if request.dialogue_context and request.dialogue_context.emotion_history:
            score += 0.3
        
        if request.cultural_context:
            score += 0.2
        
        # 情感共情通过情感强度匹配提供个性化
        if request.emotion_state and request.emotion_state.intensity > 0.6:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, emotion_state: EmotionState, context: Optional[DialogueContext]) -> float:
        """计算置信度"""
        base_confidence = 0.85  # 情感共情通常有较高置信度
        
        # 基于情感强度 - 高强度情感更容易共鸣
        if emotion_state.intensity > 0.7:
            base_confidence += 0.1
        elif emotion_state.intensity < 0.3:
            base_confidence -= 0.2  # 低强度情感难以产生强烈共鸣
        
        # 基于策略有效性
        effectiveness = self.get_effectiveness_score(emotion_state.emotion)
        base_confidence = (base_confidence + effectiveness) / 2
        
        # 特定情感的置信度调整
        if emotion_state.emotion in ["anger", "rage"]:
            base_confidence -= 0.15  # 愤怒共鸣需要更谨慎
        elif emotion_state.emotion in ["happiness", "joy"]:
            base_confidence += 0.1   # 积极情感共鸣更安全
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _calculate_resonance_strength(self, emotion_state: EmotionState) -> float:
        """计算情感共鸣强度"""
        base_strength = emotion_state.intensity
        
        # 基于情感类型调整
        if emotion_state.emotion in ["happiness", "joy", "excitement"]:
            base_strength += 0.1  # 积极情感更容易共鸣
        elif emotion_state.emotion in ["anger", "rage"]:
            base_strength -= 0.2  # 愤怒共鸣需要控制
        
        return min(max(base_strength, 0.0), 1.0)
    
    def _generate_suggested_actions(self, emotion_state: EmotionState) -> List[str]:
        """生成建议行动"""
        actions = []
        
        # 情感共情策略的行动建议更偏向情感表达和分享
        action_templates = {
            "happiness": ["与身边的人分享这份快乐", "记录这个美好的时刻", "让快乐感染更多人"],
            "sadness": ["允许自己充分感受这份悲伤", "寻找能够理解你的人倾诉", "通过创作或写作表达内心"],
            "anger": ["找到安全的方式释放愤怒", "与信任的人讨论你的感受", "等情绪平复后再做决定"],
            "fear": ["向身边的人寻求情感支持", "与有类似经历的人交流", "专注于当下的安全感"],
            "anxiety": ["与了解你的人分享担忧", "寻找能带来平静的活动", "练习情感调节技巧"],
            "excitement": ["与朋友分享你的兴奋", "充分享受这种期待感", "为即将到来的事情做准备"]
        }
        
        if emotion_state.emotion in action_templates:
            emotion_actions = action_templates[emotion_state.emotion]
            if emotion_state.intensity > 0.7:
                actions = emotion_actions[:2]
            else:
                actions = [random.choice(emotion_actions)]
        
        return actions
    
    def _determine_tone(self, emotion_state: EmotionState, personality: Optional[PersonalityProfile]) -> ResponseTone:
        """确定回应语调"""
        # 情感共情通常使用更温暖和支持性的语调
        tone_mapping = {
            "happiness": ResponseTone.WARM,
            "joy": ResponseTone.ENTHUSIASTIC,
            "excitement": ResponseTone.ENTHUSIASTIC,
            "sadness": ResponseTone.GENTLE,
            "grief": ResponseTone.GENTLE,
            "anger": ResponseTone.UNDERSTANDING,
            "fear": ResponseTone.SUPPORTIVE,
            "anxiety": ResponseTone.SUPPORTIVE,
            "surprise": ResponseTone.WARM,
            "neutral": ResponseTone.WARM
        }
        
        base_tone = tone_mapping.get(emotion_state.emotion, ResponseTone.WARM)
        
        # 基于个性调整
        if personality:
            extraversion = personality.emotional_traits.get("extraversion", 0.5)
            agreeableness = personality.emotional_traits.get("agreeableness", 0.5)
            
            # 高外向性和宜人性的组合偏向更热情的表达
            if extraversion > 0.7 and agreeableness > 0.7:
                if base_tone == ResponseTone.WARM:
                    base_tone = ResponseTone.ENTHUSIASTIC
            elif extraversion < 0.3 or agreeableness < 0.3:
                # 低外向性或宜人性偏向更温和的表达
                if base_tone == ResponseTone.ENTHUSIASTIC:
                    base_tone = ResponseTone.WARM
        
        return base_tone
