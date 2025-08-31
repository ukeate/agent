"""
认知共情策略

专注于理解和识别情感，提供理性的共情回应
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


class CognitiveEmpathyStrategy(EmpathyStrategy):
    """认知共情策略实现"""
    
    def __init__(self):
        super().__init__(EmpathyType.COGNITIVE)
        
        # 策略有效性评分 - 认知共情对所有情感都有基础有效性
        self.effectiveness_scores = {
            "happiness": 0.7,
            "joy": 0.75,
            "sadness": 0.85,
            "grief": 0.8,
            "anger": 0.8,
            "fear": 0.85,
            "anxiety": 0.9,
            "surprise": 0.7,
            "disgust": 0.7,
            "neutral": 0.6
        }
        
        # 认知共情模板库
        self.templates = {
            # 快乐类情感
            "happiness": [
                "我能感受到你现在很开心，{context}真的值得庆祝！",
                "看得出来你心情很好，这种快乐的感觉真棒！",
                "你的快乐让我也感到高兴，{context}一定很特别。"
            ],
            "joy": [
                "你现在的喜悦感真是感染人，我能感受到你内心的满足。",
                "这份纯粹的快乐很珍贵，{context}带给你的感受一定很美好。"
            ],
            "excitement": [
                "我能感受到你的兴奋劲儿，{context}一定让你很期待！",
                "你的激动心情溢于言表，这种兴奋感真是太棒了。"
            ],
            
            # 悲伤类情感  
            "sadness": [
                "我理解你现在的难过，{context}确实让人感到沉重。",
                "我能感受到你的痛苦，经历{context}一定不容易。",
                "你现在的悲伤情绪是完全可以理解的，{context}对任何人来说都不轻松。"
            ],
            "grief": [
                "我深深理解你现在的悲痛，{context}带来的失落感是如此真实。",
                "这种深层的悲伤需要时间来愈合，我理解你现在的感受。"
            ],
            "disappointment": [
                "我能理解{context}给你带来的失望感，这种落差确实不好受。",
                "失望的情绪很正常，{context}没有达到预期确实令人沮丧。"
            ],
            
            # 愤怒类情感
            "anger": [
                "我能理解你现在的愤怒，{context}确实令人感到不公。",
                "你的愤怒是有道理的，{context}的确让人难以接受。",
                "我明白{context}激起了你的怒火，这种反应很自然。"
            ],
            "frustration": [
                "我理解你的挫败感，{context}确实让人感到无能为力。",
                "这种受阻的感觉我能理解，{context}让你感到很挫折。"
            ],
            
            # 恐惧类情感
            "fear": [
                "我能理解你现在的恐惧，面对{context}感到害怕是人之常情。",
                "你的担心是可以理解的，{context}确实存在不确定性。",
                "恐惧是一种保护机制，我理解{context}让你感到不安。"
            ],
            "anxiety": [
                "我理解你现在的焦虑，{context}带来的不确定感确实令人不安。",
                "焦虑的情绪我能理解，{context}让你感到担忧是正常的。",
                "我明白{context}引起了你的焦虑，这种紧张感是可以理解的。"
            ],
            "worry": [
                "我理解你的担忧，{context}确实值得关注。",
                "你的担心很有道理，{context}的不确定性让人忧虑。"
            ],
            
            # 其他情感
            "surprise": [
                "我能感受到你的惊讶，{context}确实很出乎意料。",
                "这种意外的感觉我能理解，{context}真是让人始料未及。"
            ],
            "disgust": [
                "我理解{context}让你感到反感，这种厌恶情绪是正常的。",
                "你对{context}的反感是可以理解的。"
            ],
            "neutral": [
                "我注意到你现在的状态比较平静，{context}让你保持了内心的平衡。",
                "你现在的平和状态很好，我理解{context}对你来说是正常的。"
            ]
        }
        
        # 理性化建议模板
        self.rational_advice = {
            "sadness": [
                "虽然现在很难过，但这种情绪会随时间慢慢愈合。",
                "悲伤是处理失落的自然方式，给自己一些时间。"
            ],
            "anger": [
                "愤怒告诉我们什么是重要的，关键是如何建设性地表达。",
                "深呼吸可能有助于让思维更加清晰。"
            ],
            "fear": [
                "恐惧往往比实际危险更可怕，让我们理性分析一下情况。",
                "面对恐惧的第一步是承认它的存在。"
            ],
            "anxiety": [
                "焦虑提醒我们注意重要的事情，但不要让它主导思维。",
                "专注于可控制的因素可能会有帮助。"
            ]
        }
    
    def generate_response(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """生成认知共情响应"""
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
        
        # 添加理性化建议
        if emotion_state.emotion in self.rational_advice and emotion_state.intensity > 0.6:
            advice = random.choice(self.rational_advice[emotion_state.emotion])
            base_response += f" {advice}"
        
        # 个性化调整
        if request.personality_profile:
            base_response = self.adapt_for_personality(base_response, request.personality_profile)
        
        # 文化适配
        base_response = self._adapt_for_culture(base_response, request.cultural_context)
        
        # 计算各项指标
        comfort_level = self.calculate_comfort_level(emotion_state, request.personality_profile)
        personalization_score = self._calculate_personalization_score(request)
        confidence = self._calculate_confidence(emotion_state, context)
        
        # 生成建议行动
        suggested_actions = self._generate_suggested_actions(emotion_state)
        
        generation_time = (time.time() - start_time) * 1000
        
        return EmpathyResponse(
            response_text=base_response,
            empathy_type=EmpathyType.COGNITIVE,
            emotion_addressed=emotion_state.emotion,
            comfort_level=comfort_level,
            personalization_score=personalization_score,
            suggested_actions=suggested_actions,
            tone=self._determine_tone(emotion_state, request.personality_profile),
            confidence=confidence,
            generation_time_ms=generation_time,
            cultural_adaptation=request.cultural_context.value if request.cultural_context else None,
            template_used=f"cognitive_{emotion_state.emotion}",
            metadata={
                "strategy_type": "cognitive",
                "emotion_intensity": emotion_state.intensity,
                "context_extracted": context_info
            }
        )
    
    def is_suitable(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> float:
        """判断认知共情策略的适合度"""
        base_score = self.get_effectiveness_score(emotion_state.emotion)
        
        # 个性调整
        if personality:
            # 高开放性的人更适合认知共情
            openness = personality.emotional_traits.get("openness", 0.5)
            base_score += (openness - 0.5) * 0.2
            
            # 高尽责性的人偏好理性方法
            conscientiousness = personality.emotional_traits.get("conscientiousness", 0.5)
            base_score += (conscientiousness - 0.5) * 0.15
            
            # 低神经质的人更能接受理性分析
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            base_score += (0.5 - neuroticism) * 0.1
        
        # 上下文调整
        if context:
            # 如果最近使用了太多认知策略，降低适合度
            recent_responses = context.get_recent_responses(3)
            cognitive_count = sum(1 for r in recent_responses if r.empathy_type == EmpathyType.COGNITIVE)
            if cognitive_count >= 2:
                base_score *= 0.8
        
        # 情感强度调整
        if emotion_state.intensity > 0.8:
            # 极高强度情感可能需要更直接的情感支持
            base_score *= 0.9
        
        return min(max(base_score, 0.0), 1.0)
    
    def _extract_context_info(self, message: str, context: Optional[DialogueContext] = None) -> str:
        """提取上下文信息"""
        if not message:
            return "这种情况"
        
        # 简单的关键词提取
        keywords = []
        
        # 工作相关
        work_keywords = ["工作", "项目", "任务", "会议", "同事", "老板", "客户", "业绩", "压力"]
        if any(kw in message for kw in work_keywords):
            keywords.append("工作上的事")
        
        # 关系相关
        relationship_keywords = ["朋友", "家人", "恋人", "父母", "孩子", "伴侣", "关系"]
        if any(kw in message for kw in relationship_keywords):
            keywords.append("人际关系")
        
        # 健康相关
        health_keywords = ["身体", "健康", "生病", "医院", "疼痛", "治疗"]
        if any(kw in message for kw in health_keywords):
            keywords.append("健康问题")
        
        # 学习相关
        study_keywords = ["学习", "考试", "学校", "成绩", "课程", "老师", "作业"]
        if any(kw in message for kw in study_keywords):
            keywords.append("学习方面")
        
        if keywords:
            return f"在{keywords[0]}方面"
        else:
            return "这种情况"
    
    def _adapt_for_culture(self, response: str, cultural_context: Optional[CulturalContext]) -> str:
        """文化适配"""
        if not cultural_context:
            return response
        
        if cultural_context == CulturalContext.COLLECTIVIST:
            # 集体主义文化强调群体支持
            response = response.replace("我理解", "我们都理解")
            if "你" in response and response.count("你") > 2:
                # 适度降低直接性
                response = response.replace("你的", "这种")
        
        elif cultural_context == CulturalContext.HIGH_CONTEXT:
            # 高语境文化更含蓄
            response = response.replace("确实", "似乎")
            response = response.replace("一定", "可能")
        
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
        
        if request.multimodal_emotion:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, emotion_state: EmotionState, context: Optional[DialogueContext]) -> float:
        """计算置信度"""
        base_confidence = 0.8
        
        # 基于情感识别置信度
        if hasattr(emotion_state, 'confidence'):
            base_confidence = (base_confidence + emotion_state.confidence) / 2
        
        # 基于策略有效性
        effectiveness = self.get_effectiveness_score(emotion_state.emotion)
        base_confidence = (base_confidence + effectiveness) / 2
        
        # 基于上下文丰富程度
        if context and len(context.emotion_history) > 3:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generate_suggested_actions(self, emotion_state: EmotionState) -> List[str]:
        """生成建议行动"""
        actions = []
        
        action_templates = {
            "sadness": ["给自己一些独处时间", "和信任的朋友聊聊", "做一些让自己感到安慰的活动"],
            "anger": ["深呼吸冷静一下", "找到问题的根源", "以建设性的方式表达感受"],
            "fear": ["分析具体的担忧点", "制定应对计划", "寻求相关信息或支持"],
            "anxiety": ["专注于当下可控的事情", "尝试放松技巧", "与专业人士交流"],
            "happiness": ["分享这份快乐", "记录美好的时刻", "感恩当下的幸福"],
            "neutral": ["反思当前的状态", "设定一些小目标", "保持现有的平衡"]
        }
        
        if emotion_state.emotion in action_templates:
            # 基于情感强度选择1-2个建议
            emotion_actions = action_templates[emotion_state.emotion]
            if emotion_state.intensity > 0.7:
                actions = emotion_actions[:2]  # 高强度提供更多建议
            else:
                actions = [random.choice(emotion_actions)]
        
        return actions
    
    def _determine_tone(self, emotion_state: EmotionState, personality: Optional[PersonalityProfile]) -> ResponseTone:
        """确定回应语调"""
        # 基于情感类型的基础语调
        tone_mapping = {
            "sadness": ResponseTone.GENTLE,
            "grief": ResponseTone.GENTLE,
            "anger": ResponseTone.UNDERSTANDING,
            "fear": ResponseTone.SUPPORTIVE,
            "anxiety": ResponseTone.SUPPORTIVE,
            "happiness": ResponseTone.WARM,
            "joy": ResponseTone.ENTHUSIASTIC,
            "neutral": ResponseTone.PROFESSIONAL
        }
        
        base_tone = tone_mapping.get(emotion_state.emotion, ResponseTone.UNDERSTANDING)
        
        # 基于个性调整
        if personality:
            extraversion = personality.emotional_traits.get("extraversion", 0.5)
            if extraversion > 0.7 and base_tone != ResponseTone.GENTLE:
                # 高外向性偏向更热情的语调
                if base_tone == ResponseTone.UNDERSTANDING:
                    base_tone = ResponseTone.WARM
            elif extraversion < 0.3:
                # 低外向性偏向更专业的语调
                if base_tone in [ResponseTone.ENTHUSIASTIC, ResponseTone.WARM]:
                    base_tone = ResponseTone.PROFESSIONAL
        
        return base_tone