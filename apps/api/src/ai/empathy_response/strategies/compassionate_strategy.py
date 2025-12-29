"""
慈悲共情策略

专注于理解并提供支持行动，提供情感安慰和建设性帮助
"""

import time
import random
from typing import Dict, Any, Optional, List
from ..models import (
    EmpathyResponse, EmpathyRequest, DialogueContext, 
    EmpathyType, ResponseTone, CulturalContext, COMFORT_TECHNIQUES
)
from ...emotion_modeling.models import EmotionState, PersonalityProfile
from .base_strategy import EmpathyStrategy

class CompassionateEmpathyStrategy(EmpathyStrategy):
    """慈悲共情策略实现"""
    
    def __init__(self):
        super().__init__(EmpathyType.COMPASSIONATE)
        
        # 策略有效性评分 - 慈悲共情对困难情感和高强度情感最有效
        self.effectiveness_scores = {
            "happiness": 0.7,   # 可以提供庆祝和支持
            "joy": 0.75,
            "sadness": 0.95,    # 慈悲共情对悲伤最有效
            "grief": 0.98,      # 对悲痛有最高效力
            "disappointment": 0.9,
            "loneliness": 0.95,
            "despair": 0.98,    # 对绝望提供希望和支持
            "anger": 0.85,      # 提供建设性的愤怒管理
            "frustration": 0.88,
            "fear": 0.92,       # 提供安全感和支持
            "anxiety": 0.9,
            "panic": 0.95,
            "worry": 0.85,
            "surprise": 0.6,
            "disgust": 0.7,
            "neutral": 0.5
        }
        
        # 慈悲共情模板库
        self.templates = {
            # 悲伤类情感 - 深度支持和安慰
            "sadness": [
                "我深深地理解你的痛苦，{context}确实让人心碎。你不需要独自承受这一切，让我陪伴你度过这段艰难时光。",
                "看到你如此痛苦，我的心也在为你疼痛。{context}带来的悲伤是如此真实，但请记住，你并不孤单。",
                "你现在承受的痛苦我能感受到，{context}确实是沉重的打击。但你是坚强的，我们会一起找到走出阴霾的路。",
                "这种深深的悲伤需要时间来愈合，{context}留下的伤痕不会马上消失。我会一直陪在你身边，直到阳光重新照进你的心里。"
            ],
            "grief": [
                "失去所爱之人的痛苦是人世间最深的伤痛，我能理解{context}给你带来的巨大冲击。请允许自己悲伤，这是爱的另一种表达。",
                "悲痛是因为深爱，{context}的离开让整个世界都变了颜色。但爱不会因为离别而消失，它会以另一种方式陪伴你。",
                "面对{context}的离去，你的心正在经历最深的痛苦。让我陪你一起怀念那些美好的时光，一起承受这份思念。"
            ],
            "despair": [
                "绝望的感觉就像被黑暗包围，{context}让你觉得没有出路。但即使在最黑暗的时刻，也有微光在等待。让我成为你的那束光。",
                "我知道{context}让你感到万念俱灰，这种绝望的痛苦我能理解。但你的生命中还有未曾显现的可能性，让我们一起去发现。",
                "当{context}让你觉得世界崩塌时，请相信这种感觉会过去。我会陪伴你走过这段最黑暗的路程，直到希望重新点亮。"
            ],
            
            # 恐惧类情感 - 安全感和保护
            "fear": [
                "面对未知的恐惧是人之常情，{context}让你感到不安全是可以理解的。我会和你一起面对这些恐惧，你并不孤单。",
                "恐惧告诉我们什么是重要的，{context}触发了你内心的担忧。让我们一起制定计划，把恐惧转化为行动的力量。",
                "我理解{context}带给你的恐惧感，这种不确定性确实令人害怕。但勇敢不是不害怕，而是在害怕的时候仍然前进。"
            ],
            "anxiety": [
                "焦虑就像心中的风暴，{context}让你的思绪翻腾不止。让我做你的避风港，我们一起寻找内心的平静。",
                "我看到{context}给你带来的焦虑，这种担忧正在消耗你的能量。让我们一起学会与焦虑相处，找到内心的安宁。",
                "焦虑像是对未来的过度担心，{context}激发了你内心的不安。但我们可以专注于当下，一步一步走向平静。"
            ],
            "panic": [
                "恐慌发作时感觉世界都在旋转，{context}触发了你内心深层的恐惧。深呼吸，我就在这里，你是安全的。",
                "我知道{context}让你感到极度恐慌，这种失控的感觉很可怕。但这种感觉会过去的，让我陪伴你度过这个时刻。"
            ],
            
            # 愤怒类情感 - 建设性引导
            "anger": [
                "你的愤怒是对不公正的正当反应，{context}确实让人难以接受。让我们把这股能量转化为改变现状的力量。",
                "愤怒背后往往藏着受伤的心，{context}触碰了你的底线。让我帮你找到既能表达感受又不伤害自己的方式。",
                "我理解{context}激起的愤怒，这种强烈的情绪需要出口。让我们一起寻找建设性的方式来处理这份愤怒。"
            ],
            "frustration": [
                "被困住的感觉确实令人沮丧，{context}让你感到力不从心。让我们一起分析问题，寻找突破的方法。",
                "挫败感说明你很在乎这件事，{context}的阻碍让你感到无力。但每个难题都有解决的可能，让我们一起寻找答案。"
            ],
            
            # 积极情感 - 庆祝和分享
            "happiness": [
                "你的快乐就是最好的礼物，{context}为你的生活带来了美好。让我们一起庆祝这份幸福，它值得被记住和珍惜。",
                "看到你如此开心，我的心也被温暖填满。{context}带来的快乐是珍贵的，愿它能持续照亮你的每一天。"
            ],
            "joy": [
                "纯粹的喜悦是生命最美的体验，{context}让你的生活如此闪闪发光。这份美好值得被细细品味和分享。"
            ],
            
            # 中性情感
            "neutral": [
                "有时候内心的平静也是一种珍贵的状态，{context}让你保持了这份淡然。在这个快节奏的世界里，能够静下来思考很不容易。"
            ]
        }
        
        # 建设性行动建议模板
        self.action_suggestions = {
            "sadness": [
                "给自己时间去感受和处理这份悲伤",
                "寻找信任的朋友或专业人士倾诉",
                "通过写作、绘画等方式表达内心感受",
                "建立日常小习惯来照顾自己",
                "考虑加入支持小组与有类似经历的人交流"
            ],
            "grief": [
                "允许自己按照自己的节奏哀悼",
                "创建纪念仪式来表达对逝者的思念",
                "保持与理解你的朋友和家人的联系",
                "考虑咨询专业的哀伤辅导师",
                "记录美好的回忆，让爱以新的形式延续"
            ],
            "anger": [
                "学习健康的愤怒表达方式",
                "通过运动或其他方式释放身体的紧张",
                "写下愤怒的原因，理性分析问题根源",
                "学习冲突解决和沟通技巧",
                "寻找建设性的方式来改变引起愤怒的情况"
            ],
            "fear": [
                "制定具体的应对计划来减少不确定性",
                "练习放松和接地技巧",
                "寻求专业帮助学习恐惧管理",
                "逐步暴露疗法，小步骤面对恐惧",
                "建立安全网络，确保有人支持"
            ],
            "anxiety": [
                "学习深呼吸和正念冥想技巧",
                "建立规律的生活作息",
                "限制咖啡因和刺激性物质摄入",
                "尝试认知行为疗法技巧",
                "寻找专业的焦虑管理资源"
            ]
        }
        
        # 希望和鼓励的表达
        self.hope_expressions = [
            "虽然现在很困难，但你有内在的力量去克服这一切。",
            "这段痛苦的经历会让你变得更加坚强和智慧。",
            "每一次的眼泪都在为内心的愈合做准备。",
            "黑暗中的微光正在聚集，很快就会照亮你的道路。",
            "你的勇敢和坚持会得到回报，美好的时光正在路上。",
            "这只是人生的一个章节，更精彩的故事还在后面。"
        ]
    
    def generate_response(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """生成慈悲共情响应"""
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
        
        # 添加安慰技巧
        base_response = self._apply_comfort_techniques(base_response, emotion_state)
        
        # 为负面情感添加希望元素
        if emotion_state.valence < -0.3 and emotion_state.intensity > 0.6:
            hope_expression = random.choice(self.hope_expressions)
            base_response += f" {hope_expression}"
        
        # 个性化调整
        if request.personality_profile:
            base_response = self.adapt_for_personality(base_response, request.personality_profile)
        
        # 文化适配
        base_response = self._adapt_for_culture(base_response, request.cultural_context)
        
        # 计算各项指标
        comfort_level = self.calculate_comfort_level(emotion_state, request.personality_profile)
        # 慈悲共情通常提供最高的安慰水平
        comfort_level = min(comfort_level + 0.2, 1.0)
        
        personalization_score = self._calculate_personalization_score(request)
        confidence = self._calculate_confidence(emotion_state, context)
        
        # 生成建设性行动建议
        suggested_actions = self._generate_suggested_actions(emotion_state, request.urgency_level)
        
        generation_time = (time.time() - start_time) * 1000
        
        return EmpathyResponse(
            response_text=base_response,
            empathy_type=EmpathyType.COMPASSIONATE,
            emotion_addressed=emotion_state.emotion,
            comfort_level=comfort_level,
            personalization_score=personalization_score,
            suggested_actions=suggested_actions,
            tone=self._determine_tone(emotion_state, request.personality_profile),
            confidence=confidence,
            generation_time_ms=generation_time,
            cultural_adaptation=request.cultural_context.value if request.cultural_context else None,
            template_used=f"compassionate_{emotion_state.emotion}",
            metadata={
                "strategy_type": "compassionate",
                "emotion_intensity": emotion_state.intensity,
                "comfort_techniques_used": self._get_applied_techniques(emotion_state),
                "hope_injection": emotion_state.valence < -0.3,
                "context_extracted": context_info
            }
        )
    
    def is_suitable(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> float:
        """判断慈悲共情策略的适合度"""
        base_score = self.get_effectiveness_score(emotion_state.emotion)
        
        # 情感强度调整 - 慈悲共情对高强度负面情感特别适合
        if emotion_state.valence < -0.5 and emotion_state.intensity > 0.7:
            base_score += 0.25
        elif emotion_state.valence < 0 and emotion_state.intensity > 0.5:
            base_score += 0.15
        
        # 个性调整
        if personality:
            # 高神经质的人在困难时期更需要慈悲共情
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            if neuroticism > 0.7:
                base_score += 0.2
                
            # 低自尊或高敏感性的人适合慈悲方法
            emotional_stability = 1 - neuroticism
            if emotional_stability < 0.4:
                base_score += 0.15
                
            # 高宜人性的人欣赏慈悲和关怀
            agreeableness = personality.emotional_traits.get("agreeableness", 0.5)
            base_score += (agreeableness - 0.5) * 0.15
        
        # 上下文调整
        if context:
            # 检查是否存在情感危机迹象
            recent_emotions = context.get_recent_emotions(5)
            negative_emotions = [e for e in recent_emotions if e.valence < -0.5]
            if len(negative_emotions) >= 3:
                base_score += 0.3  # 持续负面情感需要慈悲共情
            
            # 检查情感升级
            if context.is_emotional_escalation():
                base_score += 0.25
                
            # 避免过度重复
            recent_responses = context.get_recent_responses(2)
            compassionate_count = sum(1 for r in recent_responses if r.empathy_type == EmpathyType.COMPASSIONATE)
            if compassionate_count >= 1:
                base_score *= 0.9  # 轻微减分，但慈悲共情可以连续使用
        
        # 特殊情感的加权
        crisis_emotions = ["despair", "panic", "grief", "suicidal"]  # 危机情感
        if emotion_state.emotion in crisis_emotions:
            base_score += 0.4
        
        return min(max(base_score, 0.0), 1.0)
    
    def _extract_context_info(self, message: str, context: Optional[DialogueContext] = None) -> str:
        """提取上下文信息，专注于困难和挑战"""
        if not message:
            return "这个困难"
        
        # 慈悲共情关注困难、损失、挑战等
        difficult_contexts = {
            "失去亲人": ["死亡", "去世", "离世", "逝世", "病故", "失去", "永别"],
            "分离伤痛": ["分手", "离婚", "分别", "告别", "离开", "抛弃"],
            "健康危机": ["生病", "疾病", "住院", "手术", "诊断", "治疗", "病痛"],
            "工作困难": ["失业", "辞职", "被炒", "工作压力", "职场霸凌", "考核"],
            "经济困难": ["没钱", "破产", "债务", "贷款", "经济压力", "失业"],
            "学习挫折": ["考试失败", "成绩差", "留级", "被拒", "学习困难"],
            "人际冲突": ["争吵", "冲突", "背叛", "误解", "孤立", "被排斥"],
            "未来担忧": ["不确定", "焦虑", "担心", "恐惧", "迷茫", "绝望"],
            "自我怀疑": ["失败", "无用", "不够好", "自卑", "焦虑", "迷茫"]
        }
        
        for context_key, keywords in difficult_contexts.items():
            if any(kw in message for kw in keywords):
                return context_key
        
        return "这个挑战"
    
    def _apply_comfort_techniques(self, response: str, emotion_state: EmotionState) -> str:
        """应用安慰技巧"""
        applicable_techniques = [
            technique for technique in COMFORT_TECHNIQUES
            if technique.is_applicable(emotion_state.emotion, emotion_state.intensity)
        ]
        
        if applicable_techniques:
            # 选择最有效的技巧
            best_technique = max(applicable_techniques, key=lambda t: t.effectiveness_score)
            technique_template = random.choice(best_technique.templates)
            
            # 将技巧融入回应
            if "{emotion}" in technique_template:
                technique_response = technique_template.format(emotion=emotion_state.emotion)
                response += f" {technique_response}"
        
        return response
    
    def _adapt_for_culture(self, response: str, cultural_context: Optional[CulturalContext]) -> str:
        """文化适配"""
        if not cultural_context:
            return response
        
        if cultural_context == CulturalContext.COLLECTIVIST:
            # 集体主义文化强调家庭和社区支持
            response = response.replace("我会", "我们大家会")
            response = response.replace("你并不孤单", "你有家人朋友的支持")
            
        elif cultural_context == CulturalContext.HIGH_CONTEXT:
            # 高语境文化更加含蓄表达关怀
            response = response.replace("深深地", "")
            response = response.replace("强烈的", "")
            
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
        
        # 慈悲共情通过深度理解和定制化建议提供个性化
        if request.emotion_state and request.emotion_state.intensity > 0.7:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, emotion_state: EmotionState, context: Optional[DialogueContext]) -> float:
        """计算置信度"""
        base_confidence = 0.9  # 慈悲共情对大多数情况都有效
        
        # 基于策略有效性
        effectiveness = self.get_effectiveness_score(emotion_state.emotion)
        base_confidence = (base_confidence + effectiveness) / 2
        
        # 负面高强度情感的置信度更高
        if emotion_state.valence < -0.5 and emotion_state.intensity > 0.6:
            base_confidence += 0.05
        
        # 危机情感的置信度
        crisis_emotions = ["despair", "grief", "panic", "suicidal"]
        if emotion_state.emotion in crisis_emotions:
            base_confidence += 0.05
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _generate_suggested_actions(self, emotion_state: EmotionState, urgency_level: float = 0.5) -> List[str]:
        """生成建设性行动建议"""
        actions = []
        
        # 获取预定义的行动建议
        emotion_actions = self.action_suggestions.get(emotion_state.emotion, [])
        
        if emotion_actions:
            # 基于情感强度和紧急程度选择建议数量
            if emotion_state.intensity > 0.8 or urgency_level > 0.8:
                actions = emotion_actions[:3]  # 高强度提供更多建议
            elif emotion_state.intensity > 0.5 or urgency_level > 0.5:
                actions = emotion_actions[:2]
            else:
                actions = [random.choice(emotion_actions)]
        
        # 为特定情况添加紧急建议
        if urgency_level > 0.9 or emotion_state.emotion in ["despair", "panic", "suicidal"]:
            crisis_actions = [
                "立即联系专业心理健康服务",
                "确保有人陪伴在身边",
                "拨打心理危机干预热线"
            ]
            actions = crisis_actions + actions[:1]  # 危机建议优先
        
        # 添加通用的自我照顾建议
        if emotion_state.valence < -0.5:
            self_care_actions = [
                "确保充足的睡眠和营养",
                "进行适度的身体活动",
                "限制社交媒体和负面信息摄入"
            ]
            if len(actions) < 3:
                actions.append(random.choice(self_care_actions))
        
        return actions[:4]  # 最多返回4个建议
    
    def _get_applied_techniques(self, emotion_state: EmotionState) -> List[str]:
        """获取应用的安慰技巧列表"""
        applied_techniques = []
        
        for technique in COMFORT_TECHNIQUES:
            if technique.is_applicable(emotion_state.emotion, emotion_state.intensity):
                applied_techniques.append(technique.name)
        
        return applied_techniques
    
    def _determine_tone(self, emotion_state: EmotionState, personality: Optional[PersonalityProfile]) -> ResponseTone:
        """确定回应语调"""
        # 慈悲共情主要使用支持性和温和的语调
        tone_mapping = {
            "sadness": ResponseTone.GENTLE,
            "grief": ResponseTone.GENTLE,
            "despair": ResponseTone.GENTLE,
            "anger": ResponseTone.SUPPORTIVE,
            "frustration": ResponseTone.SUPPORTIVE,
            "fear": ResponseTone.SUPPORTIVE,
            "anxiety": ResponseTone.SUPPORTIVE,
            "panic": ResponseTone.GENTLE,
            "happiness": ResponseTone.WARM,
            "joy": ResponseTone.WARM,
            "neutral": ResponseTone.SUPPORTIVE
        }
        
        base_tone = tone_mapping.get(emotion_state.emotion, ResponseTone.SUPPORTIVE)
        
        # 基于情感强度调整
        if emotion_state.intensity > 0.8 and emotion_state.valence < -0.5:
            # 极高强度的负面情感使用最温和的语调
            base_tone = ResponseTone.GENTLE
        
        # 基于个性调整
        if personality:
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            if neuroticism > 0.8:
                # 高神经质需要更温和的方法
                base_tone = ResponseTone.GENTLE
        
        return base_tone
