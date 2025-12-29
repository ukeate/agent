"""
共情响应质量评估器

评估生成的共情响应的质量和有效性
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .models import EmpathyResponse, EmpathyRequest, DialogueContext, EmpathyType
from ..emotion_modeling.models import EmotionState, PersonalityProfile

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class QualityMetrics:
    """质量评估指标"""
    emotional_alignment: float = 0.0      # 情感匹配度
    empathy_authenticity: float = 0.0     # 共情真实性
    response_appropriateness: float = 0.0 # 响应恰当性
    personalization_quality: float = 0.0  # 个性化质量
    linguistic_quality: float = 0.0       # 语言质量
    actionability: float = 0.0             # 行动可执行性
    cultural_sensitivity: float = 0.0      # 文化敏感性
    overall_score: float = 0.0             # 综合评分

class QualityAssessor:
    """共情响应质量评估器"""
    
    def __init__(self):
        """初始化质量评估器"""
        # 质量评估权重
        self.metric_weights = {
            "emotional_alignment": 0.25,
            "empathy_authenticity": 0.20,
            "response_appropriateness": 0.20,
            "personalization_quality": 0.15,
            "linguistic_quality": 0.10,
            "actionability": 0.05,
            "cultural_sensitivity": 0.05
        }
        
        # 负面模式检测
        self.negative_patterns = [
            r"你应该",           # 过于指导性
            r"你必须",           # 强制性语言
            r"这没什么大不了",    # 轻视情感
            r"你想多了",         # 否定感受
            r"不要这样想",       # 直接否定
            r"你错了",           # 批判性语言
            r"这很简单",         # 简化复杂情感
            r"别想了",           # 回避性建议
        ]
        
        # 积极模式检测
        self.positive_patterns = [
            r"我理解",
            r"我能感受到",
            r"这很正常",
            r"你不孤单",
            r"我陪伴你",
            r"你很坚强",
            r"这会过去",
            r"我支持你",
        ]
        
        # 共情标志词
        self.empathy_markers = [
            "理解", "感受", "体会", "感同身受", "陪伴", "支持",
            "关心", "在乎", "重视", "珍惜", "尊重", "接受"
        ]
        
        # 情感词汇表
        self.emotion_vocabulary = {
            "happiness": ["快乐", "高兴", "愉悦", "欣喜", "喜悦", "开心"],
            "sadness": ["悲伤", "难过", "痛苦", "沮丧", "伤心", "失落"],
            "anger": ["愤怒", "生气", "恼火", "愤慨", "火大", "不满"],
            "fear": ["恐惧", "害怕", "担心", "忧虑", "不安", "惊慌"],
            "surprise": ["惊讶", "意外", "震惊", "诧异", "惊奇", "始料未及"]
        }
    
    def assess_response(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> float:
        """
        评估共情响应质量
        
        Args:
            response: 生成的共情响应
            request: 原始请求
            context: 对话上下文
            
        Returns:
            float: 质量评分 [0,1]
        """
        try:
            metrics = self._calculate_quality_metrics(response, request, context)
            
            # 计算加权综合评分
            overall_score = sum(
                getattr(metrics, metric) * weight
                for metric, weight in self.metric_weights.items()
            )
            
            metrics.overall_score = min(max(overall_score, 0.0), 1.0)
            
            # 记录详细指标
            logger.debug(f"Quality assessment: {metrics}")
            
            return metrics.overall_score
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return 0.5  # 默认中等评分
    
    def get_detailed_assessment(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> QualityMetrics:
        """获取详细的质量评估"""
        return self._calculate_quality_metrics(response, request, context)
    
    def _calculate_quality_metrics(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> QualityMetrics:
        """计算各项质量指标"""
        metrics = QualityMetrics()
        
        # 1. 情感匹配度
        metrics.emotional_alignment = self._assess_emotional_alignment(response, request)
        
        # 2. 共情真实性
        metrics.empathy_authenticity = self._assess_empathy_authenticity(response)
        
        # 3. 响应恰当性
        metrics.response_appropriateness = self._assess_response_appropriateness(
            response, request, context
        )
        
        # 4. 个性化质量
        metrics.personalization_quality = self._assess_personalization_quality(
            response, request
        )
        
        # 5. 语言质量
        metrics.linguistic_quality = self._assess_linguistic_quality(response)
        
        # 6. 行动可执行性
        metrics.actionability = self._assess_actionability(response)
        
        # 7. 文化敏感性
        metrics.cultural_sensitivity = self._assess_cultural_sensitivity(response, request)
        
        return metrics
    
    def _assess_emotional_alignment(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest
    ) -> float:
        """评估情感匹配度"""
        if not request.emotion_state:
            return 0.5
        
        target_emotion = request.emotion_state.emotion
        response_text = response.response_text.lower()
        
        score = 0.0
        
        # 检查是否提到了相关情感词汇
        target_emotion_words = self.emotion_vocabulary.get(target_emotion, [])
        mentioned_emotion_words = sum(1 for word in target_emotion_words if word in response_text)
        
        if target_emotion_words:
            score += (mentioned_emotion_words / len(target_emotion_words)) * 0.4
        
        # 检查是否使用了恰当的共情表达
        if target_emotion in response_text or response.emotion_addressed == target_emotion:
            score += 0.3
        
        # 基于情感强度的语言强度匹配
        intensity = request.emotion_state.intensity
        strong_words = ["深深", "强烈", "非常", "极其", "十分"]
        mild_words = ["有点", "稍微", "轻微", "一些"]
        
        strong_word_count = sum(1 for word in strong_words if word in response_text)
        mild_word_count = sum(1 for word in mild_words if word in response_text)
        
        if intensity > 0.7 and strong_word_count > 0:
            score += 0.2
        elif intensity < 0.4 and mild_word_count > 0:
            score += 0.2
        elif 0.4 <= intensity <= 0.7 and strong_word_count == 0 and mild_word_count == 0:
            score += 0.1
        
        # 情感效价匹配
        valence = request.emotion_state.valence
        positive_markers = ["高兴", "开心", "棒", "好", "美好", "精彩"]
        negative_markers = ["难过", "痛苦", "困难", "不易", "艰难", "沉重"]
        
        positive_count = sum(1 for marker in positive_markers if marker in response_text)
        negative_count = sum(1 for marker in negative_markers if marker in response_text)
        
        if valence > 0.3 and positive_count > 0:
            score += 0.1
        elif valence < -0.3 and negative_count > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_empathy_authenticity(self, response: EmpathyResponse) -> float:
        """评估共情真实性"""
        text = response.response_text.lower()
        score = 0.0
        
        # 共情标志词检测
        empathy_count = sum(1 for marker in self.empathy_markers if marker in text)
        score += min(empathy_count * 0.15, 0.6)
        
        # 积极模式检测
        positive_matches = sum(1 for pattern in self.positive_patterns if re.search(pattern, text))
        score += min(positive_matches * 0.1, 0.3)
        
        # 负面模式惩罚
        negative_matches = sum(1 for pattern in self.negative_patterns if re.search(pattern, text))
        score -= negative_matches * 0.2
        
        # 第一人称使用（表现陪伴感）
        first_person_count = text.count("我") + text.count("我们")
        if first_person_count > 0:
            score += min(first_person_count * 0.05, 0.2)
        
        # 避免机械化表达
        mechanical_phrases = ["根据你的描述", "基于以上信息", "综合考虑"]
        mechanical_count = sum(1 for phrase in mechanical_phrases if phrase in text)
        score -= mechanical_count * 0.15
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_response_appropriateness(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> float:
        """评估响应恰当性"""
        score = 0.0
        
        # 策略选择恰当性
        if request.emotion_state:
            emotion = request.emotion_state.emotion
            strategy = response.empathy_type
            intensity = request.emotion_state.intensity
            
            # 高强度负面情感使用慈悲共情
            if emotion in ["sadness", "grief", "despair", "fear", "panic"] and intensity > 0.7:
                if strategy == EmpathyType.COMPASSIONATE:
                    score += 0.3
                else:
                    score += 0.1
            
            # 积极情感使用情感共情
            elif emotion in ["happiness", "joy", "excitement"] and intensity > 0.5:
                if strategy == EmpathyType.AFFECTIVE:
                    score += 0.3
                else:
                    score += 0.2
            
            # 中性或低强度情感使用认知共情
            elif intensity <= 0.5 or emotion == "neutral":
                if strategy == EmpathyType.COGNITIVE:
                    score += 0.3
                else:
                    score += 0.2
        
        # 响应长度恰当性
        text_length = len(response.response_text)
        if 50 <= text_length <= 200:
            score += 0.2
        elif 30 <= text_length <= 250:
            score += 0.15
        elif text_length < 20 or text_length > 300:
            score -= 0.1
        
        # 上下文一致性
        if context and context.response_history:
            last_response = context.response_history[-1] if context.response_history else None
            if last_response:
                # 避免完全重复
                similarity = self._calculate_text_similarity(
                    response.response_text, last_response.response_text
                )
                if similarity > 0.8:
                    score -= 0.2
                elif similarity < 0.3:
                    score += 0.1
        
        # 建议行动的相关性
        if response.suggested_actions:
            relevant_actions = 0
            for action in response.suggested_actions:
                if any(word in action.lower() for word in ["专业", "帮助", "支持", "时间", "休息"]):
                    relevant_actions += 1
            
            if response.suggested_actions:
                score += (relevant_actions / len(response.suggested_actions)) * 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_personalization_quality(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest
    ) -> float:
        """评估个性化质量"""
        score = response.personalization_score * 0.6  # 基础个性化评分
        
        # 个性特质体现
        if request.personality_profile:
            personality = request.personality_profile
            text = response.response_text.lower()
            
            # 外向性体现
            extraversion = personality.emotional_traits.get("extraversion", 0.5)
            interactive_phrases = ["一起", "我们", "聊聊", "分享"]
            interactive_count = sum(1 for phrase in interactive_phrases if phrase in text)
            
            if extraversion > 0.7 and interactive_count > 0:
                score += 0.15
            elif extraversion < 0.3 and interactive_count == 0:
                score += 0.1
            
            # 宜人性体现
            agreeableness = personality.emotional_traits.get("agreeableness", 0.5)
            warm_words = ["温暖", "关怀", "温柔", "贴心"]
            warm_count = sum(1 for word in warm_words if word in text)
            
            if agreeableness > 0.7 and warm_count > 0:
                score += 0.1
            
            # 神经质敏感性
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            gentle_words = ["轻柔", "温和", "缓慢", "慢慢"]
            gentle_count = sum(1 for word in gentle_words if word in text)
            
            if neuroticism > 0.7 and gentle_count > 0:
                score += 0.15
        
        # 文化适配体现
        if request.cultural_context and response.cultural_adaptation:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_linguistic_quality(self, response: EmpathyResponse) -> float:
        """评估语言质量"""
        text = response.response_text
        score = 0.8  # 基础分
        
        # 检查基本语法问题
        # 句号重复
        if ".." in text or "。。" in text:
            score -= 0.2
        
        # 感叹号过度使用
        exclamation_count = text.count("!") + text.count("！")
        if exclamation_count > 3:
            score -= 0.1
        
        # 问号过度使用
        question_count = text.count("?") + text.count("？")
        if question_count > 2:
            score -= 0.1
        
        # 重复词汇检查
        words = text.split()
        word_counts = {}
        for word in words:
            if len(word) > 1:  # 忽略单字
                word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 1
        if max_repetition > 3:
            score -= 0.2
        
        # 句子完整性
        sentences = re.split(r'[。！？.!?]', text)
        incomplete_sentences = sum(1 for s in sentences if len(s.strip()) < 3)
        if incomplete_sentences > len(sentences) * 0.3:
            score -= 0.15
        
        # 连贯性检查
        coherence_markers = ["因此", "所以", "但是", "然而", "同时", "另外", "而且"]
        coherence_count = sum(1 for marker in coherence_markers if marker in text)
        if len(sentences) > 2 and coherence_count == 0:
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _assess_actionability(self, response: EmpathyResponse) -> float:
        """评估行动可执行性"""
        if not response.suggested_actions:
            return 0.3  # 没有建议行动，给基础分
        
        score = 0.0
        actionable_count = 0
        
        for action in response.suggested_actions:
            action_lower = action.lower()
            
            # 具体可执行的行动
            if any(word in action_lower for word in ["联系", "寻找", "尝试", "练习", "记录", "写下"]):
                actionable_count += 1
                score += 0.3
            
            # 时间相关的具体建议
            elif any(word in action_lower for word in ["今天", "每天", "定期", "逐步", "慢慢"]):
                actionable_count += 1
                score += 0.25
            
            # 模糊的建议
            elif any(word in action_lower for word in ["考虑", "想想", "也许", "可能"]):
                score += 0.1
        
        # 建议数量恰当性
        if 1 <= len(response.suggested_actions) <= 3:
            score += 0.1
        elif len(response.suggested_actions) > 4:
            score -= 0.1
        
        return min(score, 1.0)
    
    def _assess_cultural_sensitivity(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest
    ) -> float:
        """评估文化敏感性"""
        score = 0.8  # 基础分，假设文化中性
        
        text = response.response_text.lower()
        
        # 避免文化偏见的表达
        biased_phrases = [
            "你们这种人", "按照我们的习惯", "在我们这里",
            "正常人都", "一般人会", "大家都"
        ]
        
        biased_count = sum(1 for phrase in biased_phrases if phrase in text)
        score -= biased_count * 0.3
        
        # 包容性语言
        inclusive_phrases = ["每个人都", "不同的人", "各自的", "你的方式"]
        inclusive_count = sum(1 for phrase in inclusive_phrases if phrase in text)
        score += inclusive_count * 0.1
        
        # 如果有文化适配，加分
        if request.cultural_context and response.cultural_adaptation:
            score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新评估权重"""
        # 验证权重合法性
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights do not sum to 1.0: {total_weight}")
            return False
        
        self.metric_weights.update(new_weights)
        logger.info(f"Updated quality assessment weights: {new_weights}")
        return True
    
    def add_negative_pattern(self, pattern: str):
        """添加负面模式"""
        self.negative_patterns.append(pattern)
        logger.info(f"Added negative pattern: {pattern}")
    
    def add_positive_pattern(self, pattern: str):
        """添加积极模式"""
        self.positive_patterns.append(pattern)
        logger.info(f"Added positive pattern: {pattern}")
