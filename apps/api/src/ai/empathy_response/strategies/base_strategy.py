"""
共情策略抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models import EmpathyResponse, EmpathyRequest, DialogueContext, EmpathyType
from ...emotion_modeling.models import EmotionState, PersonalityProfile


class EmpathyStrategy(ABC):
    """共情策略抽象基类"""
    
    def __init__(self, strategy_type: EmpathyType):
        self.strategy_type = strategy_type
        self.effectiveness_scores: Dict[str, float] = {}  # 针对不同情感的有效性评分
        
    @abstractmethod
    def generate_response(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """
        生成共情响应
        
        Args:
            request: 共情请求
            context: 对话上下文
            
        Returns:
            EmpathyResponse: 生成的共情响应
        """
        pass
    
    @abstractmethod
    def is_suitable(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> float:
        """
        判断策略是否适合当前情况
        
        Args:
            emotion_state: 当前情感状态
            personality: 个性画像
            context: 对话上下文
            
        Returns:
            float: 适合度评分 [0,1]
        """
        pass
    
    def calculate_comfort_level(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None
    ) -> float:
        """
        计算安慰程度
        
        Args:
            emotion_state: 情感状态
            personality: 个性画像
            
        Returns:
            float: 安慰程度 [0,1]
        """
        base_comfort = 0.5
        
        # 基于情感强度调整
        if emotion_state.intensity > 0.7:
            base_comfort += 0.2
        elif emotion_state.intensity < 0.3:
            base_comfort -= 0.1
            
        # 基于情感效价调整
        if emotion_state.valence < -0.5:  # 负面情感
            base_comfort += 0.2
        elif emotion_state.valence > 0.5:  # 正面情感
            base_comfort += 0.1
            
        # 基于个性特质调整
        if personality:
            neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
            if neuroticism > 0.7:  # 高神经质需要更多安慰
                base_comfort += 0.15
                
        return min(max(base_comfort, 0.0), 1.0)
    
    def adapt_for_personality(
        self,
        base_response: str,
        personality: PersonalityProfile
    ) -> str:
        """
        基于个性特质调整回应
        
        Args:
            base_response: 基础回应
            personality: 个性画像
            
        Returns:
            str: 调整后的回应
        """
        response = base_response
        
        # 基于外向性调整
        extraversion = personality.emotional_traits.get("extraversion", 0.5)
        if extraversion > 0.7:
            response += " 想和我多聊聊这个话题吗？"
        elif extraversion < 0.3:
            response = response.replace("我们", "我")  # 降低社交性
            
        # 基于宜人性调整温暖度
        agreeableness = personality.emotional_traits.get("agreeableness", 0.5)
        if agreeableness > 0.7:
            response = response.replace("理解", "深深理解")
            response = response.replace("感受", "真切感受")
            
        # 基于神经质调整敏感度
        neuroticism = personality.emotional_traits.get("neuroticism", 0.5)
        if neuroticism > 0.7:
            response = response.replace("确实", "可能")
            response = response.replace("一定", "或许")
            
        # 基于开放性调整表达创新性
        openness = personality.emotional_traits.get("openness", 0.5)
        if openness > 0.7:
            # 可以添加更有创意的表达
            pass
            
        return response
    
    def get_effectiveness_score(self, emotion: str) -> float:
        """获取对特定情感的有效性评分"""
        return self.effectiveness_scores.get(emotion, 0.5)
    
    def update_effectiveness(self, emotion: str, score: float):
        """更新有效性评分"""
        self.effectiveness_scores[emotion] = score