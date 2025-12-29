"""
共情策略选择器

基于情感状态、个性画像和上下文，智能选择最适合的共情策略
"""

from typing import Dict, List, Tuple, Optional
from .models import DialogueContext, EmpathyType
from .strategies.base_strategy import EmpathyStrategy
from .strategies.cognitive_strategy import CognitiveEmpathyStrategy
from .strategies.affective_strategy import AffectiveEmpathyStrategy
from .strategies.compassionate_strategy import CompassionateEmpathyStrategy
from ..emotion_modeling.models import EmotionState, PersonalityProfile

from src.core.logging import get_logger
logger = get_logger(__name__)

class StrategySelector:
    """共情策略选择器"""
    
    def __init__(self):
        # 初始化所有策略
        self.strategies: Dict[EmpathyType, EmpathyStrategy] = {
            EmpathyType.COGNITIVE: CognitiveEmpathyStrategy(),
            EmpathyType.AFFECTIVE: AffectiveEmpathyStrategy(),
            EmpathyType.COMPASSIONATE: CompassionateEmpathyStrategy()
        }
        
        # 策略权重配置
        self.strategy_weights = {
            EmpathyType.COGNITIVE: 1.0,
            EmpathyType.AFFECTIVE: 1.0,
            EmpathyType.COMPASSIONATE: 1.0
        }
        
        # 情感-策略映射的基础权重
        self.emotion_strategy_affinities = {
            # 悲伤类 - 慈悲共情最适合
            "sadness": {EmpathyType.COMPASSIONATE: 1.2, EmpathyType.AFFECTIVE: 1.0, EmpathyType.COGNITIVE: 0.8},
            "grief": {EmpathyType.COMPASSIONATE: 1.3, EmpathyType.AFFECTIVE: 1.1, EmpathyType.COGNITIVE: 0.7},
            "despair": {EmpathyType.COMPASSIONATE: 1.4, EmpathyType.AFFECTIVE: 0.9, EmpathyType.COGNITIVE: 0.6},
            "disappointment": {EmpathyType.COMPASSIONATE: 1.1, EmpathyType.COGNITIVE: 1.0, EmpathyType.AFFECTIVE: 0.9},
            
            # 快乐类 - 情感共情最适合
            "happiness": {EmpathyType.AFFECTIVE: 1.3, EmpathyType.COGNITIVE: 0.9, EmpathyType.COMPASSIONATE: 0.8},
            "joy": {EmpathyType.AFFECTIVE: 1.4, EmpathyType.COGNITIVE: 0.8, EmpathyType.COMPASSIONATE: 0.7},
            "excitement": {EmpathyType.AFFECTIVE: 1.3, EmpathyType.COGNITIVE: 0.9, EmpathyType.COMPASSIONATE: 0.8},
            
            # 愤怒类 - 需要平衡方法
            "anger": {EmpathyType.COMPASSIONATE: 1.2, EmpathyType.COGNITIVE: 1.1, EmpathyType.AFFECTIVE: 0.7},
            "frustration": {EmpathyType.COMPASSIONATE: 1.1, EmpathyType.COGNITIVE: 1.0, EmpathyType.AFFECTIVE: 0.8},
            
            # 恐惧类 - 慈悲共情和认知共情
            "fear": {EmpathyType.COMPASSIONATE: 1.3, EmpathyType.COGNITIVE: 1.1, EmpathyType.AFFECTIVE: 0.8},
            "anxiety": {EmpathyType.COMPASSIONATE: 1.2, EmpathyType.COGNITIVE: 1.1, EmpathyType.AFFECTIVE: 0.7},
            "panic": {EmpathyType.COMPASSIONATE: 1.4, EmpathyType.COGNITIVE: 0.8, EmpathyType.AFFECTIVE: 0.6},
            
            # 中性和其他 - 认知共情
            "neutral": {EmpathyType.COGNITIVE: 1.2, EmpathyType.AFFECTIVE: 0.6, EmpathyType.COMPASSIONATE: 0.8},
            "surprise": {EmpathyType.AFFECTIVE: 1.1, EmpathyType.COGNITIVE: 1.0, EmpathyType.COMPASSIONATE: 0.9}
        }
    
    def select_best_strategy(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None,
        preferred_type: Optional[EmpathyType] = None
    ) -> EmpathyStrategy:
        """
        选择最适合的共情策略
        
        Args:
            emotion_state: 当前情感状态
            personality: 个性画像
            context: 对话上下文
            preferred_type: 用户偏好的策略类型
            
        Returns:
            EmpathyStrategy: 选定的策略
        """
        try:
            # 如果有明确偏好且合理，优先使用
            if preferred_type and self._validate_preference(preferred_type, emotion_state):
                logger.info(f"Using preferred strategy: {preferred_type.value}")
                return self.strategies[preferred_type]
            
            # 计算各策略的综合评分
            strategy_scores = self._calculate_strategy_scores(
                emotion_state, personality, context
            )
            
            # 选择评分最高的策略
            best_strategy_type = max(strategy_scores.items(), key=lambda x: x[1])[0]
            best_strategy = self.strategies[best_strategy_type]
            
            logger.info(f"Selected strategy: {best_strategy_type.value}, scores: {strategy_scores}")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error in strategy selection: {e}")
            # 默认返回认知共情策略
            return self.strategies[EmpathyType.COGNITIVE]
    
    def get_strategy_rankings(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> List[Tuple[EmpathyType, float]]:
        """
        获取策略排名
        
        Returns:
            List[Tuple[EmpathyType, float]]: 按评分排序的策略列表
        """
        strategy_scores = self._calculate_strategy_scores(
            emotion_state, personality, context
        )
        
        return sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_strategy_scores(
        self,
        emotion_state: EmotionState,
        personality: Optional[PersonalityProfile] = None,
        context: Optional[DialogueContext] = None
    ) -> Dict[EmpathyType, float]:
        """计算各策略的综合评分"""
        scores = {}
        
        for strategy_type, strategy in self.strategies.items():
            # 基础适合度评分
            base_score = strategy.is_suitable(emotion_state, personality, context)
            
            # 情感亲和性加权
            affinity_weight = self._get_emotion_affinity(emotion_state.emotion, strategy_type)
            weighted_score = base_score * affinity_weight
            
            # 全局策略权重
            final_score = weighted_score * self.strategy_weights[strategy_type]
            
            scores[strategy_type] = final_score
        
        return scores
    
    def _get_emotion_affinity(self, emotion: str, strategy_type: EmpathyType) -> float:
        """获取情感对策略的亲和性权重"""
        emotion_affinities = self.emotion_strategy_affinities.get(emotion, {})
        return emotion_affinities.get(strategy_type, 1.0)
    
    def _validate_preference(self, preferred_type: EmpathyType, emotion_state: EmotionState) -> bool:
        """验证用户偏好是否合理"""
        # 检查是否存在明显不合适的组合
        unsuitable_combinations = [
            # 情感共情对极度负面情感可能不太合适
            (EmpathyType.AFFECTIVE, ["despair", "panic", "suicidal"]),
            # 认知共情对极度情感化的情况可能不够
            (EmpathyType.COGNITIVE, ["grief", "panic"] if emotion_state.intensity > 0.9 else [])
        ]
        
        for strategy_type, unsuitable_emotions in unsuitable_combinations:
            if preferred_type == strategy_type and emotion_state.emotion in unsuitable_emotions:
                return False
        
        return True
    
    def update_strategy_weights(self, strategy_type: EmpathyType, new_weight: float):
        """更新策略权重"""
        if 0.0 <= new_weight <= 2.0:  # 合理的权重范围
            self.strategy_weights[strategy_type] = new_weight
            logger.info(f"Updated {strategy_type.value} weight to {new_weight}")
    
    def get_strategy(self, strategy_type: EmpathyType) -> Optional[EmpathyStrategy]:
        """获取指定类型的策略"""
        return self.strategies.get(strategy_type)
    
    def add_custom_affinity(self, emotion: str, affinities: Dict[EmpathyType, float]):
        """添加自定义的情感-策略亲和性"""
        self.emotion_strategy_affinities[emotion] = affinities
        logger.info(f"Added custom affinity for emotion: {emotion}")
    
    def analyze_context_patterns(self, context: DialogueContext) -> Dict[str, any]:
        """分析对话上下文模式"""
        if not context or not context.response_history:
            return {"pattern": "no_history", "recommendation": "use_adaptive"}
        
        recent_strategies = [r.empathy_type for r in context.get_recent_responses(5)]
        strategy_distribution = {}
        
        for strategy in recent_strategies:
            strategy_distribution[strategy.value] = strategy_distribution.get(strategy.value, 0) + 1
        
        # 检测模式
        patterns = {}
        
        # 策略多样性
        diversity_score = len(set(recent_strategies)) / len(recent_strategies) if recent_strategies else 0
        patterns["diversity"] = diversity_score
        
        # 是否存在策略固化
        if recent_strategies and len(recent_strategies) >= 3:
            most_used = max(strategy_distribution.items(), key=lambda x: x[1])
            if most_used[1] >= len(recent_strategies) * 0.8:
                patterns["stagnation"] = most_used[0]
        
        # 情感轨迹分析
        if context.emotion_history:
            recent_emotions = context.get_recent_emotions(5)
            emotion_trend = self._analyze_emotion_trend(recent_emotions)
            patterns["emotion_trend"] = emotion_trend
        
        return patterns
    
    def _analyze_emotion_trend(self, emotions: List[EmotionState]) -> str:
        """分析情感趋势"""
        if len(emotions) < 2:
            return "insufficient_data"
        
        # 计算效价和强度的变化
        valence_changes = []
        intensity_changes = []
        
        for i in range(1, len(emotions)):
            valence_changes.append(emotions[i].valence - emotions[i-1].valence)
            intensity_changes.append(emotions[i].intensity - emotions[i-1].intensity)
        
        avg_valence_change = sum(valence_changes) / len(valence_changes)
        avg_intensity_change = sum(intensity_changes) / len(intensity_changes)
        
        if avg_valence_change > 0.1:
            return "improving"
        elif avg_valence_change < -0.1:
            return "deteriorating"
        elif avg_intensity_change > 0.1:
            return "escalating"
        elif avg_intensity_change < -0.1:
            return "calming"
        else:
            return "stable"
    
    def get_strategy_recommendations(
        self,
        emotion_state: EmotionState,
        context: Optional[DialogueContext] = None
    ) -> Dict[str, any]:
        """获取策略推荐和分析"""
        rankings = self.get_strategy_rankings(emotion_state, context=context)
        context_patterns = self.analyze_context_patterns(context) if context else {}
        
        recommendations = {
            "primary_strategy": rankings[0][0].value,
            "confidence": rankings[0][1],
            "alternatives": [(s.value, score) for s, score in rankings[1:]],
            "context_analysis": context_patterns
        }
        
        # 基于上下文模式的特殊建议
        if "stagnation" in context_patterns:
            recommendations["warning"] = f"Strategy stagnation detected: {context_patterns['stagnation']}"
            recommendations["suggestion"] = "Consider diversifying strategy selection"
        
        if context_patterns.get("emotion_trend") == "deteriorating":
            recommendations["urgent"] = "Emotional state deteriorating, consider compassionate approach"
        
        return recommendations
