"""
共情响应生成系统的核心数据模型
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..emotion_recognition.models.emotion_models import MultiModalEmotion, EmotionDimension
from ..emotion_modeling.models import EmotionState, PersonalityProfile


class EmpathyType(Enum):
    """共情类型枚举"""
    COGNITIVE = "cognitive"      # 认知共情 - 理解情感
    AFFECTIVE = "affective"      # 情感共情 - 分享情感
    COMPASSIONATE = "compassionate"  # 慈悲共情 - 提供支持


class ResponseTone(Enum):
    """回应语调类型"""
    WARM = "warm"           # 温暖
    GENTLE = "gentle"       # 温和
    ENTHUSIASTIC = "enthusiastic"  # 热情
    PROFESSIONAL = "professional"  # 专业
    SUPPORTIVE = "supportive"      # 支持性
    UNDERSTANDING = "understanding"  # 理解性


class CulturalContext(Enum):
    """文化背景枚举"""
    COLLECTIVIST = "collectivist"      # 集体主义
    INDIVIDUALIST = "individualist"    # 个人主义
    HIGH_CONTEXT = "high_context"      # 高语境文化
    LOW_CONTEXT = "low_context"        # 低语境文化


@dataclass
class EmpathyResponse:
    """共情响应数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_text: str = ""                          # 回应文本
    empathy_type: EmpathyType = EmpathyType.COGNITIVE  # 共情类型
    emotion_addressed: str = ""                       # 处理的情感
    comfort_level: float = 0.5                       # 安慰程度 [0,1]
    personalization_score: float = 0.0              # 个性化程度 [0,1]
    suggested_actions: List[str] = field(default_factory=list)  # 建议行动
    tone: ResponseTone = ResponseTone.UNDERSTANDING   # 语调风格
    confidence: float = 1.0                          # 置信度 [0,1]
    timestamp: datetime = field(default_factory=datetime.now)  # 生成时间
    generation_time_ms: float = 0.0                 # 生成耗时(毫秒)
    cultural_adaptation: Optional[str] = None        # 文化适配信息
    template_used: Optional[str] = None              # 使用的模板
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "response_text": self.response_text,
            "empathy_type": self.empathy_type.value,
            "emotion_addressed": self.emotion_addressed,
            "comfort_level": self.comfort_level,
            "personalization_score": self.personalization_score,
            "suggested_actions": self.suggested_actions,
            "tone": self.tone.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "generation_time_ms": self.generation_time_ms,
            "cultural_adaptation": self.cultural_adaptation,
            "template_used": self.template_used,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmpathyResponse':
        """从字典创建实例"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            response_text=data["response_text"],
            empathy_type=EmpathyType(data["empathy_type"]),
            emotion_addressed=data["emotion_addressed"],
            comfort_level=data["comfort_level"],
            personalization_score=data["personalization_score"],
            suggested_actions=data.get("suggested_actions", []),
            tone=ResponseTone(data["tone"]),
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            generation_time_ms=data.get("generation_time_ms", 0.0),
            cultural_adaptation=data.get("cultural_adaptation"),
            template_used=data.get("template_used"),
            metadata=data.get("metadata", {})
        )


@dataclass
class DialogueContext:
    """对话上下文数据结构"""
    user_id: str = ""
    conversation_id: str = ""
    session_id: Optional[str] = None
    
    # 情感历史
    emotion_history: List[EmotionState] = field(default_factory=list)
    
    # 回应历史
    response_history: List[EmpathyResponse] = field(default_factory=list)
    
    # 对话状态
    current_topic: str = ""
    emotional_arc: List[str] = field(default_factory=list)  # 情感弧线
    last_strategy: Optional[EmpathyType] = None
    
    # 个性化信息
    personalization_weights: Dict[str, float] = field(default_factory=dict)
    cultural_context: Optional[CulturalContext] = None
    
    # 时间信息
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # 上下文信息
    conversation_length: int = 0
    average_response_time: float = 0.0
    
    def add_emotion(self, emotion_state: EmotionState):
        """添加情感状态"""
        self.emotion_history.append(emotion_state)
        self.emotional_arc.append(emotion_state.emotion)
        self.last_update = datetime.now()
        
    def add_response(self, response: EmpathyResponse):
        """添加回应记录"""
        self.response_history.append(response)
        self.last_strategy = response.empathy_type
        self.conversation_length += 1
        self.last_update = datetime.now()
        
        # 更新平均响应时间
        if response.generation_time_ms > 0:
            total_time = self.average_response_time * (self.conversation_length - 1) + response.generation_time_ms
            self.average_response_time = total_time / self.conversation_length
    
    def get_recent_emotions(self, count: int = 5) -> List[EmotionState]:
        """获取最近的情感状态"""
        return self.emotion_history[-count:] if self.emotion_history else []
    
    def get_recent_responses(self, count: int = 3) -> List[EmpathyResponse]:
        """获取最近的回应"""
        return self.response_history[-count:] if self.response_history else []
    
    def get_emotional_pattern(self) -> Dict[str, float]:
        """获取情感模式分布"""
        if not self.emotion_history:
            return {}
        
        emotion_counts = {}
        for emotion_state in self.emotion_history:
            emotion = emotion_state.emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_count = len(self.emotion_history)
        return {emotion: count / total_count for emotion, count in emotion_counts.items()}
    
    def is_emotional_escalation(self) -> bool:
        """检测是否存在情感升级"""
        if len(self.emotion_history) < 2:
            return False
        
        recent_emotions = self.get_recent_emotions(3)
        intensities = [e.intensity for e in recent_emotions]
        
        # 检查强度是否持续上升
        return all(intensities[i] < intensities[i + 1] for i in range(len(intensities) - 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "emotion_history": [e.to_dict() for e in self.emotion_history],
            "response_history": [r.to_dict() for r in self.response_history],
            "current_topic": self.current_topic,
            "emotional_arc": self.emotional_arc,
            "last_strategy": self.last_strategy.value if self.last_strategy else None,
            "personalization_weights": self.personalization_weights,
            "cultural_context": self.cultural_context.value if self.cultural_context else None,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "conversation_length": self.conversation_length,
            "average_response_time": self.average_response_time
        }


@dataclass
class EmpathyRequest:
    """共情响应请求数据结构"""
    user_id: str = ""
    message: str = ""
    emotion_state: Optional[EmotionState] = None
    multimodal_emotion: Optional[MultiModalEmotion] = None
    personality_profile: Optional[PersonalityProfile] = None
    dialogue_context: Optional[DialogueContext] = None
    
    # 请求配置
    preferred_empathy_type: Optional[EmpathyType] = None
    max_response_length: int = 200
    cultural_context: Optional[CulturalContext] = None
    urgency_level: float = 0.5  # 紧急程度 [0,1]
    
    # 时间约束
    max_generation_time_ms: float = 300.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "message": self.message,
            "emotion_state": self.emotion_state.to_dict() if self.emotion_state else None,
            "multimodal_emotion": self.multimodal_emotion.to_dict() if self.multimodal_emotion else None,
            "personality_profile": self.personality_profile.to_dict() if self.personality_profile else None,
            "dialogue_context": self.dialogue_context.to_dict() if self.dialogue_context else None,
            "preferred_empathy_type": self.preferred_empathy_type.value if self.preferred_empathy_type else None,
            "max_response_length": self.max_response_length,
            "cultural_context": self.cultural_context.value if self.cultural_context else None,
            "urgency_level": self.urgency_level,
            "max_generation_time_ms": self.max_generation_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ComfortTechnique:
    """安慰技巧数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    applicable_emotions: List[str] = field(default_factory=list)
    intensity_range: Tuple[float, float] = (0.0, 1.0)  # 适用强度范围
    templates: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.8  # 有效性评分
    cultural_sensitivity: Dict[str, float] = field(default_factory=dict)  # 文化敏感性
    
    def is_applicable(self, emotion: str, intensity: float) -> bool:
        """检查是否适用于特定情感和强度"""
        emotion_match = emotion in self.applicable_emotions
        intensity_match = self.intensity_range[0] <= intensity <= self.intensity_range[1]
        return emotion_match and intensity_match


# 预定义的安慰技巧库
COMFORT_TECHNIQUES = [
    ComfortTechnique(
        name="validation",
        description="验证和确认用户的情感体验",
        applicable_emotions=["sadness", "anger", "fear", "anxiety"],
        intensity_range=(0.3, 1.0),
        templates=[
            "我理解你现在{emotion}的感受，这种情绪是完全正常的。",
            "你的{emotion}是可以理解的，任何人在这种情况下都会有类似的感受。",
            "感到{emotion}并不意味着你软弱，这表明你是一个有感情的人。"
        ],
        effectiveness_score=0.9,
        cultural_sensitivity={"collectivist": 0.8, "individualist": 0.9}
    ),
    ComfortTechnique(
        name="hope_injection",
        description="注入希望和积极展望",
        applicable_emotions=["sadness", "despair", "disappointment"],
        intensity_range=(0.5, 1.0),
        templates=[
            "虽然现在很困难，但这种感受是暂时的，你有力量度过难关。",
            "我相信你能够克服这个挑战，你比想象中更坚强。",
            "每个人都会经历低谷，但这也意味着好的时光即将到来。"
        ],
        effectiveness_score=0.85,
        cultural_sensitivity={"collectivist": 0.7, "individualist": 0.9}
    ),
    ComfortTechnique(
        name="presence_assurance",
        description="提供陪伴和支持的保证",
        applicable_emotions=["loneliness", "fear", "anxiety", "sadness"],
        intensity_range=(0.4, 1.0),
        templates=[
            "你不是一个人在面对这些，我会陪伴你度过这段困难时期。",
            "无论发生什么，我都在这里支持你。",
            "你有支持你的人，包括我，我们一起面对。"
        ],
        effectiveness_score=0.88,
        cultural_sensitivity={"collectivist": 0.9, "individualist": 0.8}
    )
]