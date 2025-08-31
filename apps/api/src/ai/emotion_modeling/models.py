"""
情感状态建模系统的核心数据模型
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid


class EmotionType(Enum):
    """基础情感类型枚举"""
    HAPPINESS = "happiness"
    SADNESS = "sadness" 
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    JOY = "joy"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    CONTEMPT = "contempt"
    SHAME = "shame"
    GUILT = "guilt"
    PRIDE = "pride"
    ENVY = "envy"
    LOVE = "love"
    GRATITUDE = "gratitude"
    HOPE = "hope"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"


class PersonalityTrait(Enum):
    """Big Five人格特质枚举"""
    EXTRAVERSION = "extraversion"  # 外向性
    NEUROTICISM = "neuroticism"    # 神经质
    AGREEABLENESS = "agreeableness"  # 宜人性
    CONSCIENTIOUSNESS = "conscientiousness"  # 尽责性
    OPENNESS = "openness"  # 开放性


@dataclass
class EmotionState:
    """情感状态数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    emotion: str = EmotionType.NEUTRAL.value
    intensity: float = 0.5  # 强度 [0,1]
    valence: float = 0.0    # 效价 [-1,1]
    arousal: float = 0.3    # 唤醒度 [0,1] 
    dominance: float = 0.5  # 支配性 [0,1]
    confidence: float = 1.0 # 置信度 [0,1]
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    triggers: List[str] = field(default_factory=list)  # 触发因素
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文信息
    source: str = "manual"  # 来源: manual, multimodal, text, voice等
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'emotion': self.emotion,
            'intensity': self.intensity,
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration.total_seconds() if self.duration else None,
            'triggers': self.triggers,
            'context': self.context,
            'source': self.source,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionState':
        """从字典创建实例"""
        state = cls(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data['user_id'],
            emotion=data['emotion'],
            intensity=data['intensity'],
            valence=data['valence'],
            arousal=data['arousal'],
            dominance=data['dominance'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            triggers=data.get('triggers', []),
            context=data.get('context', {}),
            source=data.get('source', 'manual'),
            session_id=data.get('session_id')
        )
        
        if data.get('duration'):
            state.duration = timedelta(seconds=data['duration'])
            
        return state
    
    def get_vad_coordinates(self) -> Tuple[float, float, float]:
        """获取VAD空间坐标"""
        return (self.valence, self.arousal, self.dominance)
    
    def is_positive(self) -> bool:
        """是否为积极情感"""
        return self.valence > 0.0
    
    def is_high_arousal(self) -> bool:
        """是否为高唤醒情感"""
        return self.arousal > 0.6


@dataclass 
class PersonalityProfile:
    """个性化情感画像"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    emotional_traits: Dict[str, float] = field(default_factory=dict)  # Big Five + 情感特质
    baseline_emotions: Dict[str, float] = field(default_factory=dict)  # 基线情感分布
    emotion_volatility: float = 0.5  # 情感波动性 [0,1]
    recovery_rate: float = 0.5  # 恢复速度 [0,1]
    dominant_emotions: List[str] = field(default_factory=list)  # 主导情感
    trigger_patterns: Dict[str, List[str]] = field(default_factory=dict)  # 触发模式
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    sample_count: int = 0  # 用于构建画像的样本数量
    confidence_score: float = 0.0  # 画像置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'emotional_traits': self.emotional_traits,
            'baseline_emotions': self.baseline_emotions,
            'emotion_volatility': self.emotion_volatility,
            'recovery_rate': self.recovery_rate,
            'dominant_emotions': self.dominant_emotions,
            'trigger_patterns': self.trigger_patterns,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'sample_count': self.sample_count,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityProfile':
        """从字典创建实例"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data['user_id'],
            emotional_traits=data.get('emotional_traits', {}),
            baseline_emotions=data.get('baseline_emotions', {}),
            emotion_volatility=data.get('emotion_volatility', 0.5),
            recovery_rate=data.get('recovery_rate', 0.5),
            dominant_emotions=data.get('dominant_emotions', []),
            trigger_patterns=data.get('trigger_patterns', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            sample_count=data.get('sample_count', 0),
            confidence_score=data.get('confidence_score', 0.0)
        )
    
    def get_trait(self, trait: PersonalityTrait) -> float:
        """获取人格特质分数"""
        return self.emotional_traits.get(trait.value, 0.5)
    
    def set_trait(self, trait: PersonalityTrait, score: float):
        """设置人格特质分数"""
        self.emotional_traits[trait.value] = max(0.0, min(1.0, score))
        self.updated_at = datetime.now()
    
    def get_dominant_emotion(self) -> Optional[str]:
        """获取最主导的情感"""
        if not self.dominant_emotions:
            return None
        return self.dominant_emotions[0]
    
    def is_high_volatility(self) -> bool:
        """是否高情感波动性"""
        return self.emotion_volatility > 0.7
    
    def is_fast_recovery(self) -> bool:
        """是否快速恢复"""
        return self.recovery_rate > 0.7


@dataclass
class EmotionTransition:
    """情感状态转换记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    from_emotion: str = ""
    to_emotion: str = ""
    transition_probability: float = 0.0  # 转换概率
    occurrence_count: int = 1  # 发生次数
    avg_duration: Optional[timedelta] = None  # 平均持续时间
    updated_at: datetime = field(default_factory=datetime.now)
    context_factors: List[str] = field(default_factory=list)  # 影响因素
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'from_emotion': self.from_emotion,
            'to_emotion': self.to_emotion,
            'transition_probability': self.transition_probability,
            'occurrence_count': self.occurrence_count,
            'avg_duration': self.avg_duration.total_seconds() if self.avg_duration else None,
            'updated_at': self.updated_at.isoformat(),
            'context_factors': self.context_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionTransition':
        """从字典创建实例"""
        transition = cls(
            id=data.get('id', str(uuid.uuid4())),
            user_id=data['user_id'],
            from_emotion=data['from_emotion'],
            to_emotion=data['to_emotion'],
            transition_probability=data['transition_probability'],
            occurrence_count=data['occurrence_count'],
            updated_at=datetime.fromisoformat(data['updated_at']),
            context_factors=data.get('context_factors', [])
        )
        
        if data.get('avg_duration'):
            transition.avg_duration = timedelta(seconds=data['avg_duration'])
            
        return transition


@dataclass
class EmotionPrediction:
    """情感预测结果"""
    user_id: str = ""
    current_emotion: str = ""
    predicted_emotions: List[Tuple[str, float]] = field(default_factory=list)  # (情感, 概率)
    confidence: float = 0.0
    time_horizon: timedelta = field(default=timedelta(hours=1))
    prediction_time: datetime = field(default_factory=datetime.now)
    factors: Dict[str, Any] = field(default_factory=dict)  # 影响预测的因素
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'current_emotion': self.current_emotion,
            'predicted_emotions': self.predicted_emotions,
            'confidence': self.confidence,
            'time_horizon': self.time_horizon.total_seconds(),
            'prediction_time': self.prediction_time.isoformat(),
            'factors': self.factors
        }
    
    def get_most_likely_emotion(self) -> Optional[Tuple[str, float]]:
        """获取最可能的情感"""
        if not self.predicted_emotions:
            return None
        return max(self.predicted_emotions, key=lambda x: x[1])


@dataclass
class EmotionStatistics:
    """情感统计信息"""
    user_id: str = ""
    time_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))
    emotion_distribution: Dict[str, float] = field(default_factory=dict)  # 情感分布
    intensity_stats: Dict[str, float] = field(default_factory=dict)  # 强度统计
    valence_stats: Dict[str, float] = field(default_factory=dict)  # 效价统计
    arousal_stats: Dict[str, float] = field(default_factory=dict)  # 唤醒度统计
    dominance_stats: Dict[str, float] = field(default_factory=dict)  # 支配性统计
    transition_counts: Dict[str, int] = field(default_factory=dict)  # 转换次数统计
    temporal_patterns: Dict[str, List[float]] = field(default_factory=dict)  # 时间模式
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'time_period': [self.time_period[0].isoformat(), self.time_period[1].isoformat()],
            'emotion_distribution': self.emotion_distribution,
            'intensity_stats': self.intensity_stats,
            'valence_stats': self.valence_stats,
            'arousal_stats': self.arousal_stats,
            'dominance_stats': self.dominance_stats,
            'transition_counts': self.transition_counts,
            'temporal_patterns': self.temporal_patterns,
            'total_samples': self.total_samples
        }