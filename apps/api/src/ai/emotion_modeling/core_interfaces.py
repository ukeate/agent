"""
情感智能系统核心接口定义
集成Story 11.1-11.6的所有情感智能模块
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field

class EmotionType(str, Enum):
    """基础情感类型"""
    HAPPINESS = "happiness"
    SADNESS = "sadness" 
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

class ModalityType(str, Enum):
    """多模态输入类型"""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    PHYSIOLOGICAL = "physiological"

class RiskLevel(str, Enum):
    """风险评估级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EmotionModelingInterface(ABC):
    """情感建模模块基础接口（标记接口）"""
    ...

@dataclass
class EmotionState:
    """情感状态表示"""
    emotion: EmotionType
    intensity: float  # 0.0-1.0
    valence: float    # -1.0 to 1.0 (negative to positive)
    arousal: float    # 0.0-1.0 (calm to excited)
    dominance: float  # 0.0-1.0 (submissive to dominant)
    confidence: float # 0.0-1.0
    timestamp: datetime

@dataclass
class MultiModalEmotion:
    """多模态情感识别结果"""
    emotions: Dict[ModalityType, EmotionState]
    fused_emotion: EmotionState
    confidence: float
    processing_time: float

@dataclass
class PersonalityProfile:
    """个性特征画像"""
    openness: float          # 开放性
    conscientiousness: float # 责任心
    extraversion: float      # 外向性
    agreeableness: float     # 宜人性
    neuroticism: float       # 神经质
    updated_at: datetime

@dataclass
class EmpathyResponse:
    """共情响应"""
    message: str
    response_type: str       # "supportive", "encouraging", "understanding"
    confidence: float
    generation_strategy: str

@dataclass
class EmotionalMemory:
    """情感记忆"""
    memory_id: str
    content: str
    emotional_context: EmotionState
    importance: float
    created_at: datetime
    last_accessed: datetime

@dataclass
class DecisionContext:
    """决策上下文"""
    decision_type: str
    factors: Dict[str, Any]
    emotional_weight: float
    rational_weight: float

@dataclass
class RiskAssessment:
    """风险评估"""
    level: RiskLevel
    factors: List[str]
    confidence: float
    intervention_needed: bool
    recommendations: List[str]

@dataclass
class SocialContext:
    """社交情感上下文"""
    participants: List[str]
    relationship_dynamics: Dict[str, float]
    cultural_factors: List[str]
    communication_style: str

@dataclass
class GroupEmotionalState:
    """群体情感状态"""
    group_emotion: EmotionState
    individual_emotions: Dict[str, EmotionState]
    consensus_level: float
    conflict_indicators: List[str]

class UnifiedEmotionalData(BaseModel):
    """统一情感数据格式"""
    user_id: str
    timestamp: datetime
    
    # Story 11.1 - 多模态情感识别
    recognition_result: Optional[MultiModalEmotion] = None
    
    # Story 11.2 - 情感状态建模
    emotional_state: Optional[EmotionState] = None
    personality_profile: Optional[PersonalityProfile] = None
    
    # Story 11.3 - 共情响应
    empathy_response: Optional[EmpathyResponse] = None
    
    # Story 11.4 - 情感记忆
    emotional_memory: Optional[EmotionalMemory] = None
    memory_relevance: Optional[float] = None
    
    # Story 11.5 - 智能决策
    decision_context: Optional[DecisionContext] = None
    risk_assessment: Optional[RiskAssessment] = None
    
    # Story 11.6 - 社交情感
    social_context: Optional[SocialContext] = None
    group_emotion: Optional[GroupEmotionalState] = None
    
    # 统一元数据
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    data_quality: float = Field(ge=0.0, le=1.0)

class EmotionalIntelligenceResponse(BaseModel):
    """统一API响应格式"""
    success: bool
    data: Optional[UnifiedEmotionalData] = None
    error: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# 核心接口定义

class EmotionRecognitionEngine(ABC):
    """多模态情感识别引擎接口"""
    
    @abstractmethod
    async def recognize_emotion(
        self, 
        input_data: Dict[ModalityType, Any]
    ) -> MultiModalEmotion:
        """识别多模态情感"""
        ...
    
    @abstractmethod
    async def get_recognition_quality(self) -> Dict[str, float]:
        """获取识别质量指标"""
        ...

class EmotionStateModeler(ABC):
    """情感状态建模器接口"""
    
    @abstractmethod
    async def update_emotional_state(
        self, 
        user_id: str, 
        new_emotion: EmotionState
    ) -> EmotionState:
        """更新情感状态"""
        ...
    
    @abstractmethod
    async def get_personality_profile(self, user_id: str) -> PersonalityProfile:
        """获取个性画像"""
        ...
    
    @abstractmethod
    async def predict_emotional_trajectory(
        self, 
        user_id: str, 
        horizon: int
    ) -> List[EmotionState]:
        """预测情感轨迹"""
        ...

class EmpathyResponseGenerator(ABC):
    """共情响应生成器接口"""
    
    @abstractmethod
    async def generate_empathy_response(
        self, 
        emotional_state: EmotionState,
        context: Optional[str] = None
    ) -> EmpathyResponse:
        """生成共情响应"""
        ...
    
    @abstractmethod
    async def evaluate_response_quality(
        self, 
        response: EmpathyResponse,
        feedback: Dict[str, Any]
    ) -> float:
        """评估响应质量"""
        ...

class EmotionalMemoryManager(ABC):
    """情感记忆管理器接口"""
    
    @abstractmethod
    async def store_emotional_memory(
        self, 
        user_id: str,
        memory: EmotionalMemory
    ) -> str:
        """存储情感记忆"""
        ...
    
    @abstractmethod
    async def retrieve_relevant_memories(
        self, 
        user_id: str,
        current_state: EmotionState,
        limit: int = 5
    ) -> List[EmotionalMemory]:
        """检索相关记忆"""
        ...
    
    @abstractmethod
    async def manage_memory_lifecycle(self, user_id: str) -> Dict[str, int]:
        """管理记忆生命周期"""
        ...

class EmotionalIntelligenceDecisionEngine(ABC):
    """情感智能决策引擎接口"""
    
    @abstractmethod
    async def make_decision(
        self, 
        context: DecisionContext,
        emotional_state: EmotionState
    ) -> Dict[str, Any]:
        """做出情感智能决策"""
        ...
    
    @abstractmethod
    async def assess_risk(
        self, 
        user_id: str,
        emotional_state: EmotionState,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """评估风险"""
        ...

class SocialEmotionalAnalyzer(ABC):
    """社交情感分析器接口"""
    
    @abstractmethod
    async def analyze_social_dynamics(
        self, 
        participants: List[str],
        conversation_data: Dict[str, Any]
    ) -> SocialContext:
        """分析社交动态"""
        ...
    
    @abstractmethod
    async def detect_group_emotion(
        self, 
        individual_emotions: Dict[str, EmotionState]
    ) -> GroupEmotionalState:
        """检测群体情感"""
        ...
    
    @abstractmethod
    async def provide_cultural_adaptation(
        self, 
        content: str,
        cultural_context: List[str]
    ) -> str:
        """提供文化适配"""
        ...

class EmotionalIntelligenceSystem(ABC):
    """统一情感智能系统接口"""
    
    def __init__(self):
        self.recognition_engine: Optional[EmotionRecognitionEngine] = None
        self.state_modeler: Optional[EmotionStateModeler] = None
        self.empathy_generator: Optional[EmpathyResponseGenerator] = None
        self.memory_manager: Optional[EmotionalMemoryManager] = None
        self.decision_engine: Optional[EmotionalIntelligenceDecisionEngine] = None
        self.social_analyzer: Optional[SocialEmotionalAnalyzer] = None
    
    @abstractmethod
    async def initialize_system(self) -> bool:
        """初始化系统"""
        ...
    
    @abstractmethod
    async def process_emotional_interaction(
        self, 
        user_id: str,
        input_data: Dict[str, Any]
    ) -> EmotionalIntelligenceResponse:
        """处理情感交互"""
        ...
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        ...
    
    @abstractmethod
    async def shutdown_system(self) -> bool:
        """关闭系统"""
        ...

# 数据流管理器接口
class EmotionalDataFlowManager(ABC):
    """情感数据流管理器"""
    
    @abstractmethod
    async def route_data(
        self, 
        data: UnifiedEmotionalData
    ) -> Dict[str, Any]:
        """路由数据到相应模块"""
        ...
    
    @abstractmethod
    async def validate_data_integrity(
        self, 
        data: UnifiedEmotionalData
    ) -> bool:
        """验证数据完整性"""
        ...
    
    @abstractmethod
    async def synchronize_modules(self) -> bool:
        """同步各模块状态"""
        ...

# 系统监控接口
class EmotionalSystemMonitor(ABC):
    """情感系统监控器"""
    
    @abstractmethod
    async def collect_performance_metrics(self) -> Dict[str, float]:
        """收集性能指标"""
        ...
    
    @abstractmethod
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        ...
    
    @abstractmethod
    async def generate_health_report(self) -> Dict[str, Any]:
        """生成健康报告"""
        ...
    
    @abstractmethod
    async def trigger_alerts(self, alert_data: Dict[str, Any]) -> bool:
        """触发告警"""
        ...
