"""
Social Emotional Understanding System - Group Emotion Models
群体情感理解系统的核心数据模型
"""

from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
from enum import Enum

class EmotionContagionType(Enum):
    """情感传染类型"""
    VIRAL = "viral"  # 病毒式传播
    CASCADE = "cascade"  # 级联传播
    AMPLIFICATION = "amplification"  # 放大效应
    DAMPENING = "dampening"  # 衰减效应

class GroupCohesionLevel(Enum):
    """群体凝聚力水平"""
    HIGH = "high"  # 高凝聚力
    MEDIUM = "medium"  # 中等凝聚力
    LOW = "low"  # 低凝聚力
    FRAGMENTED = "fragmented"  # 分化状态

@dataclass
class EmotionState:
    """个体情感状态"""
    participant_id: str
    emotion: str  # 主要情感
    intensity: float  # 强度 [0,1]
    valence: float  # 效价 [-1,1] 
    arousal: float  # 唤醒度 [0,1]
    dominance: float  # 支配度 [0,1]
    timestamp: datetime
    confidence: float = 0.8  # 识别置信度

@dataclass
class ContagionPattern:
    """情感传染模式"""
    source_participant: str  # 传染源
    target_participants: List[str]  # 传染目标
    emotion: str  # 传染的情感
    contagion_type: EmotionContagionType  # 传染类型
    strength: float  # 传染强度 [0,1]
    propagation_speed: float  # 传播速度 (msg/min)
    timestamp: datetime
    duration_seconds: int  # 持续时间

@dataclass
class EmotionalLeader:
    """情感领导者"""
    participant_id: str
    influence_score: float  # 影响力分数 [0,1]
    leadership_type: str  # 领导类型 (positive/negative/neutral)
    influenced_participants: List[str]  # 被影响的参与者
    dominant_emotions: List[str]  # 主导的情感类型
    consistency_score: float  # 一致性分数 [0,1]

@dataclass
class GroupEmotionalState:
    """群体情感状态"""
    group_id: str
    timestamp: datetime
    participants: List[str]
    
    # 情感分布
    dominant_emotion: str  # 主导情感
    emotion_distribution: Dict[str, float]  # 情感分布
    
    # 群体动态指标
    consensus_level: float  # 共识水平 [0,1]
    polarization_index: float  # 极化指数 [0,1]
    emotional_volatility: float  # 情感波动性 [0,1]
    group_cohesion: GroupCohesionLevel  # 群体凝聚力
    
    # 影响力分析
    emotional_leaders: List[EmotionalLeader]  # 情感领导者
    influence_network: Dict[str, List[str]]  # 影响网络
    
    # 传染分析
    contagion_patterns: List[ContagionPattern]  # 传染模式
    contagion_velocity: float  # 传染速度
    
    # 趋势预测
    trend_prediction: str  # 趋势预测 (escalating/stable/declining)
    stability_score: float  # 稳定性分数 [0,1]
    
    # 元数据
    analysis_confidence: float = 0.8  # 分析置信度
    data_completeness: float = 1.0  # 数据完整性

@dataclass
class GroupEmotionHistory:
    """群体情感历史记录"""
    group_id: str
    emotional_states: List[GroupEmotionalState]  # 按时间排序的状态记录
    emotion_transitions: List[Dict[str, Any]]  # 情感转换记录
    critical_events: List[Dict[str, Any]]  # 关键事件记录
    
    def get_emotion_trend(self, emotion: str, time_window_minutes: int = 60) -> List[float]:
        """获取特定情感的趋势变化"""
        recent_states = [
            state for state in self.emotional_states
            if (utc_now() - state.timestamp).seconds <= time_window_minutes * 60
        ]
        
        return [
            state.emotion_distribution.get(emotion, 0.0) 
            for state in recent_states
        ]
    
    def get_polarization_trend(self, time_window_minutes: int = 60) -> List[float]:
        """获取极化指数趋势"""
        recent_states = [
            state for state in self.emotional_states
            if (utc_now() - state.timestamp).seconds <= time_window_minutes * 60
        ]
        
        return [state.polarization_index for state in recent_states]

@dataclass
class GroupEmotionAnalysisConfig:
    """群体情感分析配置"""
    # 权重配置
    intensity_weight: float = 0.4  # 情感强度权重
    frequency_weight: float = 0.3  # 频率权重
    influence_weight: float = 0.3  # 影响力权重
    
    # 阈值配置
    consensus_threshold: float = 0.7  # 共识阈值
    polarization_threshold: float = 0.6  # 极化阈值
    contagion_threshold: float = 0.5  # 传染阈值
    leader_influence_threshold: float = 0.6  # 领导者影响力阈值
    
    # 时间窗口配置
    analysis_window_minutes: int = 30  # 分析时间窗口
    trend_prediction_minutes: int = 10  # 趋势预测窗口
    
    # 参与者配置
    min_participants: int = 2  # 最少参与者数量
    max_participants: int = 50  # 最多参与者数量
    
    # 质量控制
    min_data_points: int = 5  # 最少数据点
    confidence_threshold: float = 0.6  # 最低置信度阈值

@dataclass
class EmotionContagionEvent:
    """情感传染事件"""
    event_id: str
    group_id: str
    timestamp: datetime
    
    # 传染详情
    source_participant: str
    source_emotion: str
    source_intensity: float
    
    # 传播路径
    propagation_path: List[str]  # 传播路径
    affected_participants: List[str]  # 受影响的参与者
    
    # 传染效果
    intensity_amplification: float  # 强度放大系数
    reach_percentage: float  # 到达率 [0,1]
    conversion_rate: float  # 转换率 [0,1]
    
    # 时间特征
    propagation_time_seconds: int  # 传播时间
    peak_intensity_time: datetime  # 峰值强度时间
    decay_time_seconds: int  # 衰减时间
    
    # 分析结果
    contagion_type: EmotionContagionType
    effectiveness_score: float  # 有效性分数 [0,1]
    
    def calculate_contagion_velocity(self) -> float:
        """计算传染速度 (participants/minute)"""
        if self.propagation_time_seconds == 0:
            return 0.0
        return len(self.affected_participants) / (self.propagation_time_seconds / 60)

@dataclass
class GroupEmotionInsight:
    """群体情感洞察"""
    group_id: str
    timestamp: datetime
    
    # 关键发现
    key_findings: List[str]  # 关键发现
    emotional_climate: str  # 情感氛围描述
    dominant_patterns: List[str]  # 主导模式
    
    # 风险提醒
    risk_alerts: List[str]  # 风险提醒
    intervention_suggestions: List[str]  # 干预建议
    
    # 预测分析
    short_term_forecast: str  # 短期预测 (下5-10分钟)
    stability_assessment: str  # 稳定性评估
    
    # 量化指标
    overall_health_score: float  # 整体健康分数 [0,1]
    engagement_level: float  # 参与度水平 [0,1]
    emotional_balance: float  # 情感平衡度 [0,1]

def generate_group_id() -> str:
    """生成群体ID"""
    return f"group_{uuid.uuid4().hex[:8]}"

def generate_event_id() -> str:
    """生成事件ID"""
    return f"event_{uuid.uuid4().hex[:8]}"
