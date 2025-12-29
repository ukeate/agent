"""
Relationship Dynamics Models
人际关系动态模型

This module provides data models for analyzing interpersonal relationship dynamics:
- Relationship types and classifications
- Emotional support patterns
- Conflict and harmony detection
- Relationship health metrics
"""

from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import uuid

class RelationshipType(Enum):
    """关系类型"""
    ROMANTIC = "romantic"  # 恋人关系
    FAMILY = "family"  # 家庭关系
    FRIENDSHIP = "friendship"  # 友谊关系
    PROFESSIONAL = "professional"  # 职业关系
    MENTORSHIP = "mentorship"  # 师徒关系
    ACQUAINTANCE = "acquaintance"  # 熟人关系
    STRANGER = "stranger"  # 陌生人关系

class IntimacyLevel(Enum):
    """亲密程度等级"""
    VERY_HIGH = "very_high"  # 极高亲密度 (0.8-1.0)
    HIGH = "high"  # 高亲密度 (0.6-0.8)
    MEDIUM = "medium"  # 中等亲密度 (0.4-0.6)
    LOW = "low"  # 低亲密度 (0.2-0.4)
    VERY_LOW = "very_low"  # 极低亲密度 (0.0-0.2)

class PowerDynamics(Enum):
    """权力动态"""
    DOMINANT = "dominant"  # 支配型 (0.5-1.0)
    BALANCED = "balanced"  # 平衡型 (-0.2-0.2)
    SUBMISSIVE = "submissive"  # 从属型 (-1.0--0.5)

class SupportType(Enum):
    """支持类型"""
    EMOTIONAL = "emotional"  # 情感支持
    INFORMATIONAL = "informational"  # 信息支持
    INSTRUMENTAL = "instrumental"  # 工具性支持
    APPRAISAL = "appraisal"  # 评价支持

class ConflictStyle(Enum):
    """冲突风格"""
    COMPETING = "competing"  # 竞争型
    COLLABORATING = "collaborating"  # 合作型
    COMPROMISING = "compromising"  # 妥协型
    AVOIDING = "avoiding"  # 回避型
    ACCOMMODATING = "accommodating"  # 迁就型

@dataclass
class EmotionalSupportPattern:
    """情感支持模式"""
    support_id: str
    giver_id: str
    receiver_id: str
    support_type: SupportType
    frequency: int  # 支持频次
    intensity: float  # 支持强度 [0,1]
    reciprocity_score: float  # 互惠性分数 [0,1]
    effectiveness_score: float  # 有效性分数 [0,1]
    timestamp: datetime
    
    # 支持行为特征
    verbal_affirmation: bool = False  # 言语肯定
    active_listening: bool = False  # 积极倾听
    empathy_expression: bool = False  # 同理心表达
    problem_solving: bool = False  # 问题解决
    resource_sharing: bool = False  # 资源分享

@dataclass
class ConflictIndicator:
    """冲突指标"""
    indicator_id: str
    participants: List[str]
    conflict_type: str  # 冲突类型
    severity_level: float  # 严重程度 [0,1]
    escalation_risk: float  # 升级风险 [0,1]
    resolution_potential: float  # 解决潜力 [0,1]
    timestamp: datetime
    
    # 冲突表现
    verbal_disagreement: bool = False  # 言语分歧
    emotional_tension: bool = False  # 情感紧张
    communication_breakdown: bool = False  # 沟通中断
    value_conflict: bool = False  # 价值观冲突
    resource_competition: bool = False  # 资源竞争
    
    # 冲突处理风格
    conflict_styles: Dict[str, ConflictStyle] = None  # 每个参与者的冲突风格
    
    def __post_init__(self):
        if self.conflict_styles is None:
            self.conflict_styles = {}

@dataclass
class RelationshipMilestone:
    """关系里程碑"""
    milestone_id: str
    relationship_id: str
    milestone_type: str  # 里程碑类型
    significance_level: float  # 重要性等级 [0,1]
    emotional_impact: float  # 情感影响 [-1,1]
    relationship_change: float  # 关系变化程度 [-1,1]
    timestamp: datetime
    description: str
    
    # 里程碑类别
    positive_milestone: bool = True  # 是否为积极里程碑
    relationship_deepening: bool = False  # 关系深化
    trust_building: bool = False  # 信任建立
    boundary_setting: bool = False  # 边界设定
    conflict_resolution: bool = False  # 冲突解决

@dataclass
class RelationshipDynamics:
    """关系动态"""
    relationship_id: str
    participants: List[str]
    relationship_type: RelationshipType
    
    # 亲密度分析
    intimacy_level: IntimacyLevel
    intimacy_score: float  # 亲密程度 [0,1]
    trust_level: float  # 信任水平 [0,1]
    vulnerability_sharing: float  # 脆弱性分享 [0,1]
    
    # 权力关系
    power_balance: float  # 权力平衡 [-1,1]
    power_dynamics: PowerDynamics
    influence_patterns: Dict[str, float]  # 影响模式
    
    # 情感互惠性
    emotional_reciprocity: float  # 情感互惠性 [0,1]
    support_balance: float  # 支持平衡 [-1,1]
    empathy_symmetry: float  # 同理心对称性 [0,1]
    
    # 支持模式
    support_patterns: List[EmotionalSupportPattern]
    
    # 冲突分析
    conflict_indicators: List[ConflictIndicator]
    
    # 可选字段
    primary_support_giver: Optional[str] = None
    support_network_strength: float = 0.0
    
    # 基本指标 (有默认值)
    conflict_frequency: float = 0.0  # 冲突频率
    conflict_resolution_rate: float = 0.0  # 冲突解决率
    relationship_health: float = 0.0  # 关系健康度 [0,1]
    stability_score: float = 0.0  # 稳定性分数 [0,1]
    satisfaction_level: float = 0.0  # 满意度水平 [0,1]
    development_trend: str = "stable"  # 发展趋势
    future_outlook: str = "positive"  # 未来展望
    data_quality_score: float = 1.0  # 数据质量分数 [0,1]
    confidence_level: float = 0.8  # 分析置信度 [0,1]
    
    # 可选列表字段
    harmony_indicators: List[str] = None  # 和谐指标
    relationship_trajectory: List[float] = None  # 关系轨迹
    milestones: List[RelationshipMilestone] = None
    significant_events: List[Dict[str, Any]] = None
    analysis_timestamp: datetime = None
    
    def __post_init__(self):
        if self.harmony_indicators is None:
            self.harmony_indicators = []
        if self.relationship_trajectory is None:
            self.relationship_trajectory = []
        if self.milestones is None:
            self.milestones = []
        if self.significant_events is None:
            self.significant_events = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = utc_now()

@dataclass
class RelationshipProfile:
    """关系档案"""
    profile_id: str
    participant_ids: List[str]
    
    # 历史关系动态
    relationship_history: List[RelationshipDynamics]
    
    # 关系模式
    interaction_frequency: float  # 交互频率
    communication_style: str  # 沟通风格
    emotional_expression_pattern: str  # 情感表达模式
    
    # 关系特征
    relationship_strengths: List[str]  # 关系优势
    relationship_challenges: List[str]  # 关系挑战
    improvement_opportunities: List[str]  # 改进机会
    
    # 预测模型
    relationship_forecast: Dict[str, Any]  # 关系预测
    risk_factors: List[str]  # 风险因素
    protective_factors: List[str]  # 保护因素
    
    # 推荐建议
    relationship_recommendations: List[str]  # 关系建议
    communication_tips: List[str]  # 沟通技巧
    conflict_prevention_strategies: List[str]  # 冲突预防策略
    
    created_timestamp: datetime
    last_updated: datetime
    
    def get_current_relationship_state(self) -> Optional[RelationshipDynamics]:
        """获取当前关系状态"""
        if not self.relationship_history:
            return None
        return max(self.relationship_history, key=lambda x: x.analysis_timestamp)
    
    def get_relationship_trend(self, days: int = 30) -> List[float]:
        """获取关系趋势"""
        cutoff_time = utc_now() - timedelta(days=days)
        recent_dynamics = [
            dynamics for dynamics in self.relationship_history
            if dynamics.analysis_timestamp > cutoff_time
        ]
        
        return [dynamics.relationship_health for dynamics in recent_dynamics]
    
    def calculate_relationship_stability(self) -> float:
        """计算关系稳定性"""
        if len(self.relationship_history) < 2:
            return 0.5  # 默认值
        
        health_scores = [dynamics.relationship_health for dynamics in self.relationship_history]
        stability = 1.0 - (max(health_scores) - min(health_scores))
        
        return max(0.0, min(1.0, stability))

@dataclass
class RelationshipAnalysisConfig:
    """关系分析配置"""
    # 亲密度计算权重
    personal_disclosure_weight: float = 0.3
    emotional_support_weight: float = 0.4
    shared_experiences_weight: float = 0.2
    communication_frequency_weight: float = 0.1
    
    # 权力分析阈值
    power_imbalance_threshold: float = 0.3
    dominant_threshold: float = 0.2
    submissive_threshold: float = -0.2
    
    # 冲突检测阈值
    conflict_severity_threshold: float = 0.6
    escalation_risk_threshold: float = 0.7
    
    # 关系健康度权重
    intimacy_health_weight: float = 0.25
    trust_health_weight: float = 0.25
    reciprocity_health_weight: float = 0.2
    conflict_management_weight: float = 0.15
    satisfaction_weight: float = 0.15
    
    # 时间窗口配置
    analysis_window_days: int = 30  # 分析时间窗口
    trend_analysis_days: int = 7   # 趋势分析窗口
    
    # 质量控制
    min_interaction_threshold: int = 5  # 最少交互阈值
    confidence_threshold: float = 0.6  # 最低置信度

def generate_relationship_id(participant1: str, participant2: str) -> str:
    """生成关系ID"""
    # 确保ID的一致性：按字母顺序排列参与者
    participants = sorted([participant1, participant2])
    return f"rel_{participants[0]}_{participants[1]}_{uuid.uuid4().hex[:8]}"

def generate_support_id() -> str:
    """生成支持ID"""
    return f"support_{uuid.uuid4().hex[:8]}"

def generate_conflict_id() -> str:
    """生成冲突ID"""
    return f"conflict_{uuid.uuid4().hex[:8]}"

def generate_milestone_id() -> str:
    """生成里程碑ID"""
    return f"milestone_{uuid.uuid4().hex[:8]}"

def classify_intimacy_level(intimacy_score: float) -> IntimacyLevel:
    """根据分数分类亲密程度等级"""
    if intimacy_score >= 0.8:
        return IntimacyLevel.VERY_HIGH
    elif intimacy_score >= 0.6:
        return IntimacyLevel.HIGH
    elif intimacy_score >= 0.4:
        return IntimacyLevel.MEDIUM
    elif intimacy_score >= 0.2:
        return IntimacyLevel.LOW
    else:
        return IntimacyLevel.VERY_LOW

def classify_power_dynamics(power_balance: float) -> PowerDynamics:
    """根据权力平衡分数分类权力动态"""
    if power_balance > 0.2:
        return PowerDynamics.DOMINANT
    elif power_balance < -0.2:
        return PowerDynamics.SUBMISSIVE
    else:
        return PowerDynamics.BALANCED
