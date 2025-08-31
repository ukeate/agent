"""
情感智能决策引擎的核心数据模型
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import uuid


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(Enum):
    """决策类型枚举"""
    INTERACTION_STRATEGY = "interaction_strategy"
    INTERVENTION_PLANNING = "intervention_planning"
    RESPONSE_ADJUSTMENT = "response_adjustment"
    CRISIS_RESPONSE = "crisis_response"
    SUPPORT_PROVISION = "support_provision"


class InterventionType(Enum):
    """干预类型枚举"""
    PREVENTIVE = "preventive"  # 预防性干预
    SUPPORTIVE = "supportive"  # 支持性干预
    CORRECTIVE = "corrective"  # 矫正性干预
    CRISIS = "crisis"  # 危机干预


class SeverityLevel(Enum):
    """严重程度枚举"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class DecisionContext:
    """决策上下文"""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 情感状态信息
    current_emotion_state: Dict[str, Any] = field(default_factory=dict)
    emotion_history: List[Dict[str, Any]] = field(default_factory=list)
    personality_profile: Dict[str, Any] = field(default_factory=dict)
    
    # 环境上下文
    conversation_context: str = ""
    user_input: str = ""
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    # 历史决策信息
    previous_decisions: List[Dict[str, Any]] = field(default_factory=list)
    decision_feedback: Dict[str, float] = field(default_factory=dict)
    
    # 附加元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'context_id': self.context_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'current_emotion_state': self.current_emotion_state,
            'emotion_history': self.emotion_history,
            'personality_profile': self.personality_profile,
            'conversation_context': self.conversation_context,
            'user_input': self.user_input,
            'environmental_factors': self.environmental_factors,
            'previous_decisions': self.previous_decisions,
            'decision_feedback': self.decision_feedback,
            'metadata': self.metadata
        }


@dataclass
class EmotionalDecision:
    """情感智能决策结果"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 决策内容
    decision_type: str = DecisionType.INTERACTION_STRATEGY.value
    chosen_strategy: str = ""
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 决策置信度和推理
    confidence_score: float = 0.0  # [0,1]
    reasoning: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # 预期结果
    expected_outcome: Dict[str, float] = field(default_factory=dict)
    success_metrics: List[str] = field(default_factory=list)
    
    # 执行信息
    execution_status: str = "pending"  # pending, executing, completed, failed
    execution_details: Dict[str, Any] = field(default_factory=dict)
    
    # 效果评估
    actual_outcome: Optional[Dict[str, float]] = None
    effectiveness_score: Optional[float] = None
    user_feedback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'decision_id': self.decision_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'decision_type': self.decision_type,
            'chosen_strategy': self.chosen_strategy,
            'strategy_parameters': self.strategy_parameters,
            'confidence_score': self.confidence_score,
            'reasoning': self.reasoning,
            'evidence': self.evidence,
            'expected_outcome': self.expected_outcome,
            'success_metrics': self.success_metrics,
            'execution_status': self.execution_status,
            'execution_details': self.execution_details,
            'actual_outcome': self.actual_outcome,
            'effectiveness_score': self.effectiveness_score,
            'user_feedback': self.user_feedback
        }


@dataclass
class RiskFactor:
    """风险因子"""
    factor_type: str = ""
    score: float = 0.0  # [0,1]
    evidence: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    description: str = ""


@dataclass
class RiskAssessment:
    """情感健康风险评估"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 风险评估结果
    risk_level: str = RiskLevel.LOW.value
    risk_score: float = 0.0  # [0,1]
    risk_factors: List[RiskFactor] = field(default_factory=list)
    
    # 预测信息
    prediction_confidence: float = 0.0  # [0,1]
    prediction_timeframe: timedelta = field(default=timedelta(hours=24))
    
    # 建议行动
    recommended_actions: List[str] = field(default_factory=list)
    urgency_level: str = "low"
    alert_triggered: bool = False
    
    # 评估详情
    assessment_method: str = "multi_factor_analysis"
    data_completeness: float = 1.0  # [0,1]
    assessment_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'assessment_id': self.assessment_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'risk_level': self.risk_level,
            'risk_score': self.risk_score,
            'risk_factors': [
                {
                    'factor_type': factor.factor_type,
                    'score': factor.score,
                    'evidence': factor.evidence,
                    'weight': factor.weight,
                    'description': factor.description
                }
                for factor in self.risk_factors
            ],
            'prediction_confidence': self.prediction_confidence,
            'prediction_timeframe': self.prediction_timeframe.total_seconds(),
            'recommended_actions': self.recommended_actions,
            'urgency_level': self.urgency_level,
            'alert_triggered': self.alert_triggered,
            'assessment_method': self.assessment_method,
            'data_completeness': self.data_completeness,
            'assessment_details': self.assessment_details
        }


@dataclass
class InterventionStrategy:
    """干预策略"""
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_name: str = ""
    description: str = ""
    intervention_type: str = InterventionType.SUPPORTIVE.value
    
    # 策略参数
    implementation_steps: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    expected_duration: Optional[timedelta] = None
    
    # 效果预测
    expected_effectiveness: float = 0.5  # [0,1]
    success_metrics: List[str] = field(default_factory=list)
    priority_score: float = 0.5  # [0,1]
    
    # 适用性
    target_emotions: List[str] = field(default_factory=list)
    target_risk_levels: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'description': self.description,
            'intervention_type': self.intervention_type,
            'implementation_steps': self.implementation_steps,
            'required_resources': self.required_resources,
            'expected_duration': self.expected_duration.total_seconds() if self.expected_duration else None,
            'expected_effectiveness': self.expected_effectiveness,
            'success_metrics': self.success_metrics,
            'priority_score': self.priority_score,
            'target_emotions': self.target_emotions,
            'target_risk_levels': self.target_risk_levels,
            'contraindications': self.contraindications
        }


@dataclass
class InterventionPlan:
    """干预计划"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # 计划内容
    intervention_type: str = InterventionType.SUPPORTIVE.value
    urgency_level: str = "medium"
    target_risk_factors: List[str] = field(default_factory=list)
    
    # 策略组合
    strategies: List[InterventionStrategy] = field(default_factory=list)
    primary_strategy: Optional[InterventionStrategy] = None
    
    # 资源和时间安排
    resources: List[Dict[str, Any]] = field(default_factory=list)
    timeline: Dict[str, datetime] = field(default_factory=dict)
    
    # 成功指标
    success_metrics: List[str] = field(default_factory=list)
    monitoring_frequency: timedelta = field(default=timedelta(hours=6))
    
    # 执行状态
    status: str = "draft"  # draft, active, paused, completed, cancelled
    progress: float = 0.0  # [0,1]
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'plan_id': self.plan_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'intervention_type': self.intervention_type,
            'urgency_level': self.urgency_level,
            'target_risk_factors': self.target_risk_factors,
            'strategies': [strategy.to_dict() for strategy in self.strategies],
            'primary_strategy': self.primary_strategy.to_dict() if self.primary_strategy else None,
            'resources': self.resources,
            'timeline': {k: v.isoformat() for k, v in self.timeline.items()},
            'success_metrics': self.success_metrics,
            'monitoring_frequency': self.monitoring_frequency.total_seconds(),
            'status': self.status,
            'progress': self.progress,
            'execution_log': self.execution_log
        }


@dataclass
class CrisisAssessment:
    """危机评估结果"""
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 危机状态
    crisis_detected: bool = False
    severity_level: str = SeverityLevel.MILD.value
    crisis_type: str = "emotional"  # emotional, behavioral, cognitive
    
    # 危机指标
    indicators: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0  # [0,1]
    confidence: float = 0.0  # [0,1]
    
    # 紧急行动
    immediate_actions: List[str] = field(default_factory=list)
    professional_required: bool = False
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list)
    
    # 监护设置
    monitoring_level: str = "normal"  # normal, elevated, intensive
    check_frequency: timedelta = field(default=timedelta(hours=1))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'assessment_id': self.assessment_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'crisis_detected': self.crisis_detected,
            'severity_level': self.severity_level,
            'crisis_type': self.crisis_type,
            'indicators': self.indicators,
            'risk_score': self.risk_score,
            'confidence': self.confidence,
            'immediate_actions': self.immediate_actions,
            'professional_required': self.professional_required,
            'emergency_contacts': self.emergency_contacts,
            'monitoring_level': self.monitoring_level,
            'check_frequency': self.check_frequency.total_seconds()
        }


@dataclass
class HealthDashboardData:
    """健康仪表盘数据"""
    user_id: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    time_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now() - timedelta(days=30), datetime.now()))
    
    # 整体健康指标
    overall_health_score: float = 0.5  # [0,1]
    emotional_stability: float = 0.5  # [0,1]
    resilience_score: float = 0.5  # [0,1]
    
    # 风险评估
    current_risk_level: str = RiskLevel.LOW.value
    risk_trend: str = "stable"  # improving, stable, deteriorating
    risk_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # 情感趋势
    emotion_trends: Dict[str, List[Tuple[datetime, float]]] = field(default_factory=dict)
    dominant_emotions: List[Tuple[str, float]] = field(default_factory=list)
    emotion_volatility: float = 0.0
    
    # 干预效果
    active_interventions: int = 0
    completed_interventions: int = 0
    intervention_success_rate: float = 0.0
    
    # 目标进度
    health_goals: List[Dict[str, Any]] = field(default_factory=list)
    goal_progress: Dict[str, float] = field(default_factory=dict)
    
    # 洞察和建议
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'user_id': self.user_id,
            'generated_at': self.generated_at.isoformat(),
            'time_period': [self.time_period[0].isoformat(), self.time_period[1].isoformat()],
            'overall_health_score': self.overall_health_score,
            'emotional_stability': self.emotional_stability,
            'resilience_score': self.resilience_score,
            'current_risk_level': self.current_risk_level,
            'risk_trend': self.risk_trend,
            'risk_history': [(dt.isoformat(), score) for dt, score in self.risk_history],
            'emotion_trends': {
                emotion: [(dt.isoformat(), value) for dt, value in trend]
                for emotion, trend in self.emotion_trends.items()
            },
            'dominant_emotions': self.dominant_emotions,
            'emotion_volatility': self.emotion_volatility,
            'active_interventions': self.active_interventions,
            'completed_interventions': self.completed_interventions,
            'intervention_success_rate': self.intervention_success_rate,
            'health_goals': self.health_goals,
            'goal_progress': self.goal_progress,
            'insights': self.insights,
            'recommendations': self.recommendations
        }