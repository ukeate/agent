"""
社交情境数据模型

用于定义社交场景、角色和情境适配相关的数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class SceneType(Enum):
    """场景类型枚举"""
    FORMAL_BUSINESS = "formal_business"         # 正式商务
    INFORMAL_SOCIAL = "informal_social"         # 非正式社交
    ACADEMIC = "academic"                       # 学术场合
    FAMILY = "family"                          # 家庭环境
    ENTERTAINMENT = "entertainment"             # 娱乐场所
    HEALTHCARE = "healthcare"                  # 医疗场所
    EDUCATIONAL = "educational"                # 教育环境
    RELIGIOUS = "religious"                    # 宗教场所
    LEGAL = "legal"                           # 法律场合
    EMERGENCY = "emergency"                    # 紧急情况

class SocialRole(Enum):
    """社交角色枚举"""
    LEADER = "leader"                         # 领导者
    PEER = "peer"                            # 同级
    SUBORDINATE = "subordinate"              # 下属
    MENTOR = "mentor"                        # 导师
    STUDENT = "student"                      # 学生
    FRIEND = "friend"                        # 朋友
    FAMILY_MEMBER = "family_member"          # 家庭成员
    STRANGER = "stranger"                    # 陌生人
    PROFESSIONAL = "professional"            # 专业人士
    CLIENT = "client"                        # 客户

class CommunicationStyle(Enum):
    """沟通风格枚举"""
    DIRECT = "direct"                        # 直接
    INDIRECT = "indirect"                    # 间接
    FORMAL = "formal"                        # 正式
    CASUAL = "casual"                        # 随意
    SUPPORTIVE = "supportive"                # 支持性
    ANALYTICAL = "analytical"                # 分析性
    EXPRESSIVE = "expressive"                # 表达性
    DIPLOMATIC = "diplomatic"                # 外交性

@dataclass
class SocialContext:
    """社交情境数据结构"""
    context_id: str                                    # 情境ID
    scene_type: SceneType                             # 场景类型
    formality_level: float = 0.5                      # 正式程度 [0,1]
    privacy_level: float = 0.5                        # 隐私程度 [0,1]
    social_roles: Dict[str, SocialRole] = field(default_factory=dict)  # 社交角色映射
    cultural_background: Optional[str] = None          # 文化背景
    language_context: str = "chinese"                  # 语言环境
    emotional_norms: Dict[str, float] = field(default_factory=dict)    # 情感规范
    sensitive_topics: List[str] = field(default_factory=list)         # 敏感话题
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 场景特征
    participant_count: int = 2                         # 参与者数量
    interaction_duration: Optional[int] = None         # 互动时长(分钟)
    is_public: bool = False                           # 是否公开场合
    is_professional: bool = False                      # 是否专业场合
    has_authority_figures: bool = False                # 是否有权威人物
    
    # 情感期望
    expected_emotional_tone: str = "neutral"           # 预期情感基调
    emotional_expression_level: float = 0.5           # 情感表达水平 [0,1]
    empathy_expectation: float = 0.5                  # 共情期望 [0,1]
    conflict_tolerance: float = 0.5                   # 冲突容忍度 [0,1]

@dataclass
class RoleExpectation:
    """角色期望数据结构"""
    role: SocialRole                                  # 社交角色
    expected_behaviors: List[str]                     # 期望行为
    emotional_requirements: Dict[str, float]          # 情感要求
    communication_style: CommunicationStyle          # 沟通风格
    authority_level: float = 0.5                     # 权威程度 [0,1]
    responsibility_level: float = 0.5                # 责任程度 [0,1]
    
    # 角色约束
    forbidden_topics: List[str] = field(default_factory=list)     # 禁止话题
    required_courtesy: List[str] = field(default_factory=list)    # 必要礼貌
    emotional_boundaries: Dict[str, float] = field(default_factory=dict)  # 情感边界

@dataclass
class SceneFeatureProfile:
    """场景特征画像"""
    scene_type: SceneType                             # 场景类型
    typical_roles: List[SocialRole]                   # 典型角色
    formality_range: tuple[float, float]              # 正式性范围
    privacy_range: tuple[float, float]                # 隐私性范围
    
    # 情感特征
    dominant_emotions: List[str]                      # 主导情感
    acceptable_emotions: List[str]                    # 可接受情感
    discouraged_emotions: List[str]                   # 不鼓励情感
    
    # 互动模式
    typical_interaction_patterns: List[str]           # 典型互动模式
    preferred_communication_styles: List[CommunicationStyle]  # 偏好沟通风格
    conflict_resolution_approaches: List[str]         # 冲突解决方式
    
    # 文化考量
    cultural_variations: Dict[str, Dict[str, Any]]    # 文化变异
    language_specific_norms: Dict[str, Dict[str, Any]]  # 语言特定规范

@dataclass
class AdaptationStrategy:
    """适应策略数据结构"""
    strategy_id: str                                  # 策略ID
    target_context: SocialContext                     # 目标情境
    adaptation_rules: List[Dict[str, Any]]           # 适应规则
    
    # 表达调整
    formality_adjustments: Dict[str, str]            # 正式性调整
    emotional_intensity_modifiers: Dict[str, float]  # 情感强度修饰符
    topic_filtering_rules: List[str]                 # 话题过滤规则
    
    # 沟通风格调整
    preferred_sentence_structures: List[str]          # 偏好句式结构
    politeness_markers: List[str]                    # 礼貌标记
    cultural_specific_phrases: Dict[str, List[str]]   # 文化特定短语
    
    # 反应模式
    response_timing_preferences: Dict[str, float]     # 响应时机偏好
    empathy_expression_methods: List[str]            # 共情表达方式
    conflict_avoidance_strategies: List[str]         # 冲突回避策略

# 预定义场景特征画像
SCENE_PROFILES: Dict[SceneType, SceneFeatureProfile] = {
    SceneType.FORMAL_BUSINESS: SceneFeatureProfile(
        scene_type=SceneType.FORMAL_BUSINESS,
        typical_roles=[SocialRole.LEADER, SocialRole.PEER, SocialRole.SUBORDINATE, SocialRole.CLIENT],
        formality_range=(0.7, 1.0),
        privacy_range=(0.3, 0.7),
        dominant_emotions=["confidence", "professionalism", "focus"],
        acceptable_emotions=["enthusiasm", "concern", "determination"],
        discouraged_emotions=["anger", "sadness", "excessive_joy"],
        typical_interaction_patterns=["structured_discussion", "presentation", "negotiation", "decision_making"],
        preferred_communication_styles=[CommunicationStyle.FORMAL, CommunicationStyle.ANALYTICAL, CommunicationStyle.DIPLOMATIC],
        conflict_resolution_approaches=["mediation", "structured_negotiation", "hierarchical_decision"],
        cultural_variations={
            "collectivist": {"hierarchy_emphasis": 0.8, "group_harmony": 0.9},
            "individualist": {"direct_communication": 0.8, "personal_accountability": 0.9}
        },
        language_specific_norms={
            "chinese": {"respectful_titles": True, "indirect_refusal": True},
            "english": {"directness_accepted": True, "time_efficiency": True}
        }
    ),
    
    SceneType.INFORMAL_SOCIAL: SceneFeatureProfile(
        scene_type=SceneType.INFORMAL_SOCIAL,
        typical_roles=[SocialRole.FRIEND, SocialRole.PEER, SocialRole.STRANGER],
        formality_range=(0.0, 0.4),
        privacy_range=(0.5, 1.0),
        dominant_emotions=["happiness", "relaxation", "friendliness"],
        acceptable_emotions=["excitement", "humor", "curiosity", "mild_concern"],
        discouraged_emotions=["intense_anger", "deep_sadness", "overwhelming_anxiety"],
        typical_interaction_patterns=["casual_chat", "storytelling", "shared_activities", "mutual_support"],
        preferred_communication_styles=[CommunicationStyle.CASUAL, CommunicationStyle.EXPRESSIVE, CommunicationStyle.SUPPORTIVE],
        conflict_resolution_approaches=["open_discussion", "humor_deflection", "mutual_understanding"],
        cultural_variations={
            "collectivist": {"group_inclusion": 0.9, "face_saving": 0.8},
            "individualist": {"personal_expression": 0.8, "individual_boundaries": 0.7}
        },
        language_specific_norms={
            "chinese": {"group_oriented_language": True, "modest_self_reference": True},
            "english": {"personal_opinions_welcome": True, "casual_interruptions_ok": True}
        }
    ),
    
    SceneType.FAMILY: SceneFeatureProfile(
        scene_type=SceneType.FAMILY,
        typical_roles=[SocialRole.FAMILY_MEMBER],
        formality_range=(0.0, 0.3),
        privacy_range=(0.8, 1.0),
        dominant_emotions=["love", "care", "warmth", "protectiveness"],
        acceptable_emotions=["concern", "disappointment", "pride", "frustration"],
        discouraged_emotions=["hatred", "contempt"],
        typical_interaction_patterns=["nurturing_support", "guidance", "shared_memories", "conflict_resolution"],
        preferred_communication_styles=[CommunicationStyle.SUPPORTIVE, CommunicationStyle.DIRECT, CommunicationStyle.EXPRESSIVE],
        conflict_resolution_approaches=["family_discussion", "elder_mediation", "emotional_reconciliation"],
        cultural_variations={
            "collectivist": {"hierarchical_respect": 0.9, "family_honor": 0.8},
            "individualist": {"open_communication": 0.8, "individual_autonomy": 0.7}
        },
        language_specific_norms={
            "chinese": {"generational_respect": True, "family_harmony_priority": True},
            "english": {"emotional_openness": True, "individual_needs_recognition": True}
        }
    ),
    
    SceneType.ACADEMIC: SceneFeatureProfile(
        scene_type=SceneType.ACADEMIC,
        typical_roles=[SocialRole.MENTOR, SocialRole.STUDENT, SocialRole.PEER],
        formality_range=(0.5, 0.8),
        privacy_range=(0.2, 0.6),
        dominant_emotions=["curiosity", "intellectual_engagement", "respect"],
        acceptable_emotions=["confusion", "excitement", "mild_frustration", "determination"],
        discouraged_emotions=["contempt", "dismissiveness", "overwhelming_anxiety"],
        typical_interaction_patterns=["knowledge_sharing", "questioning", "debate", "collaborative_learning"],
        preferred_communication_styles=[CommunicationStyle.ANALYTICAL, CommunicationStyle.FORMAL, CommunicationStyle.SUPPORTIVE],
        conflict_resolution_approaches=["evidence_based_discussion", "mentor_guidance", "peer_collaboration"],
        cultural_variations={
            "collectivist": {"teacher_reverence": 0.9, "group_learning": 0.8},
            "individualist": {"critical_thinking": 0.9, "individual_contribution": 0.8}
        },
        language_specific_norms={
            "chinese": {"knowledge_humility": True, "teacher_student_hierarchy": True},
            "english": {"questioning_encouraged": True, "idea_challenges_acceptable": True}
        }
    )
}
