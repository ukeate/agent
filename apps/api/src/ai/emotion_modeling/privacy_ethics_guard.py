"""
隐私保护和伦理机制 - Story 11.6 Task 7
实现多方隐私保护、文化敏感度保护和伦理边界控制
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import re
from collections import defaultdict
from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface

from src.core.logging import get_logger
logger = get_logger(__name__)

class PrivacyLevel(Enum):
    """隐私保护级别"""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"

class EthicalRisk(Enum):
    """伦理风险类型"""
    PRIVACY_VIOLATION = "privacy_violation"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    BIAS_AMPLIFICATION = "bias_amplification"
    CONSENT_VIOLATION = "consent_violation"
    DATA_MISUSE = "data_misuse"

class ConsentType(Enum):
    """同意类型"""
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

@dataclass
class PrivacyPolicy:
    """隐私政策"""
    user_id: str
    privacy_level: PrivacyLevel
    data_retention_days: int
    sharing_permissions: Dict[str, bool]
    anonymization_required: bool
    sensitive_data_categories: List[str]
    geographic_restrictions: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class ConsentRecord:
    """同意记录"""
    user_id: str
    consent_type: ConsentType
    data_categories: List[str]
    purpose: str
    timestamp: datetime
    expiry_date: Optional[datetime]
    withdrawal_date: Optional[datetime]
    ip_address: str
    user_agent: str

@dataclass
class EthicalViolation:
    """伦理违规记录"""
    violation_id: str
    risk_type: EthicalRisk
    severity: float  # 0.0-1.0
    description: str
    context: Dict[str, Any]
    timestamp: datetime
    user_affected: str
    action_taken: str
    resolved: bool

@dataclass
class CulturalSensitivity:
    """文化敏感性配置"""
    culture_code: str
    sensitive_topics: List[str]
    prohibited_emotions: List[str]
    communication_norms: Dict[str, Any]
    privacy_expectations: Dict[str, float]
    consent_requirements: Dict[str, str]

@dataclass
class DataProcessingRecord:
    """数据处理记录"""
    processing_id: str
    user_id: str
    data_type: str
    purpose: str
    processing_method: str
    timestamp: datetime
    retention_period: int
    anonymized: bool
    shared_with: List[str]
    legal_basis: str

class PrivacyEthicsGuard(EmotionModelingInterface):
    """隐私保护和伦理机制守护系统"""
    
    def __init__(self):
        self.privacy_policies: Dict[str, PrivacyPolicy] = {}
        self.consent_records: Dict[str, List[ConsentRecord]] = defaultdict(list)
        self.ethical_violations: List[EthicalViolation] = []
        self.cultural_configs: Dict[str, CulturalSensitivity] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.monitoring_enabled = True
        
        # 初始化文化敏感性配置
        self._initialize_cultural_configs()
        
        # 设置敏感词汇过滤器
        self.sensitive_patterns = self._load_sensitive_patterns()
        
    def _initialize_cultural_configs(self) -> None:
        """初始化文化敏感性配置"""
        # 中国文化配置
        self.cultural_configs["zh-CN"] = CulturalSensitivity(
            culture_code="zh-CN",
            sensitive_topics=["政治", "宗教", "家庭矛盾", "个人隐私", "健康问题"],
            prohibited_emotions=["愤怒过激", "仇恨", "歧视"],
            communication_norms={
                "directness_preference": 0.3,  # 偏好间接沟通
                "hierarchy_respect": 0.9,      # 高度重视等级
                "face_saving": 0.8            # 重视面子
            },
            privacy_expectations={
                "family_privacy": 0.9,
                "financial_privacy": 0.8,
                "emotional_privacy": 0.7
            },
            consent_requirements={
                "emotional_analysis": "explicit",
                "data_sharing": "explicit",
                "long_term_storage": "explicit"
            }
        )
        
        # 美国文化配置
        self.cultural_configs["en-US"] = CulturalSensitivity(
            culture_code="en-US",
            sensitive_topics=["race", "religion", "politics", "mental health"],
            prohibited_emotions=["hatred", "discrimination", "harassment"],
            communication_norms={
                "directness_preference": 0.7,
                "hierarchy_respect": 0.4,
                "individual_focus": 0.8
            },
            privacy_expectations={
                "personal_data": 0.8,
                "communication_privacy": 0.7,
                "behavioral_privacy": 0.6
            },
            consent_requirements={
                "emotional_analysis": "explicit",
                "data_sharing": "explicit",
                "behavioral_tracking": "implied"
            }
        )
        
        # 欧洲文化配置（GDPR合规）
        self.cultural_configs["eu"] = CulturalSensitivity(
            culture_code="eu",
            sensitive_topics=["personal_data", "health", "political_opinions"],
            prohibited_emotions=["discrimination", "hate_speech"],
            communication_norms={
                "privacy_first": 0.9,
                "data_minimization": 0.8,
                "transparency": 0.9
            },
            privacy_expectations={
                "data_protection": 0.9,
                "right_to_deletion": 0.9,
                "data_portability": 0.8
            },
            consent_requirements={
                "all_processing": "explicit",
                "data_transfer": "explicit",
                "automated_decisions": "explicit"
            }
        )
    
    def _load_sensitive_patterns(self) -> Dict[str, List[str]]:
        """加载敏感词汇模式"""
        return {
            "personal_identifiers": [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b\d{15,19}\b",          # Credit card pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                r"\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"  # Phone
            ],
            "sensitive_content": [
                r"密码|password|pwd",
                r"银行|bank|账户|account",
                r"身份证|ID card|passport",
                r"地址|address|位置|location"
            ],
            "emotional_manipulation": [
                r"你必须|you must|强迫|force",
                r"威胁|threat|恐吓|intimidate",
                r"利用情感|exploit emotion"
            ]
        }
    
    async def validate_privacy_compliance(
        self,
        user_id: str,
        emotion_data: Dict[str, Any],
        processing_purpose: str,
        cultural_context: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """验证隐私合规性"""
        try:
            violations = []
            
            # 检查用户隐私政策
            if user_id not in self.privacy_policies:
                violations.append("No privacy policy found for user")
                return False, violations
            
            privacy_policy = self.privacy_policies[user_id]
            
            # 检查数据处理同意
            consent_valid = await self._validate_consent(
                user_id, ["emotional_data"], processing_purpose
            )
            if not consent_valid:
                violations.append("No valid consent for emotional data processing")
            
            # 检查文化敏感性
            if cultural_context and cultural_context in self.cultural_configs:
                cultural_violations = await self._check_cultural_sensitivity(
                    emotion_data, self.cultural_configs[cultural_context]
                )
                violations.extend(cultural_violations)
            
            # 检查敏感内容
            sensitive_violations = self._detect_sensitive_content(emotion_data)
            violations.extend(sensitive_violations)
            
            # 检查数据最小化原则
            if not self._check_data_minimization(emotion_data, processing_purpose):
                violations.append("Data collection violates minimization principle")
            
            # 记录处理活动
            await self._record_processing_activity(
                user_id, emotion_data, processing_purpose, violations
            )
            
            is_compliant = len(violations) == 0
            return is_compliant, violations
            
        except Exception as e:
            logger.error(f"Privacy compliance validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def _validate_consent(
        self,
        user_id: str,
        data_categories: List[str],
        purpose: str
    ) -> bool:
        """验证用户同意"""
        user_consents = self.consent_records.get(user_id, [])
        
        for category in data_categories:
            # 寻找有效的同意记录
            valid_consent = False
            
            for consent in user_consents:
                if (consent.consent_type in [ConsentType.EXPLICIT, ConsentType.IMPLIED] and
                    category in consent.data_categories and
                    consent.purpose == purpose and
                    (not consent.expiry_date or consent.expiry_date > utc_now()) and
                    not consent.withdrawal_date):
                    valid_consent = True
                    break
            
            if not valid_consent:
                return False
        
        return True
    
    async def _check_cultural_sensitivity(
        self,
        emotion_data: Dict[str, Any],
        cultural_config: CulturalSensitivity
    ) -> List[str]:
        """检查文化敏感性"""
        violations = []
        
        # 检查禁止情感
        if "emotions" in emotion_data:
            emotions = emotion_data["emotions"]
            for prohibited_emotion in cultural_config.prohibited_emotions:
                if prohibited_emotion in emotions and emotions[prohibited_emotion] > 0.5:
                    violations.append(f"Detected prohibited emotion: {prohibited_emotion}")
        
        # 检查敏感话题
        context_text = str(emotion_data.get("context", ""))
        for sensitive_topic in cultural_config.sensitive_topics:
            if sensitive_topic.lower() in context_text.lower():
                violations.append(f"Detected sensitive topic: {sensitive_topic}")
        
        # 检查沟通规范
        norms = cultural_config.communication_norms
        if ("directness" in emotion_data and 
            emotion_data["directness"] > norms.get("directness_preference", 0.5) + 0.3):
            violations.append("Communication style may be culturally inappropriate")
        
        return violations
    
    def _detect_sensitive_content(self, emotion_data: Dict[str, Any]) -> List[str]:
        """检测敏感内容"""
        violations = []
        
        # 检查所有文本字段
        text_fields = []
        if "context" in emotion_data:
            text_fields.append(str(emotion_data["context"]))
        if "message" in emotion_data:
            text_fields.append(str(emotion_data["message"]))
        
        text_content = " ".join(text_fields)
        
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    violations.append(f"Detected {category} in content")
                    break
        
        return violations
    
    def _check_data_minimization(
        self,
        emotion_data: Dict[str, Any],
        purpose: str
    ) -> bool:
        """检查数据最小化原则"""
        # 定义不同目的所需的最小数据
        purpose_requirements = {
            "emotion_analysis": ["emotions", "intensity"],
            "social_interaction": ["emotions", "intensity", "context"],
            "personalization": ["emotions", "intensity", "user_preferences"],
            "research": ["emotions", "intensity", "anonymized_context"]
        }
        
        required_fields = purpose_requirements.get(purpose, [])
        
        # 检查是否包含非必需数据
        for field in emotion_data.keys():
            if field not in required_fields and field not in ["timestamp", "user_id"]:
                # 如果包含非必需数据，需要特殊justification
                if not self._has_justification(field, purpose):
                    return False
        
        return True
    
    def _has_justification(self, field: str, purpose: str) -> bool:
        """检查是否有合理justification"""
        # 简化版本，实际应该有更复杂的justification逻辑
        justified_combinations = {
            ("social_context", "social_interaction"),
            ("cultural_context", "personalization"),
            ("detailed_emotions", "research")
        }
        
        return (field, purpose) in justified_combinations
    
    async def _record_processing_activity(
        self,
        user_id: str,
        emotion_data: Dict[str, Any],
        purpose: str,
        violations: List[str]
    ) -> None:
        """记录处理活动"""
        processing_record = DataProcessingRecord(
            processing_id=f"proc_{utc_now().isoformat()}_{user_id}",
            user_id=user_id,
            data_type="emotional_data",
            purpose=purpose,
            processing_method="automated_analysis",
            timestamp=utc_now(),
            retention_period=self.privacy_policies[user_id].data_retention_days,
            anonymized=self.privacy_policies[user_id].anonymization_required,
            shared_with=[],
            legal_basis="consent"
        )
        
        self.processing_records.append(processing_record)
        
        # 如果有违规，记录伦理违规
        if violations:
            await self._record_ethical_violation(
                EthicalRisk.PRIVACY_VIOLATION,
                user_id,
                "Privacy compliance check failed",
                {"violations": violations, "purpose": purpose}
            )
    
    async def enforce_ethical_boundaries(
        self,
        emotion_vector: EmotionVector,
        context: SocialContext,
        action: str
    ) -> Tuple[bool, Optional[str]]:
        """执行伦理边界控制"""
        try:
            # 检查情感操纵风险
            manipulation_risk = await self._assess_manipulation_risk(
                emotion_vector, context, action
            )
            
            if manipulation_risk > 0.7:
                await self._record_ethical_violation(
                    EthicalRisk.EMOTIONAL_MANIPULATION,
                    context.participants[0] if context.participants else "unknown",
                    "High emotional manipulation risk detected",
                    {"risk_score": manipulation_risk, "action": action}
                )
                return False, "Action blocked due to emotional manipulation risk"
            
            # 检查偏见放大风险
            bias_risk = await self._assess_bias_amplification(emotion_vector, context)
            
            if bias_risk > 0.6:
                await self._record_ethical_violation(
                    EthicalRisk.BIAS_AMPLIFICATION,
                    context.participants[0] if context.participants else "unknown",
                    "Bias amplification risk detected",
                    {"bias_risk": bias_risk, "context": context.scenario}
                )
                return False, "Action blocked due to bias amplification risk"
            
            # 检查文化不敏感风险
            cultural_risk = await self._assess_cultural_insensitivity(context)
            
            if cultural_risk > 0.5:
                await self._record_ethical_violation(
                    EthicalRisk.CULTURAL_INSENSITIVITY,
                    context.participants[0] if context.participants else "unknown",
                    "Cultural insensitivity risk detected",
                    {"cultural_risk": cultural_risk, "culture": context.cultural_context}
                )
                return False, "Action blocked due to cultural insensitivity risk"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Ethical boundary enforcement failed: {e}")
            return False, f"Ethical check error: {str(e)}"
    
    async def _assess_manipulation_risk(
        self,
        emotion_vector: EmotionVector,
        context: SocialContext,
        action: str
    ) -> float:
        """评估情感操纵风险"""
        risk_score = 0.0
        
        # 检查情感强度异常
        if emotion_vector.intensity > 0.8:
            risk_score += 0.3
        
        # 检查是否试图诱导特定情感
        manipulative_actions = ["persuade", "convince", "pressure", "influence_decision"]
        if action in manipulative_actions:
            risk_score += 0.4
        
        # 检查目标情感类型
        negative_emotions = ["fear", "anxiety", "guilt", "shame"]
        for emotion in negative_emotions:
            if emotion in emotion_vector.emotions and emotion_vector.emotions[emotion] > 0.6:
                risk_score += 0.3
                break
        
        # 检查上下文中的权力不平衡
        if context.power_dynamics:
            max_power_diff = max(context.power_dynamics.values()) - min(context.power_dynamics.values())
            if max_power_diff > 0.5:
                risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    async def _assess_bias_amplification(
        self,
        emotion_vector: EmotionVector,
        context: SocialContext
    ) -> float:
        """评估偏见放大风险"""
        risk_score = 0.0
        
        # 检查极化情感
        emotion_values = list(emotion_vector.emotions.values())
        if len(emotion_values) > 1:
            emotion_variance = max(emotion_values) - min(emotion_values)
            if emotion_variance > 0.7:
                risk_score += 0.3
        
        # 检查群体极化情况
        if (hasattr(context, 'group_emotions') and 
            context.group_emotions and
            len(set(context.group_emotions.values())) == 1):  # 所有人情感相同
            risk_score += 0.4
        
        # 检查历史偏见模式
        if self._has_historical_bias_pattern(context.scenario):
            risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    def _has_historical_bias_pattern(self, scenario: str) -> bool:
        """检查是否存在历史偏见模式"""
        # 检查历史违规记录中的偏见模式
        bias_violations = [
            v for v in self.ethical_violations
            if v.risk_type == EthicalRisk.BIAS_AMPLIFICATION and
            v.context.get("scenario") == scenario
        ]
        
        return len(bias_violations) > 2  # 简化判断
    
    async def _assess_cultural_insensitivity(self, context: SocialContext) -> float:
        """评估文化不敏感风险"""
        risk_score = 0.0
        
        if not context.cultural_context:
            return 0.0
        
        cultural_config = self.cultural_configs.get(context.cultural_context)
        if not cultural_config:
            return 0.2  # 未知文化的基础风险
        
        # 检查是否涉及敏感话题
        if hasattr(context, 'topics'):
            for topic in context.topics:
                if topic in cultural_config.sensitive_topics:
                    risk_score += 0.4
                    break
        
        # 检查沟通风格是否合适
        if hasattr(context, 'communication_style'):
            style_mismatch = abs(
                context.communication_style - 
                cultural_config.communication_norms.get("directness_preference", 0.5)
            )
            if style_mismatch > 0.4:
                risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    async def _record_ethical_violation(
        self,
        risk_type: EthicalRisk,
        user_id: str,
        description: str,
        context: Dict[str, Any]
    ) -> None:
        """记录伦理违规"""
        violation = EthicalViolation(
            violation_id=f"viol_{utc_now().isoformat()}_{risk_type.value}",
            risk_type=risk_type,
            severity=context.get("risk_score", 0.5),
            description=description,
            context=context,
            timestamp=utc_now(),
            user_affected=user_id,
            action_taken="blocked",
            resolved=False
        )
        
        self.ethical_violations.append(violation)
        logger.warning(f"Ethical violation recorded: {description} for user {user_id}")
    
    async def anonymize_emotional_data(
        self,
        emotion_data: Dict[str, Any],
        anonymization_level: str = "standard"
    ) -> Dict[str, Any]:
        """匿名化情感数据"""
        try:
            anonymized_data = emotion_data.copy()
            
            # 移除直接标识符
            identifiers_to_remove = ["user_id", "name", "email", "phone", "address"]
            for identifier in identifiers_to_remove:
                if identifier in anonymized_data:
                    del anonymized_data[identifier]
            
            # 生成匿名ID
            if "user_id" in emotion_data:
                anonymized_data["anonymous_id"] = hashlib.sha256(
                    emotion_data["user_id"].encode()
                ).hexdigest()[:16]
            
            # 根据匿名化级别处理
            if anonymization_level == "high":
                # 高级匿名化：添加噪音，降低精度
                if "emotions" in anonymized_data:
                    emotions = anonymized_data["emotions"]
                    # 添加少量随机噪音
                    import random
                    for emotion, value in emotions.items():
                        noise = random.uniform(-0.05, 0.05)
                        emotions[emotion] = max(0, min(1, value + noise))
                
                # 时间戳泛化
                if "timestamp" in anonymized_data:
                    timestamp = datetime.fromisoformat(anonymized_data["timestamp"])
                    # 泛化到小时级别
                    anonymized_data["timestamp"] = timestamp.replace(
                        minute=0, second=0, microsecond=0
                    ).isoformat()
            
            elif anonymization_level == "k_anonymous":
                # K-匿名化：确保至少k个用户具有相同的准标识符组合
                await self._apply_k_anonymization(anonymized_data, k=3)
            
            # 移除高度敏感的上下文信息
            if "context" in anonymized_data:
                context = anonymized_data["context"]
                sensitive_context_keys = [
                    "location", "ip_address", "device_info", 
                    "personal_details", "contacts"
                ]
                for key in sensitive_context_keys:
                    if key in context:
                        del context[key]
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return emotion_data  # 返回原数据作为fallback
    
    async def _apply_k_anonymization(
        self,
        data: Dict[str, Any],
        k: int = 3
    ) -> None:
        """应用K-匿名化"""
        # 简化的K-匿名化实现
        # 实际应用中需要更复杂的算法
        
        # 泛化地理位置
        if "location" in data:
            # 将精确位置泛化为城市级别
            location = data["location"]
            if isinstance(location, dict) and "city" in location:
                data["location"] = {"city": location["city"]}
        
        # 泛化时间信息
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])
            # 泛化到周级别
            days_to_monday = timestamp.weekday()
            monday = timestamp - timedelta(days=days_to_monday)
            data["timestamp"] = monday.replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
    
    async def manage_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        data_categories: List[str],
        purpose: str,
        ip_address: str = "",
        user_agent: str = "",
        expiry_days: Optional[int] = None
    ) -> ConsentRecord:
        """管理用户同意"""
        expiry_date = None
        if expiry_days:
            expiry_date = utc_now() + timedelta(days=expiry_days)
        
        consent_record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            data_categories=data_categories,
            purpose=purpose,
            timestamp=utc_now(),
            expiry_date=expiry_date,
            withdrawal_date=None,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.consent_records[user_id].append(consent_record)
        
        logger.info(f"Consent recorded for user {user_id}: {consent_type.value} for {purpose}")
        return consent_record
    
    async def withdraw_consent(
        self,
        user_id: str,
        data_categories: List[str],
        purpose: str
    ) -> bool:
        """撤回用户同意"""
        try:
            user_consents = self.consent_records.get(user_id, [])
            
            for consent in user_consents:
                if (consent.data_categories == data_categories and 
                    consent.purpose == purpose and
                    not consent.withdrawal_date):
                    consent.withdrawal_date = utc_now()
                    consent.consent_type = ConsentType.WITHDRAWN
                    
                    logger.info(f"Consent withdrawn for user {user_id}: {purpose}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Consent withdrawal failed: {e}")
            return False
    
    async def create_privacy_policy(
        self,
        user_id: str,
        privacy_level: PrivacyLevel,
        data_retention_days: int = 365,
        sharing_permissions: Optional[Dict[str, bool]] = None,
        cultural_context: Optional[str] = None
    ) -> PrivacyPolicy:
        """创建隐私政策"""
        default_sharing = {
            "analytics": False,
            "research": False,
            "third_party": False,
            "marketing": False
        }
        
        if sharing_permissions:
            default_sharing.update(sharing_permissions)
        
        # 根据文化背景调整默认设置
        if cultural_context in self.cultural_configs:
            cultural_config = self.cultural_configs[cultural_context]
            if cultural_config.privacy_expectations.get("data_protection", 0.5) > 0.7:
                # 高隐私期望文化
                default_sharing = {k: False for k in default_sharing.keys()}
                data_retention_days = min(data_retention_days, 180)
        
        privacy_policy = PrivacyPolicy(
            user_id=user_id,
            privacy_level=privacy_level,
            data_retention_days=data_retention_days,
            sharing_permissions=default_sharing,
            anonymization_required=privacy_level in [
                PrivacyLevel.CONFIDENTIAL, 
                PrivacyLevel.HIGHLY_CONFIDENTIAL
            ],
            sensitive_data_categories=["emotional_state", "personal_context"],
            geographic_restrictions=[],
            created_at=utc_now(),
            updated_at=utc_now()
        )
        
        self.privacy_policies[user_id] = privacy_policy
        
        logger.info(f"Privacy policy created for user {user_id} with level {privacy_level.value}")
        return privacy_policy
    
    async def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """生成合规报告"""
        try:
            # 统计违规情况
            period_violations = [
                v for v in self.ethical_violations
                if start_date <= v.timestamp <= end_date
            ]
            
            violation_stats = defaultdict(int)
            for violation in period_violations:
                violation_stats[violation.risk_type.value] += 1
            
            # 统计处理活动
            period_processing = [
                p for p in self.processing_records
                if start_date <= p.timestamp <= end_date
            ]
            
            # 统计同意状态
            consent_stats = {
                "explicit": 0,
                "implied": 0,
                "withdrawn": 0,
                "expired": 0
            }
            
            for user_consents in self.consent_records.values():
                for consent in user_consents:
                    if start_date <= consent.timestamp <= end_date:
                        consent_stats[consent.consent_type.value] += 1
            
            # 计算合规分数
            total_activities = len(period_processing)
            violation_count = len(period_violations)
            compliance_score = max(0.0, 1.0 - (violation_count / max(total_activities, 1)))
            
            return {
                "report_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "compliance_score": compliance_score,
                "total_processing_activities": total_activities,
                "total_violations": violation_count,
                "violation_breakdown": dict(violation_stats),
                "consent_statistics": consent_stats,
                "privacy_policies_count": len(self.privacy_policies),
                "cultural_contexts_supported": len(self.cultural_configs),
                "recommendations": self._generate_compliance_recommendations(
                    violation_stats, compliance_score
                )
            }
            
        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_compliance_recommendations(
        self,
        violation_stats: Dict[str, int],
        compliance_score: float
    ) -> List[str]:
        """生成合规建议"""
        recommendations = []
        
        if compliance_score < 0.8:
            recommendations.append("Improve overall compliance processes and monitoring")
        
        if violation_stats.get("privacy_violation", 0) > 0:
            recommendations.append("Enhance privacy protection mechanisms and user consent processes")
        
        if violation_stats.get("cultural_insensitivity", 0) > 0:
            recommendations.append("Expand cultural sensitivity training and update cultural configurations")
        
        if violation_stats.get("emotional_manipulation", 0) > 0:
            recommendations.append("Strengthen emotional manipulation detection and prevention")
        
        if violation_stats.get("bias_amplification", 0) > 0:
            recommendations.append("Implement bias detection and mitigation strategies")
        
        return recommendations
    
    async def cleanup_expired_data(self) -> None:
        """清理过期数据"""
        try:
            current_time = utc_now()
            
            # 清理过期的处理记录
            self.processing_records = [
                record for record in self.processing_records
                if (current_time - record.timestamp).days <= record.retention_period
            ]
            
            # 清理过期的同意记录
            for user_id in list(self.consent_records.keys()):
                valid_consents = []
                for consent in self.consent_records[user_id]:
                    if (not consent.expiry_date or 
                        consent.expiry_date > current_time):
                        valid_consents.append(consent)
                
                if valid_consents:
                    self.consent_records[user_id] = valid_consents
                else:
                    del self.consent_records[user_id]
            
            # 清理旧的违规记录（保留6个月）
            six_months_ago = current_time - timedelta(days=180)
            self.ethical_violations = [
                violation for violation in self.ethical_violations
                if violation.timestamp > six_months_ago
            ]
            
            logger.info("Expired data cleanup completed")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
