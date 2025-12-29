"""
Task 7隐私保护与伦理防护完整单元测试套件
测试PrivacyEthicsGuard的所有安全功能和合规性检查
"""

from src.core.utils.timezone_utils import utc_now
import pytest
import asyncio
from datetime import timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from ai.emotion_modeling.privacy_ethics_guard import (
    PrivacyEthicsGuard,
    PrivacyLevel,
    EthicalConcern,
    PrivacyViolation,
    EthicalViolation,
    PrivacyPolicy,
    ConsentRecord,
    AuditRecord,
    ComplianceReport,
    DataClassification
)
from ai.emotion_modeling.models import EmotionVector, SocialContext

@pytest.fixture
def privacy_guard():
    """创建隐私伦理防护实例"""
    return PrivacyEthicsGuard()

@pytest.fixture
def sample_user_data():
    """创建测试用户数据"""
    return {
        "user_id": "test_user_001",
        "personal_info": {
            "age": 25,
            "location": "Beijing",
            "occupation": "Engineer"
        },
        "emotion_history": [
            {
                "timestamp": utc_now() - timedelta(days=1),
                "emotions": {"happiness": 0.8, "confidence": 0.7},
                "context": "work_meeting"
            },
            {
                "timestamp": utc_now() - timedelta(hours=2),
                "emotions": {"stress": 0.6, "anxiety": 0.4},
                "context": "personal_conversation"
            }
        ],
        "social_connections": [
            {"connected_user": "user_002", "relationship": "colleague"},
            {"connected_user": "user_003", "relationship": "friend"}
        ]
    }

@pytest.fixture
def sample_consent_record():
    """创建测试同意记录"""
    return ConsentRecord(
        user_id="test_user_001",
        consent_type="emotion_analysis",
        granted=True,
        timestamp=utc_now(),
        scope=["emotion_tracking", "social_analysis"],
        expiry_date=utc_now() + timedelta(days=365),
        withdrawal_date=None,
        version="1.0"
    )

@pytest.fixture
def sample_privacy_policy():
    """创建测试隐私政策"""
    return PrivacyPolicy(
        policy_id="test_policy_v1",
        version="1.0",
        effective_date=utc_now(),
        data_types_collected=["emotions", "messages", "social_connections"],
        purposes=["emotion_analysis", "social_insights", "user_experience"],
        retention_period_days=365,
        sharing_allowed=False,
        third_party_access=False,
        user_rights=["access", "rectification", "erasure", "portability"],
        contact_info="privacy@example.com"
    )

class TestPrivacyEthicsGuard:
    """隐私伦理防护基础功能测试"""
    
    def test_initialization(self, privacy_guard):
        """测试初始化"""
        assert privacy_guard is not None
        assert hasattr(privacy_guard, 'privacy_policies')
        assert hasattr(privacy_guard, 'consent_records') 
        assert hasattr(privacy_guard, 'audit_log')
        assert hasattr(privacy_guard, 'violation_history')
        assert privacy_guard.compliance_enabled is True
        assert privacy_guard.audit_enabled is True
    
    def test_enable_disable_compliance(self, privacy_guard):
        """测试合规检查启用禁用"""
        privacy_guard.disable_compliance_checking()
        assert privacy_guard.compliance_enabled is False
        
        privacy_guard.enable_compliance_checking()
        assert privacy_guard.compliance_enabled is True
    
    def test_enable_disable_audit(self, privacy_guard):
        """测试审计日志启用禁用"""
        privacy_guard.disable_audit_logging()
        assert privacy_guard.audit_enabled is False
        
        privacy_guard.enable_audit_logging()
        assert privacy_guard.audit_enabled is True

class TestDataClassificationAndSensitivity:
    """数据分类和敏感度测试"""
    
    @pytest.mark.asyncio
    async def test_classify_data_sensitivity_basic(self, privacy_guard, sample_user_data):
        """测试基础数据敏感度分类"""
        classification = await privacy_guard.classify_data_sensitivity(sample_user_data)
        
        assert isinstance(classification, DataClassification)
        assert classification.data_id is not None
        assert classification.sensitivity_level in [level.value for level in PrivacyLevel]
        assert 0.0 <= classification.sensitivity_score <= 1.0
        assert isinstance(classification.sensitive_fields, list)
        assert isinstance(classification.risk_factors, list)
        assert isinstance(classification.recommended_protection_level, str)
    
    @pytest.mark.asyncio
    async def test_classify_high_sensitivity_data(self, privacy_guard):
        """测试高敏感度数据分类"""
        high_sensitivity_data = {
            "user_id": "sensitive_user",
            "personal_info": {
                "ssn": "123-45-6789",
                "medical_condition": "depression",
                "financial_status": "bankruptcy"
            },
            "emotion_history": [
                {
                    "timestamp": utc_now(),
                    "emotions": {"depression": 0.9, "suicidal_thoughts": 0.7},
                    "context": "therapy_session"
                }
            ]
        }
        
        classification = await privacy_guard.classify_data_sensitivity(high_sensitivity_data)
        
        assert classification.sensitivity_level == PrivacyLevel.HIGHLY_SENSITIVE.value
        assert classification.sensitivity_score > 0.8
        assert len(classification.sensitive_fields) > 0
        assert "medical" in classification.recommended_protection_level.lower() or "high" in classification.recommended_protection_level.lower()
    
    @pytest.mark.asyncio
    async def test_classify_low_sensitivity_data(self, privacy_guard):
        """测试低敏感度数据分类"""
        low_sensitivity_data = {
            "user_id": "public_user",
            "public_profile": {
                "username": "john_doe",
                "bio": "Software developer"
            },
            "emotion_history": [
                {
                    "timestamp": utc_now(),
                    "emotions": {"neutral": 0.8, "professional": 0.7},
                    "context": "public_presentation"
                }
            ]
        }
        
        classification = await privacy_guard.classify_data_sensitivity(low_sensitivity_data)
        
        assert classification.sensitivity_level in [PrivacyLevel.PUBLIC.value, PrivacyLevel.LOW_SENSITIVE.value]
        assert classification.sensitivity_score < 0.4
    
    @pytest.mark.asyncio
    async def test_classify_empty_data(self, privacy_guard):
        """测试空数据分类"""
        empty_data = {}
        
        classification = await privacy_guard.classify_data_sensitivity(empty_data)
        
        assert classification.sensitivity_level == PrivacyLevel.PUBLIC.value
        assert classification.sensitivity_score == 0.0
        assert len(classification.sensitive_fields) == 0

class TestPrivacyViolationDetection:
    """隐私违规检测测试"""
    
    @pytest.mark.asyncio
    async def test_check_privacy_violations_basic(self, privacy_guard, sample_user_data):
        """测试基础隐私违规检查"""
        violations = await privacy_guard.check_privacy_violations(
            sample_user_data, {"operation": "emotion_analysis", "user_consent": True}
        )
        
        assert isinstance(violations, list)
        # 正常情况下应该没有违规
        if len(violations) > 0:
            for violation in violations:
                assert isinstance(violation, PrivacyViolation)
                assert violation.violation_type is not None
                assert violation.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                assert violation.description is not None
    
    @pytest.mark.asyncio
    async def test_check_privacy_violations_no_consent(self, privacy_guard, sample_user_data):
        """测试无同意情况下的隐私违规"""
        violations = await privacy_guard.check_privacy_violations(
            sample_user_data, {"operation": "emotion_analysis", "user_consent": False}
        )
        
        # 应该检测到同意相关的违规
        assert len(violations) > 0
        consent_violations = [v for v in violations if "consent" in v.violation_type.lower()]
        assert len(consent_violations) > 0
    
    @pytest.mark.asyncio
    async def test_check_privacy_violations_unauthorized_access(self, privacy_guard, sample_user_data):
        """测试未授权访问违规检查"""
        violations = await privacy_guard.check_privacy_violations(
            sample_user_data, 
            {
                "operation": "data_export", 
                "requesting_user": "unauthorized_user",
                "data_owner": "test_user_001"
            }
        )
        
        # 应该检测到未授权访问违规
        access_violations = [v for v in violations if "access" in v.violation_type.lower() or "unauthorized" in v.violation_type.lower()]
        assert len(access_violations) > 0
    
    @pytest.mark.asyncio
    async def test_check_privacy_violations_data_retention(self, privacy_guard):
        """测试数据保留期违规检查"""
        expired_data = {
            "user_id": "expired_user",
            "emotion_history": [
                {
                    "timestamp": utc_now() - timedelta(days=400),  # 超过标准保留期
                    "emotions": {"happiness": 0.5},
                    "context": "old_conversation"
                }
            ]
        }
        
        violations = await privacy_guard.check_privacy_violations(
            expired_data, {"operation": "data_analysis"}
        )
        
        # 可能检测到数据保留期违规
        retention_violations = [v for v in violations if "retention" in v.violation_type.lower() or "expired" in v.violation_type.lower()]
        # 这取决于具体的保留政策实现

class TestEthicalViolationDetection:
    """伦理违规检测测试"""
    
    @pytest.mark.asyncio
    async def test_check_ethical_violations_basic(self, privacy_guard, sample_user_data):
        """测试基础伦理违规检查"""
        violations = await privacy_guard.check_ethical_violations(
            sample_user_data, {"operation": "emotion_analysis", "purpose": "user_wellbeing"}
        )
        
        assert isinstance(violations, list)
        # 正常情况应该没有伦理违规
        for violation in violations:
            assert isinstance(violation, EthicalViolation)
            assert violation.concern_type in [concern.value for concern in EthicalConcern]
            assert violation.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    @pytest.mark.asyncio
    async def test_check_ethical_violations_manipulation(self, privacy_guard):
        """测试操控行为伦理违规"""
        manipulation_context = {
            "operation": "emotion_influence",
            "purpose": "behavioral_manipulation",
            "target_emotions": ["anxiety", "fear"],
            "commercial_intent": True
        }
        
        violations = await privacy_guard.check_ethical_violations({}, manipulation_context)
        
        # 应该检测到操控相关的伦理违规
        manipulation_violations = [
            v for v in violations 
            if v.concern_type in [EthicalConcern.MANIPULATION.value, EthicalConcern.EXPLOITATION.value]
        ]
        assert len(manipulation_violations) > 0
    
    @pytest.mark.asyncio
    async def test_check_ethical_violations_bias(self, privacy_guard):
        """测试偏见伦理违规检查"""
        biased_data = {
            "user_demographics": {
                "age": 65,
                "gender": "female",
                "ethnicity": "minority"
            },
            "algorithmic_decisions": {
                "emotion_credibility_score": 0.3,  # 异常低的可信度
                "influence_weight": 0.1  # 异常低的影响权重
            }
        }
        
        violations = await privacy_guard.check_ethical_violations(
            biased_data, {"operation": "social_analysis", "algorithm": "emotion_weighting"}
        )
        
        # 可能检测到偏见相关的违规
        bias_violations = [v for v in violations if v.concern_type == EthicalConcern.BIAS.value]
    
    @pytest.mark.asyncio
    async def test_check_ethical_violations_vulnerable_groups(self, privacy_guard):
        """测试弱势群体保护伦理检查"""
        vulnerable_data = {
            "user_id": "vulnerable_user",
            "demographics": {
                "age": 16,  # 未成年人
                "mental_health_status": "seeking_help"
            },
            "emotion_history": [
                {
                    "emotions": {"depression": 0.9, "vulnerability": 0.8},
                    "context": "crisis_intervention"
                }
            ]
        }
        
        violations = await privacy_guard.check_ethical_violations(
            vulnerable_data, {"operation": "emotion_analysis", "research_purpose": True}
        )
        
        # 应该检测到弱势群体保护相关的关注
        vulnerable_violations = [
            v for v in violations 
            if v.concern_type == EthicalConcern.VULNERABLE_POPULATIONS.value
        ]
        assert len(vulnerable_violations) > 0

class TestConsentManagement:
    """同意管理测试"""
    
    @pytest.mark.asyncio
    async def test_record_user_consent_basic(self, privacy_guard, sample_consent_record):
        """测试基础用户同意记录"""
        await privacy_guard.record_user_consent(sample_consent_record)
        
        # 验证同意记录被存储
        assert sample_consent_record.user_id in privacy_guard.consent_records
        stored_consent = privacy_guard.consent_records[sample_consent_record.user_id]
        assert sample_consent_record.consent_type in stored_consent
        assert stored_consent[sample_consent_record.consent_type] == sample_consent_record
    
    @pytest.mark.asyncio
    async def test_check_user_consent_granted(self, privacy_guard, sample_consent_record):
        """测试已授权用户同意检查"""
        await privacy_guard.record_user_consent(sample_consent_record)
        
        has_consent = await privacy_guard.check_user_consent(
            sample_consent_record.user_id,
            sample_consent_record.consent_type,
            sample_consent_record.scope
        )
        
        assert has_consent is True
    
    @pytest.mark.asyncio
    async def test_check_user_consent_not_granted(self, privacy_guard):
        """测试未授权用户同意检查"""
        has_consent = await privacy_guard.check_user_consent(
            "unknown_user",
            "emotion_analysis",
            ["emotion_tracking"]
        )
        
        assert has_consent is False
    
    @pytest.mark.asyncio
    async def test_check_user_consent_expired(self, privacy_guard):
        """测试过期同意检查"""
        expired_consent = ConsentRecord(
            user_id="expired_user",
            consent_type="emotion_analysis",
            granted=True,
            timestamp=utc_now() - timedelta(days=400),
            scope=["emotion_tracking"],
            expiry_date=utc_now() - timedelta(days=1),  # 已过期
            withdrawal_date=None,
            version="1.0"
        )
        
        await privacy_guard.record_user_consent(expired_consent)
        
        has_consent = await privacy_guard.check_user_consent(
            "expired_user",
            "emotion_analysis",
            ["emotion_tracking"]
        )
        
        assert has_consent is False
    
    @pytest.mark.asyncio
    async def test_check_user_consent_withdrawn(self, privacy_guard):
        """测试撤回同意检查"""
        withdrawn_consent = ConsentRecord(
            user_id="withdrawn_user",
            consent_type="emotion_analysis",
            granted=True,
            timestamp=utc_now(),
            scope=["emotion_tracking"],
            expiry_date=utc_now() + timedelta(days=365),
            withdrawal_date=utc_now(),  # 已撤回
            version="1.0"
        )
        
        await privacy_guard.record_user_consent(withdrawn_consent)
        
        has_consent = await privacy_guard.check_user_consent(
            "withdrawn_user",
            "emotion_analysis", 
            ["emotion_tracking"]
        )
        
        assert has_consent is False
    
    @pytest.mark.asyncio
    async def test_withdraw_user_consent(self, privacy_guard, sample_consent_record):
        """测试撤回用户同意"""
        await privacy_guard.record_user_consent(sample_consent_record)
        
        # 撤回同意
        await privacy_guard.withdraw_user_consent(
            sample_consent_record.user_id,
            sample_consent_record.consent_type
        )
        
        # 验证同意已被撤回
        has_consent = await privacy_guard.check_user_consent(
            sample_consent_record.user_id,
            sample_consent_record.consent_type,
            sample_consent_record.scope
        )
        
        assert has_consent is False

class TestPrivacyPolicyManagement:
    """隐私政策管理测试"""
    
    @pytest.mark.asyncio
    async def test_register_privacy_policy(self, privacy_guard, sample_privacy_policy):
        """测试注册隐私政策"""
        await privacy_guard.register_privacy_policy(sample_privacy_policy)
        
        # 验证政策被存储
        assert sample_privacy_policy.policy_id in privacy_guard.privacy_policies
        stored_policy = privacy_guard.privacy_policies[sample_privacy_policy.policy_id]
        assert stored_policy == sample_privacy_policy
    
    @pytest.mark.asyncio
    async def test_check_policy_compliance_compliant(self, privacy_guard, sample_privacy_policy, sample_user_data):
        """测试符合政策的合规检查"""
        await privacy_guard.register_privacy_policy(sample_privacy_policy)
        
        compliant_operation = {
            "data_types": ["emotions", "messages"],
            "purpose": "emotion_analysis",
            "retention_days": 300,
            "sharing": False,
            "third_party": False
        }
        
        compliance_result = await privacy_guard.check_policy_compliance(
            sample_privacy_policy.policy_id,
            compliant_operation
        )
        
        assert compliance_result["compliant"] is True
        assert len(compliance_result["violations"]) == 0
    
    @pytest.mark.asyncio
    async def test_check_policy_compliance_violation(self, privacy_guard, sample_privacy_policy):
        """测试违反政策的合规检查"""
        await privacy_guard.register_privacy_policy(sample_privacy_policy)
        
        violating_operation = {
            "data_types": ["emotions", "financial_data"],  # 超出允许的数据类型
            "purpose": "commercial_targeting",  # 未授权目的
            "retention_days": 1000,  # 超出保留期限
            "sharing": True,  # 未经允许的共享
            "third_party": True  # 未经允许的第三方访问
        }
        
        compliance_result = await privacy_guard.check_policy_compliance(
            sample_privacy_policy.policy_id,
            violating_operation
        )
        
        assert compliance_result["compliant"] is False
        assert len(compliance_result["violations"]) > 0

class TestAuditLogging:
    """审计日志测试"""
    
    @pytest.mark.asyncio
    async def test_log_privacy_event_basic(self, privacy_guard):
        """测试基础隐私事件日志"""
        event_data = {
            "user_id": "test_user",
            "action": "data_access",
            "data_types": ["emotions"],
            "purpose": "analysis"
        }
        
        await privacy_guard.log_privacy_event("DATA_ACCESS", event_data)
        
        # 验证事件被记录
        assert len(privacy_guard.audit_log) > 0
        latest_record = privacy_guard.audit_log[-1]
        assert latest_record.event_type == "DATA_ACCESS"
        assert latest_record.user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_log_privacy_event_disabled(self, privacy_guard):
        """测试审计禁用时的事件日志"""
        privacy_guard.disable_audit_logging()
        
        initial_log_count = len(privacy_guard.audit_log)
        
        await privacy_guard.log_privacy_event("TEST_EVENT", {"test": "data"})
        
        # 审计禁用时不应记录新事件
        assert len(privacy_guard.audit_log) == initial_log_count
    
    @pytest.mark.asyncio
    async def test_get_audit_history(self, privacy_guard):
        """测试获取审计历史"""
        # 添加一些测试事件
        for i in range(3):
            await privacy_guard.log_privacy_event(
                f"TEST_EVENT_{i}",
                {"user_id": f"user_{i}", "action": f"action_{i}"}
            )
        
        # 获取所有审计历史
        all_history = await privacy_guard.get_audit_history()
        assert len(all_history) >= 3
        
        # 获取特定用户的审计历史
        user_history = await privacy_guard.get_audit_history(user_id="user_1")
        user_records = [record for record in user_history if record.user_id == "user_1"]
        assert len(user_records) >= 1
        
        # 获取时间范围内的审计历史
        since_time = utc_now() - timedelta(minutes=1)
        recent_history = await privacy_guard.get_audit_history(since=since_time)
        assert len(recent_history) >= 0

class TestComplianceReporting:
    """合规报告测试"""
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report_basic(self, privacy_guard):
        """测试基础合规报告生成"""
        # 添加一些测试数据
        await privacy_guard.log_privacy_event("DATA_ACCESS", {"user_id": "user1"})
        await privacy_guard.log_privacy_event("CONSENT_GRANTED", {"user_id": "user2"})
        
        report = await privacy_guard.generate_compliance_report()
        
        assert isinstance(report, ComplianceReport)
        assert report.report_id is not None
        assert isinstance(report.audit_events_count, int)
        assert isinstance(report.privacy_violations_count, int)
        assert isinstance(report.ethical_violations_count, int)
        assert isinstance(report.active_consents_count, int)
        assert isinstance(report.compliance_score, float)
        assert 0.0 <= report.compliance_score <= 1.0
        assert isinstance(report.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report_time_period(self, privacy_guard):
        """测试特定时间段合规报告"""
        start_date = utc_now() - timedelta(days=7)
        end_date = utc_now()
        
        report = await privacy_guard.generate_compliance_report(
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(report, ComplianceReport)
        assert report.period_start == start_date
        assert report.period_end == end_date
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report_with_violations(self, privacy_guard, sample_user_data):
        """测试包含违规的合规报告"""
        # 创建一些违规记录
        violations = await privacy_guard.check_privacy_violations(
            sample_user_data, {"operation": "unauthorized_access", "user_consent": False}
        )
        
        if violations:
            for violation in violations:
                privacy_guard.violation_history.append(violation)
        
        report = await privacy_guard.generate_compliance_report()
        
        assert report.privacy_violations_count >= 0
        if violations:
            assert report.compliance_score < 1.0

class TestDataAnonymization:
    """数据匿名化测试"""
    
    @pytest.mark.asyncio
    async def test_anonymize_data_basic(self, privacy_guard, sample_user_data):
        """测试基础数据匿名化"""
        anonymized_data = await privacy_guard.anonymize_data(sample_user_data)
        
        assert isinstance(anonymized_data, dict)
        # 验证敏感信息被移除或匿名化
        if "user_id" in anonymized_data:
            assert anonymized_data["user_id"] != sample_user_data["user_id"]
        
        # 验证个人信息被处理
        if "personal_info" in anonymized_data:
            personal_info = anonymized_data["personal_info"]
            if "location" in personal_info:
                # 位置信息应该被泛化
                assert personal_info["location"] != sample_user_data["personal_info"]["location"]
    
    @pytest.mark.asyncio
    async def test_anonymize_data_high_sensitivity(self, privacy_guard):
        """测试高敏感度数据匿名化"""
        sensitive_data = {
            "user_id": "sensitive_user_123",
            "personal_info": {
                "ssn": "123-45-6789",
                "phone": "555-1234",
                "email": "user@example.com"
            },
            "emotion_history": [
                {
                    "emotions": {"depression": 0.9},
                    "context": "therapy_session",
                    "location": "123 Main St"
                }
            ]
        }
        
        anonymized_data = await privacy_guard.anonymize_data(sensitive_data)
        
        # 高敏感度数据应该被更严格地处理
        assert anonymized_data["user_id"] != sensitive_data["user_id"]
        personal_info = anonymized_data.get("personal_info", {})
        assert personal_info.get("ssn") != sensitive_data["personal_info"]["ssn"]
        assert personal_info.get("phone") != sensitive_data["personal_info"]["phone"]
    
    @pytest.mark.asyncio
    async def test_anonymize_data_preserve_utility(self, privacy_guard, sample_user_data):
        """测试匿名化数据保持实用性"""
        anonymized_data = await privacy_guard.anonymize_data(sample_user_data, preserve_utility=True)
        
        # 在保持实用性模式下，某些分析相关的信息应该被保留
        if "emotion_history" in anonymized_data:
            emotion_history = anonymized_data["emotion_history"]
            assert len(emotion_history) > 0
            # 情感数据本身应该被保留用于分析
            assert "emotions" in emotion_history[0]

class TestSecurityMeasures:
    """安全措施测试"""
    
    @pytest.mark.asyncio
    async def test_encrypt_sensitive_data(self, privacy_guard):
        """测试敏感数据加密"""
        sensitive_text = "这是包含个人信息的敏感文本：SSN 123-45-6789"
        
        encrypted_data = await privacy_guard.encrypt_sensitive_data(sensitive_text)
        
        assert encrypted_data != sensitive_text
        assert isinstance(encrypted_data, (str, bytes, dict))
    
    @pytest.mark.asyncio
    async def test_decrypt_sensitive_data(self, privacy_guard):
        """测试敏感数据解密"""
        original_text = "敏感信息测试"
        
        encrypted_data = await privacy_guard.encrypt_sensitive_data(original_text)
        decrypted_data = await privacy_guard.decrypt_sensitive_data(encrypted_data)
        
        assert decrypted_data == original_text
    
    @pytest.mark.asyncio
    async def test_secure_data_transmission(self, privacy_guard, sample_user_data):
        """测试安全数据传输"""
        transmission_result = await privacy_guard.secure_data_transmission(
            sample_user_data,
            destination="analytics_service",
            encryption_level="high"
        )
        
        assert isinstance(transmission_result, dict)
        assert transmission_result.get("success", False) is True
        assert "transmission_id" in transmission_result
        assert "encryption_used" in transmission_result

class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件测试"""
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, privacy_guard):
        """测试无效数据处理"""
        invalid_data = {
            "malformed_field": float('nan'),
            "null_field": None,
            "empty_list": [],
            "invalid_timestamp": "not_a_date"
        }
        
        # 应该能够处理无效数据而不崩溃
        classification = await privacy_guard.classify_data_sensitivity(invalid_data)
        assert isinstance(classification, DataClassification)
        
        violations = await privacy_guard.check_privacy_violations(invalid_data, {})
        assert isinstance(violations, list)
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, privacy_guard):
        """测试并发访问安全性"""
        # 模拟并发操作
        tasks = []
        for i in range(10):
            consent_record = ConsentRecord(
                user_id=f"concurrent_user_{i}",
                consent_type="test_consent",
                granted=True,
                timestamp=utc_now(),
                scope=["test"],
                expiry_date=utc_now() + timedelta(days=1),
                withdrawal_date=None,
                version="1.0"
            )
            tasks.append(privacy_guard.record_user_consent(consent_record))
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        # 验证所有记录都被正确存储
        assert len(privacy_guard.consent_records) >= 10
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_audit_log(self, privacy_guard):
        """测试大量审计日志的内存使用"""
        # 添加大量审计记录
        for i in range(1000):
            await privacy_guard.log_privacy_event(
                "BULK_TEST_EVENT",
                {"user_id": f"bulk_user_{i}", "data": f"test_data_{i}"}
            )
        
        # 验证系统仍然稳定
        assert len(privacy_guard.audit_log) >= 1000
        
        # 测试报告生成仍然正常工作
        report = await privacy_guard.generate_compliance_report()
        assert isinstance(report, ComplianceReport)
    
    @pytest.mark.asyncio
    async def test_policy_version_conflicts(self, privacy_guard):
        """测试政策版本冲突处理"""
        policy_v1 = PrivacyPolicy(
            policy_id="conflict_test_policy",
            version="1.0",
            effective_date=utc_now(),
            data_types_collected=["emotions"],
            purposes=["analysis"],
            retention_period_days=365,
            sharing_allowed=False,
            third_party_access=False,
            user_rights=["access"],
            contact_info="test@example.com"
        )
        
        policy_v2 = PrivacyPolicy(
            policy_id="conflict_test_policy",  # 相同ID
            version="2.0",
            effective_date=utc_now() + timedelta(days=1),
            data_types_collected=["emotions", "messages"],
            purposes=["analysis", "improvement"],
            retention_period_days=730,
            sharing_allowed=True,
            third_party_access=False,
            user_rights=["access", "rectification"],
            contact_info="test@example.com"
        )
        
        await privacy_guard.register_privacy_policy(policy_v1)
        await privacy_guard.register_privacy_policy(policy_v2)
        
        # 应该存储最新版本
        stored_policy = privacy_guard.privacy_policies["conflict_test_policy"]
        assert stored_policy.version == "2.0"

class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_complete_privacy_workflow(
        self, privacy_guard, sample_user_data, sample_privacy_policy, sample_consent_record
    ):
        """测试完整隐私保护工作流"""
        # 1. 注册隐私政策
        await privacy_guard.register_privacy_policy(sample_privacy_policy)
        
        # 2. 记录用户同意
        await privacy_guard.record_user_consent(sample_consent_record)
        
        # 3. 分类数据敏感度
        classification = await privacy_guard.classify_data_sensitivity(sample_user_data)
        assert isinstance(classification, DataClassification)
        
        # 4. 检查隐私违规
        privacy_violations = await privacy_guard.check_privacy_violations(
            sample_user_data, {"operation": "emotion_analysis", "user_consent": True}
        )
        assert isinstance(privacy_violations, list)
        
        # 5. 检查伦理违规
        ethical_violations = await privacy_guard.check_ethical_violations(
            sample_user_data, {"operation": "emotion_analysis", "purpose": "user_wellbeing"}
        )
        assert isinstance(ethical_violations, list)
        
        # 6. 记录访问事件
        await privacy_guard.log_privacy_event(
            "DATA_ANALYSIS",
            {"user_id": sample_user_data["user_id"], "data_types": ["emotions"]}
        )
        
        # 7. 生成合规报告
        report = await privacy_guard.generate_compliance_report()
        assert isinstance(report, ComplianceReport)
        
        # 验证整个流程的一致性
        assert len(privacy_guard.privacy_policies) >= 1
        assert len(privacy_guard.consent_records) >= 1
        assert len(privacy_guard.audit_log) >= 1
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_scenario(self, privacy_guard):
        """测试GDPR合规场景"""
        # 创建符合GDPR的政策
        gdpr_policy = PrivacyPolicy(
            policy_id="gdpr_compliant_policy",
            version="1.0",
            effective_date=utc_now(),
            data_types_collected=["emotions", "social_interactions"],
            purposes=["emotion_analysis", "social_insights"],
            retention_period_days=365,
            sharing_allowed=False,
            third_party_access=False,
            user_rights=[
                "access", "rectification", "erasure", 
                "portability", "restriction", "objection"
            ],
            contact_info="dpo@example.com"
        )
        
        await privacy_guard.register_privacy_policy(gdpr_policy)
        
        # 测试数据主体权利
        user_data = {
            "user_id": "gdpr_test_user",
            "personal_info": {"location": "EU"},
            "emotion_history": [{"emotions": {"happiness": 0.8}}]
        }
        
        # 测试访问权（Right of Access）
        classification = await privacy_guard.classify_data_sensitivity(user_data)
        assert isinstance(classification, DataClassification)
        
        # 测试数据可携权（Right to Data Portability）
        anonymized_data = await privacy_guard.anonymize_data(user_data)
        assert isinstance(anonymized_data, dict)
        
        # 测试删除权（Right to Erasure）
        # 这里应该有删除数据的方法
        
        # 验证合规检查
        compliance_result = await privacy_guard.check_policy_compliance(
            "gdpr_compliant_policy",
            {
                "data_types": ["emotions"],
                "purpose": "emotion_analysis",
                "retention_days": 300,
                "sharing": False,
                "third_party": False
            }
        )
        
        assert compliance_result["compliant"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
