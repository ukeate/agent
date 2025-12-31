"""
安全合规认证和测试框架
实现企业级安全合规检查、认证和自动化测试
"""

import asyncio
import json
import hashlib
import ssl
import socket
import shutil
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from enum import Enum
import subprocess
import tempfile
import os
import re
from .security.trism import AITRiSMFramework, SecurityEvent, ThreatLevel
from .security.attack_detection import AttackDetectionManager, AttackType
from .security.auto_response import SecurityResponseManager
from .monitoring import AuditEventType, AuditLevel
from src.core.config import get_settings
from src.core.database import get_db_session, test_database_connection
from src.core.redis import get_redis, test_redis_connection
from sqlalchemy import text
import time

from src.core.logging import get_logger
logger = get_logger(__name__)

class ComplianceStandard(str, Enum):
    """合规标准"""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    AI_GOVERNANCE = "ai_governance"
    CUSTOM = "custom"

class TestSeverity(str, Enum):
    """测试严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceStatus(str, Enum):
    """合规状态"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_TESTED = "not_tested"
    REMEDIATION_REQUIRED = "remediation_required"

@dataclass
class ComplianceRequirement:
    """合规要求"""
    id: str
    title: str
    description: str
    standard: ComplianceStandard
    severity: TestSeverity
    category: str
    test_procedures: List[str] = field(default_factory=list)
    evidence_required: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "standard": self.standard.value,
            "severity": self.severity.value,
            "category": self.category,
            "test_procedures": self.test_procedures,
            "evidence_required": self.evidence_required,
            "remediation_steps": self.remediation_steps
        }

@dataclass
class TestResult:
    """测试结果"""
    requirement_id: str
    test_name: str
    status: ComplianceStatus
    score: float  # 0-100
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: utc_now())
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "score": self.score,
            "details": self.details,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time
        }

@dataclass
class ComplianceReport:
    """合规报告"""
    report_id: str
    standards: List[ComplianceStandard]
    overall_score: float
    status: ComplianceStatus
    test_results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: utc_now())
    generated_by: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "standards": [s.value for s in self.standards],
            "overall_score": self.overall_score,
            "status": self.status.value,
            "test_results": [r.to_dict() for r in self.test_results],
            "summary": self.summary,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by
        }

class SecurityComplianceTests:
    """安全合规测试"""
    
    def __init__(
        self,
        trism_framework: Optional[AITRiSMFramework] = None,
        attack_detector: Optional[AttackDetectionManager] = None,
        response_manager: Optional[SecurityResponseManager] = None
    ):
        self.trism_framework = trism_framework
        self.attack_detector = attack_detector
        self.response_manager = response_manager
        
        logger.info("安全合规测试初始化")
    
    async def test_data_encryption(self) -> TestResult:
        """测试数据加密"""
        start_time = time.time()
        result = TestResult(
            requirement_id="SEC-001",
            test_name="Data Encryption at Rest and in Transit",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            recommendations = []
            
            # 测试SSL/TLS配置
            ssl_score = await self._test_ssl_configuration()
            details["ssl_configuration"] = ssl_score
            score += ssl_score * 0.4
            
            # 测试数据存储加密
            storage_score = await self._test_storage_encryption()
            details["storage_encryption"] = storage_score
            score += storage_score * 0.3
            
            # 测试传输加密
            transport_score = await self._test_transport_encryption()
            details["transport_encryption"] = transport_score
            score += transport_score * 0.3
            
            # 评估总分
            if score >= 90:
                result.status = ComplianceStatus.COMPLIANT
            elif score >= 70:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
                recommendations.append("需要加强加密配置")
            else:
                result.status = ComplianceStatus.NON_COMPLIANT
                recommendations.append("严重的加密配置问题")
            
            result.score = score
            result.details = details
            result.recommendations = recommendations
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
            logger.error("数据加密测试失败", error=str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_ssl_configuration(self) -> float:
        """测试SSL配置"""
        try:
            if not get_settings().FORCE_HTTPS:
                return 0.0
            # 检查SSL上下文
            context = ssl.create_default_context()
            score = 0.0
            
            # 检查协议版本
            if hasattr(ssl, 'TLSVersion'):
                if context.minimum_version >= ssl.TLSVersion.TLSv1_2:
                    score += 50.0
                if context.minimum_version >= ssl.TLSVersion.TLSv1_3:
                    score += 50.0
            
            return min(score, 100.0)
        except Exception:
            return 0.0
    
    async def _test_storage_encryption(self) -> float:
        """测试存储加密"""
        try:
            dsn = get_settings().DATABASE_URL
            if any(s in dsn for s in ["sslmode=require", "sslmode=verify-full", "sslmode=verify-ca", "ssl=true"]):
                return 100.0
            return 0.0
        except Exception:
            return 0.0
    
    async def _test_transport_encryption(self) -> float:
        """测试传输加密"""
        try:
            return 100.0 if get_settings().FORCE_HTTPS else 0.0
        except Exception:
            return 0.0
    
    async def test_access_control(self) -> TestResult:
        """测试访问控制"""
        start_time = time.time()
        result = TestResult(
            requirement_id="SEC-002",
            test_name="Access Control and Authentication",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            
            # 测试身份验证
            auth_score = await self._test_authentication()
            details["authentication"] = auth_score
            score += auth_score * 0.4
            
            # 测试授权机制
            authz_score = await self._test_authorization()
            details["authorization"] = authz_score
            score += authz_score * 0.3
            
            # 测试会话管理
            session_score = await self._test_session_management()
            details["session_management"] = session_score
            score += session_score * 0.3
            
            result.score = score
            result.details = details
            
            if score >= 85:
                result.status = ComplianceStatus.COMPLIANT
            elif score >= 65:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                result.status = ComplianceStatus.NON_COMPLIANT
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_authentication(self) -> float:
        """测试身份验证"""
        try:
            settings = get_settings()
            score = 0.0
            if settings.SECRET_KEY and len(settings.SECRET_KEY) >= 32:
                score += 40.0
            if 5 <= settings.ACCESS_TOKEN_EXPIRE_MINUTES <= 120:
                score += 20.0
            if 1 <= settings.REFRESH_TOKEN_EXPIRE_DAYS <= 30:
                score += 20.0
            if 0.0 < settings.SECURITY_THRESHOLD < 1.0:
                score += 20.0
            return score
        except Exception:
            return 0.0
    
    async def _test_authorization(self) -> float:
        """测试授权机制"""
        try:
            redis_client = get_redis()
            if not redis_client:
                return 0.0
            total = await redis_client.zcard("acl:rules")
            return min(100.0, float(total) * 20.0) if total else 0.0
        except Exception:
            return 0.0
    
    async def _test_session_management(self) -> float:
        """测试会话管理"""
        try:
            settings = get_settings()
            score = 0.0
            if 1 <= settings.SESSION_TIMEOUT_MINUTES <= 1440:
                score += 50.0
            if 1 <= settings.ACCESS_TOKEN_EXPIRE_MINUTES <= settings.SESSION_TIMEOUT_MINUTES:
                score += 25.0
            if settings.REFRESH_TOKEN_EXPIRE_DAYS >= 1:
                score += 25.0
            return score
        except Exception:
            return 0.0
    
    async def test_ai_security(self) -> TestResult:
        """测试AI安全性"""
        start_time = time.time()
        result = TestResult(
            requirement_id="AI-001",
            test_name="AI Security and Governance",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            
            # 测试AI TRiSM框架
            if self.trism_framework:
                trism_score = await self._test_trism_framework()
                details["trism_framework"] = trism_score
                score += trism_score * 0.3
            
            # 测试攻击检测
            if self.attack_detector:
                attack_detection_score = await self._test_attack_detection()
                details["attack_detection"] = attack_detection_score
                score += attack_detection_score * 0.3
            
            # 测试安全响应
            if self.response_manager:
                response_score = await self._test_security_response()
                details["security_response"] = response_score
                score += response_score * 0.2
            
            # 测试模型安全
            model_security_score = await self._test_model_security()
            details["model_security"] = model_security_score
            score += model_security_score * 0.2
            
            result.score = score
            result.details = details
            
            if score >= 80:
                result.status = ComplianceStatus.COMPLIANT
            elif score >= 60:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                result.status = ComplianceStatus.NON_COMPLIANT
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_trism_framework(self) -> float:
        """测试TRiSM框架"""
        if not self.trism_framework:
            return 0.0
        
        try:
            # 测试评估功能
            evaluation = await self.trism_framework.evaluate_agent_output(
                "test-agent",
                "test output",
                {"test": True}
            )

            trust_score = float(((evaluation or {}).get("trust") or {}).get("trust_score") or 0.0)
            return max(0.0, min(100.0, trust_score * 100.0))
        except Exception:
            return 0.0
    
    async def _test_attack_detection(self) -> float:
        """测试攻击检测"""
        if not self.attack_detector:
            return 0.0
        
        try:
            # 测试各种攻击检测
            test_cases = [
                ("prompt injection test", AttackType.PROMPT_INJECTION),
                ("sensitive data: 123-45-6789", AttackType.DATA_LEAKAGE),
            ]
            
            detected_count = 0
            for test_input, expected_type in test_cases:
                results = await self.attack_detector.detect_attacks(
                    "test-agent",
                    test_input,
                    {"test": True},
                    attack_types=[expected_type],
                )

                result = results.get(expected_type)
                if result and result.attack_detected:
                    detected_count += 1
            
            return (detected_count / len(test_cases)) * 100
            
        except Exception:
            return 0.0
    
    async def _test_security_response(self) -> float:
        """测试安全响应"""
        if not self.response_manager:
            return 0.0
        
        try:
            stats = self.response_manager.get_statistics()
            total = int(((stats or {}).get("rules") or {}).get("total") or 0)
            enabled = int(((stats or {}).get("rules") or {}).get("enabled") or 0)
            return (enabled / total) * 100 if total else 0.0
        except Exception:
            return 0.0
    
    async def _test_model_security(self) -> float:
        """测试模型安全"""
        settings = get_settings()
        score = 0.0
        if not settings.DEBUG:
            score += 25.0
        if settings.FORCE_HTTPS:
            score += 25.0
        if settings.SECRET_KEY and len(settings.SECRET_KEY) >= 32:
            score += 25.0
        if settings.CSP_HEADER:
            score += 25.0
        return min(100.0, score)

class DataPrivacyTests:
    """数据隐私测试"""
    
    def __init__(self):
        from src.ai.emotion_modeling.privacy_ethics_guard import PrivacyEthicsGuard

        self.guard = PrivacyEthicsGuard()
        logger.info("数据隐私测试初始化")
    
    async def test_gdpr_compliance(self) -> TestResult:
        """测试GDPR合规性"""
        start_time = time.time()
        result = TestResult(
            requirement_id="GDPR-001",
            test_name="GDPR Data Protection Compliance",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            
            # 测试数据收集合规性
            collection_score = await self._test_data_collection()
            details["data_collection"] = collection_score
            score += collection_score * 0.25
            
            # 测试数据处理合规性
            processing_score = await self._test_data_processing()
            details["data_processing"] = processing_score
            score += processing_score * 0.25
            
            # 测试数据存储合规性
            storage_score = await self._test_data_storage()
            details["data_storage"] = storage_score
            score += storage_score * 0.25
            
            # 测试用户权利实现
            rights_score = await self._test_user_rights()
            details["user_rights"] = rights_score
            score += rights_score * 0.25
            
            result.score = score
            result.details = details
            
            if score >= 90:
                result.status = ComplianceStatus.COMPLIANT
            elif score >= 70:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                result.status = ComplianceStatus.NON_COMPLIANT
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_data_collection(self) -> float:
        """测试数据收集"""
        try:
            from src.ai.emotion_modeling.privacy_ethics_guard import PrivacyLevel, ConsentType

            user_id = "gdpr_test_user"
            purpose = "gdpr_test"
            await self.guard.create_privacy_policy(
                user_id=user_id,
                privacy_level=PrivacyLevel.CONFIDENTIAL,
                data_retention_days=180,
                cultural_context="eu",
            )
            await self.guard.manage_consent(
                user_id=user_id,
                consent_type=ConsentType.EXPLICIT,
                data_categories=["emotional_data"],
                purpose=purpose,
            )
            return 100.0
        except Exception:
            return 0.0
    
    async def _test_data_processing(self) -> float:
        """测试数据处理"""
        try:
            user_id = "gdpr_test_user"
            purpose = "gdpr_test"
            compliant, _violations = await self.guard.validate_privacy_compliance(
                user_id=user_id,
                emotion_data={"text": "test"},
                processing_purpose=purpose,
                cultural_context="eu",
            )
            return 100.0 if compliant else 0.0
        except Exception:
            return 0.0
    
    async def _test_data_storage(self) -> float:
        """测试数据存储"""
        try:
            policy = self.guard.privacy_policies.get("gdpr_test_user")
            if not policy:
                return 0.0
            return 100.0 if 0 < policy.data_retention_days <= 365 else 0.0
        except Exception:
            return 0.0
    
    async def _test_user_rights(self) -> float:
        """测试用户权利"""
        try:
            ok = await self.guard.withdraw_consent(
                user_id="gdpr_test_user",
                data_categories=["emotional_data"],
                purpose="gdpr_test",
            )
            return 100.0 if ok else 0.0
        except Exception:
            return 0.0

class OperationalSecurityTests:
    """运营安全测试"""
    
    def __init__(self):
        logger.info("运营安全测试初始化")
    
    async def test_incident_response(self) -> TestResult:
        """测试事件响应"""
        start_time = time.time()
        result = TestResult(
            requirement_id="OPS-001",
            test_name="Incident Response Procedures",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            
            # 测试事件检测
            detection_score = await self._test_incident_detection()
            details["incident_detection"] = detection_score
            score += detection_score * 0.3
            
            # 测试响应程序
            response_score = await self._test_response_procedures()
            details["response_procedures"] = response_score
            score += response_score * 0.4
            
            # 测试恢复能力
            recovery_score = await self._test_recovery_capabilities()
            details["recovery_capabilities"] = recovery_score
            score += recovery_score * 0.3
            
            result.score = score
            result.details = details
            
            if score >= 85:
                result.status = ComplianceStatus.COMPLIANT
            else:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_incident_detection(self) -> float:
        """测试事件检测"""
        try:
            async with get_db_session() as session:
                result = await session.execute(text("SELECT to_regclass('public.audit_logs')"))
                return 100.0 if result.scalar() else 0.0
        except Exception:
            return 0.0
    
    async def _test_response_procedures(self) -> float:
        """测试响应程序"""
        try:
            settings = get_settings()
            score = 0.0
            if settings.CSP_HEADER:
                score += 25.0
            if 0.0 < settings.SECURITY_THRESHOLD < 1.0:
                score += 25.0
            if 0.0 < settings.AUTO_BLOCK_THRESHOLD <= 1.0:
                score += 25.0
            if settings.MAX_REQUESTS_PER_MINUTE > 0:
                score += 25.0
            return score
        except Exception:
            return 0.0
    
    async def _test_recovery_capabilities(self) -> float:
        """测试恢复能力"""
        try:
            db_ok = await test_database_connection()
            redis_ok = await test_redis_connection()
            return ((1.0 if db_ok else 0.0) + (1.0 if redis_ok else 0.0)) / 2.0 * 100.0
        except Exception:
            return 0.0
    
    async def test_backup_and_recovery(self) -> TestResult:
        """测试备份和恢复"""
        start_time = time.time()
        result = TestResult(
            requirement_id="OPS-002",
            test_name="Backup and Recovery Systems",
            status=ComplianceStatus.NOT_TESTED,
            score=0.0,
        )
        
        try:
            score = 0.0
            details = {}
            
            # 测试备份策略
            backup_score = await self._test_backup_strategy()
            details["backup_strategy"] = backup_score
            score += backup_score * 0.5
            
            # 测试恢复测试
            recovery_test_score = await self._test_recovery_testing()
            details["recovery_testing"] = recovery_test_score
            score += recovery_test_score * 0.5
            
            result.score = score
            result.details = details
            
            if score >= 90:
                result.status = ComplianceStatus.COMPLIANT
            else:
                result.status = ComplianceStatus.PARTIALLY_COMPLIANT
            
        except Exception as e:
            result.status = ComplianceStatus.NOT_TESTED
            result.details = {"error": str(e)}
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _test_backup_strategy(self) -> float:
        """测试备份策略"""
        return 100.0 if shutil.which("pg_dump") else 0.0
    
    async def _test_recovery_testing(self) -> float:
        """测试恢复测试"""
        return 100.0 if shutil.which("pg_restore") else 0.0

class ComplianceFramework:
    """合规框架"""
    
    def __init__(
        self,
        standards: List[ComplianceStandard],
        trism_framework: Optional[AITRiSMFramework] = None,
        attack_detector: Optional[AttackDetectionManager] = None,
        response_manager: Optional[SecurityResponseManager] = None
    ):
        self.standards = standards
        self.trism_framework = trism_framework
        self.attack_detector = attack_detector
        self.response_manager = response_manager
        
        # 初始化测试套件
        self.security_tests = SecurityComplianceTests(
            trism_framework, attack_detector, response_manager
        )
        self.privacy_tests = DataPrivacyTests()
        self.ops_tests = OperationalSecurityTests()
        
        # 合规要求库
        self.requirements = self._initialize_requirements()
        
        # 测试历史
        self.test_history: List[ComplianceReport] = []
        
        logger.info("合规框架初始化", standards=[s.value for s in standards])
    
    def _initialize_requirements(self) -> Dict[str, ComplianceRequirement]:
        """初始化合规要求"""
        requirements = {}
        
        # 安全要求
        requirements["SEC-001"] = ComplianceRequirement(
            id="SEC-001",
            title="数据加密",
            description="确保所有敏感数据在传输和存储时都经过加密",
            standard=ComplianceStandard.ISO27001,
            severity=TestSeverity.CRITICAL,
            category="加密",
            test_procedures=["检查SSL/TLS配置", "验证存储加密", "测试传输加密"],
            evidence_required=["加密证书", "配置文件", "测试报告"],
            remediation_steps=["更新SSL证书", "启用数据库加密", "配置安全传输"]
        )
        
        requirements["SEC-002"] = ComplianceRequirement(
            id="SEC-002",
            title="访问控制",
            description="实施适当的身份验证和授权机制",
            standard=ComplianceStandard.SOC2,
            severity=TestSeverity.HIGH,
            category="访问控制",
            test_procedures=["测试身份验证", "验证授权机制", "检查会话管理"],
            evidence_required=["访问日志", "权限配置", "会话记录"],
            remediation_steps=["实施多因素认证", "更新权限矩阵", "加强会话安全"]
        )
        
        # AI特定要求
        requirements["AI-001"] = ComplianceRequirement(
            id="AI-001",
            title="AI安全治理",
            description="确保AI系统的安全性和可控性",
            standard=ComplianceStandard.AI_GOVERNANCE,
            severity=TestSeverity.HIGH,
            category="AI安全",
            test_procedures=["测试TRiSM框架", "验证攻击检测", "检查安全响应"],
            evidence_required=["安全评估报告", "检测日志", "响应记录"],
            remediation_steps=["优化TRiSM配置", "增强检测规则", "改进响应机制"]
        )
        
        # 隐私要求
        requirements["GDPR-001"] = ComplianceRequirement(
            id="GDPR-001",
            title="GDPR数据保护",
            description="符合GDPR数据保护要求",
            standard=ComplianceStandard.GDPR,
            severity=TestSeverity.CRITICAL,
            category="数据保护",
            test_procedures=["检查数据收集", "验证处理合法性", "测试用户权利"],
            evidence_required=["隐私政策", "同意记录", "数据流图"],
            remediation_steps=["更新隐私政策", "实施用户权利", "优化数据处理"]
        )
        
        # 运营要求
        requirements["OPS-001"] = ComplianceRequirement(
            id="OPS-001",
            title="事件响应",
            description="建立有效的安全事件响应机制",
            standard=ComplianceStandard.NIST_CYBERSECURITY,
            severity=TestSeverity.HIGH,
            category="事件响应",
            test_procedures=["测试检测能力", "验证响应程序", "检查恢复机制"],
            evidence_required=["响应计划", "演练记录", "恢复测试"],
            remediation_steps=["完善响应计划", "加强检测能力", "优化恢复程序"]
        )
        
        requirements["OPS-002"] = ComplianceRequirement(
            id="OPS-002",
            title="备份恢复",
            description="确保数据备份和灾难恢复能力",
            standard=ComplianceStandard.ISO27001,
            severity=TestSeverity.MEDIUM,
            category="业务连续性",
            test_procedures=["测试备份策略", "验证恢复程序"],
            evidence_required=["备份日志", "恢复测试报告"],
            remediation_steps=["优化备份策略", "改进恢复程序"]
        )
        
        return requirements
    
    async def run_compliance_assessment(
        self,
        target_standards: Optional[List[ComplianceStandard]] = None
    ) -> ComplianceReport:
        """运行合规评估"""
        logger.info("开始合规评估")
        
        if not target_standards:
            target_standards = self.standards
        
        report_id = f"compliance-{int(utc_now().timestamp())}"
        test_results = []
        
        # 运行安全测试
        for requirement_id, requirement in self.requirements.items():
            if requirement.standard in target_standards:
                result = await self._run_requirement_test(requirement)
                test_results.append(result)
        
        # 计算总分
        if test_results:
            overall_score = sum(r.score for r in test_results) / len(test_results)
        else:
            overall_score = 0.0
        
        # 确定整体状态
        if overall_score >= 90:
            overall_status = ComplianceStatus.COMPLIANT
        elif overall_score >= 70:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT
        
        # 生成摘要
        summary = self._generate_summary(test_results, target_standards)
        
        # 创建报告
        report = ComplianceReport(
            report_id=report_id,
            standards=target_standards,
            overall_score=overall_score,
            status=overall_status,
            test_results=test_results,
            summary=summary
        )
        
        # 保存历史
        self.test_history.append(report)
        if len(self.test_history) > 100:  # 保持最近100个报告
            self.test_history = self.test_history[-100:]
        
        logger.info("合规评估完成", 
                   overall_score=overall_score, 
                   status=overall_status.value)
        
        return report
    
    async def _run_requirement_test(self, requirement: ComplianceRequirement) -> TestResult:
        """运行特定要求的测试"""
        try:
            if requirement.id == "SEC-001":
                return await self.security_tests.test_data_encryption()
            elif requirement.id == "SEC-002":
                return await self.security_tests.test_access_control()
            elif requirement.id == "AI-001":
                return await self.security_tests.test_ai_security()
            elif requirement.id == "GDPR-001":
                return await self.privacy_tests.test_gdpr_compliance()
            elif requirement.id == "OPS-001":
                return await self.ops_tests.test_incident_response()
            elif requirement.id == "OPS-002":
                return await self.ops_tests.test_backup_and_recovery()
            else:
                # 默认测试结果
                return TestResult(
                    requirement_id=requirement.id,
                    test_name=requirement.title,
                    status=ComplianceStatus.NOT_TESTED,
                    score=0.0,
                    details={"reason": "测试未实现"}
                )
        except Exception as e:
            logger.error("要求测试失败", requirement_id=requirement.id, error=str(e))
            return TestResult(
                requirement_id=requirement.id,
                test_name=requirement.title,
                status=ComplianceStatus.NOT_TESTED,
                score=0.0,
                details={"error": str(e)}
            )
    
    def _generate_summary(
        self,
        test_results: List[TestResult],
        standards: List[ComplianceStandard]
    ) -> Dict[str, Any]:
        """生成摘要"""
        summary = {
            "total_tests": len(test_results),
            "standards_tested": [s.value for s in standards],
            "by_status": {},
            "by_severity": {},
            "top_issues": [],
            "recommendations": []
        }
        
        # 按状态统计
        for status in ComplianceStatus:
            count = sum(1 for r in test_results if r.status == status)
            summary["by_status"][status.value] = count
        
        # 按严重程度统计（基于要求）
        for severity in TestSeverity:
            matching_requirements = [
                req for req in self.requirements.values()
                if req.severity == severity and req.standard in standards
            ]
            summary["by_severity"][severity.value] = len(matching_requirements)
        
        # 找出主要问题
        failed_tests = [r for r in test_results if r.status == ComplianceStatus.NON_COMPLIANT]
        failed_tests.sort(key=lambda x: x.score)  # 分数最低的排在前面
        
        for test in failed_tests[:5]:  # 最多5个主要问题
            requirement = self.requirements.get(test.requirement_id)
            if requirement:
                summary["top_issues"].append({
                    "requirement_id": test.requirement_id,
                    "title": requirement.title,
                    "severity": requirement.severity.value,
                    "score": test.score,
                    "remediation_steps": requirement.remediation_steps
                })
        
        # 生成建议
        if summary["by_status"].get("non_compliant", 0) > 0:
            summary["recommendations"].append("优先处理不合规项目")
        
        if summary["by_status"].get("partially_compliant", 0) > 0:
            summary["recommendations"].append("改进部分合规项目")
        
        overall_score = sum(r.score for r in test_results) / len(test_results) if test_results else 0
        if overall_score < 80:
            summary["recommendations"].append("建议全面审查安全配置")
        
        return summary
    
    def get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """获取合规要求"""
        return self.requirements.get(requirement_id)
    
    def list_requirements(
        self,
        standard: Optional[ComplianceStandard] = None,
        category: Optional[str] = None
    ) -> List[ComplianceRequirement]:
        """列出合规要求"""
        requirements = list(self.requirements.values())
        
        if standard:
            requirements = [r for r in requirements if r.standard == standard]
        
        if category:
            requirements = [r for r in requirements if r.category == category]
        
        return requirements
    
    def get_latest_report(self) -> Optional[ComplianceReport]:
        """获取最新报告"""
        return self.test_history[-1] if self.test_history else None
    
    def get_compliance_trend(self, days: int = 30) -> Dict[str, Any]:
        """获取合规趋势"""
        cutoff_date = utc_now() - timedelta(days=days)
        
        recent_reports = [
            r for r in self.test_history
            if r.generated_at > cutoff_date
        ]
        
        if not recent_reports:
            return {"trend": "insufficient_data"}
        
        # 计算趋势
        scores = [r.overall_score for r in recent_reports]
        
        if len(scores) < 2:
            return {"trend": "insufficient_data", "current_score": scores[0] if scores else 0}
        
        # 计算趋势方向
        recent_avg = sum(scores[-3:]) / len(scores[-3:])  # 最近3次的平均
        earlier_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else scores[0]
        
        if recent_avg > earlier_avg + 5:
            trend = "improving"
        elif recent_avg < earlier_avg - 5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_score": scores[-1],
            "average_score": sum(scores) / len(scores),
            "report_count": len(recent_reports),
            "period_days": days
        }

# 导入time模块
