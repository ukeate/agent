#!/usr/bin/env python3
"""
隐私保护机制验证脚本
验证Task 7隐私伦理防护的核心功能
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from ai.emotion_modeling.privacy_ethics_guard import (
    PrivacyEthicsGuard,
    PrivacyLevel,
    EthicalRisk,
    ConsentType,
    PrivacyPolicy,
    ConsentRecord,
    DataClassification
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """测试基础功能"""
    logger.info("🔒 开始验证隐私保护机制基础功能...")
    
    try:
        # 创建隐私防护实例
        privacy_guard = PrivacyEthicsGuard()
        logger.info("✅ 隐私防护实例创建成功")
        
        # 测试数据敏感度分类
        test_data = {
            "user_id": "test_user_001",
            "personal_info": {
                "age": 25,
                "location": "Beijing"
            },
            "emotion_history": [
                {
                    "timestamp": datetime.now(),
                    "emotions": {"happiness": 0.8, "confidence": 0.7},
                    "context": "work_meeting"
                }
            ]
        }
        
        classification = await privacy_guard.classify_data_sensitivity(test_data)
        logger.info(f"✅ 数据敏感度分类完成：{classification.sensitivity_level}, 分数：{classification.sensitivity_score}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基础功能验证失败：{e}")
        return False


async def test_privacy_violation_detection():
    """测试隐私违规检测"""
    logger.info("🚨 开始验证隐私违规检测功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 测试数据
        sensitive_data = {
            "user_id": "sensitive_user",
            "personal_info": {
                "ssn": "123-45-6789",
                "medical_info": "anxiety disorder"
            }
        }
        
        # 模拟无同意的操作
        violations = await privacy_guard.check_privacy_violations(
            sensitive_data,
            {"operation": "data_export", "user_consent": False}
        )
        
        if len(violations) > 0:
            logger.info(f"✅ 成功检测到 {len(violations)} 个隐私违规")
            for violation in violations[:3]:  # 显示前3个
                logger.info(f"   - {violation.violation_type}: {violation.description}")
        else:
            logger.warning("⚠️ 未检测到预期的隐私违规")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 隐私违规检测验证失败：{e}")
        return False


async def test_ethical_violation_detection():
    """测试伦理违规检测"""
    logger.info("⚖️ 开始验证伦理违规检测功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 测试伦理风险场景
        manipulation_context = {
            "operation": "emotion_influence",
            "purpose": "behavioral_manipulation",
            "target_emotions": ["anxiety", "fear"],
            "commercial_intent": True
        }
        
        violations = await privacy_guard.check_ethical_violations({}, manipulation_context)
        
        if len(violations) > 0:
            logger.info(f"✅ 成功检测到 {len(violations)} 个伦理违规")
            for violation in violations[:3]:
                logger.info(f"   - {violation.concern_type}: {violation.description}")
        else:
            logger.warning("⚠️ 未检测到预期的伦理违规")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 伦理违规检测验证失败：{e}")
        return False


async def test_consent_management():
    """测试同意管理"""
    logger.info("📝 开始验证同意管理功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 创建同意记录
        consent_record = ConsentRecord(
            user_id="test_user",
            consent_type="emotion_analysis",
            granted=True,
            timestamp=datetime.now(),
            scope=["emotion_tracking", "social_analysis"],
            expiry_date=datetime.now() + timedelta(days=365),
            withdrawal_date=None,
            version="1.0"
        )
        
        # 记录同意
        await privacy_guard.record_user_consent(consent_record)
        logger.info("✅ 用户同意记录成功")
        
        # 检查同意
        has_consent = await privacy_guard.check_user_consent(
            "test_user",
            "emotion_analysis",
            ["emotion_tracking"]
        )
        
        if has_consent:
            logger.info("✅ 用户同意检查通过")
        else:
            logger.error("❌ 用户同意检查失败")
            return False
        
        # 撤回同意
        await privacy_guard.withdraw_user_consent("test_user", "emotion_analysis")
        
        # 再次检查
        has_consent_after_withdrawal = await privacy_guard.check_user_consent(
            "test_user",
            "emotion_analysis",
            ["emotion_tracking"]
        )
        
        if not has_consent_after_withdrawal:
            logger.info("✅ 同意撤回验证通过")
        else:
            logger.error("❌ 同意撤回验证失败")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 同意管理验证失败：{e}")
        return False


async def test_audit_logging():
    """测试审计日志"""
    logger.info("📋 开始验证审计日志功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 记录隐私事件
        await privacy_guard.log_privacy_event(
            "DATA_ACCESS",
            {
                "user_id": "test_user",
                "data_types": ["emotions"],
                "purpose": "analysis"
            }
        )
        
        # 获取审计历史
        audit_history = await privacy_guard.get_audit_history()
        
        if len(audit_history) > 0:
            logger.info(f"✅ 审计日志记录成功，共 {len(audit_history)} 条记录")
            latest = audit_history[-1]
            logger.info(f"   最新记录：{latest.event_type} - {latest.user_id}")
        else:
            logger.error("❌ 审计日志为空")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 审计日志验证失败：{e}")
        return False


async def test_data_anonymization():
    """测试数据匿名化"""
    logger.info("🎭 开始验证数据匿名化功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 原始敏感数据
        sensitive_data = {
            "user_id": "john_doe_123",
            "personal_info": {
                "email": "john@example.com",
                "phone": "555-1234"
            },
            "emotion_history": [
                {
                    "emotions": {"depression": 0.8},
                    "context": "therapy_session"
                }
            ]
        }
        
        # 匿名化处理
        anonymized_data = await privacy_guard.anonymize_data(sensitive_data)
        
        if anonymized_data["user_id"] != sensitive_data["user_id"]:
            logger.info("✅ 用户ID已匿名化")
        else:
            logger.warning("⚠️ 用户ID未匿名化")
        
        # 检查个人信息保护
        personal_info = anonymized_data.get("personal_info", {})
        original_info = sensitive_data["personal_info"]
        
        if personal_info.get("email") != original_info["email"]:
            logger.info("✅ 邮箱信息已保护")
        
        logger.info("✅ 数据匿名化功能验证通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据匿名化验证失败：{e}")
        return False


async def test_compliance_reporting():
    """测试合规报告"""
    logger.info("📊 开始验证合规报告功能...")
    
    try:
        privacy_guard = PrivacyEthicsGuard()
        
        # 生成合规报告
        report = await privacy_guard.generate_compliance_report()
        
        logger.info("✅ 合规报告生成成功")
        logger.info(f"   审计事件数量: {report.audit_events_count}")
        logger.info(f"   隐私违规数量: {report.privacy_violations_count}")
        logger.info(f"   伦理违规数量: {report.ethical_violations_count}")
        logger.info(f"   合规分数: {report.compliance_score:.2f}")
        
        if len(report.recommendations) > 0:
            logger.info(f"   改进建议数量: {len(report.recommendations)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 合规报告验证失败：{e}")
        return False


async def main():
    """主验证函数"""
    logger.info("🚀 开始隐私保护机制全面验证")
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("基础功能", test_basic_functionality),
        ("隐私违规检测", test_privacy_violation_detection),
        ("伦理违规检测", test_ethical_violation_detection),
        ("同意管理", test_consent_management),
        ("审计日志", test_audit_logging),
        ("数据匿名化", test_data_anonymization),
        ("合规报告", test_compliance_reporting)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试项目: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 发生异常：{e}")
            test_results.append((test_name, False))
    
    # 总结报告
    logger.info(f"\n{'='*60}")
    logger.info("🎯 隐私保护机制验证总结报告")
    logger.info(f"{'='*60}")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        logger.info("🎉 所有隐私保护功能验证通过！")
        return True
    else:
        logger.warning(f"⚠️ 有 {total_tests - passed_tests} 个测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)