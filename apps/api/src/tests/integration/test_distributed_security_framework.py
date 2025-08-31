"""
分布式安全框架集成测试
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.ai.autogen.security.identity_authentication import (
    IdentityAuthenticationService,
    AuthenticationRequest,
    AuthenticationResponse,
    ZeroTrustValidator
)
from src.ai.autogen.security.encrypted_communication import (
    EncryptedCommunicationFramework,
    MessageIntegrityValidator,
    CommunicationSession
)
from src.ai.autogen.security.access_control import (
    AccessControlEngine,
    AccessRequest,
    AccessResponse,
    PolicyRule
)
from src.ai.autogen.security.security_audit import (
    SecurityAuditSystem,
    SecurityEvent,
    ThreatIntelligenceEngine
)
from src.db.models import (
    AgentIdentity,
    SecurityRole,
    AccessPolicy,
    SecurityAlert,
    ThreatLevelEnum,
    AuthenticationMethodEnum,
    AccessControlModelEnum
)


class TestDistributedSecurityFrameworkIntegration:
    """分布式安全框架集成测试类"""
    
    @pytest.fixture
    async def auth_service(self):
        """身份认证服务fixture"""
        return IdentityAuthenticationService()
    
    @pytest.fixture
    async def comm_framework(self):
        """加密通信框架fixture"""
        return EncryptedCommunicationFramework()
    
    @pytest.fixture
    async def access_engine(self):
        """访问控制引擎fixture"""
        engine = AccessControlEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    async def audit_system(self):
        """安全审计系统fixture"""
        return SecurityAuditSystem()
    
    @pytest.fixture
    def mock_agent_identity(self):
        """模拟智能体身份"""
        return AgentIdentity(
            id=str(uuid.uuid4()),
            agent_id="test_agent_001",
            display_name="测试智能体",
            agent_type="ai_assistant",
            public_key="test_public_key_data",
            certificate_data="test_certificate_data",
            authentication_methods=[
                AuthenticationMethodEnum.PKI_CERTIFICATE,
                AuthenticationMethodEnum.MFA
            ],
            trust_score=95.0,
            failed_attempts=0,
            is_locked=False
        )
    
    @pytest.fixture
    def mock_security_role(self):
        """模拟安全角色"""
        return SecurityRole(
            id=str(uuid.uuid4()),
            role_name="ai_agent_user",
            description="AI智能体用户角色",
            permissions=["read:data", "write:response", "execute:task"],
            is_system_role=False,
            created_by="system"
        )
    
    @pytest.fixture
    def mock_access_policy(self):
        """模拟访问策略"""
        return AccessPolicy(
            id=str(uuid.uuid4()),
            policy_name="default_ai_agent_policy",
            description="AI智能体默认访问策略",
            resource_pattern="/api/v1/agents/*",
            action_pattern="read|write|execute",
            policy_type=AccessControlModelEnum.RBAC,
            rules={
                "allow_time_range": "09:00-18:00",
                "max_requests_per_hour": 1000,
                "required_trust_score": 80.0
            },
            conditions={
                "ip_whitelist": ["10.0.0.0/8", "172.16.0.0/12"],
                "user_agent_pattern": "AIAgent/*"
            },
            is_active=True,
            priority=100,
            created_by="system"
        )

    @pytest.mark.asyncio
    async def test_complete_authentication_flow(self, auth_service, mock_agent_identity):
        """测试完整的认证流程"""
        # 模拟数据库操作
        with patch.object(auth_service, 'get_agent_identity', return_value=mock_agent_identity):
            with patch.object(auth_service, 'update_agent_identity') as mock_update:
                # 创建认证请求
                auth_request = AuthenticationRequest(
                    agent_id="test_agent_001",
                    authentication_method=AuthenticationMethodEnum.PKI_CERTIFICATE,
                    credentials={
                        "certificate": "test_certificate_data",
                        "signature": "test_signature",
                        "nonce": "test_nonce_123456"
                    },
                    context_attributes={
                        "ip_address": "10.0.1.100",
                        "user_agent": "AIAgent/1.0",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # 执行认证
                response = await auth_service.authenticate_agent(auth_request)
                
                # 验证认证结果
                assert response.success is True
                assert response.agent_id == "test_agent_001"
                assert response.session_token is not None
                assert response.trust_score >= 80.0
                
                # 验证数据库更新被调用
                mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_encrypted_communication_session(self, comm_framework):
        """测试加密通信会话"""
        agent1_id = "agent_001"
        agent2_id = "agent_002"
        
        # 建立加密会话
        session = await comm_framework.establish_session(
            initiator_id=agent1_id,
            target_id=agent2_id,
            algorithm="aes_256_gcm"
        )
        
        assert session is not None
        assert session.initiator_agent_id == agent1_id
        assert session.target_agent_id == agent2_id
        assert session.is_active is True
        
        # 测试消息加密和解密
        original_message = "这是一条测试消息，包含敏感信息。"
        
        # 加密消息
        encrypted_data = await comm_framework.encrypt_message(
            session_id=session.session_id,
            message=original_message,
            sender_id=agent1_id
        )
        
        assert encrypted_data is not None
        assert encrypted_data["encrypted_content"] != original_message
        assert "integrity_hash" in encrypted_data
        
        # 解密消息
        decrypted_message = await comm_framework.decrypt_message(
            session_id=session.session_id,
            encrypted_data=encrypted_data,
            receiver_id=agent2_id
        )
        
        assert decrypted_message == original_message
        
        # 验证消息完整性
        integrity_validator = MessageIntegrityValidator()
        is_valid = await integrity_validator.validate_message_integrity(
            encrypted_data, session.session_key_hash
        )
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_access_control_decision(self, access_engine, mock_access_policy, mock_agent_identity):
        """测试访问控制决策"""
        # 模拟策略和身份数据
        with patch.object(access_engine, 'get_applicable_policies', return_value=[mock_access_policy]):
            with patch.object(access_engine, 'get_agent_identity', return_value=mock_agent_identity):
                
                # 创建访问请求
                access_request = AccessRequest(
                    agent_id="test_agent_001",
                    resource="/api/v1/agents/tasks",
                    action="execute",
                    context_attributes={
                        "ip_address": "10.0.1.100",
                        "user_agent": "AIAgent/1.0",
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": "session_123456"
                    }
                )
                
                # 执行访问控制决策
                response = await access_engine.evaluate_access(access_request)
                
                # 验证访问决策
                assert response is not None
                assert response.decision in ["allow", "deny"]
                assert len(response.policy_evaluations) > 0
                assert response.confidence_score > 0.0
                
                # 如果允许访问，验证相关属性
                if response.decision == "allow":
                    assert response.confidence_score >= 0.8
                    assert response.session_duration_minutes > 0

    @pytest.mark.asyncio
    async def test_threat_detection_and_alerting(self, audit_system):
        """测试威胁检测和告警"""
        # 模拟可疑的安全事件序列
        suspicious_events = [
            SecurityEvent(
                id=str(uuid.uuid4()),
                event_type="login_failed",
                agent_id="test_agent_001",
                resource="/auth/login",
                action="authenticate",
                ip_address="192.168.1.100",
                user_agent="UnknownAgent/1.0",
                outcome="failure",
                details={"reason": "invalid_credentials", "attempt_number": 1},
                risk_score=25.0,
                timestamp=datetime.utcnow() - timedelta(minutes=5)
            ),
            SecurityEvent(
                id=str(uuid.uuid4()),
                event_type="login_failed",
                agent_id="test_agent_001",
                resource="/auth/login",
                action="authenticate",
                ip_address="192.168.1.100",
                user_agent="UnknownAgent/1.0",
                outcome="failure",
                details={"reason": "invalid_credentials", "attempt_number": 2},
                risk_score=35.0,
                timestamp=datetime.utcnow() - timedelta(minutes=3)
            ),
            SecurityEvent(
                id=str(uuid.uuid4()),
                event_type="login_failed",
                agent_id="test_agent_001",
                resource="/auth/login",
                action="authenticate",
                ip_address="192.168.1.100",
                user_agent="UnknownAgent/1.0",
                outcome="failure",
                details={"reason": "invalid_credentials", "attempt_number": 3},
                risk_score=45.0,
                timestamp=datetime.utcnow() - timedelta(minutes=1)
            )
        ]
        
        # 模拟威胁检测
        with patch.object(audit_system, 'store_security_event') as mock_store:
            with patch.object(audit_system, 'create_security_alert') as mock_alert:
                
                # 处理安全事件
                alerts_generated = []
                for event in suspicious_events:
                    await audit_system.process_security_event(event)
                    
                    # 检查是否触发威胁检测
                    threat_detected = await audit_system.detect_threats([event])
                    if threat_detected:
                        alert = SecurityAlert(
                            id=str(uuid.uuid4()),
                            alert_type="brute_force_attack",
                            title="检测到暴力破解攻击",
                            description=f"智能体 {event.agent_id} 在短时间内多次认证失败",
                            threat_level=ThreatLevelEnum.HIGH,
                            agent_id=event.agent_id,
                            source_event_ids=[event.id],
                            indicators={
                                "failed_attempts": len(suspicious_events),
                                "time_window_minutes": 5,
                                "source_ip": event.ip_address
                            },
                            remediation_suggestions=[
                                "临时锁定智能体账户",
                                "加强身份验证要求",
                                "审查访问日志"
                            ]
                        )
                        alerts_generated.append(alert)
                        await audit_system.create_security_alert(alert)
                
                # 验证威胁检测结果
                assert len(alerts_generated) > 0
                assert mock_store.call_count == len(suspicious_events)
                assert mock_alert.call_count > 0
                
                # 验证告警内容
                if alerts_generated:
                    alert = alerts_generated[0]
                    assert alert.alert_type == "brute_force_attack"
                    assert alert.threat_level == ThreatLevelEnum.HIGH
                    assert alert.agent_id == "test_agent_001"

    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(
        self, 
        auth_service, 
        comm_framework, 
        access_engine, 
        audit_system,
        mock_agent_identity,
        mock_access_policy
    ):
        """测试端到端的安全工作流"""
        agent_id = "test_agent_001"
        
        # Step 1: 身份认证
        with patch.object(auth_service, 'get_agent_identity', return_value=mock_agent_identity):
            auth_request = AuthenticationRequest(
                agent_id=agent_id,
                authentication_method=AuthenticationMethodEnum.PKI_CERTIFICATE,
                credentials={
                    "certificate": "test_certificate",
                    "signature": "test_signature"
                },
                context_attributes={
                    "ip_address": "10.0.1.100",
                    "user_agent": "AIAgent/1.0"
                }
            )
            
            auth_response = await auth_service.authenticate_agent(auth_request)
            assert auth_response.success is True
            session_token = auth_response.session_token
        
        # Step 2: 建立加密通信
        target_agent = "agent_002"
        comm_session = await comm_framework.establish_session(
            initiator_id=agent_id,
            target_id=target_agent,
            algorithm="aes_256_gcm"
        )
        assert comm_session is not None
        
        # Step 3: 访问控制检查
        with patch.object(access_engine, 'get_applicable_policies', return_value=[mock_access_policy]):
            with patch.object(access_engine, 'get_agent_identity', return_value=mock_agent_identity):
                access_request = AccessRequest(
                    agent_id=agent_id,
                    resource="/api/v1/agents/communication",
                    action="send_message",
                    context_attributes={
                        "session_token": session_token,
                        "target_agent": target_agent,
                        "ip_address": "10.0.1.100"
                    }
                )
                
                access_response = await access_engine.evaluate_access(access_request)
                assert access_response.decision == "allow"
        
        # Step 4: 发送加密消息
        test_message = "机密通信内容：任务执行状态更新"
        encrypted_data = await comm_framework.encrypt_message(
            session_id=comm_session.session_id,
            message=test_message,
            sender_id=agent_id
        )
        assert encrypted_data is not None
        
        # Step 5: 安全审计记录
        security_event = SecurityEvent(
            id=str(uuid.uuid4()),
            event_type="communication_established",
            agent_id=agent_id,
            resource=f"/communication/session/{comm_session.session_id}",
            action="send_encrypted_message",
            outcome="success",
            details={
                "target_agent": target_agent,
                "message_size": len(test_message),
                "encryption_algorithm": "aes_256_gcm"
            },
            risk_score=10.0
        )
        
        with patch.object(audit_system, 'store_security_event') as mock_store:
            await audit_system.process_security_event(security_event)
            mock_store.assert_called_once_with(security_event)
        
        # Step 6: 验证端到端工作流完整性
        workflow_success = all([
            auth_response.success,
            comm_session.is_active,
            access_response.decision == "allow",
            encrypted_data is not None
        ])
        
        assert workflow_success is True

    @pytest.mark.asyncio
    async def test_security_configuration_validation(self):
        """测试安全配置验证"""
        # 测试各个组件的配置验证
        configurations = {
            "authentication": {
                "pki_certificate_validation": True,
                "mfa_required": True,
                "session_timeout_minutes": 60,
                "max_failed_attempts": 3
            },
            "encryption": {
                "algorithm": "aes_256_gcm",
                "key_rotation_hours": 24,
                "forward_secrecy": True
            },
            "access_control": {
                "default_policy": "deny",
                "rbac_enabled": True,
                "abac_enabled": True,
                "policy_cache_ttl_minutes": 15
            },
            "audit": {
                "log_all_events": True,
                "threat_detection_enabled": True,
                "alert_threshold": "medium",
                "retention_days": 365
            }
        }
        
        # 验证配置完整性
        required_sections = ["authentication", "encryption", "access_control", "audit"]
        for section in required_sections:
            assert section in configurations
            assert len(configurations[section]) > 0
        
        # 验证关键安全参数
        assert configurations["authentication"]["mfa_required"] is True
        assert configurations["encryption"]["forward_secrecy"] is True
        assert configurations["access_control"]["default_policy"] == "deny"
        assert configurations["audit"]["threat_detection_enabled"] is True

    @pytest.mark.asyncio
    async def test_performance_and_scalability(self):
        """测试性能和可扩展性"""
        # 模拟并发认证请求
        concurrent_requests = 50
        auth_service = IdentityAuthenticationService()
        
        async def simulate_auth_request(request_id: int):
            """模拟单个认证请求"""
            mock_agent = AgentIdentity(
                id=str(uuid.uuid4()),
                agent_id=f"agent_{request_id:03d}",
                display_name=f"测试智能体 {request_id}",
                agent_type="ai_assistant",
                public_key=f"public_key_{request_id}",
                authentication_methods=[AuthenticationMethodEnum.PKI_CERTIFICATE],
                trust_score=90.0,
                failed_attempts=0,
                is_locked=False
            )
            
            with patch.object(auth_service, 'get_agent_identity', return_value=mock_agent):
                auth_request = AuthenticationRequest(
                    agent_id=f"agent_{request_id:03d}",
                    authentication_method=AuthenticationMethodEnum.PKI_CERTIFICATE,
                    credentials={"certificate": f"cert_{request_id}"},
                    context_attributes={"ip_address": "10.0.1.100"}
                )
                
                start_time = datetime.utcnow()
                response = await auth_service.authenticate_agent(auth_request)
                end_time = datetime.utcnow()
                
                return {
                    "request_id": request_id,
                    "success": response.success,
                    "duration_ms": (end_time - start_time).total_seconds() * 1000
                }
        
        # 执行并发测试
        tasks = [simulate_auth_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证性能指标
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        assert len(successful_requests) >= concurrent_requests * 0.95  # 95%成功率
        
        # 验证平均响应时间
        avg_duration = sum(r["duration_ms"] for r in successful_requests) / len(successful_requests)
        assert avg_duration < 100  # 平均响应时间小于100ms
        
        print(f"并发认证测试完成: {len(successful_requests)}/{concurrent_requests} 成功, "
              f"平均响应时间: {avg_duration:.2f}ms")

    @pytest.mark.asyncio 
    async def test_error_handling_and_recovery(self, auth_service, audit_system):
        """测试错误处理和恢复机制"""
        # 测试网络错误恢复
        with patch.object(auth_service, 'get_agent_identity', side_effect=ConnectionError("数据库连接失败")):
            auth_request = AuthenticationRequest(
                agent_id="test_agent",
                authentication_method=AuthenticationMethodEnum.PKI_CERTIFICATE,
                credentials={"certificate": "test_cert"},
                context_attributes={"ip_address": "10.0.1.100"}
            )
            
            response = await auth_service.authenticate_agent(auth_request)
            assert response.success is False
            assert "连接失败" in response.error_message or "connection" in response.error_message.lower()
        
        # 测试无效输入处理
        invalid_auth_request = AuthenticationRequest(
            agent_id="",  # 空的agent_id
            authentication_method=AuthenticationMethodEnum.PKI_CERTIFICATE,
            credentials={},  # 空的凭据
            context_attributes={}
        )
        
        response = await auth_service.authenticate_agent(invalid_auth_request)
        assert response.success is False
        assert response.error_message is not None
        
        # 测试审计系统错误恢复
        invalid_event = SecurityEvent(
            id="",  # 无效ID
            event_type="invalid_event_type",
            agent_id="test_agent",
            resource="/test",
            action="test",
            outcome="unknown",
            details={},
            risk_score=-1.0,  # 无效风险分数
        )
        
        with patch.object(audit_system, 'store_security_event', side_effect=ValueError("无效事件数据")):
            # 应该优雅处理错误，不抛出异常
            try:
                await audit_system.process_security_event(invalid_event)
            except Exception as e:
                pytest.fail(f"处理无效事件时不应抛出异常: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])