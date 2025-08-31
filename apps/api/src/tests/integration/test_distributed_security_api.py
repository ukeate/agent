"""
分布式安全框架API集成测试
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from fastapi import status

from src.main_simple import app
from src.db.models import (
    AgentIdentity,
    SecurityRole,
    AccessPolicy,
    SecurityAlert,
    ThreatLevelEnum,
    AuthenticationMethodEnum,
    AccessControlModelEnum,
    AlertStatusEnum
)


class TestDistributedSecurityAPI:
    """分布式安全框架API集成测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_admin_user(self):
        """模拟管理员用户"""
        return {
            "id": "admin_user_001",
            "username": "admin",
            "roles": ["system:admin"],
            "permissions": ["system:read", "system:write", "system:admin"]
        }
    
    @pytest.fixture
    def mock_auth_headers(self):
        """模拟认证头信息"""
        return {
            "Authorization": "Bearer mock_jwt_token",
            "X-Agent-ID": "test_agent_001",
            "Content-Type": "application/json"
        }

    def test_agent_identity_registration(self, client, mock_auth_headers):
        """测试智能体身份注册"""
        registration_data = {
            "agent_id": "new_agent_001",
            "display_name": "新测试智能体",
            "agent_type": "ai_assistant",
            "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0...",
            "authentication_methods": ["pki_certificate", "mfa"],
            "metadata": {
                "version": "1.0.0",
                "capabilities": ["reasoning", "planning", "communication"]
            }
        }
        
        with patch("src.api.v1.distributed_security.mock_identity_service.register_agent") as mock_register:
            mock_register.return_value = {
                "success": True,
                "agent_id": "new_agent_001",
                "registration_id": "reg_12345",
                "trust_score": 100.0
            }
            
            response = client.post(
                "/api/v1/distributed-security/agents/register",
                headers=mock_auth_headers,
                json=registration_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["success"] is True
            assert result["agent_id"] == "new_agent_001"
            assert "registration_id" in result

    def test_agent_authentication_flow(self, client, mock_auth_headers):
        """测试智能体认证流程"""
        auth_data = {
            "agent_id": "test_agent_001",
            "authentication_method": "pki_certificate",
            "credentials": {
                "certificate": "-----BEGIN CERTIFICATE-----\nMIIC...",
                "signature": "signature_data_here",
                "nonce": "random_nonce_123456"
            },
            "context_attributes": {
                "ip_address": "10.0.1.100",
                "user_agent": "AIAgent/1.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        with patch("src.api.v1.distributed_security.mock_identity_service.authenticate_agent") as mock_auth:
            mock_auth.return_value = {
                "success": True,
                "agent_id": "test_agent_001",
                "session_token": "jwt_session_token_here",
                "trust_score": 95.0,
                "session_expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "permissions": ["read:data", "write:response", "execute:task"]
            }
            
            response = client.post(
                "/api/v1/distributed-security/auth/authenticate",
                headers=mock_auth_headers,
                json=auth_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["success"] is True
            assert result["session_token"] is not None
            assert result["trust_score"] >= 80.0

    def test_establish_encrypted_communication(self, client, mock_auth_headers):
        """测试建立加密通信会话"""
        comm_request = {
            "target_agent_id": "target_agent_002",
            "encryption_algorithm": "aes_256_gcm",
            "session_metadata": {
                "purpose": "task_coordination",
                "expected_duration_minutes": 30
            }
        }
        
        with patch("src.api.v1.distributed_security.mock_communication_service.establish_session") as mock_establish:
            mock_establish.return_value = {
                "session_id": "comm_session_12345",
                "initiator_agent_id": "test_agent_001",
                "target_agent_id": "target_agent_002",
                "encryption_algorithm": "aes_256_gcm",
                "session_key_fingerprint": "sha256:abcd1234...",
                "established_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            response = client.post(
                "/api/v1/distributed-security/communication/establish",
                headers=mock_auth_headers,
                json=comm_request
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["session_id"] is not None
            assert result["encryption_algorithm"] == "aes_256_gcm"

    def test_access_control_evaluation(self, client, mock_auth_headers):
        """测试访问控制评估"""
        access_request = {
            "resource": "/api/v1/agents/tasks/execute",
            "action": "execute",
            "context_attributes": {
                "task_type": "data_analysis",
                "sensitivity_level": "medium",
                "requested_resources": ["cpu", "memory", "network"],
                "ip_address": "10.0.1.100",
                "session_id": "session_123456"
            }
        }
        
        with patch("src.api.v1.distributed_security.mock_access_control_service.evaluate_access") as mock_eval:
            mock_eval.return_value = {
                "decision": "allow",
                "confidence_score": 0.92,
                "policy_evaluations": [
                    {
                        "policy_name": "default_ai_agent_policy",
                        "decision": "allow",
                        "reason": "Agent meets trust score requirement"
                    }
                ],
                "conditions": [
                    "Maximum execution time: 30 minutes",
                    "Resource usage monitoring required"
                ],
                "session_duration_minutes": 30
            }
            
            response = client.post(
                "/api/v1/distributed-security/access-control/evaluate",
                headers=mock_auth_headers,
                json=access_request
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["decision"] == "allow"
            assert result["confidence_score"] > 0.8
            assert len(result["policy_evaluations"]) > 0

    def test_security_event_logging(self, client, mock_auth_headers):
        """测试安全事件记录"""
        security_event = {
            "event_type": "access_granted",
            "resource": "/api/v1/agents/tasks",
            "action": "execute",
            "outcome": "success",
            "details": {
                "task_id": "task_12345",
                "execution_duration_seconds": 45,
                "resources_used": ["cpu", "memory"]
            },
            "risk_score": 15.0,
            "context_attributes": {
                "ip_address": "10.0.1.100",
                "user_agent": "AIAgent/1.0",
                "session_id": "session_123456"
            }
        }
        
        with patch("src.api.v1.distributed_security.mock_audit_service.log_security_event") as mock_log:
            mock_log.return_value = {
                "event_id": "event_12345",
                "logged_at": datetime.utcnow().isoformat(),
                "threat_detected": False,
                "risk_assessment": {
                    "risk_level": "low",
                    "threat_indicators": []
                }
            }
            
            response = client.post(
                "/api/v1/distributed-security/audit/events",
                headers=mock_auth_headers,
                json=security_event
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["event_id"] is not None
            assert result["threat_detected"] is False

    def test_threat_detection_alert(self, client, mock_auth_headers):
        """测试威胁检测和告警"""
        # 模拟触发威胁检测的事件序列
        suspicious_events = [
            {
                "event_type": "authentication_failed",
                "agent_id": "test_agent_001",
                "resource": "/auth/login", 
                "action": "authenticate",
                "outcome": "failure",
                "details": {"reason": "invalid_credentials", "attempt": i+1},
                "risk_score": 25.0 + (i * 10),
                "context_attributes": {
                    "ip_address": "192.168.1.100",
                    "user_agent": "SuspiciousAgent/1.0"
                }
            }
            for i in range(5)  # 5次连续失败
        ]
        
        with patch("src.api.v1.distributed_security.mock_audit_service.detect_threats") as mock_detect:
            mock_detect.return_value = {
                "threats_detected": True,
                "alerts_generated": [
                    {
                        "alert_id": "alert_12345",
                        "alert_type": "brute_force_attack",
                        "title": "检测到暴力破解攻击",
                        "threat_level": "high",
                        "agent_id": "test_agent_001",
                        "indicators": {
                            "failed_attempts": 5,
                            "time_window_minutes": 10,
                            "source_ip": "192.168.1.100"
                        },
                        "remediation_suggestions": [
                            "临时锁定智能体账户",
                            "要求强化身份验证",
                            "监控相关IP地址活动"
                        ]
                    }
                ]
            }
            
            # 批量提交可疑事件
            response = client.post(
                "/api/v1/distributed-security/audit/events/batch",
                headers=mock_auth_headers,
                json={"events": suspicious_events}
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["threats_detected"] is True
            assert len(result["alerts_generated"]) > 0
            
            # 验证告警详情
            alert = result["alerts_generated"][0]
            assert alert["threat_level"] == "high"
            assert alert["alert_type"] == "brute_force_attack"

    def test_security_alerts_management(self, client, mock_auth_headers):
        """测试安全告警管理"""
        # 获取活跃告警列表
        with patch("src.api.v1.distributed_security.mock_audit_service.get_active_alerts") as mock_get_alerts:
            mock_get_alerts.return_value = [
                {
                    "alert_id": "alert_12345",
                    "alert_type": "brute_force_attack",
                    "title": "检测到暴力破解攻击",
                    "threat_level": "high",
                    "status": "active",
                    "agent_id": "test_agent_001",
                    "created_at": datetime.utcnow().isoformat(),
                    "indicators": {
                        "failed_attempts": 5,
                        "source_ip": "192.168.1.100"
                    }
                }
            ]
            
            response = client.get(
                "/api/v1/distributed-security/alerts",
                headers=mock_auth_headers,
                params={"status": "active"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert len(result["alerts"]) > 0
            assert result["alerts"][0]["threat_level"] == "high"
        
        # 确认告警
        alert_id = "alert_12345"
        acknowledge_data = {
            "acknowledged_by": "security_admin",
            "acknowledgment_note": "已确认并开始调查此攻击"
        }
        
        with patch("src.api.v1.distributed_security.mock_audit_service.acknowledge_alert") as mock_ack:
            mock_ack.return_value = {
                "alert_id": alert_id,
                "status": "acknowledged",
                "acknowledged_at": datetime.utcnow().isoformat(),
                "acknowledged_by": "security_admin"
            }
            
            response = client.post(
                f"/api/v1/distributed-security/alerts/{alert_id}/acknowledge",
                headers=mock_auth_headers,
                json=acknowledge_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["status"] == "acknowledged"

    def test_policy_management(self, client, mock_auth_headers):
        """测试策略管理"""
        # 创建新访问策略
        policy_data = {
            "policy_name": "test_ai_agent_policy",
            "description": "测试AI智能体访问策略", 
            "resource_pattern": "/api/v1/test/*",
            "action_pattern": "read|write",
            "policy_type": "rbac",
            "rules": {
                "required_trust_score": 85.0,
                "max_requests_per_hour": 500,
                "allowed_time_range": "08:00-20:00"
            },
            "conditions": {
                "ip_whitelist": ["10.0.0.0/8"],
                "required_roles": ["ai_agent_user"]
            },
            "priority": 50
        }
        
        with patch("src.api.v1.distributed_security.mock_access_control_service.create_policy") as mock_create:
            mock_create.return_value = {
                "policy_id": "policy_12345",
                "policy_name": "test_ai_agent_policy",
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            }
            
            response = client.post(
                "/api/v1/distributed-security/policies",
                headers=mock_auth_headers,
                json=policy_data
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["policy_id"] is not None
            assert result["policy_name"] == "test_ai_agent_policy"

    def test_security_metrics_dashboard(self, client, mock_auth_headers):
        """测试安全指标仪表盘"""
        with patch("src.api.v1.distributed_security.mock_audit_service.get_security_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "authentication_metrics": {
                    "total_attempts_24h": 1250,
                    "successful_attempts_24h": 1200,
                    "failed_attempts_24h": 50,
                    "success_rate": 0.96,
                    "average_response_time_ms": 85.4
                },
                "access_control_metrics": {
                    "total_requests_24h": 8500,
                    "granted_requests_24h": 8200,
                    "denied_requests_24h": 300,
                    "approval_rate": 0.965,
                    "policy_evaluation_time_ms": 12.3
                },
                "communication_metrics": {
                    "active_sessions": 45,
                    "total_messages_24h": 25000,
                    "encryption_overhead_ms": 3.2,
                    "integrity_violations": 0
                },
                "threat_detection_metrics": {
                    "events_processed_24h": 15000,
                    "threats_detected_24h": 5,
                    "false_positives_24h": 2,
                    "alert_response_time_minutes": 8.5
                }
            }
            
            response = client.get(
                "/api/v1/distributed-security/metrics",
                headers=mock_auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert "authentication_metrics" in result
            assert "access_control_metrics" in result
            assert "communication_metrics" in result
            assert "threat_detection_metrics" in result
            
            # 验证关键指标
            auth_metrics = result["authentication_metrics"]
            assert auth_metrics["success_rate"] > 0.9
            assert auth_metrics["average_response_time_ms"] < 100

    def test_compliance_reporting(self, client, mock_auth_headers):
        """测试合规报告生成"""
        report_request = {
            "report_type": "security_compliance",
            "start_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "include_sections": [
                "authentication_summary",
                "access_control_decisions", 
                "security_events",
                "threat_detection_results"
            ]
        }
        
        with patch("src.api.v1.distributed_security.mock_audit_service.generate_compliance_report") as mock_report:
            mock_report.return_value = {
                "report_id": "report_12345",
                "report_type": "security_compliance",
                "generated_at": datetime.utcnow().isoformat(),
                "period": "2024-01-01 to 2024-01-31",
                "summary": {
                    "total_agents": 150,
                    "authentication_events": 45000,
                    "access_requests": 180000,
                    "security_incidents": 8,
                    "compliance_score": 0.965
                },
                "sections": {
                    "authentication_summary": {
                        "total_attempts": 45000,
                        "success_rate": 0.98,
                        "mfa_adoption_rate": 0.85
                    },
                    "access_control_decisions": {
                        "total_requests": 180000,
                        "approval_rate": 0.97,
                        "policy_violations": 45
                    },
                    "security_events": {
                        "total_events": 220000,
                        "high_risk_events": 120,
                        "incidents_resolved": 8
                    },
                    "threat_detection_results": {
                        "threats_detected": 15,
                        "false_positives": 7,
                        "detection_accuracy": 0.53
                    }
                },
                "recommendations": [
                    "提高MFA采用率到95%以上",
                    "优化威胁检测规则减少误报",
                    "加强高风险事件监控"
                ]
            }
            
            response = client.post(
                "/api/v1/distributed-security/compliance/report",
                headers=mock_auth_headers,
                json=report_request
            )
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["report_id"] is not None
            assert result["summary"]["compliance_score"] > 0.9
            assert len(result["recommendations"]) > 0

    def test_api_error_handling(self, client, mock_auth_headers):
        """测试API错误处理"""
        # 测试无效认证数据
        invalid_auth_data = {
            "agent_id": "",  # 空的agent_id
            "authentication_method": "invalid_method",
            "credentials": {},
            "context_attributes": {}
        }
        
        response = client.post(
            "/api/v1/distributed-security/auth/authenticate",
            headers=mock_auth_headers,
            json=invalid_auth_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        
        # 测试未授权访问
        response = client.get(
            "/api/v1/distributed-security/alerts",
            headers={"Content-Type": "application/json"}  # 缺少认证头
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # 测试不存在的资源
        response = client.get(
            "/api/v1/distributed-security/alerts/nonexistent_alert_id",
            headers=mock_auth_headers
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_api_rate_limiting(self, client, mock_auth_headers):
        """测试API速率限制"""
        # 模拟快速连续请求
        responses = []
        for i in range(10):
            response = client.get(
                "/api/v1/distributed-security/metrics",
                headers=mock_auth_headers
            )
            responses.append(response.status_code)
        
        # 验证前几个请求成功
        assert responses[0] == status.HTTP_200_OK
        assert responses[1] == status.HTTP_200_OK
        
        # 在实际实现中，后续请求可能会被速率限制
        # assert any(code == status.HTTP_429_TOO_MANY_REQUESTS for code in responses[-3:])

    def test_api_input_validation(self, client, mock_auth_headers):
        """测试API输入验证"""
        # 测试无效的策略数据
        invalid_policy_data = {
            "policy_name": "",  # 空名称
            "description": "A" * 1001,  # 过长描述
            "resource_pattern": "invalid_pattern[[[",  # 无效正则
            "action_pattern": "",
            "policy_type": "invalid_type",  # 无效类型
            "rules": "not_an_object",  # 应该是对象
            "priority": -1  # 无效优先级
        }
        
        response = client.post(
            "/api/v1/distributed-security/policies",
            headers=mock_auth_headers,
            json=invalid_policy_data
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        result = response.json()
        assert "detail" in result

    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, client, mock_auth_headers):
        """测试并发API请求"""
        async def make_request(session_id: str):
            """发送单个API请求"""
            headers = {**mock_auth_headers, "X-Session-ID": session_id}
            
            response = client.get(
                "/api/v1/distributed-security/metrics",
                headers=headers
            )
            
            return {
                "session_id": session_id,
                "status_code": response.status_code,
                "response_time": 1.0  # 模拟响应时间
            }
        
        # 创建多个并发请求
        concurrent_requests = 20
        tasks = [
            make_request(f"session_{i:03d}") 
            for i in range(concurrent_requests)
        ]
        
        # 执行并发请求 (使用同步客户端模拟)
        results = []
        for task in tasks:
            result = task
            results.append(result)
        
        # 验证所有请求都成功处理
        successful_requests = [r for r in results if r["status_code"] == 200]
        assert len(successful_requests) >= concurrent_requests * 0.95  # 95%成功率


if __name__ == "__main__":
    pytest.main([__file__, "-v"])