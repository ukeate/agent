"""
MCP工具安全测试
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.core.security.mcp_security import (
    MCPToolSecurityManager,
    ToolWhitelist,
    ToolPermissionChecker,
    MCPAuditLogger
)


@pytest.fixture
def security_manager():
    """创建MCP工具安全管理器实例"""
    return MCPToolSecurityManager()


@pytest.fixture
def tool_whitelist():
    """创建工具白名单实例"""
    return ToolWhitelist()


@pytest.fixture
def permission_checker():
    """创建权限检查器实例"""
    return ToolPermissionChecker()


@pytest.fixture
def audit_logger():
    """创建审计日志器实例"""
    return MCPAuditLogger()


class TestToolWhitelist:
    """工具白名单测试"""
    
    @pytest.mark.asyncio
    async def test_is_allowed_whitelisted(self, tool_whitelist):
        """测试白名单工具检查"""
        # 默认白名单中的工具
        assert await tool_whitelist.is_allowed("file_read") is True
        assert await tool_whitelist.is_allowed("database_query") is True
    
    @pytest.mark.asyncio
    async def test_is_allowed_not_whitelisted(self, tool_whitelist):
        """测试非白名单工具检查"""
        assert await tool_whitelist.is_allowed("dangerous_tool") is False
        assert await tool_whitelist.is_allowed("unknown_tool") is False
    
    @pytest.mark.asyncio
    async def test_add_to_whitelist(self, tool_whitelist):
        """测试添加工具到白名单"""
        await tool_whitelist.add_tool("new_tool")
        assert await tool_whitelist.is_allowed("new_tool") is True
    
    @pytest.mark.asyncio
    async def test_remove_from_whitelist(self, tool_whitelist):
        """测试从白名单移除工具"""
        await tool_whitelist.remove_tool("file_write")
        assert await tool_whitelist.is_allowed("file_write") is False
    
    @pytest.mark.asyncio
    async def test_get_whitelist(self, tool_whitelist):
        """测试获取白名单"""
        whitelist = await tool_whitelist.get_whitelist()
        assert isinstance(whitelist, list)
        assert "file_read" in whitelist


class TestToolPermissionChecker:
    """工具权限检查器测试"""
    
    @pytest.mark.asyncio
    async def test_can_use_tool_admin(self, permission_checker):
        """测试管理员使用工具权限"""
        # 管理员可以使用所有工具
        assert await permission_checker.can_use_tool(
            "admin_user", ["admin"], "system_command"
        ) is True
        assert await permission_checker.can_use_tool(
            "admin_user", ["admin"], "database_write"
        ) is True
    
    @pytest.mark.asyncio
    async def test_can_use_tool_developer(self, permission_checker):
        """测试开发者使用工具权限"""
        # 开发者可以使用大部分工具
        assert await permission_checker.can_use_tool(
            "dev_user", ["developer"], "file_read"
        ) is True
        assert await permission_checker.can_use_tool(
            "dev_user", ["developer"], "database_query"
        ) is True
        
        # 但不能使用系统命令
        assert await permission_checker.can_use_tool(
            "dev_user", ["developer"], "system_command"
        ) is False
    
    @pytest.mark.asyncio
    async def test_can_use_tool_user(self, permission_checker):
        """测试普通用户使用工具权限"""
        # 用户只能使用安全工具
        assert await permission_checker.can_use_tool(
            "normal_user", ["user"], "file_read"
        ) is True
        
        # 不能使用写入工具
        assert await permission_checker.can_use_tool(
            "normal_user", ["user"], "file_write"
        ) is False
        assert await permission_checker.can_use_tool(
            "normal_user", ["user"], "database_write"
        ) is False
    
    @pytest.mark.asyncio
    async def test_get_tool_risk_level(self, permission_checker):
        """测试获取工具风险级别"""
        assert await permission_checker.get_tool_risk_level("file_read") == "low"
        assert await permission_checker.get_tool_risk_level("file_write") == "medium"
        assert await permission_checker.get_tool_risk_level("system_command") == "critical"
        assert await permission_checker.get_tool_risk_level("unknown_tool") == "unknown"


class TestMCPAuditLogger:
    """MCP审计日志器测试"""
    
    @pytest.mark.asyncio
    @patch('src.core.security.mcp_security.get_db_session')
    async def test_log_tool_call(self, mock_get_db, audit_logger):
        """测试记录工具调用"""
        mock_session = AsyncMock()
        mock_get_db.return_value = mock_session
        
        await audit_logger.log_tool_call(
            user_id="user123",
            tool_name="file_read",
            params={"path": "/tmp/test.txt"},
            result="success",
            risk_score=0.2
        )
        
        # 验证数据库操作被调用
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.core.security.mcp_security.get_redis')
    async def test_get_recent_logs(self, mock_get_redis, audit_logger):
        """测试获取最近的日志"""
        mock_redis = AsyncMock()
        mock_redis.lrange.return_value = [
            '{"user_id": "user123", "tool_name": "file_read", "timestamp": "2024-01-01T00:00:00"}'
        ]
        mock_get_redis.return_value = mock_redis
        
        logs = await audit_logger.get_recent_logs(limit=10)
        
        assert len(logs) == 1
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["tool_name"] == "file_read"
    
    @pytest.mark.asyncio
    @patch('src.core.security.mcp_security.get_db_session')
    async def test_get_user_audit_trail(self, mock_get_db, audit_logger):
        """测试获取用户审计轨迹"""
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [
            Mock(
                id="log1",
                user_id="user123",
                tool_name="file_read",
                timestamp=datetime.utcnow()
            )
        ]
        mock_session.execute.return_value = mock_result
        mock_get_db.return_value = mock_session
        
        trail = await audit_logger.get_user_audit_trail("user123")
        
        assert len(trail) > 0
        mock_session.execute.assert_called_once()


class TestMCPToolSecurityManager:
    """MCP工具安全管理器测试"""
    
    @pytest.mark.asyncio
    @patch.object(ToolWhitelist, 'is_allowed')
    @patch.object(ToolPermissionChecker, 'can_use_tool')
    @patch.object(MCPAuditLogger, 'log_tool_call')
    async def test_authorize_tool_call_success(
        self,
        mock_log,
        mock_can_use,
        mock_is_allowed,
        security_manager
    ):
        """测试成功授权工具调用"""
        mock_is_allowed.return_value = True
        mock_can_use.return_value = True
        
        result = await security_manager.authorize_tool_call(
            user_id="user123",
            user_roles=["developer"],
            tool_name="file_read",
            tool_params={"path": "/tmp/test.txt"}
        )
        
        assert result.approved is True
        assert result.reason == "Approved"
        mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(ToolWhitelist, 'is_allowed')
    async def test_authorize_tool_call_not_whitelisted(
        self,
        mock_is_allowed,
        security_manager
    ):
        """测试未在白名单的工具调用"""
        mock_is_allowed.return_value = False
        
        result = await security_manager.authorize_tool_call(
            user_id="user123",
            user_roles=["developer"],
            tool_name="dangerous_tool",
            tool_params={}
        )
        
        assert result.approved is False
        assert "not in whitelist" in result.reason
    
    @pytest.mark.asyncio
    @patch.object(ToolWhitelist, 'is_allowed')
    @patch.object(ToolPermissionChecker, 'can_use_tool')
    async def test_authorize_tool_call_insufficient_permission(
        self,
        mock_can_use,
        mock_is_allowed,
        security_manager
    ):
        """测试权限不足的工具调用"""
        mock_is_allowed.return_value = True
        mock_can_use.return_value = False
        
        result = await security_manager.authorize_tool_call(
            user_id="user123",
            user_roles=["user"],
            tool_name="system_command",
            tool_params={}
        )
        
        assert result.approved is False
        assert "Insufficient permissions" in result.reason
    
    @pytest.mark.asyncio
    @patch.object(ToolWhitelist, 'is_allowed')
    @patch.object(ToolPermissionChecker, 'can_use_tool')
    async def test_authorize_tool_call_high_risk(
        self,
        mock_can_use,
        mock_is_allowed,
        security_manager
    ):
        """测试高风险工具调用需要审批"""
        mock_is_allowed.return_value = True
        mock_can_use.return_value = True
        
        # 模拟高风险参数
        result = await security_manager.authorize_tool_call(
            user_id="user123",
            user_roles=["developer"],
            tool_name="database_write",
            tool_params={
                "query": "DROP TABLE users;",  # 危险操作
                "database": "production"
            }
        )
        
        # 高风险操作应该需要审批
        assert result.approved is False or result.requires_approval is True
    
    @pytest.mark.asyncio
    async def test_analyze_tool_params_sql_injection(self, security_manager):
        """测试SQL注入检测"""
        result = await security_manager.analyze_tool_params(
            "database_query",
            {"query": "SELECT * FROM users WHERE id = '1' OR '1'='1'"}
        )
        
        assert result.risk_score > 0.7  # 高风险
        assert any("SQL injection" in risk for risk in result.risks)
    
    @pytest.mark.asyncio
    async def test_analyze_tool_params_path_traversal(self, security_manager):
        """测试路径遍历检测"""
        result = await security_manager.analyze_tool_params(
            "file_read",
            {"path": "../../etc/passwd"}
        )
        
        assert result.risk_score > 0.5  # 中高风险
        assert any("path traversal" in risk.lower() for risk in result.risks)
    
    @pytest.mark.asyncio
    async def test_analyze_tool_params_safe(self, security_manager):
        """测试安全参数分析"""
        result = await security_manager.analyze_tool_params(
            "file_read",
            {"path": "/tmp/user_data.txt"}
        )
        
        assert result.risk_score < 0.3  # 低风险
        assert len(result.risks) == 0
    
    @pytest.mark.asyncio
    @patch('src.core.security.mcp_security.send_approval_request')
    async def test_request_approval(self, mock_send_approval, security_manager):
        """测试请求审批"""
        mock_send_approval.return_value = "approval_123"
        
        result = await security_manager.request_approval(
            user_id="user123",
            tool_name="system_command",
            params={"command": "rm -rf /tmp/*"}
        )
        
        assert result.approved is False
        assert result.requires_approval is True
        assert result.approval_id == "approval_123"
        mock_send_approval.assert_called_once()