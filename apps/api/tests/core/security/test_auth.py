"""
JWT认证和RBAC测试
"""

import pytest
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timedelta
from unittest.mock import Mock, patch, AsyncMock
import jwt

from src.core.security.auth import JWTManager, RBACManager, get_current_user, require_role
from src.core.config import get_settings


@pytest.fixture
def jwt_manager():
    """创建JWT管理器实例"""
    return JWTManager()


@pytest.fixture
def rbac_manager():
    """创建RBAC管理器实例"""
    return RBACManager()


@pytest.fixture
def mock_user():
    """模拟用户数据"""
    return {
        "id": "user123",
        "username": "testuser",
        "email": "test@example.com",
        "roles": ["developer"]
    }


class TestJWTManager:
    """JWT管理器测试"""
    
    def test_create_access_token(self, jwt_manager, mock_user):
        """测试创建访问令牌"""
        token = jwt_manager.create_access_token(mock_user)
        
        assert token is not None
        assert isinstance(token, str)
        
        # 解码令牌验证内容
        settings = get_settings()
        decoded = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        assert decoded["sub"] == mock_user["id"]
        assert decoded["username"] == mock_user["username"]
        assert decoded["roles"] == mock_user["roles"]
        assert "exp" in decoded
    
    def test_create_access_token_with_expiry(self, jwt_manager, mock_user):
        """测试创建带自定义过期时间的令牌"""
        expires_delta = timedelta(hours=1)
        token = jwt_manager.create_access_token(mock_user, expires_delta)
        
        settings = get_settings()
        decoded = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = utc_now()
        
        # 验证过期时间约为1小时后
        assert 59 <= (exp_time - now).seconds // 60 <= 61
    
    def test_verify_token_valid(self, jwt_manager, mock_user):
        """测试验证有效令牌"""
        token = jwt_manager.create_access_token(mock_user)
        payload = jwt_manager.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == mock_user["id"]
        assert payload["username"] == mock_user["username"]
    
    def test_verify_token_invalid(self, jwt_manager):
        """测试验证无效令牌"""
        invalid_token = "invalid.token.here"
        payload = jwt_manager.verify_token(invalid_token)
        
        assert payload is None
    
    def test_verify_token_expired(self, jwt_manager, mock_user):
        """测试验证过期令牌"""
        # 创建一个已过期的令牌
        expires_delta = timedelta(seconds=-1)
        token = jwt_manager.create_access_token(mock_user, expires_delta)
        
        payload = jwt_manager.verify_token(token)
        assert payload is None
    
    def test_create_refresh_token(self, jwt_manager, mock_user):
        """测试创建刷新令牌"""
        token = jwt_manager.create_refresh_token(mock_user["id"])
        
        assert token is not None
        assert isinstance(token, str)
        
        settings = get_settings()
        decoded = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        assert decoded["sub"] == mock_user["id"]
        assert decoded["type"] == "refresh"


class TestRBACManager:
    """RBAC管理器测试"""
    
    def test_check_permission_admin(self, rbac_manager):
        """测试管理员权限检查"""
        has_permission = rbac_manager.check_permission(
            ["admin"],
            "admin:users:write"
        )
        assert has_permission is True
    
    def test_check_permission_developer(self, rbac_manager):
        """测试开发者权限检查"""
        # 开发者有权限
        assert rbac_manager.check_permission(
            ["developer"],
            "agents:read"
        ) is True
        
        # 开发者无权限
        assert rbac_manager.check_permission(
            ["developer"],
            "admin:users:delete"
        ) is False
    
    def test_check_permission_user(self, rbac_manager):
        """测试普通用户权限检查"""
        # 用户有权限
        assert rbac_manager.check_permission(
            ["user"],
            "agents:read"
        ) is True
        
        # 用户无权限
        assert rbac_manager.check_permission(
            ["user"],
            "agents:write"
        ) is False
    
    def test_check_permission_multiple_roles(self, rbac_manager):
        """测试多角色权限检查"""
        roles = ["user", "developer"]
        
        # 任一角色有权限即可
        assert rbac_manager.check_permission(
            roles,
            "agents:write"
        ) is True
    
    def test_check_permission_viewer(self, rbac_manager):
        """测试查看者权限检查"""
        assert rbac_manager.check_permission(
            ["viewer"],
            "agents:read"
        ) is True
        
        assert rbac_manager.check_permission(
            ["viewer"],
            "agents:write"
        ) is False
    
    def test_get_role_permissions(self, rbac_manager):
        """测试获取角色权限"""
        admin_perms = rbac_manager.get_role_permissions("admin")
        assert "*" in admin_perms  # 管理员有所有权限
        
        dev_perms = rbac_manager.get_role_permissions("developer")
        assert "agents:read" in dev_perms
        assert "agents:write" in dev_perms
        
        unknown_perms = rbac_manager.get_role_permissions("unknown")
        assert unknown_perms == []
    
    def test_check_tool_permission(self, rbac_manager):
        """测试工具权限检查"""
        # 管理员可以使用所有工具
        assert rbac_manager.check_tool_permission(
            ["admin"],
            "database_query"
        ) is True
        
        # 开发者可以使用大部分工具
        assert rbac_manager.check_tool_permission(
            ["developer"],
            "file_read"
        ) is True
        
        # 普通用户不能使用危险工具
        assert rbac_manager.check_tool_permission(
            ["user"],
            "system_command"
        ) is False


@pytest.mark.asyncio
class TestAuthDependencies:
    """认证依赖测试"""
    
    @patch('src.core.security.auth.jwt_manager')
    async def test_get_current_user_valid(self, mock_jwt_manager, mock_user):
        """测试获取当前用户（有效令牌）"""
        mock_jwt_manager.verify_token.return_value = {
            "sub": mock_user["id"],
            "username": mock_user["username"],
            "roles": mock_user["roles"]
        }
        
        from fastapi import Request
        request = Mock(spec=Request)
        request.headers = {"authorization": "Bearer valid_token"}
        
        with patch('src.core.security.auth.get_user_by_id', new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = mock_user
            
            user = await get_current_user(authorization="Bearer valid_token")
            
            assert user == mock_user
            mock_jwt_manager.verify_token.assert_called_once_with("valid_token")
            mock_get_user.assert_called_once_with(mock_user["id"])
    
    @patch('src.core.security.auth.jwt_manager')
    async def test_get_current_user_invalid_token(self, mock_jwt_manager):
        """测试获取当前用户（无效令牌）"""
        mock_jwt_manager.verify_token.return_value = None
        
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(authorization="Bearer invalid_token")
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "无效的认证凭据"
    
    async def test_get_current_user_no_token(self):
        """测试获取当前用户（无令牌）"""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(authorization=None)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "未提供认证凭据"
    
    @patch('src.core.security.auth.rbac_manager')
    async def test_require_role_authorized(self, mock_rbac_manager, mock_user):
        """测试角色要求（已授权）"""
        mock_rbac_manager.check_permission.return_value = True
        
        role_checker = require_role("agents:write")
        result = await role_checker(current_user=mock_user)
        
        assert result == mock_user
        mock_rbac_manager.check_permission.assert_called_once_with(
            mock_user["roles"],
            "agents:write"
        )
    
    @patch('src.core.security.auth.rbac_manager')
    async def test_require_role_unauthorized(self, mock_rbac_manager, mock_user):
        """测试角色要求（未授权）"""
        mock_rbac_manager.check_permission.return_value = False
        
        from fastapi import HTTPException
        
        role_checker = require_role("admin:users:delete")
        
        with pytest.raises(HTTPException) as exc_info:
            await role_checker(current_user=mock_user)
        
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "权限不足"