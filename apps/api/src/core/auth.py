"""
认证模块

提供基本的用户认证功能
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    获取当前用户信息
    
    在开发阶段返回默认用户
    """
    # 开发阶段：返回默认用户，跳过实际认证
    return {
        "id": "00000000-0000-4000-8000-000000000001",
        "username": "developer", 
        "email": "dev@example.com",
        "role": "admin"
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    获取可选的用户信息（允许匿名访问）
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    要求管理员权限
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证访问令牌
    
    在开发阶段直接返回默认用户
    """
    # 开发阶段：接受任何token
    if token:
        return {
            "id": "00000000-0000-4000-8000-000000000001",
            "username": "developer",
            "email": "dev@example.com", 
            "role": "admin"
        }
    return None


class AuthenticationError(Exception):
    """认证错误"""
    pass


class AuthorizationError(Exception):
    """授权错误"""
    pass