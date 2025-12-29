"""
认证模块

提供基本的用户认证功能
"""

from typing import Dict, Any, Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.security.auth import User, get_current_active_user, jwt_manager, rbac_manager
from src.models.database.user import AuthUser

security = HTTPBearer(auto_error=False)

async def get_current_user(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    获取当前用户信息
    """
    role = current_user.roles[0] if current_user.roles else "user"
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "role": role,
    }

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> Optional[Dict[str, Any]]:
    """
    获取可选的用户信息（允许匿名访问）
    """
    if not credentials or not credentials.credentials:
        return None

    try:
        token_data = jwt_manager.decode_token(credentials.credentials)
        if token_data.token_type != "access" or not token_data.user_id:
            return None
        user_uuid = uuid.UUID(token_data.user_id)
        db_user = (await db.execute(select(AuthUser).where(AuthUser.id == user_uuid))).scalar_one_or_none()
        if not db_user or not db_user.is_active:
            return None
        roles = list(db_user.roles or [])
        role = roles[0] if roles else "user"
        return {"id": str(db_user.id), "username": db_user.username, "email": db_user.email, "role": role}
    except Exception:
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
    """
    if not token:
        return None
    try:
        token_data = jwt_manager.decode_token(token)
        if token_data.token_type != "access" or not token_data.user_id:
            return None
        return {"user_id": token_data.user_id, "username": token_data.username, "scopes": token_data.scopes}
    except Exception:
        return None

class AuthenticationError(Exception):
    """认证错误"""
    ...

class AuthorizationError(Exception):
    """授权错误"""
    ...
