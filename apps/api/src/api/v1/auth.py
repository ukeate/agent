"""
身份认证API端点
"""

from datetime import timedelta
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_session
from src.core.security.auth import (
    Token,
    User,
    authenticate_user,
    get_current_active_user,
    jwt_manager,
    password_manager,
    rbac_manager,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_session)
):
    """
    用户登录获取访问令牌
    
    - **username**: 用户名
    - **password**: 密码
    - **scope**: 可选的权限范围
    """
    # 认证用户
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(
            "Failed login attempt",
            username=form_data.username
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=jwt_manager.access_token_expire_minutes)
    access_token = jwt_manager.create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "scopes": form_data.scopes or []
        },
        expires_delta=access_token_expires
    )
    
    # 创建刷新令牌
    refresh_token = jwt_manager.create_refresh_token(
        data={
            "sub": user.username,
            "user_id": user.id
        }
    )
    
    logger.info(
        "User logged in successfully",
        user_id=user.id,
        username=user.username
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=jwt_manager.access_token_expire_minutes * 60,
        refresh_token=refresh_token,
        scope=" ".join(form_data.scopes) if form_data.scopes else None
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_session)
):
    """
    使用刷新令牌获取新的访问令牌
    
    - **refresh_token**: 刷新令牌
    """
    try:
        token_data = jwt_manager.decode_token(refresh_token)
        
        # TODO: 验证refresh token类型和其他安全检查
        
        # 创建新的访问令牌
        access_token_expires = timedelta(minutes=jwt_manager.access_token_expire_minutes)
        access_token = jwt_manager.create_access_token(
            data={
                "sub": token_data.username,
                "user_id": token_data.user_id,
                "scopes": token_data.scopes
            },
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=jwt_manager.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
            scope=" ".join(token_data.scopes) if token_data.scopes else None
        )
        
    except Exception as e:
        logger.error("Failed to refresh token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    用户登出
    
    需要有效的访问令牌
    """
    # TODO: 实现令牌黑名单或撤销机制
    
    logger.info(
        "User logged out",
        user_id=current_user.id,
        username=current_user.username
    )
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前用户信息
    
    需要有效的访问令牌
    """
    return current_user


@router.put("/me")
async def update_current_user_profile(
    full_name: Optional[str] = None,
    email: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_session)
):
    """
    更新当前用户信息
    
    - **full_name**: 全名
    - **email**: 邮箱地址
    """
    # TODO: 实现用户信息更新逻辑
    
    updates = {}
    if full_name is not None:
        updates["full_name"] = full_name
    if email is not None:
        updates["email"] = email
    
    if updates:
        logger.info(
            "User profile updated",
            user_id=current_user.id,
            updates=updates
        )
    
    return {
        "message": "Profile updated successfully",
        "updated_fields": list(updates.keys())
    }


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_session)
):
    """
    修改密码
    
    - **current_password**: 当前密码
    - **new_password**: 新密码
    """
    # TODO: 验证当前密码并更新新密码
    
    # 临时实现
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # 哈希新密码
    hashed_password = password_manager.hash_password(new_password)
    
    logger.info(
        "Password changed",
        user_id=current_user.id
    )
    
    return {"message": "Password changed successfully"}


@router.get("/permissions")
async def get_current_user_permissions(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前用户权限列表
    
    需要有效的访问令牌
    """
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "roles": current_user.roles,
        "permissions": current_user.permissions
    }


@router.get("/verify")
async def verify_token(
    current_user: User = Depends(get_current_active_user)
):
    """
    验证访问令牌是否有效
    
    需要有效的访问令牌
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username
    }