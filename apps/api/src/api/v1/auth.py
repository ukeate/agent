"""
身份认证API端点
"""

import json
import uuid
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.core.security.auth import (
    Token,
    TokenData,
    User,
    authenticate_user,
    get_current_active_user,
    oauth2_scheme,
    jwt_manager,
    password_manager,
    rbac_manager,
    revoke_token,
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

_LOGIN_HISTORY_KEY_PREFIX = "auth:login_history:"
_USER_SESSIONS_KEY_PREFIX = "auth:user_sessions:"
_SESSION_KEY_PREFIX = "auth:session:"

class PermissionCheckRequest(ApiBaseModel):
    resource: str
    action: str

async def _record_login(
    *,
    redis_client,
    user_id: str,
    ip_address: Optional[str],
    user_agent: str,
    success: bool,
) -> None:
    if not redis_client or not user_id:
        return
    key = f"{_LOGIN_HISTORY_KEY_PREFIX}{user_id}"
    payload = {
        "timestamp": utc_now().isoformat(),
        "ip_address": ip_address or "",
        "user_agent": user_agent,
        "location": None,
        "success": bool(success),
    }
    await redis_client.lpush(key, json.dumps(payload, ensure_ascii=False))
    await redis_client.ltrim(key, 0, 199)

async def _store_session(
    *,
    redis_client,
    token_data: TokenData,
    ip_address: Optional[str],
    user_agent: str,
) -> None:
    if not redis_client or not token_data.user_id or not token_data.jti or not token_data.exp:
        return
    ttl = int((token_data.exp - utc_now()).total_seconds())
    if ttl <= 0:
        return
    session_payload = {
        "session_id": token_data.jti,
        "ip_address": ip_address or "",
        "user_agent": user_agent,
        "created_at": utc_now().isoformat(),
        "last_activity": utc_now().isoformat(),
        "expires_at": token_data.exp.isoformat(),
    }
    await redis_client.setex(
        f"{_SESSION_KEY_PREFIX}{token_data.jti}",
        ttl,
        json.dumps(session_payload, ensure_ascii=False),
    )
    await redis_client.sadd(f"{_USER_SESSIONS_KEY_PREFIX}{token_data.user_id}", token_data.jti)
    await redis_client.expire(
        f"{_USER_SESSIONS_KEY_PREFIX}{token_data.user_id}",
        jwt_manager.refresh_token_expire_days * 86400,
    )

async def _revoke_session_id(*, redis_client, user_id: str, session_id: str) -> None:
    raw = await redis_client.get(f"{_SESSION_KEY_PREFIX}{session_id}")
    if raw:
        try:
            data = json.loads(raw)
            exp = data.get("expires_at")
            if exp:
                await revoke_token(TokenData(jti=session_id, exp=exp))
        except Exception:
            logger.exception("撤销会话令牌失败", exc_info=True)

    await redis_client.delete(f"{_SESSION_KEY_PREFIX}{session_id}")
    await redis_client.srem(f"{_USER_SESSIONS_KEY_PREFIX}{user_id}", session_id)

class RegisterRequest(ApiBaseModel):
    username: str = Field(..., min_length=1)
    email: Optional[str] = None
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class RefreshRequest(ApiBaseModel):
    refresh_token: str

class ProfileUpdateRequest(ApiBaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None

class PasswordChangeRequest(ApiBaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db: AsyncSession = Depends(get_db)
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
    
    redis_client = get_redis()
    ip_address = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent", "") if request else ""

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

    try:
        token_data = jwt_manager.decode_token(access_token)
        await _store_session(
            redis_client=redis_client,
            token_data=token_data,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    except Exception:
        logger.exception("记录会话失败", exc_info=True)

    try:
        await _record_login(
            redis_client=redis_client,
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
        )
    except Exception:
        logger.exception("记录登录失败", exc_info=True)
    
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
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    使用刷新令牌获取新的访问令牌
    
    - **refresh_token**: 刷新令牌
    """
    try:
        token_data = jwt_manager.decode_token(request.refresh_token)
        if token_data.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not token_data.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            user_uuid = uuid.UUID(token_data.user_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        db_user = (await db.execute(select(AuthUser).where(AuthUser.id == user_uuid))).scalar_one_or_none()
        if not db_user or not db_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
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

        try:
            redis_client = get_redis()
            session_token_data = jwt_manager.decode_token(access_token)
            await _store_session(
                redis_client=redis_client,
                token_data=session_token_data,
                ip_address=None,
                user_agent="",
            )
        except Exception:
            logger.exception("记录刷新会话失败", exc_info=True)
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=jwt_manager.access_token_expire_minutes * 60,
            refresh_token=request.refresh_token,
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
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_active_user)
):
    """
    用户登出
    
    需要有效的访问令牌
    """
    try:
        token_data = jwt_manager.decode_token(token)
        await revoke_token(token_data)
        redis_client = get_redis()
        if redis_client and token_data.jti:
            await _revoke_session_id(redis_client=redis_client, user_id=current_user.id, session_id=token_data.jti)
    except Exception as e:
        logger.error("Token revoke failed", error=str(e))
    
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

@router.post("/register", response_model=User)
async def register_user(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    roles = ["user"]
    is_superuser = False

    db_user = AuthUser(
        username=request.username,
        email=request.email,
        full_name=request.full_name,
        hashed_password=password_manager.hash_password(request.password),
        roles=roles,
        is_superuser=is_superuser,
        is_active=True,
    )
    db.add(db_user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="用户名或邮箱已存在")
    await db.refresh(db_user)

    return User(
        id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        full_name=db_user.full_name,
        is_active=db_user.is_active,
        is_superuser=db_user.is_superuser,
        roles=list(db_user.roles or []),
        permissions=rbac_manager.get_user_permissions(list(db_user.roles or [])),
        created_at=db_user.created_at,
        last_login=db_user.last_login,
    )

@router.get("/users")
async def list_users(db: AsyncSession = Depends(get_db)):
    """获取用户列表"""
    result = await db.execute(select(AuthUser).order_by(AuthUser.created_at.desc()))
    users = result.scalars().all()
    payload = [
        User(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            roles=list(user.roles or []),
            permissions=rbac_manager.get_user_permissions(list(user.roles or [])),
            created_at=user.created_at,
            last_login=user.last_login,
        ).model_dump()
        for user in users
    ]
    return {"users": payload, "total": len(payload)}

@router.put("/me", response_model=User)
async def update_current_user_profile(
    request: ProfileUpdateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    更新当前用户信息
    
    - **full_name**: 全名
    - **email**: 邮箱地址
    """
    user_uuid = uuid.UUID(current_user.id)
    db_user = (await db.execute(select(AuthUser).where(AuthUser.id == user_uuid))).scalar_one_or_none()
    if not db_user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if request.full_name is not None:
        db_user.full_name = request.full_name
    if request.email is not None:
        db_user.email = request.email

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="邮箱已存在")
    await db.refresh(db_user)

    roles = list(db_user.roles or [])
    return User(
        id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        full_name=db_user.full_name,
        is_active=db_user.is_active,
        is_superuser=db_user.is_superuser,
        roles=roles,
        permissions=rbac_manager.get_user_permissions(roles),
        created_at=db_user.created_at,
        last_login=db_user.last_login,
    )

@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    修改密码
    
    - **current_password**: 当前密码
    - **new_password**: 新密码
    """
    user_uuid = uuid.UUID(current_user.id)
    db_user = (await db.execute(select(AuthUser).where(AuthUser.id == user_uuid))).scalar_one_or_none()
    if not db_user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if not password_manager.verify_password(request.current_password, db_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="当前密码错误")

    db_user.hashed_password = password_manager.hash_password(request.new_password)
    await db.commit()
    
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

@router.post("/check-permission")
async def check_permission(
    request: PermissionCheckRequest,
    current_user: User = Depends(get_current_active_user),
):
    permission = f"{request.resource}:{request.action}"
    return {"has_permission": rbac_manager.has_permission(current_user.permissions, permission)}

@router.get("/login-history")
async def get_login_history(
    limit: int = Query(10, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    redis_client = get_redis()
    if not redis_client:
        return []
    raw_items = await redis_client.lrange(f"{_LOGIN_HISTORY_KEY_PREFIX}{current_user.id}", 0, limit - 1)
    out = []
    for raw in raw_items:
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out

@router.get("/sessions")
async def get_sessions(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_active_user),
):
    redis_client = get_redis()
    if not redis_client:
        return []

    current_jti = None
    try:
        token_data = jwt_manager.decode_token(token)
        current_jti = token_data.jti
    except Exception:
        current_jti = None

    set_key = f"{_USER_SESSIONS_KEY_PREFIX}{current_user.id}"
    session_ids = list(await redis_client.smembers(set_key) or [])

    sessions = []
    for session_id in session_ids:
        raw = await redis_client.get(f"{_SESSION_KEY_PREFIX}{session_id}")
        if not raw:
            await redis_client.srem(set_key, session_id)
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        sessions.append(
            {
                "session_id": data.get("session_id", session_id),
                "ip_address": data.get("ip_address", ""),
                "user_agent": data.get("user_agent", ""),
                "created_at": data.get("created_at"),
                "last_activity": data.get("last_activity"),
                "is_current": bool(session_id == current_jti),
            }
        )
    return sessions

@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis不可用")
    await _revoke_session_id(redis_client=redis_client, user_id=current_user.id, session_id=session_id)
    return {"message": "session revoked"}

@router.delete("/sessions")
async def revoke_all_sessions(
    current_user: User = Depends(get_current_active_user),
):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis不可用")
    set_key = f"{_USER_SESSIONS_KEY_PREFIX}{current_user.id}"
    session_ids = list(await redis_client.smembers(set_key) or [])
    for session_id in session_ids:
        await _revoke_session_id(redis_client=redis_client, user_id=current_user.id, session_id=session_id)
    return {"message": "sessions revoked"}

@router.get("/verify")
async def verify_token(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_active_user)
):
    """
    验证访问令牌是否有效
    
    需要有效的访问令牌
    """
    expires_at = None
    try:
        token_data = jwt_manager.decode_token(token)
        if token_data.exp:
            expires_at = token_data.exp.isoformat()
    except Exception:
        expires_at = None
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "expires_at": expires_at,
        "permissions": current_user.permissions,
        "roles": current_user.roles,
    }
