"""
身份认证和授权系统
"""

from datetime import datetime, timedelta
import uuid
from src.core.utils.timezone_utils import from_timestamp, parse_iso_string, to_utc, utc_now
from typing import Any, Dict, List, Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import get_settings
from src.core.database import get_db
from src.core.redis import get_redis
from src.models.database.user import AuthUser

from src.core.logging import get_logger
logger = get_logger(__name__)

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

settings = get_settings()

class Token(BaseModel):
    """Token响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

class TokenData(BaseModel):
    """Token数据模型"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    token_type: Optional[str] = None
    jti: Optional[str] = None
    exp: Optional[datetime] = None

class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime
    last_login: Optional[datetime] = None

class UserInDB(User):
    """数据库用户模型"""
    hashed_password: str
    updated_at: datetime

class PasswordManager:
    """密码管理器"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)

class JWTManager:
    """JWT管理器"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = utc_now() + expires_delta
        else:
            expire = utc_now() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": utc_now(),
            "jti": str(uuid.uuid4()),
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = utc_now() + expires_delta
        else:
            expire = utc_now() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "iat": utc_now(),
            "jti": str(uuid.uuid4()),
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> TokenData:
        """解码令牌"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])
            token_type: str = payload.get("type")
            jti: str = payload.get("jti")
            exp_value = payload.get("exp")
            exp = None
            if isinstance(exp_value, (int, float)):
                exp = from_timestamp(exp_value)
            elif isinstance(exp_value, str):
                exp = parse_iso_string(exp_value)
            elif isinstance(exp_value, datetime):
                exp = to_utc(exp_value)
            
            if username is None and user_id is None:
                raise JWTError("Invalid token payload")
            
            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes,
                token_type=token_type,
                jti=jti,
                exp=exp
            )
            
        except JWTError as e:
            logger.error("JWT decode error", error=str(e))
            raise

class RBACManager:
    """基于角色的访问控制管理器"""
    
    # 角色权限映射
    ROLE_PERMISSIONS = {
        "admin": [
            "users:read", "users:write", "users:delete",
            "agents:read", "agents:write", "agents:delete",
            "tools:read", "tools:write", "tools:execute",
            "system:read", "system:write", "system:admin"
        ],
        "developer": [
            "agents:read", "agents:write",
            "tools:read", "tools:write", "tools:execute",
            "system:read"
        ],
        "user": [
            "agents:read",
            "tools:read", "tools:execute"
        ],
        "viewer": [
            "agents:read",
            "tools:read"
        ]
    }
    
    @classmethod
    def get_role_permissions(cls, role: str) -> List[str]:
        """获取角色权限"""
        return cls.ROLE_PERMISSIONS.get(role, [])
    
    @classmethod
    def get_user_permissions(cls, roles: List[str]) -> List[str]:
        """获取用户所有权限"""
        permissions = set()
        for role in roles:
            permissions.update(cls.get_role_permissions(role))
        return list(permissions)
    
    @classmethod
    def has_permission(cls, user_permissions: List[str], required_permission: str) -> bool:
        """检查是否有权限"""
        return required_permission in user_permissions
    
    @classmethod
    def has_any_permission(cls, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """检查是否有任一权限"""
        return any(perm in user_permissions for perm in required_permissions)
    
    @classmethod
    def has_all_permissions(cls, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """检查是否有所有权限"""
        return all(perm in user_permissions for perm in required_permissions)

# 全局实例
password_manager = PasswordManager()
jwt_manager = JWTManager()
rbac_manager = RBACManager()

async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str
) -> Optional[UserInDB]:
    """认证用户"""
    result = await db.execute(select(AuthUser).where(AuthUser.username == username))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        return None
    
    if not password_manager.verify_password(password, user.hashed_password):
        return None

    user.last_login = utc_now()
    await db.commit()
    await db.refresh(user)

    roles = list(user.roles or [])
    return UserInDB(
        id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        roles=roles,
        permissions=rbac_manager.get_user_permissions(roles),
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
        hashed_password=user.hashed_password,
    )

_REVOKED_JTI_PREFIX = "auth:revoked:jti:"

async def is_token_revoked(jti: Optional[str]) -> bool:
    if not jti:
        return False
    client = get_redis()
    if not client:
        return False
    return await client.exists(f"{_REVOKED_JTI_PREFIX}{jti}") > 0

async def revoke_token(token_data: TokenData) -> None:
    if not token_data.jti or not token_data.exp:
        return
    client = get_redis()
    if not client:
        return
    ttl = int((token_data.exp - utc_now()).total_seconds())
    if ttl <= 0:
        return
    await client.setex(f"{_REVOKED_JTI_PREFIX}{token_data.jti}", ttl, "1")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = jwt_manager.decode_token(token)
    except JWTError:
        raise credentials_exception

    if token_data.token_type != "access":
        raise credentials_exception

    if await is_token_revoked(token_data.jti):
        raise credentials_exception
    
    if not token_data.user_id:
        raise credentials_exception

    try:
        user_uuid = uuid.UUID(token_data.user_id)
    except Exception:
        raise credentials_exception

    result = await db.execute(select(AuthUser).where(AuthUser.id == user_uuid))
    db_user = result.scalar_one_or_none()
    if not db_user:
        raise credentials_exception

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

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

class PermissionChecker:
    """权限检查器"""
    
    def __init__(self, required_permissions: Union[str, List[str]], require_all: bool = True):
        """
        初始化权限检查器
        
        Args:
            required_permissions: 所需权限
            require_all: 是否需要所有权限（True）或任一权限（False）
        """
        if isinstance(required_permissions, str):
            self.required_permissions = [required_permissions]
        else:
            self.required_permissions = required_permissions
        self.require_all = require_all
    
    async def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        """检查用户权限"""
        if self.require_all:
            has_permission = rbac_manager.has_all_permissions(
                user.permissions,
                self.required_permissions
            )
        else:
            has_permission = rbac_manager.has_any_permission(
                user.permissions,
                self.required_permissions
            )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(self.required_permissions)}"
            )
        
        return user

# 便捷函数
def require_permission(permission: str):
    """需要单个权限"""
    return PermissionChecker(permission)

def require_any_permission(permissions: List[str]):
    """需要任一权限"""
    return PermissionChecker(permissions, require_all=False)

def require_all_permissions(permissions: List[str]):
    """需要所有权限"""
    return PermissionChecker(permissions, require_all=True)
