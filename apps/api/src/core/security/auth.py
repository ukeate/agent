"""
身份认证和授权系统
"""

from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Any, Dict, List, Optional, Union

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.database import get_session

logger = structlog.get_logger(__name__)

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
    scopes: List[str] = []
    exp: Optional[datetime] = None


class User(BaseModel):
    """用户模型"""
    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []
    permissions: List[str] = []


class UserInDB(User):
    """数据库用户模型"""
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


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
            "iat": utc_now()
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
            "iat": utc_now()
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
            
            if username is None and user_id is None:
                raise JWTError("Invalid token payload")
            
            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes,
                exp=payload.get("exp")
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
    # TODO: 实现从数据库获取用户
    # 这里是示例实现
    from src.db.models import User as DBUser
    
    # 查询用户
    # user = await db.query(DBUser).filter(DBUser.username == username).first()
    
    # 临时硬编码用户（实际应从数据库读取）
    fake_users_db = {
        "admin": {
            "id": "1",
            "username": "admin",
            "email": "admin@example.com",
            "hashed_password": password_manager.hash_password("admin123"),
            "roles": ["admin"],
            "disabled": False
        }
    }
    
    user_dict = fake_users_db.get(username)
    if not user_dict:
        return None
    
    if not password_manager.verify_password(password, user_dict["hashed_password"]):
        return None
    
    return UserInDB(
        **user_dict,
        permissions=rbac_manager.get_user_permissions(user_dict["roles"]),
        created_at=utc_now(),
        updated_at=utc_now()
    )


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_session)
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
    
    # TODO: 从数据库获取用户
    # user = await db.query(User).filter(User.id == token_data.user_id).first()
    
    # 临时实现
    if token_data.username == "admin":
        user = User(
            id="1",
            username="admin",
            email="admin@example.com",
            roles=["admin"],
            permissions=rbac_manager.get_user_permissions(["admin"])
        )
    else:
        raise credentials_exception
    
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    if current_user.disabled:
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