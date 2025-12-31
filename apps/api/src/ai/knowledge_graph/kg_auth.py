"""
知识图谱认证和安全模块 - JWT认证、权限控制、API密钥管理
"""

import asyncio
import json
import jwt
import hashlib
import secrets
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import bcrypt
from src.core.logging import get_logger, setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class Permission(Enum):
    """权限类型"""
    READ_GRAPH = "read:graph"
    WRITE_GRAPH = "write:graph"
    DELETE_GRAPH = "delete:graph"
    QUERY_SPARQL = "query:sparql"
    UPDATE_SPARQL = "update:sparql"
    MANAGE_VERSIONS = "manage:versions"
    IMPORT_DATA = "import:data"
    EXPORT_DATA = "export:data"
    MANAGE_USERS = "manage:users"
    ADMIN_ALL = "admin:all"

class Role(Enum):
    """用户角色"""
    ANONYMOUS = "anonymous"
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

# 角色权限映射
ROLE_PERMISSIONS = {
    Role.ANONYMOUS: {Permission.READ_GRAPH},
    Role.VIEWER: {
        Permission.READ_GRAPH,
        Permission.QUERY_SPARQL,
        Permission.EXPORT_DATA
    },
    Role.EDITOR: {
        Permission.READ_GRAPH,
        Permission.WRITE_GRAPH,
        Permission.QUERY_SPARQL,
        Permission.UPDATE_SPARQL,
        Permission.IMPORT_DATA,
        Permission.EXPORT_DATA,
        Permission.MANAGE_VERSIONS
    },
    Role.ADMIN: {
        Permission.READ_GRAPH,
        Permission.WRITE_GRAPH,
        Permission.DELETE_GRAPH,
        Permission.QUERY_SPARQL,
        Permission.UPDATE_SPARQL,
        Permission.MANAGE_VERSIONS,
        Permission.IMPORT_DATA,
        Permission.EXPORT_DATA,
        Permission.MANAGE_USERS
    },
    Role.SUPER_ADMIN: {Permission.ADMIN_ALL}
}

@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: Role
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    api_keys: List[str] = None

@dataclass
class APIKey:
    """API密钥"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Session:
    """会话信息"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    permissions: Set[Permission]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AuthResult:
    """认证结果"""
    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[Role] = None
    permissions: Set[Permission] = None
    session_id: Optional[str] = None
    error_message: Optional[str] = None
    expires_at: Optional[datetime] = None

class SecurityConfig:
    """安全配置"""
    
    def __init__(self):
        # JWT配置
        self.jwt_secret_key = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        
        # 密码安全
        self.password_min_length = 8
        self.password_require_special = True
        self.password_require_digits = True
        self.password_require_uppercase = True
        
        # 会话管理
        self.session_timeout_hours = 8
        self.max_sessions_per_user = 5
        
        # API密钥
        self.api_key_length = 32
        self.api_key_prefix = "kg_"
        
        # 速率限制
        self.rate_limit_requests_per_minute = 100
        self.rate_limit_window_minutes = 1
        
        # 审计日志
        self.enable_audit_logging = True
        self.log_failed_attempts = True
        
        # IP白名单/黑名单
        self.ip_whitelist: Set[str] = set()
        self.ip_blacklist: Set[str] = set()

class KnowledgeGraphAuth:
    """知识图谱认证管理器"""
    
    def __init__(self, config: SecurityConfig = None, storage_path: str = "/tmp/kg_auth"):
        self.config = config or SecurityConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 存储
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Session] = {}
        
        # 速率限制跟踪
        self.rate_limit_tracker: Dict[str, List[float]] = {}
        
        # 安全事件
        self.security_events: List[Dict[str, Any]] = []
        
        self._setup_logging()
        self._load_data()
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def _load_data(self):
        """加载认证数据"""
        try:
            # 加载用户
            users_file = self.storage_path / "users.json"
            if users_file.exists():
                with open(users_file, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                    
                for user_data in users_data:
                    user = User(**user_data)
                    user.created_at = datetime.fromisoformat(user_data["created_at"])
                    if user_data.get("last_login"):
                        user.last_login = datetime.fromisoformat(user_data["last_login"])
                    user.role = Role(user_data["role"])
                    user.permissions = {Permission(p) for p in user_data["permissions"]}
                    user.api_keys = user_data.get("api_keys", [])
                    self.users[user.user_id] = user
            
            # 加载API密钥
            keys_file = self.storage_path / "api_keys.json"
            if keys_file.exists():
                with open(keys_file, 'r', encoding='utf-8') as f:
                    keys_data = json.load(f)
                    
                for key_data in keys_data:
                    api_key = APIKey(**key_data)
                    api_key.created_at = datetime.fromisoformat(key_data["created_at"])
                    if key_data.get("expires_at"):
                        api_key.expires_at = datetime.fromisoformat(key_data["expires_at"])
                    if key_data.get("last_used"):
                        api_key.last_used = datetime.fromisoformat(key_data["last_used"])
                    api_key.permissions = {Permission(p) for p in key_data["permissions"]}
                    self.api_keys[api_key.key_id] = api_key
            
            self.logger.info(f"已加载 {len(self.users)} 个用户和 {len(self.api_keys)} 个API密钥")
            
        except Exception as e:
            self.logger.warning(f"加载认证数据失败: {e}")
    
    def _save_data(self):
        """保存认证数据"""
        try:
            # 保存用户
            users_data = []
            for user in self.users.values():
                user_dict = asdict(user)
                user_dict["created_at"] = user.created_at.isoformat()
                if user.last_login:
                    user_dict["last_login"] = user.last_login.isoformat()
                user_dict["role"] = user.role.value
                user_dict["permissions"] = [p.value for p in user.permissions]
                users_data.append(user_dict)
            
            with open(self.storage_path / "users.json", 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False)
            
            # 保存API密钥
            keys_data = []
            for api_key in self.api_keys.values():
                key_dict = asdict(api_key)
                key_dict["created_at"] = api_key.created_at.isoformat()
                if api_key.expires_at:
                    key_dict["expires_at"] = api_key.expires_at.isoformat()
                if api_key.last_used:
                    key_dict["last_used"] = api_key.last_used.isoformat()
                key_dict["permissions"] = [p.value for p in api_key.permissions]
                keys_data.append(key_dict)
            
            with open(self.storage_path / "api_keys.json", 'w', encoding='utf-8') as f:
                json.dump(keys_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存认证数据失败: {e}")
            raise
    
    async def create_user(self,
                         username: str,
                         email: str,
                         password: str,
                         role: Role = Role.VIEWER) -> User:
        """创建用户"""
        try:
            # 验证密码强度
            if not self._validate_password(password):
                raise ValueError("密码不符合安全要求")
            
            # 检查用户名和邮箱是否已存在
            if any(u.username == username for u in self.users.values()):
                raise ValueError("用户名已存在")
            
            if any(u.email == email for u in self.users.values()):
                raise ValueError("邮箱已存在")
            
            # 生成用户ID
            user_id = self._generate_user_id()
            
            # 哈希密码
            password_hash = self._hash_password(password)
            
            # 获取角色权限
            permissions = ROLE_PERMISSIONS.get(role, set())
            
            # 创建用户
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                permissions=permissions,
                created_at=utc_now(),
                api_keys=[]
            )
            
            self.users[user_id] = user
            self._save_data()
            
            self.logger.info(f"已创建用户: {username} (ID: {user_id})")
            return user
            
        except Exception as e:
            self.logger.error(f"创建用户失败 {username}: {e}")
            raise
    
    async def authenticate_user(self, username: str, password: str, ip_address: str = None) -> AuthResult:
        """用户认证"""
        try:
            # 查找用户
            user = None
            for u in self.users.values():
                if u.username == username and u.is_active:
                    user = u
                    break
            
            if not user:
                await self._log_security_event("login_failed", {"username": username, "reason": "user_not_found"})
                return AuthResult(success=False, error_message="用户名或密码错误")
            
            # 验证密码
            if not self._verify_password(password, user.password_hash):
                await self._log_security_event("login_failed", {
                    "user_id": user.user_id,
                    "username": username,
                    "reason": "invalid_password"
                })
                return AuthResult(success=False, error_message="用户名或密码错误")
            
            # 检查IP限制
            if not self._check_ip_access(ip_address):
                await self._log_security_event("login_blocked", {
                    "user_id": user.user_id,
                    "ip_address": ip_address,
                    "reason": "ip_restricted"
                })
                return AuthResult(success=False, error_message="访问被限制")
            
            # 创建会话
            session = await self._create_session(user, ip_address or "unknown")
            
            # 更新用户最后登录时间
            user.last_login = utc_now()
            self._save_data()
            
            await self._log_security_event("login_success", {
                "user_id": user.user_id,
                "username": username,
                "session_id": session.session_id
            })
            
            return AuthResult(
                success=True,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                permissions=user.permissions,
                session_id=session.session_id,
                expires_at=session.expires_at
            )
            
        except Exception as e:
            self.logger.error(f"用户认证失败 {username}: {e}")
            return AuthResult(success=False, error_message="认证系统错误")
    
    async def authenticate_api_key(self, api_key: str, ip_address: str = None) -> AuthResult:
        """API密钥认证"""
        try:
            # 提取密钥ID
            if not api_key.startswith(self.config.api_key_prefix):
                return AuthResult(success=False, error_message="无效的API密钥格式")
            
            key_data = api_key[len(self.config.api_key_prefix):]
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()
            
            # 查找API密钥
            api_key_obj = None
            for ak in self.api_keys.values():
                if ak.key_hash == key_hash and ak.is_active:
                    api_key_obj = ak
                    break
            
            if not api_key_obj:
                await self._log_security_event("api_auth_failed", {"reason": "invalid_key"})
                return AuthResult(success=False, error_message="无效的API密钥")
            
            # 检查过期时间
            if api_key_obj.expires_at and utc_now() > api_key_obj.expires_at:
                await self._log_security_event("api_auth_failed", {
                    "key_id": api_key_obj.key_id,
                    "reason": "expired"
                })
                return AuthResult(success=False, error_message="API密钥已过期")
            
            # 检查IP限制
            if not self._check_ip_access(ip_address):
                await self._log_security_event("api_auth_blocked", {
                    "key_id": api_key_obj.key_id,
                    "ip_address": ip_address,
                    "reason": "ip_restricted"
                })
                return AuthResult(success=False, error_message="访问被限制")
            
            # 更新使用记录
            api_key_obj.last_used = utc_now()
            api_key_obj.usage_count += 1
            
            # 获取用户信息
            user = self.users.get(api_key_obj.user_id)
            if not user or not user.is_active:
                return AuthResult(success=False, error_message="关联用户不存在或已停用")
            
            await self._log_security_event("api_auth_success", {
                "key_id": api_key_obj.key_id,
                "user_id": user.user_id
            })
            
            return AuthResult(
                success=True,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                permissions=api_key_obj.permissions
            )
            
        except Exception as e:
            self.logger.error(f"API密钥认证失败: {e}")
            return AuthResult(success=False, error_message="认证系统错误")
    
    async def authenticate_jwt(self, token: str) -> AuthResult:
        """JWT令牌认证"""
        try:
            # 解码JWT
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            
            if not user_id:
                return AuthResult(success=False, error_message="无效的JWT令牌")
            
            # 验证用户
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return AuthResult(success=False, error_message="用户不存在或已停用")
            
            # 验证会话
            session = self.sessions.get(session_id) if session_id else None
            if session and utc_now() > session.expires_at:
                # 会话过期，删除会话
                del self.sessions[session_id]
                session = None
            
            return AuthResult(
                success=True,
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                permissions=user.permissions,
                session_id=session_id if session else None
            )
            
        except jwt.ExpiredSignatureError:
            return AuthResult(success=False, error_message="JWT令牌已过期")
        except jwt.InvalidTokenError:
            return AuthResult(success=False, error_message="无效的JWT令牌")
        except Exception as e:
            self.logger.error(f"JWT认证失败: {e}")
            return AuthResult(success=False, error_message="认证系统错误")
    
    async def create_api_key(self,
                           user_id: str,
                           name: str,
                           permissions: Set[Permission] = None,
                           expires_days: int = None) -> Tuple[str, APIKey]:
        """创建API密钥"""
        try:
            user = self.users.get(user_id)
            if not user:
                raise ValueError("用户不存在")
            
            # 生成密钥
            key_value = secrets.token_urlsafe(self.config.api_key_length)
            full_key = f"{self.config.api_key_prefix}{key_value}"
            key_hash = hashlib.sha256(key_value.encode()).hexdigest()
            
            # 设置权限（不能超过用户权限）
            if permissions is None:
                permissions = user.permissions
            else:
                permissions = permissions & user.permissions
            
            # 计算过期时间
            expires_at = None
            if expires_days:
                expires_at = utc_now() + timedelta(days=expires_days)
            
            # 创建API密钥对象
            api_key = APIKey(
                key_id=self._generate_key_id(),
                key_hash=key_hash,
                user_id=user_id,
                name=name,
                permissions=permissions,
                created_at=utc_now(),
                expires_at=expires_at
            )
            
            self.api_keys[api_key.key_id] = api_key
            
            # 更新用户的API密钥列表
            if not user.api_keys:
                user.api_keys = []
            user.api_keys.append(api_key.key_id)
            
            self._save_data()
            
            self.logger.info(f"已为用户 {user_id} 创建API密钥 {name}")
            return full_key, api_key
            
        except Exception as e:
            self.logger.error(f"创建API密钥失败: {e}")
            raise
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """撤销API密钥"""
        try:
            api_key = self.api_keys.get(key_id)
            if not api_key:
                return False
            
            api_key.is_active = False
            self._save_data()
            
            self.logger.info(f"已撤销API密钥 {key_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"撤销API密钥失败 {key_id}: {e}")
            return False
    
    async def check_permission(self, auth_result: AuthResult, required_permission: Permission) -> bool:
        """检查权限"""
        if not auth_result.success or not auth_result.permissions:
            return False
        
        # 超级管理员拥有所有权限
        if Permission.ADMIN_ALL in auth_result.permissions:
            return True
        
        return required_permission in auth_result.permissions
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """检查速率限制"""
        current_time = time.time()
        window_start = current_time - (self.config.rate_limit_window_minutes * 60)
        
        # 获取或创建跟踪记录
        if identifier not in self.rate_limit_tracker:
            self.rate_limit_tracker[identifier] = []
        
        requests = self.rate_limit_tracker[identifier]
        
        # 清理过期请求
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # 检查限制
        if len(requests) >= self.config.rate_limit_requests_per_minute:
            return False
        
        # 记录当前请求
        requests.append(current_time)
        return True
    
    def _validate_password(self, password: str) -> bool:
        """验证密码强度"""
        if len(password) < self.config.password_min_length:
            return False
        
        if self.config.password_require_digits and not any(c.isdigit() for c in password):
            return False
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            return False
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;':\".,<>?" for c in password):
            return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
    def _check_ip_access(self, ip_address: str) -> bool:
        """检查IP访问权限"""
        if not ip_address:
            return True
        
        # 检查黑名单
        if ip_address in self.config.ip_blacklist:
            return False
        
        # 检查白名单（如果配置了白名单）
        if self.config.ip_whitelist and ip_address not in self.config.ip_whitelist:
            return False
        
        return True
    
    async def _create_session(self, user: User, ip_address: str, user_agent: str = "") -> Session:
        """创建会话"""
        session_id = self._generate_session_id()
        expires_at = utc_now() + timedelta(hours=self.config.session_timeout_hours)
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=utc_now(),
            expires_at=expires_at,
            last_activity=utc_now(),
            ip_address=ip_address,
            user_agent=user_agent,
            permissions=user.permissions
        )
        
        # 限制用户的同时会话数
        user_sessions = [s for s in self.sessions.values() if s.user_id == user.user_id]
        if len(user_sessions) >= self.config.max_sessions_per_user:
            # 删除最旧的会话
            oldest_session = min(user_sessions, key=lambda x: x.last_activity)
            del self.sessions[oldest_session.session_id]
        
        self.sessions[session_id] = session
        return session
    
    async def _log_security_event(self, event_type: str, data: Dict[str, Any]):
        """记录安全事件"""
        if self.config.enable_audit_logging:
            event = {
                "timestamp": utc_now().isoformat(),
                "event_type": event_type,
                "data": data
            }
            self.security_events.append(event)
            
            # 保持最近1000个事件
            if len(self.security_events) > 1000:
                self.security_events.pop(0)
    
    def _generate_user_id(self) -> str:
        """生成用户ID"""
        return f"user_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _generate_key_id(self) -> str:
        """生成API密钥ID"""
        return f"key_{int(time.time())}_{secrets.token_hex(8)}"
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"sess_{secrets.token_urlsafe(32)}"
    
    async def get_security_statistics(self) -> Dict[str, Any]:
        """获取安全统计"""
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() if k.is_active]),
            "active_sessions": len(self.sessions),
            "security_events_count": len(self.security_events),
            "users_by_role": {
                role.value: len([u for u in self.users.values() if u.role == role])
                for role in Role
            }
        }

# 装饰器
def require_permission(permission: Permission):
    """权限检查装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 这里应该从请求上下文中获取认证信息
            # 简化实现，实际项目中需要与FastAPI等框架集成
            auth_result = kwargs.get('auth_result')
            if not auth_result:
                raise PermissionError("需要认证")
            
            kg_auth = kwargs.get('kg_auth')
            if not kg_auth:
                raise ValueError("缺少认证管理器")
            
            if not await kg_auth.check_permission(auth_result, permission):
                raise PermissionError(f"需要权限: {permission.value}")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试认证模块
    async def test_auth():
        setup_logging()
        logger.info("测试认证模块")
        
        # 创建认证管理器
        auth = KnowledgeGraphAuth(storage_path="/tmp/test_kg_auth")
        
        # 创建用户
        user = await auth.create_user(
            username="testuser",
            email="test@example.com",
            password="Test123!@#",
            role=Role.EDITOR
        )
        logger.info("创建用户", username=user.username)
        
        # 用户登录
        auth_result = await auth.authenticate_user("testuser", "Test123!@#", "127.0.0.1")
        logger.info("用户认证结果", success=auth_result.success)
        
        if auth_result.success:
            # 创建API密钥
            api_key, key_obj = await auth.create_api_key(
                user_id=user.user_id,
                name="测试密钥",
                permissions={Permission.READ_GRAPH, Permission.QUERY_SPARQL}
            )
            logger.info("创建API密钥", api_key_prefix=api_key[:20])
            
            # API密钥认证
            api_auth_result = await auth.authenticate_api_key(api_key, "127.0.0.1")
            logger.info("API密钥认证结果", success=api_auth_result.success)
            
            # 权限检查
            has_read_perm = await auth.check_permission(auth_result, Permission.READ_GRAPH)
            has_admin_perm = await auth.check_permission(auth_result, Permission.ADMIN_ALL)
            logger.info("权限检查结果", read_permission=has_read_perm, admin_permission=has_admin_perm)
            
            # 速率限制检查
            for i in range(5):
                allowed = await auth.check_rate_limit("test_client")
                logger.info("限流检查结果", request_index=i + 1, allowed=allowed)
        
        # 获取统计信息
        stats = await auth.get_security_statistics()
        logger.info("安全统计", stats=stats)
        logger.info("认证模块测试完成")
    
    asyncio.run(test_auth())
