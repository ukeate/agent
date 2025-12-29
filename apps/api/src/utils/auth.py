"""
认证和授权工具函数
提供JWT token处理、密码加密等功能
"""

import jwt
import bcrypt
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Optional, Dict, Any
from functools import wraps
from fastapi import HTTPException, Request

from src.core.logging import get_logger
logger = get_logger(__name__)

# 配置常量
JWT_SECRET_KEY = "your-secret-key-here"  # 应该从环境变量读取
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7天

class AuthUtils:
    """认证工具类"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """密码哈希"""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"密码哈希失败: {str(e)}")
            raise
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'), 
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"密码验证失败: {str(e)}")
            return False
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = utc_now() + expires_delta
            else:
                expire = utc_now() + timedelta(minutes=JWT_EXPIRE_MINUTES)
            
            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            
            logger.debug(f"创建访问令牌成功，用户: {data.get('sub', 'unknown')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"创建访问令牌失败: {str(e)}")
            raise
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("令牌已过期")
            return None
        except jwt.JWTError as e:
            logger.warning(f"令牌验证失败: {str(e)}")
            return None
    
    @staticmethod
    def get_current_user_id(token: str) -> Optional[str]:
        """从令牌获取当前用户ID"""
        payload = AuthUtils.verify_token(token)
        if payload:
            return payload.get("sub")
        return None
    
    @staticmethod
    def extract_token_from_header(authorization: str) -> Optional[str]:
        """从请求头提取令牌"""
        if not authorization:
            return None
        
        parts = authorization.split()
        if parts[0].lower() != "bearer" or len(parts) != 2:
            return None
        
        return parts[1]

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        # 获取请求对象
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            raise HTTPException(status_code=500, detail="内部服务器错误")
        
        # 获取授权头
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="缺少认证令牌")
        
        # 提取和验证令牌
        token = AuthUtils.extract_token_from_header(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="无效的认证令牌格式")
        
        payload = AuthUtils.verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="无效或过期的认证令牌")
        
        # 将用户信息添加到请求中
        request.state.user_id = payload.get("sub")
        request.state.user_payload = payload
        
        return await f(*args, **kwargs)
    
    return decorated_function

def require_role(required_roles: list):
    """角色权限装饰器"""
    def decorator(f):
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            # 获取请求对象
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(status_code=500, detail="内部服务器错误")
            
            # 检查用户是否已认证
            if not hasattr(request.state, 'user_payload'):
                raise HTTPException(status_code=401, detail="未认证")
            
            # 检查用户角色
            user_roles = request.state.user_payload.get("roles", [])
            if not any(role in user_roles for role in required_roles):
                raise HTTPException(status_code=403, detail="权限不足")
            
            return await f(*args, **kwargs)
        
        return decorated_function
    return decorator

class SessionManager:
    """会话管理器"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, user_id: str, additional_data: Optional[Dict[str, Any]] = None) -> str:
        """创建会话"""
        import uuid
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "created_at": utc_now(),
            "last_activity": utc_now(),
            "data": additional_data or {}
        }
        
        self.active_sessions[session_id] = session_data
        logger.info(f"创建会话: {session_id}, 用户: {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话"""
        session = self.active_sessions.get(session_id)
        if session:
            # 更新最后活动时间
            session["last_activity"] = utc_now()
        return session
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """更新会话"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["data"].update(data)
            self.active_sessions[session_id]["last_activity"] = utc_now()
            return True
        return False
    
    def destroy_session(self, session_id: str) -> bool:
        """销毁会话"""
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id]["user_id"]
            del self.active_sessions[session_id]
            logger.info(f"销毁会话: {session_id}, 用户: {user_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self, expire_minutes: int = 60) -> int:
        """清理过期会话"""
        current_time = utc_now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            last_activity = session_data["last_activity"]
            if current_time - last_activity > timedelta(minutes=expire_minutes):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        if expired_sessions:
            logger.info(f"清理过期会话: {len(expired_sessions)} 个")
        
        return len(expired_sessions)

# 全局会话管理器实例
session_manager = SessionManager()
