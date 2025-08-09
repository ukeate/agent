# Backend Architecture

基于FastAPI和选择的技术栈，以下是后端特定架构的详细设计：

## Service Architecture

### Controller/Route Organization
```text
src/
├── api/
│   ├── v1/
│   │   ├── agents.py              # 智能体管理路由
│   │   ├── conversations.py       # 对话管理路由
│   │   ├── messages.py            # 消息处理路由
│   │   ├── tasks.py               # 任务管理路由
│   │   ├── dag_executions.py      # DAG执行路由
│   │   ├── knowledge.py           # 知识库路由
│   │   ├── rag.py                 # RAG查询路由
│   │   └── auth.py                # 认证路由
│   ├── deps.py                    # 依赖注入
│   ├── middleware.py              # 中间件
│   └── exceptions.py              # 异常处理
├── core/
│   ├── config.py                  # 配置管理
│   ├── security.py                # 安全相关
│   ├── database.py                # 数据库连接
│   └── logging.py                 # 日志配置
├── services/
│   ├── agent_service.py           # 智能体业务逻辑
│   ├── conversation_service.py    # 对话业务逻辑
│   ├── task_service.py            # 任务业务逻辑
│   └── rag_service.py             # RAG业务逻辑
├── models/
│   ├── database/                  # 数据库模型
│   └── schemas/                   # Pydantic数据模型
├── repositories/
│   ├── base.py                    # 基础仓储
│   ├── agent_repository.py       # 智能体数据访问
│   └── conversation_repository.py # 对话数据访问
├── ai/
│   ├── langgraph/                 # LangGraph集成
│   ├── autogen/                   # AutoGen集成
│   ├── mcp/                       # MCP协议实现
│   └── openai_client.py           # OpenAI API客户端
└── utils/
    ├── cache.py                   # 缓存工具
    ├── validators.py              # 验证器
    └── helpers.py                 # 辅助函数
```

## Database Architecture

### Data Access Layer
```python
from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from abc import ABC, abstractmethod
import uuid

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """基础仓储类，实现通用CRUD操作"""
    
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: uuid.UUID) -> Optional[ModelType]:
        """根据ID获取单个实体"""
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> tuple[List[ModelType], int]:
        """获取多个实体和总数"""
        query = select(self.model)
        count_query = select(func.count(self.model.id))
        
        # 应用过滤器
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.where(getattr(self.model, field) == value)
                    count_query = count_query.where(getattr(self.model, field) == value)
        
        # 应用排序
        if order_by and hasattr(self.model, order_by):
            query = query.order_by(getattr(self.model, order_by).desc())
        
        # 应用分页
        query = query.offset(skip).limit(limit)
        
        # 执行查询
        result = await self.db.execute(query)
        count_result = await self.db.execute(count_query)
        
        items = result.scalars().all()
        total = count_result.scalar()
        
        return items, total

    async def create(self, *, obj_in: CreateSchemaType, **kwargs) -> ModelType:
        """创建新实体"""
        obj_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
        obj_data.update(kwargs)
        db_obj = self.model(**obj_data)
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(
        self, 
        *, 
        db_obj: ModelType, 
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """更新实体"""
        obj_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
        
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def remove(self, *, id: uuid.UUID) -> bool:
        """删除实体"""
        query = delete(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        await self.db.commit()
        return result.rowcount > 0
```

## Authentication and Authorization

### Middleware/Guards
```python
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
import redis.asyncio as redis

from ..core.config import settings
from ..core.security import verify_password, create_access_token
from ..models.database.user import User
from ..repositories.user_repository import UserRepository

security = HTTPBearer()

class AuthService:
    """认证服务"""
    
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.user_repo = UserRepository(db_session)

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """验证用户凭据"""
        user = await self.user_repo.get_by_username(username)
        if not user or not user.is_active:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        return user

    async def create_user_session(self, user: User) -> dict:
        """创建用户会话"""
        # 生成访问令牌
        access_token = create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # 生成刷新令牌
        refresh_token = create_access_token(
            data={"sub": str(user.id), "type": "refresh"},
            expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # 存储会话到Redis
        session_key = f"session:{user.id}"
        session_data = {
            "user_id": str(user.id),
            "username": user.username,
            "is_active": user.is_active,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            session_key,
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            json.dumps(session_data)
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }

    async def get_current_user(self, token: str) -> Optional[User]:
        """从令牌获取当前用户"""
        try:
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
                
        except JWTError:
            return None
        
        # 检查会话状态
        session_key = f"session:{user_id}"
        session_data = await self.redis.get(session_key)
        if not session_data:
            return None
        
        # 获取用户信息
        user = await self.user_repo.get(uuid.UUID(user_id))
        if not user or not user.is_active:
            return None
        
        return user
```
