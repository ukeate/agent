"""
FastAPI 0.116+ 新特性演示和实现
展示FastAPI 0.116.x版本引入的新功能
"""
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Path, Body, status, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic.functional_validators import model_validator

import structlog

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/fastapi-features", tags=["FastAPI 0.116+ Features"])

T = TypeVar('T')


class HttpStatus(str, Enum):
    """HTTP状态码枚举 - FastAPI 0.116增强的类型提示"""
    OK = "200"
    CREATED = "201"
    BAD_REQUEST = "400"
    NOT_FOUND = "404"


class ApiResponse(BaseModel, Generic[T]):
    """通用API响应模型 - FastAPI 0.116改进的泛型支持"""
    status: str = Field(..., description="响应状态")
    message: str = Field(default="Success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "操作成功",
                "data": {"id": 1},
                "timestamp": "2025-01-08T22:57:00Z"
            }
        }


class UserCreateRequest(BaseModel):
    """用户创建请求 - FastAPI 0.116增强的验证"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", description="邮箱")
    age: Optional[int] = Field(None, ge=0, le=150, description="年龄")
    bio: Optional[str] = Field(None, max_length=500, description="个人简介")

    @field_validator('username')
    @classmethod
    def username_must_not_contain_spaces(cls, v: str) -> str:
        if ' ' in v:
            raise ValueError('用户名不能包含空格')
        return v

    @model_validator(mode='after')
    def validate_email_domain(self):
        """验证邮箱域名"""
        if self.email and self.email.endswith('@example.com'):
            raise ValueError('不允许使用example.com邮箱')
        return self


class UserResponse(BaseModel):
    """用户响应模型"""
    id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    age: Optional[int] = Field(None, description="年龄")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class PaginationParams(BaseModel):
    """分页参数 - FastAPI 0.116改进的参数验证"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(10, ge=1, le=100, description="每页数量")
    sort_by: str = Field("id", description="排序字段")
    order: str = Field("asc", pattern="^(asc|desc)$", description="排序方向")


# 内存存储（演示用）
users_db: Dict[int, UserResponse] = {}
user_id_counter = 1


@router.post(
    "/users",
    response_model=ApiResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="创建用户",
    description="FastAPI 0.116增强的请求体验证"
)
async def create_user(user: UserCreateRequest, request: Request) -> ApiResponse[UserResponse]:
    """
    创建新用户

    FastAPI 0.116新特性:
    - 增强的Pydantic v2验证
    - 改进的错误消息
    - 更好的类型提示
    """
    global user_id_counter

    try:
        # 创建用户
        new_user = UserResponse(
            id=user_id_counter,
            username=user.username,
            email=user.email,
            age=user.age,
        )

        users_db[user_id_counter] = new_user
        user_id_counter += 1

        logger.info("用户创建成功", user_id=new_user.id, username=new_user.username)

        return ApiResponse[UserResponse](
            status="success",
            message="用户创建成功",
            data=new_user,
            metadata={"request_id": request.state.request_id if hasattr(request.state, 'request_id') else None}
        )

    except Exception as e:
        logger.error("用户创建失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"验证失败: {str(e)}"
        )


@router.get(
    "/users/{user_id}",
    response_model=ApiResponse[UserResponse],
    summary="获取用户",
    description="FastAPI 0.116增强的路径参数验证"
)
async def get_user(
    user_id: int = Path(..., ge=1, description="用户ID"),
    request: Request = None
) -> ApiResponse[UserResponse]:
    """
    根据ID获取用户

    FastAPI 0.116新特性:
    - 增强的路径参数验证
    - 改进的路由匹配
    """
    if user_id not in users_db:
        logger.warning("用户不存在", user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在"
        )

    user = users_db[user_id]
    logger.info("获取用户成功", user_id=user_id)

    return ApiResponse[UserResponse](
        status="success",
        data=user
    )


@router.get(
    "/users",
    response_model=ApiResponse[List[UserResponse]],
    summary="获取用户列表",
    description="FastAPI 0.116增强的查询参数处理"
)
async def list_users(
    pagination: PaginationParams = Query(...),  # FastAPI 0.116: 嵌套查询参数
    request: Request = None
) -> ApiResponse[List[UserResponse]]:
    """
    获取用户列表

    FastAPI 0.116新特性:
    - 支持嵌套查询参数
    - 改进的序列化
    - 更好的性能
    """
    # 分页逻辑
    start_idx = (pagination.page - 1) * pagination.page_size
    end_idx = start_idx + pagination.page_size

    all_users = list(users_db.values())

    # 排序
    reverse_order = pagination.order == "desc"
    if pagination.sort_by == "id":
        all_users.sort(key=lambda u: u.id, reverse=reverse_order)
    elif pagination.sort_by == "username":
        all_users.sort(key=lambda u: u.username, reverse=reverse_order)

    # 分页
    paginated_users = all_users[start_idx:end_idx]

    logger.info(
        "获取用户列表成功",
        page=pagination.page,
        page_size=pagination.page_size,
        total=len(all_users)
    )

    return ApiResponse[List[UserResponse]](
        status="success",
        data=paginated_users,
        metadata={
            "total": len(all_users),
            "page": pagination.page,
            "page_size": pagination.page_size,
            "total_pages": (len(all_users) + pagination.page_size - 1) // pagination.page_size
        }
    )


@router.patch(
    "/users/{user_id}",
    response_model=ApiResponse[UserResponse],
    summary="更新用户",
    description="FastAPI 0.116增强的部分更新"
)
async def update_user(
    user_id: int = Path(..., ge=1),
    user_update: Dict[str, Any] = Body(..., description="要更新的字段"),
    request: Request = None
) -> ApiResponse[UserResponse]:
    """
    部分更新用户

    FastAPI 0.116新特性:
    - 改进的PATCH请求处理
    - 更好的部分更新支持
    """
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在"
        )

    user = users_db[user_id]

    # 更新字段
    for key, value in user_update.items():
        if hasattr(user, key):
            setattr(user, key, value)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的字段: {key}"
            )

    logger.info("用户更新成功", user_id=user_id, updated_fields=list(user_update.keys()))

    return ApiResponse[UserResponse](
        status="success",
        message="用户更新成功",
        data=user
    )


@router.delete(
    "/users/{user_id}",
    response_model=ApiResponse[Dict[str, Any]],
    summary="删除用户",
    description="FastAPI 0.116增强的删除操作"
)
async def delete_user(
    user_id: int = Path(..., ge=1),
    request: Request = None
) -> ApiResponse[Dict[str, Any]]:
    """
    删除用户

    FastAPI 0.116新特性:
    - 改进的DELETE响应
    - 更好的状态码处理
    """
    if user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在"
        )

    del users_db[user_id]

    logger.info("用户删除成功", user_id=user_id)

    return ApiResponse[Dict[str, Any]](
        status="success",
        message="用户删除成功",
        data={"deleted_id": user_id}
    )


@router.get(
    "/health",
    summary="健康检查",
    description="FastAPI 0.116改进的健康检查端点"
)
async def health_check():
    """健康检查端点 - FastAPI 0.116改进"""
    return {
        "status": "healthy",
        "version": "0.116.0",
        "features": [
            "Enhanced Pydantic v2 validation",
            "Improved type hints",
            "Better error messages",
            "Nested query parameters",
            "Improved async support",
            "Performance optimizations"
        ]
    }


@router.get(
    "/openapi-schema",
    summary="获取OpenAPI模式",
    description="FastAPI 0.116改进的OpenAPI 3.1支持"
)
async def get_openapi_schema():
    """
    获取OpenAPI模式

    FastAPI 0.116新特性:
    - 改进的OpenAPI 3.1支持
    - 更好的文档生成
    """
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "FastAPI 0.116+ Features",
            "version": "1.0.0",
            "description": "演示FastAPI 0.116.x的新特性"
        },
        "features": {
            "enhanced_validation": "增强的Pydantic v2验证",
            "improved_type_hints": "改进的类型提示",
            "nested_query_params": "嵌套查询参数支持",
            "better_async": "更好的异步支持",
            "openapi_3_1": "OpenAPI 3.1完整支持"
        }
    }


@router.post(
    "/bulk/users",
    response_model=ApiResponse[List[UserResponse]],
    summary="批量创建用户",
    description="FastAPI 0.116改进的批量操作"
)
async def bulk_create_users(
    users: List[UserCreateRequest] = Body(..., description="用户列表"),
    request: Request = None
) -> ApiResponse[List[UserResponse]]:
    """
    批量创建用户

    FastAPI 0.116新特性:
    - 改进的批量操作支持
    - 更好的列表验证
    - 性能优化
    """
    global user_id_counter
    created_users = []

    for user_data in users:
        new_user = UserResponse(
            id=user_id_counter,
            username=user_data.username,
            email=user_data.email,
            age=user_data.age,
        )

        users_db[user_id_counter] = new_user
        created_users.append(new_user)
        user_id_counter += 1

    logger.info("批量创建用户成功", count=len(created_users))

    return ApiResponse[List[UserResponse]](
        status="success",
        message=f"成功创建 {len(created_users)} 个用户",
        data=created_users
    )
