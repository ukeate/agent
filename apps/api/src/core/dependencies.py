"""
FastAPI依赖注入
"""

from typing import Optional
from fastapi import HTTPException, Header, status

async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> str:
    """获取当前用户，简化版本用于开发"""
    # 在实际应用中，这里应该验证JWT token或其他认证方式
    # 现在简化为返回固定用户ID用于开发测试
    if authorization and authorization.startswith("Bearer "):
        # 简单的用户ID提取，实际应用中需要解析JWT
        return "test_user_123"
    
    # 开发模式下允许无认证访问
    return "anonymous_user"

async def get_api_key(
    x_api_key: Optional[str] = Header(None)
) -> str:
    """获取API密钥"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API密钥缺失"
        )
    
    # 这里应该验证API密钥
    # 简化版本，直接返回
    return x_api_key