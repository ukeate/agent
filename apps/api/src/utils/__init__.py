"""
工具函数模块
提供认证、验证、缓存等通用工具
"""

from .auth import AuthUtils, require_auth, require_role, session_manager
from .validation import (
    Validator, 
    SchemaValidator, 
    ValidationError,
    validate_request_data,
    PydanticValidator,
    validate_pagination_params,
    validate_search_query
)
from .cache import (
    CacheManager,
    cache_manager,
    cached,
    cache_key_generator,
    RateLimiter,
    rate_limit

)

__all__ = [
    # 认证工具
    "AuthUtils",
    "require_auth", 
    "require_role",
    "session_manager",
    
    # 验证工具
    "Validator",
    "SchemaValidator",
    "ValidationError", 
    "validate_request_data",
    "PydanticValidator",
    "validate_pagination_params",
    "validate_search_query",
    
    # 缓存工具
    "CacheManager",
    "cache_manager",
    "cached",
    "cache_key_generator", 
    "RateLimiter",
    "rate_limit"
]
