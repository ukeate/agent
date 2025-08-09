"""
数据验证工具函数
提供通用的数据验证功能
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """验证错误异常"""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


class Validator:
    """数据验证器"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """验证邮箱格式"""
        if not email or not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """验证手机号格式（中国）"""
        if not phone or not isinstance(phone, str):
            return False
        
        pattern = r'^1[3-9]\d{9}$'
        return re.match(pattern, phone) is not None
    
    @staticmethod
    def is_valid_uuid(value: str) -> bool:
        """验证UUID格式"""
        if not value or not isinstance(value, str):
            return False
        
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """验证URL格式"""
        if not url or not isinstance(url, str):
            return False
        
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return re.match(pattern, url) is not None
    
    @staticmethod
    def is_valid_password(password: str, min_length: int = 8) -> tuple[bool, str]:
        """验证密码强度"""
        if not password or not isinstance(password, str):
            return False, "密码不能为空"
        
        if len(password) < min_length:
            return False, f"密码长度至少{min_length}位"
        
        # 检查是否包含数字
        if not re.search(r'\d', password):
            return False, "密码必须包含至少一个数字"
        
        # 检查是否包含字母
        if not re.search(r'[a-zA-Z]', password):
            return False, "密码必须包含至少一个字母"
        
        # 检查是否包含特殊字符
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            return False, "密码必须包含至少一个特殊字符"
        
        return True, "密码强度符合要求"
    
    @staticmethod
    def is_valid_json(value: str) -> bool:
        """验证JSON格式"""
        import json
        try:
            json.loads(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """验证必填字段"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                missing_fields.append(field)
        return missing_fields
    
    @staticmethod
    def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> List[str]:
        """验证字段类型"""
        type_errors = []
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                type_errors.append(f"字段 '{field}' 应该是 {expected_type.__name__} 类型")
        return type_errors
    
    @staticmethod
    def validate_string_length(
        value: str, 
        min_length: Optional[int] = None, 
        max_length: Optional[int] = None
    ) -> tuple[bool, str]:
        """验证字符串长度"""
        if not isinstance(value, str):
            return False, "值必须是字符串类型"
        
        if min_length is not None and len(value) < min_length:
            return False, f"字符串长度不能少于{min_length}位"
        
        if max_length is not None and len(value) > max_length:
            return False, f"字符串长度不能超过{max_length}位"
        
        return True, "字符串长度符合要求"
    
    @staticmethod
    def validate_number_range(
        value: Union[int, float], 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> tuple[bool, str]:
        """验证数字范围"""
        if not isinstance(value, (int, float)):
            return False, "值必须是数字类型"
        
        if min_value is not None and value < min_value:
            return False, f"数值不能小于{min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"数值不能大于{max_value}"
        
        return True, "数值范围符合要求"
    
    @staticmethod
    def validate_enum_value(value: Any, allowed_values: List[Any]) -> tuple[bool, str]:
        """验证枚举值"""
        if value not in allowed_values:
            return False, f"值必须是以下之一: {', '.join(map(str, allowed_values))}"
        
        return True, "枚举值符合要求"


class SchemaValidator:
    """模式验证器"""
    
    @staticmethod
    def validate_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """验证智能体配置"""
        errors = {}
        
        # 验证必填字段
        required_fields = ['name', 'type', 'model']
        missing = Validator.validate_required_fields(config, required_fields)
        if missing:
            errors['missing_fields'] = missing
        
        # 验证字段类型
        field_types = {
            'name': str,
            'type': str,
            'model': str,
            'temperature': (int, float),
            'max_tokens': int
        }
        type_errors = Validator.validate_field_types(config, field_types)
        if type_errors:
            errors['type_errors'] = type_errors
        
        # 验证特定字段
        if 'name' in config:
            valid, msg = Validator.validate_string_length(config['name'], 1, 100)
            if not valid:
                errors['name'] = msg
        
        if 'type' in config:
            valid, msg = Validator.validate_enum_value(
                config['type'], 
                ['single', 'multi', 'supervisor', 'worker']
            )
            if not valid:
                errors['type'] = msg
        
        if 'temperature' in config:
            valid, msg = Validator.validate_number_range(config['temperature'], 0.0, 2.0)
            if not valid:
                errors['temperature'] = msg
        
        return errors
    
    @staticmethod
    def validate_session_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """验证会话数据"""
        errors = {}
        
        # 验证必填字段
        required_fields = ['agents']
        missing = Validator.validate_required_fields(data, required_fields)
        if missing:
            errors['missing_fields'] = missing
        
        # 验证agents字段
        if 'agents' in data:
            if not isinstance(data['agents'], list):
                errors['agents'] = "agents字段必须是数组"
            elif len(data['agents']) == 0:
                errors['agents'] = "至少需要一个智能体"
            else:
                for i, agent_id in enumerate(data['agents']):
                    if not isinstance(agent_id, str) or not agent_id.strip():
                        errors[f'agents[{i}]'] = "智能体ID必须是非空字符串"
        
        return errors
    
    @staticmethod
    def validate_task_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """验证任务数据"""
        errors = {}
        
        # 验证必填字段
        required_fields = ['title', 'agent_id']
        missing = Validator.validate_required_fields(data, required_fields)
        if missing:
            errors['missing_fields'] = missing
        
        # 验证字段长度
        if 'title' in data:
            valid, msg = Validator.validate_string_length(data['title'], 1, 200)
            if not valid:
                errors['title'] = msg
        
        if 'description' in data:
            valid, msg = Validator.validate_string_length(data['description'], 0, 1000)
            if not valid:
                errors['description'] = msg
        
        # 验证优先级
        if 'priority' in data:
            valid, msg = Validator.validate_enum_value(
                data['priority'],
                ['low', 'medium', 'high', 'urgent']
            )
            if not valid:
                errors['priority'] = msg
        
        # 验证状态
        if 'status' in data:
            valid, msg = Validator.validate_enum_value(
                data['status'],
                ['pending', 'in_progress', 'completed', 'failed']
            )
            if not valid:
                errors['status'] = msg
        
        return errors


def validate_request_data(schema_func: Callable) -> Callable:
    """请求数据验证装饰器"""
    def decorator(f):
        from functools import wraps
        
        @wraps(f)
        async def decorated_function(*args, **kwargs):
            # 查找请求数据
            request_data = None
            for arg in args:
                if isinstance(arg, dict):
                    request_data = arg
                    break
            
            if 'data' in kwargs:
                request_data = kwargs['data']
            
            if request_data:
                # 执行验证
                errors = schema_func(request_data)
                if errors:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=422, 
                        detail={
                            "message": "数据验证失败",
                            "errors": errors
                        }
                    )
            
            return await f(*args, **kwargs)
        
        return decorated_function
    return decorator


class PydanticValidator:
    """Pydantic模型验证器"""
    
    @staticmethod
    def validate_model(data: Dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
        """使用Pydantic模型验证数据"""
        try:
            return model_class(**data)
        except ValidationError as e:
            logger.error(f"Pydantic验证失败: {e}")
            raise ValidationError(f"数据验证失败: {e}")
    
    @staticmethod
    def validate_and_convert(data: Dict[str, Any], model_class: type[BaseModel]) -> Dict[str, Any]:
        """验证并转换为字典"""
        validated_model = PydanticValidator.validate_model(data, model_class)
        return validated_model.model_dump()


# 常用验证函数
def validate_pagination_params(page: int = 1, page_size: int = 20) -> tuple[int, int]:
    """验证分页参数"""
    page = max(1, page)
    page_size = max(1, min(100, page_size))  # 限制最大页面大小
    return page, page_size


def validate_search_query(query: Optional[str]) -> Optional[str]:
    """验证搜索查询"""
    if not query:
        return None
    
    query = query.strip()
    if len(query) < 2:
        raise ValidationError("搜索关键词至少需要2个字符")
    
    if len(query) > 100:
        raise ValidationError("搜索关键词不能超过100个字符")
    
    return query