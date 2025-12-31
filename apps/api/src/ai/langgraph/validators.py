"""
类型验证器模块
提供运行时类型检查和验证功能
"""

from typing import Type, TypeVar, Any, Dict, List, get_origin, get_args, Union, Optional
from pydantic import BaseModel, ValidationError
import inspect
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import json
from .context import AgentContext, ContextVersion

T = TypeVar('T')

class TypeValidator:
    """类型验证器"""
    
    @staticmethod
    def validate_context_type(
        context: Any, 
        expected_type: Type[AgentContext]
    ) -> bool:
        """验证上下文类型是否匹配预期"""
        if not isinstance(context, AgentContext):
            return False
        
        # 检查泛型类型参数
        if hasattr(expected_type, '__args__'):
            expected_generic = get_args(expected_type)[0] if get_args(expected_type) else None
            if expected_generic and context.custom_data is not None:
                if not TypeValidator._check_type_compatibility(
                    context.custom_data, 
                    expected_generic
                ):
                    return False
        
        return True
    
    @staticmethod
    def _check_type_compatibility(value: Any, expected_type: Type) -> bool:
        """检查类型兼容性"""
        # 处理Optional类型
        if get_origin(expected_type) is Union:
            args = get_args(expected_type)
            # 检查是否为Optional（Union[T, None]）
            if type(None) in args:
                if value is None:
                    return True
                # 移除None，检查其他类型
                non_none_types = [t for t in args if t != type(None)]
                return any(
                    TypeValidator._check_type_compatibility(value, t) 
                    for t in non_none_types
                )
            else:
                # 普通Union类型
                return any(
                    TypeValidator._check_type_compatibility(value, t) 
                    for t in args
                )
        
        # 处理List类型
        if get_origin(expected_type) is list:
            if not isinstance(value, list):
                return False
            element_type = get_args(expected_type)[0] if get_args(expected_type) else Any
            return all(
                TypeValidator._check_type_compatibility(item, element_type) 
                for item in value
            )
        
        # 处理Dict类型
        if get_origin(expected_type) is dict:
            if not isinstance(value, dict):
                return False
            if get_args(expected_type):
                key_type, value_type = get_args(expected_type)
                return all(
                    TypeValidator._check_type_compatibility(k, key_type) and
                    TypeValidator._check_type_compatibility(v, value_type)
                    for k, v in value.items()
                )
            return True
        
        # 处理BaseModel子类
        if inspect.isclass(expected_type) and issubclass(expected_type, BaseModel):
            return isinstance(value, expected_type)
        
        # 基础类型检查
        return isinstance(value, expected_type)
    
    @staticmethod
    def get_type_schema(context_type: Type) -> Dict[str, Any]:
        """获取类型模式定义"""
        if inspect.isclass(context_type) and issubclass(context_type, BaseModel):
            return context_type.model_json_schema()
        
        # 对于泛型类型，获取基础类型的schema
        origin = get_origin(context_type)
        if origin and inspect.isclass(origin) and issubclass(origin, BaseModel):
            schema = origin.model_json_schema()
            # 添加泛型信息
            if get_args(context_type):
                schema['generic_args'] = [str(arg) for arg in get_args(context_type)]
            return schema
        
        # 基础类型
        return {
            "type": context_type.__name__ if hasattr(context_type, '__name__') else str(context_type),
            "module": context_type.__module__ if hasattr(context_type, '__module__') else None
        }
    
    @staticmethod
    def diagnose_type_error(context: Any, expected_type: Type) -> str:
        """诊断类型错误并提供详细信息"""
        errors = []
        
        # 检查基础类型
        if not isinstance(context, AgentContext):
            errors.append(f"期望 AgentContext 类型，但收到 {type(context).__name__}")
            return "\n".join(errors)
        
        # 验证必填字段
        try:
            context.model_dump()
        except ValidationError as e:
            errors.append("Pydantic验证错误:")
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error['loc'])
                errors.append(f"  - {field}: {error['msg']}")
        
        # 检查泛型类型
        if hasattr(expected_type, '__args__'):
            expected_generic = get_args(expected_type)[0] if get_args(expected_type) else None
            if expected_generic and context.custom_data is not None:
                if not TypeValidator._check_type_compatibility(
                    context.custom_data, 
                    expected_generic
                ):
                    errors.append(
                        f"custom_data 类型不匹配: "
                        f"期望 {expected_generic}, "
                        f"实际 {type(context.custom_data)}"
                    )
        
        # 检查版本兼容性
        if hasattr(context, 'version'):
            if context.version != ContextVersion.CURRENT:
                errors.append(
                    f"版本不匹配: 当前版本 {ContextVersion.CURRENT.value}, "
                    f"上下文版本 {context.version.value}"
                )
        
        return "\n".join(errors) if errors else "没有检测到类型错误"
    
    @staticmethod
    def validate_field_constraints(context: AgentContext) -> List[str]:
        """验证字段约束"""
        warnings = []
        
        # 检查迭代次数
        if context.step_count > context.max_iterations * 0.8:
            warnings.append(
                f"接近最大迭代次数限制: {context.step_count}/{context.max_iterations}"
            )
        
        # 检查超时
        if context.is_timeout():
            warnings.append("上下文已超时")
        
        # 检查性能标签
        if len(context.performance_tags) > 100:
            warnings.append(f"性能标签过多: {len(context.performance_tags)}")
        
        # 检查元数据大小
        metadata_size = len(json.dumps(context.metadata))
        if metadata_size > 10000:  # 10KB限制
            warnings.append(f"元数据过大: {metadata_size} bytes")
        
        return warnings

class ContextTypeGuard:
    """上下文类型守卫"""
    
    @staticmethod
    def is_valid_context(obj: Any) -> bool:
        """检查对象是否为有效的上下文"""
        return isinstance(obj, AgentContext) and validate_context(obj)
    
    @staticmethod
    def assert_context_type(
        obj: Any, 
        expected_type: Type[AgentContext] = AgentContext
    ) -> None:
        """断言对象为特定的上下文类型"""
        if not TypeValidator.validate_context_type(obj, expected_type):
            error_msg = TypeValidator.diagnose_type_error(obj, expected_type)
            raise TypeError(f"上下文类型验证失败:\n{error_msg}")
    
    @staticmethod
    def narrow_type(
        context: AgentContext, 
        custom_data_type: Type[T]
    ) -> Optional[AgentContext[T]]:
        """类型缩窄 - 验证并返回具有特定custom_data类型的上下文"""
        if context.custom_data is None:
            return None
        
        if TypeValidator._check_type_compatibility(
            context.custom_data, 
            custom_data_type
        ):
            # 类型检查通过，可以安全地进行类型断言
            return context  # type: ignore
        
        return None

def validate_context(context: AgentContext) -> bool:
    """验证上下文是否有效"""
    try:
        # 使用Pydantic的内置验证
        context.model_dump()
        
        # 额外的业务逻辑验证
        if context.step_count < 0:
            return False
        
        if context.timeout_seconds <= 0:
            return False
        
        if context.max_iterations <= 0:
            return False
        
        return True
    except (ValidationError, Exception):
        return False
