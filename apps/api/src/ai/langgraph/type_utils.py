"""
类型工具模块
提供类型安全的序列化、反序列化和类型操作工具
"""
from typing import Type, TypeVar, Any, Dict, List, Optional, get_origin, get_args
from pydantic import BaseModel, ValidationError
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import uuid
import importlib

from .context import AgentContext, ContextVersion
from .validators import TypeValidator

T = TypeVar('T')


class TypeSafeSerializer:
    """类型安全序列化器"""
    
    @classmethod
    def serialize_context(cls, context: AgentContext[T]) -> Dict[str, Any]:
        """类型安全序列化上下文"""
        # 获取基础数据
        data = context.model_dump()
        
        # 添加类型信息
        type_info = {
            "class": context.__class__.__name__,
            "module": context.__class__.__module__,
            "version": context.version.value if hasattr(context, 'version') else "1.0",
            "timestamp": utc_now().isoformat()
        }
        
        # 提取泛型类型信息
        if context.custom_data is not None:
            type_info["generic_type"] = cls._extract_type_info(context.custom_data)
        
        return {
            "data": data,
            "type_info": type_info,
            "checksum": cls._calculate_checksum(data)
        }
    
    @classmethod
    def deserialize_context(
        cls, 
        serialized: Dict[str, Any], 
        target_type: Type[AgentContext[T]] = AgentContext
    ) -> AgentContext[T]:
        """类型安全反序列化上下文"""
        # 验证数据完整性
        if "data" not in serialized or "type_info" not in serialized:
            raise ValueError("序列化数据格式无效")
        
        # 验证校验和
        if "checksum" in serialized:
            expected_checksum = cls._calculate_checksum(serialized["data"])
            if serialized["checksum"] != expected_checksum:
                raise ValueError("数据校验和不匹配，数据可能已损坏")
        
        # 提取数据
        data = serialized["data"]
        type_info = serialized["type_info"]
        
        # 版本兼容性检查
        if "version" in type_info:
            data_version = type_info["version"]
            if data_version != ContextVersion.CURRENT.value:
                # 需要版本迁移
                from .versioning import ContextMigrator
                data = ContextMigrator.migrate_context(
                    data, 
                    data_version, 
                    ContextVersion.CURRENT.value
                )
        
        # 创建上下文实例
        try:
            context = target_type(**data)
            
            # 重构custom_data如果它是Pydantic模型
            if context.custom_data and isinstance(context.custom_data, dict):
                if "generic_type" in type_info and type_info["generic_type"].get("is_pydantic_model"):
                    # 尝试重构Pydantic模型
                    type_name = type_info["generic_type"].get("type")
                    module_name = type_info["generic_type"].get("module")
                    if type_name and module_name:
                        try:
                            import importlib
                            module = importlib.import_module(module_name)
                            model_class = getattr(module, type_name)
                            if issubclass(model_class, BaseModel):
                                context.custom_data = model_class(**context.custom_data)
                        except (ImportError, AttributeError, TypeError):
                            # 如果重构失败，保持字典形式
                            pass
            
            # 验证类型匹配
            if not TypeValidator.validate_context_type(context, target_type):
                raise TypeError("反序列化的上下文类型不匹配")
            
            return context
        except ValidationError as e:
            raise ValueError(f"反序列化验证失败: {e}")
    
    @staticmethod
    def _extract_type_info(obj: Any) -> Dict[str, Any]:
        """提取对象的类型信息"""
        if obj is None:
            return {"type": "None"}
        
        type_info = {
            "type": type(obj).__name__,
            "module": type(obj).__module__ if hasattr(type(obj), '__module__') else None
        }
        
        # 对于Pydantic模型，保存schema和数据
        if isinstance(obj, BaseModel):
            type_info["schema"] = obj.model_json_schema()
            type_info["data"] = obj.model_dump()
            # 保存类信息以便重构
            type_info["is_pydantic_model"] = True
        # 对于基础类型，直接保存值
        elif isinstance(obj, (str, int, float, bool)):
            type_info["value"] = obj
        # 对于集合类型，递归提取
        elif isinstance(obj, list):
            type_info["items"] = [
                TypeSafeSerializer._extract_type_info(item) 
                for item in obj[:10]  # 只保存前10个元素的类型信息
            ]
        elif isinstance(obj, dict):
            type_info["sample"] = {
                k: TypeSafeSerializer._extract_type_info(v)
                for k, v in list(obj.items())[:5]  # 只保存前5个键值对的类型信息
            }
        
        return type_info
    
    @staticmethod
    def _calculate_checksum(data: Dict[str, Any]) -> str:
        """计算数据校验和"""
        import hashlib
        # 序列化为JSON字符串（排序键以保证一致性）
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class TypeSafeCachedNode:
    """类型安全的缓存节点"""
    
    def __init__(
        self, 
        node_func, 
        context_type: Type[AgentContext[T]],
        cache_client=None
    ):
        self.node_func = node_func
        self.context_type = context_type
        self.validator = TypeValidator()
        self.cache_client = cache_client
    
    async def __call__(self, state: Any, context: AgentContext[T]):
        """执行节点函数，带类型验证和缓存"""
        # 类型验证
        if not self.validator.validate_context_type(context, self.context_type):
            error_msg = self.validator.diagnose_type_error(context, self.context_type)
            raise TypeError(f"上下文类型不匹配:\n{error_msg}")
        
        # 生成类型安全的缓存键
        cache_key = self.generate_typed_cache_key(state, context)
        
        # 尝试从缓存获取
        if self.cache_client and context.cache_namespace:
            cached = await self.get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # 执行函数
        result = await self.node_func(state, context)
        
        # 保存到缓存
        if self.cache_client and context.cache_namespace:
            await self.save_to_cache(cache_key, result)
        
        return result
    
    def generate_typed_cache_key(
        self, 
        state: Any, 
        context: AgentContext[T]
    ) -> str:
        """生成类型安全的缓存键"""
        import hashlib
        
        # 组合关键信息
        key_parts = [
            context.cache_namespace or "default",
            context.user_id,
            context.session_id,
            context.version.value,
            str(self.context_type),
            str(state) if state else "no_state"
        ]
        
        # 添加custom_data的类型信息
        if context.custom_data is not None:
            key_parts.append(str(type(context.custom_data)))
        
        # 生成哈希键
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if not self.cache_client:
            return None
        
        try:
            cached_data = await self.cache_client.get(cache_key)
            if cached_data:
                # 反序列化缓存数据
                return json.loads(cached_data)
        except Exception:
            # 缓存错误不应影响正常执行
            pass
        
        return None
    
    async def save_to_cache(self, cache_key: str, data: Any) -> None:
        """保存数据到缓存"""
        if not self.cache_client:
            return
        
        try:
            # 序列化数据
            serialized = json.dumps(data, default=str)
            # 设置1小时过期
            await self.cache_client.set(cache_key, serialized, expire=3600)
        except Exception:
            # 缓存错误不应影响正常执行
            pass


class TypeRegistry:
    """类型注册表 - 管理自定义类型"""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, type_name: str, type_class: Type) -> None:
        """注册自定义类型"""
        cls._registry[type_name] = type_class
    
    @classmethod
    def get(cls, type_name: str) -> Optional[Type]:
        """获取注册的类型"""
        return cls._registry.get(type_name)
    
    @classmethod
    def resolve_type(cls, type_info: Dict[str, Any]) -> Optional[Type]:
        """从类型信息解析类型"""
        # 尝试从注册表获取
        if "type" in type_info:
            registered = cls.get(type_info["type"])
            if registered:
                return registered
        
        # 尝试动态导入
        if "module" in type_info and "class" in type_info:
            try:
                module = importlib.import_module(type_info["module"])
                return getattr(module, type_info["class"])
            except (ImportError, AttributeError):
                pass
        
        return None
    
    @classmethod
    def list_registered(cls) -> List[str]:
        """列出所有注册的类型"""
        return list(cls._registry.keys())


def create_typed_context(
    user_id: str,
    session_id: str,
    custom_type: Type[T],
    custom_data: Optional[T] = None,
    **kwargs
) -> AgentContext[T]:
    """创建类型化的上下文"""
    from .context import create_context
    
    context = create_context(user_id, session_id, **kwargs)
    
    if custom_data is not None:
        # 验证custom_data类型
        if not TypeValidator._check_type_compatibility(custom_data, custom_type):
            raise TypeError(
                f"custom_data类型不匹配: 期望 {custom_type}, 实际 {type(custom_data)}"
            )
        context.custom_data = custom_data
    
    return context


def cast_context(
    context: AgentContext,
    target_type: Type[T]
) -> AgentContext[T]:
    """类型转换上下文（带运行时检查）"""
    if context.custom_data is not None:
        if not TypeValidator._check_type_compatibility(
            context.custom_data, 
            target_type
        ):
            raise TypeError(
                f"无法转换上下文类型: custom_data类型 {type(context.custom_data)} "
                f"与目标类型 {target_type} 不兼容"
            )
    
    return context  # type: ignore