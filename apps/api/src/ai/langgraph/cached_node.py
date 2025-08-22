"""
缓存节点装饰器和包装器
提供透明的节点级缓存功能
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Optional, Dict, Union, Awaitable

from .context import AgentContext
from .cache_factory import get_node_cache
from .caching import NodeCache

logger = logging.getLogger(__name__)


class CachedNodeWrapper:
    """缓存节点包装器"""
    
    def __init__(
        self,
        node_func: Callable,
        node_name: Optional[str] = None,
        cache: Optional[NodeCache] = None,
        ttl: Optional[int] = None,
        cache_key_func: Optional[Callable] = None,
        enabled: bool = True
    ):
        self.node_func = node_func
        self.node_name = node_name or node_func.__name__
        self.cache = cache or get_node_cache()
        self.ttl = ttl
        self.cache_key_func = cache_key_func
        self.enabled = enabled
        
        # 保持原函数的元数据
        self.__name__ = node_func.__name__
        self.__doc__ = node_func.__doc__
        self.__module__ = node_func.__module__
        self.__qualname__ = getattr(node_func, '__qualname__', node_func.__name__)
        
        # 检查是否是异步函数
        self.is_async = inspect.iscoroutinefunction(node_func)
    
    def _extract_context_and_inputs(self, *args, **kwargs) -> tuple[Optional[AgentContext], Dict[str, Any]]:
        """从函数参数中提取上下文和输入"""
        context = None
        inputs = {}
        
        # 检查参数中的AgentContext
        for arg in args:
            if isinstance(arg, AgentContext):
                context = arg
                break
        
        # 检查关键字参数中的context
        if context is None and 'context' in kwargs:
            potential_context = kwargs.get('context')
            if isinstance(potential_context, AgentContext):
                context = potential_context
        
        # 构建输入参数字典（排除context）
        inputs = kwargs.copy()
        inputs.pop('context', None)
        
        # 添加位置参数（排除context）
        non_context_args = [arg for arg in args if not isinstance(arg, AgentContext)]
        if non_context_args:
            inputs['_args'] = non_context_args
        
        return context, inputs
    
    def _generate_cache_key(self, context: Optional[AgentContext], inputs: Dict[str, Any]) -> Optional[str]:
        """生成缓存键"""
        if not context:
            # 没有上下文时，使用简化的键生成策略
            logger.warning(f"节点 {self.node_name} 没有上下文信息，使用简化缓存键")
            return None
        
        try:
            if self.cache_key_func:
                # 使用自定义缓存键生成函数
                return self.cache_key_func(self.node_name, context, inputs)
            else:
                # 使用默认缓存键生成策略
                return self.cache.generate_cache_key(self.node_name, context, inputs)
        except Exception as e:
            logger.error(f"缓存键生成失败 {self.node_name}: {e}")
            return None
    
    async def __call__(self, *args, **kwargs) -> Any:
        """执行缓存节点"""
        # 如果缓存未启用，直接执行原函数
        if not self.enabled or not self.cache.config.enabled:
            if self.is_async:
                return await self.node_func(*args, **kwargs)
            else:
                return self.node_func(*args, **kwargs)
        
        # 提取上下文和输入
        context, inputs = self._extract_context_and_inputs(*args, **kwargs)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(context, inputs)
        
        if not cache_key:
            logger.debug(f"节点 {self.node_name} 无法生成缓存键，直接执行")
            if self.is_async:
                return await self.node_func(*args, **kwargs)
            else:
                return self.node_func(*args, **kwargs)
        
        try:
            # 尝试从缓存获取结果
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"节点 {self.node_name} 缓存命中: {cache_key}")
                return cached_result
            
            logger.debug(f"节点 {self.node_name} 缓存未命中，执行计算: {cache_key}")
            
            # 执行原函数
            if self.is_async:
                result = await self.node_func(*args, **kwargs)
            else:
                result = self.node_func(*args, **kwargs)
            
            # 将结果存储到缓存
            cache_success = await self.cache.set(cache_key, result, ttl=self.ttl)
            if cache_success:
                logger.debug(f"节点 {self.node_name} 结果已缓存: {cache_key}")
            else:
                logger.warning(f"节点 {self.node_name} 缓存存储失败: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"节点 {self.node_name} 缓存操作失败: {e}")
            # 缓存失败时降级到直接执行
            if self.is_async:
                return await self.node_func(*args, **kwargs)
            else:
                return self.node_func(*args, **kwargs)


def cached_node(
    name: Optional[str] = None,
    ttl: Optional[int] = None,
    cache: Optional[NodeCache] = None,
    cache_key_func: Optional[Callable] = None,
    enabled: bool = True
):
    """
    缓存节点装饰器
    
    Args:
        name: 节点名称，默认使用函数名
        ttl: 缓存TTL（秒），None使用默认值
        cache: 缓存实例，None使用默认缓存
        cache_key_func: 自定义缓存键生成函数
        enabled: 是否启用缓存
    
    Example:
        @cached_node(name="analysis_node", ttl=1800)
        async def analyze_data(state, context: AgentContext):
            # 数据分析逻辑
            return {"result": "analysis_complete"}
    """
    def decorator(func: Callable) -> CachedNodeWrapper:
        return CachedNodeWrapper(
            node_func=func,
            node_name=name,
            cache=cache,
            ttl=ttl,
            cache_key_func=cache_key_func,
            enabled=enabled
        )
    
    return decorator


def invalidate_node_cache(
    node_name: str,
    context: Optional[AgentContext] = None,
    inputs: Optional[Dict[str, Any]] = None,
    pattern: Optional[str] = None,
    cache: Optional[NodeCache] = None
) -> Awaitable[bool]:
    """
    使节点缓存失效
    
    Args:
        node_name: 节点名称
        context: 上下文信息
        inputs: 输入参数
        pattern: 匹配模式，优先级高于具体参数
        cache: 缓存实例
    
    Returns:
        是否成功失效缓存
    """
    async def _invalidate():
        cache_instance = cache or get_node_cache()
        
        try:
            if pattern:
                # 使用模式清理
                count = await cache_instance.clear(pattern)
                logger.info(f"按模式清理节点缓存: pattern={pattern}, count={count}")
                return count > 0
            elif context and inputs is not None:
                # 清理特定缓存键
                cache_key = cache_instance.generate_cache_key(node_name, context, inputs)
                success = await cache_instance.delete(cache_key)
                logger.info(f"清理节点缓存: node={node_name}, key={cache_key}, success={success}")
                return success
            else:
                # 清理所有相关缓存
                node_pattern = f"*:{node_name}:*"
                count = await cache_instance.clear(node_pattern)
                logger.info(f"清理节点所有缓存: node={node_name}, count={count}")
                return count > 0
        except Exception as e:
            logger.error(f"节点缓存失效失败: {e}")
            return False
    
    return _invalidate()


def cache_warmup(
    node_func: Callable,
    contexts: list[AgentContext],
    inputs_list: list[Dict[str, Any]],
    node_name: Optional[str] = None,
    cache: Optional[NodeCache] = None
) -> Awaitable[Dict[str, Any]]:
    """
    缓存预热
    
    Args:
        node_func: 节点函数
        contexts: 上下文列表
        inputs_list: 输入参数列表
        node_name: 节点名称
        cache: 缓存实例
    
    Returns:
        预热统计信息
    """
    async def _warmup():
        cache_instance = cache or get_node_cache()
        node_name_resolved = node_name or node_func.__name__
        
        success_count = 0
        error_count = 0
        
        for context, inputs in zip(contexts, inputs_list):
            try:
                # 生成缓存键
                cache_key = cache_instance.generate_cache_key(node_name_resolved, context, inputs)
                
                # 检查是否已存在缓存
                if await cache_instance.exists(cache_key):
                    logger.debug(f"缓存预热跳过（已存在）: {cache_key}")
                    continue
                
                # 执行函数并缓存结果
                if inspect.iscoroutinefunction(node_func):
                    result = await node_func(context=context, **inputs)
                else:
                    result = node_func(context=context, **inputs)
                
                # 存储到缓存
                await cache_instance.set(cache_key, result)
                success_count += 1
                logger.debug(f"缓存预热成功: {cache_key}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"缓存预热失败: {e}")
        
        stats = {
            "success_count": success_count,
            "error_count": error_count,
            "total_count": len(contexts),
            "node_name": node_name_resolved
        }
        
        logger.info(f"缓存预热完成: {stats}")
        return stats
    
    return _warmup()