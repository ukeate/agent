"""
事件路由和过滤系统
实现高级事件路由、过滤和转换功能
"""

import re
import asyncio
from typing import Pattern, Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from enum import Enum
from .events import Event, EventType, EventPriority
from .event_processors import EventProcessor, EventContext, ProcessingResult

from src.core.logging import get_logger
logger = get_logger(__name__)

class FilterOperator(str, Enum):
    """过滤操作符"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"

@dataclass
class FilterCondition:
    """过滤条件"""
    field: str  # 字段路径，支持嵌套如 "data.user.id"
    operator: FilterOperator
    value: Any
    case_sensitive: bool = True
    
    def evaluate(self, event: Event) -> bool:
        """评估条件是否满足"""
        try:
            # 获取字段值
            field_value = self._get_field_value(event, self.field)
            
            # 根据操作符评估
            if self.operator == FilterOperator.EXISTS:
                return field_value is not None
            elif self.operator == FilterOperator.NOT_EXISTS:
                return field_value is None
            elif field_value is None:
                return False  # 字段不存在时，除了EXISTS/NOT_EXISTS外都返回False
            
            # 字符串操作的大小写处理
            if isinstance(field_value, str) and not self.case_sensitive:
                field_value = field_value.lower()
                if isinstance(self.value, str):
                    compare_value = self.value.lower()
                else:
                    compare_value = self.value
            else:
                compare_value = self.value
            
            # 执行比较
            if self.operator == FilterOperator.EQUALS:
                return field_value == compare_value
            elif self.operator == FilterOperator.NOT_EQUALS:
                return field_value != compare_value
            elif self.operator == FilterOperator.GREATER_THAN:
                return field_value > compare_value
            elif self.operator == FilterOperator.GREATER_THAN_OR_EQUAL:
                return field_value >= compare_value
            elif self.operator == FilterOperator.LESS_THAN:
                return field_value < compare_value
            elif self.operator == FilterOperator.LESS_THAN_OR_EQUAL:
                return field_value <= compare_value
            elif self.operator == FilterOperator.IN:
                return field_value in compare_value
            elif self.operator == FilterOperator.NOT_IN:
                return field_value not in compare_value
            elif self.operator == FilterOperator.CONTAINS:
                return compare_value in str(field_value)
            elif self.operator == FilterOperator.REGEX:
                pattern = re.compile(compare_value)
                return bool(pattern.match(str(field_value)))
            
            return False
            
        except Exception as e:
            logger.debug(f"过滤条件评估失败", field=self.field, error=str(e))
            return False
    
    def _get_field_value(self, event: Event, field_path: str) -> Any:
        """获取嵌套字段值"""
        parts = field_path.split('.')
        value = event
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value

class EventFilter:
    """事件过滤器"""
    
    def __init__(
        self,
        name: str = None,
        event_type_pattern: Optional[Pattern] = None,
        source_pattern: Optional[Pattern] = None,
        target_pattern: Optional[Pattern] = None,
        conditions: Optional[List[FilterCondition]] = None,
        custom_filter: Optional[Callable[[Event], bool]] = None,
        logic: str = "AND"  # "AND" 或 "OR"
    ):
        self.name = name or "EventFilter"
        self.event_type_pattern = event_type_pattern
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.conditions = conditions or []
        self.custom_filter = custom_filter
        self.logic = logic.upper()
        
        # 统计信息
        self.stats = {
            "events_evaluated": 0,
            "events_matched": 0,
            "events_rejected": 0
        }
    
    def matches(self, event: Event) -> bool:
        """检查事件是否匹配过滤条件"""
        self.stats["events_evaluated"] += 1
        
        # 检查事件类型模式
        if self.event_type_pattern:
            event_type_str = event.type.value if hasattr(event.type, 'value') else str(event.type)
            if not self.event_type_pattern.match(event_type_str):
                self.stats["events_rejected"] += 1
                return False
        
        # 检查源模式
        if self.source_pattern and hasattr(event, 'source'):
            if not self.source_pattern.match(event.source or ''):
                self.stats["events_rejected"] += 1
                return False
        
        # 检查目标模式
        if self.target_pattern and hasattr(event, 'target'):
            if not self.target_pattern.match(event.target or ''):
                self.stats["events_rejected"] += 1
                return False
        
        # 评估条件
        if self.conditions:
            results = [condition.evaluate(event) for condition in self.conditions]
            
            if self.logic == "AND":
                condition_result = all(results)
            else:  # OR
                condition_result = any(results)
            
            if not condition_result:
                self.stats["events_rejected"] += 1
                return False
        
        # 自定义过滤器
        if self.custom_filter and not self.custom_filter(event):
            self.stats["events_rejected"] += 1
            return False
        
        self.stats["events_matched"] += 1
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取过滤器统计信息"""
        return {
            "name": self.name,
            **self.stats,
            "match_rate": (
                self.stats["events_matched"] / self.stats["events_evaluated"]
                if self.stats["events_evaluated"] > 0 else 0
            )
        }

class EventTransformer:
    """事件转换器"""
    
    def __init__(
        self,
        name: str = None,
        transform_func: Optional[Callable[[Event], Event]] = None
    ):
        self.name = name or "EventTransformer"
        self.transform_func = transform_func or (lambda e: e)
        self.stats = {
            "events_transformed": 0,
            "transform_errors": 0
        }
    
    async def transform(self, event: Event) -> Optional[Event]:
        """转换事件"""
        try:
            transformed = self.transform_func(event)
            self.stats["events_transformed"] += 1
            return transformed
        except Exception as e:
            logger.error(f"事件转换失败", transformer=self.name, error=str(e))
            self.stats["transform_errors"] += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取转换器统计信息"""
        return {
            "name": self.name,
            **self.stats
        }

@dataclass
class Route:
    """路由规则"""
    name: str
    filter: EventFilter
    processors: List[EventProcessor] = field(default_factory=list)
    transformers: List[EventTransformer] = field(default_factory=list)
    priority: int = 10  # 路由优先级
    enabled: bool = True
    stop_on_match: bool = False  # 匹配后是否停止后续路由
    
    def __lt__(self, other):
        """用于优先级排序"""
        return self.priority < other.priority

class EventRouter:
    """事件路由器"""
    
    def __init__(self, name: str = "EventRouter"):
        self.name = name
        self.routes: List[Route] = []
        self.default_processors: List[EventProcessor] = []
        self.global_transformers: List[EventTransformer] = []
        
        # 路由缓存
        self._route_cache: Dict[str, List[Route]] = {}
        self._cache_size = 1000
        
        # 统计信息
        self.stats = {
            "events_routed": 0,
            "routes_matched": 0,
            "default_routes_used": 0,
            "routing_errors": 0
        }
    
    def add_route(
        self,
        filter: EventFilter,
        processors: List[EventProcessor],
        transformers: Optional[List[EventTransformer]] = None,
        name: str = None,
        priority: int = 10,
        stop_on_match: bool = False
    ) -> Route:
        """添加路由规则"""
        route = Route(
            name=name or f"Route-{len(self.routes)}",
            filter=filter,
            processors=processors,
            transformers=transformers or [],
            priority=priority,
            stop_on_match=stop_on_match
        )
        
        self.routes.append(route)
        # 按优先级排序
        self.routes.sort()
        
        # 清除缓存
        self._route_cache.clear()
        
        logger.info(f"添加路由规则", route_name=route.name, priority=priority)
        
        return route
    
    def remove_route(self, route_name: str) -> bool:
        """移除路由规则"""
        for i, route in enumerate(self.routes):
            if route.name == route_name:
                del self.routes[i]
                self._route_cache.clear()
                logger.info(f"移除路由规则", route_name=route_name)
                return True
        return False
    
    def enable_route(self, route_name: str) -> bool:
        """启用路由规则"""
        for route in self.routes:
            if route.name == route_name:
                route.enabled = True
                self._route_cache.clear()
                return True
        return False
    
    def disable_route(self, route_name: str) -> bool:
        """禁用路由规则"""
        for route in self.routes:
            if route.name == route_name:
                route.enabled = False
                self._route_cache.clear()
                return True
        return False
    
    def add_default_processor(self, processor: EventProcessor) -> None:
        """添加默认处理器（当没有路由匹配时使用）"""
        self.default_processors.append(processor)
    
    def add_global_transformer(self, transformer: EventTransformer) -> None:
        """添加全局转换器（应用于所有事件）"""
        self.global_transformers.append(transformer)
    
    async def route_event(
        self,
        event: Event,
        context: Optional[EventContext] = None
    ) -> List[Tuple[EventProcessor, Event]]:
        """路由事件到合适的处理器"""
        self.stats["events_routed"] += 1
        
        try:
            # 应用全局转换器
            transformed_event = await self._apply_global_transformers(event)
            if not transformed_event:
                return []
            
            # 检查缓存
            cache_key = self._get_cache_key(transformed_event)
            if cache_key in self._route_cache:
                matched_routes = self._route_cache[cache_key]
            else:
                # 查找匹配的路由
                matched_routes = []
                for route in self.routes:
                    if not route.enabled:
                        continue
                    
                    if route.filter.matches(transformed_event):
                        matched_routes.append(route)
                        self.stats["routes_matched"] += 1
                        
                        if route.stop_on_match:
                            break
                
                # 更新缓存
                if len(self._route_cache) < self._cache_size:
                    self._route_cache[cache_key] = matched_routes
            
            # 收集处理器
            processor_event_pairs = []
            
            if matched_routes:
                for route in matched_routes:
                    # 应用路由级转换器
                    route_event = transformed_event
                    for transformer in route.transformers:
                        route_event = await transformer.transform(route_event)
                        if not route_event:
                            break
                    
                    if route_event:
                        for processor in route.processors:
                            processor_event_pairs.append((processor, route_event))
            else:
                # 使用默认处理器
                if self.default_processors:
                    self.stats["default_routes_used"] += 1
                    for processor in self.default_processors:
                        processor_event_pairs.append((processor, transformed_event))
            
            return processor_event_pairs
            
        except Exception as e:
            logger.error(f"事件路由失败", error=str(e))
            self.stats["routing_errors"] += 1
            return []
    
    async def _apply_global_transformers(self, event: Event) -> Optional[Event]:
        """应用全局转换器"""
        transformed = event
        for transformer in self.global_transformers:
            transformed = await transformer.transform(transformed)
            if not transformed:
                return None
        return transformed
    
    def _get_cache_key(self, event: Event) -> str:
        """生成事件缓存键"""
        # 基于事件类型和关键属性生成缓存键
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        source = getattr(event, 'source', '')
        target = getattr(event, 'target', '')
        priority = getattr(event, 'priority', EventPriority.NORMAL)
        priority_str = priority.value if hasattr(priority, 'value') else str(priority)
        
        return f"{event_type}:{source}:{target}:{priority_str}"
    
    def get_route_stats(self) -> List[Dict[str, Any]]:
        """获取所有路由的统计信息"""
        route_stats = []
        for route in self.routes:
            stats = {
                "name": route.name,
                "priority": route.priority,
                "enabled": route.enabled,
                "filter_stats": route.filter.get_stats(),
                "processor_count": len(route.processors),
                "transformer_count": len(route.transformers)
            }
            route_stats.append(stats)
        return route_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """获取路由器统计信息"""
        return {
            "name": self.name,
            **self.stats,
            "route_count": len(self.routes),
            "enabled_route_count": sum(1 for r in self.routes if r.enabled),
            "default_processor_count": len(self.default_processors),
            "global_transformer_count": len(self.global_transformers),
            "cache_size": len(self._route_cache),
            "route_stats": self.get_route_stats()
        }

class EventAggregator:
    """事件聚合器"""
    
    def __init__(
        self,
        name: str = "EventAggregator",
        window_size: int = 100,
        time_window_seconds: int = 60
    ):
        self.name = name
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds
        
        # 聚合窗口
        self.event_windows: Dict[str, List[Event]] = {}
        self.window_start_times: Dict[str, datetime] = {}
        
        # 聚合函数
        self.aggregation_functions: Dict[str, Callable[[List[Event]], Any]] = {}
        
        # 统计信息
        self.stats = {
            "events_aggregated": 0,
            "windows_created": 0,
            "aggregations_computed": 0
        }
    
    def add_aggregation_function(
        self,
        name: str,
        func: Callable[[List[Event]], Any]
    ) -> None:
        """添加聚合函数"""
        self.aggregation_functions[name] = func
    
    async def add_event(self, event: Event, window_key: str = "default") -> None:
        """添加事件到聚合窗口"""
        # 初始化窗口
        if window_key not in self.event_windows:
            self.event_windows[window_key] = []
            self.window_start_times[window_key] = utc_now()
            self.stats["windows_created"] += 1
        
        # 检查时间窗口
        current_time = utc_now()
        window_start = self.window_start_times[window_key]
        
        if (current_time - window_start).total_seconds() > self.time_window_seconds:
            # 时间窗口过期，重置
            await self._flush_window(window_key)
            self.event_windows[window_key] = []
            self.window_start_times[window_key] = current_time
        
        # 添加事件
        self.event_windows[window_key].append(event)
        self.stats["events_aggregated"] += 1
        
        # 检查窗口大小
        if len(self.event_windows[window_key]) >= self.window_size:
            await self._flush_window(window_key)
    
    async def _flush_window(self, window_key: str) -> None:
        """刷新聚合窗口"""
        if window_key not in self.event_windows:
            return
        
        events = self.event_windows[window_key]
        if not events:
            return
        
        # 执行聚合函数
        aggregation_results = {}
        for func_name, func in self.aggregation_functions.items():
            try:
                result = func(events)
                aggregation_results[func_name] = result
                self.stats["aggregations_computed"] += 1
            except Exception as e:
                logger.error(f"聚合函数执行失败", function=func_name, error=str(e))
        
        # 记录聚合结果
        logger.info(
            "聚合窗口刷新",
            window_key=window_key,
            event_count=len(events),
            results=aggregation_results
        )
        
        # 清空窗口
        self.event_windows[window_key] = []
    
    async def flush_all_windows(self) -> None:
        """刷新所有聚合窗口"""
        for window_key in list(self.event_windows.keys()):
            await self._flush_window(window_key)
    
    def get_window_stats(self, window_key: str = "default") -> Dict[str, Any]:
        """获取窗口统计信息"""
        if window_key not in self.event_windows:
            return {}
        
        events = self.event_windows[window_key]
        window_start = self.window_start_times[window_key]
        current_time = utc_now()
        
        return {
            "window_key": window_key,
            "event_count": len(events),
            "window_age_seconds": (current_time - window_start).total_seconds(),
            "window_fullness": len(events) / self.window_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "name": self.name,
            **self.stats,
            "active_windows": len(self.event_windows),
            "aggregation_functions": list(self.aggregation_functions.keys())
        }

# 预定义的转换器
class JsonEventTransformer(EventTransformer):
    """JSON事件转换器"""
    
    def __init__(self):
        super().__init__(name="JsonEventTransformer")
    
    async def transform(self, event: Event) -> Optional[Event]:
        """将事件数据转换为JSON兼容格式"""
        try:
            if hasattr(event, 'data') and isinstance(event.data, dict):
                # 确保所有值都是JSON可序列化的
                import json
                json.dumps(event.data)  # 测试序列化
            
            return await super().transform(event)
        except Exception as e:
            logger.error("JSON转换失败", error=str(e))
            return None

class EventEnricher(EventTransformer):
    """事件增强器"""
    
    def __init__(self, enrichment_data: Dict[str, Any]):
        self.enrichment_data = enrichment_data
        super().__init__(name="EventEnricher")
    
    async def transform(self, event: Event) -> Optional[Event]:
        """向事件添加额外数据"""
        try:
            if hasattr(event, 'data') and isinstance(event.data, dict):
                event.data.update(self.enrichment_data)
            
            return await super().transform(event)
        except Exception as e:
            logger.error("事件增强失败", error=str(e))
            return None
