"""
变更追踪器 - 实时跟踪知识图谱的变更和审计日志
"""

import asyncio
import json
import time
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Set, Callable, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from src.core.logging import get_logger, setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class EventType(Enum):
    """事件类型"""
    TRIPLE_ADDED = "triple_added"
    TRIPLE_REMOVED = "triple_removed"
    TRIPLE_MODIFIED = "triple_modified"
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_DELETED = "entity_deleted"
    RELATION_CREATED = "relation_created"
    RELATION_UPDATED = "relation_updated"
    RELATION_DELETED = "relation_deleted"
    GRAPH_LOADED = "graph_loaded"
    GRAPH_CLEARED = "graph_cleared"
    QUERY_EXECUTED = "query_executed"
    BULK_OPERATION = "bulk_operation"
    VERSION_CREATED = "version_created"
    VERSION_RESTORED = "version_restored"

class Priority(Enum):
    """事件优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ChangeEvent:
    """变更事件"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: str
    session_id: Optional[str] = None
    affected_resources: List[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: Priority = Priority.NORMAL
    source: str = "unknown"
    correlation_id: Optional[str] = None

@dataclass
class AuditEntry:
    """审计条目"""
    audit_id: str
    event: ChangeEvent
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

@dataclass
class ChangeStatistics:
    """变更统计"""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_user: Dict[str, int]
    events_by_priority: Dict[str, int]
    average_processing_time_ms: float
    error_rate: float
    events_per_minute: float
    most_active_resources: List[str]

class ChangeListener:
    """变更监听器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    async def on_change(self, event: ChangeEvent) -> bool:
        """处理变更事件"""
        raise NotImplementedError
    
    def should_process(self, event: ChangeEvent) -> bool:
        """判断是否应该处理此事件"""
        return self.enabled

class DatabaseChangeListener(ChangeListener):
    """数据库变更监听器"""
    
    def __init__(self, connection_string: str = None):
        super().__init__("DatabaseListener")
        self.connection_string = connection_string
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def on_change(self, event: ChangeEvent) -> bool:
        """将变更保存到数据库"""
        try:
            # 简化实现 - 实际应该连接到数据库
            self.logger.info(f"保存到数据库: {event.event_type} by {event.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"保存到数据库失败: {e}")
            return False

class WebSocketChangeListener(ChangeListener):
    """WebSocket变更监听器"""
    
    def __init__(self):
        super().__init__("WebSocketListener")
        self.connections: Set[Any] = set()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def on_change(self, event: ChangeEvent) -> bool:
        """通过WebSocket广播变更"""
        try:
            message = {
                "type": "change_event",
                "event": asdict(event),
                "timestamp": event.timestamp.isoformat()
            }
            
            # 简化实现 - 实际应该通过WebSocket发送
            self.logger.info(f"WebSocket广播: {event.event_type}")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket广播失败: {e}")
            return False

class FileChangeListener(ChangeListener):
    """文件变更监听器"""
    
    def __init__(self, log_file: str = "/tmp/kg_changes.log"):
        super().__init__("FileListener")
        self.log_file = log_file
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def on_change(self, event: ChangeEvent) -> bool:
        """将变更写入文件"""
        try:
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "user_id": event.user_id,
                "affected_resources": event.affected_resources or [],
                "metadata": event.metadata or {}
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            return True
        except Exception as e:
            self.logger.error(f"写入文件失败: {e}")
            return False

class ChangeTracker:
    """变更追踪器"""
    
    def __init__(self, 
                 max_events_in_memory: int = 10000,
                 auto_flush_interval: int = 60,
                 enable_real_time_stats: bool = True):
        self.max_events_in_memory = max_events_in_memory
        self.auto_flush_interval = auto_flush_interval
        self.enable_real_time_stats = enable_real_time_stats
        
        # 事件存储
        self.events: deque[ChangeEvent] = deque(maxlen=max_events_in_memory)
        self.audit_log: deque[AuditEntry] = deque(maxlen=max_events_in_memory)
        
        # 监听器
        self.listeners: Dict[str, ChangeListener] = {}
        
        # 统计信息
        self.stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_user": defaultdict(int),
            "events_by_priority": defaultdict(int),
            "processing_times": deque(maxlen=1000),
            "error_count": 0
        }
        
        # 事件过滤器
        self.event_filters: List[Callable[[ChangeEvent], bool]] = []
        
        # 异步处理
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # 批处理
        self.batch_events: List[ChangeEvent] = []
        self.batch_size = 100
        self.batch_timeout = 5.0
        
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def start(self):
        """启动变更追踪器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        
        # 启动自动刷新任务
        if self.auto_flush_interval > 0:
            asyncio.create_task(self._auto_flush_task())
        
        self.logger.info("变更追踪器已启动")
    
    async def stop(self):
        """停止变更追踪器"""
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                raise
        
        # 处理剩余事件
        await self._flush_batch_events()
        
        self.logger.info("变更追踪器已停止")
    
    def add_listener(self, listener: ChangeListener):
        """添加变更监听器"""
        self.listeners[listener.name] = listener
        self.logger.info(f"已添加监听器: {listener.name}")
    
    def remove_listener(self, listener_name: str):
        """移除变更监听器"""
        if listener_name in self.listeners:
            del self.listeners[listener_name]
            self.logger.info(f"已移除监听器: {listener_name}")
    
    def add_event_filter(self, filter_func: Callable[[ChangeEvent], bool]):
        """添加事件过滤器"""
        self.event_filters.append(filter_func)
    
    async def track_change(self, 
                          event_type: EventType,
                          user_id: str,
                          affected_resources: List[str] = None,
                          old_value: Any = None,
                          new_value: Any = None,
                          metadata: Dict[str, Any] = None,
                          priority: Priority = Priority.NORMAL,
                          source: str = "api",
                          session_id: str = None,
                          correlation_id: str = None) -> str:
        """
        跟踪变更事件
        
        Args:
            event_type: 事件类型
            user_id: 用户ID
            affected_resources: 受影响的资源
            old_value: 旧值
            new_value: 新值
            metadata: 元数据
            priority: 优先级
            source: 事件源
            session_id: 会话ID
            correlation_id: 关联ID
            
        Returns:
            str: 事件ID
        """
        # 生成事件ID
        event_id = self._generate_event_id()
        
        # 创建事件
        event = ChangeEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=utc_now(),
            user_id=user_id,
            session_id=session_id,
            affected_resources=affected_resources or [],
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
            priority=priority,
            source=source,
            correlation_id=correlation_id
        )
        
        # 应用过滤器
        if not self._should_process_event(event):
            return event_id
        
        # 添加到处理队列
        await self.processing_queue.put(event)
        
        return event_id
    
    async def get_events(self,
                        limit: int = 100,
                        offset: int = 0,
                        event_type: Optional[EventType] = None,
                        user_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[ChangeEvent]:
        """获取事件列表"""
        events = list(self.events)
        
        # 过滤
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # 排序 - 按时间倒序
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 分页
        return events[offset:offset + limit]
    
    async def get_audit_log(self,
                           limit: int = 100,
                           offset: int = 0,
                           success_only: bool = False) -> List[AuditEntry]:
        """获取审计日志"""
        audit_entries = list(self.audit_log)
        
        if success_only:
            audit_entries = [a for a in audit_entries if a.success]
        
        # 排序
        audit_entries.sort(key=lambda x: x.event.timestamp, reverse=True)
        
        # 分页
        return audit_entries[offset:offset + limit]
    
    async def get_statistics(self) -> ChangeStatistics:
        """获取变更统计"""
        total_events = self.stats["total_events"]
        processing_times = list(self.stats["processing_times"])
        
        # 计算每分钟事件数
        if len(self.events) > 1:
            time_span = (self.events[0].timestamp - self.events[-1].timestamp).total_seconds() / 60
            events_per_minute = len(self.events) / max(time_span, 1)
        else:
            events_per_minute = 0
        
        # 最活跃的资源
        resource_counts = defaultdict(int)
        for event in self.events:
            for resource in event.affected_resources:
                resource_counts[resource] += 1
        
        most_active_resources = sorted(
            resource_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return ChangeStatistics(
            total_events=total_events,
            events_by_type=dict(self.stats["events_by_type"]),
            events_by_user=dict(self.stats["events_by_user"]),
            events_by_priority=dict(self.stats["events_by_priority"]),
            average_processing_time_ms=sum(processing_times) / len(processing_times) if processing_times else 0,
            error_rate=self.stats["error_count"] / max(total_events, 1),
            events_per_minute=events_per_minute,
            most_active_resources=[resource for resource, _ in most_active_resources]
        )
    
    async def get_user_activity(self, user_id: str, hours: int = 24) -> List[ChangeEvent]:
        """获取用户活动记录"""
        start_time = utc_now() - timedelta(hours=hours)
        
        return [
            event for event in self.events
            if event.user_id == user_id and event.timestamp >= start_time
        ]
    
    async def get_resource_history(self, resource_uri: str, limit: int = 50) -> List[ChangeEvent]:
        """获取资源变更历史"""
        return [
            event for event in list(self.events)[-limit:]
            if resource_uri in event.affected_resources
        ]
    
    async def search_events(self, 
                           query: str,
                           fields: List[str] = None,
                           limit: int = 100) -> List[ChangeEvent]:
        """搜索事件"""
        if not fields:
            fields = ["user_id", "affected_resources", "metadata"]
        
        matching_events = []
        query_lower = query.lower()
        
        for event in self.events:
            # 搜索指定字段
            matches = False
            
            if "user_id" in fields and query_lower in event.user_id.lower():
                matches = True
            
            if "affected_resources" in fields:
                for resource in event.affected_resources:
                    if query_lower in resource.lower():
                        matches = True
                        break
            
            if "metadata" in fields and event.metadata:
                metadata_str = json.dumps(event.metadata, ensure_ascii=False).lower()
                if query_lower in metadata_str:
                    matches = True
            
            if matches:
                matching_events.append(event)
                if len(matching_events) >= limit:
                    break
        
        return matching_events
    
    def _generate_event_id(self) -> str:
        """生成事件ID"""
        timestamp = int(time.time() * 1000000)  # 微秒时间戳
        return f"evt_{timestamp}_{len(self.events)}"
    
    def _should_process_event(self, event: ChangeEvent) -> bool:
        """判断是否应该处理事件"""
        for filter_func in self.event_filters:
            if not filter_func(event):
                return False
        return True
    
    async def _process_events(self):
        """处理事件队列"""
        while self.is_running:
            try:
                # 获取事件（带超时）
                try:
                    event = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                start_time = time.time()
                success = True
                error_message = None
                
                try:
                    # 存储事件
                    self.events.append(event)
                    
                    # 更新统计
                    self._update_statistics(event)
                    
                    # 批处理
                    self.batch_events.append(event)
                    
                    # 检查是否需要刷新批处理
                    if (len(self.batch_events) >= self.batch_size or
                        event.priority in [Priority.HIGH, Priority.CRITICAL]):
                        await self._flush_batch_events()
                    
                except Exception as e:
                    success = False
                    error_message = str(e)
                    self.stats["error_count"] += 1
                    self.logger.error(f"处理事件失败 {event.event_id}: {e}")
                
                # 记录处理时间
                processing_time = (time.time() - start_time) * 1000
                self.stats["processing_times"].append(processing_time)
                
                # 创建审计条目
                audit_entry = AuditEntry(
                    audit_id=f"audit_{event.event_id}",
                    event=event,
                    processing_time_ms=processing_time,
                    success=success,
                    error_message=error_message
                )
                
                self.audit_log.append(audit_entry)
                
                # 标记任务完成
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"事件处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _flush_batch_events(self):
        """刷新批处理事件"""
        if not self.batch_events:
            return
        
        events_to_process = self.batch_events.copy()
        self.batch_events.clear()
        
        # 并发处理监听器
        tasks = []
        for event in events_to_process:
            for listener in self.listeners.values():
                if listener.should_process(event):
                    tasks.append(self._call_listener_safe(listener, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_listener_safe(self, listener: ChangeListener, event: ChangeEvent):
        """安全调用监听器"""
        try:
            await listener.on_change(event)
        except Exception as e:
            self.logger.error(f"监听器 {listener.name} 处理事件失败: {e}")
    
    async def _auto_flush_task(self):
        """自动刷新任务"""
        while self.is_running:
            await asyncio.sleep(self.auto_flush_interval)
            await self._flush_batch_events()
    
    def _update_statistics(self, event: ChangeEvent):
        """更新统计信息"""
        self.stats["total_events"] += 1
        self.stats["events_by_type"][event.event_type.value] += 1
        self.stats["events_by_user"][event.user_id] += 1
        self.stats["events_by_priority"][event.priority.value] += 1

# 便捷函数
async def create_change_tracker(
    listeners: List[ChangeListener] = None,
    max_events: int = 10000,
    auto_flush_interval: int = 60
) -> ChangeTracker:
    """
    创建变更追踪器的便捷函数
    
    Args:
        listeners: 监听器列表
        max_events: 内存中最大事件数
        auto_flush_interval: 自动刷新间隔(秒)
        
    Returns:
        ChangeTracker: 变更追踪器实例
    """
    tracker = ChangeTracker(
        max_events_in_memory=max_events,
        auto_flush_interval=auto_flush_interval
    )
    
    # 添加默认监听器
    if not listeners:
        listeners = [
            FileChangeListener(),
            DatabaseChangeListener(),
            WebSocketChangeListener()
        ]
    
    for listener in listeners:
        tracker.add_listener(listener)
    
    await tracker.start()
    return tracker

if __name__ == "__main__":
    # 测试变更追踪器
    async def test_change_tracker():
        setup_logging()
        logger.info("测试变更追踪器")
        
        # 创建追踪器
        tracker = ChangeTracker(max_events_in_memory=1000)
        
        # 添加监听器
        tracker.add_listener(FileChangeListener("/tmp/test_changes.log"))
        
        # 启动
        await tracker.start()
        
        try:
            # 记录一些变更事件
            await tracker.track_change(
                event_type=EventType.ENTITY_CREATED,
                user_id="test_user",
                affected_resources=["ex:John"],
                new_value={"name": "John Doe", "age": 30},
                metadata={"operation": "create_entity"}
            )
            
            await tracker.track_change(
                event_type=EventType.TRIPLE_ADDED,
                user_id="test_user",
                affected_resources=["ex:John", "ex:age"],
                old_value=None,
                new_value={"subject": "ex:John", "predicate": "ex:age", "object": "30"}
            )
            
            await tracker.track_change(
                event_type=EventType.ENTITY_UPDATED,
                user_id="admin_user",
                affected_resources=["ex:John"],
                old_value={"age": 30},
                new_value={"age": 31},
                priority=Priority.HIGH
            )
            
            # 等待处理完成
            await asyncio.sleep(1)
            
            # 获取事件
            events = await tracker.get_events(limit=10)
            logger.info("记录事件数量", total=len(events))
            
            # 获取统计信息
            stats = await tracker.get_statistics()
            logger.info(
                "统计信息",
                total_events=stats.total_events,
                events_per_minute=stats.events_per_minute,
            )
            
            # 搜索事件
            search_results = await tracker.search_events("John")
            logger.info("搜索结果", keyword="John", total=len(search_results))
            
            # 获取用户活动
            user_activity = await tracker.get_user_activity("test_user")
            logger.info("用户活动统计", user_id="test_user", total=len(user_activity))
            
        finally:
            await tracker.stop()
        
        logger.info("变更追踪器测试完成")
    
    asyncio.run(test_change_tracker())
