"""
用户反馈收集服务

负责收集和缓冲各类用户反馈信号，支持批量处理和实时传输。
实现反馈事件的去重、缓冲、验证和质量初评。
"""

import asyncio
import json
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque

from models.schemas.feedback import FeedbackType
from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EventPriority(str, Enum):
    """事件优先级"""
    HIGH = "high"        # 显式反馈 - 立即处理
    MEDIUM = "medium"    # 关键隐式反馈 - 快速处理
    LOW = "low"         # 一般隐式反馈 - 批量处理


@dataclass
class CollectedEvent:
    """收集的事件数据结构"""
    event_id: str
    user_id: str
    session_id: str
    item_id: str
    feedback_type: FeedbackType
    raw_value: Any
    context: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


class FeedbackBuffer:
    """反馈事件缓冲器"""
    
    def __init__(self, max_size: int = 1000, flush_interval: float = 5.0):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: deque[CollectedEvent] = deque()
        self.priority_buffers: Dict[EventPriority, deque[CollectedEvent]] = {
            EventPriority.HIGH: deque(),
            EventPriority.MEDIUM: deque(),
            EventPriority.LOW: deque()
        }
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        
    async def add_event(self, event: CollectedEvent) -> bool:
        """添加事件到缓冲区"""
        async with self._lock:
            # 检查缓冲区是否已满
            if len(self.buffer) >= self.max_size:
                logger.warning(f"Buffer full, dropping event {event.event_id}")
                return False
                
            # 按优先级分类缓冲
            self.priority_buffers[event.priority].append(event)
            self.buffer.append(event)
            
            logger.debug(f"Added event {event.event_id} with priority {event.priority}")
            return True
    
    async def should_flush(self) -> bool:
        """检查是否应该刷新缓冲区"""
        current_time = time.time()
        time_threshold = current_time - self._last_flush >= self.flush_interval
        size_threshold = len(self.buffer) >= self.max_size * 0.8  # 80% 阈值
        
        # 高优先级事件立即刷新
        has_high_priority = len(self.priority_buffers[EventPriority.HIGH]) > 0
        
        return time_threshold or size_threshold or has_high_priority
    
    async def flush(self) -> List[CollectedEvent]:
        """刷新缓冲区，返回所有事件"""
        async with self._lock:
            if not self.buffer:
                return []
                
            # 按优先级排序输出
            events = []
            for priority in [EventPriority.HIGH, EventPriority.MEDIUM, EventPriority.LOW]:
                while self.priority_buffers[priority]:
                    events.append(self.priority_buffers[priority].popleft())
            
            # 清空缓冲区
            self.buffer.clear()
            self._last_flush = time.time()
            
            logger.info(f"Flushed {len(events)} events from buffer")
            return events


class EventDeduplicator:
    """事件去重器"""
    
    def __init__(self, window_seconds: int = 300):  # 5分钟去重窗口
        self.window_seconds = window_seconds
        self.seen_events: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
    def _generate_event_key(self, event: CollectedEvent) -> str:
        """生成事件唯一标识"""
        key_components = [
            event.user_id,
            event.item_id,
            event.feedback_type.value,
            str(event.raw_value),
        ]
        return "|".join(key_components)
    
    async def is_duplicate(self, event: CollectedEvent) -> bool:
        """检查事件是否重复"""
        async with self._lock:
            event_key = self._generate_event_key(event)
            current_time = utc_now()
            
            # 清理过期的去重记录
            await self._cleanup_expired_entries(current_time)
            
            # 检查是否在去重窗口内
            if event_key in self.seen_events:
                last_seen = self.seen_events[event_key]
                if (current_time - last_seen).total_seconds() < self.window_seconds:
                    logger.debug(f"Duplicate event detected: {event_key}")
                    return True
            
            # 记录新事件
            self.seen_events[event_key] = current_time
            return False
    
    async def _cleanup_expired_entries(self, current_time: datetime):
        """清理过期的去重记录"""
        expired_keys = []
        for key, timestamp in self.seen_events.items():
            if (current_time - timestamp).total_seconds() > self.window_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.seen_events[key]


class EventValidator:
    """事件验证器"""
    
    @staticmethod
    def validate_implicit_event(event: CollectedEvent) -> bool:
        """验证隐式反馈事件"""
        try:
            feedback_type = event.feedback_type
            raw_value = event.raw_value
            
            if feedback_type == FeedbackType.CLICK:
                return isinstance(raw_value, bool) or raw_value in [0, 1]
                
            elif feedback_type == FeedbackType.DWELL_TIME:
                return isinstance(raw_value, (int, float)) and 0 <= raw_value <= 3600  # 最大1小时
                
            elif feedback_type == FeedbackType.SCROLL_DEPTH:
                return isinstance(raw_value, (int, float)) and 0 <= raw_value <= 100  # 百分比
                
            elif feedback_type == FeedbackType.VIEW:
                return isinstance(raw_value, bool) or raw_value in [0, 1]
                
            elif feedback_type == FeedbackType.HOVER:
                return isinstance(raw_value, (int, float)) and raw_value >= 0
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating implicit event: {e}")
            return False
    
    @staticmethod
    def validate_explicit_event(event: CollectedEvent) -> bool:
        """验证显式反馈事件"""
        try:
            feedback_type = event.feedback_type
            raw_value = event.raw_value
            
            if feedback_type == FeedbackType.RATING:
                return isinstance(raw_value, (int, float)) and 1 <= raw_value <= 5
                
            elif feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE]:
                return isinstance(raw_value, bool) or raw_value in [0, 1]
                
            elif feedback_type == FeedbackType.BOOKMARK:
                return isinstance(raw_value, bool) or raw_value in [0, 1]
                
            elif feedback_type == FeedbackType.SHARE:
                return isinstance(raw_value, bool) or raw_value in [0, 1]
                
            elif feedback_type == FeedbackType.COMMENT:
                return isinstance(raw_value, str) and len(raw_value.strip()) > 0
                
            return False
            
        except Exception as e:
            logger.error(f"Error validating explicit event: {e}")
            return False


class FeedbackCollector:
    """反馈收集器主类"""
    
    def __init__(self):
        self.buffer = FeedbackBuffer(
            max_size=settings.FEEDBACK_BUFFER_SIZE or 1000,
            flush_interval=settings.FEEDBACK_FLUSH_INTERVAL or 5.0
        )
        self.deduplicator = EventDeduplicator(
            window_seconds=settings.FEEDBACK_DEDUP_WINDOW or 300
        )
        self.validator = EventValidator()
        
        # 统计信息
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "duplicates_filtered": 0,
            "validation_failures": 0,
            "buffer_overflows": 0
        }
        
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
    async def start(self):
        """启动收集器"""
        if self._running:
            return
            
        self._running = True
        
        # 启动后台任务
        task = asyncio.create_task(self._background_flush_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        logger.info("Feedback collector started")
    
    async def stop(self):
        """停止收集器"""
        self._running = False
        
        # 取消所有后台任务
        for task in self._background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        
        # 最后一次刷新缓冲区
        await self._flush_events()
        
        logger.info("Feedback collector stopped")
    
    async def collect_implicit_feedback(
        self, 
        user_id: str,
        session_id: str,
        item_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """收集隐式反馈"""
        self.stats["total_received"] += 1
        
        try:
            # 解析反馈类型
            feedback_type = FeedbackType(event_type)
            
            # 创建事件对象
            event = CollectedEvent(
                event_id=f"{user_id}_{session_id}_{item_id}_{int(time.time() * 1000)}",
                user_id=user_id,
                session_id=session_id,
                item_id=item_id,
                feedback_type=feedback_type,
                raw_value=event_data.get("value"),
                context=context or {},
                timestamp=utc_now(),
                priority=EventPriority.LOW,  # 隐式反馈低优先级
                client_ip=event_data.get("client_ip"),
                user_agent=event_data.get("user_agent")
            )
            
            # 验证事件
            if not self.validator.validate_implicit_event(event):
                self.stats["validation_failures"] += 1
                logger.warning(f"Invalid implicit event: {event.event_id}")
                return False
            
            # 去重检查
            if await self.deduplicator.is_duplicate(event):
                self.stats["duplicates_filtered"] += 1
                return False
            
            # 添加到缓冲区
            if await self.buffer.add_event(event):
                self.stats["total_processed"] += 1
                logger.debug(f"Collected implicit feedback: {event.event_id}")
                return True
            else:
                self.stats["buffer_overflows"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error collecting implicit feedback: {e}")
            return False
    
    async def collect_explicit_feedback(
        self,
        user_id: str,
        session_id: str,
        item_id: str,
        feedback_type: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """收集显式反馈"""
        self.stats["total_received"] += 1
        
        try:
            # 解析反馈类型
            fb_type = FeedbackType(feedback_type)
            
            # 创建事件对象
            event = CollectedEvent(
                event_id=f"{user_id}_{session_id}_{item_id}_{int(time.time() * 1000)}",
                user_id=user_id,
                session_id=session_id,
                item_id=item_id,
                feedback_type=fb_type,
                raw_value=value,
                context=context or {},
                timestamp=utc_now(),
                priority=EventPriority.HIGH,  # 显式反馈高优先级
            )
            
            # 验证事件
            if not self.validator.validate_explicit_event(event):
                self.stats["validation_failures"] += 1
                logger.warning(f"Invalid explicit event: {event.event_id}")
                return False
            
            # 去重检查（显式反馈也要去重）
            if await self.deduplicator.is_duplicate(event):
                self.stats["duplicates_filtered"] += 1
                return False
            
            # 添加到缓冲区
            if await self.buffer.add_event(event):
                self.stats["total_processed"] += 1
                logger.info(f"Collected explicit feedback: {event.event_id}")
                return True
            else:
                self.stats["buffer_overflows"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error collecting explicit feedback: {e}")
            return False
    
    async def collect_batch_feedback(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """批量收集反馈"""
        results = {
            "total": len(events),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for event_data in events:
            try:
                if event_data.get("is_explicit", False):
                    success = await self.collect_explicit_feedback(
                        user_id=event_data["user_id"],
                        session_id=event_data["session_id"],
                        item_id=event_data["item_id"],
                        feedback_type=event_data["feedback_type"],
                        value=event_data["value"],
                        context=event_data.get("context")
                    )
                else:
                    success = await self.collect_implicit_feedback(
                        user_id=event_data["user_id"],
                        session_id=event_data["session_id"],
                        item_id=event_data["item_id"],
                        event_type=event_data["event_type"],
                        event_data=event_data,
                        context=event_data.get("context")
                    )
                
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))
                logger.error(f"Error in batch feedback collection: {e}")
        
        logger.info(f"Batch collection completed: {results}")
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取收集器统计信息"""
        current_buffer_size = len(self.buffer.buffer)
        buffer_utilization = (current_buffer_size / self.buffer.max_size) * 100
        
        return {
            **self.stats,
            "current_buffer_size": current_buffer_size,
            "buffer_utilization_percent": round(buffer_utilization, 2),
            "dedup_entries": len(self.deduplicator.seen_events),
            "is_running": self._running
        }
    
    async def _background_flush_task(self):
        """后台定期刷新任务"""
        while self._running:
            try:
                if await self.buffer.should_flush():
                    await self._flush_events()
                    
                await asyncio.sleep(1)  # 每秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush task: {e}")
                await asyncio.sleep(5)  # 错误后等待5秒
    
    async def _flush_events(self):
        """刷新事件到处理器"""
        events = await self.buffer.flush()
        if not events:
            return
            
        # 这里应该调用反馈处理器
        # 为了避免循环依赖，我们使用事件或消息队列
        # 暂时记录日志
        logger.info(f"Flushing {len(events)} events for processing")
        
        # TODO: 集成反馈处理器
        # await self.feedback_processor.process_events(events)


# 全局收集器实例
_feedback_collector: Optional[FeedbackCollector] = None


async def get_feedback_collector() -> FeedbackCollector:
    """获取全局反馈收集器实例"""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
        await _feedback_collector.start()
    return _feedback_collector


async def shutdown_feedback_collector():
    """关闭反馈收集器"""
    global _feedback_collector
    if _feedback_collector is not None:
        await _feedback_collector.stop()
        _feedback_collector = None