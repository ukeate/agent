"""
流式数据缓冲管理

提供流式数据的缓冲、背压控制和状态追踪功能。
"""

from typing import Any, Dict, Optional, List, Callable, AsyncIterator
from dataclasses import dataclass, field
import asyncio
import time
from enum import Enum
from collections import deque

from src.core.logging import get_logger
logger = get_logger(__name__)

class BufferStatus(str, Enum):
    """缓冲区状态"""
    NORMAL = "normal"
    HIGH_USAGE = "high_usage"
    FULL = "full"
    OVERFLOW = "overflow"

@dataclass
class BufferMetrics:
    """缓冲区指标"""
    current_size: int = 0
    max_size: int = 0
    total_added: int = 0
    total_removed: int = 0
    overflow_count: int = 0
    high_watermark_hits: int = 0
    status: BufferStatus = BufferStatus.NORMAL
    last_updated: float = field(default_factory=time.time)

class StreamBuffer:
    """流式数据缓冲器"""
    
    def __init__(
        self, 
        max_size: int = 1000,
        high_watermark: float = 0.8,
        low_watermark: float = 0.3,
        overflow_strategy: str = "drop_oldest"
    ):
        self.max_size = max_size
        self.high_watermark = int(max_size * high_watermark)
        self.low_watermark = int(max_size * low_watermark)
        self.overflow_strategy = overflow_strategy  # drop_oldest, drop_newest, block
        
        self._buffer = deque(maxlen=max_size)
        self._metrics = BufferMetrics(max_size=max_size)
        self._condition = asyncio.Condition()
        self._overflow_callbacks: List[Callable] = []
        self._watermark_callbacks: List[Callable] = []
        
    async def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """
        添加项目到缓冲区
        
        Args:
            item: 要缓冲的数据项
            timeout: 超时时间(秒)，仅在block策略下有效
            
        Returns:
            bool: 是否成功添加
        """
        async with self._condition:
            current_size = len(self._buffer)
            
            # 检查是否已满
            if current_size >= self.max_size:
                if self.overflow_strategy == "drop_oldest":
                    dropped = self._buffer.popleft()  # 移除最旧的
                    logger.debug(f"缓冲区已满，丢弃最旧项目: {type(dropped)}")
                elif self.overflow_strategy == "drop_newest":
                    logger.debug(f"缓冲区已满，丢弃新项目: {type(item)}")
                    self._metrics.overflow_count += 1
                    return False
                elif self.overflow_strategy == "block":
                    # 等待有空间
                    start_time = time.time()
                    while len(self._buffer) >= self.max_size:
                        if timeout and (time.time() - start_time) > timeout:
                            return False
                        await self._condition.wait()
                else:
                    raise ValueError(f"未知的溢出策略: {self.overflow_strategy}")
            
            # 添加项目
            self._buffer.append(item)
            self._metrics.current_size = len(self._buffer)
            self._metrics.total_added += 1
            self._metrics.last_updated = time.time()
            
            # 检查水位线
            await self._check_watermarks()
            
            # 通知等待的消费者
            self._condition.notify_all()
            
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        从缓冲区获取项目
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            Optional[Any]: 获取的项目，超时或空时返回None
        """
        async with self._condition:
            start_time = time.time()
            
            while len(self._buffer) == 0:
                if timeout and (time.time() - start_time) > timeout:
                    return None
                await self._condition.wait()
            
            item = self._buffer.popleft()
            self._metrics.current_size = len(self._buffer)
            self._metrics.total_removed += 1
            self._metrics.last_updated = time.time()
            
            # 检查水位线
            await self._check_watermarks()
            
            # 通知等待的生产者
            self._condition.notify_all()
            
            return item
    
    async def peek(self) -> Optional[Any]:
        """查看缓冲区中的第一个项目而不移除它"""
        async with self._condition:
            return self._buffer[0] if self._buffer else None
    
    async def get_many(self, count: int, timeout: Optional[float] = None) -> List[Any]:
        """
        批量获取多个项目
        
        Args:
            count: 要获取的项目数量
            timeout: 超时时间(秒)
            
        Returns:
            List[Any]: 获取的项目列表
        """
        items = []
        start_time = time.time()
        
        for _ in range(count):
            remaining_timeout = None
            if timeout:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0:
                    break
            
            item = await self.get(timeout=remaining_timeout)
            if item is None:
                break
            items.append(item)
        
        return items
    
    async def drain(self) -> List[Any]:
        """清空缓冲区并返回所有项目"""
        async with self._condition:
            items = list(self._buffer)
            self._buffer.clear()
            self._metrics.current_size = 0
            self._metrics.total_removed += len(items)
            self._metrics.last_updated = time.time()
            self._condition.notify_all()
            return items
    
    async def _check_watermarks(self):
        """检查水位线并触发相应回调"""
        current_size = len(self._buffer)
        usage_ratio = current_size / self.max_size
        
        old_status = self._metrics.status
        
        # 更新状态
        if current_size >= self.max_size:
            self._metrics.status = BufferStatus.FULL
            if current_size > self.max_size:
                self._metrics.status = BufferStatus.OVERFLOW
                self._metrics.overflow_count += 1
        elif current_size >= self.high_watermark:
            self._metrics.status = BufferStatus.HIGH_USAGE
            if old_status != BufferStatus.HIGH_USAGE:
                self._metrics.high_watermark_hits += 1
        else:
            self._metrics.status = BufferStatus.NORMAL
        
        # 触发状态变化回调
        if old_status != self._metrics.status:
            await self._trigger_callbacks()
    
    async def _trigger_callbacks(self):
        """触发状态变化回调"""
        callbacks = []
        
        if self._metrics.status in [BufferStatus.FULL, BufferStatus.OVERFLOW]:
            callbacks.extend(self._overflow_callbacks)
        
        if self._metrics.status in [BufferStatus.HIGH_USAGE, BufferStatus.FULL]:
            callbacks.extend(self._watermark_callbacks)
        
        # 异步执行回调
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._metrics)
                else:
                    callback(self._metrics)
            except Exception as e:
                logger.error(f"执行缓冲区回调时出错: {e}")
    
    def add_overflow_callback(self, callback: Callable):
        """添加溢出回调"""
        self._overflow_callbacks.append(callback)
    
    def add_watermark_callback(self, callback: Callable):
        """添加水位线回调"""
        self._watermark_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """移除回调"""
        if callback in self._overflow_callbacks:
            self._overflow_callbacks.remove(callback)
        if callback in self._watermark_callbacks:
            self._watermark_callbacks.remove(callback)
    
    @property
    def size(self) -> int:
        """获取当前缓冲区大小"""
        return len(self._buffer)
    
    @property
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return len(self._buffer) == 0
    
    @property
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self._buffer) >= self.max_size
    
    @property
    def usage_ratio(self) -> float:
        """获取缓冲区使用率"""
        return len(self._buffer) / self.max_size
    
    def get_metrics(self) -> BufferMetrics:
        """获取缓冲区指标"""
        self._metrics.current_size = len(self._buffer)
        self._metrics.last_updated = time.time()
        return self._metrics
    
    def reset_metrics(self):
        """重置指标"""
        current_size = len(self._buffer)
        self._metrics = BufferMetrics(
            current_size=current_size,
            max_size=self.max_size
        )

class MultiStreamBuffer:
    """多流缓冲管理器"""
    
    def __init__(self, default_max_size: int = 1000):
        self.default_max_size = default_max_size
        self.buffers: Dict[str, StreamBuffer] = {}
        self._lock = asyncio.Lock()
    
    async def get_buffer(self, stream_id: str, max_size: Optional[int] = None) -> StreamBuffer:
        """获取或创建指定流的缓冲区"""
        async with self._lock:
            if stream_id not in self.buffers:
                size = max_size or self.default_max_size
                self.buffers[stream_id] = StreamBuffer(max_size=size)
            return self.buffers[stream_id]
    
    async def remove_buffer(self, stream_id: str) -> Optional[List[Any]]:
        """移除指定流的缓冲区并返回剩余数据"""
        async with self._lock:
            if stream_id in self.buffers:
                buffer = self.buffers.pop(stream_id)
                return await buffer.drain()
            return None
    
    async def get_all_metrics(self) -> Dict[str, BufferMetrics]:
        """获取所有缓冲区的指标"""
        async with self._lock:
            return {
                stream_id: buffer.get_metrics() 
                for stream_id, buffer in self.buffers.items()
            }
    
    async def cleanup_empty_buffers(self):
        """清理空的缓冲区"""
        async with self._lock:
            empty_buffers = [
                stream_id for stream_id, buffer in self.buffers.items()
                if buffer.is_empty
            ]
            
            for stream_id in empty_buffers:
                self.buffers.pop(stream_id)
            
            if empty_buffers:
                logger.info(f"清理了 {len(empty_buffers)} 个空缓冲区")
    
    def get_buffer_count(self) -> int:
        """获取缓冲区数量"""
        return len(self.buffers)
