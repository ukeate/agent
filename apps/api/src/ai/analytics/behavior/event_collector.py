"""
行为事件采集器

负责采集、缓冲、压缩和批量处理用户行为事件。
"""

import asyncio
import json
import gzip
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from collections import defaultdict, deque
from ..models import BehaviorEvent, BulkEventRequest
from ..storage.event_store import EventStore

from src.core.logging import get_logger
logger = get_logger(__name__)

class EventCollector:
    """行为事件采集器"""
    
    def __init__(
        self,
        event_store: EventStore,
        batch_size: int = 1000,
        flush_interval: int = 5,
        max_buffer_size: int = 10000,
        compression_threshold: int = 100,
        enable_compression: bool = True
    ):
        self.event_store = event_store
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        
        # 事件缓冲区
        self.buffer: deque[BehaviorEvent] = deque()
        self.buffer_lock = asyncio.Lock()
        
        # 统计信息
        self.stats = {
            'total_events': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'compression_ratio': 0.0,
            'last_flush_time': utc_now(),
            'buffer_overflows': 0
        }
        
        # 质量监控
        self.quality_metrics = {
            'duplicate_events': 0,
            'invalid_events': 0,
            'late_events': 0,
            'out_of_order_events': 0
        }
        
        # 启动后台刷新任务
        self._flush_task = None
        self._running = False
    
    async def start(self):
        """启动采集器"""
        if self._running:
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"事件采集器已启动, 批量大小: {self.batch_size}, 刷新间隔: {self.flush_interval}s")
    
    async def stop(self):
        """停止采集器"""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                raise
        
        # 最后刷新缓冲区
        await self.flush()
        logger.info("事件采集器已停止")
    
    async def collect_event(self, event: BehaviorEvent) -> bool:
        """采集单个事件"""
        try:
            # 数据质量检查
            if not await self._validate_event(event):
                self.quality_metrics['invalid_events'] += 1
                return False
            
            async with self.buffer_lock:
                # 检查缓冲区是否已满
                if len(self.buffer) >= self.max_buffer_size:
                    logger.warning(f"缓冲区已满({len(self.buffer)}), 强制刷新")
                    self.stats['buffer_overflows'] += 1
                    await self._flush_buffer()
                
                self.buffer.append(event)
                self.stats['total_events'] += 1
                
                # 检查是否需要立即刷新
                if len(self.buffer) >= self.batch_size:
                    await self._flush_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"采集事件失败: {e}")
            return False
    
    async def collect_batch(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """批量采集事件"""
        results = {
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for event in events:
            try:
                success = await self.collect_event(event)
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(str(e))
        
        return results
    
    async def flush(self) -> bool:
        """手动刷新缓冲区"""
        async with self.buffer_lock:
            return await self._flush_buffer()
    
    async def _flush_buffer(self) -> bool:
        """内部刷新缓冲区方法"""
        if not self.buffer:
            return True
        
        try:
            # 复制缓冲区内容
            events_to_flush = list(self.buffer)
            self.buffer.clear()
            
            # 创建批量请求
            bulk_request = BulkEventRequest(
                events=events_to_flush,
                compression='gzip' if self.enable_compression and len(events_to_flush) >= self.compression_threshold else None
            )
            
            # 写入存储
            success = await self.event_store.store_events_batch(bulk_request)
            
            if success:
                self.stats['successful_batches'] += 1
                self.stats['last_flush_time'] = utc_now()
                logger.debug(f"成功刷新{len(events_to_flush)}个事件")
            else:
                self.stats['failed_batches'] += 1
                # 失败时重新入队(可选策略)
                # self.buffer.extendleft(reversed(events_to_flush))
                logger.error(f"刷新{len(events_to_flush)}个事件失败")
            
            return success
            
        except Exception as e:
            self.stats['failed_batches'] += 1
            logger.error(f"刷新缓冲区异常: {e}")
            return False
    
    async def _periodic_flush(self):
        """定期刷新缓冲区"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self.buffer:
                    async with self.buffer_lock:
                        await self._flush_buffer()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期刷新异常: {e}")
    
    async def _validate_event(self, event: BehaviorEvent) -> bool:
        """验证事件数据质量"""
        try:
            # 检查必填字段
            if not event.user_id or not event.session_id or not event.event_name:
                return False
            
            # 检查时间戳合理性(不能超过未来1小时或过去24小时)
            now = utc_now()
            if event.timestamp > now + timedelta(hours=1):
                self.quality_metrics['late_events'] += 1
                return False
            
            if event.timestamp < now - timedelta(hours=24):
                self.quality_metrics['late_events'] += 1
                # 仍然接受,但标记为迟到事件
            
            # 检查事件数据大小(避免过大的事件)
            event_size = len(json.dumps(event.event_data, default=str))
            if event_size > 10240:  # 10KB限制
                logger.warning(f"事件数据过大: {event_size}字节")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"事件验证失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取采集器统计信息"""
        buffer_size = len(self.buffer)
        buffer_utilization = buffer_size / self.max_buffer_size
        
        return {
            **self.stats,
            **self.quality_metrics,
            'buffer_size': buffer_size,
            'buffer_utilization': buffer_utilization,
            'is_running': self._running,
            'last_updated': utc_now().isoformat()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_events': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'compression_ratio': 0.0,
            'last_flush_time': utc_now(),
            'buffer_overflows': 0
        }
        
        self.quality_metrics = {
            'duplicate_events': 0,
            'invalid_events': 0,
            'late_events': 0,
            'out_of_order_events': 0
        }

class CompressionUtils:
    """事件数据压缩工具"""
    
    @staticmethod
    def compress_events(events: List[BehaviorEvent], method: str = 'gzip') -> bytes:
        """压缩事件列表"""
        try:
            # 序列化事件
            data = json.dumps([event.model_dump(mode="json") for event in events])
            
            if method == 'gzip':
                return gzip.compress(data.encode('utf-8'))
            else:
                raise ValueError(f"不支持的压缩方法: {method}")
                
        except Exception as e:
            logger.error(f"事件压缩失败: {e}")
            raise
    
    @staticmethod
    def decompress_events(compressed_data: bytes, method: str = 'gzip') -> List[Dict[str, Any]]:
        """解压缩事件列表"""
        try:
            if method == 'gzip':
                data = gzip.decompress(compressed_data).decode('utf-8')
                return json.loads(data)
            else:
                raise ValueError(f"不支持的解压缩方法: {method}")
                
        except Exception as e:
            logger.error(f"事件解压缩失败: {e}")
            raise
    
    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """计算压缩比"""
        if original_size == 0:
            return 0.0
        return (original_size - compressed_size) / original_size

class EventCollectorPool:
    """事件采集器池
    
    用于处理高并发采集需求，按用户或会话分片处理。
    """
    
    def __init__(
        self,
        event_store: EventStore,
        pool_size: int = 4,
        **collector_kwargs
    ):
        self.event_store = event_store
        self.pool_size = pool_size
        self.collector_kwargs = collector_kwargs
        
        # 创建采集器池
        self.collectors: List[EventCollector] = []
        for i in range(pool_size):
            collector = EventCollector(event_store, **collector_kwargs)
            self.collectors.append(collector)
        
        self._round_robin_index = 0
    
    async def start(self):
        """启动所有采集器"""
        for collector in self.collectors:
            await collector.start()
        logger.info(f"采集器池已启动, 池大小: {self.pool_size}")
    
    async def stop(self):
        """停止所有采集器"""
        for collector in self.collectors:
            await collector.stop()
        logger.info("采集器池已停止")
    
    async def collect_event(self, event: BehaviorEvent) -> bool:
        """采集事件(负载均衡)"""
        # 基于用户ID进行哈希分片
        collector_index = hash(event.user_id) % self.pool_size
        return await self.collectors[collector_index].collect_event(event)
    
    async def collect_batch(self, events: List[BehaviorEvent]) -> Dict[str, Any]:
        """批量采集事件"""
        # 按用户ID分组
        user_groups = defaultdict(list)
        for event in events:
            user_groups[event.user_id].append(event)
        
        # 并发处理各组
        tasks = []
        for user_id, user_events in user_groups.items():
            collector_index = hash(user_id) % self.pool_size
            task = self.collectors[collector_index].collect_batch(user_events)
            tasks.append(task)
        
        # 汇总结果
        results = await asyncio.gather(*tasks)
        total_result = {
            'successful': sum(r['successful'] for r in results),
            'failed': sum(r['failed'] for r in results),
            'errors': [error for r in results for error in r['errors']]
        }
        
        return total_result
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        collectors_stats = [collector.get_stats() for collector in self.collectors]
        
        total_stats = {
            'pool_size': self.pool_size,
            'total_events': sum(stats['total_events'] for stats in collectors_stats),
            'successful_batches': sum(stats['successful_batches'] for stats in collectors_stats),
            'failed_batches': sum(stats['failed_batches'] for stats in collectors_stats),
            'buffer_overflows': sum(stats['buffer_overflows'] for stats in collectors_stats),
            'collectors': collectors_stats
        }
        
        return total_stats
