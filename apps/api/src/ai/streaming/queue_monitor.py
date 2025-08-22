"""
队列深度监控器

监控各种队列的深度和处理速度，为背压控制提供数据支持。
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import time
from datetime import datetime, timedelta
from collections import deque
import weakref

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """队列指标"""
    name: str
    current_size: int
    max_size: int
    enqueue_rate: float  # 入队速率（项目/秒）
    dequeue_rate: float  # 出队速率（项目/秒）
    average_wait_time: float  # 平均等待时间（秒）
    oldest_item_age: float  # 最老项目年龄（秒）
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def utilization(self) -> float:
        """队列利用率 (0-1)"""
        return self.current_size / self.max_size if self.max_size > 0 else 0
    
    @property
    def is_overloaded(self) -> bool:
        """是否过载"""
        return self.utilization > 0.8 or self.enqueue_rate > self.dequeue_rate * 1.2
    
    @property
    def throughput_ratio(self) -> float:
        """吞吐量比率（出队/入队）"""
        return self.dequeue_rate / self.enqueue_rate if self.enqueue_rate > 0 else 1.0


@dataclass
class QueueOperation:
    """队列操作记录"""
    operation: str  # enqueue, dequeue
    timestamp: float
    queue_size_before: int
    queue_size_after: int
    item_id: Optional[str] = None
    processing_time: Optional[float] = None


class QueueMonitor:
    """队列监控器"""
    
    def __init__(self, name: str, max_size: int, monitoring_window: float = 60.0):
        self.name = name
        self.max_size = max_size
        self.monitoring_window = monitoring_window
        
        # 当前状态
        self.current_size = 0
        self.is_monitoring = False
        
        # 操作历史
        self.operations: deque = deque(maxlen=10000)
        self.item_timestamps: Dict[str, float] = {}
        
        # 速率计算
        self._last_metrics_time = time.time()
        self._last_enqueue_count = 0
        self._last_dequeue_count = 0
        
        # 回调函数
        self.overload_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[QueueMetrics] = []
    
    async def start_monitoring(self, check_interval: float = 5.0):
        """启动监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(check_interval))
        logger.info(f"队列监控已启动: {self.name}")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"队列监控已停止: {self.name}")
    
    async def _monitor_loop(self, check_interval: float):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._calculate_metrics()
                self._metrics_history.append(metrics)
                
                # 限制历史记录大小
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-500:]
                
                # 检查是否过载
                if metrics.is_overloaded:
                    await self._trigger_overload_callbacks(metrics)
                
                # 触发指标回调
                await self._trigger_metrics_callbacks(metrics)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"队列监控错误 {self.name}: {e}")
                await asyncio.sleep(check_interval)
    
    def record_enqueue(self, item_id: Optional[str] = None):
        """记录入队操作"""
        timestamp = time.time()
        
        operation = QueueOperation(
            operation="enqueue",
            timestamp=timestamp,
            queue_size_before=self.current_size,
            queue_size_after=self.current_size + 1,
            item_id=item_id
        )
        
        self.operations.append(operation)
        self.current_size += 1
        
        if item_id:
            self.item_timestamps[item_id] = timestamp
    
    def record_dequeue(self, item_id: Optional[str] = None, processing_time: Optional[float] = None):
        """记录出队操作"""
        timestamp = time.time()
        
        operation = QueueOperation(
            operation="dequeue",
            timestamp=timestamp,
            queue_size_before=self.current_size,
            queue_size_after=max(0, self.current_size - 1),
            item_id=item_id,
            processing_time=processing_time
        )
        
        self.operations.append(operation)
        self.current_size = max(0, self.current_size - 1)
        
        # 清理项目时间戳
        if item_id and item_id in self.item_timestamps:
            del self.item_timestamps[item_id]
    
    def _calculate_metrics(self) -> QueueMetrics:
        """计算队列指标"""
        current_time = time.time()
        window_start = current_time - self.monitoring_window
        
        # 过滤时间窗口内的操作
        recent_ops = [op for op in self.operations if op.timestamp >= window_start]
        
        # 计算速率
        enqueue_count = len([op for op in recent_ops if op.operation == "enqueue"])
        dequeue_count = len([op for op in recent_ops if op.operation == "dequeue"])
        
        enqueue_rate = enqueue_count / self.monitoring_window
        dequeue_rate = dequeue_count / self.monitoring_window
        
        # 计算平均等待时间
        dequeue_ops = [op for op in recent_ops if op.operation == "dequeue" and op.processing_time]
        avg_wait_time = sum(op.processing_time for op in dequeue_ops) / len(dequeue_ops) if dequeue_ops else 0
        
        # 计算最老项目年龄
        oldest_age = 0
        if self.item_timestamps:
            oldest_timestamp = min(self.item_timestamps.values())
            oldest_age = current_time - oldest_timestamp
        
        return QueueMetrics(
            name=self.name,
            current_size=self.current_size,
            max_size=self.max_size,
            enqueue_rate=enqueue_rate,
            dequeue_rate=dequeue_rate,
            average_wait_time=avg_wait_time,
            oldest_item_age=oldest_age
        )
    
    async def _trigger_overload_callbacks(self, metrics: QueueMetrics):
        """触发过载回调"""
        for callback in self.overload_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"过载回调执行失败: {e}")
    
    async def _trigger_metrics_callbacks(self, metrics: QueueMetrics):
        """触发指标回调"""
        for callback in self.metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")
    
    def add_overload_callback(self, callback: Callable):
        """添加过载回调"""
        self.overload_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """添加指标回调"""
        self.metrics_callbacks.append(callback)
    
    def get_current_metrics(self) -> QueueMetrics:
        """获取当前指标"""
        return self._calculate_metrics()
    
    def get_metrics_history(self, limit: int = 100) -> List[QueueMetrics]:
        """获取指标历史"""
        return self._metrics_history[-limit:] if limit else self._metrics_history
    
    def get_recent_operations(self, limit: int = 100) -> List[QueueOperation]:
        """获取最近操作记录"""
        return list(self.operations)[-limit:] if limit else list(self.operations)
    
    def reset_metrics(self):
        """重置指标"""
        self.operations.clear()
        self.item_timestamps.clear()
        self._metrics_history.clear()
        self.current_size = 0


class QueueMonitorManager:
    """队列监控管理器"""
    
    def __init__(self):
        self.monitors: Dict[str, QueueMonitor] = {}
        self._global_callbacks: List[Callable] = []
        self.is_running = False
    
    def register_queue(self, name: str, max_size: int, monitoring_window: float = 60.0) -> QueueMonitor:
        """注册队列监控"""
        monitor = QueueMonitor(name, max_size, monitoring_window)
        self.monitors[name] = monitor
        
        # 添加全局回调
        monitor.add_overload_callback(self._global_overload_handler)
        monitor.add_metrics_callback(self._global_metrics_handler)
        
        logger.info(f"已注册队列监控: {name}")
        return monitor
    
    async def start_all(self, check_interval: float = 5.0):
        """启动所有监控"""
        self.is_running = True
        
        for monitor in self.monitors.values():
            await monitor.start_monitoring(check_interval)
        
        logger.info("所有队列监控已启动")
    
    async def stop_all(self):
        """停止所有监控"""
        self.is_running = False
        
        for monitor in self.monitors.values():
            await monitor.stop_monitoring()
        
        logger.info("所有队列监控已停止")
    
    def get_monitor(self, name: str) -> Optional[QueueMonitor]:
        """获取监控器"""
        return self.monitors.get(name)
    
    def get_all_metrics(self) -> Dict[str, QueueMetrics]:
        """获取所有队列指标"""
        return {
            name: monitor.get_current_metrics()
            for name, monitor in self.monitors.items()
        }
    
    def get_overloaded_queues(self) -> List[str]:
        """获取过载的队列"""
        overloaded = []
        for name, monitor in self.monitors.items():
            metrics = monitor.get_current_metrics()
            if metrics.is_overloaded:
                overloaded.append(name)
        return overloaded
    
    def add_global_callback(self, callback: Callable):
        """添加全局回调"""
        self._global_callbacks.append(callback)
    
    async def _global_overload_handler(self, metrics: QueueMetrics):
        """全局过载处理器"""
        logger.warning(f"队列过载: {metrics.name} (利用率: {metrics.utilization:.2f})")
        
        for callback in self._global_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("overload", metrics)
                else:
                    callback("overload", metrics)
            except Exception as e:
                logger.error(f"全局过载回调执行失败: {e}")
    
    async def _global_metrics_handler(self, metrics: QueueMetrics):
        """全局指标处理器"""
        for callback in self._global_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("metrics", metrics)
                else:
                    callback("metrics", metrics)
            except Exception as e:
                logger.error(f"全局指标回调执行失败: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        all_metrics = self.get_all_metrics()
        overloaded_queues = self.get_overloaded_queues()
        
        total_items = sum(m.current_size for m in all_metrics.values())
        total_capacity = sum(m.max_size for m in all_metrics.values())
        avg_utilization = sum(m.utilization for m in all_metrics.values()) / len(all_metrics) if all_metrics else 0
        
        return {
            "total_queues": len(self.monitors),
            "overloaded_queues": len(overloaded_queues),
            "overloaded_queue_names": overloaded_queues,
            "total_items": total_items,
            "total_capacity": total_capacity,
            "system_utilization": total_items / total_capacity if total_capacity > 0 else 0,
            "average_utilization": avg_utilization,
            "is_running": self.is_running
        }


# 全局队列监控管理器实例
queue_monitor_manager = QueueMonitorManager()