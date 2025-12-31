"""
企业级流控机制
为高吞吐量场景实现背压机制和流量控制
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory

from src.core.utils.async_utils import create_task_with_logging
from collections import deque, defaultdict
from .enterprise_config import get_config_manager

from src.core.logging import get_logger
logger = get_logger(__name__)

class BackpressureStrategy(str, Enum):
    """背压策略"""
    QUEUE_SIZE = "queue_size"       # 队列大小控制
    THROUGHPUT = "throughput"       # 吞吐量控制
    LATENCY = "latency"            # 延迟控制
    RESOURCE = "resource"          # 资源使用率控制
    ADAPTIVE = "adaptive"          # 自适应控制

class DropPolicy(str, Enum):
    """丢弃策略"""
    OLDEST = "oldest"             # 丢弃最旧的任务
    NEWEST = "newest"             # 丢弃最新的任务
    RANDOM = "random"             # 随机丢弃
    PRIORITY = "priority"         # 按优先级丢弃

@dataclass
class FlowControlMetrics:
    """流控指标"""
    timestamp: datetime = field(default_factory=utc_factory)
    queue_size: int = 0
    throughput: float = 0.0  # 任务/秒
    avg_latency: float = 0.0  # 毫秒
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 1.0
    backpressure_triggered: bool = False
    dropped_tasks: int = 0
    throttled_tasks: int = 0

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    priority: int = 0
    created_at: datetime = field(default_factory=utc_factory)
    deadline: Optional[datetime] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BackpressureController(ABC):
    """背压控制器抽象类"""
    
    @abstractmethod
    async def should_apply_backpressure(self, metrics: FlowControlMetrics) -> bool:
        """判断是否应该应用背压"""
        ...
    
    @abstractmethod
    async def calculate_throttle_rate(self, metrics: FlowControlMetrics) -> float:
        """计算限流比例 (0-1)"""
        ...

class QueueBasedBackpressure(BackpressureController):
    """基于队列大小的背压控制"""
    
    def __init__(self, threshold: int = None):
        self.threshold = threshold or get_config_manager().get_int('FLOW_CONTROL_QUEUE_SIZE_THRESHOLD', 1000)
    
    async def should_apply_backpressure(self, metrics: FlowControlMetrics) -> bool:
        return metrics.queue_size > self.threshold
    
    async def calculate_throttle_rate(self, metrics: FlowControlMetrics) -> float:
        if metrics.queue_size <= self.threshold:
            return 0.0
        
        # 基于队列大小计算限流比例
        overflow_ratio = (metrics.queue_size - self.threshold) / self.threshold
        return min(overflow_ratio, 1.0)

class ThroughputBasedBackpressure(BackpressureController):
    """基于吞吐量的背压控制"""
    
    def __init__(self, max_throughput: float = 100.0):
        self.max_throughput = max_throughput
    
    async def should_apply_backpressure(self, metrics: FlowControlMetrics) -> bool:
        return metrics.throughput > self.max_throughput
    
    async def calculate_throttle_rate(self, metrics: FlowControlMetrics) -> float:
        if metrics.throughput <= self.max_throughput:
            return 0.0
        
        excess_ratio = (metrics.throughput - self.max_throughput) / self.max_throughput
        return min(excess_ratio * 0.5, 0.8)  # 最多限制80%

class AdaptiveBackpressure(BackpressureController):
    """自适应背压控制"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.queue_controller = QueueBasedBackpressure()
        self.throughput_controller = ThroughputBasedBackpressure()
        
    async def should_apply_backpressure(self, metrics: FlowControlMetrics) -> bool:
        # 综合多个指标判断
        queue_pressure = await self.queue_controller.should_apply_backpressure(metrics)
        throughput_pressure = await self.throughput_controller.should_apply_backpressure(metrics)
        
        # 资源使用率过高
        resource_pressure = (metrics.cpu_usage > 0.9 or metrics.memory_usage > 0.85)
        
        # 延迟过高
        latency_pressure = metrics.avg_latency > 5000  # 5秒
        
        # 成功率下降
        quality_pressure = metrics.success_rate < 0.8
        
        return any([queue_pressure, throughput_pressure, resource_pressure, 
                   latency_pressure, quality_pressure])
    
    async def calculate_throttle_rate(self, metrics: FlowControlMetrics) -> float:
        rates = []
        
        # 获取各种控制器的限流建议
        if await self.queue_controller.should_apply_backpressure(metrics):
            rates.append(await self.queue_controller.calculate_throttle_rate(metrics))
        
        if await self.throughput_controller.should_apply_backpressure(metrics):
            rates.append(await self.throughput_controller.calculate_throttle_rate(metrics))
        
        # 资源压力
        if metrics.cpu_usage > 0.9:
            rates.append((metrics.cpu_usage - 0.9) / 0.1 * 0.6)
        
        if metrics.memory_usage > 0.85:
            rates.append((metrics.memory_usage - 0.85) / 0.15 * 0.5)
        
        # 延迟压力
        if metrics.avg_latency > 5000:
            rates.append(min((metrics.avg_latency - 5000) / 10000, 0.7))
        
        # 质量压力
        if metrics.success_rate < 0.8:
            rates.append((0.8 - metrics.success_rate) / 0.8 * 0.4)
        
        return max(rates) if rates else 0.0

class FlowController:
    """流量控制器"""
    
    def __init__(self, 
                 strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE,
                 drop_policy: DropPolicy = DropPolicy.OLDEST,
                 max_queue_size: int = None):
        
        self.config = get_config_manager()
        self.strategy = strategy
        self.drop_policy = DropPolicy(drop_policy)
        self.max_queue_size = max_queue_size or self.config.get_int('FLOW_CONTROL_QUEUE_SIZE_THRESHOLD', 1000)
        
        # 任务队列
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queues: Dict[int, deque] = defaultdict(deque)
        
        # 指标收集
        self.metrics_history: deque = deque(maxlen=100)
        self.current_metrics = FlowControlMetrics()
        
        # 背压控制器
        self.backpressure_controller = self._create_backpressure_controller()
        
        # 统计信息
        self.total_tasks = 0
        self.dropped_tasks = 0
        self.throttled_tasks = 0
        self.completion_times: deque = deque(maxlen=1000)
        
        # 运行状态
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        logger.info(f"Flow controller initialized with strategy: {strategy}")
    
    def _create_backpressure_controller(self) -> BackpressureController:
        """创建背压控制器"""
        if self.strategy == BackpressureStrategy.QUEUE_SIZE:
            return QueueBasedBackpressure()
        elif self.strategy == BackpressureStrategy.THROUGHPUT:
            return ThroughputBasedBackpressure()
        elif self.strategy == BackpressureStrategy.ADAPTIVE:
            return AdaptiveBackpressure()
        else:
            return AdaptiveBackpressure()  # 默认使用自适应
    
    async def start(self):
        """启动流控器"""
        if self.running:
            return
        
        self.running = True
        
        # 启动指标收集任务
        metrics_task = create_task_with_logging(self._collect_metrics())
        self.worker_tasks.append(metrics_task)
        
        logger.info("Flow controller started")
    
    async def stop(self):
        """停止流控器"""
        self.running = False
        
        # 取消所有工作任务
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Flow controller stopped")
    
    async def submit_task(self, 
                         task_id: str,
                         task_data: Any,
                         priority: int = 0,
                         deadline: Optional[datetime] = None) -> bool:
        """提交任务到流控队列"""
        task_info = TaskInfo(
            task_id=task_id,
            priority=priority,
            created_at=utc_now(),
            deadline=deadline,
            metadata={'data': task_data}
        )
        
        # 检查是否应该应用背压
        if await self.backpressure_controller.should_apply_backpressure(self.current_metrics):
            throttle_rate = await self.backpressure_controller.calculate_throttle_rate(self.current_metrics)
            
            # 根据限流比例决定是否丢弃任务
            import random
            if random.random() < throttle_rate:
                await self._handle_task_drop(task_info, "backpressure_throttling")
                return False
        
        # 检查队列容量
        if self.task_queue.qsize() >= self.max_queue_size:
            await self._handle_queue_overflow(task_info)
            return False
        
        # 将任务加入队列
        await self.task_queue.put(task_info)
        self.total_tasks += 1
        
        return True
    
    async def get_task(self) -> Optional[TaskInfo]:
        """从队列获取任务"""
        try:
            # 等待任务，但添加超时避免无限等待
            task_info = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            return task_info
        except asyncio.TimeoutError:
            return None
    
    async def complete_task(self, task_id: str, success: bool = True, execution_time: float = 0.0):
        """标记任务完成"""
        if execution_time > 0:
            self.completion_times.append(execution_time)
        
        # 更新成功率
        if hasattr(self, '_recent_results'):
            if not hasattr(self, '_recent_results'):
                self._recent_results = deque(maxlen=100)
        else:
            self._recent_results = deque(maxlen=100)
        
        self._recent_results.append(success)
    
    async def _handle_queue_overflow(self, new_task: TaskInfo):
        """处理队列溢出"""
        if self.drop_policy == DropPolicy.NEWEST:
            # 丢弃新任务
            await self._handle_task_drop(new_task, "queue_overflow_newest")
        elif self.drop_policy == DropPolicy.OLDEST:
            # 丢弃最旧的任务
            try:
                oldest_task = self.task_queue.get_nowait()
                await self._handle_task_drop(oldest_task, "queue_overflow_oldest")
                await self.task_queue.put(new_task)
            except asyncio.QueueEmpty:
                await self._handle_task_drop(new_task, "queue_overflow_empty")
        elif self.drop_policy == DropPolicy.PRIORITY:
            # 基于优先级丢弃
            await self._handle_priority_based_drop(new_task)
        else:
            # 随机丢弃
            import random
            if random.random() < 0.5:
                await self._handle_task_drop(new_task, "queue_overflow_random")
    
    async def _handle_task_drop(self, task_info: TaskInfo, reason: str):
        """处理任务丢弃"""
        self.dropped_tasks += 1
        self.current_metrics.dropped_tasks = self.dropped_tasks
        
        logger.warning(f"Task dropped: {task_info.task_id}, reason: {reason}")
    
    async def _handle_priority_based_drop(self, new_task: TaskInfo):
        """基于优先级的丢弃策略"""
        # 简化实现：如果新任务优先级更高，丢弃低优先级任务
        try:
            existing_task = self.task_queue.get_nowait()
            if new_task.priority > existing_task.priority:
                await self._handle_task_drop(existing_task, "priority_replacement")
                await self.task_queue.put(new_task)
            else:
                await self.task_queue.put(existing_task)
                await self._handle_task_drop(new_task, "priority_rejection")
        except asyncio.QueueEmpty:
            await self.task_queue.put(new_task)
    
    async def _collect_metrics(self):
        """收集流控指标"""
        while self.running:
            try:
                # 收集当前指标
                metrics = await self._calculate_current_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # 检查背压状态
                if await self.backpressure_controller.should_apply_backpressure(metrics):
                    metrics.backpressure_triggered = True
                    logger.debug(f"Backpressure triggered: queue_size={metrics.queue_size}, "
                               f"throughput={metrics.throughput:.2f}, cpu={metrics.cpu_usage:.2f}")
                
                await asyncio.sleep(1.0)  # 每秒收集一次指标
                
            except Exception as e:
                logger.error(f"Error collecting flow control metrics: {e}")
                await asyncio.sleep(5.0)
    
    async def _calculate_current_metrics(self) -> FlowControlMetrics:
        """计算当前指标"""
        metrics = FlowControlMetrics()
        
        # 队列大小
        metrics.queue_size = self.task_queue.qsize()
        
        # 计算吞吐量 (基于最近的任务完成时间)
        if len(self.completion_times) > 0:
            recent_times = list(self.completion_times)[-60:]  # 最近60个任务
            if len(recent_times) > 1:
                time_span = sum(recent_times)
                if time_span > 0:
                    metrics.throughput = len(recent_times) / (time_span / 1000)  # 任务/秒
        
        # 计算平均延迟
        if len(self.completion_times) > 0:
            metrics.avg_latency = statistics.mean(self.completion_times)
        
        # 资源使用率 (简化实现)
        try:
            import psutil
            metrics.cpu_usage = psutil.cpu_percent() / 100.0
            metrics.memory_usage = psutil.virtual_memory().percent / 100.0
        except Exception as e:
            logger.warning("获取系统资源指标失败", error=str(e))
        
        # 成功率
        if hasattr(self, '_recent_results') and len(self._recent_results) > 0:
            metrics.success_rate = sum(self._recent_results) / len(self._recent_results)
        
        # 统计信息
        metrics.dropped_tasks = self.dropped_tasks
        metrics.throttled_tasks = self.throttled_tasks
        
        return metrics
    
    def get_current_metrics(self) -> FlowControlMetrics:
        """获取当前指标"""
        return self.current_metrics
    
    def get_metrics_history(self) -> List[FlowControlMetrics]:
        """获取指标历史"""
        return list(self.metrics_history)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_tasks': self.total_tasks,
            'dropped_tasks': self.dropped_tasks,
            'throttled_tasks': self.throttled_tasks,
            'drop_rate': self.dropped_tasks / max(self.total_tasks, 1),
            'throttle_rate': self.throttled_tasks / max(self.total_tasks, 1),
            'current_queue_size': self.task_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'strategy': self.strategy,
            'drop_policy': self.drop_policy
        }

class CircuitBreaker:
    """断路器实现"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """通过断路器调用函数"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置断路器"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """成功调用后的处理"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """失败调用后的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# 全局流控实例
_flow_controller: Optional[FlowController] = None

def get_flow_controller() -> FlowController:
    """获取全局流控器实例"""
    global _flow_controller
    if _flow_controller is None:
        _flow_controller = FlowController()
    return _flow_controller

async def init_flow_controller(strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE):
    """初始化全局流控器"""
    global _flow_controller
    _flow_controller = FlowController(strategy=strategy)
    await _flow_controller.start()
