"""
背压管理器

实现流式处理的背压控制和动态流量限制，防止系统过载。
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import asyncio
import time
from enum import Enum
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import weakref

from src.core.logging import get_logger
logger = get_logger(__name__)

class ThrottleLevel(str, Enum):
    """限流级别"""
    NONE = "none"
    LIGHT = "light" 
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"

class PressureSource(str, Enum):
    """压力来源"""
    BUFFER_OVERFLOW = "buffer_overflow"
    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CONNECTION_LIMIT = "connection_limit"

@dataclass
class PressureMetrics:
    """压力指标"""
    source: PressureSource
    current_value: float
    threshold: float
    severity: float  # 0-1，1表示最严重
    timestamp: datetime = field(default_factory=utc_now)
    
    @property
    def is_over_threshold(self) -> bool:
        return self.current_value > self.threshold

@dataclass
class ThrottleAction:
    """限流动作"""
    level: ThrottleLevel
    action_type: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    applied_at: datetime = field(default_factory=utc_now)
    duration: Optional[float] = None

class BackpressureManager:
    """背压管理器"""
    
    def __init__(
        self,
        max_buffer_size: int = 1000,
        high_watermark: float = 0.8,
        critical_watermark: float = 0.95,
        check_interval: float = 1.0
    ):
        self.max_buffer_size = max_buffer_size
        self.high_watermark = high_watermark
        self.critical_watermark = critical_watermark
        self.check_interval = check_interval
        
        # 当前状态
        self.current_throttle_level = ThrottleLevel.NONE
        self.buffer_usage = 0
        self.is_monitoring = False
        
        # 压力监控
        self.pressure_metrics: Dict[PressureSource, PressureMetrics] = {}
        self.active_throttles: List[ThrottleAction] = []
        
        # 阈值配置
        self.thresholds = {
            PressureSource.BUFFER_OVERFLOW: 0.8,
            PressureSource.CPU_HIGH: 0.85,
            PressureSource.MEMORY_HIGH: 0.90,
            PressureSource.QUEUE_DEPTH: 0.75,
            PressureSource.ERROR_RATE: 0.05,  # 5%错误率
            PressureSource.CONNECTION_LIMIT: 0.9
        }
        
        # 回调函数
        self.throttle_callbacks: List[Callable] = []
        self.release_callbacks: List[Callable] = []
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._metrics_history: List[Dict[str, Any]] = []
        
    async def start_monitoring(self):
        """启动背压监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_pressure())
        logger.info("背压监控已启动")
    
    async def stop_monitoring(self):
        """停止背压监控"""
        self.is_monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                raise
            
        logger.info("背压监控已停止")
    
    async def _monitor_pressure(self):
        """持续监控系统压力"""
        while self.is_monitoring:
            try:
                # 收集压力指标
                await self._collect_pressure_metrics()
                
                # 分析压力状态
                pressure_level = self._analyze_pressure()
                
                # 应用或释放限流
                await self._apply_throttle_if_needed(pressure_level)
                
                # 记录历史
                self._record_metrics_history()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"背压监控错误: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_pressure_metrics(self):
        """收集压力指标"""
        # 缓冲区使用率
        if self.max_buffer_size > 0:
            buffer_ratio = self.buffer_usage / self.max_buffer_size
            self.pressure_metrics[PressureSource.BUFFER_OVERFLOW] = PressureMetrics(
                source=PressureSource.BUFFER_OVERFLOW,
                current_value=buffer_ratio,
                threshold=self.thresholds[PressureSource.BUFFER_OVERFLOW],
                severity=min(1.0, buffer_ratio / self.critical_watermark)
            )
        
        # 系统资源监控
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent() / 100.0
            self.pressure_metrics[PressureSource.CPU_HIGH] = PressureMetrics(
                source=PressureSource.CPU_HIGH,
                current_value=cpu_percent,
                threshold=self.thresholds[PressureSource.CPU_HIGH],
                severity=max(0, (cpu_percent - 0.5) / 0.5)  # 50%以上开始计算严重程度
            )
            
            # 内存使用率
            memory_percent = psutil.virtual_memory().percent / 100.0
            self.pressure_metrics[PressureSource.MEMORY_HIGH] = PressureMetrics(
                source=PressureSource.MEMORY_HIGH,
                current_value=memory_percent,
                threshold=self.thresholds[PressureSource.MEMORY_HIGH],
                severity=max(0, (memory_percent - 0.6) / 0.4)  # 60%以上开始计算严重程度
            )
            
        except ImportError:
            # 无psutil则跳过系统指标
            self.pressure_metrics[PressureSource.CPU_HIGH] = PressureMetrics(
                source=PressureSource.CPU_HIGH,
                current_value=0.0,
                threshold=self.thresholds[PressureSource.CPU_HIGH],
                severity=0.0
            )
            self.pressure_metrics[PressureSource.MEMORY_HIGH] = PressureMetrics(
                source=PressureSource.MEMORY_HIGH,
                current_value=0.0,
                threshold=self.thresholds[PressureSource.MEMORY_HIGH],
                severity=0.0
            )
    
    def _analyze_pressure(self) -> ThrottleLevel:
        """分析当前压力水平"""
        max_severity = 0
        over_threshold_count = 0
        
        for metric in self.pressure_metrics.values():
            if metric.is_over_threshold:
                over_threshold_count += 1
                max_severity = max(max_severity, metric.severity)
        
        # 根据严重程度和超阈值指标数量确定限流级别
        if over_threshold_count == 0:
            return ThrottleLevel.NONE
        elif max_severity < 0.3:
            return ThrottleLevel.LIGHT
        elif max_severity < 0.6:
            return ThrottleLevel.MODERATE
        elif max_severity < 0.9:
            return ThrottleLevel.HEAVY
        else:
            return ThrottleLevel.SEVERE
    
    async def _apply_throttle_if_needed(self, pressure_level: ThrottleLevel):
        """根据压力水平应用限流"""
        if pressure_level != self.current_throttle_level:
            if pressure_level == ThrottleLevel.NONE:
                await self._release_throttling()
            else:
                await self._apply_throttling(pressure_level)
            
            self.current_throttle_level = pressure_level
    
    async def _apply_throttling(self, level: ThrottleLevel):
        """应用限流"""
        logger.warning(f"应用背压限流: {level.value}")
        
        # 清除旧的限流动作
        self.active_throttles.clear()
        
        # 根据级别应用不同的限流策略
        if level == ThrottleLevel.LIGHT:
            await self._apply_light_throttling()
        elif level == ThrottleLevel.MODERATE:
            await self._apply_moderate_throttling()
        elif level == ThrottleLevel.HEAVY:
            await self._apply_heavy_throttling()
        elif level == ThrottleLevel.SEVERE:
            await self._apply_severe_throttling()
        
        # 执行限流回调
        for callback in self.throttle_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(level)
                else:
                    callback(level)
            except Exception as e:
                logger.error(f"限流回调执行失败: {e}")
    
    async def _apply_light_throttling(self):
        """轻度限流"""
        actions = [
            ThrottleAction(
                level=ThrottleLevel.LIGHT,
                action_type="delay_processing",
                target="stream_processor",
                parameters={"delay_ms": 10}
            ),
            ThrottleAction(
                level=ThrottleLevel.LIGHT,
                action_type="reduce_batch_size",
                target="batch_processor", 
                parameters={"reduction_factor": 0.9}
            )
        ]
        self.active_throttles.extend(actions)
    
    async def _apply_moderate_throttling(self):
        """中度限流"""
        actions = [
            ThrottleAction(
                level=ThrottleLevel.MODERATE,
                action_type="delay_processing",
                target="stream_processor",
                parameters={"delay_ms": 50}
            ),
            ThrottleAction(
                level=ThrottleLevel.MODERATE,
                action_type="reduce_batch_size",
                target="batch_processor",
                parameters={"reduction_factor": 0.7}
            ),
            ThrottleAction(
                level=ThrottleLevel.MODERATE,
                action_type="reject_low_priority",
                target="task_scheduler",
                parameters={"min_priority": 3}
            )
        ]
        self.active_throttles.extend(actions)
    
    async def _apply_heavy_throttling(self):
        """重度限流"""
        actions = [
            ThrottleAction(
                level=ThrottleLevel.HEAVY,
                action_type="delay_processing",
                target="stream_processor",
                parameters={"delay_ms": 100}
            ),
            ThrottleAction(
                level=ThrottleLevel.HEAVY,
                action_type="reduce_batch_size",
                target="batch_processor",
                parameters={"reduction_factor": 0.5}
            ),
            ThrottleAction(
                level=ThrottleLevel.HEAVY,
                action_type="reject_low_priority",
                target="task_scheduler",
                parameters={"min_priority": 5}
            ),
            ThrottleAction(
                level=ThrottleLevel.HEAVY,
                action_type="limit_connections",
                target="stream_server",
                parameters={"max_connections": 50}
            )
        ]
        self.active_throttles.extend(actions)
    
    async def _apply_severe_throttling(self):
        """严重限流"""
        actions = [
            ThrottleAction(
                level=ThrottleLevel.SEVERE,
                action_type="delay_processing",
                target="stream_processor",
                parameters={"delay_ms": 500}
            ),
            ThrottleAction(
                level=ThrottleLevel.SEVERE,
                action_type="pause_batch_processing",
                target="batch_processor",
                parameters={"pause_duration": 30}
            ),
            ThrottleAction(
                level=ThrottleLevel.SEVERE,
                action_type="reject_new_requests",
                target="api_gateway",
                parameters={"reject_probability": 0.5}
            ),
            ThrottleAction(
                level=ThrottleLevel.SEVERE,
                action_type="emergency_gc",
                target="system",
                parameters={}
            )
        ]
        self.active_throttles.extend(actions)
    
    async def _release_throttling(self):
        """释放限流"""
        if self.current_throttle_level != ThrottleLevel.NONE:
            logger.info("释放背压限流")
            
            self.active_throttles.clear()
            
            # 执行释放回调
            for callback in self.release_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"释放限流回调执行失败: {e}")
    
    def _record_metrics_history(self):
        """记录指标历史"""
        metrics_snapshot = {
            "timestamp": utc_now().isoformat(),
            "throttle_level": self.current_throttle_level.value,
            "buffer_usage": self.buffer_usage,
            "pressure_metrics": {
                source.value: {
                    "value": metric.current_value,
                    "threshold": metric.threshold,
                    "severity": metric.severity,
                    "over_threshold": metric.is_over_threshold
                }
                for source, metric in self.pressure_metrics.items()
            },
            "active_throttles": len(self.active_throttles)
        }
        
        self._metrics_history.append(metrics_snapshot)
        
        # 限制历史记录大小
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-500:]
    
    def update_buffer_usage(self, usage: int):
        """更新缓冲区使用情况"""
        self.buffer_usage = usage
    
    def add_throttle_callback(self, callback: Callable):
        """添加限流回调"""
        self.throttle_callbacks.append(callback)
    
    def add_release_callback(self, callback: Callable):
        """添加释放限流回调"""
        self.release_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "throttle_level": self.current_throttle_level.value,
            "buffer_usage": self.buffer_usage,
            "buffer_usage_ratio": self.buffer_usage / self.max_buffer_size if self.max_buffer_size > 0 else 0,
            "is_monitoring": self.is_monitoring,
            "pressure_metrics": {
                source.value: {
                    "current_value": metric.current_value,
                    "threshold": metric.threshold,
                    "severity": metric.severity,
                    "over_threshold": metric.is_over_threshold
                }
                for source, metric in self.pressure_metrics.items()
            },
            "active_throttles": [
                {
                    "level": action.level.value,
                    "action_type": action.action_type,
                    "target": action.target,
                    "parameters": action.parameters,
                    "applied_at": action.applied_at.isoformat()
                }
                for action in self.active_throttles
            ]
        }
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指标历史"""
        return self._metrics_history[-limit:] if limit else self._metrics_history
    
    def configure_threshold(self, source: PressureSource, threshold: float):
        """配置阈值"""
        self.thresholds[source] = threshold
        logger.info(f"已更新 {source.value} 阈值为 {threshold}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """获取所有阈值配置"""
        return {source.value: threshold for source, threshold in self.thresholds.items()}

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, rate: int, per: float = 1.0, burst: Optional[int] = None):
        self.rate = rate  # 每个时间窗口允许的请求数
        self.per = per    # 时间窗口长度（秒）
        self.burst = burst or rate  # 突发请求限制
        
        self.allowance = float(rate)
        self.last_check = time.time()
        self._lock = asyncio.Lock()
        
        # 统计信息
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0
    
    async def acquire(self, tokens: int = 1) -> bool:
        """获取令牌"""
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            
            # 补充令牌
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.burst:
                self.allowance = self.burst
            
            self.total_requests += 1
            
            # 检查是否有足够的令牌
            if self.allowance >= tokens:
                self.allowance -= tokens
                self.total_allowed += 1
                return True
            else:
                self.total_rejected += 1
                return False
    
    async def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """等待令牌可用"""
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # 计算等待时间
            wait_time = tokens / (self.rate / self.per)
            wait_time = min(wait_time, 1.0)  # 最大等待1秒
            
            await asyncio.sleep(wait_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "rate": self.rate,
            "per": self.per,
            "burst": self.burst,
            "current_allowance": self.allowance,
            "total_requests": self.total_requests,
            "total_allowed": self.total_allowed,
            "total_rejected": self.total_rejected,
            "rejection_rate": self.total_rejected / self.total_requests if self.total_requests > 0 else 0
        }
    
    def reset(self):
        """重置限制器"""
        self.allowance = float(self.rate)
        self.last_check = time.time()
        self.total_requests = 0
        self.total_allowed = 0
        self.total_rejected = 0

class CircuitBreaker:
    """熔断器"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        async with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await self._on_success()
                return result
            except self.expected_exception as e:
                await self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    async def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    async def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }
