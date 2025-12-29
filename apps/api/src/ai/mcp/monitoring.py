"""MCP性能监控和指标收集"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from collections import defaultdict, deque
import threading

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation_name: str
    execution_time: float
    success: bool
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_factory)
    server_type: Optional[str] = None
    tool_name: Optional[str] = None

@dataclass
class AggregatedMetrics:
    """聚合指标数据类"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=utc_factory)

    @property
    def success_rate(self) -> float:
        """成功率百分比"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def average_execution_time(self) -> float:
        """平均执行时间"""
        if self.total_calls == 0:
            return 0.0
        return self.total_execution_time / self.total_calls

class MCPMonitor:
    """MCP性能监控器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.aggregated_metrics: Dict[str, AggregatedMetrics] = defaultdict(AggregatedMetrics)
        self.lock = threading.Lock()
        
        # 定期清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动定期清理任务"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # 每小时清理一次
                self._cleanup_old_metrics()
        
        try:
            # 尝试获取当前运行的事件循环
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # 如果没有运行的事件循环，稍后创建
            self._cleanup_task = None

    def _cleanup_old_metrics(self):
        """清理旧的指标数据"""
        cutoff_time = utc_now() - timedelta(hours=24)
        
        with self.lock:
            # 清理历史记录中超过24小时的数据
            while (self.metrics_history and 
                   self.metrics_history[0].timestamp < cutoff_time):
                self.metrics_history.popleft()
            
            logger.info(f"Cleaned up old metrics, current history size: {len(self.metrics_history)}")

    def record_operation(
        self,
        operation_name: str,
        execution_time: float,
        success: bool,
        error_type: Optional[str] = None,
        server_type: Optional[str] = None,
        tool_name: Optional[str] = None
    ):
        """记录操作指标"""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            success=success,
            error_type=error_type,
            server_type=server_type,
            tool_name=tool_name
        )
        
        with self.lock:
            # 添加到历史记录
            self.metrics_history.append(metric)
            
            # 更新聚合指标
            key = f"{server_type}.{tool_name}" if server_type and tool_name else operation_name
            agg = self.aggregated_metrics[key]
            
            agg.total_calls += 1
            agg.total_execution_time += execution_time
            agg.last_updated = utc_now()
            
            if success:
                agg.successful_calls += 1
            else:
                agg.failed_calls += 1
                if error_type:
                    agg.error_counts[error_type] = agg.error_counts.get(error_type, 0) + 1
            
            # 更新执行时间统计
            agg.min_execution_time = min(agg.min_execution_time, execution_time)
            agg.max_execution_time = max(agg.max_execution_time, execution_time)
        
        # 记录到日志
        logger.info(
            f"Operation recorded: {operation_name}",
            extra={
                "operation_name": operation_name,
                "execution_time": execution_time,
                "success": success,
                "error_type": error_type,
                "server_type": server_type,
                "tool_name": tool_name
            }
        )

    def get_metrics_summary(
        self, 
        operation_filter: Optional[str] = None,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            # 时间过滤
            filtered_metrics = list(self.metrics_history)
            if time_window_hours:
                cutoff_time = utc_now() - timedelta(hours=time_window_hours)
                filtered_metrics = [
                    m for m in filtered_metrics 
                    if m.timestamp >= cutoff_time
                ]
            
            # 操作过滤
            if operation_filter:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if operation_filter in m.operation_name
                ]
            
            if not filtered_metrics:
                return {
                    "total_operations": 0,
                    "success_rate": 0.0,
                    "average_execution_time": 0.0,
                    "error_distribution": {},
                    "time_window_hours": time_window_hours,
                    "operation_filter": operation_filter
                }
            
            # 计算统计信息
            total_operations = len(filtered_metrics)
            successful_operations = sum(1 for m in filtered_metrics if m.success)
            total_execution_time = sum(m.execution_time for m in filtered_metrics)
            
            # 错误分布
            error_counts = defaultdict(int)
            for metric in filtered_metrics:
                if not metric.success and metric.error_type:
                    error_counts[metric.error_type] += 1
            
            # 按服务器类型分组
            server_stats = defaultdict(lambda: {"total": 0, "successful": 0, "execution_time": 0.0})
            for metric in filtered_metrics:
                if metric.server_type:
                    stats = server_stats[metric.server_type]
                    stats["total"] += 1
                    stats["execution_time"] += metric.execution_time
                    if metric.success:
                        stats["successful"] += 1
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": total_operations - successful_operations,
                "success_rate": (successful_operations / total_operations) * 100,
                "average_execution_time": total_execution_time / total_operations,
                "total_execution_time": total_execution_time,
                "error_distribution": dict(error_counts),
                "server_statistics": {
                    server: {
                        "total_calls": stats["total"],
                        "successful_calls": stats["successful"],
                        "success_rate": (stats["successful"] / stats["total"]) * 100,
                        "average_execution_time": stats["execution_time"] / stats["total"]
                    }
                    for server, stats in server_stats.items()
                },
                "time_window_hours": time_window_hours,
                "operation_filter": operation_filter,
                "generated_at": utc_now().isoformat()
            }

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取聚合指标"""
        with self.lock:
            result = {}
            for key, metrics in self.aggregated_metrics.items():
                result[key] = {
                    "total_calls": metrics.total_calls,
                    "successful_calls": metrics.successful_calls,
                    "failed_calls": metrics.failed_calls,
                    "success_rate": metrics.success_rate,
                    "average_execution_time": metrics.average_execution_time,
                    "min_execution_time": metrics.min_execution_time if metrics.min_execution_time != float('inf') else 0.0,
                    "max_execution_time": metrics.max_execution_time,
                    "error_counts": dict(metrics.error_counts),
                    "last_updated": metrics.last_updated.isoformat()
                }
            return result

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误记录"""
        with self.lock:
            error_metrics = [
                m for m in reversed(self.metrics_history)
                if not m.success
            ][:limit]
            
            return [
                {
                    "operation_name": m.operation_name,
                    "error_type": m.error_type,
                    "execution_time": m.execution_time,
                    "timestamp": m.timestamp.isoformat(),
                    "server_type": m.server_type,
                    "tool_name": m.tool_name
                }
                for m in error_metrics
            ]

    def reset_metrics(self):
        """重置所有指标"""
        with self.lock:
            self.metrics_history.clear()
            self.aggregated_metrics.clear()
        
        logger.info("All metrics have been reset")

    async def close(self):
        """关闭监控器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                raise
        
        logger.info("MCP monitor closed")

# 全局监控实例
_global_monitor: Optional[MCPMonitor] = None

def get_mcp_monitor() -> MCPMonitor:
    """获取全局MCP监控实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MCPMonitor()
    return _global_monitor

class MonitoringContextManager:
    """监控上下文管理器"""
    
    def __init__(
        self,
        operation_name: str,
        server_type: Optional[str] = None,
        tool_name: Optional[str] = None,
        monitor: Optional[MCPMonitor] = None
    ):
        self.operation_name = operation_name
        self.server_type = server_type
        self.tool_name = tool_name
        self.monitor = monitor or get_mcp_monitor()
        self.start_time: Optional[float] = None
        self.success = False
        self.error_type: Optional[str] = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
        
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.success = True
        else:
            self.success = False
            self.error_type = exc_type.__name__
        
        self.monitor.record_operation(
            operation_name=self.operation_name,
            execution_time=execution_time,
            success=self.success,
            error_type=self.error_type,
            server_type=self.server_type,
            tool_name=self.tool_name
        )

def monitor_operation(
    operation_name: str,
    server_type: Optional[str] = None,
    tool_name: Optional[str] = None
):
    """监控操作装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with MonitoringContextManager(
                operation_name=operation_name,
                server_type=server_type,
                tool_name=tool_name
            ):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

async def get_monitor_dependency() -> MCPMonitor:
    """FastAPI依赖注入：获取监控器"""
    return get_mcp_monitor()
from src.core.logging import get_logger
