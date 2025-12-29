"""
高并发性能优化模块
实现智能体系统的性能优化、资源管理和负载均衡
"""

import asyncio
import time
import threading
import multiprocessing
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
import weakref
import gc
from .events import Event, EventType, EventPriority
from .async_manager import AsyncAgentManager
from .distributed_events import DistributedEventCoordinator

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import orjson as json
    JSON_FAST = True
except ImportError:
    import json
    JSON_FAST = False

class OptimizationLevel(str, Enum):
    """优化级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AGGRESSIVE = "aggressive"

class ResourceType(str, Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    DISK = "disk"

@dataclass
class PerformanceProfile:
    """性能配置文件"""
    max_concurrent_tasks: int = 100
    max_worker_threads: int = 50
    max_worker_processes: int = 4
    event_queue_size: int = 10000
    connection_pool_size: int = 20
    cache_size: int = 1000
    gc_interval: int = 60
    optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM
    enable_profiling: bool = False
    enable_memory_profiling: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_worker_threads": self.max_worker_threads,
            "max_worker_processes": self.max_worker_processes,
            "event_queue_size": self.event_queue_size,
            "connection_pool_size": self.connection_pool_size,
            "cache_size": self.cache_size,
            "gc_interval": self.gc_interval,
            "optimization_level": self.optimization_level.value,
            "enable_profiling": self.enable_profiling,
            "enable_memory_profiling": self.enable_memory_profiling
        }

@dataclass
class ResourceMetrics:
    """资源指标"""
    timestamp: datetime = field(default_factory=lambda: utc_now())
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_rss: int = 0
    memory_vms: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    network_sent: int = 0
    network_recv: int = 0
    active_threads: int = 0
    active_tasks: int = 0
    queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_rss": self.memory_rss,
            "memory_vms": self.memory_vms,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "network_sent": self.network_sent,
            "network_recv": self.network_recv,
            "active_threads": self.active_threads,
            "active_tasks": self.active_tasks,
            "queue_size": self.queue_size
        }

class AsyncTaskPool:
    """异步任务池"""
    
    def __init__(
        self,
        max_concurrent: int = 100,
        max_queue_size: int = 1000,
        enable_priority: bool = True
    ):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.enable_priority = enable_priority
        
        # 任务队列
        if enable_priority:
            self.task_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        else:
            self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # 活跃任务
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        
        # 统计
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "peak_concurrent": 0
        }
        
        # 控制
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        logger.info("异步任务池初始化", max_concurrent=max_concurrent)
    
    async def start(self):
        """启动任务池"""
        if self.running:
            return
        
        self.running = True
        
        # 启动工作者
        for i in range(min(self.max_concurrent, 10)):  # 最多10个worker
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        logger.info("异步任务池启动", workers=len(self.worker_tasks))
    
    async def stop(self):
        """停止任务池"""
        self.running = False
        
        # 取消所有工作者
        for worker in self.worker_tasks:
            worker.cancel()
        
        # 等待工作者完成
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # 取消活跃任务
        for task in self.active_tasks.values():
            task.cancel()
        
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        logger.info("异步任务池停止")
    
    async def submit(
        self,
        coro: Callable,
        priority: int = 0,
        task_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """提交任务"""
        if not task_id:
            task_id = f"task-{self.stats['submitted']}"
        
        task_data = {
            "id": task_id,
            "coro": coro,
            "kwargs": kwargs,
            "priority": priority,
            "timeout": timeout,
            "submitted_at": time.time()
        }
        
        try:
            if self.enable_priority:
                await self.task_queue.put((priority, task_data))
            else:
                await self.task_queue.put(task_data)
            
            self.stats["submitted"] += 1
            return task_id
            
        except asyncio.QueueFull:
            raise RuntimeError("任务队列已满")
    
    async def _worker(self, worker_id: str):
        """工作者协程"""
        logger.debug("工作者启动", worker_id=worker_id)
        
        while self.running:
            try:
                # 获取任务
                if self.enable_priority:
                    try:
                        priority, task_data = await asyncio.wait_for(
                            self.task_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                else:
                    try:
                        task_data = await asyncio.wait_for(
                            self.task_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                # 执行任务
                await self._execute_task(task_data)
                
                # 更新统计
                current_concurrent = len(self.active_tasks)
                if current_concurrent > self.stats["peak_concurrent"]:
                    self.stats["peak_concurrent"] = current_concurrent
                
            except Exception as e:
                logger.error("工作者异常", worker_id=worker_id, error=str(e))
        
        logger.debug("工作者停止", worker_id=worker_id)
    
    async def _execute_task(self, task_data: Dict[str, Any]):
        """执行任务"""
        task_id = task_data["id"]
        coro = task_data["coro"]
        kwargs = task_data["kwargs"]
        timeout = task_data["timeout"]
        
        start_time = time.time()
        
        try:
            # 创建任务
            if asyncio.iscoroutinefunction(coro):
                task = asyncio.create_task(coro(**kwargs))
            else:
                task = asyncio.create_task(asyncio.to_thread(coro, **kwargs))
            
            self.active_tasks[task_id] = task
            
            # 执行任务（带超时）
            if timeout:
                result = await asyncio.wait_for(task, timeout=timeout)
            else:
                result = await task
            
            # 记录成功
            execution_time = time.time() - start_time
            self.completed_tasks.append({
                "id": task_id,
                "result": str(result)[:1000],  # 限制结果长度
                "execution_time": execution_time,
                "completed_at": time.time()
            })
            
            self.stats["completed"] += 1
            
            logger.debug("任务完成", task_id=task_id, execution_time=execution_time)
            
        except asyncio.CancelledError:
            self.stats["cancelled"] += 1
            logger.debug("任务取消", task_id=task_id)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tasks.append({
                "id": task_id,
                "error": str(e),
                "execution_time": execution_time,
                "failed_at": time.time()
            })
            
            self.stats["failed"] += 1
            logger.error("任务失败", task_id=task_id, error=str(e))
            
        finally:
            # 清理
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            # 保持历史记录在合理大小
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-500:]
            if len(self.failed_tasks) > 1000:
                self.failed_tasks = self.failed_tasks[-500:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "recent_completed": len(self.completed_tasks),
            "recent_failed": len(self.failed_tasks)
        }

class ConnectionPool:
    """连接池"""
    
    def __init__(
        self,
        max_connections: int = 20,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0
    ):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        self.active_connections: Dict[str, Any] = {}
        self.idle_connections: List[Tuple[str, Any, float]] = []  # id, conn, last_used
        self.connection_counter = 0
        
        self._lock = asyncio.Lock()
        
        logger.info("连接池初始化", max_connections=max_connections)
    
    async def acquire(self, connection_factory: Callable) -> Tuple[str, Any]:
        """获取连接"""
        async with self._lock:
            # 尝试从空闲连接池获取
            current_time = time.time()
            
            # 清理过期连接
            self.idle_connections = [
                (conn_id, conn, last_used)
                for conn_id, conn, last_used in self.idle_connections
                if current_time - last_used < self.idle_timeout
            ]
            
            # 获取空闲连接
            if self.idle_connections:
                conn_id, conn, _ = self.idle_connections.pop(0)
                self.active_connections[conn_id] = conn
                return conn_id, conn
            
            # 创建新连接
            if len(self.active_connections) < self.max_connections:
                self.connection_counter += 1
                conn_id = f"conn-{self.connection_counter}"
                
                try:
                    conn = await asyncio.wait_for(
                        connection_factory(), 
                        timeout=self.connection_timeout
                    )
                    self.active_connections[conn_id] = conn
                    return conn_id, conn
                    
                except Exception as e:
                    logger.error("创建连接失败", error=str(e))
                    raise
            
            # 连接池已满
            raise RuntimeError("连接池已满")
    
    async def release(self, conn_id: str):
        """释放连接"""
        async with self._lock:
            if conn_id in self.active_connections:
                conn = self.active_connections.pop(conn_id)
                self.idle_connections.append((conn_id, conn, time.time()))
    
    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            # 关闭活跃连接
            for conn in self.active_connections.values():
                try:
                    if hasattr(conn, 'close'):
                        await conn.close()
                except Exception as e:
                    logger.error("关闭连接失败", error=str(e))
            
            # 关闭空闲连接
            for _, conn, _ in self.idle_connections:
                try:
                    if hasattr(conn, 'close'):
                        await conn.close()
                except Exception as e:
                    logger.error("关闭连接失败", error=str(e))
            
            self.active_connections.clear()
            self.idle_connections.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "active_connections": len(self.active_connections),
            "idle_connections": len(self.idle_connections),
            "max_connections": self.max_connections,
            "utilization": len(self.active_connections) / self.max_connections
        }

class MemoryCache:
    """内存缓存"""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: float = 3600.0,  # 1小时
        enable_lru: bool = True
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.enable_lru = enable_lru
        
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: List[str] = []  # LRU tracking
        
        self._lock = asyncio.Lock()
        
        # 统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }
        
        logger.info("内存缓存初始化", max_size=max_size, ttl=ttl)
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # 检查过期
                if time.time() - timestamp > self.ttl:
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.stats["misses"] += 1
                    return None
                
                # 更新LRU
                if self.enable_lru:
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                
                self.stats["hits"] += 1
                return value
            
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any):
        """设置缓存项"""
        async with self._lock:
            current_time = time.time()
            
            # 检查是否需要清理空间
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_one()
            
            # 设置值
            self.cache[key] = (value, current_time)
            
            # 更新LRU
            if self.enable_lru:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            self.stats["sets"] += 1
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False
    
    async def _evict_one(self):
        """淘汰一个缓存项"""
        if not self.cache:
            return
        
        if self.enable_lru and self.access_order:
            # LRU淘汰
            key_to_evict = self.access_order.pop(0)
        else:
            # 随机淘汰
            key_to_evict = next(iter(self.cache))
        
        if key_to_evict in self.cache:
            del self.cache[key_to_evict]
            self.stats["evictions"] += 1
    
    async def clear(self):
        """清空缓存"""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "utilization": len(self.cache) / self.max_size
        }

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 1000
        
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        logger.info("资源监控器初始化", interval=interval)
    
    async def start(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("资源监控启动")
    
    async def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                raise
        logger.info("资源监控停止")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录大小
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history // 2:]
                
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error("资源监控异常", error=str(e))
                await asyncio.sleep(self.interval)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """收集资源指标"""
        metrics = ResourceMetrics()
        
        try:
            import psutil
            
            # CPU使用率
            metrics.cpu_usage = psutil.cpu_percent()
            
            # 内存信息
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.percent
            
            # 进程内存信息
            process = psutil.Process()
            process_memory = process.memory_info()
            metrics.memory_rss = process_memory.rss
            metrics.memory_vms = process_memory.vms
            
            # 网络信息
            network = psutil.net_io_counters()
            metrics.network_sent = network.bytes_sent
            metrics.network_recv = network.bytes_recv
            
            # 线程信息
            metrics.active_threads = process.num_threads()
            
            # 任务信息
            current_task = asyncio.current_task()
            if current_task:
                all_tasks = asyncio.all_tasks()
                metrics.active_tasks = len(all_tasks)
            
        except Exception as e:
            logger.error("收集资源指标失败", error=str(e))
        
        return metrics
    
    def get_latest_metrics(self) -> Optional[ResourceMetrics]:
        """获取最新指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        cutoff_time = utc_now() - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # 计算统计信息
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": recent_metrics[-1].cpu_usage
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": recent_metrics[-1].memory_usage,
                "rss_mb": recent_metrics[-1].memory_rss / (1024 * 1024)
            },
            "tasks": {
                "active": recent_metrics[-1].active_tasks,
                "threads": recent_metrics[-1].active_threads
            }
        }

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.current_loads: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = {}
        
        self.balance_strategies = {
            "round_robin": self._round_robin,
            "least_connections": self._least_connections,
            "least_response_time": self._least_response_time,
            "weighted_random": self._weighted_random
        }
        
        self.current_strategy = "least_connections"
        self.round_robin_index = 0
        
        logger.info("负载均衡器初始化")
    
    def add_worker(
        self,
        worker_id: str,
        weight: float = 1.0,
        max_connections: int = 100
    ):
        """添加工作者"""
        self.workers[worker_id] = {
            "weight": weight,
            "max_connections": max_connections,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "last_used": time.time()
        }
        self.current_loads[worker_id] = 0.0
        self.request_counts[worker_id] = 0
        
        logger.info("添加工作者", worker_id=worker_id, weight=weight)
    
    def remove_worker(self, worker_id: str):
        """移除工作者"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.current_loads[worker_id]
            del self.request_counts[worker_id]
            logger.info("移除工作者", worker_id=worker_id)
    
    def select_worker(self) -> Optional[str]:
        """选择工作者"""
        if not self.workers:
            return None
        
        strategy_func = self.balance_strategies.get(self.current_strategy)
        if strategy_func:
            return strategy_func()
        
        return self._round_robin()
    
    def _round_robin(self) -> str:
        """轮询策略"""
        worker_ids = list(self.workers.keys())
        if not worker_ids:
            return None
        
        selected = worker_ids[self.round_robin_index % len(worker_ids)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections(self) -> str:
        """最少连接策略"""
        if not self.workers:
            return None
        
        return min(self.current_loads.keys(), key=lambda w: self.current_loads[w])
    
    def _least_response_time(self) -> str:
        """最短响应时间策略"""
        if not self.workers:
            return None
        
        return min(
            self.workers.keys(),
            key=lambda w: self.workers[w]["avg_response_time"]
        )
    
    def _weighted_random(self) -> str:
        """加权随机策略"""
        import random
        
        if not self.workers:
            return None
        
        weights = [self.workers[w]["weight"] for w in self.workers.keys()]
        worker_ids = list(self.workers.keys())
        
        return random.choices(worker_ids, weights=weights)[0]
    
    def update_worker_load(self, worker_id: str, load: float):
        """更新工作者负载"""
        if worker_id in self.current_loads:
            self.current_loads[worker_id] = load
    
    def update_worker_stats(
        self,
        worker_id: str,
        response_time: float,
        success: bool
    ):
        """更新工作者统计"""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        
        # 更新响应时间（移动平均）
        alpha = 0.1  # 平滑因子
        worker["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * worker["avg_response_time"]
        )
        
        # 更新请求计数
        self.request_counts[worker_id] += 1
        
        # 更新最后使用时间
        worker["last_used"] = time.time()
        
        logger.debug(
            "更新工作者统计",
            worker_id=worker_id,
            response_time=response_time,
            success=success
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "workers": dict(self.workers),
            "current_loads": dict(self.current_loads),
            "request_counts": dict(self.request_counts),
            "strategy": self.current_strategy,
            "total_workers": len(self.workers),
            "total_requests": sum(self.request_counts.values())
        }

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(
        self,
        profile: PerformanceProfile,
        agent_manager: Optional[AsyncAgentManager] = None,
        distributed_coordinator: Optional[DistributedEventCoordinator] = None
    ):
        self.profile = profile
        self.agent_manager = agent_manager
        self.distributed_coordinator = distributed_coordinator
        
        # 核心组件
        self.task_pool = AsyncTaskPool(
            max_concurrent=profile.max_concurrent_tasks,
            max_queue_size=profile.event_queue_size
        )
        
        self.connection_pool = ConnectionPool(
            max_connections=profile.connection_pool_size
        )
        
        self.cache = MemoryCache(
            max_size=profile.cache_size
        )
        
        self.resource_monitor = ResourceMonitor()
        self.load_balancer = LoadBalancer()
        
        # 线程池和进程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=profile.max_worker_threads
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=profile.max_worker_processes
        )
        
        # 优化控制
        self.gc_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # 性能统计
        self.performance_stats = {
            "optimizations_applied": 0,
            "gc_collections": 0,
            "cache_optimizations": 0,
            "load_balancing_decisions": 0
        }
        
        logger.info("性能优化器初始化", profile=profile.to_dict())
    
    async def start(self):
        """启动优化器"""
        logger.info("启动性能优化器")
        
        # 启动核心组件
        await self.task_pool.start()
        await self.resource_monitor.start()
        
        # 设置事件循环策略
        if UVLOOP_AVAILABLE and self.profile.optimization_level in [
            OptimizationLevel.HIGH, OptimizationLevel.AGGRESSIVE
        ]:
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                logger.info("启用uvloop优化")
            except Exception as e:
                logger.warning("无法启用uvloop", error=str(e))
        
        # 启动后台任务
        if self.profile.gc_interval > 0:
            self.gc_task = asyncio.create_task(self._gc_loop())
        
        self.optimization_task = asyncio.create_task(self._optimization_loop())
    
    async def stop(self):
        """停止优化器"""
        logger.info("停止性能优化器")
        
        # 停止后台任务
        if self.gc_task:
            self.gc_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        # 停止核心组件
        await self.task_pool.stop()
        await self.resource_monitor.stop()
        await self.connection_pool.close_all()
        
        # 关闭执行器
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
    
    async def _gc_loop(self):
        """垃圾回收循环"""
        while True:
            try:
                await asyncio.sleep(self.profile.gc_interval)
                
                # 执行垃圾回收
                before_objects = len(gc.get_objects())
                collected = gc.collect()
                after_objects = len(gc.get_objects())
                
                self.performance_stats["gc_collections"] += 1
                
                logger.debug(
                    "垃圾回收完成",
                    collected=collected,
                    objects_before=before_objects,
                    objects_after=after_objects
                )
                
            except Exception as e:
                logger.error("垃圾回收失败", error=str(e))
    
    async def _optimization_loop(self):
        """优化循环"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒优化一次
                
                await self._apply_optimizations()
                
            except Exception as e:
                logger.error("优化循环失败", error=str(e))
    
    async def _apply_optimizations(self):
        """应用优化"""
        metrics = self.resource_monitor.get_latest_metrics()
        if not metrics:
            return
        
        optimizations_applied = 0
        
        # CPU优化
        if metrics.cpu_usage > 80:
            await self._optimize_cpu_usage()
            optimizations_applied += 1
        
        # 内存优化
        if metrics.memory_usage > 85:
            await self._optimize_memory_usage()
            optimizations_applied += 1
        
        # 缓存优化
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 0.5:
            await self._optimize_cache()
            optimizations_applied += 1
        
        # 负载均衡优化
        if self.agent_manager:
            await self._optimize_load_balancing()
            optimizations_applied += 1
        
        if optimizations_applied > 0:
            self.performance_stats["optimizations_applied"] += optimizations_applied
            logger.info("应用性能优化", count=optimizations_applied)
    
    async def _optimize_cpu_usage(self):
        """优化CPU使用"""
        logger.debug("优化CPU使用")
        
        # 减少并发任务数
        if self.task_pool.max_concurrent > 50:
            self.task_pool.max_concurrent = max(50, self.task_pool.max_concurrent - 10)
        
        # 调整工作者数量
        active_workers = len(self.task_pool.worker_tasks)
        if active_workers > 5:
            # 取消一些工作者
            for _ in range(min(2, active_workers - 3)):
                if self.task_pool.worker_tasks:
                    worker = self.task_pool.worker_tasks.pop()
                    worker.cancel()
    
    async def _optimize_memory_usage(self):
        """优化内存使用"""
        logger.debug("优化内存使用")
        
        # 清理缓存
        cache_size = len(self.cache.cache)
        if cache_size > self.profile.cache_size * 0.8:
            # 清理最老的20%
            items_to_remove = int(cache_size * 0.2)
            keys_to_remove = list(self.cache.cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                await self.cache.delete(key)
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理弱引用
        import weakref
        weakref.getweakrefs(self)
    
    async def _optimize_cache(self):
        """优化缓存"""
        logger.debug("优化缓存策略")
        
        # 增加缓存大小
        current_size = self.cache.max_size
        if current_size < self.profile.cache_size * 2:
            self.cache.max_size = min(
                self.profile.cache_size * 2,
                current_size + 200
            )
        
        self.performance_stats["cache_optimizations"] += 1
    
    async def _optimize_load_balancing(self):
        """优化负载均衡"""
        logger.debug("优化负载均衡")
        
        if not self.agent_manager:
            return
        
        try:
            # 获取所有智能体
            agents = await self.agent_manager.list_agents()
            
            # 更新负载均衡器
            for agent in agents:
                agent_id = agent.get("id", "")
                if agent_id:
                    # 计算负载（基于任务数）
                    load = agent.get("active_tasks", 0) / 100.0  # 归一化
                    self.load_balancer.update_worker_load(agent_id, load)
            
            self.performance_stats["load_balancing_decisions"] += 1
            
        except Exception as e:
            logger.error("负载均衡优化失败", error=str(e))
    
    async def submit_task(
        self,
        coro: Callable,
        priority: int = 0,
        use_thread_pool: bool = False,
        use_process_pool: bool = False,
        **kwargs
    ) -> str:
        """提交任务"""
        if use_process_pool:
            # 使用进程池
            future = self.process_pool.submit(coro, **kwargs)
            task_id = f"process-{id(future)}"
            return task_id
        elif use_thread_pool:
            # 使用线程池
            future = self.thread_pool.submit(coro, **kwargs)
            task_id = f"thread-{id(future)}"
            return task_id
        else:
            # 使用异步任务池
            return await self.task_pool.submit(coro, priority=priority, **kwargs)
    
    def get_performance_overview(self) -> Dict[str, Any]:
        """获取性能总览"""
        return {
            "profile": self.profile.to_dict(),
            "stats": self.performance_stats,
            "task_pool": self.task_pool.get_stats(),
            "connection_pool": self.connection_pool.get_stats(),
            "cache": self.cache.get_stats(),
            "load_balancer": self.load_balancer.get_stats(),
            "resource_metrics": self.resource_monitor.get_metrics_summary(),
            "thread_pool_active": self.thread_pool._threads,
            "process_pool_active": len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
        }
