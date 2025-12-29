"""
推理查询优化器 - 推理性能优化和资源管理

实现功能:
- 推理查询的解析和优化
- 推理结果的分层缓存策略
- 并发推理的任务调度
- 推理性能监控和调优
- 资源分配和负载均衡

技术栈:
- 查询计划优化
- 多级缓存架构
- 异步任务调度
- 性能监控和指标收集
"""

import asyncio
import time
import hashlib
import base64
from src.core.utils import secure_pickle as pickle
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import numpy as np
from queue import PriorityQueue
from uuid import uuid4

from src.core.logging import get_logger
logger = get_logger(__name__)

class CacheLevel(str, Enum):
    """缓存级别"""
    L1_MEMORY = "l1_memory"      # 内存缓存（最快）
    L2_REDIS = "l2_redis"        # Redis缓存（中等）
    L3_DATABASE = "l3_database"   # 数据库缓存（最慢）

class OptimizationStrategy(str, Enum):
    """优化策略"""
    CACHE_FIRST = "cache_first"           # 缓存优先
    PARALLEL_EXECUTION = "parallel"       # 并行执行
    QUERY_REWRITING = "query_rewrite"     # 查询重写
    RESOURCE_AWARE = "resource_aware"     # 资源感知
    ADAPTIVE = "adaptive"                 # 自适应

class ReasoningPriority(int, Enum):
    """推理优先级"""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=utc_now)
    expiry: Optional[datetime] = None
    size_bytes: int = 0
    hit_rate: float = 0.0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expiry is None:
            return False
        return utc_now() > self.expiry
    
    def access(self):
        """访问记录"""
        self.access_count += 1
        self.last_access = utc_now()

@dataclass
class ReasoningTask:
    """推理任务"""
    task_id: str
    query_type: str
    parameters: Dict[str, Any]
    priority: ReasoningPriority = ReasoningPriority.MEDIUM
    created_at: datetime = field(default_factory=utc_now)
    deadline: Optional[datetime] = None
    estimated_duration: float = 0.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def __lt__(self, other):
        """优先级比较（用于优先队列）"""
        return self.priority.value > other.priority.value

@dataclass
class PerformanceMetrics:
    """性能指标"""
    query_id: str
    query_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_used_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    parallel_workers: int = 1
    optimization_applied: List[str] = field(default_factory=list)
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class CacheManager:
    """分层缓存管理器"""
    
    def __init__(self, max_memory_size: int = 100 * 1024 * 1024):  # 100MB
        self.max_memory_size = max_memory_size
        self.current_size = 0
        
        # 多级缓存
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # LRU缓存
        self.l2_cache: Optional[Any] = None  # Redis客户端
        self.l3_cache: Optional[Any] = None  # 数据库连接
        
        # 缓存统计
        self.hit_stats = defaultdict(int)
        self.miss_stats = defaultdict(int)
        
        # 锁
        self._lock = threading.RLock()
    
    async def get(self, key: str, cache_levels: List[CacheLevel] = None) -> Optional[Any]:
        """从缓存获取数据"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]
        
        for level in cache_levels:
            if level == CacheLevel.L1_MEMORY:
                value = await self._get_l1(key)
                if value is not None:
                    self.hit_stats[level] += 1
                    return value
                else:
                    self.miss_stats[level] += 1
            
            elif level == CacheLevel.L2_REDIS and self.l2_cache:
                value = await self._get_l2(key)
                if value is not None:
                    self.hit_stats[level] += 1
                    # 回写到L1缓存
                    await self._set_l1(key, value)
                    return value
                else:
                    self.miss_stats[level] += 1
            
            elif level == CacheLevel.L3_DATABASE and self.l3_cache:
                value = await self._get_l3(key)
                if value is not None:
                    self.hit_stats[level] += 1
                    # 回写到上级缓存
                    await self._set_l1(key, value)
                    if self.l2_cache:
                        await self._set_l2(key, value)
                    return value
                else:
                    self.miss_stats[level] += 1
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_levels: List[CacheLevel] = None):
        """设置缓存数据"""
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY]
        
        for level in cache_levels:
            if level == CacheLevel.L1_MEMORY:
                await self._set_l1(key, value, ttl)
            elif level == CacheLevel.L2_REDIS and self.l2_cache:
                await self._set_l2(key, value, ttl)
            elif level == CacheLevel.L3_DATABASE and self.l3_cache:
                await self._set_l3(key, value, ttl)
    
    async def _get_l1(self, key: str) -> Optional[Any]:
        """L1内存缓存获取"""
        with self._lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if entry.is_expired():
                    del self.l1_cache[key]
                    self.current_size -= entry.size_bytes
                    return None
                
                entry.access()
                # LRU更新
                self.l1_cache.move_to_end(key)
                return entry.value
        return None
    
    async def _set_l1(self, key: str, value: Any, ttl: Optional[int] = None):
        """L1内存缓存设置"""
        with self._lock:
            # 序列化计算大小
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # 检查空间并清理
            while self.current_size + size_bytes > self.max_memory_size and self.l1_cache:
                # LRU淘汰
                old_key, old_entry = self.l1_cache.popitem(last=False)
                self.current_size -= old_entry.size_bytes
            
            # 设置过期时间
            expiry = None
            if ttl:
                expiry = utc_now() + timedelta(seconds=ttl)
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=utc_now(),
                expiry=expiry,
                size_bytes=size_bytes
            )
            
            self.l1_cache[key] = entry
            self.current_size += size_bytes
    
    async def _get_l2(self, key: str) -> Optional[Any]:
        """L2 Redis缓存获取"""
        if not self.l2_cache:
            return None
        try:
            raw = await self.l2_cache.get(key)
            if raw is None:
                return None
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            payload = base64.b64decode(raw)
            return pickle.loads(payload)
        except Exception as e:
            logger.warning("L2缓存读取失败", key=key, error=str(e))
            return None
    
    async def _set_l2(self, key: str, value: Any, ttl: Optional[int] = None):
        """L2 Redis缓存设置"""
        if not self.l2_cache:
            return
        try:
            payload = pickle.dumps(value)
            encoded = base64.b64encode(payload).decode("ascii")
            if ttl:
                if hasattr(self.l2_cache, "setex"):
                    await self.l2_cache.setex(key, ttl, encoded)
                else:
                    await self.l2_cache.set(key, encoded, ex=ttl)
            else:
                await self.l2_cache.set(key, encoded)
        except Exception as e:
            logger.warning("L2缓存写入失败", key=key, error=str(e))
    
    async def _get_l3(self, key: str) -> Optional[Any]:
        """L3数据库缓存获取"""
        if not self.l3_cache:
            return None
        try:
            if hasattr(self.l3_cache, "get"):
                result = self.l3_cache.get(key)
                return await result if asyncio.iscoroutine(result) else result
            from sqlalchemy import select
            from sqlalchemy.ext.asyncio import AsyncSession
            from src.ai.reasoning.models import ReasoningCacheModel

            if isinstance(self.l3_cache, AsyncSession):
                result = await self.l3_cache.execute(
                    select(ReasoningCacheModel).where(ReasoningCacheModel.cache_key == key)
                )
                cache_entry = result.scalar_one_or_none()
                if not cache_entry:
                    return None
                if cache_entry.expires_at and cache_entry.expires_at < utc_now():
                    await self.l3_cache.delete(cache_entry)
                    await self.l3_cache.commit()
                    return None
                cache_entry.hit_count = (cache_entry.hit_count or 0) + 1
                await self.l3_cache.commit()
                return cache_entry.result
            raise TypeError("L3缓存需要实现get/set接口或提供AsyncSession")
        except Exception as e:
            logger.warning("L3缓存读取失败", key=key, error=str(e))
            return None
    
    async def _set_l3(self, key: str, value: Any, ttl: Optional[int] = None):
        """L3数据库缓存设置"""
        if not self.l3_cache:
            return
        try:
            if hasattr(self.l3_cache, "set"):
                result = self.l3_cache.set(key, value, ttl=ttl)
                if asyncio.iscoroutine(result):
                    await result
                return
            from sqlalchemy import select
            from sqlalchemy.ext.asyncio import AsyncSession
            from src.ai.reasoning.models import ReasoningCacheModel

            if not isinstance(self.l3_cache, AsyncSession):
                raise TypeError("L3缓存需要实现get/set接口或提供AsyncSession")

            expires_at = utc_now() + timedelta(seconds=ttl or 3600)
            result = await self.l3_cache.execute(
                select(ReasoningCacheModel).where(ReasoningCacheModel.cache_key == key)
            )
            cache_entry = result.scalar_one_or_none()
            if cache_entry:
                cache_entry.result = value
                cache_entry.ttl_seconds = ttl or cache_entry.ttl_seconds or 3600
                cache_entry.expires_at = expires_at
            else:
                problem_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
                cache_entry = ReasoningCacheModel(
                    id=uuid4(),
                    cache_key=key,
                    problem_hash=problem_hash,
                    strategy="knowledge_graph",
                    result=value,
                    hit_count=0,
                    ttl_seconds=ttl or 3600,
                    created_at=utc_now(),
                    expires_at=expires_at,
                )
                self.l3_cache.add(cache_entry)
            await self.l3_cache.commit()
        except Exception as e:
            logger.warning("L3缓存写入失败", key=key, error=str(e))
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_hits = sum(self.hit_stats.values())
        total_misses = sum(self.miss_stats.values())
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        
        return {
            "l1_entries": len(self.l1_cache),
            "l1_size_mb": self.current_size / (1024 * 1024),
            "l1_max_size_mb": self.max_memory_size / (1024 * 1024),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "level_hits": dict(self.hit_stats),
            "level_misses": dict(self.miss_stats)
        }

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # 任务队列
        self.task_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, ReasoningTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # 执行器
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        
        # 资源监控
        self.resource_monitor = ResourceMonitor()
        
        # 统计
        self.total_scheduled = 0
        self.total_completed = 0
        self.total_failed = 0
    
    async def start(self):
        """启动调度器"""
        self.running = True
        # 启动调度循环
        asyncio.create_task(self._scheduling_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """停止调度器"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Task scheduler stopped")
    
    async def submit_task(self, task: ReasoningTask) -> str:
        """提交任务"""
        if self.task_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Task queue is full")
        
        self.task_queue.put(task)
        self.total_scheduled += 1
        logger.debug(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id
    
    async def _scheduling_loop(self):
        """调度循环"""
        while self.running:
            try:
                # 检查资源可用性
                if not self.resource_monitor.has_available_resources():
                    await asyncio.sleep(0.1)
                    continue
                
                # 获取最高优先级任务
                if not self.task_queue.empty():
                    task = self.task_queue.get_nowait()
                    
                    # 检查依赖
                    if self._check_dependencies(task):
                        await self._execute_task(task)
                    else:
                        # 重新放入队列
                        self.task_queue.put(task)
                
                await asyncio.sleep(0.01)  # 避免忙等待
                
            except Exception as e:
                logger.error(f"Scheduling error: {str(e)}")
                await asyncio.sleep(1)
    
    def _check_dependencies(self, task: ReasoningTask) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    async def _execute_task(self, task: ReasoningTask):
        """执行任务"""
        try:
            self.active_tasks[task.task_id] = task
            
            # 在线程池中执行
            future = self.executor.submit(self._run_task, task)
            
            # 异步等待结果
            result = await asyncio.wrap_future(future)
            
            # 任务完成
            self.completed_tasks[task.task_id] = result
            self.total_completed += 1
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            self.total_failed += 1
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _run_task(self, task: ReasoningTask) -> Any:
        """运行任务（在线程池中执行）"""
        # 任务执行逻辑占位符
        # 这里应该调用相应的推理引擎
        time.sleep(0.1)  # 模拟计算时间
        return {"task_id": task.task_id, "result": "completed"}
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """获取调度器统计"""
        return {
            "total_scheduled": self.total_scheduled,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "completion_rate": self.total_completed / self.total_scheduled if self.total_scheduled > 0 else 0.0,
            "success_rate": self.total_completed / (self.total_completed + self.total_failed) if (self.total_completed + self.total_failed) > 0 else 0.0
        }

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # CPU使用率阈值
        self.memory_threshold = 80.0  # 内存使用率阈值
        self.disk_threshold = 90.0  # 磁盘使用率阈值
    
    def get_system_resources(self) -> Dict[str, float]:
        """获取系统资源使用情况"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_available_gb": memory.available / (1024 ** 3),
                "disk_free_gb": disk.free / (1024 ** 3)
            }
        except Exception as e:
            logger.error(f"Failed to get system resources: {str(e)}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "memory_available_gb": 0.0,
                "disk_free_gb": 0.0
            }
    
    def has_available_resources(self) -> bool:
        """检查是否有可用资源"""
        resources = self.get_system_resources()
        
        return (
            resources["cpu_percent"] < self.cpu_threshold and
            resources["memory_percent"] < self.memory_threshold and
            resources["disk_percent"] < self.disk_threshold
        )

class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.optimization_rules = []
        self.query_statistics = defaultdict(list)
    
    def optimize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """优化查询"""
        optimized_query = query.copy()
        applied_optimizations = []
        
        # 应用优化规则
        for rule in self.optimization_rules:
            try:
                result = rule(optimized_query)
                if result:
                    optimized_query, optimization_name = result
                    applied_optimizations.append(optimization_name)
            except Exception as e:
                logger.error(f"Optimization rule failed: {str(e)}")
        
        # 记录优化统计
        query_type = query.get("type", "unknown")
        self.query_statistics[query_type].append({
            "original_complexity": self._estimate_complexity(query),
            "optimized_complexity": self._estimate_complexity(optimized_query),
            "optimizations": applied_optimizations,
            "timestamp": utc_now()
        })
        
        return optimized_query
    
    def _estimate_complexity(self, query: Dict[str, Any]) -> int:
        """估计查询复杂度"""
        # 简化的复杂度估计
        complexity = 1
        
        if "max_hops" in query:
            complexity *= query["max_hops"]
        
        if "constraints" in query:
            complexity *= len(query["constraints"])
        
        if "entities" in query:
            complexity *= len(query["entities"])
        
        return complexity
    
    def add_optimization_rule(self, rule_func: Callable):
        """添加优化规则"""
        self.optimization_rules.append(rule_func)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计"""
        stats = {}
        
        for query_type, records in self.query_statistics.items():
            if records:
                original_complexities = [r["original_complexity"] for r in records]
                optimized_complexities = [r["optimized_complexity"] for r in records]
                
                stats[query_type] = {
                    "total_queries": len(records),
                    "avg_original_complexity": np.mean(original_complexities),
                    "avg_optimized_complexity": np.mean(optimized_complexities),
                    "avg_complexity_reduction": (np.mean(original_complexities) - np.mean(optimized_complexities)) / np.mean(original_complexities) if np.mean(original_complexities) > 0 else 0.0,
                    "common_optimizations": self._get_common_optimizations(records)
                }
        
        return stats
    
    def _get_common_optimizations(self, records: List[Dict]) -> Dict[str, int]:
        """获取常用优化"""
        optimization_counts = defaultdict(int)
        for record in records:
            for opt in record["optimizations"]:
                optimization_counts[opt] += 1
        return dict(optimization_counts)

class ReasoningOptimizer:
    """推理优化器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 组件初始化
        self.cache_manager = CacheManager(
            max_memory_size=self.config.get("cache_size_mb", 100) * 1024 * 1024
        )
        self.task_scheduler = TaskScheduler(
            max_workers=self.config.get("max_workers", 4),
            max_queue_size=self.config.get("max_queue_size", 1000)
        )
        self.query_optimizer = QueryOptimizer()
        self.resource_monitor = ResourceMonitor()
        
        # 性能指标
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_strategies: Dict[str, bool] = {
            OptimizationStrategy.CACHE_FIRST: True,
            OptimizationStrategy.PARALLEL_EXECUTION: True,
            OptimizationStrategy.QUERY_REWRITING: True,
            OptimizationStrategy.RESOURCE_AWARE: True,
            OptimizationStrategy.ADAPTIVE: False
        }
        
        # 添加默认优化规则
        self._setup_default_optimizations()
    
    async def start(self):
        """启动优化器"""
        await self.task_scheduler.start()
        logger.info("Reasoning optimizer started")
    
    async def stop(self):
        """停止优化器"""
        await self.task_scheduler.stop()
        logger.info("Reasoning optimizer stopped")
    
    async def optimize_reasoning_request(self,
                                       query: Dict[str, Any],
                                       priority: ReasoningPriority = ReasoningPriority.MEDIUM) -> Any:
        """优化推理请求"""
        start_time = utc_now()
        query_id = self._generate_query_id(query)
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            query_id=query_id,
            query_type=query.get("type", "unknown"),
            start_time=start_time
        )
        
        try:
            # 1. 缓存检查
            if self.optimization_strategies[OptimizationStrategy.CACHE_FIRST]:
                cache_key = self._generate_cache_key(query)
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result is not None:
                    metrics.cache_hits = 1
                    metrics.optimization_applied.append("cache_hit")
                    metrics.end_time = utc_now()
                    metrics.execution_time_ms = (metrics.end_time - start_time).total_seconds() * 1000
                    self.performance_history.append(metrics)
                    return cached_result
                else:
                    metrics.cache_misses = 1
            
            # 2. 查询优化
            if self.optimization_strategies[OptimizationStrategy.QUERY_REWRITING]:
                optimized_query = self.query_optimizer.optimize_query(query)
                metrics.optimization_applied.append("query_rewrite")
            else:
                optimized_query = query
            
            # 3. 资源感知调度
            if self.optimization_strategies[OptimizationStrategy.RESOURCE_AWARE]:
                if not self.resource_monitor.has_available_resources():
                    # 降级策略
                    optimized_query = self._apply_resource_constraints(optimized_query)
                    metrics.optimization_applied.append("resource_constrained")
            
            # 4. 并行执行
            if self.optimization_strategies[OptimizationStrategy.PARALLEL_EXECUTION]:
                result = await self._execute_parallel_reasoning(optimized_query, metrics)
                metrics.optimization_applied.append("parallel_execution")
            else:
                result = await self._execute_sequential_reasoning(optimized_query, metrics)
            
            # 5. 缓存结果
            if self.optimization_strategies[OptimizationStrategy.CACHE_FIRST] and result is not None:
                cache_key = self._generate_cache_key(query)
                await self.cache_manager.set(cache_key, result, ttl=3600)  # 1小时TTL
            
            # 更新性能指标
            metrics.end_time = utc_now()
            metrics.execution_time_ms = (metrics.end_time - start_time).total_seconds() * 1000
            metrics.memory_used_mb = self._get_memory_usage()
            metrics.cpu_usage_percent = self.resource_monitor.get_system_resources()["cpu_percent"]
            
            self.performance_history.append(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning optimization failed for query {query_id}: {str(e)}")
            metrics.end_time = utc_now()
            metrics.execution_time_ms = (metrics.end_time - start_time).total_seconds() * 1000
            self.performance_history.append(metrics)
            raise
    
    def _setup_default_optimizations(self):
        """设置默认优化规则"""
        
        def limit_hops_rule(query: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
            """限制跳数优化规则"""
            if query.get("max_hops", 0) > 5:
                optimized = query.copy()
                optimized["max_hops"] = 5
                return optimized, "limit_max_hops"
            return None
        
        def constraint_reordering_rule(query: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
            """约束重排序规则"""
            if "constraints" in query and len(query["constraints"]) > 1:
                # 按选择性排序约束（简化实现）
                optimized = query.copy()
                optimized["constraints"] = sorted(query["constraints"])
                return optimized, "reorder_constraints"
            return None
        
        self.query_optimizer.add_optimization_rule(limit_hops_rule)
        self.query_optimizer.add_optimization_rule(constraint_reordering_rule)
    
    def _apply_resource_constraints(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """应用资源约束"""
        constrained_query = query.copy()
        
        # 减少并行度
        if "max_parallel" in constrained_query:
            constrained_query["max_parallel"] = min(2, constrained_query["max_parallel"])
        
        # 减少结果数量
        if "max_results" in constrained_query:
            constrained_query["max_results"] = min(50, constrained_query["max_results"])
        
        # 减少跳数
        if "max_hops" in constrained_query:
            constrained_query["max_hops"] = min(3, constrained_query["max_hops"])
        
        return constrained_query
    
    async def _execute_parallel_reasoning(self, query: Dict[str, Any], metrics: PerformanceMetrics) -> Any:
        """并行执行推理"""
        # 简化的并行执行实现
        max_workers = min(4, query.get("max_parallel", 2))
        metrics.parallel_workers = max_workers
        
        # 模拟并行任务
        tasks = []
        for i in range(max_workers):
            task = ReasoningTask(
                task_id=f"{metrics.query_id}_{i}",
                query_type=query.get("type", "unknown"),
                parameters=query,
                priority=ReasoningPriority.MEDIUM
            )
            tasks.append(task)
        
        # 提交任务
        futures = []
        for task in tasks:
            task_id = await self.task_scheduler.submit_task(task)
            futures.append(task_id)
        
        # 等待完成（简化实现）
        await asyncio.sleep(0.1)
        
        return {"parallel_results": len(futures), "query_id": metrics.query_id}
    
    async def _execute_sequential_reasoning(self, query: Dict[str, Any], metrics: PerformanceMetrics) -> Any:
        """顺序执行推理"""
        metrics.parallel_workers = 1
        
        # 模拟顺序执行
        task = ReasoningTask(
            task_id=metrics.query_id,
            query_type=query.get("type", "unknown"),
            parameters=query,
            priority=ReasoningPriority.MEDIUM
        )
        
        task_id = await self.task_scheduler.submit_task(task)
        
        # 等待完成（简化实现）
        await asyncio.sleep(0.1)
        
        return {"sequential_result": task_id, "query_id": metrics.query_id}
    
    def _generate_query_id(self, query: Dict[str, Any]) -> str:
        """生成查询ID"""
        query_str = json.dumps(query, sort_keys=True)
        hash_obj = hashlib.md5(query_str.encode())
        return hash_obj.hexdigest()[:16]
    
    def _generate_cache_key(self, query: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 移除时间戳等非确定性字段
        cache_query = {k: v for k, v in query.items() if k not in ["timestamp", "request_id"]}
        query_str = json.dumps(cache_query, sort_keys=True)
        hash_obj = hashlib.sha256(query_str.encode())
        return f"reasoning_cache:{hash_obj.hexdigest()}"
    
    def _get_memory_usage(self) -> float:
        """获取内存使用情况"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # MB
        except:
            return 0.0
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-100:]  # 最近100个查询
        
        avg_execution_time = np.mean([m.execution_time_ms for m in recent_metrics])
        cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        avg_parallel_workers = np.mean([m.parallel_workers for m in recent_metrics])
        
        optimization_counts = defaultdict(int)
        for metrics in recent_metrics:
            for opt in metrics.optimization_applied:
                optimization_counts[opt] += 1
        
        return {
            "total_queries": len(self.performance_history),
            "recent_queries": len(recent_metrics),
            "avg_execution_time_ms": avg_execution_time,
            "cache_hit_rate": cache_hit_rate,
            "avg_parallel_workers": avg_parallel_workers,
            "optimization_frequency": dict(optimization_counts),
            "cache_statistics": self.cache_manager.get_cache_statistics(),
            "scheduler_statistics": self.task_scheduler.get_scheduler_statistics(),
            "query_optimizer_statistics": self.query_optimizer.get_optimization_statistics(),
            "enabled_strategies": [k for k, v in self.optimization_strategies.items() if v]
        }
    
    def configure_strategy(self, strategy: OptimizationStrategy, enabled: bool):
        """配置优化策略"""
        self.optimization_strategies[strategy] = enabled
        logger.info(f"Optimization strategy {strategy} {'enabled' if enabled else 'disabled'}")
    
    def clear_performance_history(self):
        """清除性能历史"""
        self.performance_history.clear()
        logger.info("Performance history cleared")
