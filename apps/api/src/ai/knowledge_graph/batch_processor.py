"""
知识图谱批处理器模块

提供高性能批量文档处理能力，支持：
- 异步并发处理
- 分布式任务调度
- 内存优化和资源管理
- 进度监控和错误恢复
- 缓存机制优化
"""

import asyncio
import time
import gc
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
import weakref
from .data_models import Entity, Relation, KnowledgeGraph, BatchProcessingResult
from .entity_recognizer import MultiModelEntityRecognizer
from .relation_extractor import RelationExtractor
from .entity_linker import EntityLinker
from .multilingual_processor import MultilingualProcessor

from src.core.logging import get_logger
logger = get_logger(__name__)

class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CacheStrategy(Enum):
    """缓存策略枚举"""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"

@dataclass
class BatchTask:
    """批处理任务数据结构"""
    task_id: str
    document_id: str
    text: str
    language: Optional[str] = None
    priority: int = 0
    created_at: datetime = field(default_factory=utc_factory)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchConfig:
    """批处理配置"""
    # 并发控制
    max_concurrent_tasks: int = 50
    max_concurrent_per_model: int = 10
    worker_pool_size: int = mp.cpu_count()
    
    # 性能优化
    chunk_size: int = 1000  # 文档分块大小
    memory_limit_mb: int = 2048  # 内存限制
    gc_threshold: int = 100  # 垃圾回收阈值
    
    # 缓存配置
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    cache_size_limit: int = 10000
    cache_ttl_seconds: int = 3600
    
    # 重试和超时
    task_timeout_seconds: int = 300
    retry_delay_seconds: int = 5
    max_retries: int = 3
    
    # 分布式配置
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    worker_node_id: Optional[str] = None

@dataclass
class ProcessingMetrics:
    """处理指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    throughput_docs_per_minute: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    start_time: datetime = field(default_factory=utc_factory)
    last_update: datetime = field(default_factory=utc_factory)

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, limit_mb: int = 2048, gc_threshold: int = 100):
        self.limit_mb = limit_mb
        self.gc_threshold = gc_threshold
        self.processed_count = 0
        self.logger = get_logger(__name__)
    
    def check_memory_usage(self) -> float:
        """检查内存使用情况"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    
    def should_gc(self) -> bool:
        """判断是否需要垃圾回收"""
        self.processed_count += 1
        
        if self.processed_count >= self.gc_threshold:
            return True
            
        memory_usage = self.check_memory_usage()
        if memory_usage > self.limit_mb * 0.8:  # 80%阈值
            return True
            
        return False
    
    def perform_gc(self):
        """执行垃圾回收"""
        try:
            collected = gc.collect()
            self.processed_count = 0
            memory_after = self.check_memory_usage()
            self.logger.debug(f"垃圾回收完成: 回收对象数={collected}, 内存使用={memory_after:.1f}MB")
        except Exception as e:
            self.logger.warning(f"垃圾回收失败: {e}")

class ResultCache:
    """结果缓存管理器"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.HYBRID, 
                 size_limit: int = 10000, ttl_seconds: int = 3600):
        self.strategy = strategy
        self.size_limit = size_limit
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def _generate_key(self, text: str, language: Optional[str] = None) -> str:
        """生成缓存键"""
        content = f"{text}:{language or 'auto'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, language: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if self.strategy == CacheStrategy.NONE:
            return None
            
        key = self._generate_key(text, language)
        
        with self.lock:
            if key in self.cache:
                result, cached_time = self.cache[key]
                if utc_now() - cached_time < timedelta(seconds=self.ttl_seconds):
                    self.access_times[key] = utc_now()
                    self.hit_count += 1
                    return result
                else:
                    # 过期删除
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            self.miss_count += 1
            return None
    
    def put(self, text: str, result: Dict[str, Any], language: Optional[str] = None):
        """存储缓存结果"""
        if self.strategy == CacheStrategy.NONE:
            return
            
        key = self._generate_key(text, language)
        
        with self.lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.size_limit:
                self._evict_lru()
            
            self.cache[key] = (result, utc_now())
            self.access_times[key] = utc_now()
    
    def _evict_lru(self):
        """LRU淘汰策略"""
        if not self.access_times:
            return
            
        # 找到最久未访问的键
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # 删除缓存条目
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.task_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, BatchTask] = {}
        self.completed_tasks: Dict[str, BatchTask] = {}
        self.failed_tasks: Dict[str, BatchTask] = {}
        self.cancelled_tasks: Dict[str, BatchTask] = {}
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.logger = get_logger(__name__)
    
    async def add_task(self, task: BatchTask):
        """添加任务到队列"""
        if task.priority > 0:
            await self.priority_queue.put((-task.priority, task.task_id, task))
        else:
            await self.task_queue.put(task)
        
        self.logger.debug(f"任务已添加到队列: {task.task_id}")
    
    async def get_next_task(self) -> Optional[BatchTask]:
        """获取下一个待处理任务"""
        try:
            # 优先处理高优先级任务
            if not self.priority_queue.empty():
                _, task_id, task = await asyncio.wait_for(
                    self.priority_queue.get(), timeout=0.1
                )
                return task
            
            # 处理普通任务
            if not self.task_queue.empty():
                task = await asyncio.wait_for(
                    self.task_queue.get(), timeout=0.1
                )
                return task
        except asyncio.TimeoutError:
            self.logger.debug("获取任务超时", exc_info=True)
        
        return None
    
    def mark_task_started(self, task: BatchTask):
        """标记任务开始处理"""
        task.status = ProcessingStatus.PROCESSING
        task.started_at = utc_now()
        self.active_tasks[task.task_id] = task
    
    def mark_task_completed(self, task: BatchTask, result: Dict[str, Any]):
        """标记任务完成"""
        task.status = ProcessingStatus.COMPLETED
        task.completed_at = utc_now()
        task.result = result
        
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        
        self.completed_tasks[task.task_id] = task
    
    def mark_task_failed(self, task: BatchTask, error: str):
        """标记任务失败"""
        task.status = ProcessingStatus.FAILED
        task.completed_at = utc_now()
        task.error = error
        task.retry_count += 1
        
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        
        # 判断是否需要重试
        if task.retry_count < task.max_retries:
            # 重新加入队列进行重试
            asyncio.create_task(self.add_task(task))
            self.logger.info(f"任务将重试: {task.task_id}, 重试次数: {task.retry_count}")
        else:
            self.failed_tasks[task.task_id] = task
            self.logger.error(f"任务最终失败: {task.task_id}, 错误: {error}")

    def mark_task_cancelled(self, task: BatchTask):
        """标记任务取消"""
        task.status = ProcessingStatus.CANCELLED
        task.completed_at = utc_now()
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        self.cancelled_tasks[task.task_id] = task
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.task_queue.qsize() + self.priority_queue.qsize()
    
    def get_active_count(self) -> int:
        """获取活动任务数量"""
        return len(self.active_tasks)

class BatchProcessor:
    """知识图谱批处理器"""
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.entity_recognizer = MultiModelEntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.entity_linker = EntityLinker()
        self.multilingual_processor = MultilingualProcessor()
        
        # 组件初始化
        self.task_scheduler = TaskScheduler(self.config)
        self.memory_manager = MemoryManager(
            self.config.memory_limit_mb, 
            self.config.gc_threshold
        )
        self.result_cache = ResultCache(
            self.config.cache_strategy,
            self.config.cache_size_limit,
            self.config.cache_ttl_seconds
        )
        
        # 指标和状态
        self.metrics = ProcessingMetrics()
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self._initialized = False
        self.batch_task_ids: Dict[str, List[str]] = {}
        self.batch_created_at: Dict[str, datetime] = {}
        self.cancelled_batches: set[str] = set()
        
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """初始化批处理器"""
        try:
            if self._initialized:
                return
            # 初始化各个组件
            await self.entity_recognizer.initialize()
            await self.relation_extractor.initialize()
            await self.entity_linker.initialize()
            await self.multilingual_processor.initialize()
            self._initialized = True
            
            self.logger.info("批处理器初始化完成")
            
        except Exception as e:
            self.logger.error(f"批处理器初始化失败: {e}")
            raise
    
    async def start_workers(self, num_workers: Optional[int] = None):
        """启动工作协程"""
        if self.is_running:
            return
        
        self.is_running = True
        worker_count = num_workers or self.config.worker_pool_size
        
        # 启动工作协程
        for i in range(worker_count):
            worker_task = asyncio.create_task(
                self._worker_loop(f"worker-{i}")
            )
            self.worker_tasks.append(worker_task)
        
        self.logger.info(f"已启动 {worker_count} 个工作协程")
    
    async def stop_workers(self):
        """停止工作协程"""
        self.is_running = False
        
        # 等待所有工作协程完成
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            self.worker_tasks.clear()
        
        self.logger.info("所有工作协程已停止")
    
    async def _worker_loop(self, worker_id: str):
        """工作协程主循环"""
        self.logger.debug(f"工作协程启动: {worker_id}")
        
        while self.is_running:
            try:
                # 获取下一个任务
                task = await self.task_scheduler.get_next_task()
                if not task:
                    await asyncio.sleep(0.1)  # 短暂休眠
                    continue
                
                # 处理任务
                async with self.task_scheduler.semaphore:
                    await self._process_single_task(task)
                
                # 内存管理
                if self.memory_manager.should_gc():
                    self.memory_manager.perform_gc()
                
            except Exception as e:
                self.logger.error(f"工作协程 {worker_id} 发生错误: {e}")
                await asyncio.sleep(1)  # 错误恢复延迟
    
    async def _process_single_task(self, task: BatchTask):
        """处理单个任务"""
        try:
            batch_id = task.metadata.get("batch_id")
            if batch_id and batch_id in self.cancelled_batches:
                self.task_scheduler.mark_task_cancelled(task)
                self._update_metrics(task, False, failed=True)
                return

            self.task_scheduler.mark_task_started(task)
            
            # 检查缓存
            cached_result = self.result_cache.get(task.text, task.language)
            if cached_result:
                self.task_scheduler.mark_task_completed(task, cached_result)
                self._update_metrics(task, True)
                return
            
            # 执行知识提取
            start_time = time.time()

            extract_entities = bool(task.metadata.get("extract_entities", True))
            extract_relations = bool(task.metadata.get("extract_relations", True))
            link_entities = bool(task.metadata.get("link_entities", True))
            confidence_threshold = task.metadata.get("confidence_threshold")
            
            if task.language and task.language != "auto":
                # 使用指定语言处理
                entities = (
                    await self.entity_recognizer.extract_entities(
                        task.text, task.language, confidence_threshold=confidence_threshold
                    )
                    if extract_entities
                    else []
                )
                relations = (
                    await self.relation_extractor.extract_relations(
                        task.text, entities, confidence_threshold=confidence_threshold
                    )
                    if extract_relations and entities
                    else []
                )
                linked_entities = await self.entity_linker.link_entities(entities) if link_entities else []
            else:
                # 使用多语言处理器
                multilingual_result = await self.multilingual_processor.process_multilingual_text(
                    task.text
                )
                entities = multilingual_result.entities if extract_entities else []
                relations = (
                    multilingual_result.relations
                    if extract_relations and entities
                    else []
                )
                linked_entities = multilingual_result.linked_entities if link_entities else []
            
            processing_time = time.time() - start_time
            
            # 构建结果
            result = {
                "document_id": task.document_id,
                "entities": [entity.__dict__ for entity in entities],
                "relations": [relation.__dict__ for relation in relations],
                "linked_entities": [entity.__dict__ for entity in linked_entities],
                "processing_time": processing_time,
                "language": task.language,
                "metadata": task.metadata
            }
            
            # 缓存结果
            self.result_cache.put(task.text, result, task.language)
            
            # 标记任务完成
            self.task_scheduler.mark_task_completed(task, result)
            self._update_metrics(task, False)
            
        except Exception as e:
            error_msg = f"任务处理失败: {str(e)}"
            self.task_scheduler.mark_task_failed(task, error_msg)
            self._update_metrics(task, False, failed=True)
    
    def _update_metrics(self, task: BatchTask, cache_hit: bool, failed: bool = False):
        """更新处理指标"""
        if failed:
            self.metrics.failed_tasks += 1
        else:
            self.metrics.completed_tasks += 1
        
        # 更新处理时间
        if task.started_at and task.completed_at:
            processing_time = (task.completed_at - task.started_at).total_seconds()
            total_completed = self.metrics.completed_tasks + self.metrics.failed_tasks
            
            if total_completed > 1:
                self.metrics.average_processing_time = (
                    self.metrics.average_processing_time * (total_completed - 1) + processing_time
                ) / total_completed
            else:
                self.metrics.average_processing_time = processing_time
        
        # 更新吞吐量
        elapsed_time = (utc_now() - self.metrics.start_time).total_seconds() / 60
        if elapsed_time > 0:
            total_processed = self.metrics.completed_tasks + self.metrics.failed_tasks
            self.metrics.throughput_docs_per_minute = total_processed / elapsed_time
        
        # 更新其他指标
        self.metrics.memory_usage_mb = self.memory_manager.check_memory_usage()
        self.metrics.cache_hit_rate = self.result_cache.get_hit_rate()
        
        total_tasks = self.metrics.completed_tasks + self.metrics.failed_tasks
        if total_tasks > 0:
            self.metrics.error_rate = self.metrics.failed_tasks / total_tasks
        
        self.metrics.last_update = utc_now()
    
    async def process_batch(self, documents: List[Dict[str, Any]], 
                          priority: int = 0,
                          batch_id: Optional[str] = None) -> str:
        """处理文档批次"""
        if not self._initialized:
            await self.initialize()
        await self.start_workers()

        batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
        self.batch_created_at[batch_id] = utc_now()
        
        # 创建批处理任务
        tasks = []
        for i, doc in enumerate(documents):
            metadata = dict(doc.get("metadata") or {})
            metadata["batch_id"] = batch_id
            if "extract_entities" in doc:
                metadata["extract_entities"] = bool(doc["extract_entities"])
            if "extract_relations" in doc:
                metadata["extract_relations"] = bool(doc["extract_relations"])
            if "link_entities" in doc:
                metadata["link_entities"] = bool(doc["link_entities"])
            if "confidence_threshold" in doc:
                metadata["confidence_threshold"] = doc.get("confidence_threshold")
            task = BatchTask(
                task_id=f"{batch_id}_task_{i}",
                document_id=doc.get("id", f"doc_{i}"),
                text=doc["text"],
                language=doc.get("language"),
                priority=priority,
                metadata=metadata
            )
            tasks.append(task)
        self.batch_task_ids[batch_id] = [t.task_id for t in tasks]
        
        # 添加任务到调度器
        for task in tasks:
            await self.task_scheduler.add_task(task)
        
        self.metrics.total_tasks += len(tasks)
        
        self.logger.info(f"批处理任务已创建: {batch_id}, 任务数量: {len(tasks)}")
        return batch_id

    async def submit_batch(self, batch_id: str, documents: List[str], config: Dict[str, Any]) -> str:
        """提交批处理任务"""
        docs = []
        for idx, text in enumerate(documents):
            docs.append(
                {
                    "id": f"doc_{idx}",
                    "text": text,
                    "language": config.get("language"),
                    "metadata": config.get("metadata") or {},
                    "extract_entities": config.get("extract_entities", True),
                    "extract_relations": config.get("extract_relations", True),
                    "link_entities": config.get("link_entities", True),
                    "confidence_threshold": config.get("confidence_threshold"),
                }
            )
        return await self.process_batch(docs, priority=int(config.get("priority", 0) or 0), batch_id=batch_id)

    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """获取批处理任务状态"""
        task_ids = self.batch_task_ids.get(batch_id)
        if not task_ids:
            return None
        task_set = set(task_ids)

        completed = [t for tid, t in self.task_scheduler.completed_tasks.items() if tid in task_set]
        failed = [t for tid, t in self.task_scheduler.failed_tasks.items() if tid in task_set]
        cancelled = [t for tid, t in self.task_scheduler.cancelled_tasks.items() if tid in task_set]
        active = [t for tid, t in self.task_scheduler.active_tasks.items() if tid in task_set]

        processed = len(completed) + len(failed) + len(cancelled) + len(active)
        total = len(task_ids)

        status = "pending"
        if batch_id in self.cancelled_batches:
            status = "cancelled"
        elif processed > 0:
            status = "processing"
        if processed >= total and not active:
            status = "failed" if (len(failed) + len(cancelled)) == total else "completed"

        created_at = self.batch_created_at.get(batch_id, utc_now())
        updated_at = utc_now()
        processing_time = (updated_at - created_at).total_seconds()
        successful = len(completed)
        failed_count = len(failed) + len(cancelled)
        return {
            "batch_id": batch_id,
            "status": status,
            "total_documents": total,
            "processed_documents": min(processed, total),
            "successful_documents": successful,
            "failed_documents": failed_count,
            "success_rate": successful / total if total else 0.0,
            "results": [],
            "errors": [
                {"task_id": t.task_id, "document_id": t.document_id, "error": t.error or "cancelled"}
                for t in failed + cancelled
            ],
            "processing_time": processing_time,
            "metrics": self.metrics.__dict__.copy(),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    async def list_batches(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出批处理任务摘要"""
        batch_items = list(self.batch_task_ids.items())
        batch_items.sort(
            key=lambda item: self.batch_created_at.get(item[0], utc_now()),
            reverse=True
        )
        summaries: List[Dict[str, Any]] = []
        for batch_id, task_ids in batch_items[:limit]:
            task_set = set(task_ids)
            completed = [t for tid, t in self.task_scheduler.completed_tasks.items() if tid in task_set]
            failed = [t for tid, t in self.task_scheduler.failed_tasks.items() if tid in task_set]
            cancelled = [t for tid, t in self.task_scheduler.cancelled_tasks.items() if tid in task_set]
            active = [t for tid, t in self.task_scheduler.active_tasks.items() if tid in task_set]

            processed = len(completed) + len(failed) + len(cancelled) + len(active)
            total = len(task_ids)
            status = "pending"
            if batch_id in self.cancelled_batches:
                status = "cancelled"
            elif processed > 0:
                status = "processing"
            if processed >= total and not active:
                status = "failed" if (len(failed) + len(cancelled)) == total else "completed"

            created_at = self.batch_created_at.get(batch_id, utc_now())
            last_times = [created_at]
            for task in completed + failed + cancelled + active:
                if task.completed_at:
                    last_times.append(task.completed_at)
                if task.started_at:
                    last_times.append(task.started_at)
            updated_at = max(last_times) if last_times else utc_now()
            successful = len(completed)
            failed_count = len(failed) + len(cancelled)
            progress = (processed / total) * 100 if total else 0.0
            summaries.append({
                "batch_id": batch_id,
                "status": status,
                "total_documents": total,
                "processed_documents": min(processed, total),
                "successful_documents": successful,
                "failed_documents": failed_count,
                "progress": progress,
                "created_at": created_at,
                "updated_at": updated_at,
            })
        return summaries

    async def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """获取批处理任务完整结果"""
        task_ids = self.batch_task_ids.get(batch_id)
        if not task_ids:
            return None
        task_set = set(task_ids)

        completed = [t for tid, t in self.task_scheduler.completed_tasks.items() if tid in task_set]
        failed = [t for tid, t in self.task_scheduler.failed_tasks.items() if tid in task_set]
        cancelled = [t for tid, t in self.task_scheduler.cancelled_tasks.items() if tid in task_set]

        return {
            "batch_id": batch_id,
            "status": "cancelled" if batch_id in self.cancelled_batches else "completed",
            "total_documents": len(task_ids),
            "successful_documents": len(completed),
            "failed_documents": len(failed) + len(cancelled),
            "results": [t.result for t in completed if t.result],
            "errors": [
                {"task_id": t.task_id, "document_id": t.document_id, "error": t.error or "cancelled"}
                for t in failed + cancelled
            ],
        }

    async def cancel_batch(self, batch_id: str) -> bool:
        """取消批处理任务"""
        if batch_id not in self.batch_task_ids:
            return False
        self.cancelled_batches.add(batch_id)
        return True
    
    async def process_documents_stream(self, documents: AsyncGenerator[Dict[str, Any], None], 
                                     batch_size: int = 100) -> AsyncGenerator[BatchProcessingResult, None]:
        """流式处理文档"""
        batch = []
        
        async for doc in documents:
            batch.append(doc)
            
            if len(batch) >= batch_size:
                batch_id = await self.process_batch(batch)
                
                # 等待批次完成并返回结果
                result = await self.wait_for_batch_completion(batch_id)
                yield result
                
                batch = []
        
        # 处理剩余文档
        if batch:
            batch_id = await self.process_batch(batch)
            result = await self.wait_for_batch_completion(batch_id)
            yield result
    
    async def wait_for_batch_completion(self, batch_id: str, 
                                      timeout: Optional[float] = None) -> BatchProcessingResult:
        """等待批次处理完成"""
        timeout = timeout or self.config.task_timeout_seconds
        start_time = time.time()
        task_ids = self.batch_task_ids.get(batch_id) or []
        task_set = set(task_ids)
        
        while time.time() - start_time < timeout:
            # 检查批次任务状态
            done = (
                set(self.task_scheduler.completed_tasks) |
                set(self.task_scheduler.failed_tasks) |
                set(self.task_scheduler.cancelled_tasks)
            )
            if task_set and task_set.issubset(done):
                break
                
            await asyncio.sleep(0.5)
        
        # 收集结果
        completed = [
            t for tid, t in self.task_scheduler.completed_tasks.items()
            if tid in task_set
        ]
        failed = [
            t for tid, t in self.task_scheduler.failed_tasks.items()
            if tid in task_set
        ]
        cancelled = [
            t for tid, t in self.task_scheduler.cancelled_tasks.items()
            if tid in task_set
        ]
        
        results = []
        errors = []
        
        for task in completed:
            if task.result:
                results.append(task.result)
        
        for task in failed:
            if task.error:
                errors.append({
                    "task_id": task.task_id,
                    "document_id": task.document_id,
                    "error": task.error
                })
        for task in cancelled:
            errors.append(
                {"task_id": task.task_id, "document_id": task.document_id, "error": "cancelled"}
            )
        
        return BatchProcessingResult(
            batch_id=batch_id,
            total_documents=len(task_ids),
            successful_documents=len(completed),
            failed_documents=len(failed) + len(cancelled),
            results=results,
            errors=errors,
            processing_time=(utc_now() - self.metrics.start_time).total_seconds(),
            metrics=self.metrics.__dict__.copy()
        )
    
    def get_processing_status(self) -> Dict[str, Any]:
        """获取处理状态"""
        return {
            "is_running": self.is_running,
            "queue_size": self.task_scheduler.get_queue_size(),
            "active_tasks": self.task_scheduler.get_active_count(),
            "worker_count": len(self.worker_tasks),
            "metrics": self.metrics.__dict__.copy(),
            "cache_stats": {
                "hit_rate": self.result_cache.get_hit_rate(),
                "cache_size": len(self.result_cache.cache)
            },
            "memory_usage_mb": self.memory_manager.check_memory_usage()
        }
    
    async def shutdown(self):
        """关闭批处理器"""
        self.logger.info("正在关闭批处理器...")
        
        # 停止工作协程
        await self.stop_workers()
        
        # 清理缓存
        self.result_cache.clear()
        
        # 执行最终垃圾回收
        self.memory_manager.perform_gc()
        
        self.logger.info("批处理器已关闭")

class DistributedBatchProcessor(BatchProcessor):
    """分布式批处理器"""
    
    def __init__(self, config: BatchConfig, redis_client=None):
        super().__init__(config)
        self.redis_client = redis_client
        self.node_id = config.worker_node_id or f"node_{int(time.time())}"
        
        if config.enable_distributed and not redis_client:
            raise ValueError("分布式模式需要提供 Redis 客户端")
    
    async def register_worker_node(self):
        """注册工作节点"""
        if not self.config.enable_distributed:
            return
            
        node_info = {
            "node_id": self.node_id,
            "status": "active",
            "registered_at": utc_now().isoformat(),
            "capabilities": {
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "worker_pool_size": self.config.worker_pool_size
            }
        }
        
        await self.redis_client.hset(
            "knowledge_graph:workers", 
            self.node_id, 
            json.dumps(node_info)
        )
        
        self.logger.info(f"工作节点已注册: {self.node_id}")
    
    async def distribute_batch_tasks(self, documents: List[Dict[str, Any]]) -> str:
        """分布式批次任务分发"""
        if not self.config.enable_distributed:
            return await self.process_batch(documents)
        
        # 获取可用工作节点
        worker_nodes = await self.redis_client.hgetall("knowledge_graph:workers")
        active_nodes = [
            node_id for node_id, info in worker_nodes.items()
            if json.loads(info)["status"] == "active"
        ]
        
        if not active_nodes:
            self.logger.warning("没有可用的工作节点，使用本地处理")
            return await self.process_batch(documents)
        
        # 分发任务到不同节点
        batch_id = f"distributed_batch_{int(time.time() * 1000)}"
        chunk_size = max(1, len(documents) // len(active_nodes))
        
        for i, node_id in enumerate(active_nodes):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(active_nodes) - 1 else len(documents)
            node_documents = documents[start_idx:end_idx]
            
            if node_documents:
                # 将任务推送到 Redis 队列
                task_data = {
                    "batch_id": batch_id,
                    "node_id": node_id,
                    "documents": node_documents,
                    "created_at": utc_now().isoformat()
                }
                
                await self.redis_client.lpush(
                    f"knowledge_graph:tasks:{node_id}",
                    json.dumps(task_data)
                )
        
        self.logger.info(f"分布式批处理任务已分发: {batch_id}")
        return batch_id

# 导出主要类
__all__ = [
    "BatchProcessor",
    "DistributedBatchProcessor", 
    "BatchConfig",
    "BatchTask",
    "ProcessingMetrics",
    "ProcessingStatus",
    "CacheStrategy"
]
