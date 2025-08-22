"""
批处理引擎

提供大规模并行任务执行、任务管理和进度追踪功能。
"""

from typing import List, Dict, Any, Callable, Optional, AsyncIterator, Union
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import uuid
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from typing import Tuple

from .checkpoint_manager import checkpoint_manager, CheckpointConfig
from .batch_types import BatchJob, BatchTask, BatchStatus, TaskPriority

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批处理引擎"""
    
    def __init__(
        self, 
        max_workers: int = 10,
        max_concurrent_jobs: int = 5,
        task_timeout: float = 300.0,  # 5分钟
        enable_checkpointing: bool = True,
        checkpoint_config: CheckpointConfig = None
    ):
        self.max_workers = max_workers
        self.max_concurrent_jobs = max_concurrent_jobs
        self.default_task_timeout = task_timeout
        self.enable_checkpointing = enable_checkpointing
        
        # 工作管理
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 作业和任务管理
        self.jobs: Dict[str, BatchJob] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.job_dependencies: Dict[str, List[str]] = {}
        
        # 性能指标
        self._total_jobs_processed = 0
        self._total_tasks_processed = 0
        self._average_task_time = 0.0
        
        # 控制标志
        self._running = False
        self._paused = False
        
        # 新增：容错和一致性保证
        self._circuit_breakers: Dict[str, Dict] = {}  # task_type -> breaker_state
        self._failed_task_cache: Dict[str, List[str]] = {}  # job_id -> [failed_task_ids]
        self._consistency_validators: Dict[str, Callable] = {}  # task_type -> validator
        self._transaction_managers: Dict[str, Any] = {}  # 事务管理器
        
        # 检查点管理
        if self.enable_checkpointing:
            checkpoint_manager.config = checkpoint_config or CheckpointConfig()
        
    def register_task_handler(self, task_type: str, handler: Callable):
        """注册任务处理器"""
        self.task_handlers[task_type] = handler
        logger.info(f"注册任务处理器: {task_type}")
    
    def register_consistency_validator(self, task_type: str, validator: Callable[[Any, Any], bool]):
        """注册数据一致性验证器"""
        self._consistency_validators[task_type] = validator
        logger.info(f"注册一致性验证器: {task_type}")
    
    def _get_circuit_breaker_state(self, task_type: str) -> Dict:
        """获取熔断器状态"""
        if task_type not in self._circuit_breakers:
            self._circuit_breakers[task_type] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': 0,
                'success_count': 0,
                'timeout': 60  # 熔断超时时间（秒）
            }
        return self._circuit_breakers[task_type]
    
    def _update_circuit_breaker(self, task_type: str, success: bool, threshold: int = 5):
        """更新熔断器状态"""
        breaker = self._get_circuit_breaker_state(task_type)
        current_time = time.time()
        
        if success:
            breaker['success_count'] += 1
            if breaker['state'] == 'half_open' and breaker['success_count'] >= 3:
                # 半开状态下连续成功，关闭熔断器
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
                logger.info(f"熔断器关闭: {task_type}")
        else:
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = current_time
            breaker['success_count'] = 0
            
            if breaker['state'] == 'closed' and breaker['failure_count'] >= threshold:
                # 失败次数达到阈值，打开熔断器
                breaker['state'] = 'open'
                logger.warning(f"熔断器打开: {task_type} (失败次数: {breaker['failure_count']})")
        
        # 检查熔断器超时
        if (breaker['state'] == 'open' and 
            current_time - breaker['last_failure_time'] > breaker['timeout']):
            breaker['state'] = 'half_open'
            logger.info(f"熔断器半开: {task_type}")
    
    def _is_circuit_breaker_open(self, task_type: str, threshold: int = 5) -> bool:
        """检查熔断器是否打开"""
        breaker = self._get_circuit_breaker_state(task_type)
        return breaker['state'] == 'open'
    
    async def start(self):
        """启动批处理引擎"""
        if self._running:
            logger.warning("批处理引擎已在运行")
            return
            
        self._running = True
        logger.info(f"启动批处理引擎 (工作者: {self.max_workers})")
        
        # 启动检查点管理
        if self.enable_checkpointing:
            await checkpoint_manager.start_auto_save()
        
        # 启动工作者
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # 启动结果处理器
        self._result_processor = asyncio.create_task(self._process_results())
        
    async def stop(self):
        """停止批处理引擎"""
        if not self._running:
            return
            
        self._running = False
        logger.info("正在停止批处理引擎...")
        
        # 停止检查点管理
        if self.enable_checkpointing:
            await checkpoint_manager.stop_auto_save()
        
        # 取消所有工作者
        for worker in self.workers:
            worker.cancel()
        
        # 等待工作者完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # 停止结果处理器
        if hasattr(self, '_result_processor'):
            self._result_processor.cancel()
            
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        self.workers.clear()
        logger.info("批处理引擎已停止")
    
    async def submit_job(self, job: BatchJob) -> str:
        """提交批处理作业"""
        if not self._running:
            raise RuntimeError("批处理引擎未运行")
        
        # 验证任务处理器
        for task in job.tasks:
            if task.type not in self.task_handlers:
                raise ValueError(f"未找到任务类型处理器: {task.type}")
        
        # 存储作业
        self.jobs[job.id] = job
        
        # 注册检查点管理
        if self.enable_checkpointing:
            await checkpoint_manager.register_job(job)
        
        # 检查依赖关系
        ready_tasks = self._get_ready_tasks(job)
        
        # 提交就绪任务到队列
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.utcnow()
        
        for task in ready_tasks:
            await self._enqueue_task(job.id, task)
        
        logger.info(f"提交批处理作业: {job.id} (任务数: {len(job.tasks)})")
        return job.id
    
    async def submit_batch(self, tasks: List[Dict[str, Any]], job_config: Optional[Dict] = None) -> str:
        """便捷方法：提交任务批次"""
        job_id = str(uuid.uuid4())
        
        # 创建任务对象
        batch_tasks = []
        for i, task_data in enumerate(tasks):
            task = BatchTask(
                id=f"{job_id}-task-{i}",
                type=task_data.get("type", "default"),
                data=task_data.get("data"),
                priority=task_data.get("priority", TaskPriority.NORMAL),
                max_retries=task_data.get("max_retries", 3),
                timeout=task_data.get("timeout"),
                dependencies=task_data.get("dependencies", []),
                tags=task_data.get("tags", {})
            )
            batch_tasks.append(task)
        
        # 创建作业
        config = job_config or {}
        job = BatchJob(
            id=job_id,
            name=config.get("name", f"batch-job-{job_id}"),
            tasks=batch_tasks,
            priority=config.get("priority", TaskPriority.NORMAL),
            max_parallel_tasks=config.get("max_parallel_tasks", 10),
            timeout=config.get("timeout"),
            tags=config.get("tags", {}),
            continue_on_failure=config.get("continue_on_failure", True),
            failure_threshold=config.get("failure_threshold", 0.1)
        )
        
        return await self.submit_job(job)
    
    def _get_ready_tasks(self, job: BatchJob) -> List[BatchTask]:
        """获取就绪执行的任务"""
        ready_tasks = []
        
        for task in job.tasks:
            if task.status != BatchStatus.PENDING:
                continue
                
            # 检查依赖任务是否完成
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self._find_task_in_job(job, dep_id)
                if not dep_task or dep_task.status != BatchStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        return ready_tasks
    
    def _find_task_in_job(self, job: BatchJob, task_id: str) -> Optional[BatchTask]:
        """在作业中查找指定任务"""
        for task in job.tasks:
            if task.id == task_id:
                return task
        return None
    
    async def _enqueue_task(self, job_id: str, task: BatchTask):
        """将任务加入执行队列"""
        task.status = BatchStatus.PENDING
        await self.task_queue.put((task.priority, job_id, task))
    
    async def _worker(self, worker_id: str):
        """工作者协程"""
        logger.info(f"启动工作者: {worker_id}")
        
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(1)
                    continue
                
                # 获取任务
                try:
                    priority, job_id, task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                job = self.jobs.get(job_id)
                if not job:
                    logger.warning(f"作业不存在: {job_id}")
                    continue
                
                # 执行任务
                await self._execute_task(worker_id, job, task)
                
            except Exception as e:
                logger.error(f"工作者 {worker_id} 执行出错: {e}")
        
        logger.info(f"工作者停止: {worker_id}")
    
    async def _execute_task(self, worker_id: str, job: BatchJob, task: BatchTask):
        """执行单个任务"""
        # 检查熔断器状态
        if self._is_circuit_breaker_open(task.type, task.circuit_breaker_threshold):
            logger.warning(f"熔断器打开，跳过任务: {task.id} (类型: {task.type})")
            task.error = "熔断器打开"
            task.status = BatchStatus.FAILED
            await self.result_queue.put(('circuit_breaker', job.id, task))
            return
        
        # 验证数据完整性
        if not task.verify_data_integrity():
            logger.error(f"任务数据完整性验证失败: {task.id}")
            task.error = "数据完整性验证失败"
            task.status = BatchStatus.FAILED
            await self.result_queue.put(('data_integrity_failed', job.id, task))
            return
        
        task.status = BatchStatus.RUNNING
        task.started_at = datetime.utcnow()
        start_time = time.time()
        success = False
        
        try:
            logger.debug(f"工作者 {worker_id} 开始执行任务: {task.id}")
            
            # 获取任务处理器
            handler = self.task_handlers.get(task.type)
            if not handler:
                raise ValueError(f"未找到任务处理器: {task.type}")
            
            # 设置超时
            timeout = task.timeout or self.default_task_timeout
            
            # 执行任务（包装在事务中）
            result = await self._execute_with_transaction(handler, task, timeout)
            
            # 验证结果一致性
            if task.type in self._consistency_validators:
                validator = self._consistency_validators[task.type]
                if not validator(task.data, result):
                    raise ValueError("结果一致性验证失败")
            
            # 记录成功结果
            task.result = result
            task.status = BatchStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.execution_time = time.time() - start_time
            success = True
            
            logger.debug(f"任务完成: {task.id} (耗时: {task.execution_time:.2f}s)")
            
            # 发送结果通知
            await self.result_queue.put(('completed', job.id, task))
            
        except asyncio.TimeoutError:
            task.error = "任务超时"
            task.status = BatchStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.execution_time = time.time() - start_time
            
            logger.warning(f"任务超时: {task.id}")
            await self._handle_task_failure(job, task, 'timeout')
            
        except Exception as e:
            task.error = str(e)
            task.error_details = {
                "exception_type": type(e).__name__,
                "worker_id": worker_id,
                "execution_time": time.time() - start_time
            }
            task.status = BatchStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.execution_time = time.time() - start_time
            
            logger.error(f"任务执行失败: {task.id} - {e}")
            await self._handle_task_failure(job, task, 'failed')
        
        # 更新熔断器状态
        self._update_circuit_breaker(task.type, success, task.circuit_breaker_threshold)
    
    async def _execute_with_transaction(self, handler: Callable, task: BatchTask, timeout: float):
        """在事务中执行任务"""
        # 对于幂等任务，可以安全重试
        if not task.idempotent:
            logger.warning(f"非幂等任务执行: {task.id}")
        
        # 执行任务
        if asyncio.iscoroutinefunction(handler):
            # 异步处理器
            result = await asyncio.wait_for(
                handler(task.data), 
                timeout=timeout
            )
        else:
            # 同步处理器，在线程池中执行
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, 
                    handler, 
                    task.data
                ),
                timeout=timeout
            )
        
        return result
    
    async def _handle_task_failure(self, job: BatchJob, task: BatchTask, failure_type: str):
        """处理任务失败"""
        # 记录失败任务
        if job.id not in self._failed_task_cache:
            self._failed_task_cache[job.id] = []
        self._failed_task_cache[job.id].append(task.id)
        
        # 检查是否需要重试
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            
            # 计算重试延迟
            retry_delay = task.get_next_retry_delay()
            
            # 重置任务状态以便重试
            task.status = BatchStatus.PENDING
            task.started_at = None
            task.error = None
            task.error_details = None
            
            logger.info(f"任务重试: {task.id} (第{task.retry_count}次, 延迟: {retry_delay:.1f}s)")
            
            # 延迟后重新加入队列
            asyncio.create_task(self._delayed_retry(job.id, task, retry_delay))
        else:
            # 达到最大重试次数
            await self.result_queue.put((failure_type, job.id, task))
    
    async def _delayed_retry(self, job_id: str, task: BatchTask, delay: float):
        """延迟重试"""
        await asyncio.sleep(delay)
        await self._enqueue_task(job_id, task)
    
    async def _process_results(self):
        """处理任务结果"""
        while self._running:
            try:
                # 获取结果
                try:
                    event_type, job_id, task = await asyncio.wait_for(
                        self.result_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                job = self.jobs.get(job_id)
                if not job:
                    continue
                
                # 更新作业统计
                if event_type == 'completed':
                    job.completed_tasks += 1
                elif event_type in ['failed', 'timeout']:
                    job.failed_tasks += 1
                
                # 检查是否需要提交更多任务
                if event_type == 'completed':
                    ready_tasks = self._get_ready_tasks(job)
                    for ready_task in ready_tasks:
                        await self._enqueue_task(job_id, ready_task)
                
                # 检查作业是否完成
                await self._check_job_completion(job)
                
                # 检查失败率是否超过阈值
                if not job.continue_on_failure and job.failure_rate > job.failure_threshold:
                    await self._cancel_job(job.id, "失败率超过阈值")
                
            except Exception as e:
                logger.error(f"结果处理出错: {e}")
    
    async def _check_job_completion(self, job: BatchJob):
        """检查作业是否完成"""
        if job.status not in [BatchStatus.RUNNING]:
            return
        
        total_processed = job.completed_tasks + job.failed_tasks + job.cancelled_tasks
        
        if total_processed >= job.total_tasks:
            # 作业完成
            if job.failed_tasks == 0:
                job.status = BatchStatus.COMPLETED
            else:
                job.status = BatchStatus.FAILED
            
            job.completed_at = datetime.utcnow()
            self._total_jobs_processed += 1
            
            logger.info(f"作业完成: {job.id} (状态: {job.status.value}, "
                       f"成功: {job.completed_tasks}, 失败: {job.failed_tasks})")
    
    async def _cancel_job(self, job_id: str, reason: str = "用户取消"):
        """取消作业"""
        job = self.jobs.get(job_id)
        if not job or job.status not in [BatchStatus.PENDING, BatchStatus.RUNNING]:
            return False
        
        # 取消所有未完成的任务
        for task in job.tasks:
            if task.status in [BatchStatus.PENDING, BatchStatus.RUNNING]:
                task.status = BatchStatus.CANCELLED
                job.cancelled_tasks += 1
        
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        logger.info(f"取消作业: {job_id} (原因: {reason})")
        return True
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """获取作业状态"""
        return self.jobs.get(job_id)
    
    async def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取作业进度详情"""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        # 按状态分组任务
        status_counts = {}
        for task in job.tasks:
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        # 计算平均执行时间
        completed_tasks = [t for t in job.tasks if t.status == BatchStatus.COMPLETED and t.execution_time]
        avg_execution_time = 0
        if completed_tasks:
            avg_execution_time = sum(t.execution_time for t in completed_tasks) / len(completed_tasks)
        
        # 估算剩余时间
        remaining_tasks = job.total_tasks - job.completed_tasks - job.failed_tasks - job.cancelled_tasks
        estimated_remaining_time = remaining_tasks * avg_execution_time if avg_execution_time > 0 else None
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": job.progress,
            "total_tasks": job.total_tasks,
            "completed_tasks": job.completed_tasks,
            "failed_tasks": job.failed_tasks,
            "cancelled_tasks": job.cancelled_tasks,
            "status_breakdown": status_counts,
            "success_rate": job.success_rate,
            "failure_rate": job.failure_rate,
            "average_execution_time": avg_execution_time,
            "estimated_remaining_time": estimated_remaining_time,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """取消作业"""
        return await self._cancel_job(job_id, "用户取消")
    
    async def pause_job(self, job_id: str) -> bool:
        """暂停作业"""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchStatus.RUNNING:
            return False
        
        job.status = BatchStatus.PAUSED
        logger.info(f"暂停作业: {job_id}")
        return True
    
    async def resume_job(self, job_id: str) -> bool:
        """恢复作业"""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchStatus.PAUSED:
            return False
        
        job.status = BatchStatus.RUNNING
        
        # 重新提交暂停的任务
        ready_tasks = self._get_ready_tasks(job)
        for task in ready_tasks:
            await self._enqueue_task(job_id, task)
        
        logger.info(f"恢复作业: {job_id}")
        return True
    
    def pause_processing(self):
        """暂停整体处理"""
        self._paused = True
        logger.info("暂停批处理")
    
    def resume_processing(self):
        """恢复整体处理"""
        self._paused = False
        logger.info("恢复批处理")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        active_jobs = len([j for j in self.jobs.values() if j.status == BatchStatus.RUNNING])
        total_tasks_in_queue = self.task_queue.qsize()
        
        return {
            "total_jobs_processed": self._total_jobs_processed,
            "total_tasks_processed": self._total_tasks_processed,
            "active_jobs": active_jobs,
            "total_jobs": len(self.jobs),
            "tasks_in_queue": total_tasks_in_queue,
            "active_workers": len([w for w in self.workers if not w.done()]),
            "max_workers": self.max_workers,
            "processing_paused": self._paused,
            "engine_running": self._running
        }
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """清理完成的作业"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                job.completed_at and job.completed_at.timestamp() < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            # 注销检查点管理
            if self.enable_checkpointing:
                await checkpoint_manager.unregister_job(job_id)
            
            del self.jobs[job_id]
            
            # 清理失败任务缓存
            if job_id in self._failed_task_cache:
                del self._failed_task_cache[job_id]
        
        if jobs_to_remove:
            logger.info(f"清理了 {len(jobs_to_remove)} 个已完成的作业")
    
    async def create_checkpoint(self, job_id: str) -> Optional[str]:
        """手动创建检查点"""
        if not self.enable_checkpointing:
            return None
        
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return await checkpoint_manager.create_checkpoint(job, 'manual')
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> Optional[str]:
        """从检查点恢复作业"""
        if not self.enable_checkpointing:
            return None
        
        job = await checkpoint_manager.restore_job(checkpoint_id)
        if not job:
            return None
        
        # 重新提交作业
        return await self.submit_job(job)
    
    async def list_job_checkpoints(self, job_id: str):
        """列出作业的检查点"""
        if not self.enable_checkpointing:
            return []
        
        return await checkpoint_manager.list_job_checkpoints(job_id)
    
    async def retry_failed_tasks(self, job_id: str) -> bool:
        """重试作业中的失败任务"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        failed_task_ids = self._failed_task_cache.get(job_id, [])
        if not failed_task_ids:
            return False
        
        retry_count = 0
        for task in job.tasks:
            if task.id in failed_task_ids and task.status == BatchStatus.FAILED:
                # 重置任务状态
                task.status = BatchStatus.PENDING
                task.retry_count = 0
                task.error = None
                task.error_details = None
                task.started_at = None
                task.completed_at = None
                
                # 重新提交任务
                await self._enqueue_task(job_id, task)
                retry_count += 1
        
        if retry_count > 0:
            # 清理失败缓存
            self._failed_task_cache[job_id] = []
            logger.info(f"重试失败任务: {job_id} ({retry_count}个任务)")
        
        return retry_count > 0
    
    async def get_fault_tolerance_stats(self) -> Dict[str, Any]:
        """获取容错统计信息"""
        circuit_breaker_stats = {}
        for task_type, breaker in self._circuit_breakers.items():
            circuit_breaker_stats[task_type] = {
                'state': breaker['state'],
                'failure_count': breaker['failure_count'],
                'success_count': breaker['success_count']
            }
        
        failed_tasks_count = sum(len(tasks) for tasks in self._failed_task_cache.values())
        
        stats = {
            'circuit_breakers': circuit_breaker_stats,
            'total_failed_tasks': failed_tasks_count,
            'jobs_with_failures': len(self._failed_task_cache),
            'registered_validators': len(self._consistency_validators)
        }
        
        if self.enable_checkpointing:
            checkpoint_stats = await checkpoint_manager.get_checkpoint_stats()
            stats['checkpoints'] = checkpoint_stats
        
        return stats