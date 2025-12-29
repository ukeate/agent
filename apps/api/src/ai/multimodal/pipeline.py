"""
多模态处理管道
"""

import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .processor import MultimodalProcessor
from .types import MultimodalContent, ProcessingResult, ProcessingStatus, ProcessingOptions

from src.core.logging import get_logger
logger = get_logger(__name__)

# 移除不需要的数据库导入，该项目使用dataclass而非SQLAlchemy

@dataclass
class ProcessingTask:
    """处理任务"""
    content: MultimodalContent
    options: ProcessingOptions
    priority: int = 1
    submitted_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

class ProcessingPipeline:
    """多模态处理管道"""
    
    def __init__(
        self, 
        processor: MultimodalProcessor,
        max_concurrent_tasks: int = 5
    ):
        self.processor = processor
        self.max_concurrent_tasks = max_concurrent_tasks
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queue: List[ProcessingTask] = []
        self.is_running = False
        self.active_tasks = 0
        self.processing_results: Dict[str, ProcessingResult] = {}
        self.submission_times: Dict[str, datetime] = {}
        self._workers: List[asyncio.Task] = []
        
    async def start(self):
        """启动处理管道"""
        if self.is_running:
            logger.warning("处理管道已经在运行")
            return
            
        self.is_running = True
        logger.info(f"启动处理管道，并发数: {self.max_concurrent_tasks}")
        
        # 创建工作任务
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._process_worker(i))
            self._workers.append(worker)
    
    async def stop(self):
        """停止处理管道"""
        logger.info("停止处理管道")
        self.is_running = False
        
        # 等待所有工作任务完成
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        
        logger.info("处理管道已停止")
    
    async def submit_for_processing(
        self,
        content: MultimodalContent,
        options: Optional[ProcessingOptions] = None,
        priority: int = 1
    ) -> str:
        """提交内容进行处理"""
        options = options or ProcessingOptions()
        
        task = ProcessingTask(
            content=content,
            options=options,
            priority=priority,
            submitted_at=utc_now()
        )
        
        # 根据优先级插入队列
        self.submission_times[content.content_id] = task.submitted_at
        await self._add_to_priority_queue(task)
        
        logger.info(
            f"提交处理任务",
            content_id=content.content_id,
            content_type=content.content_type,
            priority=priority
        )
        
        return content.content_id
    
    async def submit_batch(
        self,
        contents: List[MultimodalContent],
        options: Optional[ProcessingOptions] = None,
        priority: int = 1
    ) -> List[str]:
        """批量提交内容处理"""
        content_ids = []
        
        for content in contents:
            content_id = await self.submit_for_processing(
                content,
                options,
                priority
            )
            content_ids.append(content_id)
        
        logger.info(f"批量提交 {len(contents)} 个处理任务")
        return content_ids
    
    async def get_processing_status(self, content_id: str) -> Optional[ProcessingResult]:
        """获取处理状态"""
        return self.processing_results.get(content_id)
    
    async def get_all_results(self) -> Dict[str, ProcessingResult]:
        """获取所有处理结果"""
        return self.processing_results.copy()
    
    async def wait_for_completion(
        self, 
        content_id: str, 
        timeout: float = 300
    ) -> Optional[ProcessingResult]:
        """等待处理完成"""
        start_time = asyncio.get_running_loop().time()
        
        while asyncio.get_running_loop().time() - start_time < timeout:
            result = self.processing_results.get(content_id)
            if result and result.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                return result
            
            await asyncio.sleep(0.5)
        
        logger.warning(f"等待处理超时: {content_id}")
        return None
    
    async def _process_worker(self, worker_id: int):
        """处理工作线程"""
        logger.info(f"工作线程 {worker_id} 启动")
        
        while self.is_running:
            try:
                # 从优先级队列获取任务
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                self.active_tasks += 1
                
                logger.info(
                    f"工作线程 {worker_id} 开始处理",
                    content_id=task.content.content_id,
                    content_type=task.content.content_type
                )
                
                # 处理内容
                result = await self._process_task(task)
                
                # 保存结果
                self.processing_results[task.content.content_id] = result
                
                logger.info(
                    f"工作线程 {worker_id} 完成处理",
                    content_id=task.content.content_id,
                    status=result.status,
                    duration=result.processing_time
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 错误: {e}")
            finally:
                self.active_tasks = max(0, self.active_tasks - 1)
        
        logger.info(f"工作线程 {worker_id} 停止")
    
    async def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """处理单个任务"""
        try:
            # 处理内容
            result = await self.processor.process_content(
                task.content,
                task.options
            )
            
            return result
            
        except Exception as e:
            # 错误重试逻辑
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(
                    f"处理失败，重试",
                    content_id=task.content.content_id,
                    retry_count=task.retry_count,
                    error=str(e)
                )
                
                # 重新加入队列
                await self._add_to_priority_queue(task)
                
                # 返回临时失败结果
                return ProcessingResult(
                    content_id=task.content.content_id,
                    status=ProcessingStatus.PROCESSING,
                    extracted_data={},
                    confidence_score=0.0,
                    processing_time=0,
                    error_message=f"重试中 ({task.retry_count}/{task.max_retries})"
                )
            else:
                logger.error(
                    f"处理最终失败",
                    content_id=task.content.content_id,
                    error=str(e)
                )
                
                return ProcessingResult(
                    content_id=task.content.content_id,
                    status=ProcessingStatus.FAILED,
                    extracted_data={},
                    confidence_score=0.0,
                    processing_time=0,
                    error_message=str(e)
                )
    
    async def _add_to_priority_queue(self, task: ProcessingTask):
        """添加任务到优先级队列"""
        # 简单的优先级队列实现
        self.priority_queue.append(task)
        self.priority_queue.sort(key=lambda x: (-x.priority, x.submitted_at or utc_now()))
        
        # 通知有新任务
        await self.processing_queue.put(None)
    
    async def _get_next_task(self) -> Optional[ProcessingTask]:
        """获取下一个待处理任务"""
        try:
            # 等待新任务通知
            await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
            
            # 从优先级队列获取任务
            if self.priority_queue:
                return self.priority_queue.pop(0)
            
        except asyncio.TimeoutError:
            self.logger.debug("等待任务超时", exc_info=True)
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        completed_results = [
            r for r in self.processing_results.values()
            if r.status == ProcessingStatus.COMPLETED
        ]
        failed_results = [
            r for r in self.processing_results.values()
            if r.status == ProcessingStatus.FAILED
        ]
        completed_jobs = len(completed_results)
        failed_jobs = len(failed_results)
        pending_jobs = len(self.priority_queue)
        processing_jobs = self.active_tasks
        total_jobs = pending_jobs + processing_jobs + completed_jobs + failed_jobs
        wait_times: List[float] = []
        
        for result in completed_results:
            submitted_at = self.submission_times.get(result.content_id)
            if submitted_at and result.created_at:
                wait_times.append(max(0.0, (result.created_at - submitted_at).total_seconds()))
        
        average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
        average_processing_time = (
            sum(r.processing_time for r in completed_results) / len(completed_results)
            if completed_results else 0.0
        )
        
        return {
            "total_jobs": total_jobs,
            "pending_jobs": pending_jobs,
            "processing_jobs": processing_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "average_wait_time": average_wait_time,
            "average_processing_time": average_processing_time,
            "is_running": self.is_running,
        }

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, pipeline: ProcessingPipeline):
        self.pipeline = pipeline
        
    async def process_batch(
        self,
        contents: List[MultimodalContent],
        options: Optional[ProcessingOptions] = None,
        wait_for_completion: bool = True,
        timeout: float = 600
    ) -> List[ProcessingResult]:
        """批量处理内容"""
        # 提交批量任务
        content_ids = await self.pipeline.submit_batch(
            contents,
            options,
            priority=2  # 批量任务使用较高优先级
        )
        
        if not wait_for_completion:
            return [
                ProcessingResult(
                    content_id=content_id,
                    status=ProcessingStatus.PENDING,
                    extracted_data={},
                    confidence_score=0.0,
                    processing_time=0.0,
                    created_at=utc_now(),
                )
                for content_id in content_ids
            ]
        
        # 等待所有任务完成
        results = []
        start_time = asyncio.get_running_loop().time()
        
        for content_id in content_ids:
            remaining_timeout = timeout - (asyncio.get_running_loop().time() - start_time)
            if remaining_timeout <= 0:
                logger.warning(f"批量处理超时")
                break
            
            result = await self.pipeline.wait_for_completion(
                content_id,
                timeout=remaining_timeout
            )
            
            if result:
                results.append(result)
            else:
                # 创建超时结果
                results.append(ProcessingResult(
                    content_id=content_id,
                    status=ProcessingStatus.FAILED,
                    extracted_data={},
                    confidence_score=0.0,
                    processing_time=0,
                    error_message="处理超时"
                ))
        
        logger.info(
            f"批量处理完成",
            total=len(contents),
            completed=len([r for r in results if r.status == ProcessingStatus.COMPLETED]),
            failed=len([r for r in results if r.status == ProcessingStatus.FAILED])
        )
        
        return results
