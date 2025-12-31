"""
背压任务处理器
处理经过流控的任务队列，实现高吞吐量场景的任务调度
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from .flow_control import FlowController, TaskInfo
from typing import TYPE_CHECKING

from src.core.logging import get_logger
logger = get_logger(__name__)

# 延迟导入避免循环依赖

if TYPE_CHECKING:
    from .enterprise import EnterpriseAgentManager

class BackpressureTaskProcessor:
    """背压任务处理器"""
    
    def __init__(self, 
                 flow_controller: FlowController,
                 enterprise_manager: "EnterpriseAgentManager"):
        self.flow_controller = flow_controller
        self.enterprise_manager = enterprise_manager
        self.running = False
        self.worker_tasks = []
        self.max_workers = 5  # 最大并发处理任务数
    
    async def start(self):
        """启动任务处理器"""
        if self.running:
            return
        
        self.running = True
        
        # 启动多个工作协程
        for i in range(self.max_workers):
            worker = create_task_with_logging(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        logger.info(f"Backpressure task processor started with {self.max_workers} workers")
    
    async def stop(self):
        """停止任务处理器"""
        self.running = False
        
        # 取消所有工作任务
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Backpressure task processor stopped")
    
    async def _worker_loop(self, worker_id: str):
        """工作协程循环"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # 从流控队列获取任务
                task_info = await self.flow_controller.get_task()
                
                if task_info is None:
                    continue
                
                # 处理任务
                await self._process_task(worker_id, task_info)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {e}")
                await asyncio.sleep(1.0)  # 短暂延迟后重试
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, worker_id: str, task_info: TaskInfo):
        """处理单个任务"""
        start_time = time.time()
        success = False
        
        try:
            logger.debug(f"Worker {worker_id} processing task {task_info.task_id}")
            
            # 从任务元数据中提取参数
            task_data = task_info.metadata.get('data', {})
            pool_id = task_data.get('pool_id')
            task_type = task_data.get('task_type')
            description = task_data.get('description')
            input_data = task_data.get('input_data', {})
            timeout_seconds = task_data.get('timeout_seconds', 300)
            
            # 检查任务是否过期
            if task_info.deadline and utc_now() > task_info.deadline:
                logger.warning(f"Task {task_info.task_id} expired, skipping")
                return
            
            # 选择智能体
            agent_id = await self.enterprise_manager._select_agent_from_pool(pool_id)
            if not agent_id:
                logger.warning(f"No available agents in pool {pool_id} for task {task_info.task_id}")
                return
            
            # 实际提交任务到智能体
            actual_task_id = await self.enterprise_manager.submit_task(
                agent_id=agent_id,
                task_type=task_type,
                description=description,
                input_data=input_data,
                timeout_seconds=timeout_seconds
            )
            
            # 等待任务完成 (简化实现，实际可能需要更复杂的状态跟踪)
            # 这里我们假设任务会通过事件机制异步完成
            logger.info(f"Task {task_info.task_id} submitted as {actual_task_id} to agent {agent_id}")
            success = True
            
        except Exception as e:
            logger.error(f"Failed to process task {task_info.task_id}: {e}")
            success = False
            
        finally:
            # 计算执行时间
            execution_time = (time.time() - start_time) * 1000  # 毫秒
            
            # 通知流控器任务完成
            await self.flow_controller.complete_task(
                task_info.task_id, 
                success=success, 
                execution_time=execution_time
            )
            
            logger.debug(f"Task {task_info.task_id} completed in {execution_time:.2f}ms, success: {success}")

# 全局任务处理器实例
_task_processor: Optional[BackpressureTaskProcessor] = None

def get_task_processor() -> Optional[BackpressureTaskProcessor]:
    """获取全局任务处理器实例"""
    return _task_processor

async def init_task_processor(flow_controller: FlowController, 
                            enterprise_manager: "EnterpriseAgentManager"):
    """初始化全局任务处理器"""
    global _task_processor
    _task_processor = BackpressureTaskProcessor(flow_controller, enterprise_manager)
    await _task_processor.start()

async def shutdown_task_processor():
    """关闭全局任务处理器"""
    global _task_processor
    if _task_processor:
        await _task_processor.stop()
        _task_processor = None
