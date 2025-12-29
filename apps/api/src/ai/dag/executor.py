"""
DAG任务执行引擎
提供任务编排和并发执行能力
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from .models import DAGWorkflow, DAGTask, TaskStatus, DAGNode

from src.core.logging import get_logger
logger = get_logger(__name__)

class DAGExecutor:
    """DAG执行引擎"""
    
    def __init__(self, max_workers: int = 4, task_timeout: int = 300):
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_handlers: Dict[str, Callable] = {}
        self._running_tasks: Set[str] = set()
        
    def register_task_handler(self, task_type: str, handler: Callable):
        """注册任务类型处理器"""
        self._task_handlers[task_type] = handler
    
    async def execute_workflow(self, workflow: DAGWorkflow) -> Dict[str, Any]:
        """执行DAG工作流"""
        if not workflow.validate_dag():
            raise ValueError("工作流包含循环依赖，无法执行")
        
        logger.info(f"开始执行工作流: {workflow.name} ({workflow.id})")
        start_time = utc_now()
        
        try:
            # 执行任务直到完成
            while not workflow.is_completed():
                ready_tasks = workflow.get_ready_tasks()
                
                if not ready_tasks:
                    # 检查是否有运行中的任务
                    running_tasks = [
                        task for task in workflow.tasks.values() 
                        if task.status == TaskStatus.RUNNING
                    ]
                    
                    if not running_tasks:
                        # 没有可执行任务也没有运行中的任务，可能出现死锁
                        logger.error("工作流出现死锁状态")
                        break
                    
                    # 等待运行中的任务完成
                    await asyncio.sleep(0.1)
                    continue
                
                # 并发执行准备好的任务
                tasks_futures = []
                for task in ready_tasks:
                    if task.node.id not in self._running_tasks:
                        future = self._execute_task(task)
                        tasks_futures.append(future)
                        self._running_tasks.add(task.node.id)
                
                # 等待至少一个任务完成
                if tasks_futures:
                    await asyncio.wait(tasks_futures, return_when=asyncio.FIRST_COMPLETED)
            
            end_time = utc_now()
            duration = (end_time - start_time).total_seconds()
            
            # 生成执行报告
            completed_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.COMPLETED]
            failed_tasks = workflow.get_failed_tasks()
            
            result = {
                "workflow_id": workflow.id,
                "status": "completed" if not failed_tasks else "failed",
                "duration_seconds": duration,
                "total_tasks": len(workflow.tasks),
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "task_results": {
                    task_id: task.result 
                    for task_id, task in workflow.tasks.items() 
                    if task.result is not None
                },
                "failed_task_errors": {
                    task_id: task.error 
                    for task_id, task in workflow.tasks.items() 
                    if task.error is not None
                }
            }
            
            logger.info(f"工作流执行完成: {workflow.name}, 耗时: {duration:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"工作流执行异常: {str(e)}")
            raise
    
    async def _execute_task(self, task: DAGTask) -> Any:
        """执行单个任务"""
        task.start()
        logger.info(f"开始执行任务: {task.node.name} ({task.node.id})")
        
        try:
            # 获取任务处理器
            task_type = task.node.metadata.get("type", "default")
            handler = self._task_handlers.get(task_type)
            
            if not handler:
                raise ValueError(f"未找到任务类型处理器: {task_type}")
            
            # 准备任务参数
            task_params = task.node.metadata.get("params", {})
            
            # 执行任务（支持同步和异步处理器）
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(task_params), 
                    timeout=self.task_timeout
                )
            else:
                # 在线程池中执行同步任务
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    handler, 
                    task_params
                )
            
            task.complete(result)
            logger.info(f"任务执行成功: {task.node.name}")
            
        except asyncio.TimeoutError:
            error_msg = f"任务执行超时: {task.node.name}"
            logger.error(error_msg)
            task.fail(error_msg)
            
        except Exception as e:
            error_msg = f"任务执行失败: {task.node.name}, 错误: {str(e)}"
            logger.error(error_msg)
            task.fail(error_msg)
            
            # 尝试重试
            if task.can_retry():
                logger.info(f"任务将重试: {task.node.name}, 重试次数: {task.retry_count + 1}")
                task.retry()
                # 重新调度任务执行
                await asyncio.sleep(1)  # 等待1秒后重试
                return await self._execute_task(task)
        
        finally:
            self._running_tasks.discard(task.node.id)
        
        return task.result
    
    def create_simple_workflow(self, name: str, tasks_config: List[Dict[str, Any]]) -> DAGWorkflow:
        """创建简单的串行工作流"""
        workflow = DAGWorkflow(name=name, description=f"简单串行工作流: {name}")
        
        previous_node_id = None
        for i, task_config in enumerate(tasks_config):
            node = DAGNode(
                name=task_config.get("name", f"Task-{i+1}"),
                description=task_config.get("description", ""),
                metadata=task_config.get("metadata", {})
            )
            
            node_id = workflow.add_node(node)
            
            # 添加串行依赖
            if previous_node_id:
                workflow.add_dependency(previous_node_id, node_id)
            
            previous_node_id = node_id
        
        return workflow
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        logger.info("DAG执行器已关闭")

# 默认任务处理器示例
async def default_task_handler(params: Dict[str, Any]) -> Any:
    """默认任务处理器"""
    await asyncio.sleep(0.1)  # 模拟任务执行
    return {"status": "completed", "params": params}

def echo_task_handler(params: Dict[str, Any]) -> Any:
    """回显任务处理器（同步）"""
    return {"echo": params.get("message", "Hello World")}

# 预注册默认处理器
def get_default_executor() -> DAGExecutor:
    """获取预配置的默认执行器"""
    executor = DAGExecutor()
    executor.register_task_handler("default", default_task_handler)
    executor.register_task_handler("echo", echo_task_handler)
    return executor
