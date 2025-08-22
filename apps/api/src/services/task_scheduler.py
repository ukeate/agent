"""
后台任务调度器
定期执行pending任务
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import structlog

from src.services.task_executor import task_executor

logger = structlog.get_logger(__name__)


class TaskScheduler:
    """后台任务调度器"""
    
    def __init__(self, poll_interval: int = 15, max_tasks_per_cycle: int = 3):
        self.poll_interval = poll_interval  # 轮询间隔（秒）
        self.max_tasks_per_cycle = max_tasks_per_cycle  # 每次最多执行任务数
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_execution = None
        
    async def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("任务调度器已在运行中")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("任务调度器已启动", poll_interval=self.poll_interval)
    
    async def stop(self):
        """停止调度器"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("任务调度器已停止")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("任务调度器循环已开始")
        
        while self._running:
            try:
                # 记录执行时间
                cycle_start = datetime.now()
                
                # 执行pending任务
                results = await task_executor.execute_pending_tasks(self.max_tasks_per_cycle)
                
                # 记录执行结果
                if results:
                    successful_tasks = sum(1 for r in results if r.get("success", False))
                    failed_tasks = len(results) - successful_tasks
                    
                    logger.info("任务调度周期完成", 
                              total_tasks=len(results),
                              successful=successful_tasks,
                              failed=failed_tasks,
                              cycle_duration=(datetime.now() - cycle_start).total_seconds())
                
                self._last_execution = cycle_start
                
                # 等待下次轮询
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                logger.info("任务调度器被取消")
                break
            except Exception as e:
                logger.error("任务调度器循环异常", error=str(e))
                # 出现异常后等待一段时间再继续
                await asyncio.sleep(min(self.poll_interval, 30))
    
    async def force_execution(self) -> dict:
        """强制执行一次任务调度"""
        try:
            logger.info("强制执行任务调度")
            start_time = datetime.now()
            
            results = await task_executor.execute_pending_tasks(self.max_tasks_per_cycle)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "executed_tasks": len(results),
                "successful_tasks": sum(1 for r in results if r.get("success", False)),
                "duration_seconds": duration,
                "results": results
            }
            
        except Exception as e:
            logger.error("强制执行任务调度失败", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_status(self) -> dict:
        """获取调度器状态"""
        return {
            "running": self._running,
            "poll_interval": self.poll_interval,
            "max_tasks_per_cycle": self.max_tasks_per_cycle,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "executor_status": task_executor.get_status()
        }


# 全局任务调度器实例
task_scheduler = TaskScheduler()