"""
后台任务调度器
定期执行pending任务
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Optional
from src.services.task_executor import task_executor

from src.core.logging import get_logger
logger = get_logger(__name__)

class TaskScheduler:
    """后台任务调度器"""
    
    def __init__(self, poll_interval: int = 120, max_tasks_per_cycle: int = 3):
        self.poll_interval = poll_interval  # 轮询间隔（秒） - 从15秒增加到120秒（2分钟）
        self.max_tasks_per_cycle = max_tasks_per_cycle  # 每次最多执行任务数
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_execution = None
        self._consecutive_empty_cycles = 0  # 连续空循环计数
        
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
                raise
        
        logger.info("任务调度器已停止")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        logger.info("任务调度器循环已开始")
        
        while self._running:
            try:
                # 记录执行时间
                cycle_start = utc_now()
                
                # 执行pending任务
                results = await task_executor.execute_pending_tasks(self.max_tasks_per_cycle)
                
                # 记录执行结果并实施智能退避
                if results and len(results) > 0:
                    # 有任务执行，重置空循环计数
                    self._consecutive_empty_cycles = 0
                    
                    successful_tasks = sum(1 for r in results if r.get("success", False))
                    failed_tasks = len(results) - successful_tasks
                    
                    logger.info("任务调度周期完成", 
                              total_tasks=len(results),
                              successful=successful_tasks,
                              failed=failed_tasks,
                              cycle_duration=(utc_now() - cycle_start).total_seconds())
                else:
                    # 无任务执行，增加空循环计数
                    self._consecutive_empty_cycles += 1
                    
                    # 每10次空循环记录一次日志
                    if self._consecutive_empty_cycles % 10 == 0:
                        logger.debug("任务调度器连续空循环", 
                                   consecutive_empty_cycles=self._consecutive_empty_cycles)
                
                self._last_execution = cycle_start
                
                # 智能退避：连续空循环时增加等待时间
                if self._consecutive_empty_cycles > 5:
                    # 连续5次以上空循环时，将轮询间隔翻倍（最大600秒=10分钟）
                    adaptive_interval = min(self.poll_interval * 2, 600)
                    logger.debug("采用智能退避机制", 
                               base_interval=self.poll_interval,
                               adaptive_interval=adaptive_interval,
                               empty_cycles=self._consecutive_empty_cycles)
                    await asyncio.sleep(adaptive_interval)
                else:
                    # 正常轮询间隔
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
            start_time = utc_now()
            
            results = await task_executor.execute_pending_tasks(self.max_tasks_per_cycle)
            
            duration = (utc_now() - start_time).total_seconds()
            
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
