"""异步工具：在线程池中运行同步IO"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Awaitable

from src.core.logging import get_logger

# 限制线程数，防止占满默认执行器
_io_executor = ThreadPoolExecutor(max_workers=8)
_logger = get_logger(__name__)
_background_tasks: set[asyncio.Task] = set()

async def run_sync_io(func: Callable[[], Any]) -> Any:
    """将同步阻塞IO放入独立线程池运行"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_io_executor, func)

def create_task_with_logging(
    coro: Awaitable[Any],
    *,
    logger: Any | None = None,
    keep_reference: bool = True,
) -> asyncio.Task:
    """创建后台任务并记录异常"""
    task = asyncio.create_task(coro)
    if keep_reference:
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    log = logger or _logger

    def _done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            log.error("后台任务异常", exc_info=True)

    task.add_done_callback(_done)
    return task
