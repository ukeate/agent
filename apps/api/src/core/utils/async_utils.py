"""异步工具：在线程池中运行同步IO"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

# 限制线程数，防止占满默认执行器
_io_executor = ThreadPoolExecutor(max_workers=8)

async def run_sync_io(func: Callable[[], Any]) -> Any:
    """将同步阻塞IO放入独立线程池运行"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_io_executor, func)
