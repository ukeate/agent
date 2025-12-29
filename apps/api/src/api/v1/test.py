"""
测试端点 - 用于验证异步处理能力
"""

import asyncio
from typing import Any
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from src.api.exceptions import ValidationError
from src.core.database import get_db
from src.core.redis import RedisCache, get_cache

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/test", tags=["测试"])

@router.get("/async-db")
async def test_async_database(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """测试异步数据库操作"""
    from src.core.config import get_settings

    settings = get_settings()

    if settings.TESTING or db is None:
        raise HTTPException(
            status_code=503,
            detail="database not available for async test"
        )

    try:
        # 执行简单的异步数据库查询
        result = await db.execute(
            text("SELECT NOW() as current_time, version() as db_version")
        )
        row = result.fetchone()

        return {
            "status": "success",
            "message": "异步数据库连接测试成功",
            "data": {
                "current_time": str(row.current_time),
                "database_version": row.db_version,
            },
        }
    except Exception as e:
        logger.error("Async database test failed", error=str(e), exc_info=True)
        raise ValidationError("数据库连接测试失败", details={"error": str(e)}) from e

@router.get("/async-redis")
async def test_async_redis(cache: RedisCache = Depends(get_cache)) -> dict[str, Any]:
    """测试异步Redis操作"""
    from src.core.config import get_settings

    settings = get_settings()

    if settings.TESTING or cache is None:
        raise HTTPException(
            status_code=503,
            detail="redis not available for async test"
        )

    try:
        test_key = "test:async:operation"
        test_value = "Hello Redis from async FastAPI!"

        # 设置值
        set_result = await cache.set(test_key, test_value, ttl=60)
        if not set_result:
            raise ValidationError("Redis设置操作失败")

        # 获取值
        get_result = await cache.get(test_key)
        if get_result != test_value:
            raise ValidationError("Redis获取操作失败")

        # 检查键存在
        exists_result = await cache.exists(test_key)
        if not exists_result:
            raise ValidationError("Redis键存在检查失败")

        # 删除键
        delete_result = await cache.delete(test_key)

        return {
            "status": "success",
            "message": "异步Redis连接测试成功",
            "data": {
                "set_result": set_result,
                "get_result": get_result,
                "exists_result": exists_result,
                "delete_result": delete_result,
            },
        }
    except ValidationError:
        raise
    except Exception as e:
        logger.error("Async Redis test failed", error=str(e), exc_info=True)
        raise ValidationError("Redis连接测试失败", details={"error": str(e)}) from e

@router.get("/concurrent")
async def test_concurrent_requests() -> dict[str, Any]:
    """测试并发请求处理能力"""
    start = asyncio.get_running_loop().time()
    concurrency = 50
    await asyncio.gather(*(asyncio.sleep(0.02) for _ in range(concurrency)))
    duration_ms = (asyncio.get_running_loop().time() - start) * 1000
    return {
        "status": "success",
        "message": "并发任务执行完成",
        "data": {"concurrency": concurrency, "duration_ms": duration_ms},
    }

@router.get("/mixed-async")
async def test_mixed_async_operations(
    db: AsyncSession = Depends(get_db), cache: RedisCache = Depends(get_cache)
) -> dict[str, Any]:
    """测试混合异步操作（数据库 + Redis + 计算）"""
    from src.core.config import get_settings

    settings = get_settings()

    if settings.TESTING or db is None or cache is None:
        raise HTTPException(
            status_code=503,
            detail="dependencies not available for mixed async test"
        )

    try:
        # 并发执行多种异步操作
        db_task = db.execute(text("SELECT NOW() as timestamp"))
        redis_task = cache.set("test:mixed", "mixed_operation_test", ttl=30)
        compute_task = asyncio.sleep(0.05)

        # 等待所有任务完成
        db_result, redis_result, _ = await asyncio.gather(
            db_task, redis_task, compute_task
        )

        db_row = db_result.fetchone()

        return {
            "status": "success",
            "message": "混合异步操作测试成功",
            "data": {
                "database_timestamp": str(db_row.timestamp),
                "redis_operation": redis_result,
                "compute_completed": True,
            },
        }
    except Exception as e:
        logger.error("Mixed async operations test failed", error=str(e), exc_info=True)
        raise ValidationError("混合异步操作测试失败", details={"error": str(e)}) from e
