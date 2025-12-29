"""
Redis配置和连接管理
"""

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from .config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

# 全局Redis客户端和连接池
redis_client: redis.Redis | None = None
connection_pool: ConnectionPool | None = None

async def init_redis() -> None:
    """初始化Redis连接"""
    global redis_client, connection_pool

    settings = get_settings()

    logger.info("开始初始化Redis连接", redis_url=settings.REDIS_URL)

    # 创建连接池
    connection_pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=200,
        retry_on_timeout=True,
        decode_responses=True,  # 自动解码为字符串
    )

    # 创建Redis客户端
    redis_client = redis.Redis(connection_pool=connection_pool)

    logger.info("Redis连接初始化完成")

async def close_redis() -> None:
    """关闭Redis连接"""
    global redis_client, connection_pool

    if redis_client:
        logger.info("关闭Redis连接")
        await redis_client.aclose()

    if connection_pool:
        await connection_pool.aclose()

    logger.info("Redis连接已关闭")

def get_redis() -> redis.Redis | None:
    """FastAPI依赖注入函数：获取Redis客户端（使用全局实例）"""
    if not redis_client:
        logger.warning("Redis未初始化")
        return None
    return redis_client

async def test_redis_connection() -> bool:
    """测试Redis连接"""
    try:
        client = get_redis()
        if not client:
            logger.error("Redis客户端未初始化")
            return False

        response = await client.ping()
        if response:
            logger.info("Redis连接测试成功")
            return True
        logger.error("Redis连接测试失败：ping无响应")
        return False
    except Exception as e:
        logger.error("Redis连接测试失败", error=str(e), exc_info=True)
        return False

class RedisCache:
    """Redis缓存操作类"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> str | None:
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error("Redis读取失败", key=key, error=str(e))
            return None

    async def set(self, key: str, value: str, ttl: int = 300) -> bool:
        try:
            return await self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.error("Redis写入失败", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Redis删除失败", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error("Redis存在性检查失败", key=key, error=str(e))
            return False

def get_cache() -> RedisCache:
    """FastAPI依赖注入函数：获取Redis缓存操作实例（复用全局客户端）"""
    from .config import get_settings

    settings = get_settings()

    if settings.TESTING:
        return None
    redis_instance = get_redis()
    return RedisCache(redis_instance) if redis_instance else None
