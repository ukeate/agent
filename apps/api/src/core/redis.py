"""
Redis配置和连接管理
"""


import redis.asyncio as redis
import structlog
from redis.asyncio import ConnectionPool

from .config import get_settings

logger = structlog.get_logger(__name__)

# 全局Redis客户端和连接池
redis_client: redis.Redis | None = None
connection_pool: ConnectionPool | None = None


async def init_redis() -> None:
    """初始化Redis连接"""
    global redis_client, connection_pool

    settings = get_settings()

    logger.info("Initializing Redis connection", redis_url=settings.REDIS_URL)

    # 创建连接池
    connection_pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=20,
        retry_on_timeout=True,
        decode_responses=True,  # 自动解码为字符串
    )

    # 创建Redis客户端
    redis_client = redis.Redis(connection_pool=connection_pool)

    logger.info("Redis connection initialized successfully")


async def close_redis() -> None:
    """关闭Redis连接"""
    global redis_client, connection_pool

    if redis_client:
        logger.info("Closing Redis connection")
        await redis_client.close()

    if connection_pool:
        await connection_pool.disconnect()

    logger.info("Redis connection closed")


def get_redis() -> redis.Redis:
    """FastAPI依赖注入函数：获取Redis客户端"""
    if not redis_client:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return redis_client


async def test_redis_connection() -> bool:
    """测试Redis连接"""
    try:
        if not redis_client:
            logger.error("Redis client not initialized")
            return False

        # 测试ping命令
        response = await redis_client.ping()
        if response:
            logger.info("Redis connection test successful")
            return True
        else:
            logger.error("Redis connection test failed: no response to ping")
            return False
    except Exception as e:
        logger.error("Redis connection test failed", error=str(e), exc_info=True)
        return False


class RedisCache:
    """Redis缓存操作类"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> str | None:
        """获取缓存值"""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            return None

    async def set(self, key: str, value: str, ttl: int = 300) -> bool:
        """设置缓存值"""
        try:
            return await self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error("Redis exists failed", key=key, error=str(e))
            return False


def get_cache() -> RedisCache:
    """FastAPI依赖注入函数：获取Redis缓存操作实例"""
    from .config import get_settings

    settings = get_settings()

    if settings.TESTING:
        # 测试模式下返回None，由端点处理
        return None
    else:
        redis_instance = get_redis()
        return RedisCache(redis_instance)
