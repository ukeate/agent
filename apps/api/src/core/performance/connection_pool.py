"""
连接池管理
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from asyncpg import Pool, create_pool
from redis.asyncio import ConnectionPool as RedisConnectionPool
from redis.asyncio import Redis
from src.core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class ConnectionPoolManager:
    """连接池管理器"""
    
    def __init__(self):
        self.db_pool: Optional[Pool] = None
        self.redis_pool: Optional[RedisConnectionPool] = None
        self.max_db_connections = settings.VECTOR_MAX_CONNECTIONS
        self.min_db_connections = 5
        self.max_redis_connections = 50
        self.connection_stats = {
            "db": {
                "active": 0,
                "idle": 0,
                "total": 0,
                "acquired": 0,
                "released": 0,
                "errors": 0
            },
            "redis": {
                "active": 0,
                "total": 0,
                "errors": 0
            }
        }
    
    async def initialize(self):
        """初始化连接池"""
        await self._init_db_pool()
        await self._init_redis_pool()
    
    async def _init_db_pool(self):
        """初始化数据库连接池"""
        try:
            # 解析数据库URL
            db_url = settings.DATABASE_URL.replace(
                "postgresql+asyncpg://",
                "postgresql://"
            )
            
            self.db_pool = await create_pool(
                db_url,
                min_size=self.min_db_connections,
                max_size=self.max_db_connections,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            
            logger.info(
                "Database connection pool initialized",
                min_size=self.min_db_connections,
                max_size=self.max_db_connections
            )
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise
    
    async def _init_redis_pool(self):
        """初始化Redis连接池"""
        try:
            self.redis_pool = RedisConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=self.max_redis_connections,
                decode_responses=True
            )
            
            logger.info(
                "Redis connection pool initialized",
                max_connections=self.max_redis_connections
            )
        except Exception as e:
            logger.error("Failed to initialize Redis pool", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_db_connection(self):
        """获取数据库连接"""
        if not self.db_pool:
            await self._init_db_pool()
        
        self.connection_stats["db"]["acquired"] += 1
        
        try:
            async with self.db_pool.acquire() as connection:
                self.connection_stats["db"]["active"] += 1
                yield connection
        except Exception as e:
            self.connection_stats["db"]["errors"] += 1
            logger.error("Database connection error", error=str(e))
            raise
        finally:
            self.connection_stats["db"]["active"] -= 1
            self.connection_stats["db"]["released"] += 1
    
    async def get_redis_connection(self) -> Redis:
        """获取Redis连接"""
        if not self.redis_pool:
            await self._init_redis_pool()
        
        try:
            redis = Redis(connection_pool=self.redis_pool)
            self.connection_stats["redis"]["active"] += 1
            return redis
        except Exception as e:
            self.connection_stats["redis"]["errors"] += 1
            logger.error("Redis connection error", error=str(e))
            raise
    
    async def execute_query(self, query: str, *args, **kwargs):
        """执行数据库查询"""
        async with self.get_db_connection() as conn:
            return await conn.fetch(query, *args, **kwargs)
    
    async def execute_command(self, query: str, *args, **kwargs):
        """执行数据库命令"""
        async with self.get_db_connection() as conn:
            return await conn.execute(query, *args, **kwargs)
    
    async def close(self):
        """关闭连接池"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")
        
        if self.redis_pool:
            await self.redis_pool.aclose()
            logger.info("Redis connection pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        stats = self.connection_stats.copy()
        
        if self.db_pool:
            stats["db"].update({
                "pool_size": self.db_pool.get_size(),
                "pool_min_size": self.db_pool.get_min_size(),
                "pool_max_size": self.db_pool.get_max_size(),
                "pool_free_size": self.db_pool.get_idle_size()
            })
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health = {
            "db": False,
            "redis": False
        }
        
        # 检查数据库连接
        try:
            async with self.get_db_connection() as conn:
                await conn.fetchval("SELECT 1")
                health["db"] = True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
        
        # 检查Redis连接
        try:
            redis = await self.get_redis_connection()
            await redis.ping()
            health["redis"] = True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
        
        return health
    
    async def optimize_pools(self):
        """优化连接池"""
        # 根据使用情况动态调整连接池大小
        stats = self.get_pool_stats()
        
        # 数据库连接池优化
        if self.db_pool and stats["db"]["active"] > stats["db"]["pool_size"] * 0.8:
            # 如果活跃连接超过80%，考虑增加连接池大小
            logger.info(
                "Database pool optimization suggested",
                active=stats["db"]["active"],
                pool_size=stats["db"]["pool_size"]
            )
        
        # Redis连接池优化
        if stats["redis"]["errors"] > 10:
            # 如果错误过多，重新初始化连接池
            logger.warning("Redis pool has too many errors, reinitializing")
            await self._init_redis_pool()

# 全局连接池管理器实例
pool_manager = ConnectionPoolManager()
