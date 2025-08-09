"""
数据库配置和连接管理
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """数据库模型基类"""

    pass


# 全局数据库引擎和会话工厂
engine = None
async_session_factory = None


async def init_database() -> None:
    """初始化数据库连接"""
    global engine, async_session_factory

    settings = get_settings()

    logger.info("Initializing database connection", database_url=settings.DATABASE_URL)

    # 创建异步引擎
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,  # 在调试模式下打印SQL
        pool_size=5,  # 连接池最小连接数
        max_overflow=15,  # 连接池最大额外连接数
        pool_pre_ping=True,  # 连接前验证连接有效性
        pool_recycle=3600,  # 连接回收时间（1小时）
    )

    # 创建会话工厂
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    logger.info("Database connection initialized successfully")


async def close_database() -> None:
    """关闭数据库连接"""
    global engine

    if engine:
        logger.info("Closing database connection")
        await engine.dispose()
        logger.info("Database connection closed")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话的上下文管理器"""
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI依赖注入函数：获取数据库会话"""
    from .config import get_settings

    settings = get_settings()

    if settings.TESTING:
        # 测试模式下返回None，由端点处理
        yield None
    else:
        async with get_db_session() as session:
            yield session


async def test_database_connection() -> bool:
    """测试数据库连接"""
    try:
        async with get_db_session() as session:
            # 执行简单查询测试连接
            result = await session.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.info("Database connection test successful")
                return True
            else:
                logger.error("Database connection test failed: unexpected result")
                return False
    except Exception as e:
        logger.error("Database connection test failed", error=str(e), exc_info=True)
        return False
