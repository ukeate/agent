"""
数据库配置和连接管理
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from .config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

class Base(DeclarativeBase):
    """数据库模型基类"""
    ...

# 全局数据库引擎和会话工厂
engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None

async def init_database() -> None:
    """初始化数据库连接"""
    global engine, async_session_factory

    settings = get_settings()

    logger.info("开始初始化数据库连接")

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

    logger.info("数据库连接初始化完成")

async def close_database() -> None:
    """关闭数据库连接"""
    global engine

    if engine:
        logger.info("关闭数据库连接")
        await engine.dispose()
        logger.info("数据库连接已关闭")

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
    async with get_db_session() as session:
        yield session

async def test_database_connection() -> bool:
    """测试数据库连接"""
    try:
        async with get_db_session() as session:
            # 执行简单查询测试连接
            result = await session.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.info("数据库连接测试成功")
                return True
            else:
                logger.error("数据库连接测试失败：返回值异常")
                return False
    except Exception as e:
        logger.error("数据库连接测试失败", error=str(e), exc_info=True)
        return False
