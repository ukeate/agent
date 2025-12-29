import asyncio
from src.core.database import Base, engine, init_database, close_database
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
初始化数据库表的简单脚本
"""

async def create_tables():
    """创建数据库表"""
    logger.info("开始创建数据库表...")
    
    try:
        # 初始化数据库连接
        await init_database()
        
        # 从core.database获取已初始化的engine
        from src.core.database import engine
        
        # 确保导入所有模型
        from src.models.database import api_key, event_tracking, experiment, session, supervisor, user, workflow
        from src.ai.langgraph import checkpoints
        from src.db import emotional_memory_models
        
        logger.info("导入的表模型:")
        for table_name in Base.metadata.tables.keys():
            logger.info(f"  - {table_name}")
        
        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(emotional_memory_models.Base.metadata.create_all)
        
        logger.info("✅ 数据库表创建成功！")
        
    except Exception as e:
        logger.error(f"❌ 创建数据库表失败: {e}")
        raise
    finally:
        await close_database()

if __name__ == "__main__":
    setup_logging()
    asyncio.run(create_tables())
