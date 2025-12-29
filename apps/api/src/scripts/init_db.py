import asyncio
import sys
from pathlib import Path
from src.core.database import engine, Base, init_database, close_database
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
数据库初始化脚本
创建所需的数据库表
"""

sys.path.append(str(Path(__file__).parent.parent))

async def create_tables():
    """创建所有数据库表"""
    logger.info("开始创建数据库表...")
    
    try:
        # 确保所有模型都被导入
        logger.info("导入数据库模型...")
        
        # 导入所有模型以确保它们被注册到Base.metadata
        import models.database.workflow
        import ai.langgraph.checkpoints
        
        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("数据库表创建成功！")
        
        # 打印创建的表信息
        logger.info("\n已创建的表:")
        for table in Base.metadata.tables.keys():
            logger.info(f"  - {table}")
            
    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        raise

async def main():
    """主函数"""
    try:
        # 初始化数据库连接
        await init_database()
        
        # 创建表
        await create_tables()
        
    finally:
        # 关闭数据库连接
        await close_database()

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
