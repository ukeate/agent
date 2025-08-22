#!/usr/bin/env python3
"""
数据库初始化脚本
创建所需的数据库表
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import engine, Base, init_database, close_database


async def create_tables():
    """创建所有数据库表"""
    print("开始创建数据库表...")
    
    try:
        # 确保所有模型都被导入
        print("导入数据库模型...")
        
        # 导入所有模型以确保它们被注册到Base.metadata
        import models.database.workflow
        import ai.langgraph.checkpoints
        
        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("数据库表创建成功！")
        
        # 打印创建的表信息
        print("\n已创建的表:")
        for table in Base.metadata.tables.keys():
            print(f"  - {table}")
            
    except Exception as e:
        print(f"创建数据库表失败: {e}")
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
    asyncio.run(main())