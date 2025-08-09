#!/usr/bin/env python3
"""
数据库种子数据生成脚本
创建默认的智能体配置
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "apps" / "api" / "src"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)


async def create_default_agents():
    """创建默认智能体配置"""
    settings = get_settings()
    
    # 创建异步数据库引擎
    engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # 默认智能体配置
    default_agents = [
        {
            "name": "代码专家",
            "role": "code_expert",
            "capabilities": ["代码生成", "代码审查", "调试", "重构"],
            "configuration": {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 4096,
                "tools": ["code_execution", "file_operations"],
                "system_prompt": "你是一位专业的代码专家，专注于高质量代码的生成、审查和优化。"
            }
        },
        {
            "name": "系统架构师",
            "role": "architect",
            "capabilities": ["系统设计", "技术选型", "架构评估", "文档编写"],
            "configuration": {
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 4096,
                "tools": ["documentation", "diagram_generation"],
                "system_prompt": "你是一位经验丰富的系统架构师，负责设计可扩展、可维护的软件架构。"
            }
        },
        {
            "name": "文档专家",
            "role": "doc_expert",
            "capabilities": ["技术文档", "API文档", "用户手册", "代码注释"],
            "configuration": {
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 4096,
                "tools": ["markdown_generation", "file_operations"],
                "system_prompt": "你是一位专业的技术文档专家，擅长创建清晰、准确、易懂的技术文档。"
            }
        },
        {
            "name": "任务调度器",
            "role": "supervisor",
            "capabilities": ["任务分解", "智能体协调", "工作流管理", "质量控制"],
            "configuration": {
                "model": "gpt-4o-mini",
                "temperature": 0.6,
                "max_tokens": 4096,
                "tools": ["task_management", "agent_coordination"],
                "system_prompt": "你是智能体团队的协调者，负责任务分解、分配和质量管控。"
            }
        },
        {
            "name": "知识检索专家",
            "role": "rag_specialist",
            "capabilities": ["语义搜索", "知识整合", "答案生成", "内容验证"],
            "configuration": {
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 4096,
                "tools": ["vector_search", "knowledge_management"],
                "system_prompt": "你是知识检索和整合专家，擅长从大量信息中找到相关内容并生成准确答案。"
            }
        }
    ]
    
    try:
        async with async_session() as session:
            # 检查agents表是否存在
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'agents'
                );
            """))
            table_exists = result.scalar()
            
            if not table_exists:
                logger.warning("agents表不存在，跳过种子数据插入")
                return
            
            # 清空现有智能体数据
            await session.execute(text("DELETE FROM agents"))
            
            # 插入默认智能体
            for agent in default_agents:
                insert_sql = text("""
                    INSERT INTO agents (name, role, capabilities, configuration, status)
                    VALUES (:name, :role, :capabilities, :configuration, 'idle')
                """)
                
                await session.execute(insert_sql, {
                    "name": agent["name"],
                    "role": agent["role"],
                    "capabilities": agent["capabilities"],
                    "configuration": json.dumps(agent["configuration"], ensure_ascii=False)
                })
            
            await session.commit()
            logger.info(f"成功插入 {len(default_agents)} 个默认智能体")
            
    except Exception as e:
        logger.error(f"插入种子数据失败: {e}")
        raise
    finally:
        await engine.dispose()


async def main():
    """主函数"""
    logger.info("开始执行数据库种子数据生成...")
    
    try:
        await create_default_agents()
        logger.info("种子数据生成完成")
    except Exception as e:
        logger.error(f"种子数据生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())