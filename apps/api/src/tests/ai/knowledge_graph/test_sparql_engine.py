"""
SPARQL引擎测试
"""

import pytest
import asyncio
from ai.knowledge_graph.sparql_engine import (

    SPARQLEngine, 
    SPARQLQuery, 
    QueryType,
    execute_sparql_query
)
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

@pytest.mark.asyncio
async def test_sparql_engine_creation():
    """测试SPARQL引擎创建"""
    engine = SPARQLEngine()
    assert engine is not None
    assert hasattr(engine, 'execute_query')

@pytest.mark.asyncio
async def test_simple_query_execution():
    """测试简单查询执行"""
    query_text = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
    
    result = await execute_sparql_query(
        query_text,
        QueryType.SELECT,
        timeout_seconds=10
    )
    
    assert result.success
    assert result.query_id is not None
    assert result.execution_time_ms >= 0
    assert isinstance(result.results, list)

@pytest.mark.asyncio
async def test_ask_query():
    """测试ASK查询"""
    query_text = "ASK WHERE { ?s ?p ?o }"
    
    result = await execute_sparql_query(
        query_text,
        QueryType.ASK,
        timeout_seconds=10
    )
    
    assert result.success
    assert result.result_type == "boolean"

@pytest.mark.asyncio
async def test_invalid_query():
    """测试无效查询"""
    query_text = "INVALID SPARQL QUERY"
    
    result = await execute_sparql_query(
        query_text,
        QueryType.SELECT,
        timeout_seconds=10
    )
    
    # 应该失败但不抛出异常
    assert not result.success
    assert result.error_message is not None

if __name__ == "__main__":
    setup_logging()
    # 运行简单测试
    async def run_tests():
        logger.info("测试SPARQL引擎...")
        
        try:
            await test_sparql_engine_creation()
            logger.info("✓ SPARQL引擎创建测试通过")
        except Exception as e:
            logger.error(f"✗ SPARQL引擎创建测试失败: {e}")
        
        try:
            await test_simple_query_execution()
            logger.info("✓ 简单查询执行测试通过")
        except Exception as e:
            logger.error(f"✗ 简单查询执行测试失败: {e}")
        
        try:
            await test_ask_query()
            logger.info("✓ ASK查询测试通过")
        except Exception as e:
            logger.error(f"✗ ASK查询测试失败: {e}")
        
        try:
            await test_invalid_query()
            logger.info("✓ 无效查询测试通过")
        except Exception as e:
            logger.error(f"✗ 无效查询测试失败: {e}")
        
        logger.info("SPARQL引擎测试完成")
    
    asyncio.run(run_tests())
