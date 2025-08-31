"""
图数据库核心组件测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# 直接导入避开__init__.py的依赖问题
from ai.knowledge_graph.graph_database import (
    Neo4jGraphDatabase,
    GraphDatabaseConfig,
    GraphDatabaseError,
    ConnectionPoolStats
)


@pytest.mark.unit
class TestGraphDatabaseConfig:
    """图数据库配置测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = GraphDatabaseConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="graph"
        )
        
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database == "graph"
        assert config.connection_timeout == 30
        assert config.max_connection_pool_size == 50


@pytest.mark.unit
class TestNeo4jGraphDatabase:
    """Neo4j图数据库测试"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return GraphDatabaseConfig(
            uri="bolt://localhost:7687",
            username="test",
            password="test",
            database="test",
            connection_timeout=5,
            max_connection_pool_size=10
        )
    
    @pytest.fixture
    def mock_driver(self):
        """Mock驱动器"""
        driver = Mock()
        session = Mock()
        
        # Mock查询结果
        record = Mock()
        record.values.return_value = ["test_value"]
        record.data.return_value = {"test": "data"}
        
        result = Mock()
        result.__aiter__ = AsyncMock(return_value=iter([record]))
        result.consume = AsyncMock()
        result.single = AsyncMock(return_value=record)
        
        session.run = AsyncMock(return_value=result)
        session.close = AsyncMock()
        
        driver.session = Mock(return_value=session)
        driver.close = AsyncMock()
        
        return driver
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, config, mock_driver):
        """测试数据库初始化"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            
            await db.initialize()
            
            assert db.is_initialized
            assert db._driver is not None
    
    @pytest.mark.asyncio
    async def test_database_close(self, config, mock_driver):
        """测试数据库关闭"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            await db.close()
            
            mock_driver.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, config, mock_driver):
        """测试健康检查成功"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            # Mock返回版本信息
            mock_session = mock_driver.session.return_value
            mock_result = Mock()
            mock_record = Mock()
            mock_record.single.return_value = {"version": "5.15.0"}
            mock_result.single = AsyncMock(return_value=mock_record)
            mock_session.run.return_value = mock_result
            
            is_healthy = await db.health_check()
            
            assert is_healthy
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, config, mock_driver):
        """测试健康检查失败"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            # Mock抛出异常
            mock_session = mock_driver.session.return_value
            mock_session.run.side_effect = Exception("Connection failed")
            
            is_healthy = await db.health_check()
            
            assert not is_healthy
    
    @pytest.mark.asyncio
    async def test_execute_read_query_success(self, config, mock_driver):
        """测试读查询执行成功"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            query = "MATCH (n) RETURN n LIMIT 10"
            parameters = {"limit": 10}
            
            result = await db.execute_read_query(query, parameters)
            
            assert isinstance(result, list)
            mock_driver.session.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_write_query_success(self, config, mock_driver):
        """测试写查询执行成功"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            query = "CREATE (n:Person {name: $name}) RETURN n"
            parameters = {"name": "张三"}
            
            result = await db.execute_write_query(query, parameters)
            
            assert isinstance(result, list)
            mock_driver.session.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_transaction_success(self, config, mock_driver):
        """测试事务执行成功"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            async def transaction_func(tx):
                await tx.run("CREATE (n:Person {name: 'Test'})")
                return "success"
            
            result = await db.execute_transaction(transaction_func)
            
            assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_transaction_rollback(self, config, mock_driver):
        """测试事务回滚"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            async def failing_transaction(tx):
                await tx.run("CREATE (n:Person {name: 'Test'})")
                raise Exception("Transaction failed")
            
            with pytest.raises(Exception, match="Transaction failed"):
                await db.execute_transaction(failing_transaction)
    
    def test_connection_stats(self, config, mock_driver):
        """测试连接统计"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            
            stats = db.get_connection_stats()
            
            assert isinstance(stats, dict)
            assert "active_connections" in stats
            assert "created_connections" in stats
    
    @pytest.mark.asyncio
    async def test_database_not_initialized_error(self, config):
        """测试未初始化数据库错误"""
        db = Neo4jGraphDatabase(config)
        
        with pytest.raises(GraphDatabaseError, match="Database not initialized"):
            await db.execute_read_query("MATCH (n) RETURN n")
    
    @pytest.mark.asyncio
    async def test_query_execution_error(self, config, mock_driver):
        """测试查询执行错误"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            # Mock抛出查询错误
            mock_session = mock_driver.session.return_value
            mock_session.run.side_effect = Exception("Invalid query")
            
            with pytest.raises(GraphDatabaseError, match="Query execution failed"):
                await db.execute_read_query("INVALID QUERY")


@pytest.mark.integration
class TestNeo4jIntegration:
    """Neo4j集成测试（需要真实数据库连接）"""
    
    @pytest.mark.neo4j_integration
    @pytest.mark.asyncio
    async def test_real_database_connection(self):
        """测试真实数据库连接（需要Neo4j运行）"""
        config = GraphDatabaseConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="test",
            database="test"
        )
        
        db = Neo4jGraphDatabase(config)
        
        try:
            await db.initialize()
            
            # 测试基本查询
            result = await db.execute_read_query("RETURN 1 as test")
            assert len(result) == 1
            assert result[0]["test"] == 1
            
        finally:
            await db.close()
    
    @pytest.mark.neo4j_integration
    @pytest.mark.asyncio 
    async def test_crud_operations(self):
        """测试CRUD操作"""
        config = GraphDatabaseConfig(
            uri="bolt://localhost:7687",
            username="neo4j", 
            password="test",
            database="test"
        )
        
        db = Neo4jGraphDatabase(config)
        
        try:
            await db.initialize()
            
            # 创建测试节点
            create_result = await db.execute_write_query(
                "CREATE (p:TestPerson {name: $name, id: $id}) RETURN p",
                {"name": "测试用户", "id": "test_001"}
            )
            assert len(create_result) == 1
            
            # 读取测试节点
            read_result = await db.execute_read_query(
                "MATCH (p:TestPerson {id: $id}) RETURN p",
                {"id": "test_001"}
            )
            assert len(read_result) == 1
            assert read_result[0]["p"]["name"] == "测试用户"
            
            # 更新测试节点
            await db.execute_write_query(
                "MATCH (p:TestPerson {id: $id}) SET p.updated = true",
                {"id": "test_001"}
            )
            
            # 验证更新
            updated_result = await db.execute_read_query(
                "MATCH (p:TestPerson {id: $id}) RETURN p.updated as updated",
                {"id": "test_001"}
            )
            assert updated_result[0]["updated"] is True
            
            # 删除测试节点
            await db.execute_write_query(
                "MATCH (p:TestPerson {id: $id}) DELETE p",
                {"id": "test_001"}
            )
            
            # 验证删除
            delete_result = await db.execute_read_query(
                "MATCH (p:TestPerson {id: $id}) RETURN p",
                {"id": "test_001"}
            )
            assert len(delete_result) == 0
            
        finally:
            await db.close()


@pytest.mark.performance
class TestDatabasePerformance:
    """数据库性能测试"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, config, mock_driver):
        """测试并发查询性能"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            async def run_query():
                return await db.execute_read_query("RETURN 1")
            
            # 并发执行多个查询
            tasks = [run_query() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            for result in results:
                assert len(result) >= 0  # Mock返回可能为空
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_connection_pool_behavior(self, config, mock_driver):
        """测试连接池行为"""
        with patch('ai.knowledge_graph.graph_database.GraphDatabase.driver', mock_driver):
            db = Neo4jGraphDatabase(config)
            await db.initialize()
            
            # 获取初始连接统计
            initial_stats = db.get_connection_stats()
            
            # 执行多个查询
            for _ in range(5):
                await db.execute_read_query("RETURN 1")
            
            # 检查连接统计变化
            final_stats = db.get_connection_stats()
            assert isinstance(final_stats, dict)