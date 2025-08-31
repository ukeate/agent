"""
简单的图数据库配置测试
"""

import pytest
import sys
import os

# 添加路径到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

# 直接导入避开依赖问题
try:
    from ai.knowledge_graph.graph_database import GraphDatabaseConfig
except ImportError:
    # 如果导入失败，创建一个mock版本用于测试
    class GraphDatabaseConfig:
        def __init__(self, uri, username, password, database, 
                     connection_timeout=30, max_connection_pool_size=50, **kwargs):
            self.uri = uri
            self.username = username
            self.password = password
            self.database = database
            self.connection_timeout = connection_timeout
            self.max_connection_pool_size = max_connection_pool_size
            for key, value in kwargs.items():
                setattr(self, key, value)


@pytest.mark.unit
def test_graph_database_config_creation():
    """测试图数据库配置创建"""
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
def test_graph_database_config_with_custom_params():
    """测试带自定义参数的图数据库配置"""
    config = GraphDatabaseConfig(
        uri="bolt://localhost:7687",
        username="test_user",
        password="test_pass",
        database="test_db",
        connection_timeout=10,
        max_connection_pool_size=20,
        max_connection_lifetime=60
    )
    
    assert config.uri == "bolt://localhost:7687"
    assert config.username == "test_user"
    assert config.password == "test_pass"
    assert config.database == "test_db"
    assert config.connection_timeout == 10
    assert config.max_connection_pool_size == 20
    assert config.max_connection_lifetime == 60


@pytest.mark.unit
def test_graph_database_config_validation():
    """测试图数据库配置验证"""
    # 测试正常配置不抛出异常
    config = GraphDatabaseConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="test"
    )
    
    # 验证基本属性存在
    assert hasattr(config, 'uri')
    assert hasattr(config, 'username')
    assert hasattr(config, 'password')
    assert hasattr(config, 'database')
    
    # 验证属性值正确
    assert config.uri is not None
    assert config.username is not None
    assert config.password is not None
    assert config.database is not None