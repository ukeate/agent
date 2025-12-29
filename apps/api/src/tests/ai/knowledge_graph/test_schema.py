"""
图模式管理测试
"""

import pytest
from unittest.mock import Mock, AsyncMock
from src.ai.knowledge_graph.schema import (
    GraphSchema,
    SchemaManager,
    NodeDefinition,
    RelationshipDefinition,
    IndexDefinition,
    ConstraintDefinition,
    SchemaValidationError
)

@pytest.mark.unit
class TestGraphSchema:
    """图模式测试"""
    
    def test_schema_creation(self):
        """测试模式创建"""
        schema = GraphSchema()
        
        assert schema.nodes == {}
        assert schema.relationships == {}
        assert schema.indexes == {}
        assert schema.constraints == {}
    
    def test_add_node_definition(self):
        """测试添加节点定义"""
        schema = GraphSchema()
        
        node_def = NodeDefinition(
            label="Person",
            properties={
                "name": "string",
                "age": "integer", 
                "email": "string"
            },
            required_properties=["name"],
            description="人员节点"
        )
        
        schema.add_node("Person", node_def)
        
        assert "Person" in schema.nodes
        assert schema.nodes["Person"] == node_def
    
    def test_add_relationship_definition(self):
        """测试添加关系定义"""
        schema = GraphSchema()
        
        rel_def = RelationshipDefinition(
            type="WORKS_FOR",
            properties={
                "since": "date",
                "position": "string"
            },
            description="工作关系"
        )
        
        schema.add_relationship("WORKS_FOR", rel_def)
        
        assert "WORKS_FOR" in schema.relationships
        assert schema.relationships["WORKS_FOR"] == rel_def
    
    def test_add_index_definition(self):
        """测试添加索引定义"""
        schema = GraphSchema()
        
        index_def = IndexDefinition(
            name="person_name_index",
            labels=["Person"],
            properties=["name"],
            type="btree",
            description="人员姓名索引"
        )
        
        schema.add_index("person_name_index", index_def)
        
        assert "person_name_index" in schema.indexes
        assert schema.indexes["person_name_index"] == index_def
    
    def test_add_constraint_definition(self):
        """测试添加约束定义"""
        schema = GraphSchema()
        
        constraint_def = ConstraintDefinition(
            name="person_id_unique",
            labels=["Person"],
            properties=["id"],
            type="unique",
            description="人员ID唯一约束"
        )
        
        schema.add_constraint("person_id_unique", constraint_def)
        
        assert "person_id_unique" in schema.constraints
        assert schema.constraints["person_id_unique"] == constraint_def
    
    def test_schema_to_dict(self):
        """测试模式序列化"""
        schema = GraphSchema()
        
        # 添加节点定义
        node_def = NodeDefinition(
            label="Person",
            properties={"name": "string"},
            required_properties=["name"]
        )
        schema.add_node("Person", node_def)
        
        schema_dict = schema.to_dict()
        
        assert "nodes" in schema_dict
        assert "relationships" in schema_dict
        assert "indexes" in schema_dict
        assert "constraints" in schema_dict
        assert "Person" in schema_dict["nodes"]
    
    def test_schema_from_dict(self):
        """测试模式反序列化"""
        schema_dict = {
            "nodes": {
                "Person": {
                    "label": "Person",
                    "properties": {"name": "string"},
                    "required_properties": ["name"],
                    "description": ""
                }
            },
            "relationships": {},
            "indexes": {},
            "constraints": {}
        }
        
        schema = GraphSchema.from_dict(schema_dict)
        
        assert "Person" in schema.nodes
        assert schema.nodes["Person"].label == "Person"

@pytest.mark.unit 
class TestSchemaManager:
    """模式管理器测试"""
    
    @pytest.fixture
    def mock_graph_db(self):
        """Mock图数据库"""
        db = Mock()
        db.execute_write_query = AsyncMock()
        db.execute_read_query = AsyncMock()
        return db
    
    @pytest.fixture
    def schema_manager(self, mock_graph_db):
        """模式管理器夹具"""
        return SchemaManager(mock_graph_db)
    
    @pytest.mark.asyncio
    async def test_create_schema(self, schema_manager, mock_graph_db):
        """测试创建模式"""
        schema = GraphSchema()
        
        # 添加节点定义
        node_def = NodeDefinition(
            label="Person", 
            properties={"name": "string", "age": "integer"},
            required_properties=["name"]
        )
        schema.add_node("Person", node_def)
        
        # 添加索引定义
        index_def = IndexDefinition(
            name="person_name_index",
            labels=["Person"],
            properties=["name"],
            type="btree"
        )
        schema.add_index("person_name_index", index_def)
        
        # 添加约束定义
        constraint_def = ConstraintDefinition(
            name="person_id_unique",
            labels=["Person"], 
            properties=["id"],
            type="unique"
        )
        schema.add_constraint("person_id_unique", constraint_def)
        
        await schema_manager.create_schema(schema)
        
        # 验证调用了正确的数据库操作
        assert mock_graph_db.execute_write_query.call_count >= 2  # 至少创建索引和约束
    
    @pytest.mark.asyncio
    async def test_get_current_schema(self, schema_manager, mock_graph_db):
        """测试获取当前模式"""
        # Mock返回现有索引和约束
        mock_graph_db.execute_read_query.side_effect = [
            # 返回索引信息
            [
                {
                    "name": "person_name_index",
                    "labelsOrTypes": ["Person"],
                    "properties": ["name"],
                    "type": "btree"
                }
            ],
            # 返回约束信息  
            [
                {
                    "name": "person_id_unique",
                    "labelsOrTypes": ["Person"],
                    "properties": ["id"],
                    "type": "unique"
                }
            ],
            # 返回节点标签
            [{"label": "Person"}],
            # 返回关系类型
            [{"type": "WORKS_FOR"}]
        ]
        
        schema = await schema_manager.get_current_schema()
        
        assert isinstance(schema, GraphSchema)
        assert mock_graph_db.execute_read_query.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_validate_schema_success(self, schema_manager):
        """测试模式验证成功"""
        schema = GraphSchema()
        
        node_def = NodeDefinition(
            label="Person",
            properties={"name": "string"},
            required_properties=["name"]
        )
        schema.add_node("Person", node_def)
        
        # 应该不抛出异常
        schema_manager.validate_schema(schema)
    
    def test_validate_schema_missing_required_property(self, schema_manager):
        """测试模式验证缺少必需属性"""
        schema = GraphSchema()
        
        # 节点定义有必需属性但为空
        node_def = NodeDefinition(
            label="Person",
            properties={},  # 空属性
            required_properties=["name"]  # 但要求name属性
        )
        schema.add_node("Person", node_def)
        
        with pytest.raises(SchemaValidationError, match="Required property"):
            schema_manager.validate_schema(schema)
    
    def test_validate_schema_invalid_property_type(self, schema_manager):
        """测试模式验证无效属性类型"""
        schema = GraphSchema()
        
        # 使用无效的属性类型
        node_def = NodeDefinition(
            label="Person",
            properties={"age": "invalid_type"},  # 无效类型
            required_properties=[]
        )
        schema.add_node("Person", node_def)
        
        with pytest.raises(SchemaValidationError, match="Invalid property type"):
            schema_manager.validate_schema(schema)
    
    @pytest.mark.asyncio
    async def test_drop_index(self, schema_manager, mock_graph_db):
        """测试删除索引"""
        await schema_manager.drop_index("test_index")
        
        mock_graph_db.execute_write_query.assert_called_once()
        call_args = mock_graph_db.execute_write_query.call_args[0]
        assert "DROP INDEX" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_drop_constraint(self, schema_manager, mock_graph_db):
        """测试删除约束"""
        await schema_manager.drop_constraint("test_constraint")
        
        mock_graph_db.execute_write_query.assert_called_once()
        call_args = mock_graph_db.execute_write_query.call_args[0]
        assert "DROP CONSTRAINT" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_schema_diff(self, schema_manager, mock_graph_db):
        """测试模式差异对比"""
        # Mock当前模式
        mock_graph_db.execute_read_query.side_effect = [
            # 索引
            [{"name": "old_index", "labelsOrTypes": ["Person"], "properties": ["name"]}],
            # 约束
            [{"name": "old_constraint", "labelsOrTypes": ["Person"], "properties": ["id"]}],
            # 节点
            [{"label": "Person"}],
            # 关系
            [{"type": "WORKS_FOR"}]
        ]
        
        # 创建新模式
        new_schema = GraphSchema()
        index_def = IndexDefinition(
            name="new_index",
            labels=["Person"],
            properties=["email"],
            type="btree"
        )
        new_schema.add_index("new_index", index_def)
        
        diff = await schema_manager.schema_diff(new_schema)
        
        assert "indexes_to_create" in diff
        assert "indexes_to_drop" in diff
        assert "constraints_to_create" in diff
        assert "constraints_to_drop" in diff
    
    @pytest.mark.asyncio
    async def test_apply_schema_migration(self, schema_manager, mock_graph_db):
        """测试应用模式迁移"""
        migration_plan = {
            "indexes_to_create": [
                IndexDefinition("new_index", ["Person"], ["name"], "btree")
            ],
            "indexes_to_drop": ["old_index"],
            "constraints_to_create": [
                ConstraintDefinition("new_constraint", ["Person"], ["id"], "unique")
            ],
            "constraints_to_drop": ["old_constraint"]
        }
        
        await schema_manager.apply_migration(migration_plan)
        
        # 验证执行了正确的操作数量
        # 应该调用: 删除旧索引、删除旧约束、创建新索引、创建新约束
        assert mock_graph_db.execute_write_query.call_count == 4

@pytest.mark.integration
class TestSchemaIntegration:
    """模式集成测试"""
    
    @pytest.mark.neo4j_integration
    @pytest.mark.asyncio
    async def test_real_schema_operations(self, test_neo4j_config):
        """测试真实模式操作"""
        from src.ai.knowledge_graph.graph_database import Neo4jGraphDatabase
        
        db = Neo4jGraphDatabase(test_neo4j_config)
        schema_manager = SchemaManager(db)
        
        try:
            await db.initialize()
            
            # 创建测试模式
            schema = GraphSchema()
            
            # 添加节点定义
            node_def = NodeDefinition(
                label="TestNode",
                properties={"name": "string", "value": "integer"},
                required_properties=["name"]
            )
            schema.add_node("TestNode", node_def)
            
            # 添加索引
            index_def = IndexDefinition(
                name="test_node_name_index",
                labels=["TestNode"],
                properties=["name"],
                type="btree"
            )
            schema.add_index("test_node_name_index", index_def)
            
            # 创建模式
            await schema_manager.create_schema(schema)
            
            # 获取当前模式并验证
            current_schema = await schema_manager.get_current_schema()
            assert "test_node_name_index" in current_schema.indexes
            
            # 清理测试数据
            await schema_manager.drop_index("test_node_name_index")
            
        finally:
            await db.close()

@pytest.mark.performance
class TestSchemaPerformance:
    """模式性能测试"""
    
    @pytest.mark.slow
    def test_large_schema_validation(self):
        """测试大型模式验证性能"""
        schema = GraphSchema()
        
        # 创建大量节点定义
        for i in range(1000):
            node_def = NodeDefinition(
                label=f"Node{i}",
                properties={"id": "string", "value": "integer"},
                required_properties=["id"]
            )
            schema.add_node(f"Node{i}", node_def)
        
        schema_manager = SchemaManager(Mock())
        
        # 验证性能应该在合理范围内
        import time
        start_time = time.time()
        schema_manager.validate_schema(schema)
        end_time = time.time()
        
        # 验证时间应该少于1秒
        assert (end_time - start_time) < 1.0
