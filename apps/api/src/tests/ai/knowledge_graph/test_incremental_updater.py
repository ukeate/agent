"""
增量更新器测试
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.ai.knowledge_graph.incremental_updater import (
    IncrementalUpdater,
    ConflictResolutionStrategy,
    UpdateResult,
    EntityUpdate,
    RelationUpdate,
    UpdateConflict
)

@pytest.mark.unit
class TestEntityUpdate:
    """实体更新测试"""
    
    def test_entity_update_creation(self):
        """测试实体更新创建"""
        update = EntityUpdate(
            entity_id="test_001",
            canonical_form="张三",
            entity_type="PERSON",
            properties={"age": 30, "occupation": "工程师"},
            confidence=0.95,
            source="document_001"
        )
        
        assert update.entity_id == "test_001"
        assert update.canonical_form == "张三"
        assert update.entity_type == "PERSON"
        assert update.properties["age"] == 30
        assert update.confidence == 0.95
        assert update.source == "document_001"
    
    def test_entity_update_to_dict(self):
        """测试实体更新序列化"""
        update = EntityUpdate(
            entity_id="test_001",
            canonical_form="张三",
            entity_type="PERSON",
            properties={"age": 30},
            confidence=0.95,
            source="document_001"
        )
        
        update_dict = update.to_dict()
        
        assert update_dict["entity_id"] == "test_001"
        assert update_dict["canonical_form"] == "张三"
        assert "timestamp" in update_dict

@pytest.mark.unit
class TestRelationUpdate:
    """关系更新测试"""
    
    def test_relation_update_creation(self):
        """测试关系更新创建"""
        update = RelationUpdate(
            relation_id="rel_001",
            relation_type="WORKS_FOR",
            source_entity_id="entity_001",
            target_entity_id="entity_002",
            properties={"since": "2020", "position": "工程师"},
            confidence=0.90,
            source="document_001"
        )
        
        assert update.relation_id == "rel_001"
        assert update.relation_type == "WORKS_FOR"
        assert update.source_entity_id == "entity_001"
        assert update.target_entity_id == "entity_002"
        assert update.confidence == 0.90

@pytest.mark.unit
class TestIncrementalUpdater:
    """增量更新器测试"""
    
    @pytest.fixture
    def mock_graph_db(self):
        """Mock图数据库"""
        db = Mock()
        db.execute_read_query = AsyncMock()
        db.execute_write_query = AsyncMock()
        db.execute_transaction = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_quality_manager(self):
        """Mock质量管理器"""
        quality_manager = Mock()
        quality_manager.validate_entity = AsyncMock(return_value=True)
        quality_manager.validate_relation = AsyncMock(return_value=True)
        return quality_manager
    
    @pytest.fixture
    def updater(self, mock_graph_db, mock_quality_manager):
        """增量更新器夹具"""
        return IncrementalUpdater(mock_graph_db, mock_quality_manager)
    
    @pytest.mark.asyncio
    async def test_process_entity_update_new(self, updater, mock_graph_db):
        """测试处理新实体更新"""
        # Mock实体不存在
        mock_graph_db.execute_read_query.return_value = []
        mock_graph_db.execute_write_query.return_value = [{"created": True}]
        
        entity_update = EntityUpdate(
            entity_id="new_entity",
            canonical_form="新实体",
            entity_type="PERSON",
            properties={"age": 25},
            confidence=0.95,
            source="doc_001"
        )
        
        result = await updater.process_entity_update(entity_update)
        
        assert result.operation == "create"
        assert result.success
        assert not result.conflicts
    
    @pytest.mark.asyncio
    async def test_process_entity_update_existing_no_conflict(self, updater, mock_graph_db):
        """测试处理现有实体更新（无冲突）"""
        # Mock实体存在且相似度高
        mock_graph_db.execute_read_query.return_value = [{
            "id": "existing_entity",
            "canonical_form": "张三",
            "type": "PERSON",
            "properties": {"age": 30},
            "confidence": 0.90,
            "last_updated": utc_now().isoformat()
        }]
        mock_graph_db.execute_write_query.return_value = [{"updated": True}]
        
        entity_update = EntityUpdate(
            entity_id="existing_entity",
            canonical_form="张三",
            entity_type="PERSON",
            properties={"age": 31, "occupation": "工程师"},  # 更新信息
            confidence=0.95,
            source="doc_002"
        )
        
        result = await updater.process_entity_update(entity_update)
        
        assert result.operation == "update"
        assert result.success
    
    @pytest.mark.asyncio
    async def test_process_entity_update_with_conflict(self, updater, mock_graph_db):
        """测试处理实体更新冲突"""
        # Mock实体存在但信息冲突
        mock_graph_db.execute_read_query.return_value = [{
            "id": "conflict_entity",
            "canonical_form": "张三",
            "type": "PERSON", 
            "properties": {"age": 30, "occupation": "医生"},  # 冲突信息
            "confidence": 0.98,  # 更高置信度
            "last_updated": utc_now().isoformat()
        }]
        
        entity_update = EntityUpdate(
            entity_id="conflict_entity",
            canonical_form="张三",
            entity_type="PERSON",
            properties={"age": 25, "occupation": "工程师"},  # 冲突信息
            confidence=0.85,  # 较低置信度
            source="doc_003"
        )
        
        result = await updater.process_entity_update(entity_update)
        
        assert len(result.conflicts) > 0
        assert result.conflicts[0].field in ["age", "occupation"]
    
    @pytest.mark.asyncio
    async def test_process_relation_update_new(self, updater, mock_graph_db):
        """测试处理新关系更新"""
        # Mock关系不存在
        mock_graph_db.execute_read_query.return_value = []
        mock_graph_db.execute_write_query.return_value = [{"created": True}]
        
        relation_update = RelationUpdate(
            relation_id="new_relation",
            relation_type="WORKS_FOR",
            source_entity_id="entity_001",
            target_entity_id="entity_002",
            properties={"since": "2024"},
            confidence=0.90,
            source="doc_001"
        )
        
        result = await updater.process_relation_update(relation_update)
        
        assert result.operation == "create"
        assert result.success
    
    @pytest.mark.asyncio
    async def test_intelligent_entity_merge(self, updater, mock_graph_db):
        """测试智能实体合并"""
        # Mock找到相似实体
        mock_graph_db.execute_read_query.side_effect = [
            # 第一次查询：查找相似实体
            [{
                "id": "similar_entity",
                "canonical_form": "张三丰",
                "type": "PERSON",
                "properties": {"age": 35},
                "embedding": [0.1, 0.2, 0.3] * 100
            }],
            # 第二次查询：获取实体详细信息
            [{
                "id": "similar_entity",
                "canonical_form": "张三丰", 
                "type": "PERSON",
                "properties": {"age": 35},
                "confidence": 0.92
            }]
        ]
        
        entity_update = EntityUpdate(
            entity_id="new_entity",
            canonical_form="张三",  # 相似但不完全相同
            entity_type="PERSON",
            properties={"age": 33},
            confidence=0.88,
            source="doc_004",
            embedding=[0.12, 0.18, 0.28] * 100  # 相似的嵌入向量
        )
        
        result = await updater.intelligent_entity_merge(entity_update, similarity_threshold=0.85)
        
        # 应该建议合并
        assert "merge_candidate" in result
    
    @pytest.mark.asyncio
    async def test_resolve_conflict_highest_confidence(self, updater):
        """测试解决冲突 - 最高置信度策略"""
        conflict = UpdateConflict(
            field="age",
            existing_value=30,
            new_value=25,
            existing_confidence=0.95,
            new_confidence=0.80
        )
        
        resolved = await updater.resolve_conflict(
            conflict,
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        
        # 应该选择现有值（更高置信度）
        assert resolved == 30
    
    @pytest.mark.asyncio
    async def test_resolve_conflict_newest_value(self, updater):
        """测试解决冲突 - 最新值策略"""
        conflict = UpdateConflict(
            field="age",
            existing_value=30,
            new_value=25,
            existing_timestamp=datetime(2024, 1, 1),
            new_timestamp=datetime(2024, 2, 1)  # 更新
        )
        
        resolved = await updater.resolve_conflict(
            conflict,
            strategy=ConflictResolutionStrategy.NEWEST_VALUE
        )
        
        # 应该选择新值（更新的时间戳）
        assert resolved == 25
    
    @pytest.mark.asyncio
    async def test_batch_update(self, updater, mock_graph_db):
        """测试批量更新"""
        # Mock数据库返回
        mock_graph_db.execute_read_query.return_value = []  # 没有现有实体
        mock_graph_db.execute_transaction.return_value = "success"
        
        # 准备批量更新数据
        entity_updates = [
            EntityUpdate("e1", "实体1", "PERSON", {"age": 20}, 0.9, "doc1"),
            EntityUpdate("e2", "实体2", "PERSON", {"age": 25}, 0.85, "doc2")
        ]
        
        relation_updates = [
            RelationUpdate("r1", "KNOWS", "e1", "e2", {}, 0.8, "doc1")
        ]
        
        results = await updater.batch_update(entity_updates, relation_updates)
        
        assert len(results["entity_results"]) == 2
        assert len(results["relation_results"]) == 1
        assert all(r.success for r in results["entity_results"])
    
    def test_calculate_similarity(self, updater):
        """测试相似度计算"""
        entity1 = {
            "canonical_form": "张三",
            "properties": {"age": 30, "occupation": "工程师"}
        }
        
        entity2 = {
            "canonical_form": "张三丰",
            "properties": {"age": 32, "occupation": "工程师"}
        }
        
        similarity = updater._calculate_similarity(entity1, entity2)
        
        # 应该有一定相似度（姓名相似，职业相同）
        assert 0.0 < similarity < 1.0
    
    def test_calculate_embedding_similarity(self, updater):
        """测试嵌入向量相似度计算"""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.8, 0.6, 0.0, 0.0]  # 与第一个向量相似
        embedding3 = [0.0, 0.0, 1.0, 0.0]  # 与第一个向量不同
        
        similarity1 = updater._calculate_embedding_similarity(embedding1, embedding2)
        similarity2 = updater._calculate_embedding_similarity(embedding1, embedding3)
        
        # embedding1和embedding2应该更相似
        assert similarity1 > similarity2
        assert 0.0 <= similarity1 <= 1.0
        assert 0.0 <= similarity2 <= 1.0
    
    @pytest.mark.asyncio
    async def test_rollback_update(self, updater, mock_graph_db):
        """测试回滚更新"""
        mock_graph_db.execute_write_query.return_value = [{"rollback": "success"}]
        
        update_id = "update_123"
        
        success = await updater.rollback_update(update_id)
        
        assert success
        mock_graph_db.execute_write_query.assert_called_once()

@pytest.mark.integration  
class TestIncrementalUpdaterIntegration:
    """增量更新器集成测试"""
    
    @pytest.mark.neo4j_integration
    @pytest.mark.asyncio
    async def test_real_entity_update_workflow(self, test_neo4j_config):
        """测试真实实体更新工作流"""
        from src.ai.knowledge_graph.graph_database import Neo4jGraphDatabase
        from src.ai.knowledge_graph.quality_manager import QualityManager
        
        db = Neo4jGraphDatabase(test_neo4j_config)
        quality_manager = QualityManager(db)
        updater = IncrementalUpdater(db, quality_manager)
        
        try:
            await db.initialize()
            
            # 创建新实体
            entity_update = EntityUpdate(
                entity_id="test_entity_001",
                canonical_form="测试人员",
                entity_type="PERSON",
                properties={"age": 28, "department": "工程部"},
                confidence=0.92,
                source="test_document"
            )
            
            result = await updater.process_entity_update(entity_update)
            
            assert result.success
            assert result.operation == "create"
            
            # 更新实体
            update2 = EntityUpdate(
                entity_id="test_entity_001",
                canonical_form="测试人员",
                entity_type="PERSON", 
                properties={"age": 29, "department": "研发部", "title": "高级工程师"},
                confidence=0.94,
                source="updated_document"
            )
            
            result2 = await updater.process_entity_update(update2)
            
            assert result2.success
            assert result2.operation == "update"
            
            # 清理测试数据
            await db.execute_write_query(
                "MATCH (e:PERSON {id: $id}) DELETE e",
                {"id": "test_entity_001"}
            )
            
        finally:
            await db.close()

@pytest.mark.performance
class TestUpdaterPerformance:
    """更新器性能测试"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_batch_update_performance(self, updater, mock_graph_db):
        """测试大批量更新性能"""
        import time
        
        # Mock数据库快速响应
        mock_graph_db.execute_read_query.return_value = []
        mock_graph_db.execute_transaction.return_value = "success"
        
        # 创建大量更新
        entity_updates = [
            EntityUpdate(f"entity_{i}", f"实体{i}", "PERSON", {"index": i}, 0.9, "doc")
            for i in range(1000)
        ]
        
        start_time = time.time()
        results = await updater.batch_update(entity_updates, [])
        end_time = time.time()
        
        # 验证性能
        assert len(results["entity_results"]) == 1000
        assert (end_time - start_time) < 5.0  # 应该在5秒内完成
    
    @pytest.mark.slow
    def test_similarity_calculation_performance(self, updater):
        """测试相似度计算性能"""
        import time
        
        # 创建复杂实体
        entity1 = {
            "canonical_form": "张三" * 100,  # 长文本
            "properties": {f"prop_{i}": f"value_{i}" for i in range(100)}  # 多属性
        }
        
        entity2 = {
            "canonical_form": "李四" * 100,
            "properties": {f"prop_{i}": f"different_value_{i}" for i in range(100)}
        }
        
        start_time = time.time()
        
        # 计算多次相似度
        for _ in range(100):
            updater._calculate_similarity(entity1, entity2)
        
        end_time = time.time()
        
        # 性能应该合理
        assert (end_time - start_time) < 1.0  # 100次计算在1秒内
