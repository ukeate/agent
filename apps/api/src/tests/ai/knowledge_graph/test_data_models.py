"""
知识图谱数据模型测试

测试核心数据结构的功能和验证逻辑
"""

import pytest
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4
from ai.knowledge_graph.data_models import (
    Entity, Relation, KnowledgeGraph, TripleStore,
    EntityType, RelationType,
    EntityModel, RelationModel,
    ExtractionRequest, ExtractionResponse,
    BatchProcessingResult, BatchProcessingRequest, BatchProcessingResponse
)

class TestEntityType:
    """实体类型测试"""
    
    def test_entity_types_exist(self):
        """测试所有实体类型是否存在"""
        expected_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "GPE", "DATE", "TIME",
            "MONEY", "PERCENTAGE", "FACILITY", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LAW", "LANGUAGE", "NATIONALITY", "RELIGION",
            "CARDINAL", "ORDINAL", "QUANTITY", "MISC",
            "COUNTRY", "CITY", "PROVINCE", "UNIVERSITY", "COMPANY"
        ]
        
        for entity_type in expected_types:
            assert hasattr(EntityType, entity_type)
            assert EntityType[entity_type].value == entity_type

class TestRelationType:
    """关系类型测试"""
    
    def test_basic_relation_types(self):
        """测试基础关系类型"""
        basic_types = [
            "LOCATED_IN", "WORKS_FOR", "BORN_IN", "FOUNDED_BY",
            "PART_OF", "OWNS", "MARRIED_TO", "PARENT_OF"
        ]
        
        for relation_type in basic_types:
            assert hasattr(RelationType, relation_type)

class TestEntity:
    """实体测试"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(
            text="苹果公司",
            label=EntityType.COMPANY,
            start=0,
            end=3,
            confidence=0.95
        )
        
        assert entity.text == "苹果公司"
        assert entity.label == EntityType.COMPANY
        assert entity.start == 0
        assert entity.end == 3
        assert entity.confidence == 0.95
        assert entity.entity_id is not None
        assert isinstance(entity.entity_id, str)
    
    def test_entity_with_metadata(self):
        """测试带元数据的实体"""
        metadata = {"source": "test", "confidence_breakdown": {"model1": 0.9, "model2": 0.8}}
        
        entity = Entity(
            text="北京",
            label=EntityType.CITY,
            start=5,
            end=7,
            confidence=0.88,
            canonical_form="北京市",
            language="zh",
            metadata=metadata
        )
        
        assert entity.canonical_form == "北京市"
        assert entity.language == "zh"
        assert entity.metadata == metadata
    
    def test_entity_to_dict(self):
        """测试实体转字典"""
        entity = Entity(
            text="张三",
            label=EntityType.PERSON,
            start=0,
            end=2,
            confidence=0.9
        )
        
        entity_dict = entity.to_dict()
        
        assert entity_dict["text"] == "张三"
        assert entity_dict["label"] == "PERSON"
        assert entity_dict["start"] == 0
        assert entity_dict["end"] == 2
        assert entity_dict["confidence"] == 0.9
        assert "entity_id" in entity_dict

class TestRelation:
    """关系测试"""
    
    def test_relation_creation(self):
        """测试关系创建"""
        subject = Entity(
            text="张三",
            label=EntityType.PERSON,
            start=0,
            end=2,
            confidence=0.9
        )
        
        obj = Entity(
            text="苹果公司",
            label=EntityType.COMPANY,
            start=5,
            end=8,
            confidence=0.95
        )
        
        relation = Relation(
            subject=subject,
            predicate=RelationType.WORKS_FOR,
            object=obj,
            confidence=0.85,
            context="张三在苹果公司工作",
            source_sentence="张三在苹果公司工作"
        )
        
        assert relation.subject == subject
        assert relation.predicate == RelationType.WORKS_FOR
        assert relation.object == obj
        assert relation.confidence == 0.85
        assert relation.context == "张三在苹果公司工作"
        assert relation.relation_id is not None
    
    def test_relation_to_dict(self):
        """测试关系转字典"""
        subject = Entity(
            text="北京",
            label=EntityType.CITY,
            start=0,
            end=2,
            confidence=0.9
        )
        
        obj = Entity(
            text="中国",
            label=EntityType.COUNTRY,
            start=5,
            end=7,
            confidence=0.95
        )
        
        relation = Relation(
            subject=subject,
            predicate=RelationType.LOCATED_IN,
            object=obj,
            confidence=0.9,
            context="北京位于中国",
            source_sentence="北京位于中国"
        )
        
        relation_dict = relation.to_dict()
        
        assert relation_dict["predicate"] == "LOCATED_IN"
        assert relation_dict["confidence"] == 0.9
        assert "subject" in relation_dict
        assert "object" in relation_dict
        assert "relation_id" in relation_dict

class TestTripleStore:
    """三元组存储测试"""
    
    def test_triple_store_creation(self):
        """测试三元组存储创建"""
        store = TripleStore()
        assert len(store.triples) == 0
        assert store.size() == 0
    
    def test_add_relation(self):
        """测试添加关系"""
        store = TripleStore()
        
        subject = Entity(
            text="张三",
            label=EntityType.PERSON,
            start=0,
            end=2,
            confidence=0.9
        )
        
        obj = Entity(
            text="苹果公司",
            label=EntityType.COMPANY,
            start=5,
            end=8,
            confidence=0.95
        )
        
        relation = Relation(
            subject=subject,
            predicate=RelationType.WORKS_FOR,
            object=obj,
            confidence=0.85,
            context="张三在苹果公司工作",
            source_sentence="张三在苹果公司工作"
        )
        
        store.add_relation(relation)
        
        assert store.size() == 1
        assert len(store.triples) == 1
        
        triple = list(store.triples)[0]
        assert triple[0] == "张三"  # subject
        assert triple[1] == "WORKS_FOR"  # predicate
        assert triple[2] == "苹果公司"  # object
    
    def test_query_triples(self):
        """测试查询三元组"""
        store = TripleStore()
        
        # 添加多个关系
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("李四", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 0, 3, 0.95),
            Entity("北京", EntityType.CITY, 0, 2, 0.9)
        ]
        
        relations = [
            Relation(entities[0], RelationType.WORKS_FOR, entities[2], 0.85, "ctx1", "sent1"),
            Relation(entities[1], RelationType.WORKS_FOR, entities[2], 0.88, "ctx2", "sent2"),
            Relation(entities[2], RelationType.LOCATED_IN, entities[3], 0.92, "ctx3", "sent3")
        ]
        
        for relation in relations:
            store.add_relation(relation)
        
        # 查询特定主语的关系
        zhang_relations = store.query_triples(subject="张三")
        assert len(zhang_relations) == 1
        assert zhang_relations[0][1] == "WORKS_FOR"
        
        # 查询特定宾语的关系
        apple_relations = store.query_triples(object="苹果公司")
        assert len(apple_relations) == 2
        
        # 查询特定谓词的关系
        works_relations = store.query_triples(predicate="WORKS_FOR")
        assert len(works_relations) == 2

class TestKnowledgeGraph:
    """知识图谱测试"""
    
    def test_knowledge_graph_creation(self):
        """测试知识图谱创建"""
        kg = KnowledgeGraph()
        
        assert kg.graph_id is not None
        assert len(kg.entities) == 0
        assert len(kg.relations) == 0
        assert kg.triple_store.size() == 0
    
    def test_add_entity(self):
        """测试添加实体"""
        kg = KnowledgeGraph()
        
        entity = Entity(
            text="张三",
            label=EntityType.PERSON,
            start=0,
            end=2,
            confidence=0.9
        )
        
        kg.add_entity(entity)
        
        assert len(kg.entities) == 1
        assert entity.entity_id in kg.entities
        assert kg.entities[entity.entity_id] == entity
    
    def test_add_relation(self):
        """测试添加关系"""
        kg = KnowledgeGraph()
        
        subject = Entity("张三", EntityType.PERSON, 0, 2, 0.9)
        obj = Entity("苹果公司", EntityType.COMPANY, 5, 8, 0.95)
        
        relation = Relation(
            subject=subject,
            predicate=RelationType.WORKS_FOR,
            object=obj,
            confidence=0.85,
            context="张三在苹果公司工作",
            source_sentence="张三在苹果公司工作"
        )
        
        kg.add_relation(relation)
        
        # 检查关系和实体都被添加
        assert len(kg.relations) == 1
        assert len(kg.entities) == 2
        assert kg.triple_store.size() == 1
    
    def test_get_entity_relations(self):
        """测试获取实体关系"""
        kg = KnowledgeGraph()
        
        # 创建实体
        zhang = Entity("张三", EntityType.PERSON, 0, 2, 0.9)
        apple = Entity("苹果公司", EntityType.COMPANY, 5, 8, 0.95)
        beijing = Entity("北京", EntityType.CITY, 10, 12, 0.92)
        
        # 创建关系
        relation1 = Relation(zhang, RelationType.WORKS_FOR, apple, 0.85, "ctx1", "sent1")
        relation2 = Relation(apple, RelationType.LOCATED_IN, beijing, 0.88, "ctx2", "sent2")
        
        kg.add_relation(relation1)
        kg.add_relation(relation2)
        
        # 获取张三相关的关系
        zhang_relations = kg.get_entity_relations(zhang.entity_id)
        assert len(zhang_relations) == 1
        assert zhang_relations[0].predicate == RelationType.WORKS_FOR
        
        # 获取苹果公司相关的关系
        apple_relations = kg.get_entity_relations(apple.entity_id)
        assert len(apple_relations) == 2
    
    def test_merge_knowledge_graphs(self):
        """测试合并知识图谱"""
        kg1 = KnowledgeGraph()
        kg2 = KnowledgeGraph()
        
        # 向kg1添加实体和关系
        zhang = Entity("张三", EntityType.PERSON, 0, 2, 0.9)
        apple = Entity("苹果公司", EntityType.COMPANY, 5, 8, 0.95)
        relation1 = Relation(zhang, RelationType.WORKS_FOR, apple, 0.85, "ctx1", "sent1")
        kg1.add_relation(relation1)
        
        # 向kg2添加实体和关系
        li = Entity("李四", EntityType.PERSON, 0, 2, 0.9)
        google = Entity("谷歌", EntityType.COMPANY, 5, 7, 0.93)
        relation2 = Relation(li, RelationType.WORKS_FOR, google, 0.87, "ctx2", "sent2")
        kg2.add_relation(relation2)
        
        # 合并
        kg1.merge(kg2)
        
        assert len(kg1.entities) == 4
        assert len(kg1.relations) == 2
        assert kg1.triple_store.size() == 2
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        kg = KnowledgeGraph()
        
        # 添加多个实体和关系
        entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("李四", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 0, 3, 0.95),
            Entity("北京", EntityType.CITY, 0, 2, 0.9)
        ]
        
        relations = [
            Relation(entities[0], RelationType.WORKS_FOR, entities[2], 0.85, "ctx1", "sent1"),
            Relation(entities[1], RelationType.WORKS_FOR, entities[2], 0.88, "ctx2", "sent2"),
            Relation(entities[2], RelationType.LOCATED_IN, entities[3], 0.92, "ctx3", "sent3")
        ]
        
        for relation in relations:
            kg.add_relation(relation)
        
        stats = kg.get_statistics()
        
        assert stats["total_entities"] == 4
        assert stats["total_relations"] == 3
        assert stats["total_triples"] == 3
        assert stats["entity_types"]["PERSON"] == 2
        assert stats["entity_types"]["COMPANY"] == 1
        assert stats["entity_types"]["CITY"] == 1
        assert stats["relation_types"]["WORKS_FOR"] == 2
        assert stats["relation_types"]["LOCATED_IN"] == 1

class TestPydanticModels:
    """Pydantic模型测试"""
    
    def test_entity_model_validation(self):
        """测试实体模型验证"""
        # 有效模型
        entity_model = EntityModel(
            text="张三",
            label="PERSON",
            start=0,
            end=2,
            confidence=0.9
        )
        
        assert entity_model.text == "张三"
        assert entity_model.label == "PERSON"
        assert entity_model.confidence == 0.9
        
        # 测试验证错误
        with pytest.raises(ValueError):
            # end必须大于start
            EntityModel(
                text="测试",
                label="PERSON",
                start=5,
                end=3,  # 错误：end <= start
                confidence=0.9
            )
        
        with pytest.raises(ValueError):
            # 无效的实体类型
            EntityModel(
                text="测试",
                label="INVALID_TYPE",
                start=0,
                end=2,
                confidence=0.9
            )
    
    def test_extraction_request_validation(self):
        """测试抽取请求验证"""
        # 有效请求
        request = ExtractionRequest(
            text="张三在苹果公司工作。",
            language="zh",
            confidence_threshold=0.8
        )
        
        assert request.text == "张三在苹果公司工作。"
        assert request.language == "zh"
        assert request.extract_entities is True
        assert request.extract_relations is True
        assert request.link_entities is True
        
        # 测试文本长度限制
        with pytest.raises(ValueError):
            ExtractionRequest(text="")  # 空文本
        
        with pytest.raises(ValueError):
            ExtractionRequest(text="x" * 50001)  # 文本过长
    
    def test_batch_processing_request_validation(self):
        """测试批处理请求验证"""
        # 有效请求
        documents = [
            {"id": "doc1", "text": "张三在苹果公司工作。"},
            {"id": "doc2", "text": "李四在谷歌工作。"}
        ]
        
        request = BatchProcessingRequest(
            documents=documents,
            priority=5
        )
        
        assert len(request.documents) == 2
        assert request.priority == 5
        
        # 测试文档验证
        with pytest.raises(ValueError):
            # 文档缺少text字段
            BatchProcessingRequest(
                documents=[{"id": "doc1"}]
            )
        
        with pytest.raises(ValueError):
            # 文档数量过多
            BatchProcessingRequest(
                documents=[{"text": "test"}] * 1001
            )

class TestBatchProcessingResult:
    """批处理结果测试"""
    
    def test_batch_processing_result_creation(self):
        """测试批处理结果创建"""
        results = [
            {"document_id": "doc1", "entities": [], "relations": []},
            {"document_id": "doc2", "entities": [], "relations": []}
        ]
        
        errors = [
            {"document_id": "doc3", "error": "Processing failed"}
        ]
        
        metrics = {
            "total_processing_time": 10.5,
            "average_entities_per_doc": 2.3,
            "cache_hit_rate": 0.15
        }
        
        result = BatchProcessingResult(
            batch_id="batch_123",
            total_documents=3,
            successful_documents=2,
            failed_documents=1,
            results=results,
            errors=errors,
            processing_time=10.5,
            metrics=metrics
        )
        
        assert result.batch_id == "batch_123"
        assert result.total_documents == 3
        assert result.successful_documents == 2
        assert result.failed_documents == 1
        assert len(result.results) == 2
        assert len(result.errors) == 1
        assert result.processing_time == 10.5
    
    def test_batch_processing_result_to_dict(self):
        """测试批处理结果转字典"""
        result = BatchProcessingResult(
            batch_id="batch_456",
            total_documents=10,
            successful_documents=8,
            failed_documents=2,
            results=[],
            errors=[],
            processing_time=25.3,
            metrics={}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["batch_id"] == "batch_456"
        assert result_dict["total_documents"] == 10
        assert result_dict["successful_documents"] == 8
        assert result_dict["failed_documents"] == 2
        assert result_dict["success_rate"] == 0.8  # 8/10
        assert result_dict["processing_time"] == 25.3
        assert "created_at" in result_dict

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
