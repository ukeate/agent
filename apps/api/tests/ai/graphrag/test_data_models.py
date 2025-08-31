"""
GraphRAG数据模型测试

测试GraphRAG系统的核心数据模型
"""

import pytest
from datetime import datetime

from src.ai.graphrag.data_models import (
    QueryType,
    RetrievalMode,
    GraphContext,
    ReasoningPath,
    KnowledgeSource,
    GraphRAGRequest,
    GraphRAGResponse,
    QueryDecomposition,
    FusionResult,
    EntityRecognitionResult,
    GraphRAGConfig,
    create_graph_rag_request,
    create_empty_graph_context,
    validate_graph_rag_request
)


class TestDataModels:
    """数据模型测试类"""
    
    def test_graph_context_creation(self):
        """测试GraphContext创建"""
        context = GraphContext(
            entities=[{"id": "1", "name": "test"}],
            relations=[{"type": "RELATED", "source": "1", "target": "2"}],
            subgraph={},
            reasoning_paths=[],
            expansion_depth=2,
            confidence_score=0.8
        )
        
        assert len(context.entities) == 1
        assert len(context.relations) == 1
        assert context.expansion_depth == 2
        assert context.confidence_score == 0.8
        
        # 测试to_dict方法
        context_dict = context.to_dict()
        assert "entities" in context_dict
        assert "relations" in context_dict
    
    def test_reasoning_path_creation(self):
        """测试ReasoningPath创建"""
        path = ReasoningPath(
            path_id="test_path",
            entities=["entity1", "entity2"],
            relations=["RELATED"],
            path_score=0.9,
            explanation="Test reasoning path",
            evidence=[{"fact": "test"}],
            hops_count=1
        )
        
        assert path.path_id == "test_path"
        assert len(path.entities) == 2
        assert path.path_score == 0.9
        
        # 测试to_dict方法
        path_dict = path.to_dict()
        assert "path_id" in path_dict
        assert "entities" in path_dict
    
    def test_knowledge_source_creation(self):
        """测试KnowledgeSource创建"""
        source = KnowledgeSource(
            source_type="vector",
            content="Test content",
            confidence=0.7,
            metadata={"source": "test"}
        )
        
        assert source.source_type == "vector"
        assert source.content == "Test content"
        assert source.confidence == 0.7
        
        # 测试to_dict方法
        source_dict = source.to_dict()
        assert "source_type" in source_dict
        assert "content" in source_dict
    
    def test_query_decomposition_creation(self):
        """测试QueryDecomposition创建"""
        decomposition = QueryDecomposition(
            original_query="Test query",
            sub_queries=["sub1", "sub2"],
            entity_queries=[{"entity": "test"}],
            relation_queries=[{"entity1": "a", "entity2": "b"}],
            decomposition_strategy="simple",
            complexity_score=0.5
        )
        
        assert decomposition.original_query == "Test query"
        assert len(decomposition.sub_queries) == 2
        assert decomposition.complexity_score == 0.5
        
        # 测试to_dict方法
        decomp_dict = decomposition.to_dict()
        assert "original_query" in decomp_dict
        assert "sub_queries" in decomp_dict
    
    def test_entity_recognition_result(self):
        """测试EntityRecognitionResult创建"""
        result = EntityRecognitionResult(
            text="Apple",
            canonical_form="Apple Inc",
            entity_type="ORGANIZATION",
            confidence=0.9,
            start_pos=0,
            end_pos=5,
            metadata={"source": "test"}
        )
        
        assert result.text == "Apple"
        assert result.canonical_form == "Apple Inc"
        assert result.confidence == 0.9
        
        # 测试to_dict方法
        result_dict = result.to_dict()
        assert "text" in result_dict
        assert "canonical_form" in result_dict
    
    def test_graphrag_config(self):
        """测试GraphRAGConfig"""
        config = GraphRAGConfig()
        
        # 测试默认值
        assert config.max_expansion_depth == 3
        assert config.confidence_threshold == 0.6
        assert config.enable_caching == True
        
        # 测试to_dict方法
        config_dict = config.to_dict()
        assert "max_expansion_depth" in config_dict
        assert "confidence_threshold" in config_dict
    
    def test_create_graph_rag_request(self):
        """测试创建GraphRAG请求"""
        request = create_graph_rag_request(
            query="Test query",
            retrieval_mode=RetrievalMode.HYBRID,
            max_docs=10
        )
        
        assert request["query"] == "Test query"
        assert request["retrieval_mode"] == RetrievalMode.HYBRID
        assert request["max_docs"] == 10
    
    def test_create_empty_graph_context(self):
        """测试创建空的图谱上下文"""
        context = create_empty_graph_context()
        
        assert len(context.entities) == 0
        assert len(context.relations) == 0
        assert context.confidence_score == 0.0
    
    def test_validate_graph_rag_request(self):
        """测试GraphRAG请求验证"""
        # 有效请求
        valid_request = create_graph_rag_request(
            query="Valid query",
            max_docs=5,
            confidence_threshold=0.7
        )
        errors = validate_graph_rag_request(valid_request)
        assert len(errors) == 0
        
        # 无效请求 - 空查询
        invalid_request = create_graph_rag_request(
            query="",
            max_docs=5
        )
        errors = validate_graph_rag_request(invalid_request)
        assert len(errors) > 0
        assert any("查询不能为空" in error for error in errors)
        
        # 无效请求 - 无效的max_docs
        invalid_request2 = create_graph_rag_request(
            query="Test",
            max_docs=0
        )
        errors = validate_graph_rag_request(invalid_request2)
        assert len(errors) > 0
        assert any("max_docs必须大于0" in error for error in errors)
    
    def test_post_init_methods(self):
        """测试__post_init__方法"""
        # GraphContext post_init
        context = GraphContext(
            entities=None,  # 会被转换为空列表
            relations=None,  # 会被转换为空列表
            subgraph=None,  # 会被转换为空字典
            reasoning_paths=None,  # 会被转换为空列表
            expansion_depth=1,
            confidence_score=1.5  # 会被限制在1.0
        )
        
        assert isinstance(context.entities, list)
        assert isinstance(context.relations, list)
        assert isinstance(context.subgraph, dict)
        assert isinstance(context.reasoning_paths, list)
        assert context.confidence_score == 1.0
        
        # ReasoningPath post_init
        path = ReasoningPath(
            path_id="",  # 会自动生成UUID
            entities=None,  # 会被转换为空列表
            relations=None,  # 会被转换为空列表
            path_score=1.5,  # 会被限制在1.0
            explanation="test",
            evidence=None,  # 会被转换为空列表
            hops_count=1
        )
        
        assert path.path_id != ""
        assert isinstance(path.entities, list)
        assert isinstance(path.relations, list)
        assert isinstance(path.evidence, list)
        assert path.path_score == 1.0
        
        # KnowledgeSource post_init
        source = KnowledgeSource(
            source_type="test",
            content="test",
            confidence=1.5,  # 会被限制在1.0
            metadata=None  # 会被转换为空字典
        )
        
        assert isinstance(source.metadata, dict)
        assert source.confidence == 1.0
        
        # QueryDecomposition post_init  
        decomp = QueryDecomposition(
            original_query="test",
            sub_queries=None,  # 会被转换为空列表
            entity_queries=None,  # 会被转换为空列表
            relation_queries=None,  # 会被转换为空列表
            decomposition_strategy="test",
            complexity_score=1.5  # 会被限制在1.0
        )
        
        assert isinstance(decomp.sub_queries, list)
        assert isinstance(decomp.entity_queries, list)
        assert isinstance(decomp.relation_queries, list)
        assert decomp.complexity_score == 1.0
        
        # EntityRecognitionResult post_init
        entity = EntityRecognitionResult(
            text="test",
            canonical_form="test",
            entity_type="test",
            confidence=1.5,  # 会被限制在1.0
            start_pos=0,
            end_pos=4,
            metadata=None  # 会被转换为空字典
        )
        
        assert isinstance(entity.metadata, dict)
        assert entity.confidence == 1.0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__])