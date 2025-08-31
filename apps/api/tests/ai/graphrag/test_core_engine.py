"""
GraphRAG核心引擎集成测试

测试GraphRAG系统的完整查询流程，包括多模式检索、知识融合和推理路径生成
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.ai.graphrag.core_engine import GraphRAGEngine, get_graphrag_engine
from src.ai.graphrag.data_models import (
    GraphRAGRequest,
    GraphRAGResponse,
    RetrievalMode,
    QueryType,
    GraphContext,
    ReasoningPath,
    KnowledgeSource,
    FusionResult,
    create_graph_rag_request
)


class TestGraphRAGEngine:
    """GraphRAG引擎集成测试"""
    
    @pytest.fixture
    async def mock_dependencies(self):
        """模拟依赖组件"""
        with patch('src.ai.graphrag.core_engine.QueryAnalyzer') as mock_analyzer, \
             patch('src.ai.graphrag.core_engine.CacheManager') as mock_cache, \
             patch('src.ai.graphrag.core_engine.KnowledgeFusion') as mock_fusion, \
             patch('src.ai.graphrag.core_engine.ReasoningEngine') as mock_reasoning, \
             patch('src.ai.rag.retriever.get_rag_retriever') as mock_rag, \
             patch('src.ai.graphrag.core_engine.get_neo4j_driver') as mock_neo4j:
            
            # 配置模拟对象
            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            mock_cache_instance = AsyncMock()
            mock_cache.return_value = mock_cache_instance
            
            mock_fusion_instance = AsyncMock()
            mock_fusion.return_value = mock_fusion_instance
            
            mock_reasoning_instance = AsyncMock()
            mock_reasoning.return_value = mock_reasoning_instance
            
            mock_rag_instance = AsyncMock()
            mock_rag.return_value = mock_rag_instance
            
            mock_neo4j_instance = AsyncMock()
            mock_neo4j.return_value = mock_neo4j_instance
            
            yield {
                'analyzer': mock_analyzer_instance,
                'cache': mock_cache_instance,
                'fusion': mock_fusion_instance,
                'reasoning': mock_reasoning_instance,
                'rag': mock_rag_instance,
                'neo4j': mock_neo4j_instance
            }
    
    @pytest.fixture
    async def sample_request(self):
        """示例GraphRAG请求"""
        return create_graph_rag_request(
            query="什么是机器学习",
            retrieval_mode=RetrievalMode.HYBRID,
            max_docs=10,
            include_reasoning=True,
            expansion_depth=2,
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    async def sample_decomposition(self):
        """示例查询分解结果"""
        from src.ai.graphrag.data_models import QueryDecomposition
        return QueryDecomposition(
            original_query="什么是机器学习",
            sub_queries=["机器学习定义", "机器学习应用"],
            entity_queries=[{"entity": "机器学习", "type": "CONCEPT"}],
            relation_queries=[{"entity1": "机器学习", "entity2": "人工智能"}],
            decomposition_strategy="semantic",
            complexity_score=0.6
        )
    
    @pytest.fixture
    async def sample_graph_context(self):
        """示例图谱上下文"""
        return GraphContext(
            entities=[
                {"id": "ml", "name": "机器学习", "type": "CONCEPT"},
                {"id": "ai", "name": "人工智能", "type": "CONCEPT"}
            ],
            relations=[
                {"type": "PART_OF", "source": "ml", "target": "ai"}
            ],
            subgraph={"nodes": 2, "edges": 1},
            reasoning_paths=[],
            expansion_depth=1,
            confidence_score=0.8
        )
    
    async def test_engine_initialization(self, mock_dependencies):
        """测试引擎初始化"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        assert engine.analyzer is not None
        assert engine.cache_manager is not None
        assert engine.knowledge_fusion is not None
        assert engine.reasoning_engine is not None
        
        # 验证性能计数器初始化
        assert engine.performance_counters["total_queries"] == 0
        assert engine.performance_counters["cache_hits"] == 0
    
    async def test_enhanced_query_with_cache_hit(self, mock_dependencies, sample_request):
        """测试缓存命中的增强查询"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟缓存命中
        cached_response = {
            "query_id": "test-id",
            "original_query": sample_request["query"],
            "final_answer": "机器学习是人工智能的一个子领域",
            "knowledge_sources": [],
            "graph_context": {"entities": [], "relations": []},
            "reasoning_paths": [],
            "fusion_result": {"confidence": 0.8, "sources": []},
            "performance_metrics": {"total_time": 0.1}
        }
        mock_dependencies['cache'].get_cached_result.return_value = cached_response
        
        result = await engine.enhanced_query(sample_request)
        
        assert result["query_id"] == "test-id"
        assert result["final_answer"] == "机器学习是人工智能的一个子领域"
        assert engine.performance_counters["cache_hits"] == 1
    
    async def test_enhanced_query_full_pipeline(self, mock_dependencies, sample_request, 
                                              sample_decomposition, sample_graph_context):
        """测试完整的增强查询流程"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟无缓存
        mock_dependencies['cache'].get_cached_result.return_value = None
        
        # 模拟查询分析
        mock_dependencies['analyzer'].analyze_query.return_value = sample_decomposition
        
        # 模拟向量检索结果
        vector_results = {
            "documents": [
                {"content": "机器学习是AI的分支", "score": 0.9},
                {"content": "ML用于数据分析", "score": 0.8}
            ]
        }
        mock_dependencies['rag'].query.return_value = vector_results
        
        # 模拟图谱上下文扩展
        engine._expand_graph_context = AsyncMock(return_value=sample_graph_context)
        
        # 模拟推理路径生成
        reasoning_paths = [
            ReasoningPath(
                path_id="path1",
                entities=["机器学习", "人工智能"],
                relations=["PART_OF"],
                path_score=0.9,
                explanation="机器学习是人工智能的一部分",
                evidence=[{"fact": "ML ⊆ AI"}],
                hops_count=1
            )
        ]
        mock_dependencies['reasoning'].generate_reasoning_paths.return_value = reasoning_paths
        
        # 模拟知识融合
        fusion_result = FusionResult(
            fused_knowledge=[
                KnowledgeSource(
                    source_type="fusion",
                    content="机器学习是人工智能的子领域，用于数据分析和模式识别",
                    confidence=0.9,
                    metadata={"fusion_method": "weighted_average"}
                )
            ],
            confidence_score=0.9,
            fusion_strategy="weighted_consensus",
            source_weights={"vector": 0.6, "graph": 0.4},
            conflict_resolution="evidence_based",
            metadata={"sources_count": 2}
        )
        mock_dependencies['fusion'].fuse_knowledge_sources.return_value = fusion_result
        
        # 执行查询
        result = await engine.enhanced_query(sample_request)
        
        # 验证结果
        assert result["success"] is True
        assert "query_id" in result
        assert result["original_query"] == sample_request["query"]
        assert "final_answer" in result
        assert len(result["knowledge_sources"]) > 0
        assert result["graph_context"] == sample_graph_context.to_dict()
        assert len(result["reasoning_paths"]) == 1
        assert result["fusion_result"]["confidence_score"] == 0.9
        
        # 验证性能指标
        assert "performance_metrics" in result
        assert "total_time" in result["performance_metrics"]
        
        # 验证缓存调用
        mock_dependencies['cache'].cache_result.assert_called_once()
    
    async def test_vector_only_retrieval(self, mock_dependencies, sample_request):
        """测试纯向量检索模式"""
        # 修改请求为向量模式
        vector_request = sample_request.copy()
        vector_request["retrieval_mode"] = RetrievalMode.VECTOR_ONLY
        
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟无缓存和基本设置
        mock_dependencies['cache'].get_cached_result.return_value = None
        mock_dependencies['analyzer'].analyze_query.return_value = AsyncMock()
        mock_dependencies['rag'].query.return_value = {"documents": []}
        engine._expand_graph_context = AsyncMock(return_value=GraphContext(
            entities=[], relations=[], subgraph={}, reasoning_paths=[],
            expansion_depth=0, confidence_score=0.0
        ))
        mock_dependencies['reasoning'].generate_reasoning_paths.return_value = []
        mock_dependencies['fusion'].fuse_knowledge_sources.return_value = FusionResult(
            fused_knowledge=[], confidence_score=0.5, fusion_strategy="none",
            source_weights={}, conflict_resolution="none", metadata={}
        )
        
        result = await engine.enhanced_query(vector_request)
        
        # 在向量模式下，图谱扩展应该被跳过或最小化
        assert result["success"] is True
        assert result["graph_context"]["expansion_depth"] == 0
    
    async def test_graph_only_retrieval(self, mock_dependencies, sample_request, sample_graph_context):
        """测试纯图谱检索模式"""
        # 修改请求为图谱模式
        graph_request = sample_request.copy()
        graph_request["retrieval_mode"] = RetrievalMode.GRAPH_ONLY
        
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟无缓存和基本设置
        mock_dependencies['cache'].get_cached_result.return_value = None
        mock_dependencies['analyzer'].analyze_query.return_value = AsyncMock()
        engine._expand_graph_context = AsyncMock(return_value=sample_graph_context)
        mock_dependencies['reasoning'].generate_reasoning_paths.return_value = []
        mock_dependencies['fusion'].fuse_knowledge_sources.return_value = FusionResult(
            fused_knowledge=[], confidence_score=0.7, fusion_strategy="graph_based",
            source_weights={"graph": 1.0}, conflict_resolution="none", metadata={}
        )
        
        result = await engine.enhanced_query(graph_request)
        
        # 在图谱模式下，向量检索应该被跳过
        assert result["success"] is True
        assert result["graph_context"]["confidence_score"] == 0.8
        mock_dependencies['rag'].query.assert_not_called()
    
    async def test_error_handling(self, mock_dependencies, sample_request):
        """测试错误处理"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟查询分析器错误
        mock_dependencies['cache'].get_cached_result.return_value = None
        mock_dependencies['analyzer'].analyze_query.side_effect = Exception("分析器错误")
        
        result = await engine.enhanced_query(sample_request)
        
        assert result["success"] is False
        assert "error" in result
        assert "分析器错误" in result["error"]
    
    async def test_performance_metrics(self, mock_dependencies, sample_request):
        """测试性能指标收集"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 设置基本模拟
        mock_dependencies['cache'].get_cached_result.return_value = None
        mock_dependencies['analyzer'].analyze_query.return_value = AsyncMock()
        mock_dependencies['rag'].query.return_value = {"documents": []}
        engine._expand_graph_context = AsyncMock(return_value=AsyncMock())
        mock_dependencies['reasoning'].generate_reasoning_paths.return_value = []
        mock_dependencies['fusion'].fuse_knowledge_sources.return_value = FusionResult(
            fused_knowledge=[], confidence_score=0.5, fusion_strategy="none",
            source_weights={}, conflict_resolution="none", metadata={}
        )
        
        result = await engine.enhanced_query(sample_request)
        
        # 验证性能指标
        assert "performance_metrics" in result
        metrics = result["performance_metrics"]
        assert "total_time" in metrics
        assert "query_analysis_time" in metrics
        assert "retrieval_time" in metrics
        assert "graph_expansion_time" in metrics
        assert "reasoning_time" in metrics
        assert "fusion_time" in metrics
        
        # 验证计数器更新
        assert engine.performance_counters["total_queries"] == 1
    
    async def test_get_performance_stats(self, mock_dependencies):
        """测试性能统计获取"""
        engine = GraphRAGEngine()
        await engine.initialize()
        
        # 模拟一些查询历史
        engine.performance_counters["total_queries"] = 10
        engine.performance_counters["cache_hits"] = 3
        engine.query_times = [0.1, 0.2, 0.15, 0.25, 0.18]
        
        stats = await engine.get_performance_stats()
        
        assert stats["engine_status"] == "initialized"
        assert stats["total_queries"] == 10
        assert stats["cache_hits"] == 3
        assert stats["cache_hit_rate"] == 0.3
        assert "average_query_time" in stats
        assert "query_times_percentiles" in stats
    
    async def test_singleton_pattern(self):
        """测试单例模式"""
        engine1 = await get_graphrag_engine()
        engine2 = await get_graphrag_engine()
        
        assert engine1 is engine2


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__])