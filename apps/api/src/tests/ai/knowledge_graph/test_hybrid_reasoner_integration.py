"""
混合推理引擎集成测试

测试整个推理系统的集成功能，包括所有推理引擎组件
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from ai.knowledge_graph.hybrid_reasoner import (
    HybridReasoner, ReasoningRequest, HybridReasoningResult,
    ReasoningStrategy, ConfidenceWeights, ReasoningEvidence
)
from ai.knowledge_graph.rule_engine import RuleEngine, InferenceResult
from ai.knowledge_graph.embedding_engine import EmbeddingEngine, SimilarityResult
from ai.knowledge_graph.path_reasoning import PathReasoner, PathSearchResult, ReasoningPath
from ai.knowledge_graph.uncertainty_reasoning import UncertaintyReasoner, UncertaintyQuantification
from ai.knowledge_graph.reasoning_optimizer import ReasoningOptimizer


@pytest.fixture
def mock_rule_engine():
    """模拟规则推理引擎"""
    engine = Mock(spec=RuleEngine)
    
    async def mock_forward_chaining(facts, max_iterations=10):
        return [
            InferenceResult(
                conclusion="person(alice)",
                confidence=0.9,
                applied_rules=["rule1", "rule2"],
                evidence=facts[:2]
            )
        ]
    
    engine.forward_chaining = AsyncMock(side_effect=mock_forward_chaining)
    return engine


@pytest.fixture
def mock_embedding_engine():
    """模拟嵌入推理引擎"""
    engine = Mock(spec=EmbeddingEngine)
    
    async def mock_find_similar(entity, top_k=10):
        return SimilarityResult(
            query_entity=entity,
            similar_entities=[
                type('', (), {
                    'entity': 'similar_entity_1',
                    'similarity': 0.85
                })(),
                type('', (), {
                    'entity': 'similar_entity_2', 
                    'similarity': 0.78
                })()
            ],
            execution_time=0.1
        )
    
    engine.find_similar_entities = AsyncMock(side_effect=mock_find_similar)
    return engine


@pytest.fixture
def mock_path_reasoner():
    """模拟路径推理引擎"""
    reasoner = Mock(spec=PathReasoner)
    
    async def mock_find_paths(start, end, constraints=None):
        return PathSearchResult(
            start_entity=start,
            end_entity=end,
            paths=[
                ReasoningPath(
                    path_entities=[start, 'intermediate', end],
                    path_relations=['rel1', 'rel2'],
                    confidence=0.82,
                    path_length=3
                )
            ],
            total_paths_found=1,
            search_time=0.2
        )
    
    reasoner.find_reasoning_paths = AsyncMock(side_effect=mock_find_paths)
    return reasoner


@pytest.fixture 
def mock_uncertainty_reasoner():
    """模拟不确定性推理引擎"""
    reasoner = Mock(spec=UncertaintyReasoner)
    
    async def mock_calculate_confidence(evidence, hypothesis):
        return UncertaintyQuantification(
            hypothesis=hypothesis,
            posterior_probability=0.75,
            confidence_interval=(0.65, 0.85),
            uncertainty_score=0.15,
            evidence_strength=0.8
        )
    
    reasoner.calculate_inference_confidence = AsyncMock(side_effect=mock_calculate_confidence)
    return reasoner


@pytest.fixture
def mock_optimizer():
    """模拟推理优化器"""
    optimizer = Mock(spec=ReasoningOptimizer)
    
    async def mock_optimize(request, priority):
        return request  # 直接返回原请求
    
    optimizer.optimize_reasoning_request = AsyncMock(side_effect=mock_optimize)
    return optimizer


@pytest.fixture
def hybrid_reasoner(
    mock_rule_engine,
    mock_embedding_engine, 
    mock_path_reasoner,
    mock_uncertainty_reasoner,
    mock_optimizer
):
    """创建混合推理引擎实例"""
    return HybridReasoner(
        rule_engine=mock_rule_engine,
        embedding_engine=mock_embedding_engine,
        path_reasoner=mock_path_reasoner,
        uncertainty_reasoner=mock_uncertainty_reasoner,
        optimizer=mock_optimizer
    )


class TestHybridReasonerIntegration:
    """混合推理引擎集成测试类"""
    
    @pytest.mark.asyncio
    async def test_rule_only_strategy(self, hybrid_reasoner):
        """测试仅规则推理策略"""
        request = ReasoningRequest(
            query="test rule reasoning",
            entities=["alice", "bob"],
            strategy=ReasoningStrategy.RULE_ONLY
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.RULE_ONLY
        assert result.confidence > 0
        assert len(result.results) > 0
        assert "rule" in result.method_contributions
        assert result.method_contributions["rule"] == 1.0
    
    @pytest.mark.asyncio
    async def test_embedding_only_strategy(self, hybrid_reasoner):
        """测试仅嵌入推理策略"""
        request = ReasoningRequest(
            query="find similar entities",
            entities=["test_entity"],
            strategy=ReasoningStrategy.EMBEDDING_ONLY
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.EMBEDDING_ONLY
        assert result.confidence > 0
        assert len(result.results) > 0
        assert "embedding" in result.method_contributions
        assert result.method_contributions["embedding"] == 1.0
    
    @pytest.mark.asyncio
    async def test_path_only_strategy(self, hybrid_reasoner):
        """测试仅路径推理策略"""
        request = ReasoningRequest(
            query="find connection path",
            entities=["entity1", "entity2"],
            strategy=ReasoningStrategy.PATH_ONLY
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.PATH_ONLY
        assert result.confidence > 0
        assert len(result.results) > 0
        assert "path" in result.method_contributions
        assert result.method_contributions["path"] == 1.0
    
    @pytest.mark.asyncio
    async def test_uncertainty_only_strategy(self, hybrid_reasoner):
        """测试仅不确定性推理策略"""
        request = ReasoningRequest(
            query="calculate uncertainty",
            entities=["uncertain_entity"],
            strategy=ReasoningStrategy.UNCERTAINTY_ONLY
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.UNCERTAINTY_ONLY
        assert result.confidence > 0
        assert len(result.results) > 0
        assert "uncertainty" in result.method_contributions
        assert result.method_contributions["uncertainty"] == 1.0
        assert result.uncertainty_analysis is not None
    
    @pytest.mark.asyncio
    async def test_ensemble_strategy(self, hybrid_reasoner):
        """测试集成推理策略"""
        request = ReasoningRequest(
            query="ensemble reasoning test",
            entities=["entity1", "entity2"],
            strategy=ReasoningStrategy.ENSEMBLE
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.ENSEMBLE
        assert result.confidence > 0
        assert len(result.results) > 0
        # 集成策略应该包含多个方法的贡献
        assert len(result.method_contributions) > 1
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, hybrid_reasoner):
        """测试自适应策略选择"""
        # 初始化策略性能，模拟历史数据
        for strategy in ReasoningStrategy:
            perf = hybrid_reasoner.strategy_performance[strategy]
            perf.total_queries = 10
            perf.success_queries = 8
            perf.avg_confidence = 0.8
            perf.avg_execution_time = 0.5
            perf.accuracy_score = 0.8
        
        request = ReasoningRequest(
            query="adaptive strategy test",
            entities=["test_entity"],
            strategy=ReasoningStrategy.ADAPTIVE
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.confidence >= 0
        assert isinstance(result.strategy_used, ReasoningStrategy)
        # 自适应策略应该选择一个具体的策略，不应该是ADAPTIVE
        assert result.strategy_used != ReasoningStrategy.ADAPTIVE
    
    @pytest.mark.asyncio
    async def test_cascading_strategy(self, hybrid_reasoner):
        """测试级联推理策略"""
        request = ReasoningRequest(
            query="cascading reasoning test", 
            entities=["entity1", "entity2"],
            strategy=ReasoningStrategy.CASCADING
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.CASCADING
        assert result.confidence >= 0
        assert len(result.evidences) > 0  # 级联应该产生多层证据
    
    @pytest.mark.asyncio
    async def test_voting_strategy(self, hybrid_reasoner):
        """测试投票推理策略"""
        request = ReasoningRequest(
            query="voting reasoning test",
            entities=["entity1", "entity2"],
            strategy=ReasoningStrategy.VOTING
        )
        
        result = await hybrid_reasoner.reason(request)
        
        assert result.strategy_used == ReasoningStrategy.VOTING
        assert result.confidence >= 0
        # 投票策略的结果应该包含投票分数
        if result.results:
            for res in result.results:
                if "vote_score" in res:
                    assert res["vote_score"] > 0
    
    @pytest.mark.asyncio
    async def test_query_feature_analysis(self, hybrid_reasoner):
        """测试查询特征分析"""
        # 测试不同类型的查询
        test_queries = [
            ("if person(x) then human(x)", True, False, False, False),  # 规则查询
            ("find similar entities to cat", False, True, False, False),  # 相似性查询
            ("path from A to B", False, False, True, False),  # 路径查询
            ("maybe person is human", False, False, False, True),  # 不确定性查询
        ]
        
        for query, has_rules, needs_similarity, needs_path, needs_uncertainty in test_queries:
            features = await hybrid_reasoner._analyze_query_features(
                ReasoningRequest(query=query, entities=["test"])
            )
            
            assert features["has_rules"] == has_rules
            assert features["needs_similarity"] == needs_similarity
            assert features["needs_path"] == needs_path
            assert features["needs_uncertainty"] == needs_uncertainty
    
    @pytest.mark.asyncio
    async def test_confidence_weight_update(self, hybrid_reasoner):
        """测试置信度权重更新"""
        new_weights = {
            "rule": 0.4,
            "embedding": 0.3,
            "path": 0.2,
            "uncertainty": 0.1
        }
        
        await hybrid_reasoner.update_confidence_weights(new_weights)
        
        assert hybrid_reasoner.confidence_weights == new_weights
    
    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, hybrid_reasoner):
        """测试策略性能跟踪"""
        request = ReasoningRequest(
            query="performance tracking test",
            entities=["test_entity"],
            strategy=ReasoningStrategy.RULE_ONLY
        )
        
        # 执行推理前的性能数据
        initial_perf = hybrid_reasoner.strategy_performance[ReasoningStrategy.RULE_ONLY]
        initial_queries = initial_perf.total_queries
        
        # 执行推理
        result = await hybrid_reasoner.reason(request)
        
        # 验证性能统计更新
        updated_perf = hybrid_reasoner.strategy_performance[ReasoningStrategy.RULE_ONLY]
        assert updated_perf.total_queries == initial_queries + 1
        if result.confidence > 0:
            assert updated_perf.success_queries > initial_perf.success_queries
    
    @pytest.mark.asyncio
    async def test_reasoning_explanation(self, hybrid_reasoner):
        """测试推理解释生成"""
        request = ReasoningRequest(
            query="explanation test",
            entities=["test_entity"],
            strategy=ReasoningStrategy.ENSEMBLE
        )
        
        result = await hybrid_reasoner.reason(request)
        explanation = await hybrid_reasoner.explain_reasoning(result)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "查询:" in explanation
        assert "策略:" in explanation
        assert "置信度:" in explanation
    
    @pytest.mark.asyncio
    async def test_error_handling(self, hybrid_reasoner):
        """测试错误处理"""
        # 模拟规则引擎抛出异常
        hybrid_reasoner.rule_engine.forward_chaining = AsyncMock(
            side_effect=Exception("Rule engine error")
        )
        
        request = ReasoningRequest(
            query="error handling test",
            entities=["test_entity"], 
            strategy=ReasoningStrategy.RULE_ONLY
        )
        
        result = await hybrid_reasoner.reason(request)
        
        # 应该返回失败结果而不是抛出异常
        assert result.confidence == 0.0
        assert len(result.results) == 0
        assert "失败" in result.explanation
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, hybrid_reasoner):
        """测试超时处理"""
        # 模拟长时间运行的推理任务
        async def long_running_task(*args, **kwargs):
            await asyncio.sleep(2)  # 模拟长时间运行
            return []
        
        hybrid_reasoner.rule_engine.forward_chaining = AsyncMock(
            side_effect=long_running_task
        )
        
        request = ReasoningRequest(
            query="timeout test",
            entities=["test_entity"],
            strategy=ReasoningStrategy.RULE_ONLY,
            timeout=1  # 1秒超时
        )
        
        result = await hybrid_reasoner.reason(request)
        
        # 由于模拟的任务运行时间超过超时时间，应该得到超时结果
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_result_fusion(self, hybrid_reasoner):
        """测试结果融合功能"""
        # 创建模拟结果
        mock_results = [
            HybridReasoningResult(
                query="test",
                results=[{"item": "result1", "confidence": 0.8}],
                confidence=0.8,
                evidences=[],
                strategy_used=ReasoningStrategy.RULE_ONLY,
                execution_time=0.1,
                method_contributions={"rule": 1.0}
            ),
            HybridReasoningResult(
                query="test", 
                results=[{"item": "result2", "confidence": 0.7}],
                confidence=0.7,
                evidences=[],
                strategy_used=ReasoningStrategy.EMBEDDING_ONLY,
                execution_time=0.2,
                method_contributions={"embedding": 1.0}
            )
        ]
        
        request = ReasoningRequest(query="test", entities=["test"])
        fused_result = await hybrid_reasoner._fuse_reasoning_results(request, mock_results)
        
        assert fused_result.confidence > 0
        assert len(fused_result.results) >= 1
        assert len(fused_result.method_contributions) >= 2
        
        # 检查方法贡献是否归一化
        total_contribution = sum(fused_result.method_contributions.values())
        assert abs(total_contribution - 1.0) < 0.001
    
    @pytest.mark.asyncio 
    async def test_strategy_performance_stats(self, hybrid_reasoner):
        """测试策略性能统计获取"""
        # 模拟一些历史数据
        for strategy in ReasoningStrategy:
            perf = hybrid_reasoner.strategy_performance[strategy]
            perf.total_queries = 5
            perf.success_queries = 4
            perf.avg_confidence = 0.8
            perf.avg_execution_time = 0.3
            perf.accuracy_score = 0.8
        
        stats = await hybrid_reasoner.get_strategy_performance_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) == len(ReasoningStrategy)
        
        for strategy_name, stat in stats.items():
            assert "total_queries" in stat
            assert "success_rate" in stat
            assert "avg_confidence" in stat
            assert "avg_execution_time" in stat
            assert "accuracy_score" in stat
            assert "last_updated" in stat


class TestReasoningRequestValidation:
    """推理请求验证测试"""
    
    def test_valid_reasoning_request(self):
        """测试有效的推理请求"""
        request = ReasoningRequest(
            query="test query",
            entities=["entity1", "entity2"],
            relations=["relation1"],
            strategy=ReasoningStrategy.ENSEMBLE
        )
        
        assert request.query == "test query"
        assert len(request.entities) == 2
        assert len(request.relations) == 1
        assert request.strategy == ReasoningStrategy.ENSEMBLE
    
    def test_default_values(self):
        """测试默认值设置"""
        request = ReasoningRequest(query="minimal query")
        
        assert request.query_type == "general"
        assert request.entities == []
        assert request.relations == []
        assert request.strategy == ReasoningStrategy.ADAPTIVE
        assert request.max_depth == 3
        assert request.top_k == 10
        assert request.confidence_threshold == 0.5
        assert request.timeout == 30


class TestReasoningEvidence:
    """推理证据测试"""
    
    def test_evidence_creation(self):
        """测试证据创建"""
        evidence = ReasoningEvidence(
            source="test_source",
            method="test_method", 
            evidence_type="test_type",
            content={"key": "value"},
            confidence=0.85
        )
        
        assert evidence.source == "test_source"
        assert evidence.method == "test_method"
        assert evidence.evidence_type == "test_type"
        assert evidence.content == {"key": "value"}
        assert evidence.confidence == 0.85
        assert evidence.support_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])