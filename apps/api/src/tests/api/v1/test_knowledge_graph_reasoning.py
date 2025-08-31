"""
知识图推理API端点测试

测试推理引擎API的所有端点和功能
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from api.v1.knowledge_graph_reasoning import router, get_hybrid_reasoner
from ai.knowledge_graph.hybrid_reasoner import (
    HybridReasoner, HybridReasoningResult, ReasoningStrategy, ReasoningEvidence
)


@pytest.fixture
def mock_hybrid_reasoner():
    """模拟混合推理引擎"""
    reasoner = Mock(spec=HybridReasoner)
    
    # 模拟推理结果
    mock_result = HybridReasoningResult(
        query="test query",
        results=[
            {"entity": "result1", "confidence": 0.8},
            {"entity": "result2", "confidence": 0.7}
        ],
        confidence=0.75,
        evidences=[
            ReasoningEvidence(
                source="rule_engine",
                method="forward_chaining", 
                evidence_type="rule_application",
                content={"rule": "test_rule"},
                confidence=0.8
            )
        ],
        strategy_used=ReasoningStrategy.ENSEMBLE,
        execution_time=0.5,
        method_contributions={"rule": 0.4, "embedding": 0.3, "path": 0.3},
        explanation="测试推理完成"
    )
    
    reasoner.reason = AsyncMock(return_value=mock_result)
    
    # 模拟性能统计
    mock_stats = {
        "rule_only": {
            "total_queries": 10,
            "success_rate": 0.8,
            "avg_confidence": 0.75,
            "avg_execution_time": 0.3,
            "accuracy_score": 0.8,
            "last_updated": utc_now().isoformat()
        },
        "ensemble": {
            "total_queries": 15,
            "success_rate": 0.9,
            "avg_confidence": 0.85,
            "avg_execution_time": 0.5,
            "accuracy_score": 0.85,
            "last_updated": utc_now().isoformat()
        }
    }
    
    reasoner.get_strategy_performance_stats = AsyncMock(return_value=mock_stats)
    reasoner.update_confidence_weights = AsyncMock()
    reasoner.explain_reasoning = AsyncMock(return_value="详细的推理解释")
    
    # 模拟adaptive_thresholds属性
    reasoner.adaptive_thresholds = {
        "high_confidence": 0.8,
        "medium_confidence": 0.6,
        "low_confidence": 0.4
    }
    
    return reasoner


@pytest.fixture
def client():
    """创建测试客户端"""
    from main import app
    return TestClient(app)


class TestReasoningQueryEndpoint:
    """推理查询端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_query_reasoning_success(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试成功的推理查询"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "query": "find connection between entity1 and entity2",
            "entities": ["entity1", "entity2"],
            "relations": ["connected_to"],
            "strategy": "ensemble",
            "top_k": 5
        }
        
        response = client.post("/api/v1/kg-reasoning/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["query"] == request_data["query"]
        assert data["confidence"] == 0.75
        assert data["strategy_used"] == "ensemble"
        assert len(data["results"]) == 2
        assert data["evidences_count"] == 1
        assert "method_contributions" in data
        assert "execution_time" in data
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_query_reasoning_with_different_strategies(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试不同推理策略"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        strategies = ["rule_only", "embedding_only", "path_only", "uncertainty_only", "adaptive", "cascading", "voting"]
        
        for strategy in strategies:
            request_data = {
                "query": f"test {strategy} strategy",
                "entities": ["test_entity"],
                "strategy": strategy
            }
            
            response = client.post("/api/v1/kg-reasoning/query", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_query_reasoning_validation_error(self, mock_get_reasoner, client):
        """测试请求验证错误"""
        mock_get_reasoner.return_value = Mock()
        
        # 缺少必需字段
        request_data = {
            "entities": ["entity1"],
            # 缺少query字段
        }
        
        response = client.post("/api/v1/kg-reasoning/query", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_query_reasoning_engine_error(self, mock_get_reasoner, client):
        """测试推理引擎内部错误"""
        mock_reasoner = Mock()
        mock_reasoner.reason = AsyncMock(side_effect=Exception("Internal engine error"))
        mock_get_reasoner.return_value = mock_reasoner
        
        request_data = {
            "query": "test query",
            "entities": ["entity1"]
        }
        
        response = client.post("/api/v1/kg-reasoning/query", json=request_data)
        assert response.status_code == 500


class TestBatchReasoningEndpoint:
    """批量推理端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_batch_reasoning_parallel(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试并行批量推理"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "queries": [
                {
                    "query": "test query 1",
                    "entities": ["entity1"]
                },
                {
                    "query": "test query 2", 
                    "entities": ["entity2"]
                },
                {
                    "query": "test query 3",
                    "entities": ["entity3"]
                }
            ],
            "parallel": True
        }
        
        response = client.post("/api/v1/kg-reasoning/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["total_queries"] == 3
        assert data["successful_queries"] == 3
        assert data["failed_queries"] == 0
        assert len(data["results"]) == 3
        assert "total_execution_time" in data
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_batch_reasoning_serial(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试串行批量推理"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "queries": [
                {"query": "test query 1", "entities": ["entity1"]},
                {"query": "test query 2", "entities": ["entity2"]}
            ],
            "parallel": False
        }
        
        response = client.post("/api/v1/kg-reasoning/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_queries"] == 2
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner') 
    def test_batch_reasoning_empty_queries(self, mock_get_reasoner, client):
        """测试空查询列表"""
        mock_get_reasoner.return_value = Mock()
        
        request_data = {
            "queries": []
        }
        
        response = client.post("/api/v1/kg-reasoning/batch", json=request_data)
        assert response.status_code == 400
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_batch_reasoning_with_failures(self, mock_get_reasoner, client):
        """测试包含失败的批量推理"""
        mock_reasoner = Mock()
        
        # 第一个查询成功，第二个失败
        async def mock_reason(request):
            if "success" in request.query:
                return HybridReasoningResult(
                    query=request.query,
                    results=[{"result": "success"}],
                    confidence=0.8,
                    evidences=[],
                    strategy_used=ReasoningStrategy.ENSEMBLE,
                    execution_time=0.1,
                    method_contributions={"rule": 1.0},
                    explanation="成功"
                )
            else:
                raise Exception("Reasoning failed")
        
        mock_reasoner.reason = AsyncMock(side_effect=mock_reason)
        mock_get_reasoner.return_value = mock_reasoner
        
        request_data = {
            "queries": [
                {"query": "success query", "entities": ["entity1"]},
                {"query": "fail query", "entities": ["entity2"]}
            ],
            "parallel": True
        }
        
        response = client.post("/api/v1/kg-reasoning/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["successful_queries"] == 1
        assert data["failed_queries"] == 1


class TestStrategyPerformanceEndpoint:
    """策略性能端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_get_strategy_performance(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试获取策略性能统计"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        response = client.get("/api/v1/kg-reasoning/strategies/performance")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "strategies" in data
        assert "summary" in data
        
        # 检查汇总信息
        summary = data["summary"]
        assert "total_strategies" in summary
        assert "total_queries" in summary
        assert "avg_success_rate" in summary
        assert "avg_confidence" in summary
        assert "avg_execution_time" in summary
        
        # 检查具体策略统计
        strategies = data["strategies"]
        assert "rule_only" in strategies
        assert "ensemble" in strategies
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_strategy_performance_error(self, mock_get_reasoner, client):
        """测试策略性能获取错误"""
        mock_reasoner = Mock()
        mock_reasoner.get_strategy_performance_stats = AsyncMock(
            side_effect=Exception("Performance stats error")
        )
        mock_get_reasoner.return_value = mock_reasoner
        
        response = client.get("/api/v1/kg-reasoning/strategies/performance")
        assert response.status_code == 500


class TestConfigurationEndpoint:
    """配置端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_update_confidence_weights(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试更新置信度权重"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "confidence_weights": {
                "rule": 0.4,
                "embedding": 0.3,
                "path": 0.2,
                "uncertainty": 0.1
            }
        }
        
        response = client.post("/api/v1/kg-reasoning/config", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "confidence_weights" in data["updated_configs"]
        mock_hybrid_reasoner.update_confidence_weights.assert_called_once()
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_update_adaptive_thresholds(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试更新自适应阈值"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "adaptive_thresholds": {
                "high_confidence": 0.9,
                "medium_confidence": 0.7,
                "low_confidence": 0.3
            }
        }
        
        response = client.post("/api/v1/kg-reasoning/config", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "adaptive_thresholds" in data["updated_configs"]
        
        # 验证阈值已更新
        updated_thresholds = mock_hybrid_reasoner.adaptive_thresholds
        assert updated_thresholds["high_confidence"] == 0.9
        assert updated_thresholds["medium_confidence"] == 0.7
        assert updated_thresholds["low_confidence"] == 0.3
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_update_multiple_configs(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试同时更新多个配置"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        request_data = {
            "confidence_weights": {"rule": 0.5, "embedding": 0.5},
            "adaptive_thresholds": {"high_confidence": 0.95},
            "cache_settings": {"enabled": True}
        }
        
        response = client.post("/api/v1/kg-reasoning/config", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["updated_configs"]) == 3
        assert "confidence_weights" in data["updated_configs"]
        assert "adaptive_thresholds" in data["updated_configs"]
        assert "cache_settings" in data["updated_configs"]


class TestExplanationEndpoint:
    """解释端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_explain_reasoning_result(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试推理结果解释"""
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        result_data = {
            "query": "test query",
            "results": [{"entity": "result1"}],
            "confidence": 0.8,
            "strategy_used": "ensemble",
            "execution_time": 0.5,
            "method_contributions": {"rule": 0.6, "embedding": 0.4}
        }
        
        response = client.post("/api/v1/kg-reasoning/explain", json=result_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "explanation" in data
        assert isinstance(data["explanation"], str)
        mock_hybrid_reasoner.explain_reasoning.assert_called_once()
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_explain_reasoning_error(self, mock_get_reasoner, client):
        """测试解释生成错误"""
        mock_reasoner = Mock()
        mock_reasoner.explain_reasoning = AsyncMock(
            side_effect=Exception("Explanation generation failed")
        )
        mock_get_reasoner.return_value = mock_reasoner
        
        result_data = {
            "query": "test query",
            "results": [],
            "confidence": 0.0
        }
        
        response = client.post("/api/v1/kg-reasoning/explain", json=result_data)
        assert response.status_code == 500


class TestHealthCheckEndpoint:
    """健康检查端点测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_health_check_healthy(self, mock_get_reasoner, client, mock_hybrid_reasoner):
        """测试健康检查 - 健康状态"""
        # 模拟健康的推理引擎
        test_result = HybridReasoningResult(
            query="test query",
            results=[],
            confidence=0.5,
            evidences=[],
            strategy_used=ReasoningStrategy.RULE_ONLY,
            execution_time=0.1,
            method_contributions={"rule": 1.0},
            explanation="测试通过"
        )
        mock_hybrid_reasoner.reason = AsyncMock(return_value=test_result)
        mock_get_reasoner.return_value = mock_hybrid_reasoner
        
        response = client.get("/api/v1/kg-reasoning/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["reasoner_initialized"] is True
        assert "timestamp" in data
        assert data["test_reasoning"] == "passed"
        assert "test_confidence" in data
    
    @patch('api.v1.knowledge_graph_reasoning.get_hybrid_reasoner')
    def test_health_check_unhealthy(self, mock_get_reasoner, client):
        """测试健康检查 - 不健康状态"""
        mock_reasoner = Mock()
        mock_reasoner.reason = AsyncMock(side_effect=Exception("Test failed"))
        mock_get_reasoner.return_value = mock_reasoner
        
        response = client.get("/api/v1/kg-reasoning/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["reasoner_initialized"] is True
        assert data["test_reasoning"] == "failed"
        assert "test_error" in data
    
    def test_health_check_no_reasoner(self, client):
        """测试健康检查 - 推理引擎未初始化"""
        # 不使用mock，让推理引擎为None
        with patch('api.v1.knowledge_graph_reasoning._hybrid_reasoner', None):
            response = client.get("/api/v1/kg-reasoning/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert data["reasoner_initialized"] is False


class TestReasoningEngineInitialization:
    """推理引擎初始化测试"""
    
    @patch('api.v1.knowledge_graph_reasoning.RuleEngine')
    @patch('api.v1.knowledge_graph_reasoning.EmbeddingEngine')
    @patch('api.v1.knowledge_graph_reasoning.PathReasoner')
    @patch('api.v1.knowledge_graph_reasoning.UncertaintyReasoner')
    @patch('api.v1.knowledge_graph_reasoning.ReasoningOptimizer')
    @patch('api.v1.knowledge_graph_reasoning.HybridReasoner')
    @pytest.mark.asyncio
    async def test_reasoner_initialization_success(
        self, mock_hybrid, mock_optimizer, mock_uncertainty, 
        mock_path, mock_embedding, mock_rule
    ):
        """测试推理引擎成功初始化"""
        # 设置全局变量为None，强制重新初始化
        import api.v1.knowledge_graph_reasoning
        api.v1.knowledge_graph_reasoning._hybrid_reasoner = None
        
        # 创建mock实例
        mock_rule_instance = Mock()
        mock_embedding_instance = Mock()
        mock_path_instance = Mock()
        mock_uncertainty_instance = Mock()
        mock_optimizer_instance = Mock()
        mock_hybrid_instance = Mock()
        
        mock_rule.return_value = mock_rule_instance
        mock_embedding.return_value = mock_embedding_instance
        mock_path.return_value = mock_path_instance
        mock_uncertainty.return_value = mock_uncertainty_instance
        mock_optimizer.return_value = mock_optimizer_instance
        mock_hybrid.return_value = mock_hybrid_instance
        
        # 调用初始化函数
        reasoner = await get_hybrid_reasoner()
        
        # 验证所有组件都被创建
        mock_rule.assert_called_once()
        mock_embedding.assert_called_once_with(model_name="TransE", embedding_dim=256)
        mock_path.assert_called_once()
        mock_uncertainty.assert_called_once()
        mock_optimizer.assert_called_once()
        mock_hybrid.assert_called_once_with(
            rule_engine=mock_rule_instance,
            embedding_engine=mock_embedding_instance,
            path_reasoner=mock_path_instance,
            uncertainty_reasoner=mock_uncertainty_instance,
            optimizer=mock_optimizer_instance
        )
        
        assert reasoner == mock_hybrid_instance
    
    @patch('api.v1.knowledge_graph_reasoning.RuleEngine')
    @pytest.mark.asyncio
    async def test_reasoner_initialization_failure(self, mock_rule):
        """测试推理引擎初始化失败"""
        # 设置全局变量为None，强制重新初始化
        import api.v1.knowledge_graph_reasoning
        api.v1.knowledge_graph_reasoning._hybrid_reasoner = None
        
        # 模拟初始化失败
        mock_rule.side_effect = Exception("Initialization failed")
        
        # 初始化应该抛出HTTPException
        with pytest.raises(Exception):  # HTTPException will be raised
            await get_hybrid_reasoner()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])