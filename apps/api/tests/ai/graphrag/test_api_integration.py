"""
GraphRAG API集成测试

测试GraphRAG系统的REST API接口，包括独立API和RAG集成API
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.v1.graphrag import router as graphrag_router
from src.api.v1.rag import router as rag_router


@pytest.fixture
def app():
    """创建测试应用"""
    app = FastAPI()
    app.include_router(graphrag_router)
    app.include_router(rag_router)
    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


class TestGraphRAGAPI:
    """GraphRAG API集成测试"""
    
    @pytest.fixture
    def sample_request(self):
        """示例API请求"""
        return {
            "query": "什么是深度学习",
            "retrieval_mode": "hybrid",
            "max_docs": 10,
            "include_reasoning": True,
            "expansion_depth": 2,
            "confidence_threshold": 0.7,
            "filters": {"domain": "AI"}
        }
    
    @pytest.fixture
    def mock_engine_response(self):
        """模拟引擎响应"""
        return {
            "success": True,
            "query_id": "test-query-id",
            "original_query": "什么是深度学习",
            "final_answer": "深度学习是机器学习的一个子领域",
            "knowledge_sources": [
                {
                    "source_type": "vector",
                    "content": "深度学习使用神经网络",
                    "confidence": 0.9,
                    "metadata": {"source": "wiki"}
                }
            ],
            "graph_context": {
                "entities": [{"id": "dl", "name": "深度学习"}],
                "relations": [{"type": "IS_A", "source": "dl", "target": "ml"}],
                "expansion_depth": 2,
                "confidence_score": 0.8
            },
            "reasoning_paths": [
                {
                    "path_id": "path1",
                    "entities": ["深度学习", "机器学习"],
                    "relations": ["IS_A"],
                    "path_score": 0.9,
                    "explanation": "深度学习是机器学习的一种方法"
                }
            ],
            "fusion_result": {
                "confidence_score": 0.85,
                "fusion_strategy": "weighted_consensus",
                "source_weights": {"vector": 0.6, "graph": 0.4}
            },
            "performance_metrics": {
                "total_time": 1.5,
                "query_analysis_time": 0.2,
                "retrieval_time": 0.8,
                "reasoning_time": 0.5
            }
        }
    
    def test_graphrag_query_success(self, client, sample_request, mock_engine_response):
        """测试GraphRAG查询成功"""
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.enhanced_query.return_value = mock_engine_response
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query", json=sample_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["query_id"] == "test-query-id"
            assert data["final_answer"] == "深度学习是机器学习的一个子领域"
            assert len(data["knowledge_sources"]) == 1
            assert data["graph_context"]["expansion_depth"] == 2
            assert len(data["reasoning_paths"]) == 1
            assert "performance_metrics" in data
    
    def test_graphrag_query_empty_query(self, client):
        """测试空查询请求"""
        request = {"query": "", "retrieval_mode": "hybrid"}
        
        response = client.post("/graphrag/query", json=request)
        
        assert response.status_code == 400
        assert "查询不能为空" in response.json()["detail"]
    
    def test_graphrag_query_invalid_retrieval_mode(self, client):
        """测试无效检索模式"""
        request = {
            "query": "test query",
            "retrieval_mode": "invalid_mode"
        }
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query", json=request)
            
            # 应该处理无效的检索模式
            assert response.status_code in [400, 500]
    
    def test_graphrag_query_engine_error(self, client, sample_request):
        """测试引擎错误处理"""
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.enhanced_query.side_effect = Exception("引擎错误")
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query", json=sample_request)
            
            assert response.status_code == 500
            assert "引擎错误" in response.json()["detail"]
    
    def test_graphrag_analyze_query(self, client):
        """测试查询分析接口"""
        request = {
            "query": "什么是机器学习算法",
            "include_entities": True,
            "include_relations": True
        }
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_analysis = {
                "success": True,
                "original_query": "什么是机器学习算法",
                "query_type": "factual",
                "complexity_score": 0.6,
                "sub_queries": ["机器学习定义", "算法类型"],
                "entities": [
                    {
                        "text": "机器学习",
                        "canonical_form": "机器学习",
                        "entity_type": "CONCEPT",
                        "confidence": 0.9
                    }
                ],
                "relations": [
                    {"entity1": "机器学习", "entity2": "算法", "relation": "HAS_TYPE"}
                ]
            }
            mock_engine.query_analyzer.analyze_query.return_value = mock_analysis
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query/analyze", json=request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["query_type"] == "factual"
            assert len(data["entities"]) == 1
            assert len(data["relations"]) == 1
    
    def test_graphrag_reasoning_paths(self, client):
        """测试推理路径生成接口"""
        request = {
            "query": "深度学习如何工作",
            "entities": ["深度学习", "神经网络"],
            "max_paths": 5,
            "max_depth": 3
        }
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_paths = {
                "success": True,
                "reasoning_paths": [
                    {
                        "path_id": "path1",
                        "entities": ["深度学习", "神经网络", "反向传播"],
                        "relations": ["USES", "IMPLEMENTS"],
                        "path_score": 0.9,
                        "explanation": "深度学习使用神经网络并通过反向传播学习",
                        "hops_count": 2
                    }
                ]
            }
            mock_engine.reasoning_engine.generate_reasoning_paths.return_value = mock_paths
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query/reasoning", json=request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["reasoning_paths"]) == 1
            assert data["reasoning_paths"][0]["path_score"] == 0.9
    
    def test_graphrag_health_check(self, client):
        """测试GraphRAG健康检查"""
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_stats = {
                "engine_status": "initialized",
                "total_queries": 100,
                "cache_hits": 30,
                "cache_hit_rate": 0.3,
                "average_query_time": 1.2
            }
            mock_engine.get_performance_stats.return_value = mock_stats
            mock_get_engine.return_value = mock_engine
            
            response = client.get("/graphrag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["details"]["total_queries"] == 100
            assert data["details"]["cache_hit_rate"] == 0.3
    
    def test_graphrag_health_check_error(self, client):
        """测试健康检查错误情况"""
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_get_engine.side_effect = Exception("引擎未初始化")
            
            response = client.get("/graphrag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "引擎未初始化" in data["error"]


class TestRAGIntegratedGraphRAG:
    """RAG集成的GraphRAG API测试"""
    
    def test_rag_graphrag_query_success(self, client, mock_engine_response):
        """测试RAG集成的GraphRAG查询"""
        request = {
            "query": "什么是机器学习",
            "retrieval_mode": "hybrid",
            "max_docs": 10
        }
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.enhanced_query.return_value = mock_engine_response
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/rag/graphrag/query", json=request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["query_id"] == "test-query-id"
    
    def test_rag_graphrag_health_check(self, client):
        """测试RAG集成的GraphRAG健康检查"""
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_stats = {
                "engine_status": "initialized",
                "total_queries": 50,
                "average_query_time": 0.8
            }
            mock_engine.get_performance_stats.return_value = mock_stats
            mock_get_engine.return_value = mock_engine
            
            response = client.get("/rag/graphrag/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["details"]["total_queries"] == 50


class TestValidationAndErrorHandling:
    """验证和错误处理测试"""
    
    def test_request_validation(self, client):
        """测试请求验证"""
        # 测试无效的max_docs
        invalid_request = {
            "query": "test query",
            "max_docs": 0
        }
        
        with patch('src.ai.graphrag.data_models.validate_graph_rag_request') as mock_validate:
            mock_validate.return_value = ["max_docs必须大于0"]
            
            response = client.post("/graphrag/query", json=invalid_request)
            
            assert response.status_code == 400
            assert "max_docs必须大于0" in response.json()["detail"]
    
    def test_large_request_handling(self, client):
        """测试大型请求处理"""
        large_request = {
            "query": "x" * 10000,  # 非常长的查询
            "retrieval_mode": "hybrid",
            "max_docs": 1000,  # 大量文档请求
            "expansion_depth": 10  # 深度扩展
        }
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.enhanced_query.return_value = {
                "success": False,
                "error": "查询太长或复杂"
            }
            mock_get_engine.return_value = mock_engine
            
            response = client.post("/graphrag/query", json=large_request)
            
            # 应该能处理大型请求，即使返回错误
            assert response.status_code in [400, 500]
    
    def test_concurrent_requests(self, client, sample_request, mock_engine_response):
        """测试并发请求处理"""
        import concurrent.futures
        import threading
        
        with patch('src.ai.graphrag.core_engine.get_graphrag_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.enhanced_query.return_value = mock_engine_response
            mock_get_engine.return_value = mock_engine
            
            def make_request():
                return client.post("/graphrag/query", json=sample_request)
            
            # 并发发送多个请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(5)]
                responses = [future.result() for future in futures]
            
            # 所有请求都应该成功
            for response in responses:
                assert response.status_code == 200
                assert response.json()["success"] is True


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__])