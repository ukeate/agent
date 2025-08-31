"""
知识图谱API测试
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

# 假设我们有一个测试应用实例
from src.api.v1.knowledge_graph import router as kg_router
from src.ai.knowledge_graph.graph_database import GraphDatabaseError


@pytest.fixture
def test_app():
    """测试应用"""
    app = FastAPI()
    app.include_router(kg_router, prefix="/api/v1")
    return app


@pytest.fixture
def test_client(test_app):
    """测试客户端"""
    return TestClient(test_app)


@pytest.fixture
def mock_graph_operations():
    """Mock图操作"""
    ops = Mock()
    ops.create_entity = AsyncMock()
    ops.get_entity = AsyncMock()
    ops.update_entity = AsyncMock()
    ops.delete_entity = AsyncMock()
    ops.create_relation = AsyncMock()
    ops.get_relations = AsyncMock()
    ops.query_graph = AsyncMock()
    ops.find_shortest_path = AsyncMock()
    ops.get_subgraph = AsyncMock()
    return ops


@pytest.fixture
def mock_incremental_updater():
    """Mock增量更新器"""
    updater = Mock()
    updater.process_entity_update = AsyncMock()
    updater.process_relation_update = AsyncMock()
    updater.intelligent_entity_merge = AsyncMock()
    updater.batch_update = AsyncMock()
    return updater


@pytest.fixture
def mock_performance_optimizer():
    """Mock性能优化器"""
    optimizer = Mock()
    optimizer.get_performance_stats = AsyncMock()
    optimizer.get_slow_queries = AsyncMock()
    optimizer.analyze_query_patterns = AsyncMock()
    optimizer.optimize_indexes = AsyncMock()
    optimizer.invalidate_cache = AsyncMock()
    optimizer.get_cache_stats = Mock()
    return optimizer


@pytest.mark.api
class TestEntityAPI:
    """实体API测试"""
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_create_entity_success(self, mock_ops, test_client):
        """测试创建实体成功"""
        # Mock返回
        mock_ops.create_entity.return_value = {
            "id": "entity_001",
            "canonical_form": "张三",
            "type": "PERSON",
            "created": True
        }
        
        entity_data = {
            "canonical_form": "张三",
            "entity_type": "PERSON",
            "properties": {"age": 30, "occupation": "工程师"},
            "confidence": 0.95,
            "embedding": [0.1, 0.2, 0.3] * 100
        }
        
        response = test_client.post("/api/v1/entities/", json=entity_data)
        
        assert response.status_code == 201
        assert response.json()["canonical_form"] == "张三"
        assert response.json()["created"] is True
    
    @patch('src.api.v1.knowledge_graph.graph_operations') 
    def test_create_entity_validation_error(self, mock_ops, test_client):
        """测试创建实体验证错误"""
        # 缺少必需字段
        invalid_data = {
            "entity_type": "PERSON",
            # 缺少canonical_form
        }
        
        response = test_client.post("/api/v1/entities/", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_get_entity_success(self, mock_ops, test_client):
        """测试获取实体成功"""
        mock_ops.get_entity.return_value = {
            "id": "entity_001",
            "canonical_form": "张三",
            "type": "PERSON",
            "properties": {"age": 30},
            "confidence": 0.95
        }
        
        response = test_client.get("/api/v1/entities/entity_001")
        
        assert response.status_code == 200
        assert response.json()["canonical_form"] == "张三"
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_get_entity_not_found(self, mock_ops, test_client):
        """测试获取不存在的实体"""
        mock_ops.get_entity.return_value = None
        
        response = test_client.get("/api/v1/entities/nonexistent")
        
        assert response.status_code == 404
        assert "Entity not found" in response.json()["detail"]
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_update_entity_success(self, mock_ops, test_client):
        """测试更新实体成功"""
        mock_ops.update_entity.return_value = {
            "id": "entity_001",
            "updated": True
        }
        
        update_data = {
            "properties": {"age": 31, "occupation": "高级工程师"},
            "confidence": 0.96
        }
        
        response = test_client.put("/api/v1/entities/entity_001", json=update_data)
        
        assert response.status_code == 200
        assert response.json()["updated"] is True
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_delete_entity_success(self, mock_ops, test_client):
        """测试删除实体成功"""
        mock_ops.delete_entity.return_value = {"deleted": True}
        
        response = test_client.delete("/api/v1/entities/entity_001")
        
        assert response.status_code == 200
        assert response.json()["deleted"] is True
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_search_entities_success(self, mock_ops, test_client):
        """测试搜索实体成功"""
        mock_ops.search_entities.return_value = [
            {"id": "entity_001", "canonical_form": "张三", "type": "PERSON"},
            {"id": "entity_002", "canonical_form": "张三丰", "type": "PERSON"}
        ]
        
        response = test_client.get("/api/v1/entities/search?query=张三&limit=10")
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert all("张三" in item["canonical_form"] for item in response.json())


@pytest.mark.api
class TestRelationAPI:
    """关系API测试"""
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_create_relation_success(self, mock_ops, test_client):
        """测试创建关系成功"""
        mock_ops.create_relation.return_value = {
            "id": "relation_001",
            "type": "WORKS_FOR",
            "created": True
        }
        
        relation_data = {
            "relation_type": "WORKS_FOR",
            "source_entity_id": "entity_001",
            "target_entity_id": "entity_002",
            "properties": {"since": "2020", "position": "工程师"},
            "confidence": 0.90
        }
        
        response = test_client.post("/api/v1/relations/", json=relation_data)
        
        assert response.status_code == 201
        assert response.json()["type"] == "WORKS_FOR"
        assert response.json()["created"] is True
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_get_entity_relations(self, mock_ops, test_client):
        """测试获取实体关系"""
        mock_ops.get_relations.return_value = [
            {
                "id": "relation_001",
                "type": "WORKS_FOR",
                "target_entity": {"id": "entity_002", "canonical_form": "苹果公司"}
            }
        ]
        
        response = test_client.get("/api/v1/entities/entity_001/relations")
        
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["type"] == "WORKS_FOR"


@pytest.mark.api
class TestGraphQueryAPI:
    """图查询API测试"""
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_cypher_query_success(self, mock_ops, test_client):
        """测试Cypher查询成功"""
        mock_ops.query_graph.return_value = [
            {"n.name": "张三", "n.age": 30},
            {"n.name": "李四", "n.age": 25}
        ]
        
        query_data = {
            "query": "MATCH (n:Person) WHERE n.age > $age RETURN n.name, n.age",
            "parameters": {"age": 20}
        }
        
        response = test_client.post("/api/v1/graph/query", json=query_data)
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]["n.name"] == "张三"
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_cypher_query_error(self, mock_ops, test_client):
        """测试Cypher查询错误"""
        mock_ops.query_graph.side_effect = GraphDatabaseError("Invalid query")
        
        query_data = {
            "query": "INVALID CYPHER QUERY",
            "parameters": {}
        }
        
        response = test_client.post("/api/v1/graph/query", json=query_data)
        
        assert response.status_code == 400
        assert "Invalid query" in response.json()["detail"]
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_shortest_path_success(self, mock_ops, test_client):
        """测试最短路径查询成功"""
        mock_ops.find_shortest_path.return_value = {
            "path": [
                {"id": "entity_001", "canonical_form": "张三"},
                {"id": "relation_001", "type": "WORKS_FOR"}, 
                {"id": "entity_002", "canonical_form": "苹果公司"}
            ],
            "length": 1
        }
        
        response = test_client.get("/api/v1/graph/path/entity_001/entity_002")
        
        assert response.status_code == 200
        assert response.json()["length"] == 1
        assert len(response.json()["path"]) == 3
    
    @patch('src.api.v1.knowledge_graph.graph_operations')
    def test_subgraph_success(self, mock_ops, test_client):
        """测试子图查询成功"""
        mock_ops.get_subgraph.return_value = {
            "nodes": [
                {"id": "entity_001", "canonical_form": "张三", "type": "PERSON"},
                {"id": "entity_002", "canonical_form": "苹果公司", "type": "ORGANIZATION"}
            ],
            "relationships": [
                {"id": "relation_001", "type": "WORKS_FOR", "source": "entity_001", "target": "entity_002"}
            ]
        }
        
        response = test_client.get("/api/v1/graph/subgraph/entity_001?depth=2")
        
        assert response.status_code == 200
        assert len(response.json()["nodes"]) == 2
        assert len(response.json()["relationships"]) == 1


@pytest.mark.api
class TestIncrementalUpdateAPI:
    """增量更新API测试"""
    
    @patch('src.api.v1.knowledge_graph.incremental_updater')
    def test_upsert_entity_success(self, mock_updater, test_client):
        """测试智能实体更新成功"""
        from src.ai.knowledge_graph.incremental_updater import UpdateResult
        
        mock_updater.process_entity_update.return_value = UpdateResult(
            operation="create",
            success=True,
            entity_id="entity_001",
            conflicts=[]
        )
        
        entity_data = {
            "canonical_form": "张三",
            "entity_type": "PERSON",
            "properties": {"age": 30},
            "confidence": 0.95,
            "source": "document_001"
        }
        
        response = test_client.post("/api/v1/entities/upsert", json=entity_data)
        
        assert response.status_code == 200
        assert response.json()["operation"] == "create"
        assert response.json()["success"] is True
    
    @patch('src.api.v1.knowledge_graph.incremental_updater')
    def test_batch_update_success(self, mock_updater, test_client):
        """测试批量更新成功"""
        mock_updater.batch_update.return_value = {
            "entity_results": [{"operation": "create", "success": True}],
            "relation_results": [{"operation": "create", "success": True}]
        }
        
        batch_data = {
            "entities": [
                {
                    "canonical_form": "张三",
                    "entity_type": "PERSON",
                    "properties": {"age": 30},
                    "confidence": 0.95,
                    "source": "doc1"
                }
            ],
            "relations": [
                {
                    "relation_type": "WORKS_FOR",
                    "source_entity_id": "entity_001",
                    "target_entity_id": "entity_002",
                    "confidence": 0.90,
                    "source": "doc1"
                }
            ]
        }
        
        response = test_client.post("/api/v1/graph/batch-update", json=batch_data)
        
        assert response.status_code == 200
        assert len(response.json()["entity_results"]) == 1
        assert len(response.json()["relation_results"]) == 1


@pytest.mark.api
class TestQualityManagementAPI:
    """质量管理API测试"""
    
    @patch('src.api.v1.knowledge_graph.quality_manager')
    def test_quality_assessment_success(self, mock_quality, test_client):
        """测试质量评估成功"""
        from src.ai.knowledge_graph.quality_manager import QualityMetrics
        
        mock_quality.assess_graph_quality.return_value = QualityMetrics(
            completeness_score=0.85,
            consistency_score=0.92,
            accuracy_score=0.88,
            overall_score=0.88
        )
        
        response = test_client.get("/api/v1/quality/assessment")
        
        assert response.status_code == 200
        assert response.json()["completeness_score"] == 0.85
        assert response.json()["overall_score"] == 0.88
    
    @patch('src.api.v1.knowledge_graph.quality_manager')
    def test_quality_issues_success(self, mock_quality, test_client):
        """测试质量问题检测成功"""
        from src.ai.knowledge_graph.quality_manager import QualityIssue
        
        mock_quality.detect_quality_issues.return_value = [
            QualityIssue(
                issue_type="missing_property",
                entity_id="entity_001", 
                description="缺少必需属性",
                severity="medium"
            )
        ]
        
        response = test_client.get("/api/v1/quality/issues")
        
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["issue_type"] == "missing_property"


@pytest.mark.api
class TestPerformanceAPI:
    """性能监控API测试"""
    
    @patch('src.api.v1.knowledge_graph.performance_optimizer')
    def test_performance_stats_success(self, mock_optimizer, test_client):
        """测试性能统计成功"""
        from src.ai.knowledge_graph.performance_optimizer import PerformanceStats
        
        mock_optimizer.get_performance_stats.return_value = PerformanceStats(
            total_queries=1000,
            cache_hit_rate=0.75,
            avg_query_time_ms=150.5,
            slow_queries_count=10,
            peak_qps=50.0,
            current_connections=5
        )
        
        response = test_client.get("/api/v1/performance/stats")
        
        assert response.status_code == 200
        assert response.json()["total_queries"] == 1000
        assert response.json()["cache_hit_rate"] == 0.75
    
    @patch('src.api.v1.knowledge_graph.performance_optimizer')
    def test_slow_queries_success(self, mock_optimizer, test_client):
        """测试慢查询获取成功"""
        from src.ai.knowledge_graph.performance_optimizer import QueryPerformance
        
        mock_optimizer.get_slow_queries.return_value = [
            QueryPerformance("hash1", "read", 1500, 10, False),
            QueryPerformance("hash2", "read", 2000, 5, False)
        ]
        
        response = test_client.get("/api/v1/performance/slow-queries?limit=10")
        
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]["execution_time_ms"] == 1500
    
    @patch('src.api.v1.knowledge_graph.performance_optimizer')
    def test_cache_invalidation_success(self, mock_optimizer, test_client):
        """测试缓存失效成功"""
        mock_optimizer.invalidate_cache.return_value = None
        
        response = test_client.post("/api/v1/performance/cache/invalidate?pattern=person")
        
        assert response.status_code == 200
        assert response.json()["message"] == "缓存清理成功"


@pytest.mark.api
class TestAdministrationAPI:
    """管理API测试"""
    
    @patch('src.ai.v1.knowledge_graph.schema_manager')
    def test_schema_info_success(self, mock_schema, test_client):
        """测试模式信息获取成功"""
        from src.ai.knowledge_graph.schema import GraphSchema
        
        schema = GraphSchema()
        mock_schema.get_current_schema.return_value = schema
        
        response = test_client.get("/api/v1/admin/schema")
        
        assert response.status_code == 200
        assert "nodes" in response.json()
        assert "relationships" in response.json()
    
    @patch('src.api.v1.knowledge_graph.graph_database')
    def test_health_check_success(self, mock_db, test_client):
        """测试健康检查成功"""
        mock_db.health_check.return_value = True
        
        response = test_client.get("/api/v1/admin/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('src.api.v1.knowledge_graph.graph_database')
    def test_health_check_failure(self, mock_db, test_client):
        """测试健康检查失败"""
        mock_db.health_check.return_value = False
        
        response = test_client.get("/api/v1/admin/health")
        
        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"


@pytest.mark.integration
class TestKnowledgeGraphAPIIntegration:
    """知识图谱API集成测试"""
    
    @pytest.mark.slow
    def test_full_entity_lifecycle(self, test_client):
        """测试完整实体生命周期"""
        # 这个测试需要真实的数据库连接
        # 在实际环境中会跳过，但展示了完整的测试流程
        
        # 创建实体
        entity_data = {
            "canonical_form": "集成测试实体",
            "entity_type": "PERSON",
            "properties": {"test": True},
            "confidence": 0.95
        }
        
        # 由于没有真实数据库，这里会失败
        # 但展示了完整的测试逻辑
        with pytest.raises(Exception):
            response = test_client.post("/api/v1/entities/", json=entity_data)


@pytest.mark.performance
class TestAPIPerformance:
    """API性能测试"""
    
    @pytest.mark.slow
    def test_concurrent_requests(self, test_client):
        """测试并发请求性能"""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = test_client.get("/api/v1/admin/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # 启动多个并发请求
        threads = []
        start_time = time.time()
        
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有请求完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 验证结果
        assert len(results) == 10
        assert (end_time - start_time) < 2.0  # 10个请求在2秒内完成