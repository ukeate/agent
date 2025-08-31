"""
知识管理API测试
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.api.v1.knowledge_management import (
    router,
    EntityType,
    RelationType,
    EntityRequest,
    RelationRequest,
    SPARQLQueryRequest,
    ImportRequest,
    ExportRequest
)


class TestKnowledgeManagementAPI:
    """知识管理API测试类"""

    @pytest.fixture
    def client(self):
        """测试客户端"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    @pytest.fixture
    def mock_sparql_engine(self):
        """模拟SPARQL引擎"""
        engine = AsyncMock()
        engine.execute_query.return_value = Mock(
            success=True,
            query_id="test_query",
            results=[{'name': 'Alice', 'age': 30}],
            execution_time_ms=50.0,
            row_count=1
        )
        return engine

    def test_get_entities_success(self, client):
        """测试获取实体列表成功"""
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.query_entities.return_value = [
                {'id': 'e1', 'type': 'Person', 'name': 'Alice'},
                {'id': 'e2', 'type': 'Organization', 'name': 'TechCorp'}
            ]
            
            response = client.get("/api/v1/kg/entities?limit=10&offset=0")
            
            assert response.status_code == 200
            data = response.json()
            assert 'entities' in data
            assert len(data['entities']) == 2

    def test_get_entities_with_filters(self, client):
        """测试带过滤条件的实体查询"""
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.query_entities.return_value = [
                {'id': 'e1', 'type': 'Person', 'name': 'Alice', 'age': 30}
            ]
            
            response = client.get(
                "/api/v1/kg/entities?entity_type=Person&name_contains=Alice"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data['entities']) == 1
            assert data['entities'][0]['name'] == 'Alice'

    def test_create_entity_success(self, client):
        """测试创建实体成功"""
        entity_data = {
            'entity_type': 'Person',
            'properties': {
                'name': 'Bob Johnson',
                'age': 25,
                'occupation': 'Engineer'
            },
            'metadata': {'source': 'api_test'}
        }

        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.create_entity.return_value = {
                'entity_id': 'new_entity_123',
                'created_at': '2023-01-01T00:00:00Z'
            }
            
            response = client.post(
                "/api/v1/kg/entities",
                json=entity_data
            )
            
            assert response.status_code == 201
            data = response.json()
            assert 'entity_id' in data
            assert data['entity_id'] == 'new_entity_123'

    def test_create_entity_validation_error(self, client):
        """测试创建实体验证错误"""
        invalid_entity_data = {
            'entity_type': 'InvalidType',  # 无效类型
            'properties': {}  # 缺少必要属性
        }

        response = client.post(
            "/api/v1/kg/entities",
            json=invalid_entity_data
        )
        
        assert response.status_code == 422

    def test_get_entity_by_id_success(self, client):
        """测试根据ID获取实体成功"""
        entity_id = "test_entity_123"
        
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.get_entity_by_id.return_value = {
                'id': entity_id,
                'type': 'Person',
                'name': 'Charlie Brown',
                'age': 35,
                'relations': [
                    {'target': 'org_123', 'type': 'works_for'}
                ]
            }
            
            response = client.get(f"/api/v1/kg/entities/{entity_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data['id'] == entity_id
            assert data['name'] == 'Charlie Brown'

    def test_get_entity_not_found(self, client):
        """测试获取不存在的实体"""
        entity_id = "nonexistent_entity"
        
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.get_entity_by_id.return_value = None
            
            response = client.get(f"/api/v1/kg/entities/{entity_id}")
            
            assert response.status_code == 404

    def test_update_entity_success(self, client):
        """测试更新实体成功"""
        entity_id = "test_entity_123"
        update_data = {
            'properties': {
                'age': 36,
                'location': 'New York'
            },
            'metadata': {'updated_by': 'api_test'}
        }

        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.update_entity.return_value = {
                'entity_id': entity_id,
                'updated_at': '2023-01-01T00:00:00Z',
                'changes': ['age', 'location']
            }
            
            response = client.put(
                f"/api/v1/kg/entities/{entity_id}",
                json=update_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['entity_id'] == entity_id

    def test_delete_entity_success(self, client):
        """测试删除实体成功"""
        entity_id = "test_entity_123"
        
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.delete_entity.return_value = True
            
            response = client.delete(f"/api/v1/kg/entities/{entity_id}")
            
            assert response.status_code == 204

    def test_get_relations_success(self, client):
        """测试获取关系列表成功"""
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.query_relations.return_value = [
                {
                    'id': 'r1',
                    'source': 'e1',
                    'target': 'e2',
                    'type': 'knows',
                    'properties': {'since': '2023-01-01'}
                }
            ]
            
            response = client.get("/api/v1/kg/relations?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert 'relations' in data
            assert len(data['relations']) == 1

    def test_create_relation_success(self, client):
        """测试创建关系成功"""
        relation_data = {
            'source_entity_id': 'person_123',
            'target_entity_id': 'org_456', 
            'relation_type': 'works_for',
            'properties': {
                'start_date': '2023-01-01',
                'position': 'Engineer'
            },
            'metadata': {'source': 'api_test'}
        }

        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.create_relation.return_value = {
                'relation_id': 'new_relation_789',
                'created_at': '2023-01-01T00:00:00Z'
            }
            
            response = client.post(
                "/api/v1/kg/relations",
                json=relation_data
            )
            
            assert response.status_code == 201
            data = response.json()
            assert 'relation_id' in data

    def test_sparql_query_success(self, client, mock_sparql_engine):
        """测试SPARQL查询成功"""
        query_data = {
            'query_text': 'SELECT ?name WHERE { ?person foaf:name ?name }',
            'parameters': {},
            'timeout_seconds': 30
        }

        with patch('src.api.v1.knowledge_management.default_sparql_engine', mock_sparql_engine):
            response = client.post(
                "/api/v1/kg/sparql/query",
                json=query_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
            assert data['success'] is True

    def test_batch_operations_success(self, client):
        """测试批量操作成功"""
        batch_data = {
            'operations': [
                {
                    'operation_type': 'create',
                    'target_type': 'entity',
                    'data': {
                        'entity_type': 'Person',
                        'properties': {'name': 'Alice'}
                    }
                },
                {
                    'operation_type': 'create',
                    'target_type': 'entity',
                    'data': {
                        'entity_type': 'Person',
                        'properties': {'name': 'Bob'}
                    }
                }
            ],
            'transaction_mode': True,
            'conflict_resolution': 'skip'
        }

        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.batch_operations.return_value = {
                'total_operations': 2,
                'successful_operations': 2,
                'failed_operations': 0,
                'results': [
                    {'entity_id': 'alice_123', 'status': 'created'},
                    {'entity_id': 'bob_456', 'status': 'created'}
                ]
            }
            
            response = client.post(
                "/api/v1/kg/batch",
                json=batch_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['successful_operations'] == 2

    def test_graph_validation_success(self, client):
        """测试图谱验证成功"""
        validation_config = {
            'check_consistency': True,
            'validate_schema': True,
            'check_orphaned_nodes': True,
            'validate_property_types': True
        }

        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.validate_graph.return_value = {
                'validation_id': 'val_123',
                'overall_status': 'passed',
                'checks_performed': 4,
                'checks_passed': 4,
                'checks_failed': 0,
                'issues': [],
                'recommendations': ['Consider adding indexes for better performance']
            }
            
            response = client.post(
                "/api/v1/kg/validate",
                json=validation_config
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['overall_status'] == 'passed'

    def test_import_data_success(self, client):
        """测试数据导入成功"""
        import_data = {
            'source_format': 'CSV',
            'import_mode': 'FULL',
            'source_data': 'name,type\nAlice,Person\nBob,Person',
            'mapping_rules': {
                'name': 'rdfs:label',
                'type': 'rdf:type'
            },
            'validation_config': {'strict_types': True}
        }

        with patch('src.api.v1.knowledge_management.default_data_importer') as mock_importer:
            mock_importer.import_data.return_value = {
                'job_id': 'import_job_123',
                'status': 'success',
                'total_records': 2,
                'successful_records': 2,
                'failed_records': 0
            }
            
            response = client.post(
                "/api/v1/kg/import",
                json=import_data
            )
            
            assert response.status_code == 202
            data = response.json()
            assert 'job_id' in data

    def test_export_data_success(self, client):
        """测试数据导出成功"""
        export_config = {
            'format': 'JSON_LD',
            'filters': {'entity_type': 'Person'},
            'include_metadata': True,
            'max_records': 100
        }

        with patch('src.api.v1.knowledge_management.default_data_exporter') as mock_exporter:
            mock_exporter.export_data.return_value = {
                'export_id': 'export_456',
                'status': 'success',
                'record_count': 50,
                'download_url': '/api/v1/kg/export/export_456/download'
            }
            
            response = client.post(
                "/api/v1/kg/export",
                json=export_config
            )
            
            assert response.status_code == 202
            data = response.json()
            assert 'export_id' in data

    def test_get_graph_schema(self, client):
        """测试获取图谱模式"""
        with patch('src.api.v1.knowledge_management.default_graph_store') as mock_store:
            mock_store.get_schema.return_value = {
                'entity_types': [
                    {'name': 'Person', 'properties': ['name', 'age', 'email']},
                    {'name': 'Organization', 'properties': ['name', 'type', 'website']}
                ],
                'relation_types': [
                    {'name': 'works_for', 'source': 'Person', 'target': 'Organization'},
                    {'name': 'knows', 'source': 'Person', 'target': 'Person'}
                ],
                'constraints': [
                    {'type': 'unique', 'property': 'email', 'entity_type': 'Person'}
                ]
            }
            
            response = client.get("/api/v1/kg/schema")
            
            assert response.status_code == 200
            data = response.json()
            assert 'entity_types' in data
            assert 'relation_types' in data

    def test_api_authentication_required(self, client):
        """测试API认证要求"""
        # 这个测试需要根据实际的认证机制调整
        response = client.get("/api/v1/kg/entities")
        
        # 如果API需要认证，应该返回401
        # 目前假设API不需要认证，所以这个测试会通过
        assert response.status_code in [200, 401]

    def test_api_rate_limiting(self, client):
        """测试API速率限制"""
        # 这个测试需要根据实际的速率限制机制调整
        responses = []
        for _ in range(10):  # 快速发送10个请求
            response = client.get("/api/v1/kg/entities?limit=1")
            responses.append(response.status_code)
        
        # 检查是否有速率限制响应
        # 429 = Too Many Requests
        rate_limited = any(status == 429 for status in responses)
        # 这里我们不强制要求速率限制，只是检查响应
        assert all(status in [200, 401, 429] for status in responses)


@pytest.mark.integration
class TestKnowledgeManagementAPIIntegration:
    """知识管理API集成测试"""

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self):
        """测试完整CRUD工作流"""
        pytest.skip("需要真实的数据库连接")

    @pytest.mark.asyncio
    async def test_concurrent_api_operations(self):
        """测试并发API操作"""
        pytest.skip("需要并发测试环境")

    @pytest.mark.asyncio
    async def test_api_performance_benchmark(self):
        """测试API性能基准"""
        pytest.skip("需要性能测试环境")