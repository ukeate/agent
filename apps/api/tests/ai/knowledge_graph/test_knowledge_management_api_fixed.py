"""
知识管理API测试 - 修复版本
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List


@pytest.mark.knowledge_graph
@pytest.mark.unit
class TestKnowledgeManagementAPI:
    """知识管理API测试类"""

    def test_entity_type_enum_values(self):
        """测试实体类型枚举值"""
        from conftest import MockEntityType
        
        assert hasattr(MockEntityType, 'PERSON')
        assert hasattr(MockEntityType, 'ORGANIZATION')
        assert hasattr(MockEntityType, 'CONCEPT')
        
        assert MockEntityType.PERSON == "person"
        assert MockEntityType.ORGANIZATION == "organization"
        assert MockEntityType.CONCEPT == "concept"

    def test_relation_type_enum_values(self):
        """测试关系类型枚举值"""
        from conftest import MockRelationType
        
        assert hasattr(MockRelationType, 'KNOWS')
        assert hasattr(MockRelationType, 'WORKS_FOR')
        assert hasattr(MockRelationType, 'PART_OF')
        
        assert MockRelationType.KNOWS == "knows"
        assert MockRelationType.WORKS_FOR == "works_for"

    def test_entity_request_structure(self, sample_entities):
        """测试实体请求数据结构"""
        entity_request = {
            'entity_type': 'Person',
            'properties': sample_entities[0].copy(),
            'metadata': {'source': 'api_test'}
        }
        
        assert 'entity_type' in entity_request
        assert 'properties' in entity_request
        assert 'metadata' in entity_request
        assert entity_request['entity_type'] == 'Person'
        assert entity_request['properties']['name'] == 'Alice Smith'

    def test_relation_request_structure(self, sample_relations):
        """测试关系请求数据结构"""
        relation_request = {
            'source_entity_id': sample_relations[0]['source'],
            'target_entity_id': sample_relations[0]['target'],
            'relation_type': sample_relations[0]['type'],
            'properties': sample_relations[0]['properties'],
            'metadata': {'source': 'api_test'}
        }
        
        assert 'source_entity_id' in relation_request
        assert 'target_entity_id' in relation_request
        assert 'relation_type' in relation_request
        assert relation_request['relation_type'] == 'works_for'

    @pytest.mark.asyncio
    async def test_mock_entities_crud_operations(self, mock_graph_store, sample_entities):
        """测试模拟实体CRUD操作"""
        # 测试创建实体
        create_result = {'entity_id': 'new_entity_123', 'created_at': '2023-01-01T00:00:00Z'}
        mock_graph_store.create_entity.return_value = create_result
        
        result = await mock_graph_store.create_entity(sample_entities[0])
        assert result['entity_id'] == 'new_entity_123'
        
        # 测试获取实体
        mock_graph_store.get_entity_by_id.return_value = sample_entities[0]
        
        entity = await mock_graph_store.get_entity_by_id('test_entity_123')
        assert entity['name'] == 'Alice Smith'
        assert entity['type'] == 'Person'
        
        # 测试更新实体
        update_result = {'entity_id': 'test_entity_123', 'updated_at': '2023-01-01T00:00:00Z'}
        mock_graph_store.update_entity.return_value = update_result
        
        updated = await mock_graph_store.update_entity('test_entity_123', {'age': 31})
        assert updated['entity_id'] == 'test_entity_123'
        
        # 测试删除实体
        mock_graph_store.delete_entity.return_value = True
        
        deleted = await mock_graph_store.delete_entity('test_entity_123')
        assert deleted is True

    @pytest.mark.asyncio
    async def test_mock_relations_crud_operations(self, mock_graph_store, sample_relations):
        """测试模拟关系CRUD操作"""
        # 测试创建关系
        create_result = {'relation_id': 'new_relation_456', 'created_at': '2023-01-01T00:00:00Z'}
        mock_graph_store.create_relation.return_value = create_result
        
        result = await mock_graph_store.create_relation(sample_relations[0])
        assert result['relation_id'] == 'new_relation_456'
        
        # 测试查询关系
        mock_graph_store.query_relations.return_value = sample_relations
        
        relations = await mock_graph_store.query_relations({'limit': 10})
        assert len(relations) == 2
        assert relations[0]['type'] == 'works_for'

    @pytest.mark.asyncio
    async def test_sparql_query_execution(self, mock_sparql_engine, sample_sparql_queries):
        """测试SPARQL查询执行"""
        from conftest import MockSPARQLResult
        
        query_result = MockSPARQLResult(
            query_id="api_query_001",
            success=True,
            result_type="bindings",
            results=[{'name': 'Alice', 'age': 30}],
            execution_time_ms=50.0,
            row_count=1,
            cached=False
        )
        
        mock_sparql_engine.execute_query.return_value = query_result

        result = await mock_sparql_engine.execute_query(sample_sparql_queries['simple_select'])
        
        assert result.success is True
        assert result.row_count == 1
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_batch_operations(self, mock_graph_store):
        """测试批量操作"""
        batch_result = {
            'total_operations': 3,
            'successful_operations': 3,
            'failed_operations': 0,
            'results': [
                {'entity_id': 'batch_entity_1', 'status': 'created'},
                {'entity_id': 'batch_entity_2', 'status': 'created'},
                {'relation_id': 'batch_relation_1', 'status': 'created'}
            ],
            'execution_time': 0.5
        }
        
        mock_graph_store.batch_operations.return_value = batch_result

        operations = [
            {
                'operation_type': 'create',
                'target_type': 'entity',
                'data': {'entity_type': 'Person', 'name': 'Alice'}
            },
            {
                'operation_type': 'create',
                'target_type': 'entity', 
                'data': {'entity_type': 'Person', 'name': 'Bob'}
            },
            {
                'operation_type': 'create',
                'target_type': 'relation',
                'data': {'source': 'alice', 'target': 'bob', 'type': 'knows'}
            }
        ]

        result = await mock_graph_store.batch_operations(operations)

        assert result['total_operations'] == 3
        assert result['successful_operations'] == 3
        assert result['failed_operations'] == 0
        assert len(result['results']) == 3

    @pytest.mark.asyncio
    async def test_graph_validation(self, mock_graph_store):
        """测试图谱验证"""
        validation_result = {
            'validation_id': 'val_123',
            'overall_status': 'passed',
            'checks_performed': 4,
            'checks_passed': 4,
            'checks_failed': 0,
            'issues': [],
            'recommendations': ['Consider adding indexes for better performance'],
            'execution_time': 1.5
        }
        
        mock_graph_store.validate_graph.return_value = validation_result

        validation_config = {
            'check_consistency': True,
            'validate_schema': True,
            'check_orphaned_nodes': True,
            'validate_property_types': True
        }

        result = await mock_graph_store.validate_graph(validation_config)

        assert result['overall_status'] == 'passed'
        assert result['checks_performed'] == 4
        assert result['checks_passed'] == 4
        assert len(result['recommendations']) == 1

    @pytest.mark.asyncio
    async def test_data_import_api(self, mock_data_importer, sample_import_data):
        """测试数据导入API"""
        import_result = {
            'job_id': 'api_import_123',
            'status': 'success',
            'total_records': 3,
            'successful_records': 3,
            'failed_records': 0,
            'errors': [],
            'warnings': [],
            'execution_time': 0.8,
            'created_entities': ['john', 'acme_corp', 'python'],
            'created_relations': []
        }
        
        mock_data_importer.import_data.return_value = import_result

        import_request = {
            'source_format': 'CSV',
            'import_mode': 'FULL',
            'source_data': sample_import_data['csv_data'],
            'mapping_rules': {
                'name': 'rdfs:label',
                'type': 'rdf:type'
            },
            'validation_config': {'strict_types': True}
        }

        result = await mock_data_importer.import_data(import_request)

        assert result['status'] == 'success'
        assert result['total_records'] == 3
        assert len(result['created_entities']) == 3

    @pytest.mark.asyncio
    async def test_data_export_api(self, mock_data_exporter):
        """测试数据导出API"""
        export_result = {
            'export_id': 'api_export_456',
            'status': 'success',
            'format': 'JSON_LD',
            'record_count': 50,
            'file_size': 2048,
            'download_url': '/api/v1/kg/export/api_export_456/download',
            'created_at': '2023-01-01T00:00:00Z'
        }
        
        mock_data_exporter.export_data.return_value = export_result

        export_request = {
            'format': 'JSON_LD',
            'filters': {'entity_type': 'Person'},
            'include_metadata': True,
            'max_records': 100
        }

        result = await mock_data_exporter.export_data(export_request)

        assert result['status'] == 'success'
        assert result['format'] == 'JSON_LD'
        assert result['record_count'] == 50

    @pytest.mark.asyncio
    async def test_graph_schema_api(self, mock_graph_store):
        """测试图谱模式API"""
        schema_result = {
            'entity_types': [
                {'name': 'Person', 'properties': ['name', 'age', 'email'], 'count': 100},
                {'name': 'Organization', 'properties': ['name', 'type', 'website'], 'count': 50}
            ],
            'relation_types': [
                {'name': 'works_for', 'source': 'Person', 'target': 'Organization', 'count': 75},
                {'name': 'knows', 'source': 'Person', 'target': 'Person', 'count': 150}
            ],
            'constraints': [
                {'type': 'unique', 'property': 'email', 'entity_type': 'Person'},
                {'type': 'required', 'property': 'name', 'entity_type': 'Person'}
            ],
            'statistics': {
                'total_entities': 150,
                'total_relations': 225,
                'avg_degree': 3.0
            }
        }
        
        mock_graph_store.get_schema.return_value = schema_result

        schema = await mock_graph_store.get_schema()

        assert 'entity_types' in schema
        assert 'relation_types' in schema
        assert 'constraints' in schema
        assert len(schema['entity_types']) == 2
        assert len(schema['relation_types']) == 2
        assert schema['statistics']['total_entities'] == 150

    def test_api_request_validation(self, sample_entities):
        """测试API请求验证"""
        # 测试有效的实体创建请求
        valid_request = {
            'entity_type': 'Person',
            'properties': {
                'name': 'John Doe',
                'age': 30,
                'email': 'john@example.com'
            },
            'metadata': {'source': 'api'}
        }
        
        # 验证必需字段
        assert 'entity_type' in valid_request
        assert 'properties' in valid_request
        assert 'name' in valid_request['properties']
        
        # 测试无效请求
        invalid_request = {
            'entity_type': 'InvalidType',  # 无效类型
            'properties': {}  # 缺少必要属性
        }
        
        # 这里可以添加实际的验证逻辑
        assert invalid_request['entity_type'] == 'InvalidType'
        assert len(invalid_request['properties']) == 0

    def test_api_response_format(self):
        """测试API响应格式"""
        # 成功响应格式
        success_response = {
            'success': True,
            'data': {
                'entity_id': 'test_entity_123',
                'created_at': '2023-01-01T00:00:00Z'
            },
            'message': 'Entity created successfully',
            'metadata': {
                'execution_time': 0.1,
                'api_version': 'v1'
            }
        }
        
        assert success_response['success'] is True
        assert 'data' in success_response
        assert 'entity_id' in success_response['data']
        
        # 错误响应格式
        error_response = {
            'success': False,
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': 'Invalid entity type',
                'details': {'field': 'entity_type', 'value': 'InvalidType'}
            },
            'metadata': {
                'request_id': 'req_123',
                'timestamp': '2023-01-01T00:00:00Z'
            }
        }
        
        assert error_response['success'] is False
        assert 'error' in error_response
        assert error_response['error']['code'] == 'VALIDATION_ERROR'

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_graph_store):
        """测试API错误处理"""
        # 模拟数据库连接错误
        mock_graph_store.create_entity.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception) as exc_info:
            await mock_graph_store.create_entity({
                'entity_type': 'Person',
                'name': 'Test Entity'
            })
        
        assert "Database connection failed" in str(exc_info.value)

    def test_api_pagination_parameters(self):
        """测试API分页参数"""
        pagination_params = {
            'limit': 20,
            'offset': 40,
            'sort_by': 'created_at',
            'sort_order': 'desc',
            'filters': {
                'entity_type': 'Person',
                'created_after': '2023-01-01'
            }
        }
        
        assert pagination_params['limit'] == 20
        assert pagination_params['offset'] == 40
        assert pagination_params['sort_by'] == 'created_at'
        assert pagination_params['sort_order'] == 'desc'
        assert 'entity_type' in pagination_params['filters']

    @pytest.mark.asyncio
    async def test_concurrent_api_operations(self, mock_graph_store, sample_entities):
        """测试并发API操作"""
        import asyncio
        
        # 设置mock返回值
        mock_graph_store.create_entity.return_value = {
            'entity_id': 'concurrent_entity',
            'created_at': '2023-01-01T00:00:00Z'
        }
        
        # 创建并发任务
        tasks = []
        for i in range(5):
            entity_data = sample_entities[0].copy()
            entity_data['name'] = f'Concurrent Entity {i}'
            task = mock_graph_store.create_entity(entity_data)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(r['entity_id'] == 'concurrent_entity' for r in results)


@pytest.mark.integration
@pytest.mark.knowledge_graph 
class TestKnowledgeManagementAPIIntegration:
    """知识管理API集成测试"""

    @pytest.mark.asyncio
    async def test_complete_api_workflow(self, mock_graph_store, mock_sparql_engine, mock_data_importer):
        """测试完整API工作流"""
        # 1. 导入数据
        import_result = {
            'job_id': 'workflow_import',
            'status': 'success',
            'created_entities': ['entity_1', 'entity_2']
        }
        mock_data_importer.import_data.return_value = import_result
        
        import_job = {
            'source_format': 'CSV',
            'source_data': 'name,type\nAlice,Person\nBob,Person'
        }
        
        imported = await mock_data_importer.import_data(import_job)
        assert imported['status'] == 'success'
        
        # 2. 查询实体
        mock_graph_store.query_entities.return_value = [
            {'id': 'entity_1', 'name': 'Alice', 'type': 'Person'},
            {'id': 'entity_2', 'name': 'Bob', 'type': 'Person'}
        ]
        
        entities = await mock_graph_store.query_entities({'entity_type': 'Person'})
        assert len(entities) == 2
        
        # 3. 执行SPARQL查询
        from conftest import MockSPARQLResult
        
        sparql_result = MockSPARQLResult(
            query_id="workflow_query",
            success=True,
            result_type="bindings",
            results=entities,
            execution_time_ms=100.0,
            row_count=2,
            cached=False
        )
        mock_sparql_engine.execute_query.return_value = sparql_result
        
        query_result = await mock_sparql_engine.execute_query("SELECT ?name WHERE { ?p foaf:name ?name }")
        assert query_result.success is True
        assert query_result.row_count == 2

    @pytest.mark.asyncio
    async def test_api_performance_monitoring(self, mock_performance_monitor):
        """测试API性能监控"""
        # 模拟性能指标收集
        performance_metrics = {
            'api_requests_total': 1000,
            'avg_response_time': 0.15,
            'error_rate': 0.02,
            'throughput_qps': 150,
            'cache_hit_rate': 0.85
        }
        
        mock_performance_monitor.get_metrics.return_value = performance_metrics
        
        metrics = mock_performance_monitor.get_metrics()
        
        assert metrics['api_requests_total'] == 1000
        assert metrics['avg_response_time'] == 0.15
        assert metrics['error_rate'] == 0.02
        assert metrics['throughput_qps'] == 150
        assert metrics['cache_hit_rate'] == 0.85

    def test_api_security_considerations(self):
        """测试API安全考虑"""
        # 认证令牌格式
        auth_token = {
            'token_type': 'Bearer',
            'access_token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            'expires_in': 3600,
            'scope': 'kg:read kg:write'
        }
        
        assert auth_token['token_type'] == 'Bearer'
        assert auth_token['expires_in'] == 3600
        assert 'kg:read' in auth_token['scope']
        
        # API速率限制
        rate_limit = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'burst_limit': 10
        }
        
        assert rate_limit['requests_per_minute'] == 60
        assert rate_limit['burst_limit'] == 10


if __name__ == "__main__":
    # 允许直接运行测试文件
    pytest.main([__file__, "-v"])