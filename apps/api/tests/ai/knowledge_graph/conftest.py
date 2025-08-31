"""
知识图谱模块测试配置和fixtures
"""

import sys
import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# 添加项目源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

# 模拟缺失的依赖
sys.modules['spacy'] = Mock()
sys.modules['slowapi'] = Mock()
sys.modules['rdflib'] = Mock()
sys.modules['rdflib.plugins.sparql'] = Mock()

# 模拟OpenTelemetry
sys.modules['opentelemetry'] = Mock()
sys.modules['opentelemetry.sdk'] = Mock()
sys.modules['opentelemetry.instrumentation'] = Mock()

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_graph_store():
    """模拟图数据库存储"""
    store = Mock()
    store.get_statistics.return_value = {
        'total_entities': 1000,
        'total_relations': 2000,
        'entity_types': ['Person', 'Organization', 'Concept']
    }
    store.db_type = "mock"
    store.create_entity = AsyncMock(return_value={'entity_id': 'test_entity_123'})
    store.update_entity = AsyncMock(return_value={'entity_id': 'test_entity_123'})
    store.delete_entity = AsyncMock(return_value=True)
    store.get_entity_by_id = AsyncMock(return_value={
        'id': 'test_entity_123',
        'type': 'Person',
        'name': 'Test Entity'
    })
    store.query_entities = AsyncMock(return_value=[])
    store.query_relations = AsyncMock(return_value=[])
    store.create_relation = AsyncMock(return_value={'relation_id': 'test_relation_456'})
    store.batch_operations = AsyncMock(return_value={
        'total_operations': 0,
        'successful_operations': 0,
        'failed_operations': 0
    })
    store.validate_graph = AsyncMock(return_value={
        'validation_id': 'val_123',
        'overall_status': 'passed'
    })
    store.get_schema = AsyncMock(return_value={
        'entity_types': [],
        'relation_types': []
    })
    return store

@pytest.fixture
def mock_cache_manager():
    """模拟缓存管理器"""
    cache = AsyncMock()
    cache.get_query_result.return_value = None
    cache.cache_query_result = AsyncMock()
    return cache

@pytest.fixture
def mock_version_manager():
    """模拟版本管理器"""
    manager = AsyncMock()
    manager.create_import_version.return_value = Mock(version_id="import_v1")
    manager.finalize_import_version = AsyncMock()
    manager.rollback_version = AsyncMock()
    manager.create_version = AsyncMock()
    manager.compare_versions = AsyncMock()
    manager.rollback_to_version = AsyncMock(return_value=True)
    manager.list_versions = AsyncMock(return_value=[])
    manager.update_version_metadata = AsyncMock(return_value=True)
    return manager

@pytest.fixture
def mock_change_tracker():
    """模拟变更追踪器"""
    tracker = AsyncMock()
    tracker.record_change = AsyncMock(return_value=True)
    tracker.record_rollback = AsyncMock()
    tracker.get_changes_between_versions = AsyncMock(return_value=[])
    tracker.get_change_history = AsyncMock(return_value=[])
    return tracker

@pytest.fixture
def mock_data_importer():
    """模拟数据导入器"""
    importer = AsyncMock()
    importer.import_data = AsyncMock()
    return importer

@pytest.fixture
def mock_data_exporter():
    """模拟数据导出器"""
    exporter = AsyncMock()
    exporter.export_data = AsyncMock()
    return exporter

@pytest.fixture
def mock_sparql_engine():
    """模拟SPARQL引擎"""
    engine = AsyncMock()
    engine.execute_query = AsyncMock()
    engine.explain_query = AsyncMock()
    return engine

@pytest.fixture
def sample_entities():
    """示例实体数据"""
    return [
        {
            'id': 'person_1',
            'type': 'Person',
            'name': 'Alice Smith',
            'age': 30,
            'email': 'alice@example.com'
        },
        {
            'id': 'org_1',
            'type': 'Organization',
            'name': 'TechCorp',
            'type_detail': 'Technology Company'
        },
        {
            'id': 'concept_1',
            'type': 'Concept',
            'name': 'Artificial Intelligence',
            'description': 'AI technology concept'
        }
    ]

@pytest.fixture
def sample_relations():
    """示例关系数据"""
    return [
        {
            'id': 'rel_1',
            'source': 'person_1',
            'target': 'org_1',
            'type': 'works_for',
            'properties': {'since': '2023-01-01'}
        },
        {
            'id': 'rel_2', 
            'source': 'person_1',
            'target': 'concept_1',
            'type': 'knows_about',
            'properties': {'expertise_level': 'expert'}
        }
    ]

@pytest.fixture
def sample_sparql_queries():
    """示例SPARQL查询"""
    return {
        'simple_select': 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10',
        'complex_select': '''
            SELECT ?person ?name ?age WHERE {
                ?person rdf:type foaf:Person .
                ?person foaf:name ?name .
                ?person foaf:age ?age .
                FILTER(?age > 30)
            }
            ORDER BY ?name
        ''',
        'construct_query': 'CONSTRUCT { ?s a ?type } WHERE { ?s rdf:type ?type }',
        'ask_query': 'ASK { ?s rdf:type foaf:Person }'
    }

@pytest.fixture
def sample_import_data():
    """示例导入数据"""
    return {
        'csv_data': '''name,type,description
John,Person,A software engineer
ACME Corp,Organization,Technology company
Python,Concept,Programming language''',
        'json_ld_data': {
            "@context": {
                "foaf": "http://xmlns.com/foaf/0.1/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            },
            "@graph": [
                {
                    "@id": "person1",
                    "rdf:type": "foaf:Person",
                    "foaf:name": "Alice Smith"
                }
            ]
        }
    }

@pytest.fixture
def mock_performance_monitor():
    """模拟性能监控器"""
    monitor = Mock()
    monitor.record_query_time = Mock()
    monitor.record_operation_time = Mock()
    monitor.get_metrics = Mock(return_value={
        'avg_query_time': 0.5,
        'total_queries': 100,
        'cache_hit_rate': 0.8
    })
    return monitor

# 模拟知识图谱核心类型
class MockSPARQLQuery:
    """模拟SPARQL查询类"""
    def __init__(self, query_id, query_text, query_type, parameters, **kwargs):
        self.query_id = query_id
        self.query_text = query_text
        self.query_type = query_type
        self.parameters = parameters
        self.timeout_seconds = kwargs.get('timeout_seconds', 30)
        self.use_cache = kwargs.get('use_cache', True)

class MockSPARQLResult:
    """模拟SPARQL查询结果类"""
    def __init__(self, query_id, success, result_type, results, execution_time_ms, row_count, cached, **kwargs):
        self.query_id = query_id
        self.success = success
        self.result_type = result_type
        self.results = results
        self.execution_time_ms = execution_time_ms
        self.row_count = row_count
        self.cached = cached
        self.error_message = kwargs.get('error_message')

class MockImportJob:
    """模拟导入任务类"""
    def __init__(self, job_id, source_format, import_mode, source_data, mapping_rules, validation_config, metadata):
        self.data = {
            'job_id': job_id,
            'source_format': source_format,
            'import_mode': import_mode,
            'source_data': source_data,
            'mapping_rules': mapping_rules,
            'validation_config': validation_config,
            'metadata': metadata
        }
    
    def __getitem__(self, key):
        return self.data[key]
    
    def get(self, key, default=None):
        return self.data.get(key, default)

class MockGraphVersion:
    """模拟图谱版本类"""
    def __init__(self, version_id, version_number, parent_version, created_at, created_by, description, metadata, statistics, checksum):
        self.version_id = version_id
        self.version_number = version_number
        self.parent_version = parent_version
        self.created_at = created_at
        self.created_by = created_by
        self.description = description
        self.metadata = metadata
        self.statistics = statistics
        self.checksum = checksum

# 注册模拟类到全局命名空间
@pytest.fixture(autouse=True)
def setup_mock_classes(monkeypatch):
    """自动设置模拟类"""
    # 模拟枚举类型
    class MockQueryType:
        SELECT = "select"
        CONSTRUCT = "construct"
        ASK = "ask" 
        DESCRIBE = "describe"
        UPDATE = "update"
    
    class MockImportFormat:
        CSV = "csv"
        JSON_LD = "json_ld"
        TURTLE = "turtle"
        RDF_XML = "rdf_xml"
        N_TRIPLES = "n_triples"
        EXCEL = "excel"
    
    class MockImportMode:
        FULL = "full"
        INCREMENTAL = "incremental"
        REPLACE = "replace"
        MERGE = "merge"
    
    class MockEntityType:
        PERSON = "person"
        ORGANIZATION = "organization"
        CONCEPT = "concept"
        EVENT = "event"
        LOCATION = "location"
        DOCUMENT = "document"
        CUSTOM = "custom"
    
    class MockRelationType:
        KNOWS = "knows"
        WORKS_FOR = "works_for"
        PART_OF = "part_of"
        RELATES_TO = "relates_to"
    
    # 将模拟类添加到sys.modules中
    knowledge_graph_mock = Mock()
    knowledge_graph_mock.SPARQLQuery = MockSPARQLQuery
    knowledge_graph_mock.SPARQLResult = MockSPARQLResult
    knowledge_graph_mock.ImportJob = MockImportJob
    knowledge_graph_mock.GraphVersion = MockGraphVersion
    knowledge_graph_mock.QueryType = MockQueryType
    knowledge_graph_mock.ImportFormat = MockImportFormat
    knowledge_graph_mock.ImportMode = MockImportMode
    knowledge_graph_mock.EntityType = MockEntityType
    knowledge_graph_mock.RelationType = MockRelationType
    
    sys.modules['src.ai.knowledge_graph.sparql_engine'] = knowledge_graph_mock
    sys.modules['src.ai.knowledge_graph.data_importer'] = knowledge_graph_mock
    sys.modules['src.ai.knowledge_graph.data_exporter'] = knowledge_graph_mock
    sys.modules['src.ai.knowledge_graph.version_manager'] = knowledge_graph_mock
    sys.modules['src.ai.knowledge_graph.kg_models'] = knowledge_graph_mock
    sys.modules['src.api.v1.knowledge_management'] = knowledge_graph_mock