"""
知识图谱测试配置

提供测试夹具和公共配置
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

# 动态知识图谱存储系统不需要这些导入
# from src.ai.knowledge_graph.data_models import Entity, Relation, EntityType, RelationType

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# 这些夹具依赖于其他知识图谱组件，暂时注释
# @pytest.fixture
# def sample_entities():
#     """提供示例实体数据"""
#     return [
#         Entity("张三", EntityType.PERSON, 0, 2, 0.95),
#         Entity("李四", EntityType.PERSON, 5, 7, 0.93),
#         Entity("苹果公司", EntityType.COMPANY, 10, 14, 0.97),
#         Entity("谷歌", EntityType.COMPANY, 17, 19, 0.94),
#         Entity("北京", EntityType.CITY, 22, 24, 0.91),
#         Entity("加州", EntityType.LOCATION, 27, 29, 0.89)
#     ]

# @pytest.fixture
# def sample_relations(sample_entities):
#     """提供示例关系数据"""
#     return [
#         Relation(
#             subject=sample_entities[0],  # 张三
#             predicate=RelationType.WORKS_FOR,
#             object=sample_entities[2],   # 苹果公司
#             confidence=0.88,
#             context="张三在苹果公司工作",
#             source_sentence="张三在苹果公司工作"
#         ),
#         Relation(
#             subject=sample_entities[1],  # 李四
#             predicate=RelationType.WORKS_FOR,
#             object=sample_entities[3],   # 谷歌
#             confidence=0.86,
#             context="李四在谷歌工作",
#             source_sentence="李四在谷歌工作"
#         ),
#         Relation(
#             subject=sample_entities[2],  # 苹果公司
#             predicate=RelationType.LOCATED_IN,
#             object=sample_entities[4],   # 北京
#             confidence=0.85,
#             context="苹果公司位于北京",
#             source_sentence="苹果公司位于北京"
#         ),
#         Relation(
#             subject=sample_entities[3],  # 谷歌
#             predicate=RelationType.LOCATED_IN,
#             object=sample_entities[5],   # 加州
#             confidence=0.83,
#             context="谷歌位于加州",
#             source_sentence="谷歌位于加州"
#         )
#     ]

@pytest.fixture
def sample_chinese_text():
    """提供中文测试文本"""
    return "张三在位于北京的苹果公司工作，他是一名高级软件工程师。李四在谷歌负责人工智能研发。"

@pytest.fixture
def sample_english_text():
    """提供英文测试文本"""
    return "John Smith works for Apple Inc. in California. He is a senior software engineer specializing in machine learning."

@pytest.fixture
def sample_mixed_text():
    """提供中英混合测试文本"""
    return "张三 works at Apple Inc. 他负责 AI research in 北京分公司。"

@pytest.fixture
def mock_entity_recognizer():
    """Mock实体识别器"""
    recognizer = Mock()
    recognizer.extract_entities = AsyncMock()
    recognizer.initialize = AsyncMock()
    recognizer.is_loaded = Mock(return_value=True)
    return recognizer

@pytest.fixture
def mock_relation_extractor():
    """Mock关系抽取器"""
    extractor = Mock()
    extractor.extract_relations = AsyncMock()
    extractor.initialize = AsyncMock()
    extractor.is_loaded = Mock(return_value=True)
    return extractor

@pytest.fixture
def mock_entity_linker():
    """Mock实体链接器"""
    linker = Mock()
    linker.link_entities = AsyncMock()
    linker.initialize = AsyncMock()
    linker.is_loaded = Mock(return_value=True)
    return linker

@pytest.fixture
def mock_multilingual_processor():
    """Mock多语言处理器"""
    processor = Mock()
    processor.process_multilingual_text = AsyncMock()
    processor.initialize = AsyncMock()
    return processor

@pytest.fixture
def sample_documents():
    """提供示例文档数据"""
    return [
        {
            "id": "doc_001",
            "text": "张三在苹果公司工作，负责iOS开发。",
            "metadata": {"source": "hr_system", "department": "engineering"}
        },
        {
            "id": "doc_002", 
            "text": "李四是谷歌的数据科学家，专注于机器学习。",
            "metadata": {"source": "linkedin", "verified": True}
        },
        {
            "id": "doc_003",
            "text": "王五创立了一家AI创业公司，总部位于深圳。",
            "metadata": {"source": "news", "date": "2024-01-15"}
        }
    ]

@pytest.fixture
def sample_api_request():
    """提供示例API请求数据"""
    return {
        "text": "马云创立了阿里巴巴集团，总部位于杭州。",
        "language": "zh",
        "extract_entities": True,
        "extract_relations": True,
        "link_entities": True,
        "confidence_threshold": 0.7,
        "extraction_config": {
            "use_cache": True,
            "max_entities": 50,
            "max_relations": 20
        }
    }

@pytest.fixture
def sample_batch_request():
    """提供示例批处理请求数据"""
    return {
        "documents": [
            {"id": "batch_doc_1", "text": "张三在苹果公司工作。"},
            {"id": "batch_doc_2", "text": "李四在谷歌工作。"},
            {"id": "batch_doc_3", "text": "王五在微软工作。"},
            {"id": "batch_doc_4", "text": "赵六在百度工作。"}
        ],
        "priority": 5,
        "language": "zh",
        "extract_entities": True,
        "extract_relations": True,
        "link_entities": False,
        "confidence_threshold": 0.8,
        "batch_settings": {
            "timeout_seconds": 300,
            "retry_count": 2
        }
    }

# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")

# 知识图谱存储系统测试夹具
@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j驱动器"""
    from unittest.mock import Mock, AsyncMock
    
    driver = Mock()
    session = Mock()
    
    # Mock异步会话
    session.run = AsyncMock()
    session.close = AsyncMock()
    
    # Mock事务
    tx = Mock()
    tx.run = AsyncMock()
    
    session.begin_transaction = Mock(return_value=tx)
    driver.session = Mock(return_value=session)
    driver.close = AsyncMock()
    
    return driver

@pytest.fixture
def sample_graph_entities():
    """提供图数据库测试实体"""
    return [
        {
            "id": "entity_001",
            "canonical_form": "张三",
            "type": "PERSON",
            "properties": {
                "age": 30,
                "occupation": "工程师",
                "confidence": 0.95
            },
            "embedding": [0.1, 0.2, 0.3] * 128
        },
        {
            "id": "entity_002", 
            "canonical_form": "苹果公司",
            "type": "ORGANIZATION",
            "properties": {
                "industry": "科技",
                "founded": 1976,
                "confidence": 0.98
            },
            "embedding": [0.2, 0.3, 0.4] * 128
        }
    ]

@pytest.fixture
def sample_graph_relations():
    """提供图数据库测试关系"""
    return [
        {
            "id": "relation_001",
            "type": "WORKS_FOR",
            "source_id": "entity_001",
            "target_id": "entity_002", 
            "properties": {
                "since": "2020",
                "position": "高级工程师",
                "confidence": 0.90
            }
        }
    ]

@pytest.fixture
async def test_neo4j_config():
    """测试用Neo4j配置"""
    from src.ai.knowledge_graph.graph_database import GraphDatabaseConfig
    
    return GraphDatabaseConfig(
        uri="bolt://localhost:7687",
        username="test",
        password="test",
        database="test",
        connection_timeout=5,
        max_connection_lifetime=30,
        max_connection_pool_size=10,
        connection_acquisition_timeout=5
    )

# 测试跳过条件
def pytest_collection_modifyitems(config, items):
    """根据条件跳过测试"""
    import pytest
    
    # 跳过需要外部依赖的测试
    skip_external = pytest.mark.skip(reason="requires external dependencies")
    skip_neo4j = pytest.mark.skip(reason="requires Neo4j database")
    
    for item in items:
        # 如果测试需要实际的NLP模型，添加跳过标记
        if "real_model" in item.keywords:
            item.add_marker(skip_external)
        # 如果测试需要真实Neo4j连接
        if "neo4j_integration" in item.keywords:
            item.add_marker(skip_neo4j)

# 异步测试支持
@pytest.fixture(scope="function")
async def async_test_client():
    """异步测试客户端"""
    from httpx import AsyncClient, ASGITransport
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

# 临时文件和目录
@pytest.fixture
def temp_cache_dir(tmp_path):
    """临时缓存目录"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def temp_model_dir(tmp_path):
    """临时模型目录"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
