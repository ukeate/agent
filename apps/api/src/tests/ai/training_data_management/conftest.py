"""
训练数据管理系统测试配置和共享fixtures
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import tempfile
import os
from src.ai.training_data_management.models import (
    DataSource, DataRecord, AnnotationTask, Annotation, DataVersion,
    AnnotationStatus, AnnotationTaskStatus
)

@pytest.fixture(scope="session")
def event_loop():
    """创建一个会话级别的事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_db_session():
    """Mock数据库会话"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.add_all = MagicMock()
    mock_session.merge = MagicMock()
    mock_session.delete = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    return mock_session

@pytest.fixture
def sample_data_source():
    """示例数据源"""
    return DataSource(
        source_id="test-source",
        source_type="file",
        name="Test Data Source",
        description="A test data source for unit testing",
        config={
            'file_path': '/test/data.json',
            'format': 'json'
        },
        created_at=utc_now()
    )

@pytest.fixture
def sample_data_records():
    """示例数据记录列表"""
    return [
        DataRecord(
            record_id="rec1",
            source_id="test-source",
            raw_data={'text': 'Hello world', 'title': 'Test 1'},
            processed_data={'content': 'Hello world', 'word_count': 2},
            metadata={'source': 'test'},
            quality_score=0.9,
            status='processed',
            created_at=utc_now()
        ),
        DataRecord(
            record_id="rec2",
            source_id="test-source",
            raw_data={'text': 'Test message', 'title': 'Test 2'},
            processed_data={'content': 'Test message', 'word_count': 2},
            metadata={'source': 'test'},
            quality_score=0.8,
            status='processed',
            created_at=utc_now()
        )
    ]

@pytest.fixture
def sample_annotation_task():
    """示例标注任务"""
    return AnnotationTask(
        task_id="task1",
        name="Test Classification Task",
        description="A test classification task",
        task_type="text_classification",
        record_ids=["rec1", "rec2"],
        schema={
            'type': 'object',
            'properties': {
                'label': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}
            },
            'required': ['label']
        },
        annotators=['user1', 'user2'],
        created_by="test_user",
        guidelines="Test guidelines",
        status=AnnotationTaskStatus.ACTIVE,
        created_at=utc_now()
    )

@pytest.fixture
def sample_annotations():
    """示例标注结果列表"""
    return [
        Annotation(
            annotation_id="ann1",
            task_id="task1",
            record_id="rec1",
            annotator_id="user1",
            annotation_data={'label': 'positive', 'confidence': 0.9},
            status=AnnotationStatus.SUBMITTED,
            created_at=utc_now()
        ),
        Annotation(
            annotation_id="ann2",
            task_id="task1",
            record_id="rec1",
            annotator_id="user2",
            annotation_data={'label': 'positive', 'confidence': 0.8},
            status=AnnotationStatus.SUBMITTED,
            created_at=utc_now()
        )
    ]

@pytest.fixture
def sample_data_version():
    """示例数据版本"""
    return DataVersion(
        version_id="v1.0.0",
        dataset_name="test-dataset",
        version_number="1.0.0",
        description="Initial test version",
        created_by="test_user",
        metadata={'total_records': 2, 'avg_quality_score': 0.85},
        created_at=utc_now()
    )

@pytest.fixture
def temp_file():
    """临时文件fixture"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_directory():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """自动Mock外部依赖"""
    with patch('transformers.pipeline') as mock_pipeline, \
         patch('torch.cuda.is_available', return_value=False), \
         patch('langchain_community.llms.LlamaCpp') as mock_llm, \
         patch('requests.get') as mock_requests, \
         patch('aiohttp.ClientSession') as mock_session:
        
        # Mock transformers pipeline
        mock_sentiment = MagicMock()
        mock_sentiment.return_value = [{'label': 'POSITIVE', 'score': 0.9}]
        mock_pipeline.return_value = mock_sentiment
        
        # Mock HTTP requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ok'}
        mock_requests.return_value = mock_response
        
        # Mock aiohttp session
        mock_context = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_context
        mock_http_response = AsyncMock()
        mock_http_response.status = 200
        mock_http_response.json.return_value = {'data': []}
        mock_context.get.return_value.__aenter__.return_value = mock_http_response
        
        yield {
            'pipeline': mock_pipeline,
            'llm': mock_llm,
            'requests': mock_requests,
            'session': mock_session
        }

@pytest.fixture
def mock_preprocessing_dependencies():
    """Mock预处理相关的重型依赖"""
    with patch('spacy.load') as mock_spacy, \
         patch('stanza.Pipeline') as mock_stanza, \
         patch('nltk.download') as mock_nltk_download, \
         patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_tfidf:
        
        # Mock spaCy
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.lang_ = 'en'
        mock_nlp.return_value = mock_doc
        mock_spacy.return_value = mock_nlp
        
        # Mock stanza
        mock_stanza_pipeline = MagicMock()
        mock_stanza.return_value = mock_stanza_pipeline
        
        # Mock scikit-learn TfidfVectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = MagicMock()
        mock_tfidf.return_value = mock_vectorizer
        
        yield {
            'spacy': mock_spacy,
            'stanza': mock_stanza,
            'nltk_download': mock_nltk_download,
            'tfidf': mock_tfidf
        }

# 测试工具函数
def assert_record_equality(record1: DataRecord, record2: DataRecord):
    """比较两个DataRecord是否相等的辅助函数"""
    assert record1.record_id == record2.record_id
    assert record1.source_id == record2.source_id
    assert record1.raw_data == record2.raw_data
    assert record1.status == record2.status

def create_mock_query_result(items: list):
    """创建模拟数据库查询结果"""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.all.return_value = items
    mock_query.first.return_value = items[0] if items else None
    mock_query.count.return_value = len(items)
    return mock_query

@pytest.fixture
def mock_language_detection():
    """Mock语言检测"""
    with patch('langdetect.detect') as mock_detect:
        mock_detect.return_value = 'en'
        yield mock_detect

@pytest.fixture
def disable_network():
    """禁用网络请求，确保测试隔离"""
    with patch('socket.socket'), \
         patch('urllib3.poolmanager.PoolManager'), \
         patch('requests.Session.request'), \
         patch('httpx.AsyncClient'):
        yield
