"""
数据预处理器测试
"""

import pytest
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from src.ai.training_data_management.models import DataRecord
from src.ai.training_data_management.preprocessor import DataPreprocessor

class TestDataPreprocessor:
    """数据预处理器测试"""
    
    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()
    
    @pytest.fixture
    def sample_records(self):
        return [
            DataRecord(
                record_id="rec1",
                source_id="src1",
                raw_data={
                    'text': '  Hello    world!  ',
                    'title': 'Test Title',
                    'created_at': '2024-01-01T00:00:00Z',
                    'author': 'John Doe'
                }
            ),
            DataRecord(
                record_id="rec2", 
                source_id="src1",
                raw_data={
                    'content': 'Short',
                    'author': 'Jane Smith',
                    'category': 'test'
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_text_cleaning(self, preprocessor, sample_records):
        """测试文本清理"""
        
        processed_records = await preprocessor.preprocess_data(
            sample_records, 
            rules=['text_cleaning']
        )
        
        assert len(processed_records) == 2
        
        # 检查文本清理结果
        first_record = processed_records[0]
        assert first_record.processed_data['text'] == 'Hello world!'
        assert first_record.status == 'processed'
        assert first_record.processed_at is not None
    
    @pytest.mark.asyncio 
    async def test_format_standardization(self, preprocessor, sample_records):
        """测试格式标准化"""
        
        processed_records = await preprocessor.preprocess_data(
            sample_records,
            rules=['format_standardization']
        )
        
        # 检查字段映射
        first_record = processed_records[0]
        assert 'content' in first_record.processed_data  # text -> content
        assert 'title' in first_record.processed_data
        assert 'timestamp' in first_record.processed_data  # created_at -> timestamp
        assert 'author' in first_record.processed_data
    
    @pytest.mark.asyncio
    async def test_deduplication(self, preprocessor, sample_records):
        """测试数据去重"""
        
        processed_records = await preprocessor.preprocess_data(
            sample_records,
            rules=['deduplication']
        )
        
        # 检查是否添加了内容哈希
        for record in processed_records:
            assert '_metadata' in record.processed_data
            assert 'content_hash' in record.processed_data['_metadata']
    
    @pytest.mark.asyncio
    async def test_data_enrichment(self, preprocessor, sample_records):
        """测试数据丰富化"""
        
        processed_records = await preprocessor.preprocess_data(
            sample_records,
            rules=['data_enrichment']
        )
        
        # 检查丰富化数据
        first_record = processed_records[0]
        assert '_metadata' in first_record.processed_data
        assert '_analysis' in first_record.processed_data
        assert '_structure' in first_record.processed_data
        
        # 检查分析数据
        analysis = first_record.processed_data['_analysis']
        assert 'character_count' in analysis
        assert 'word_count' in analysis
        assert 'language' in analysis
        assert 'sentiment' in analysis
    
    @pytest.mark.asyncio
    async def test_quality_filtering(self, preprocessor):
        """测试质量过滤"""
        
        # 创建质量不合格的记录
        low_quality_record = DataRecord(
            record_id="rec_low",
            source_id="src1",
            raw_data={'text': 'x'},  # 太短
            metadata={'required_fields': ['text']}
        )
        
        processed_records = await preprocessor.preprocess_data(
            [low_quality_record],
            rules=['quality_filtering'],
            custom_config={'min_content_length': 10}
        )
        
        # 应该有错误记录
        assert len(processed_records) == 1
        assert processed_records[0].status == 'error'
        assert 'error' in processed_records[0].metadata
    
    @pytest.mark.asyncio
    async def test_data_validation(self, preprocessor, sample_records):
        """测试数据验证"""
        
        validation_config = {
            'validation_rules': {
                'title': {
                    'type': 'string',
                    'min_length': 5,
                    'max_length': 100
                },
                'author': {
                    'type': 'string',
                    'pattern': r'^[A-Za-z\s]+$'
                }
            }
        }
        
        processed_records = await preprocessor.preprocess_data(
            sample_records,
            rules=['validation'],
            custom_config=validation_config
        )
        
        # 应该通过验证
        assert all(record.status == 'processed' for record in processed_records)
    
    @pytest.mark.asyncio
    async def test_normalization(self, preprocessor):
        """测试数据标准化"""
        
        record = DataRecord(
            record_id="rec_norm",
            source_id="src1",
            raw_data={
                'price': '$19.99',
                'quantity': '5',
                'is_available': 'true',
                'tags': 'tag1, tag2, tag3'
            }
        )
        
        normalization_config = {
            'numeric_fields': ['price', 'quantity'],
            'boolean_fields': ['is_available'],
            'list_fields': ['tags']
        }
        
        processed_records = await preprocessor.preprocess_data(
            [record],
            rules=['normalization'],
            custom_config=normalization_config
        )
        
        processed_data = processed_records[0].processed_data
        assert processed_data['price'] == 19.99
        assert processed_data['quantity'] == 5.0
        assert processed_data['is_available'] is True
        assert processed_data['tags'] == ['tag1', 'tag2', 'tag3']
    
    def test_quality_score_calculation(self, preprocessor):
        """测试质量分数计算"""
        
        # 高质量数据
        high_quality_data = {
            'title': 'Great Title',
            'content': 'This is a well-structured content with good length and multiple fields providing comprehensive information.',
            'author': 'John Doe',
            'category': 'Technology',
            'tags': ['AI', 'ML']
        }
        
        score = preprocessor._calculate_quality_score(high_quality_data)
        assert 0.7 <= score <= 1.0
        
        # 低质量数据
        low_quality_data = {
            'text': 'short'
        }
        
        score = preprocessor._calculate_quality_score(low_quality_data)
        assert score < 0.5
    
    def test_language_detection(self, preprocessor):
        """测试语言检测"""
        
        # 中文文本
        chinese_text = "这是一段中文文本，用于测试语言检测功能。"
        assert preprocessor._detect_language(chinese_text) == 'zh'
        
        # 英文文本
        english_text = "This is an English text for testing language detection."
        assert preprocessor._detect_language(english_text) == 'en'
        
        # 混合文本（中文占主导）
        mixed_text = "这是中文和English混合的文本。"
        assert preprocessor._detect_language(mixed_text) == 'zh'
    
    def test_sentiment_analysis(self, preprocessor):
        """测试情感分析"""
        
        # 积极情感
        positive_text = "This is a great and excellent product. I love it!"
        assert preprocessor._analyze_sentiment(positive_text) == 'positive'
        
        # 消极情感
        negative_text = "This is terrible and awful. I hate it!"
        assert preprocessor._analyze_sentiment(negative_text) == 'negative'
        
        # 中性情感
        neutral_text = "This is a product with some features."
        assert preprocessor._analyze_sentiment(neutral_text) == 'neutral'
        
        # 中文情感
        positive_chinese = "这个产品很好，我很喜欢。"
        assert preprocessor._analyze_sentiment(positive_chinese) == 'positive'
    
    def test_timestamp_normalization(self, preprocessor):
        """测试时间戳标准化"""
        
        # ISO格式
        iso_time = "2024-01-01T12:00:00Z"
        normalized = preprocessor._normalize_timestamp(iso_time)
        assert normalized == "2024-01-01T12:00:00+00:00"
        
        # Unix时间戳
        unix_time = 1704110400  # 2024-01-01 12:00:00 UTC
        normalized = preprocessor._normalize_timestamp(unix_time)
        assert "2024-01-01T12:00:00" in normalized
        
        # 常见格式
        common_format = "2024-01-01 12:00:00"
        normalized = preprocessor._normalize_timestamp(common_format)
        assert "2024-01-01T12:00:00" in normalized
    
    def test_url_normalization(self, preprocessor):
        """测试URL标准化"""
        
        # 已有协议
        full_url = "https://example.com/path"
        assert preprocessor._normalize_url(full_url) == "https://example.com/path"
        
        # 缺少协议的www链接
        www_url = "www.example.com"
        assert preprocessor._normalize_url(www_url) == "https://www.example.com"
        
        # 缺少协议的普通链接
        simple_url = "example.com"
        assert preprocessor._normalize_url(simple_url) == "https://example.com"
    
    def test_boolean_normalization(self, preprocessor):
        """测试布尔值标准化"""
        
        assert preprocessor._normalize_boolean("true") is True
        assert preprocessor._normalize_boolean("false") is False
        assert preprocessor._normalize_boolean("1") is True
        assert preprocessor._normalize_boolean("0") is False
        assert preprocessor._normalize_boolean("yes") is True
        assert preprocessor._normalize_boolean("no") is False
        assert preprocessor._normalize_boolean(1) is True
        assert preprocessor._normalize_boolean(0) is False
        assert preprocessor._normalize_boolean(True) is True
        assert preprocessor._normalize_boolean(False) is False
    
    def test_nesting_depth_calculation(self, preprocessor):
        """测试嵌套深度计算"""
        
        # 简单对象
        simple_obj = {"key": "value"}
        assert preprocessor._calculate_nesting_depth(simple_obj) == 1
        
        # 嵌套对象
        nested_obj = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        assert preprocessor._calculate_nesting_depth(nested_obj) == 3
        
        # 包含数组的对象
        array_obj = {
            "items": [
                {"name": "item1"},
                {"name": "item2"}
            ]
        }
        assert preprocessor._calculate_nesting_depth(array_obj) == 3
    
    @pytest.mark.asyncio
    async def test_custom_preprocessing_rules(self, preprocessor, sample_records):
        """测试自定义预处理规则"""
        
        # 只应用特定规则
        custom_rules = ['text_cleaning', 'format_standardization']
        
        processed_records = await preprocessor.preprocess_data(
            sample_records,
            rules=custom_rules
        )
        
        # 验证只应用了指定的规则
        first_record = processed_records[0]
        assert first_record.status == 'processed'
        
        # 应该有清理和标准化，但没有丰富化数据
        assert 'content' in first_record.processed_data  # 标准化
        assert '_analysis' not in first_record.processed_data  # 未丰富化
    
    @pytest.mark.asyncio
    async def test_error_handling_in_preprocessing(self, preprocessor):
        """测试预处理错误处理"""
        
        # 创建会导致错误的记录
        error_record = DataRecord(
            record_id="error_rec",
            source_id="src1",
            raw_data={'text': 'valid content'},
            metadata={'required_fields': ['missing_field']}  # 缺少必需字段
        )
        
        processed_records = await preprocessor.preprocess_data(
            [error_record],
            rules=['quality_filtering']
        )
        
        assert len(processed_records) == 1
        assert processed_records[0].status == 'error'
        assert 'error' in processed_records[0].metadata
        assert 'error_timestamp' in processed_records[0].metadata
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, preprocessor):
        """测试并行处理能力"""
        
        # 创建大量记录
        records = []
        for i in range(100):
            record = DataRecord(
                record_id=f"rec_{i}",
                source_id="src1",
                raw_data={
                    'text': f'This is test content number {i}',
                    'id': i,
                    'category': 'test'
                }
            )
            records.append(record)
        
        processed_records = await preprocessor.preprocess_data(
            records,
            rules=['text_cleaning', 'format_standardization', 'data_enrichment']
        )
        
        assert len(processed_records) == 100
        assert all(record.status == 'processed' for record in processed_records)
        assert all(record.quality_score is not None for record in processed_records)
    
    def test_preprocessing_rule_registry(self, preprocessor):
        """测试预处理规则注册"""
        
        # 检查所有规则都已注册
        expected_rules = [
            'text_cleaning',
            'deduplication', 
            'format_standardization',
            'quality_filtering',
            'data_enrichment',
            'normalization',
            'validation'
        ]
        
        for rule in expected_rules:
            assert rule in preprocessor.preprocessing_rules
            assert preprocessor.preprocessing_rules[rule].enabled
            assert callable(preprocessor.preprocessing_rules[rule].function)
