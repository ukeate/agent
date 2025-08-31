"""
数据收集器测试
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.training_data_management.models import DataSource
from src.ai.training_data_management.collectors import (
    FileDataCollector, 
    APIDataCollector, 
    WebDataCollector,
    CollectorFactory
)


class TestFileDataCollector:
    """文件数据收集器测试"""
    
    @pytest.fixture
    def file_source(self):
        return DataSource(
            source_id="test-file",
            source_type="file",
            name="Test File",
            description="Test file source",
            config={
                'file_path': 'test_data.json',
                'format': 'json'
            }
        )
    
    @pytest.mark.asyncio
    async def test_collect_json_file(self, file_source):
        """测试JSON文件数据收集"""
        
        test_data = [
            {'id': 1, 'text': 'Hello world'},
            {'id': 2, 'text': 'Test message'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            file_source.config['file_path'] = temp_file
            collector = FileDataCollector(file_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 2
            assert collected_records[0].raw_data['id'] == 1
            assert collected_records[1].raw_data['text'] == 'Test message'
            assert all(record.source_id == file_source.source_id for record in collected_records)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_collect_jsonl_file(self, file_source):
        """测试JSONL文件数据收集"""
        
        test_data = [
            {'id': 1, 'text': 'Line 1'},
            {'id': 2, 'text': 'Line 2'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            file_source.config.update({
                'file_path': temp_file,
                'format': 'jsonl'
            })
            
            collector = FileDataCollector(file_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 2
            assert all(record.source_id == file_source.source_id for record in collected_records)
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_collect_text_file(self, file_source):
        """测试文本文件数据收集"""
        
        test_lines = ["Line 1", "Line 2", "Line 3"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(test_lines))
            temp_file = f.name
        
        try:
            file_source.config.update({
                'file_path': temp_file,
                'format': 'text',
                'split_by': 'line'
            })
            
            collector = FileDataCollector(file_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 3
            assert collected_records[0].raw_data['text'] == 'Line 1'
            assert collected_records[0].raw_data['line_number'] == 1
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_auto_format_detection(self, file_source):
        """测试自动格式检测"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([{"test": "data"}], f)
            temp_file = f.name
        
        try:
            file_source.config.update({
                'file_path': temp_file,
                'format': 'auto'  # 自动检测
            })
            
            collector = FileDataCollector(file_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 1
            assert collected_records[0].raw_data['test'] == 'data'
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_generate_record_id(self, file_source):
        """测试记录ID生成"""
        
        collector = FileDataCollector(file_source)
        
        data1 = {'text': 'Hello world', 'id': 123}
        data2 = {'id': 123, 'text': 'Hello world'}  # 顺序不同
        data3 = {'text': 'Hello world', 'id': 456}  # 内容不同
        
        id1 = collector.generate_record_id(data1)
        id2 = collector.generate_record_id(data2)
        id3 = collector.generate_record_id(data3)
        
        assert id1 == id2  # 相同数据应生成相同ID
        assert id1 != id3  # 不同数据应生成不同ID
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, file_source):
        """测试文件不存在的情况"""
        
        file_source.config['file_path'] = '/nonexistent/file.json'
        collector = FileDataCollector(file_source)
        
        with pytest.raises(FileNotFoundError):
            async for _ in collector.collect_data():
                pass
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, file_source):
        """测试无效JSON的处理"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
            temp_file = f.name
        
        try:
            file_source.config.update({
                'file_path': temp_file,
                'format': 'jsonl'
            })
            
            collector = FileDataCollector(file_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            # 应该只收集有效的JSON行
            assert len(collected_records) == 2
            assert collected_records[0].raw_data['valid'] == 'json'
            assert collected_records[1].raw_data['another'] == 'valid'
            
        finally:
            Path(temp_file).unlink(missing_ok=True)


class TestAPIDataCollector:
    """API数据收集器测试"""
    
    @pytest.fixture
    def api_source(self):
        return DataSource(
            source_id="test-api",
            source_type="api",
            name="Test API",
            description="Test API source",
            config={
                'url': 'https://api.example.com/data',
                'headers': {'Authorization': 'Bearer token'},
                'batch_size': 10,
                'delay': 0  # 测试时不延时
            }
        )
    
    @pytest.mark.asyncio
    async def test_collect_api_data(self, api_source):
        """测试API数据收集"""
        
        # 模拟API响应
        mock_responses = [
            [{'id': 1, 'text': 'Item 1'}, {'id': 2, 'text': 'Item 2'}],
            []  # 空响应表示结束
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_context
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = mock_responses
            
            mock_context.get.return_value.__aenter__.return_value = mock_response
            
            collector = APIDataCollector(api_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 2
            assert collected_records[0].raw_data['id'] == 1
            assert collected_records[1].raw_data['text'] == 'Item 2'
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, api_source):
        """测试API错误处理"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_context
            
            mock_response = AsyncMock()
            mock_response.status = 404
            
            mock_context.get.return_value.__aenter__.return_value = mock_response
            
            collector = APIDataCollector(api_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            # 404错误应该终止收集，返回空列表
            assert len(collected_records) == 0
    
    @pytest.mark.asyncio
    async def test_different_response_formats(self, api_source):
        """测试不同的API响应格式"""
        
        mock_responses = [
            {'data': [{'id': 1, 'text': 'Item 1'}]},  # 包装在data字段中
            {'items': [{'id': 2, 'text': 'Item 2'}]},  # 包装在items字段中
            [{'id': 3, 'text': 'Item 3'}],  # 直接返回数组
            []
        ]
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_context
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = mock_responses
            
            mock_context.get.return_value.__aenter__.return_value = mock_response
            
            collector = APIDataCollector(api_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 3
            assert all(record.source_id == api_source.source_id for record in collected_records)


class TestWebDataCollector:
    """网页数据收集器测试"""
    
    @pytest.fixture
    def web_source(self):
        return DataSource(
            source_id="test-web",
            source_type="web",
            name="Test Web",
            description="Test web source",
            config={
                'urls': ['https://example.com/page1', 'https://example.com/page2'],
                'selectors': {
                    'title': 'h1',
                    'content': 'p'
                },
                'delay': 0  # 测试时不延时
            }
        )
    
    @pytest.mark.asyncio
    async def test_collect_web_data(self, web_source):
        """测试网页数据收集"""
        
        mock_html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>First paragraph</p>
            <p>Second paragraph</p>
        </body>
        </html>
        """
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_context
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = mock_html
            
            mock_context.get.return_value.__aenter__.return_value = mock_response
            
            collector = WebDataCollector(web_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            assert len(collected_records) == 2  # 两个URL
            
            # 检查解析的数据
            record = collected_records[0]
            assert 'url' in record.raw_data
            assert 'title' in record.raw_data
            assert record.raw_data['title'] == 'Test Page'
    
    @pytest.mark.asyncio
    async def test_web_error_handling(self, web_source):
        """测试网页错误处理"""
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_context = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_context
            
            mock_response = AsyncMock()
            mock_response.status = 404
            
            mock_context.get.return_value.__aenter__.return_value = mock_response
            
            collector = WebDataCollector(web_source)
            collected_records = []
            
            async for record in collector.collect_data():
                collected_records.append(record)
            
            # 404错误应该跳过，但不应该中断其他URL的处理
            # 由于都是404，应该返回空列表
            assert len(collected_records) == 0


class TestCollectorFactory:
    """收集器工厂测试"""
    
    def test_create_file_collector(self):
        """测试创建文件收集器"""
        
        source = DataSource(
            source_id="test",
            source_type="file",
            name="Test",
            description="Test",
            config={}
        )
        
        collector = CollectorFactory.create_collector(source)
        assert isinstance(collector, FileDataCollector)
    
    def test_create_api_collector(self):
        """测试创建API收集器"""
        
        source = DataSource(
            source_id="test",
            source_type="api",
            name="Test",
            description="Test",
            config={}
        )
        
        collector = CollectorFactory.create_collector(source)
        assert isinstance(collector, APIDataCollector)
    
    def test_create_web_collector(self):
        """测试创建网页收集器"""
        
        source = DataSource(
            source_id="test",
            source_type="web",
            name="Test",
            description="Test",
            config={}
        )
        
        collector = CollectorFactory.create_collector(source)
        assert isinstance(collector, WebDataCollector)
    
    def test_unsupported_source_type(self):
        """测试不支持的数据源类型"""
        
        source = DataSource(
            source_id="test",
            source_type="unsupported",
            name="Test",
            description="Test",
            config={}
        )
        
        with pytest.raises(ValueError, match="Unsupported source type"):
            CollectorFactory.create_collector(source)
    
    def test_register_custom_collector(self):
        """测试注册自定义收集器"""
        
        class CustomCollector:
            def __init__(self, source):
                self.source = source
        
        # 注册自定义收集器
        CollectorFactory.register_collector("custom", CustomCollector)
        
        source = DataSource(
            source_id="test",
            source_type="custom",
            name="Test",
            description="Test",
            config={}
        )
        
        collector = CollectorFactory.create_collector(source)
        assert isinstance(collector, CustomCollector)