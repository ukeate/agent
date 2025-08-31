"""
数据收集管理器测试
"""

import pytest
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.training_data_management.models import DataSource, DataRecord
from src.ai.training_data_management.manager import DataCollectionManager


class TestDataCollectionManager:
    """数据收集管理器测试"""
    
    @pytest.fixture
    def data_manager(self):
        mock_db_session = AsyncMock()
        return DataCollectionManager(mock_db_session)
    
    @pytest.fixture
    def sample_source(self):
        return DataSource(
            source_id="test-source",
            source_type="file",
            name="Test Source",
            description="Test data source",
            config={
                'file_path': 'test_data.json',
                'format': 'json'
            }
        )
    
    @pytest.fixture
    def sample_records(self):
        return [
            DataRecord(
                record_id="rec1",
                source_id="test-source",
                raw_data={'text': 'Hello world'},
                status='collected'
            ),
            DataRecord(
                record_id="rec2",
                source_id="test-source",
                raw_data={'text': 'Test message'},
                status='collected'
            )
        ]
    
    @pytest.mark.asyncio
    async def test_add_data_source(self, data_manager, sample_source):
        """测试添加数据源"""
        
        with patch.object(data_manager.db, 'add') as mock_add:
            with patch.object(data_manager.db, 'commit') as mock_commit:
                added_source = await data_manager.add_data_source(sample_source)
                
                assert added_source.source_id == "test-source"
                assert added_source.source_type == "file"
                assert added_source.name == "Test Source"
                
                mock_add.assert_called_once_with(sample_source)
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_data_source(self, data_manager, sample_source):
        """测试获取数据源"""
        
        with patch.object(data_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = sample_source
            
            source = await data_manager.get_data_source("test-source")
            
            assert source.source_id == "test-source"
            assert source.name == "Test Source"
    
    @pytest.mark.asyncio
    async def test_list_data_sources(self, data_manager):
        """测试列出数据源"""
        
        mock_sources = [
            MagicMock(source_id="src1", name="Source 1"),
            MagicMock(source_id="src2", name="Source 2")
        ]
        
        with patch.object(data_manager.db, 'query') as mock_query:
            mock_query.return_value.all.return_value = mock_sources
            
            sources = await data_manager.list_data_sources()
            
            assert len(sources) == 2
            assert sources[0].source_id == "src1"
            assert sources[1].source_id == "src2"
    
    @pytest.mark.asyncio
    async def test_collect_data_from_source(self, data_manager, sample_source):
        """测试从数据源收集数据"""
        
        mock_collector = AsyncMock()
        mock_collector.collect_data.return_value = iter([
            DataRecord(record_id="rec1", source_id="test-source", raw_data={'text': 'Test'}),
            DataRecord(record_id="rec2", source_id="test-source", raw_data={'text': 'Data'})
        ])
        
        with patch.object(data_manager, 'get_data_source') as mock_get_source:
            mock_get_source.return_value = sample_source
            
            with patch('src.ai.training_data_management.collectors.CollectorFactory.create_collector') as mock_factory:
                mock_factory.return_value = mock_collector
                
                with patch.object(data_manager, '_save_collected_records') as mock_save:
                    collected_records = await data_manager.collect_data_from_source("test-source")
                    
                    assert len(collected_records) == 2
                    assert collected_records[0].record_id == "rec1"
                    assert collected_records[1].record_id == "rec2"
                    
                    mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_data_batch(self, data_manager):
        """测试批量数据收集"""
        
        source_ids = ["src1", "src2"]
        
        mock_results = [
            [DataRecord(record_id="rec1", source_id="src1", raw_data={'text': 'Data 1'})],
            [DataRecord(record_id="rec2", source_id="src2", raw_data={'text': 'Data 2'})]
        ]
        
        with patch.object(data_manager, 'collect_data_from_source') as mock_collect:
            mock_collect.side_effect = mock_results
            
            batch_results = await data_manager.collect_data_batch(source_ids)
            
            assert len(batch_results) == 2
            assert batch_results["src1"][0].record_id == "rec1"
            assert batch_results["src2"][0].record_id == "rec2"
    
    @pytest.mark.asyncio
    async def test_preprocess_records(self, data_manager, sample_records):
        """测试记录预处理"""
        
        mock_preprocessor = AsyncMock()
        processed_records = [
            DataRecord(
                record_id="rec1",
                source_id="test-source",
                raw_data={'text': 'Hello world'},
                processed_data={'content': 'Hello world', 'word_count': 2},
                status='processed'
            ),
            DataRecord(
                record_id="rec2",
                source_id="test-source", 
                raw_data={'text': 'Test message'},
                processed_data={'content': 'Test message', 'word_count': 2},
                status='processed'
            )
        ]
        
        mock_preprocessor.preprocess_data.return_value = processed_records
        
        with patch('src.ai.training_data_management.preprocessor.DataPreprocessor') as mock_preprocessor_class:
            mock_preprocessor_class.return_value = mock_preprocessor
            
            with patch.object(data_manager, '_update_records') as mock_update:
                result = await data_manager.preprocess_records(
                    sample_records,
                    rules=['text_cleaning', 'format_standardization']
                )
                
                assert len(result) == 2
                assert all(record.status == 'processed' for record in result)
                mock_update.assert_called_once_with(processed_records)
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics(self, data_manager):
        """测试获取收集统计信息"""
        
        # 模拟数据库查询结果
        mock_record_counts = [
            MagicMock(source_id="src1", count=10),
            MagicMock(source_id="src2", count=15)
        ]
        
        mock_status_counts = [
            MagicMock(status="collected", count=20),
            MagicMock(status="processed", count=5)
        ]
        
        mock_quality_stats = MagicMock(
            avg_quality=0.85,
            min_quality=0.6,
            max_quality=1.0
        )
        
        with patch.object(data_manager.db, 'query') as mock_query:
            # 设置多个查询的返回值
            mock_query.return_value.with_entities.return_value.group_by.return_value.all.side_effect = [
                mock_record_counts,
                mock_status_counts
            ]
            mock_query.return_value.with_entities.return_value.first.return_value = mock_quality_stats
            
            stats = await data_manager.get_collection_statistics()
            
            assert 'total_records' in stats
            assert 'records_by_source' in stats
            assert 'records_by_status' in stats
            assert 'quality_statistics' in stats
            
            # 验证统计数据
            assert stats['records_by_source']['src1'] == 10
            assert stats['records_by_source']['src2'] == 15
            assert stats['records_by_status']['collected'] == 20
            assert stats['records_by_status']['processed'] == 5
    
    @pytest.mark.asyncio
    async def test_reprocess_records(self, data_manager):
        """测试重新处理记录"""
        
        record_ids = ["rec1", "rec2"]
        
        mock_records = [
            MagicMock(
                record_id="rec1",
                raw_data={'text': 'Hello'},
                status='processed'
            ),
            MagicMock(
                record_id="rec2",
                raw_data={'text': 'World'},
                status='processed'
            )
        ]
        
        with patch.object(data_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_records
            
            with patch.object(data_manager, 'preprocess_records') as mock_preprocess:
                mock_preprocess.return_value = mock_records
                
                reprocessed = await data_manager.reprocess_records(
                    record_ids=record_ids,
                    rules=['text_cleaning', 'quality_filtering']
                )
                
                assert len(reprocessed) == 2
                mock_preprocess.assert_called_once_with(
                    mock_records,
                    rules=['text_cleaning', 'quality_filtering']
                )
    
    @pytest.mark.asyncio
    async def test_delete_records(self, data_manager):
        """测试删除记录"""
        
        record_ids = ["rec1", "rec2"]
        
        with patch.object(data_manager.db, 'query') as mock_query:
            mock_delete = mock_query.return_value.filter.return_value.delete
            mock_delete.return_value = 2  # 删除了2条记录
            
            with patch.object(data_manager.db, 'commit') as mock_commit:
                deleted_count = await data_manager.delete_records(record_ids)
                
                assert deleted_count == 2
                mock_delete.assert_called_once()
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_records(self, data_manager):
        """测试搜索记录"""
        
        search_params = {
            'text_query': 'hello',
            'source_id': 'test-source',
            'status': 'processed',
            'min_quality_score': 0.8
        }
        
        mock_records = [
            MagicMock(
                record_id="rec1",
                raw_data={'text': 'Hello world'},
                quality_score=0.9
            )
        ]
        
        with patch.object(data_manager.db, 'query') as mock_query:
            # 构建复杂的查询链
            mock_filter_chain = mock_query.return_value
            for _ in range(4):  # 4个过滤条件
                mock_filter_chain = mock_filter_chain.filter.return_value
            mock_filter_chain.limit.return_value.offset.return_value.all.return_value = mock_records
            
            results = await data_manager.search_records(**search_params)
            
            assert len(results) == 1
            assert results[0].record_id == "rec1"
    
    @pytest.mark.asyncio
    async def test_get_record_by_id(self, data_manager):
        """测试根据ID获取记录"""
        
        mock_record = MagicMock(
            record_id="rec1",
            raw_data={'text': 'Hello'},
            status='processed'
        )
        
        with patch.object(data_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = mock_record
            
            record = await data_manager.get_record_by_id("rec1")
            
            assert record.record_id == "rec1"
            assert record.status == 'processed'
    
    @pytest.mark.asyncio
    async def test_update_data_source(self, data_manager, sample_source):
        """测试更新数据源"""
        
        updates = {
            'name': 'Updated Source Name',
            'description': 'Updated description',
            'config': {'new_setting': 'value'}
        }
        
        with patch.object(data_manager, 'get_data_source') as mock_get_source:
            mock_get_source.return_value = sample_source
            
            with patch.object(data_manager.db, 'commit') as mock_commit:
                updated_source = await data_manager.update_data_source(
                    source_id="test-source",
                    **updates
                )
                
                assert updated_source.name == 'Updated Source Name'
                assert updated_source.description == 'Updated description'
                assert updated_source.config['new_setting'] == 'value'
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_data_source(self, data_manager, sample_source):
        """测试删除数据源"""
        
        with patch.object(data_manager, 'get_data_source') as mock_get_source:
            mock_get_source.return_value = sample_source
            
            with patch.object(data_manager.db, 'delete') as mock_delete:
                with patch.object(data_manager.db, 'commit') as mock_commit:
                    success = await data_manager.delete_data_source("test-source")
                    
                    assert success is True
                    mock_delete.assert_called_once_with(sample_source)
                    mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_collected_records(self, data_manager, sample_records):
        """测试保存收集的记录"""
        
        with patch.object(data_manager.db, 'add_all') as mock_add_all:
            with patch.object(data_manager.db, 'commit') as mock_commit:
                await data_manager._save_collected_records(sample_records)
                
                mock_add_all.assert_called_once_with(sample_records)
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_records(self, data_manager, sample_records):
        """测试更新记录"""
        
        with patch.object(data_manager.db, 'merge') as mock_merge:
            with patch.object(data_manager.db, 'commit') as mock_commit:
                await data_manager._update_records(sample_records)
                
                assert mock_merge.call_count == len(sample_records)
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collection_with_error_handling(self, data_manager, sample_source):
        """测试收集过程中的错误处理"""
        
        mock_collector = AsyncMock()
        mock_collector.collect_data.side_effect = Exception("Collection failed")
        
        with patch.object(data_manager, 'get_data_source') as mock_get_source:
            mock_get_source.return_value = sample_source
            
            with patch('src.ai.training_data_management.collectors.CollectorFactory.create_collector') as mock_factory:
                mock_factory.return_value = mock_collector
                
                with pytest.raises(Exception, match="Collection failed"):
                    await data_manager.collect_data_from_source("test-source")
    
    @pytest.mark.asyncio
    async def test_parallel_collection(self, data_manager):
        """测试并行数据收集"""
        
        source_ids = ["src1", "src2", "src3"]
        
        # 模拟不同的收集时间
        async def mock_collect(source_id):
            await asyncio.sleep(0.1)  # 模拟I/O操作
            return [DataRecord(record_id=f"rec_{source_id}", source_id=source_id, raw_data={'id': source_id})]
        
        with patch.object(data_manager, 'collect_data_from_source') as mock_collect_method:
            mock_collect_method.side_effect = mock_collect
            
            start_time = asyncio.get_event_loop().time()
            results = await data_manager.collect_data_batch(source_ids, parallel=True)
            end_time = asyncio.get_event_loop().time()
            
            # 并行执行应该比串行快
            assert end_time - start_time < 0.3  # 应该小于串行执行时间(0.3秒)
            assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_collection_progress_tracking(self, data_manager):
        """测试收集进度跟踪"""
        
        source_ids = ["src1", "src2", "src3"]
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress)
        
        with patch.object(data_manager, 'collect_data_from_source') as mock_collect:
            async def mock_collect_with_progress(source_id):
                return [DataRecord(record_id=f"rec_{source_id}", source_id=source_id, raw_data={})]
            
            mock_collect.side_effect = mock_collect_with_progress
            
            results = await data_manager.collect_data_batch(
                source_ids,
                progress_callback=progress_callback
            )
            
            # 验证进度回调被调用
            assert len(progress_updates) > 0
            assert progress_updates[-1]['completed'] == len(source_ids)
            assert progress_updates[-1]['total'] == len(source_ids)