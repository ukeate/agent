"""
数据版本管理器测试
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.training_data_management.models import DataRecord, DataVersion
from src.ai.training_data_management.version_manager import DataVersionManager


class TestDataVersionManager:
    """数据版本管理器测试"""
    
    @pytest.fixture
    def version_manager(self):
        mock_db_session = AsyncMock()
        return DataVersionManager(mock_db_session)
    
    @pytest.fixture
    def sample_records(self):
        return [
            DataRecord(
                record_id="rec1",
                source_id="src1",
                raw_data={'text': 'Hello world'},
                processed_data={'content': 'Hello world', 'word_count': 2},
                status='processed',
                quality_score=0.9
            ),
            DataRecord(
                record_id="rec2",
                source_id="src1",
                raw_data={'text': 'Test message'},
                processed_data={'content': 'Test message', 'word_count': 2},
                status='processed',
                quality_score=0.8
            )
        ]
    
    @pytest.fixture
    def sample_version(self):
        return DataVersion(
            version_id="v1.0.0",
            name="Initial Version",
            description="Initial data version",
            record_ids=["rec1", "rec2"],
            metadata={
                'total_records': 2,
                'avg_quality_score': 0.85,
                'creation_date': '2024-01-01'
            }
        )
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_manager, sample_records):
        """测试创建数据版本"""
        
        version_config = {
            'name': 'Test Version',
            'description': 'Test data version',
            'version_id': 'v1.0.0'
        }
        
        with patch.object(version_manager.db, 'add') as mock_add:
            with patch.object(version_manager.db, 'commit') as mock_commit:
                with patch.object(version_manager, '_calculate_version_statistics') as mock_stats:
                    mock_stats.return_value = {
                        'total_records': 2,
                        'avg_quality_score': 0.85,
                        'record_types': ['text'],
                        'sources': ['src1']
                    }
                    
                    version = await version_manager.create_version(
                        records=sample_records,
                        **version_config
                    )
                    
                    assert version.version_id == 'v1.0.0'
                    assert version.name == 'Test Version'
                    assert len(version.record_ids) == 2
                    assert version.metadata['total_records'] == 2
                    
                    mock_add.assert_called_once()
                    mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_version(self, version_manager, sample_version):
        """测试获取版本信息"""
        
        with patch.object(version_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = sample_version
            
            version = await version_manager.get_version("v1.0.0")
            
            assert version.version_id == "v1.0.0"
            assert version.name == "Initial Version"
            assert len(version.record_ids) == 2
    
    @pytest.mark.asyncio
    async def test_list_versions(self, version_manager):
        """测试列出所有版本"""
        
        mock_versions = [
            MagicMock(
                version_id="v1.0.0",
                name="Version 1",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
            ),
            MagicMock(
                version_id="v1.1.0",
                name="Version 1.1",
                created_at=datetime(2024, 1, 15, tzinfo=timezone.utc)
            )
        ]
        
        with patch.object(version_manager.db, 'query') as mock_query:
            mock_query.return_value.order_by.return_value.all.return_value = mock_versions
            
            versions = await version_manager.list_versions()
            
            assert len(versions) == 2
            assert versions[0].version_id == "v1.0.0"
            assert versions[1].version_id == "v1.1.0"
    
    @pytest.mark.asyncio
    async def test_compare_versions(self, version_manager):
        """测试版本比较"""
        
        version1 = DataVersion(
            version_id="v1.0.0",
            record_ids=["rec1", "rec2", "rec3"],
            metadata={'total_records': 3, 'avg_quality_score': 0.8}
        )
        
        version2 = DataVersion(
            version_id="v1.1.0",
            record_ids=["rec2", "rec3", "rec4", "rec5"],
            metadata={'total_records': 4, 'avg_quality_score': 0.85}
        )
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.side_effect = [version1, version2]
            
            comparison = await version_manager.compare_versions("v1.0.0", "v1.1.0")
            
            assert comparison['version_1'] == "v1.0.0"
            assert comparison['version_2'] == "v1.1.0"
            assert comparison['added_records'] == ["rec4", "rec5"]
            assert comparison['removed_records'] == ["rec1"]
            assert comparison['common_records'] == ["rec2", "rec3"]
            assert comparison['metadata_changes']['total_records'] == {'v1.0.0': 3, 'v1.1.0': 4}
    
    @pytest.mark.asyncio
    async def test_merge_versions(self, version_manager):
        """测试版本合并"""
        
        source_versions = ["v1.0.0", "v1.1.0"]
        
        version1_records = ["rec1", "rec2"]
        version2_records = ["rec3", "rec4"]
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.side_effect = [
                MagicMock(record_ids=version1_records),
                MagicMock(record_ids=version2_records)
            ]
            
            with patch.object(version_manager, 'create_version') as mock_create_version:
                mock_create_version.return_value = MagicMock(version_id="v2.0.0")
                
                with patch.object(version_manager.db, 'query') as mock_query:
                    mock_query.return_value.filter.return_value.all.return_value = []  # 模拟记录查询
                    
                    merged_version = await version_manager.merge_versions(
                        source_versions=source_versions,
                        target_version_id="v2.0.0",
                        name="Merged Version",
                        description="Merged from v1.0.0 and v1.1.0"
                    )
                    
                    assert merged_version.version_id == "v2.0.0"
                    
                    # 验证create_version被调用时包含了所有记录
                    call_args = mock_create_version.call_args
                    assert 'records' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_rollback_to_version(self, version_manager, sample_version):
        """测试回滚到指定版本"""
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'query') as mock_query:
                mock_records = [
                    MagicMock(record_id="rec1"),
                    MagicMock(record_id="rec2")
                ]
                mock_query.return_value.filter.return_value.all.return_value = mock_records
                
                with patch.object(version_manager, 'create_version') as mock_create_version:
                    mock_create_version.return_value = MagicMock(version_id="v1.0.0-rollback")
                    
                    rollback_version = await version_manager.rollback_to_version(
                        target_version_id="v1.0.0"
                    )
                    
                    assert rollback_version.version_id == "v1.0.0-rollback"
    
    @pytest.mark.asyncio
    async def test_export_version_json(self, version_manager, sample_version):
        """测试导出版本为JSON格式"""
        
        mock_records = [
            MagicMock(
                record_id="rec1",
                raw_data={'text': 'Hello'},
                processed_data={'content': 'Hello', 'word_count': 1},
                quality_score=0.9
            ),
            MagicMock(
                record_id="rec2",
                raw_data={'text': 'World'},
                processed_data={'content': 'World', 'word_count': 1},
                quality_score=0.8
            )
        ]
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'query') as mock_query:
                mock_query.return_value.filter.return_value.all.return_value = mock_records
                
                export_data = await version_manager.export_version(
                    version_id="v1.0.0",
                    format="json"
                )
                
                # 验证导出的数据是有效的JSON
                parsed_data = json.loads(export_data)
                assert 'version_info' in parsed_data
                assert 'records' in parsed_data
                assert len(parsed_data['records']) == 2
                assert parsed_data['version_info']['version_id'] == "v1.0.0"
    
    @pytest.mark.asyncio
    async def test_export_version_csv(self, version_manager, sample_version):
        """测试导出版本为CSV格式"""
        
        mock_records = [
            MagicMock(
                record_id="rec1",
                raw_data={'text': 'Hello'},
                processed_data={'content': 'Hello', 'word_count': 1},
                quality_score=0.9,
                status='processed'
            )
        ]
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'query') as mock_query:
                mock_query.return_value.filter.return_value.all.return_value = mock_records
                
                export_data = await version_manager.export_version(
                    version_id="v1.0.0",
                    format="csv"
                )
                
                # 验证CSV格式
                lines = export_data.strip().split('\n')
                assert len(lines) >= 2  # 至少包含头部和一行数据
                assert 'record_id' in lines[0]  # 检查CSV头部
    
    @pytest.mark.asyncio
    async def test_export_version_to_file(self, version_manager, sample_version):
        """测试导出版本到文件"""
        
        mock_records = [
            MagicMock(
                record_id="rec1",
                raw_data={'text': 'Hello'},
                processed_data={'content': 'Hello'},
                quality_score=0.9
            )
        ]
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'query') as mock_query:
                mock_query.return_value.filter.return_value.all.return_value = mock_records
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / "export.json"
                    
                    await version_manager.export_version_to_file(
                        version_id="v1.0.0",
                        file_path=str(file_path),
                        format="json"
                    )
                    
                    # 验证文件已创建并包含数据
                    assert file_path.exists()
                    
                    with open(file_path, 'r') as f:
                        content = f.read()
                        parsed_data = json.loads(content)
                        assert 'version_info' in parsed_data
                        assert 'records' in parsed_data
    
    def test_calculate_version_statistics(self, version_manager, sample_records):
        """测试计算版本统计信息"""
        
        stats = version_manager._calculate_version_statistics(sample_records)
        
        assert stats['total_records'] == 2
        assert stats['avg_quality_score'] == 0.85  # (0.9 + 0.8) / 2
        assert stats['min_quality_score'] == 0.8
        assert stats['max_quality_score'] == 0.9
        assert 'src1' in stats['sources']
        assert stats['status_distribution']['processed'] == 2
        assert stats['total_size'] > 0  # 应该计算数据大小
    
    def test_generate_version_id(self, version_manager):
        """测试生成版本ID"""
        
        # 测试语义化版本生成
        version_id = version_manager._generate_version_id()
        assert version_id.count('.') == 2  # 应该是 major.minor.patch 格式
        
        # 测试自定义前缀
        version_id_with_prefix = version_manager._generate_version_id(prefix="data")
        assert version_id_with_prefix.startswith("data-")
    
    @pytest.mark.asyncio
    async def test_delete_version(self, version_manager, sample_version):
        """测试删除版本"""
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'delete') as mock_delete:
                with patch.object(version_manager.db, 'commit') as mock_commit:
                    success = await version_manager.delete_version("v1.0.0")
                    
                    assert success is True
                    mock_delete.assert_called_once_with(sample_version)
                    mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_version_not_found(self, version_manager):
        """测试版本不存在的情况"""
        
        with patch.object(version_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = None
            
            version = await version_manager.get_version("nonexistent")
            assert version is None
    
    @pytest.mark.asyncio
    async def test_update_version_metadata(self, version_manager, sample_version):
        """测试更新版本元数据"""
        
        new_metadata = {
            'updated_field': 'new_value',
            'description': 'Updated description'
        }
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.return_value = sample_version
            
            with patch.object(version_manager.db, 'commit') as mock_commit:
                updated_version = await version_manager.update_version_metadata(
                    version_id="v1.0.0",
                    metadata=new_metadata
                )
                
                # 验证元数据已更新
                assert 'updated_field' in updated_version.metadata
                assert updated_version.metadata['updated_field'] == 'new_value'
                mock_commit.assert_called_once()
    
    def test_calculate_data_size(self, version_manager):
        """测试计算数据大小"""
        
        test_data = {
            'text': 'Hello world',
            'metadata': {'key': 'value'},
            'numbers': [1, 2, 3]
        }
        
        size = version_manager._calculate_data_size(test_data)
        assert size > 0
        assert isinstance(size, int)
    
    @pytest.mark.asyncio
    async def test_get_version_diff(self, version_manager):
        """测试获取版本差异"""
        
        version1 = DataVersion(
            version_id="v1.0.0",
            record_ids=["rec1", "rec2"],
            metadata={'total_records': 2, 'feature_A': True}
        )
        
        version2 = DataVersion(
            version_id="v1.1.0", 
            record_ids=["rec2", "rec3"],
            metadata={'total_records': 2, 'feature_A': False, 'feature_B': True}
        )
        
        with patch.object(version_manager, 'get_version') as mock_get_version:
            mock_get_version.side_effect = [version1, version2]
            
            diff = await version_manager.get_version_diff("v1.0.0", "v1.1.0")
            
            assert diff['records']['added'] == ["rec3"]
            assert diff['records']['removed'] == ["rec1"]
            assert diff['metadata']['changed']['feature_A'] == {'old': True, 'new': False}
            assert diff['metadata']['added']['feature_B'] == True