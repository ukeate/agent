"""
版本管理测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.ai.knowledge_graph.version_manager import (
    VersionManager,
    GraphVersion,
    ChangeRecord
)
from src.ai.knowledge_graph.change_tracker import (
    ChangeTracker
)


class TestVersionManager:
    """版本管理器测试类"""

    @pytest.fixture
    def mock_graph_store(self):
        """模拟图数据库存储"""
        store = Mock()
        store.create_snapshot = AsyncMock()
        store.restore_from_snapshot = AsyncMock()
        store.get_current_state = AsyncMock()
        return store

    @pytest.fixture
    def mock_change_tracker(self):
        """模拟变更追踪器"""
        tracker = AsyncMock()
        tracker.record_change = AsyncMock()
        tracker.record_rollback = AsyncMock()
        tracker.get_changes_between_versions = AsyncMock()
        return tracker

    @pytest.fixture
    def version_manager(self, mock_graph_store, mock_change_tracker):
        """版本管理器实例"""
        return VersionManager(mock_graph_store, mock_change_tracker)

    @pytest.mark.asyncio
    async def test_create_version_success(self, version_manager):
        """测试创建版本成功"""
        with patch.object(version_manager, '_create_graph_snapshot') as mock_snapshot:
            mock_snapshot.return_value = {
                'statistics': {'entities': 100, 'relations': 200},
                'checksum': 'abc123',
                'data': {'nodes': [], 'edges': []}
            }
            
            with patch.object(version_manager, '_get_current_version_id') as mock_current:
                mock_current.return_value = 'v1.0.0'
                
                with patch.object(version_manager, '_get_current_user') as mock_user:
                    mock_user.return_value = 'test_user'
                    
                    with patch.object(version_manager, '_store_version') as mock_store:
                        mock_store.return_value = None
                        
                        version = await version_manager.create_version(
                            "Test version creation",
                            {"author": "test_user", "purpose": "testing"}
                        )

                        assert isinstance(version, GraphVersion)
                        assert version.description == "Test version creation"
                        assert version.created_by == "test_user"
                        assert version.parent_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_version_comparison(self, version_manager):
        """测试版本比较"""
        v1_snapshot = {
            'entities': [
                {'id': 'e1', 'type': 'Person', 'name': 'Alice'},
                {'id': 'e2', 'type': 'Concept', 'name': 'AI'}
            ],
            'relations': [
                {'id': 'r1', 'source': 'e1', 'target': 'e2', 'type': 'knows_about'}
            ]
        }
        
        v2_snapshot = {
            'entities': [
                {'id': 'e1', 'type': 'Person', 'name': 'Alice Smith'},  # 修改
                {'id': 'e2', 'type': 'Concept', 'name': 'AI'},
                {'id': 'e3', 'type': 'Organization', 'name': 'TechCorp'}  # 新增
            ],
            'relations': [
                {'id': 'r1', 'source': 'e1', 'target': 'e2', 'type': 'knows_about'},
                {'id': 'r2', 'source': 'e1', 'target': 'e3', 'type': 'works_for'}  # 新增
            ]
        }

        with patch.object(version_manager, '_load_version_snapshot') as mock_load:
            def side_effect(version_id):
                if version_id == 'v1.0':
                    return v1_snapshot
                elif version_id == 'v2.0':
                    return v2_snapshot
                
            mock_load.side_effect = side_effect

            with patch.object(version_manager, '_calculate_differences') as mock_diff:
                mock_diff.return_value = {
                    'added_entities': [{'id': 'e3', 'name': 'TechCorp'}],
                    'removed_entities': [],
                    'modified_entities': [{'id': 'e1', 'old_name': 'Alice', 'new_name': 'Alice Smith'}],
                    'added_relations': [{'id': 'r2', 'type': 'works_for'}],
                    'removed_relations': [],
                    'modified_relations': [],
                    'statistics_diff': {'entities_added': 1, 'relations_added': 1}
                }

                comparison = await version_manager.compare_versions('v1.0', 'v2.0')

                assert comparison['version_1'] == 'v1.0'
                assert comparison['version_2'] == 'v2.0'
                assert len(comparison['added_entities']) == 1
                assert len(comparison['modified_entities']) == 1

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, version_manager):
        """测试版本回滚"""
        target_version = 'v1.5'
        
        with patch.object(version_manager, '_load_version_snapshot') as mock_load:
            mock_load.return_value = {
                'entities': [{'id': 'e1', 'name': 'Alice'}],
                'relations': [],
                'metadata': {'version': target_version}
            }
            
            with patch.object(version_manager, 'create_version') as mock_create:
                backup_version = GraphVersion(
                    version_id='backup_v1',
                    version_number='backup_1.0',
                    parent_version='v2.0',
                    created_at=datetime.now(),
                    created_by='system',
                    description='Backup before rollback',
                    metadata={},
                    statistics={},
                    checksum='backup123'
                )
                mock_create.return_value = backup_version
                
                with patch.object(version_manager, '_restore_from_snapshot') as mock_restore:
                    mock_restore.return_value = None
                    
                    success = await version_manager.rollback_to_version(target_version)

                    assert success is True
                    version_manager.change_tracker.record_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_failure_recovery(self, version_manager):
        """测试回滚失败时的恢复"""
        target_version = 'v1.0'
        
        with patch.object(version_manager, '_load_version_snapshot') as mock_load:
            mock_load.return_value = {'data': 'snapshot'}
            
            with patch.object(version_manager, 'create_version') as mock_create:
                backup_version = Mock(version_id='backup_v1')
                mock_create.return_value = backup_version
                
                with patch.object(version_manager, '_restore_from_snapshot') as mock_restore:
                    # 模拟恢复失败
                    mock_restore.side_effect = Exception("Restore failed")
                    
                    with patch.object(version_manager, 'rollback_to_version') as mock_rollback:
                        mock_rollback.return_value = True
                        
                        with pytest.raises(Exception) as exc_info:
                            await version_manager.rollback_to_version(target_version)
                        
                        assert "Restore failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_version_list_and_history(self, version_manager):
        """测试版本列表和历史"""
        mock_versions = [
            GraphVersion(
                version_id=f'v{i}',
                version_number=f'1.{i}',
                parent_version=f'v{i-1}' if i > 0 else None,
                created_at=datetime.now(),
                created_by='test_user',
                description=f'Version {i}',
                metadata={},
                statistics={'entities': i * 10},
                checksum=f'hash{i}'
            )
            for i in range(5)
        ]

        with patch.object(version_manager, '_list_all_versions') as mock_list:
            mock_list.return_value = mock_versions
            
            versions = await version_manager.list_versions(limit=10, offset=0)
            
            assert len(versions) == 5
            assert all(isinstance(v, GraphVersion) for v in versions)

    @pytest.mark.asyncio
    async def test_version_metadata_update(self, version_manager):
        """测试版本元数据更新"""
        version_id = 'v1.0'
        new_metadata = {
            'tags': ['production', 'stable'],
            'description_updated': 'Updated description',
            'performance_metrics': {'query_time': 0.5}
        }

        with patch.object(version_manager, '_update_version_metadata') as mock_update:
            mock_update.return_value = True
            
            success = await version_manager.update_version_metadata(version_id, new_metadata)
            
            assert success is True
            mock_update.assert_called_once_with(version_id, new_metadata)

    def test_version_number_generation(self, version_manager):
        """测试版本号生成"""
        with patch.object(version_manager, '_get_latest_version_number') as mock_latest:
            mock_latest.return_value = '1.5.2'
            
            with patch.object(version_manager, '_increment_version_number') as mock_increment:
                mock_increment.return_value = '1.5.3'
                
                new_version = version_manager._generate_version_number()
                
                assert new_version == '1.5.3'


class TestChangeTracker:
    """变更追踪器测试类"""

    @pytest.fixture
    def change_tracker(self):
        """变更追踪器实例"""
        return ChangeTracker()

    @pytest.mark.asyncio
    async def test_record_entity_creation(self, change_tracker):
        """测试记录实体创建"""
        change = ChangeRecord(
            change_id='change_001',
            version_id='v1.0',
            operation_type='create',
            target_type='entity',
            target_id='entity_123',
            old_data=None,
            new_data={'name': 'Alice', 'type': 'Person'},
            timestamp=datetime.now(),
            user_id='user_001',
            reason='Initial entity creation'
        )

        with patch.object(change_tracker, '_store_change_record') as mock_store:
            mock_store.return_value = True
            
            success = await change_tracker.record_change(change)
            
            assert success is True
            mock_store.assert_called_once_with(change)

    @pytest.mark.asyncio
    async def test_record_entity_update(self, change_tracker):
        """测试记录实体更新"""
        change = ChangeRecord(
            change_id='change_002',
            version_id='v1.1',
            operation_type='update',
            target_type='entity',
            target_id='entity_123',
            old_data={'name': 'Alice', 'age': 30},
            new_data={'name': 'Alice Smith', 'age': 31},
            timestamp=datetime.now(),
            user_id='user_001',
            reason='Profile update'
        )

        with patch.object(change_tracker, '_store_change_record') as mock_store:
            mock_store.return_value = True
            
            success = await change_tracker.record_change(change)
            
            assert success is True

    @pytest.mark.asyncio
    async def test_record_entity_deletion(self, change_tracker):
        """测试记录实体删除"""
        change = ChangeRecord(
            change_id='change_003',
            version_id='v1.2',
            operation_type='delete',
            target_type='entity',
            target_id='entity_123',
            old_data={'name': 'Alice Smith', 'age': 31},
            new_data=None,
            timestamp=datetime.now(),
            user_id='user_001',
            reason='Data cleanup'
        )

        with patch.object(change_tracker, '_store_change_record') as mock_store:
            mock_store.return_value = True
            
            success = await change_tracker.record_change(change)
            
            assert success is True

    @pytest.mark.asyncio
    async def test_get_change_history(self, change_tracker):
        """测试获取变更历史"""
        mock_changes = [
            ChangeRecord(
                change_id=f'change_{i:03d}',
                version_id='v1.0',
                operation_type=['create', 'update', 'delete'][i % 3],
                target_type='entity',
                target_id=f'entity_{i}',
                old_data={} if i % 3 != 0 else None,
                new_data={} if i % 3 != 2 else None,
                timestamp=datetime.now(),
                user_id='user_001',
                reason=f'Change {i}'
            )
            for i in range(10)
        ]

        with patch.object(change_tracker, '_query_change_history') as mock_query:
            mock_query.return_value = mock_changes[:5]  # 限制返回5个
            
            changes = await change_tracker.get_change_history(
                target_id='entity_123',
                limit=5,
                offset=0
            )
            
            assert len(changes) == 5
            assert all(isinstance(c, ChangeRecord) for c in changes)

    @pytest.mark.asyncio
    async def test_get_changes_between_versions(self, change_tracker):
        """测试获取版本间变更"""
        with patch.object(change_tracker, '_query_changes_between_versions') as mock_query:
            mock_query.return_value = [
                ChangeRecord(
                    change_id='change_001',
                    version_id='v1.1',
                    operation_type='create',
                    target_type='entity',
                    target_id='new_entity',
                    old_data=None,
                    new_data={'name': 'New Entity'},
                    timestamp=datetime.now(),
                    user_id='user_001',
                    reason='New feature'
                )
            ]
            
            changes = await change_tracker.get_changes_between_versions('v1.0', 'v1.1')
            
            assert len(changes) == 1
            assert changes[0].operation_type == 'create'


@pytest.mark.integration
class TestVersionManagementIntegration:
    """版本管理集成测试"""

    @pytest.mark.asyncio
    async def test_full_version_workflow(self):
        """测试完整版本管理工作流"""
        pytest.skip("需要真实的数据库连接")

    @pytest.mark.asyncio
    async def test_concurrent_version_operations(self):
        """测试并发版本操作"""
        pytest.skip("需要并发测试环境")

    @pytest.mark.asyncio
    async def test_version_performance_large_graph(self):
        """测试大型图谱的版本管理性能"""
        pytest.skip("需要大型测试数据集")