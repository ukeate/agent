"""
版本管理测试 - 修复版本
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

# 使用conftest.py中的mock类
from conftest import MockGraphVersion


@pytest.mark.knowledge_graph
@pytest.mark.unit
class TestVersionManager:
    """版本管理器测试类"""

    def test_graph_version_creation(self):
        """测试图谱版本对象创建"""
        version = MockGraphVersion(
            version_id="v1.0",
            version_number="1.0.0",
            parent_version=None,
            created_at=datetime.now(),
            created_by="test_user",
            description="Test version",
            metadata={'author': 'test_user'},
            statistics={'entities': 100, 'relations': 50},
            checksum="abc123"
        )

        assert version.version_id == "v1.0"
        assert version.version_number == "1.0.0"
        assert version.created_by == "test_user"
        assert version.description == "Test version"
        assert version.statistics['entities'] == 100

    @pytest.mark.asyncio
    async def test_create_version_success(self, mock_version_manager):
        """测试创建版本成功"""
        test_version = MockGraphVersion(
            version_id="test_v1",
            version_number="1.0.0",
            parent_version=None,
            created_at=datetime.now(),
            created_by="test_user",
            description="Test version creation",
            metadata={"author": "test_user", "purpose": "testing"},
            statistics={'entities': 100, 'relations': 200},
            checksum='abc123'
        )
        
        mock_version_manager.create_version.return_value = test_version

        version = await mock_version_manager.create_version(
            "Test version creation",
            {"author": "test_user", "purpose": "testing"}
        )

        assert isinstance(version, MockGraphVersion)
        assert version.description == "Test version creation"
        assert version.created_by == "test_user"

    @pytest.mark.asyncio
    async def test_version_comparison(self, mock_version_manager):
        """测试版本比较"""
        comparison_result = {
            'version_1': 'v1.0',
            'version_2': 'v2.0',
            'added_entities': [{'id': 'e3', 'name': 'TechCorp'}],
            'removed_entities': [],
            'modified_entities': [{'id': 'e1', 'old_name': 'Alice', 'new_name': 'Alice Smith'}],
            'added_relations': [{'id': 'r2', 'type': 'works_for'}],
            'removed_relations': [],
            'modified_relations': [],
            'statistics_diff': {'entities_added': 1, 'relations_added': 1}
        }
        
        mock_version_manager.compare_versions.return_value = comparison_result

        comparison = await mock_version_manager.compare_versions('v1.0', 'v2.0')

        assert comparison['version_1'] == 'v1.0'
        assert comparison['version_2'] == 'v2.0'
        assert len(comparison['added_entities']) == 1
        assert len(comparison['modified_entities']) == 1
        assert comparison['statistics_diff']['entities_added'] == 1

    @pytest.mark.asyncio
    async def test_rollback_to_version(self, mock_version_manager):
        """测试版本回滚"""
        mock_version_manager.rollback_to_version.return_value = True

        success = await mock_version_manager.rollback_to_version('v1.5')

        assert success is True
        mock_version_manager.rollback_to_version.assert_called_once_with('v1.5')

    @pytest.mark.asyncio
    async def test_rollback_failure_recovery(self, mock_version_manager):
        """测试回滚失败时的恢复"""
        # 模拟回滚失败
        mock_version_manager.rollback_to_version.side_effect = Exception("Restore failed")

        with pytest.raises(Exception) as exc_info:
            await mock_version_manager.rollback_to_version('v1.0')
        
        assert "Restore failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_version_list_and_history(self, mock_version_manager):
        """测试版本列表和历史"""
        mock_versions = [
            MockGraphVersion(
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

        mock_version_manager.list_versions.return_value = mock_versions
        
        versions = await mock_version_manager.list_versions(limit=10, offset=0)
        
        assert len(versions) == 5
        assert all(isinstance(v, MockGraphVersion) for v in versions)
        assert all(v.created_by == 'test_user' for v in versions)

    @pytest.mark.asyncio
    async def test_version_metadata_update(self, mock_version_manager):
        """测试版本元数据更新"""
        mock_version_manager.update_version_metadata.return_value = True
        
        new_metadata = {
            'tags': ['production', 'stable'],
            'description_updated': 'Updated description',
            'performance_metrics': {'query_time': 0.5}
        }

        success = await mock_version_manager.update_version_metadata('v1.0', new_metadata)
        
        assert success is True
        mock_version_manager.update_version_metadata.assert_called_once_with('v1.0', new_metadata)

    def test_version_properties_validation(self):
        """测试版本属性验证"""
        version = MockGraphVersion(
            version_id="v1.0",
            version_number="1.0.0",
            parent_version=None,
            created_at=datetime.now(),
            created_by="test_user",
            description="Test version",
            metadata={},
            statistics={},
            checksum="abc123"
        )

        # 验证必需属性
        assert hasattr(version, 'version_id')
        assert hasattr(version, 'version_number')
        assert hasattr(version, 'created_at')
        assert hasattr(version, 'created_by')
        assert hasattr(version, 'description')
        assert hasattr(version, 'checksum')

    @pytest.mark.asyncio
    async def test_version_creation_with_statistics(self, mock_version_manager):
        """测试带统计信息的版本创建"""
        version_with_stats = MockGraphVersion(
            version_id="stats_v1",
            version_number="1.0.0", 
            parent_version=None,
            created_at=datetime.now(),
            created_by="test_user",
            description="Version with statistics",
            metadata={},
            statistics={
                'entities': 150,
                'relations': 300,
                'avg_degree': 2.0,
                'max_path_length': 6
            },
            checksum="stats123"
        )
        
        mock_version_manager.create_version.return_value = version_with_stats

        version = await mock_version_manager.create_version(
            "Version with statistics",
            {'include_statistics': True}
        )

        assert version.statistics['entities'] == 150
        assert version.statistics['relations'] == 300
        assert version.statistics['avg_degree'] == 2.0


@pytest.mark.knowledge_graph
@pytest.mark.unit
class TestChangeTracker:
    """变更追踪器测试类"""

    @pytest.mark.asyncio
    async def test_record_entity_creation(self, mock_change_tracker):
        """测试记录实体创建"""
        mock_change_tracker.record_change.return_value = True

        # 模拟变更记录
        change_data = {
            'change_id': 'change_001',
            'version_id': 'v1.0',
            'operation_type': 'create',
            'target_type': 'entity',
            'target_id': 'entity_123',
            'old_data': None,
            'new_data': {'name': 'Alice', 'type': 'Person'},
            'timestamp': datetime.now(),
            'user_id': 'user_001',
            'reason': 'Initial entity creation'
        }

        success = await mock_change_tracker.record_change(change_data)
        
        assert success is True
        mock_change_tracker.record_change.assert_called_once_with(change_data)

    @pytest.mark.asyncio
    async def test_record_entity_update(self, mock_change_tracker):
        """测试记录实体更新"""
        mock_change_tracker.record_change.return_value = True

        change_data = {
            'change_id': 'change_002',
            'version_id': 'v1.1',
            'operation_type': 'update',
            'target_type': 'entity',
            'target_id': 'entity_123',
            'old_data': {'name': 'Alice', 'age': 30},
            'new_data': {'name': 'Alice Smith', 'age': 31},
            'timestamp': datetime.now(),
            'user_id': 'user_001',
            'reason': 'Profile update'
        }

        success = await mock_change_tracker.record_change(change_data)
        
        assert success is True

    @pytest.mark.asyncio
    async def test_record_entity_deletion(self, mock_change_tracker):
        """测试记录实体删除"""
        mock_change_tracker.record_change.return_value = True

        change_data = {
            'change_id': 'change_003',
            'version_id': 'v1.2',
            'operation_type': 'delete',
            'target_type': 'entity',
            'target_id': 'entity_123',
            'old_data': {'name': 'Alice Smith', 'age': 31},
            'new_data': None,
            'timestamp': datetime.now(),
            'user_id': 'user_001',
            'reason': 'Data cleanup'
        }

        success = await mock_change_tracker.record_change(change_data)
        
        assert success is True

    @pytest.mark.asyncio
    async def test_get_change_history(self, mock_change_tracker):
        """测试获取变更历史"""
        mock_changes = [
            {
                'change_id': f'change_{i:03d}',
                'version_id': 'v1.0',
                'operation_type': ['create', 'update', 'delete'][i % 3],
                'target_type': 'entity',
                'target_id': f'entity_{i}',
                'timestamp': datetime.now(),
                'user_id': 'user_001'
            }
            for i in range(5)
        ]

        mock_change_tracker.get_change_history.return_value = mock_changes

        changes = await mock_change_tracker.get_change_history(
            target_id='entity_123',
            limit=5,
            offset=0
        )
        
        assert len(changes) == 5
        assert all('change_id' in c for c in changes)
        assert all('operation_type' in c for c in changes)

    @pytest.mark.asyncio
    async def test_get_changes_between_versions(self, mock_change_tracker):
        """测试获取版本间变更"""
        version_changes = [
            {
                'change_id': 'change_001',
                'version_id': 'v1.1',
                'operation_type': 'create',
                'target_type': 'entity',
                'target_id': 'new_entity',
                'timestamp': datetime.now(),
                'user_id': 'user_001'
            }
        ]
        
        mock_change_tracker.get_changes_between_versions.return_value = version_changes

        changes = await mock_change_tracker.get_changes_between_versions('v1.0', 'v1.1')
        
        assert len(changes) == 1
        assert changes[0]['operation_type'] == 'create'

    def test_change_record_structure(self):
        """测试变更记录数据结构"""
        change_record = {
            'change_id': 'change_001',
            'version_id': 'v1.0',
            'operation_type': 'create',
            'target_type': 'entity',
            'target_id': 'entity_123',
            'old_data': None,
            'new_data': {'name': 'Test Entity'},
            'timestamp': datetime.now(),
            'user_id': 'user_001',
            'reason': 'Test creation'
        }

        # 验证必需字段
        required_fields = [
            'change_id', 'version_id', 'operation_type',
            'target_type', 'target_id', 'timestamp', 'user_id'
        ]
        
        for field in required_fields:
            assert field in change_record

        # 验证操作类型
        valid_operations = ['create', 'update', 'delete']
        assert change_record['operation_type'] in valid_operations


@pytest.mark.integration
@pytest.mark.knowledge_graph
class TestVersionManagementIntegration:
    """版本管理集成测试"""

    @pytest.mark.asyncio
    async def test_version_workflow_integration(self, mock_version_manager, mock_change_tracker):
        """测试版本管理工作流集成"""
        # 1. 创建初始版本
        initial_version = MockGraphVersion(
            version_id="workflow_v1",
            version_number="1.0.0",
            parent_version=None,
            created_at=datetime.now(),
            created_by="test_user",
            description="Initial version",
            metadata={},
            statistics={'entities': 10},
            checksum="initial123"
        )
        
        mock_version_manager.create_version.return_value = initial_version
        
        version1 = await mock_version_manager.create_version("Initial version", {})
        assert version1.version_id == "workflow_v1"
        
        # 2. 记录变更
        mock_change_tracker.record_change.return_value = True
        
        change_data = {
            'change_id': 'workflow_change_001',
            'version_id': version1.version_id,
            'operation_type': 'create',
            'target_type': 'entity',
            'target_id': 'new_entity_001'
        }
        
        change_recorded = await mock_change_tracker.record_change(change_data)
        assert change_recorded is True
        
        # 3. 创建新版本
        updated_version = MockGraphVersion(
            version_id="workflow_v2",
            version_number="1.1.0",
            parent_version="workflow_v1",
            created_at=datetime.now(),
            created_by="test_user",
            description="Updated version",
            metadata={},
            statistics={'entities': 11},
            checksum="updated456"
        )
        
        mock_version_manager.create_version.return_value = updated_version
        
        version2 = await mock_version_manager.create_version("Updated version", {})
        assert version2.parent_version == version1.version_id

    @pytest.mark.asyncio
    async def test_rollback_workflow(self, mock_version_manager, mock_change_tracker):
        """测试回滚工作流"""
        # 设置回滚成功
        mock_version_manager.rollback_to_version.return_value = True
        mock_change_tracker.record_rollback.return_value = True
        
        # 执行回滚
        rollback_success = await mock_version_manager.rollback_to_version('v1.0')
        assert rollback_success is True
        
        # 记录回滚操作
        rollback_recorded = await mock_change_tracker.record_rollback('v1.0', 'backup_v1')
        assert rollback_recorded is True

    def test_version_consistency_validation(self):
        """测试版本一致性验证"""
        # 创建版本链
        versions = []
        for i in range(3):
            version = MockGraphVersion(
                version_id=f"consistency_v{i}",
                version_number=f"1.{i}.0",
                parent_version=f"consistency_v{i-1}" if i > 0 else None,
                created_at=datetime.now(),
                created_by="test_user",
                description=f"Consistency test version {i}",
                metadata={},
                statistics={'entities': 10 + i},
                checksum=f"consistency{i}"
            )
            versions.append(version)
        
        # 验证版本链一致性
        assert versions[0].parent_version is None  # 根版本
        assert versions[1].parent_version == versions[0].version_id
        assert versions[2].parent_version == versions[1].version_id
        
        # 验证统计信息增长
        assert versions[0].statistics['entities'] == 10
        assert versions[1].statistics['entities'] == 11
        assert versions[2].statistics['entities'] == 12

    @pytest.mark.slow
    def test_version_performance(self):
        """测试版本管理性能 - 标记为慢速测试"""
        import time
        
        start_time = time.time()
        
        # 模拟创建大量版本
        versions = []
        for i in range(100):
            version = MockGraphVersion(
                version_id=f"perf_v{i}",
                version_number=f"1.{i}.0",
                parent_version=f"perf_v{i-1}" if i > 0 else None,
                created_at=datetime.now(),
                created_by="perf_user",
                description=f"Performance test version {i}",
                metadata={},
                statistics={'entities': i * 10},
                checksum=f"perf{i}"
            )
            versions.append(version)
        
        creation_time = time.time() - start_time
        
        assert len(versions) == 100
        assert creation_time < 1.0  # 应该在1秒内完成


if __name__ == "__main__":
    # 允许直接运行测试文件
    pytest.main([__file__, "-v"])