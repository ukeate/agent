import pytest
import asyncio
import tempfile
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, mock_open

from ....ai.fault_tolerance.backup_manager import (
    BackupManager,
    BackupType,
    BackupRecord
)

@pytest.fixture
def temp_backup_dir():
    """创建临时备份目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def backup_config(temp_backup_dir):
    return {
        "backup_interval": 1,  # 1秒用于快速测试
        "retention_days": 1,
        "backup_location": temp_backup_dir,
        "auto_backup_components": ["agent-1", "agent-2"]
    }

@pytest.fixture
def backup_manager(backup_config):
    storage_backend = Mock()
    return BackupManager(
        storage_backend=storage_backend,
        config=backup_config
    )

@pytest.fixture
def sample_component_data():
    return {
        "component_id": "agent-1",
        "timestamp": datetime.now().isoformat(),
        "state": {"status": "active", "memory": "test_data"},
        "configuration": {"type": "chat_agent", "params": {"max_tokens": 1000}},
        "tasks": [{"task_id": "task-1", "status": "running"}],
        "metrics": {"cpu": 45.0, "memory": 60.0}
    }

@pytest.mark.asyncio
class TestBackupManager:
    
    async def test_backup_manager_initialization(self, backup_manager, temp_backup_dir):
        """测试备份管理器初始化"""
        assert backup_manager.backup_interval == 1
        assert backup_manager.retention_days == 1
        assert backup_manager.backup_location == temp_backup_dir
        assert backup_manager.running is False
        assert len(backup_manager.backup_records) == 0
    
    async def test_start_and_stop(self, backup_manager):
        """测试启动和停止功能"""
        assert not backup_manager.running
        
        # 启动
        await backup_manager.start()
        assert backup_manager.running
        
        # 等待一小段时间让循环开始
        await asyncio.sleep(0.1)
        
        # 停止
        await backup_manager.stop()
        assert not backup_manager.running
    
    async def test_collect_component_data(self, backup_manager):
        """测试收集组件数据"""
        with patch.object(backup_manager, '_get_component_state') as mock_state, \
             patch.object(backup_manager, '_get_component_config') as mock_config, \
             patch.object(backup_manager, '_get_component_tasks') as mock_tasks, \
             patch.object(backup_manager, '_get_component_metrics') as mock_metrics:
            
            mock_state.return_value = {"status": "active"}
            mock_config.return_value = {"type": "agent"}
            mock_tasks.return_value = [{"task_id": "task-1"}]
            mock_metrics.return_value = {"cpu": 50.0}
            
            data = await backup_manager._collect_component_data("agent-1")
            
            assert data is not None
            assert data["component_id"] == "agent-1"
            assert "timestamp" in data
            assert data["state"]["status"] == "active"
            assert data["configuration"]["type"] == "agent"
            assert len(data["tasks"]) == 1
            assert data["metrics"]["cpu"] == 50.0
    
    async def test_create_backup_success(self, backup_manager, sample_component_data):
        """测试成功创建备份"""
        with patch.object(backup_manager, '_collect_component_data', return_value=sample_component_data), \
             patch.object(backup_manager, '_store_backup_data') as mock_store, \
             patch.object(backup_manager, '_verify_backup', return_value=True) as mock_verify:
            
            backup_record = await backup_manager.create_backup("agent-1", BackupType.FULL_BACKUP)
            
            assert backup_record is not None
            assert backup_record.component_id == "agent-1"
            assert backup_record.backup_type == BackupType.FULL_BACKUP
            assert backup_record.valid is True
            assert backup_record.size > 0
            assert len(backup_record.checksum) == 64  # SHA256 hash length
            
            # 验证存储被调用
            mock_store.assert_called_once()
            mock_verify.assert_called_once()
            
            # 验证备份记录被添加
            assert len(backup_manager.backup_records) == 1
            assert backup_manager.backup_records[0] == backup_record
    
    async def test_create_backup_no_data(self, backup_manager):
        """测试无数据时的备份创建"""
        with patch.object(backup_manager, '_collect_component_data', return_value=None):
            backup_record = await backup_manager.create_backup("agent-1")
            
            assert backup_record is None
            assert len(backup_manager.backup_records) == 0
    
    async def test_create_backup_failure(self, backup_manager, sample_component_data):
        """测试备份创建失败"""
        with patch.object(backup_manager, '_collect_component_data', return_value=sample_component_data), \
             patch.object(backup_manager, '_store_backup_data', side_effect=Exception("Storage failed")):
            
            backup_record = await backup_manager.create_backup("agent-1")
            
            assert backup_record is None
            assert len(backup_manager.backup_records) == 0
    
    async def test_store_and_load_backup_data(self, backup_manager, temp_backup_dir):
        """测试存储和加载备份数据"""
        test_data = b"test backup data"
        file_path = os.path.join(temp_backup_dir, "test_backup.backup")
        
        # 测试存储
        await backup_manager._store_backup_data(file_path, test_data)
        
        # 验证文件被创建
        assert os.path.exists(file_path)
        
        # 测试加载
        loaded_data = await backup_manager._load_backup_data(file_path)
        assert loaded_data == test_data
    
    async def test_verify_backup(self, backup_manager, temp_backup_dir):
        """测试验证备份"""
        test_data = b"test backup data for verification"
        checksum = hashlib.sha256(test_data).hexdigest()
        file_path = os.path.join(temp_backup_dir, "verify_test.backup")
        
        # 创建备份记录
        backup_record = BackupRecord(
            backup_id="test_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=len(test_data),
            checksum=checksum,
            metadata={},
            storage_path=file_path
        )
        
        # 写入数据
        with open(file_path, 'wb') as f:
            f.write(test_data)
        
        # 验证备份
        is_valid = await backup_manager._verify_backup(backup_record)
        assert is_valid is True
        
        # 测试校验和不匹配
        backup_record.checksum = "invalid_checksum"
        is_valid = await backup_manager._verify_backup(backup_record)
        assert is_valid is False
        
        # 测试文件不存在
        os.remove(file_path)
        is_valid = await backup_manager._verify_backup(backup_record)
        assert is_valid is False
    
    async def test_verify_backup_integrity(self, backup_manager):
        """测试备份数据完整性验证"""
        test_data = b"test data for integrity check"
        expected_checksum = hashlib.sha256(test_data).hexdigest()
        
        # 测试正确的校验和
        is_valid = await backup_manager._verify_backup_integrity(test_data, expected_checksum)
        assert is_valid is True
        
        # 测试错误的校验和
        is_valid = await backup_manager._verify_backup_integrity(test_data, "wrong_checksum")
        assert is_valid is False
    
    async def test_restore_backup_success(self, backup_manager, sample_component_data, temp_backup_dir):
        """测试成功恢复备份"""
        # 创建备份
        serialized_data = pickle.dumps(sample_component_data)
        checksum = hashlib.sha256(serialized_data).hexdigest()
        file_path = os.path.join(temp_backup_dir, "restore_test.backup")
        
        backup_record = BackupRecord(
            backup_id="restore_test",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=len(serialized_data),
            checksum=checksum,
            metadata={},
            storage_path=file_path,
            valid=True
        )
        
        # 写入序列化数据
        with open(file_path, 'wb') as f:
            f.write(serialized_data)
        
        backup_manager.backup_records.append(backup_record)
        
        with patch.object(backup_manager, '_restore_component_data', return_value=True) as mock_restore:
            success = await backup_manager.restore_backup("restore_test")
            
            assert success is True
            mock_restore.assert_called_once_with("agent-1", sample_component_data)
    
    async def test_restore_backup_not_found(self, backup_manager):
        """测试恢复不存在的备份"""
        success = await backup_manager.restore_backup("non_existent_backup")
        assert success is False
    
    async def test_restore_backup_invalid(self, backup_manager, temp_backup_dir):
        """测试恢复无效备份"""
        backup_record = BackupRecord(
            backup_id="invalid_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=100,
            checksum="invalid",
            metadata={},
            storage_path=os.path.join(temp_backup_dir, "invalid.backup"),
            valid=False  # 标记为无效
        )
        
        backup_manager.backup_records.append(backup_record)
        
        success = await backup_manager.restore_backup("invalid_backup")
        assert success is False
    
    async def test_restore_backup_integrity_failure(self, backup_manager, temp_backup_dir):
        """测试备份完整性检查失败"""
        test_data = b"corrupted data"
        file_path = os.path.join(temp_backup_dir, "corrupted.backup")
        
        backup_record = BackupRecord(
            backup_id="corrupted_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=len(test_data),
            checksum="wrong_checksum",  # 错误的校验和
            metadata={},
            storage_path=file_path,
            valid=True
        )
        
        # 写入数据
        with open(file_path, 'wb') as f:
            f.write(test_data)
        
        backup_manager.backup_records.append(backup_record)
        
        success = await backup_manager.restore_backup("corrupted_backup")
        assert success is False
    
    async def test_restore_backup_with_target(self, backup_manager, sample_component_data, temp_backup_dir):
        """测试恢复备份到指定目标组件"""
        serialized_data = pickle.dumps(sample_component_data)
        checksum = hashlib.sha256(serialized_data).hexdigest()
        file_path = os.path.join(temp_backup_dir, "target_test.backup")
        
        backup_record = BackupRecord(
            backup_id="target_test",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=len(serialized_data),
            checksum=checksum,
            metadata={},
            storage_path=file_path,
            valid=True
        )
        
        with open(file_path, 'wb') as f:
            f.write(serialized_data)
        
        backup_manager.backup_records.append(backup_record)
        
        with patch.object(backup_manager, '_restore_component_data', return_value=True) as mock_restore:
            success = await backup_manager.restore_backup("target_test", "agent-2")
            
            assert success is True
            # 验证恢复到目标组件
            mock_restore.assert_called_once_with("agent-2", sample_component_data)
    
    async def test_restore_component_data(self, backup_manager, sample_component_data):
        """测试恢复组件数据"""
        with patch.object(backup_manager, '_restore_component_state') as mock_state, \
             patch.object(backup_manager, '_restore_component_config') as mock_config, \
             patch.object(backup_manager, '_restore_component_tasks') as mock_tasks:
            
            success = await backup_manager._restore_component_data("agent-1", sample_component_data)
            
            assert success is True
            mock_state.assert_called_once_with("agent-1", sample_component_data["state"])
            mock_config.assert_called_once_with("agent-1", sample_component_data["configuration"])
            mock_tasks.assert_called_once_with("agent-1", sample_component_data["tasks"])
    
    async def test_cleanup_expired_backups(self, backup_manager, temp_backup_dir):
        """测试清理过期备份"""
        # 创建过期备份记录
        old_time = datetime.now() - timedelta(days=2)  # 2天前
        recent_time = datetime.now() - timedelta(hours=1)  # 1小时前
        
        # 过期备份
        expired_backup = BackupRecord(
            backup_id="expired_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=old_time,
            size=100,
            checksum="checksum1",
            metadata={},
            storage_path=os.path.join(temp_backup_dir, "expired.backup")
        )
        
        # 有效备份
        valid_backup = BackupRecord(
            backup_id="valid_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=recent_time,
            size=100,
            checksum="checksum2",
            metadata={},
            storage_path=os.path.join(temp_backup_dir, "valid.backup")
        )
        
        # 创建实际文件
        with open(expired_backup.storage_path, 'wb') as f:
            f.write(b"expired data")
        with open(valid_backup.storage_path, 'wb') as f:
            f.write(b"valid data")
        
        backup_manager.backup_records = [expired_backup, valid_backup]
        
        await backup_manager._cleanup_expired_backups()
        
        # 验证过期备份被删除
        assert len(backup_manager.backup_records) == 1
        assert backup_manager.backup_records[0].backup_id == "valid_backup"
        
        # 验证文件被删除
        assert not os.path.exists(expired_backup.storage_path)
        assert os.path.exists(valid_backup.storage_path)
    
    async def test_get_backup_statistics_empty(self, backup_manager):
        """测试获取空的备份统计信息"""
        stats = backup_manager.get_backup_statistics()
        
        assert stats["total_backups"] == 0
        assert stats["valid_backups"] == 0
        assert stats["total_size"] == 0
        assert stats["backup_types"] == {}
        assert stats["components"] == {}
    
    async def test_get_backup_statistics_with_data(self, backup_manager):
        """测试获取有数据的备份统计信息"""
        now = datetime.now()
        
        # 添加备份记录
        backup1 = BackupRecord(
            backup_id="backup1",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=now,
            size=1000,
            checksum="checksum1",
            metadata={},
            storage_path="/path/backup1.backup",
            valid=True
        )
        
        backup2 = BackupRecord(
            backup_id="backup2",
            backup_type=BackupType.INCREMENTAL_BACKUP,
            component_id="agent-1",
            created_at=now - timedelta(hours=1),
            size=500,
            checksum="checksum2",
            metadata={},
            storage_path="/path/backup2.backup",
            valid=True
        )
        
        backup3 = BackupRecord(
            backup_id="backup3",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-2",
            created_at=now - timedelta(hours=2),
            size=800,
            checksum="checksum3",
            metadata={},
            storage_path="/path/backup3.backup",
            valid=False  # 无效备份
        )
        
        backup_manager.backup_records = [backup1, backup2, backup3]
        
        stats = backup_manager.get_backup_statistics()
        
        assert stats["total_backups"] == 3
        assert stats["valid_backups"] == 2
        assert stats["total_size"] == 2300  # 1000 + 500 + 800
        
        # 备份类型统计
        assert stats["backup_types"]["full_backup"] == 2
        assert stats["backup_types"]["incremental_backup"] == 1
        
        # 组件统计
        assert stats["components"]["agent-1"]["count"] == 2
        assert stats["components"]["agent-1"]["total_size"] == 1500
        assert stats["components"]["agent-2"]["count"] == 1
        assert stats["components"]["agent-2"]["total_size"] == 800
    
    async def test_get_backup_records(self, backup_manager):
        """测试获取备份记录"""
        backup1 = BackupRecord(
            backup_id="backup1",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=1000,
            checksum="checksum1",
            metadata={},
            storage_path="/path/backup1.backup"
        )
        
        backup2 = BackupRecord(
            backup_id="backup2",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-2",
            created_at=datetime.now(),
            size=500,
            checksum="checksum2",
            metadata={},
            storage_path="/path/backup2.backup"
        )
        
        backup_manager.backup_records = [backup1, backup2]
        
        # 获取所有记录
        all_records = backup_manager.get_backup_records()
        assert len(all_records) == 2
        
        # 按组件过滤
        agent1_records = backup_manager.get_backup_records("agent-1")
        assert len(agent1_records) == 1
        assert agent1_records[0].component_id == "agent-1"
    
    async def test_validate_all_backups(self, backup_manager, temp_backup_dir):
        """测试验证所有备份"""
        # 创建有效备份
        valid_data = b"valid backup data"
        valid_checksum = hashlib.sha256(valid_data).hexdigest()
        valid_path = os.path.join(temp_backup_dir, "valid.backup")
        
        valid_backup = BackupRecord(
            backup_id="valid_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-1",
            created_at=datetime.now(),
            size=len(valid_data),
            checksum=valid_checksum,
            metadata={},
            storage_path=valid_path,
            valid=True
        )
        
        with open(valid_path, 'wb') as f:
            f.write(valid_data)
        
        # 创建无效备份
        invalid_backup = BackupRecord(
            backup_id="invalid_backup",
            backup_type=BackupType.FULL_BACKUP,
            component_id="agent-2",
            created_at=datetime.now(),
            size=100,
            checksum="wrong_checksum",
            metadata={},
            storage_path=os.path.join(temp_backup_dir, "nonexistent.backup"),
            valid=True
        )
        
        backup_manager.backup_records = [valid_backup, invalid_backup]
        
        validation_results = await backup_manager.validate_all_backups()
        
        assert validation_results["valid_backup"] is True
        assert validation_results["invalid_backup"] is False
        
        # 验证备份记录的valid字段被更新
        assert valid_backup.valid is True
        assert invalid_backup.valid is False
    
    async def test_auto_backup_loop(self, backup_manager):
        """测试自动备份循环"""
        with patch.object(backup_manager, '_perform_scheduled_backups') as mock_scheduled:
            await backup_manager.start()
            
            # 等待自动备份循环执行
            await asyncio.sleep(1.5)  # 等待超过backup_interval
            
            await backup_manager.stop()
            
            # 验证定时备份被调用
            mock_scheduled.assert_called()
    
    async def test_perform_scheduled_backups(self, backup_manager):
        """测试执行定时备份"""
        with patch.object(backup_manager, 'create_backup') as mock_create:
            mock_create.return_value = Mock()  # 模拟备份记录
            
            await backup_manager._perform_scheduled_backups()
            
            # 验证对配置中的组件执行备份
            assert mock_create.call_count == 2  # agent-1和agent-2
            mock_create.assert_any_call("agent-1", BackupType.INCREMENTAL_BACKUP)
            mock_create.assert_any_call("agent-2", BackupType.INCREMENTAL_BACKUP)