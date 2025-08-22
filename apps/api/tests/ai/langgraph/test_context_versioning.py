"""
上下文版本管理测试
测试版本迁移、兼容性检查和版本管理功能
"""
import pytest
from typing import Dict, Any, List
from datetime import datetime
import copy

from src.ai.langgraph.context import ContextVersion
from src.ai.langgraph.versioning import (
    ContextMigrator,
    VersionCompatibilityChecker,
    ContextVersionManager
)


class TestContextMigrator:
    """测试上下文迁移器"""
    
    def test_find_migration_path(self):
        """测试查找迁移路径"""
        # 正向迁移路径
        path = ContextMigrator._find_migration_path("1.0", "1.2")
        assert path == ["1.1", "1.2"]
        
        path = ContextMigrator._find_migration_path("1.0", "1.1")
        assert path == ["1.1"]
        
        path = ContextMigrator._find_migration_path("1.1", "1.2")
        assert path == ["1.2"]
        
        # 相同版本
        path = ContextMigrator._find_migration_path("1.1", "1.1")
        assert path == []
        
        # 反向迁移（不支持）
        path = ContextMigrator._find_migration_path("1.2", "1.0")
        assert path == []
        
        # 无效版本
        path = ContextMigrator._find_migration_path("1.0", "2.0")
        assert path == []
    
    def test_is_compatible(self):
        """测试版本兼容性检查"""
        # 相同版本兼容
        assert ContextMigrator.is_compatible("1.1", "1.1")
        
        # 可以升级的版本兼容
        assert ContextMigrator.is_compatible("1.0", "1.1")
        assert ContextMigrator.is_compatible("1.0", "1.2")
        assert ContextMigrator.is_compatible("1.1", "1.2")
        
        # 不能降级
        assert not ContextMigrator.is_compatible("1.2", "1.0")
        assert not ContextMigrator.is_compatible("1.2", "1.1")
        
        # 无效版本不兼容
        assert not ContextMigrator.is_compatible("1.0", "3.0")
    
    def test_migrate_v1_0_to_v1_1(self):
        """测试从1.0迁移到1.1"""
        v1_0_data = {
            "user_id": "test_user",
            "session_id": "test_session",
            "version": "1.0",
            "status": "running"
        }
        
        migrated = ContextMigrator.migrate_v1_0_to_v1_1(copy.deepcopy(v1_0_data))
        
        # 验证新增字段
        assert "user_preferences" in migrated
        assert migrated["user_preferences"]["language"] == "zh-CN"
        assert migrated["user_preferences"]["timezone"] == "Asia/Shanghai"
        assert migrated["user_preferences"]["theme"] == "light"
        
        assert "session_context" in migrated
        assert migrated["session_context"]["session_id"] == "test_session"
        assert migrated["session_context"]["message_count"] == 0
        
        assert "performance_tags" in migrated
        assert migrated["performance_tags"] == []
        
        # 验证原有字段保留
        assert migrated["user_id"] == "test_user"
        assert migrated["status"] == "running"
    
    def test_migrate_v1_1_to_v1_2(self):
        """测试从1.1迁移到1.2"""
        v1_1_data = {
            "user_id": "test_user",
            "session_id": "test_session",
            "version": "1.1",
            "workflow_id": "old_workflow_id",
            "user_preferences": {
                "language": "en-US",
                "timezone": "UTC",
                "theme": "dark"
            },
            "session_context": {
                "session_id": "test_session",
                "message_count": 5
            },
            "performance_tags": ["fast", "cached"]
        }
        
        migrated = ContextMigrator.migrate_v1_1_to_v1_2(copy.deepcopy(v1_1_data))
        
        # 验证workflow_metadata迁移
        assert "workflow_metadata" in migrated
        assert migrated["workflow_metadata"]["workflow_id"] == "old_workflow_id"
        assert migrated["workflow_metadata"]["workflow_version"] == "1.0"
        assert migrated["workflow_metadata"]["execution_path"] == []
        assert migrated["workflow_metadata"]["checkpoints"] == []
        
        # 验证旧workflow_id字段被移除
        assert "workflow_id" not in migrated
        
        # 验证user_preferences扩展
        assert migrated["user_preferences"]["notification_enabled"] is True
        assert migrated["user_preferences"]["custom_settings"] == {}
        
        # 验证session_context扩展
        assert migrated["session_context"]["interaction_mode"] == "chat"
        
        # 验证原有数据保留
        assert migrated["user_preferences"]["language"] == "en-US"
        assert migrated["session_context"]["message_count"] == 5
        assert migrated["performance_tags"] == ["fast", "cached"]
    
    def test_migrate_context_full_path(self):
        """测试完整迁移路径"""
        v1_0_data = {
            "user_id": "migrate_user",
            "session_id": "migrate_session",
            "version": "1.0",
            "status": "running"
        }
        
        # 从1.0迁移到1.2
        migrated = ContextMigrator.migrate_context(
            copy.deepcopy(v1_0_data), 
            "1.0", 
            "1.2"
        )
        
        # 验证最终版本
        assert migrated["version"] == "1.2"
        
        # 验证所有新字段都存在
        assert "user_preferences" in migrated
        assert "session_context" in migrated
        assert "workflow_metadata" in migrated
        assert "performance_tags" in migrated
        
        # 验证1.2特有字段
        assert migrated["user_preferences"]["notification_enabled"] is True
        assert migrated["user_preferences"]["custom_settings"] == {}
        assert migrated["session_context"]["interaction_mode"] == "chat"
    
    def test_migrate_same_version(self):
        """测试相同版本迁移（应该返回原数据）"""
        data = {
            "user_id": "same_user",
            "session_id": "same_session",
            "version": "1.1"
        }
        
        migrated = ContextMigrator.migrate_context(
            copy.deepcopy(data), 
            "1.1", 
            "1.1"
        )
        
        assert migrated == data
    
    def test_migrate_invalid_path(self):
        """测试无效迁移路径"""
        data = {"version": "1.2"}
        
        with pytest.raises(ValueError) as exc_info:
            ContextMigrator.migrate_context(data, "1.2", "1.0")
        assert "无法找到从版本" in str(exc_info.value)


class TestVersionCompatibilityChecker:
    """测试版本兼容性检查器"""
    
    def test_check_compatibility(self):
        """测试功能兼容性检查"""
        # 1.0版本功能
        compat = VersionCompatibilityChecker.check_compatibility(
            "1.0",
            {
                "basic_context": "",
                "user_preferences": "",
                "workflow_metadata": ""
            }
        )
        assert compat["basic_context"] is True
        assert compat["user_preferences"] is False
        assert compat["workflow_metadata"] is False
        
        # 1.1版本功能
        compat = VersionCompatibilityChecker.check_compatibility(
            "1.1",
            {
                "basic_context": "",
                "user_preferences": "",
                "session_context": "",
                "workflow_metadata": "",
                "performance_tags": ""
            }
        )
        assert compat["basic_context"] is True
        assert compat["user_preferences"] is True
        assert compat["session_context"] is True
        assert compat["workflow_metadata"] is False
        assert compat["performance_tags"] is True
        
        # 1.2版本功能
        compat = VersionCompatibilityChecker.check_compatibility(
            "1.2",
            {
                "basic_context": "",
                "user_preferences": "",
                "session_context": "",
                "workflow_metadata": "",
                "performance_tags": "",
                "generic_support": ""
            }
        )
        # 1.2版本支持所有功能
        for feature, is_compatible in compat.items():
            assert is_compatible is True
    
    def test_check_unknown_version(self):
        """测试未知版本的兼容性检查"""
        compat = VersionCompatibilityChecker.check_compatibility(
            "3.0",  # 不存在的版本
            {"basic_context": "", "user_preferences": ""}
        )
        
        # 未知版本应该返回所有功能不兼容
        assert compat["basic_context"] is False
        assert compat["user_preferences"] is False
    
    def test_get_minimum_version(self):
        """测试获取最低版本"""
        # 只需要基础功能
        min_version = VersionCompatibilityChecker.get_minimum_version(
            ["basic_context"]
        )
        assert min_version == "1.0"
        
        # 需要1.1版本功能
        min_version = VersionCompatibilityChecker.get_minimum_version(
            ["basic_context", "user_preferences", "session_context"]
        )
        assert min_version == "1.1"
        
        # 需要1.2版本功能
        min_version = VersionCompatibilityChecker.get_minimum_version(
            ["workflow_metadata", "generic_support"]
        )
        assert min_version == "1.2"
        
        # 不存在的功能返回最新版本
        min_version = VersionCompatibilityChecker.get_minimum_version(
            ["unknown_feature"]
        )
        assert min_version == "1.2"


class TestContextVersionManager:
    """测试上下文版本管理器"""
    
    def test_upgrade_context(self):
        """测试上下文升级"""
        manager = ContextVersionManager()
        
        # 1.0升级到最新
        v1_0_data = {
            "user_id": "upgrade_user",
            "session_id": "upgrade_session",
            "version": "1.0"
        }
        
        upgraded = manager.upgrade_context(copy.deepcopy(v1_0_data))
        assert upgraded["version"] == ContextVersion.CURRENT.value
        assert "user_preferences" in upgraded
        assert "workflow_metadata" in upgraded
        
        # 1.1升级到指定版本
        v1_1_data = {
            "user_id": "upgrade_user",
            "session_id": "upgrade_session",
            "version": "1.1",
            "user_preferences": {"language": "zh-CN"}
        }
        
        upgraded = manager.upgrade_context(
            copy.deepcopy(v1_1_data), 
            target_version="1.2"
        )
        assert upgraded["version"] == "1.2"
        assert "workflow_metadata" in upgraded
    
    def test_upgrade_already_latest(self):
        """测试已是最新版本的升级"""
        manager = ContextVersionManager()
        
        latest_data = {
            "user_id": "latest_user",
            "session_id": "latest_session",
            "version": ContextVersion.CURRENT.value
        }
        
        upgraded = manager.upgrade_context(copy.deepcopy(latest_data))
        assert upgraded == latest_data
    
    def test_downgrade_context(self):
        """测试上下文降级"""
        manager = ContextVersionManager()
        
        # 1.2降级到1.0
        v1_2_data = {
            "user_id": "downgrade_user",
            "session_id": "downgrade_session",
            "version": "1.2",
            "user_preferences": {
                "language": "en-US",
                "notification_enabled": True,
                "custom_settings": {"key": "value"}
            },
            "session_context": {
                "session_id": "downgrade_session",
                "interaction_mode": "chat"
            },
            "workflow_metadata": {
                "workflow_id": "wf_123",
                "workflow_version": "2.0"
            },
            "performance_tags": ["tag1", "tag2"],
            "custom_data": {"test": "data"}
        }
        
        downgraded = manager.downgrade_context(
            copy.deepcopy(v1_2_data), 
            "1.0"
        )
        
        assert downgraded["version"] == "1.0"
        # 1.0版本不应该有这些字段
        assert "user_preferences" not in downgraded
        assert "session_context" not in downgraded
        assert "workflow_metadata" not in downgraded
        assert "performance_tags" not in downgraded
        assert "custom_data" not in downgraded
        # 但应该恢复workflow_id
        assert downgraded["workflow_id"] == "wf_123"
    
    def test_downgrade_to_v1_1(self):
        """测试降级到1.1版本"""
        manager = ContextVersionManager()
        
        v1_2_data = {
            "user_id": "downgrade_user",
            "session_id": "downgrade_session",
            "version": "1.2",
            "user_preferences": {
                "language": "en-US",
                "theme": "dark",
                "notification_enabled": True,
                "custom_settings": {"key": "value"}
            },
            "session_context": {
                "session_id": "downgrade_session",
                "message_count": 10,
                "interaction_mode": "chat"
            },
            "workflow_metadata": {
                "workflow_id": "wf_456"
            },
            "custom_data": {"test": "data"}
        }
        
        downgraded = manager.downgrade_context(
            copy.deepcopy(v1_2_data), 
            "1.1"
        )
        
        assert downgraded["version"] == "1.1"
        # 1.1版本有user_preferences但没有某些字段
        assert "user_preferences" in downgraded
        assert "notification_enabled" not in downgraded["user_preferences"]
        assert "custom_settings" not in downgraded["user_preferences"]
        # 1.1版本有session_context但没有interaction_mode
        assert "session_context" in downgraded
        assert "interaction_mode" not in downgraded["session_context"]
        # 1.1版本没有workflow_metadata和custom_data
        assert "workflow_metadata" not in downgraded
        assert "custom_data" not in downgraded
    
    def test_validate_version_requirements(self):
        """测试版本需求验证"""
        manager = ContextVersionManager()
        
        # 1.0版本数据
        v1_0_data = {"version": "1.0"}
        
        # 检查基础功能
        is_valid, missing = manager.validate_version_requirements(
            v1_0_data,
            ["basic_context"]
        )
        assert is_valid is True
        assert len(missing) == 0
        
        # 检查1.1功能
        is_valid, missing = manager.validate_version_requirements(
            v1_0_data,
            ["user_preferences", "session_context"]
        )
        assert is_valid is False
        assert "user_preferences" in missing
        assert "session_context" in missing
        
        # 1.2版本数据
        v1_2_data = {"version": "1.2"}
        
        # 检查所有功能
        is_valid, missing = manager.validate_version_requirements(
            v1_2_data,
            ["basic_context", "user_preferences", "workflow_metadata", "generic_support"]
        )
        assert is_valid is True
        assert len(missing) == 0


class TestVersionMigrationIntegration:
    """版本迁移集成测试"""
    
    def test_full_migration_cycle(self):
        """测试完整的迁移周期"""
        manager = ContextVersionManager()
        
        # 创建1.0版本数据
        original = {
            "user_id": "cycle_user",
            "session_id": "cycle_session",
            "version": "1.0",
            "status": "running",
            "metadata": {"original": "data"}
        }
        
        # 升级到1.2
        upgraded = manager.upgrade_context(copy.deepcopy(original))
        assert upgraded["version"] == "1.2"
        assert "user_preferences" in upgraded
        assert "workflow_metadata" in upgraded
        assert upgraded["metadata"]["original"] == "data"  # 原数据保留
        
        # 降级回1.0
        downgraded = manager.downgrade_context(copy.deepcopy(upgraded), "1.0")
        assert downgraded["version"] == "1.0"
        assert "user_preferences" not in downgraded
        assert "workflow_metadata" not in downgraded
        assert downgraded["metadata"]["original"] == "data"  # 原数据仍保留
    
    def test_migration_data_preservation(self):
        """测试迁移过程中数据保留"""
        manager = ContextVersionManager()
        
        # 创建包含自定义数据的上下文
        data = {
            "user_id": "preserve_user",
            "session_id": "preserve_session",
            "version": "1.0",
            "custom_field_1": "value1",
            "custom_field_2": {"nested": "data"},
            "metadata": {
                "key1": "value1",
                "key2": [1, 2, 3]
            }
        }
        
        # 升级
        upgraded = manager.upgrade_context(copy.deepcopy(data))
        
        # 验证自定义字段保留
        assert upgraded["custom_field_1"] == "value1"
        assert upgraded["custom_field_2"]["nested"] == "data"
        assert upgraded["metadata"]["key1"] == "value1"
        assert upgraded["metadata"]["key2"] == [1, 2, 3]
    
    def test_migration_with_datetime(self):
        """测试包含datetime的迁移"""
        v1_1_data = {
            "user_id": "datetime_user",
            "session_id": "datetime_session",
            "version": "1.1",
            "last_updated": "2024-01-01T12:00:00",
            "user_preferences": {"language": "zh-CN"}
        }
        
        migrated = ContextMigrator.migrate_context(
            copy.deepcopy(v1_1_data),
            "1.1",
            "1.2"
        )
        
        # last_updated应该被处理但不丢失
        assert "last_updated" in migrated