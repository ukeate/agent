"""
上下文版本管理模块
处理不同版本上下文的迁移和兼容性
"""

from typing import Dict, Any, Tuple, Callable, Optional, List
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import copy
from .context import ContextVersion

class ContextMigrator:
    """上下文版本迁移器"""
    
    # 版本迁移规则映射
    MIGRATION_RULES: Dict[Tuple[str, str], str] = {
        ("1.0", "1.1"): "migrate_v1_0_to_v1_1",
        ("1.1", "1.2"): "migrate_v1_1_to_v1_2",
    }
    
    @classmethod
    def migrate_context(
        cls, 
        context_data: Dict[str, Any], 
        from_version: str, 
        to_version: str
    ) -> Dict[str, Any]:
        """迁移上下文数据到新版本"""
        if from_version == to_version:
            return context_data
        
        # 复制数据以避免修改原始数据
        migrated_data = copy.deepcopy(context_data)
        
        # 找到迁移路径
        migration_path = cls._find_migration_path(from_version, to_version)
        if not migration_path:
            raise ValueError(
                f"无法找到从版本 {from_version} 到 {to_version} 的迁移路径"
            )
        
        # 按顺序执行迁移
        current_version = from_version
        for next_version in migration_path:
            migration_key = (current_version, next_version)
            if migration_key in cls.MIGRATION_RULES:
                method_name = cls.MIGRATION_RULES[migration_key]
                migration_method = getattr(cls, method_name, None)
                if migration_method:
                    migrated_data = migration_method(migrated_data)
                    migrated_data["version"] = next_version
                else:
                    raise ValueError(
                        f"迁移方法 {method_name} 未实现"
                    )
            current_version = next_version
        
        return migrated_data
    
    @classmethod
    def _find_migration_path(cls, from_version: str, to_version: str) -> List[str]:
        """找到版本迁移路径"""
        # 简单的线性迁移路径
        versions = ["1.0", "1.1", "1.2"]
        
        try:
            from_idx = versions.index(from_version)
            to_idx = versions.index(to_version)
        except ValueError:
            return []
        
        if from_idx >= to_idx:
            return []
        
        return versions[from_idx + 1:to_idx + 1]
    
    @classmethod
    def is_compatible(cls, context_version: str, required_version: str) -> bool:
        """检查版本兼容性"""
        # 相同版本总是兼容
        if context_version == required_version:
            return True
        
        # 检查是否可以迁移
        migration_path = cls._find_migration_path(context_version, required_version)
        return len(migration_path) > 0
    
    @classmethod
    def migrate_v1_0_to_v1_1(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """从1.0版本迁移到1.1版本"""
        # 1.1版本新增了user_preferences字段
        if "user_preferences" not in data:
            data["user_preferences"] = {
                "language": "zh-CN",
                "timezone": "Asia/Shanghai",
                "theme": "light"
            }
        
        # 1.1版本新增了session_context字段
        if "session_context" not in data:
            data["session_context"] = {
                "session_id": data.get("session_id", ""),
                "created_at": utc_now().isoformat(),
                "last_active": utc_now().isoformat(),
                "message_count": 0
            }
        
        # 1.1版本新增了performance_tags字段
        if "performance_tags" not in data:
            data["performance_tags"] = []
        
        return data
    
    @classmethod
    def migrate_v1_1_to_v1_2(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """从1.1版本迁移到1.2版本"""
        # 1.2版本新增了workflow_metadata字段
        if "workflow_metadata" not in data:
            data["workflow_metadata"] = {
                "workflow_id": data.get("workflow_id"),
                "workflow_version": "1.0",
                "parent_workflow_id": None,
                "execution_path": [],
                "checkpoints": []
            }
            # 移除旧的workflow_id字段（已迁移到workflow_metadata中）
            if "workflow_id" in data:
                del data["workflow_id"]
        
        # 1.2版本扩展了user_preferences
        if "user_preferences" in data:
            if "notification_enabled" not in data["user_preferences"]:
                data["user_preferences"]["notification_enabled"] = True
            if "custom_settings" not in data["user_preferences"]:
                data["user_preferences"]["custom_settings"] = {}
        
        # 1.2版本扩展了session_context
        if "session_context" in data:
            if "interaction_mode" not in data["session_context"]:
                data["session_context"]["interaction_mode"] = "chat"
        
        # 确保last_updated是datetime类型
        if "last_updated" in data and isinstance(data["last_updated"], str):
            try:
                # 尝试解析ISO格式的时间字符串
                data["last_updated"] = datetime.fromisoformat(data["last_updated"])
            except (ValueError, TypeError):
                data["last_updated"] = None
        
        return data

class VersionCompatibilityChecker:
    """版本兼容性检查器"""
    
    @staticmethod
    def check_compatibility(
        context_version: str,
        feature_requirements: Dict[str, str]
    ) -> Dict[str, bool]:
        """检查版本对特定功能的兼容性"""
        compatibility_matrix = {
            "1.0": {
                "basic_context": True,
                "user_preferences": False,
                "session_context": False,
                "workflow_metadata": False,
                "performance_tags": False,
                "generic_support": False
            },
            "1.1": {
                "basic_context": True,
                "user_preferences": True,
                "session_context": True,
                "workflow_metadata": False,
                "performance_tags": True,
                "generic_support": False
            },
            "1.2": {
                "basic_context": True,
                "user_preferences": True,
                "session_context": True,
                "workflow_metadata": True,
                "performance_tags": True,
                "generic_support": True
            }
        }
        
        if context_version not in compatibility_matrix:
            return {feature: False for feature in feature_requirements}
        
        version_features = compatibility_matrix[context_version]
        return {
            feature: version_features.get(feature, False)
            for feature in feature_requirements
        }
    
    @staticmethod
    def get_minimum_version(features: List[str]) -> str:
        """获取支持所有指定功能的最低版本"""
        version_features = {
            "1.0": ["basic_context"],
            "1.1": ["basic_context", "user_preferences", "session_context", "performance_tags"],
            "1.2": [
                "basic_context", "user_preferences", "session_context", 
                "workflow_metadata", "performance_tags", "generic_support"
            ]
        }
        
        for version in ["1.0", "1.1", "1.2"]:
            if all(feature in version_features[version] for feature in features):
                return version
        
        # 如果没有版本支持所有功能，返回最新版本
        return "1.2"

class ContextVersionManager:
    """上下文版本管理器"""
    
    def __init__(self):
        self.migrator = ContextMigrator()
        self.checker = VersionCompatibilityChecker()
    
    def upgrade_context(
        self, 
        context_data: Dict[str, Any], 
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """升级上下文到指定版本或最新版本"""
        current_version = context_data.get("version", "1.0")
        target = target_version or ContextVersion.CURRENT.value
        
        if current_version == target:
            return context_data
        
        return self.migrator.migrate_context(
            context_data, 
            current_version, 
            target
        )
    
    def downgrade_context(
        self, 
        context_data: Dict[str, Any], 
        target_version: str
    ) -> Dict[str, Any]:
        """降级上下文到指定版本（可能丢失数据）"""
        current_version = context_data.get("version", ContextVersion.CURRENT.value)
        
        # 复制数据以避免修改原始数据
        downgraded_data = context_data.copy()
        
        # 降级通过移除新版本字段实现
        if target_version == "1.0":
            # 恢复workflow_id字段（在移除workflow_metadata之前）
            if "workflow_metadata" in downgraded_data:
                downgraded_data["workflow_id"] = downgraded_data["workflow_metadata"].get("workflow_id")
            
            # 移除1.1和1.2版本的字段
            fields_to_remove = [
                "user_preferences", "session_context", "performance_tags",
                "workflow_metadata", "custom_data"
            ]
            for field in fields_to_remove:
                downgraded_data.pop(field, None)
        
        elif target_version == "1.1":
            # 移除1.2版本的字段
            fields_to_remove = ["workflow_metadata", "custom_data"]
            for field in fields_to_remove:
                downgraded_data.pop(field, None)
            
            # 简化user_preferences和session_context
            if "user_preferences" in downgraded_data:
                downgraded_data["user_preferences"].pop("notification_enabled", None)
                downgraded_data["user_preferences"].pop("custom_settings", None)
            
            if "session_context" in downgraded_data:
                downgraded_data["session_context"].pop("interaction_mode", None)
        
        downgraded_data["version"] = target_version
        return downgraded_data
    
    def validate_version_requirements(
        self,
        context_data: Dict[str, Any],
        required_features: List[str]
    ) -> Tuple[bool, List[str]]:
        """验证上下文版本是否满足功能需求"""
        context_version = context_data.get("version", "1.0")
        compatibility = self.checker.check_compatibility(
            context_version, 
            {feature: "" for feature in required_features}
        )
        
        missing_features = [
            feature for feature, is_compatible in compatibility.items()
            if not is_compatible
        ]
        
        return len(missing_features) == 0, missing_features
