import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, asdict
from redis.asyncio import Redis

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class FeatureVersion:
    """特征版本信息"""
    version_id: str
    feature_schema: Dict[str, Any]
    created_at: datetime
    description: str
    is_active: bool = False
    parent_version: Optional[str] = None
    changes: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.changes is None:
            self.changes = []
        if self.metadata is None:
            self.metadata = {}

class FeatureVersionControl:
    """特征版本控制系统"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.version_prefix = "feature_version"
        self.current_version_key = f"{self.version_prefix}:current"
        self.version_history_key = f"{self.version_prefix}:history"
        
    async def create_version(
        self,
        feature_schema: Dict[str, Any],
        description: str,
        parent_version: Optional[str] = None,
        auto_activate: bool = False
    ) -> FeatureVersion:
        """创建新的特征版本
        
        Args:
            feature_schema: 特征模式定义
            description: 版本描述
            parent_version: 父版本ID
            auto_activate: 是否自动激活
            
        Returns:
            FeatureVersion: 创建的特征版本
        """
        try:
            # 生成版本ID
            version_id = self._generate_version_id(feature_schema)
            
            # 检查版本是否已存在
            if await self._version_exists(version_id):
                logger.warning(f"特征版本已存在: {version_id}")
                return await self.get_version(version_id)
            
            # 获取父版本信息
            changes = []
            if parent_version:
                parent = await self.get_version(parent_version)
                if parent:
                    changes = self._calculate_changes(
                        parent.feature_schema,
                        feature_schema
                    )
            
            # 创建版本对象
            version = FeatureVersion(
                version_id=version_id,
                feature_schema=feature_schema,
                created_at=utc_now(),
                description=description,
                is_active=auto_activate,
                parent_version=parent_version,
                changes=changes,
                metadata={
                    "feature_count": len(feature_schema),
                    "feature_types": self._get_feature_types(feature_schema)
                }
            )
            
            # 保存到Redis
            await self._save_version(version)
            
            # 添加到版本历史
            await self.redis.zadd(
                self.version_history_key,
                {version_id: version.created_at.timestamp()}
            )
            
            # 如果自动激活，设置为当前版本
            if auto_activate:
                await self.activate_version(version_id)
            
            logger.info(f"创建特征版本: {version_id}")
            return version
            
        except Exception as e:
            logger.error(f"创建特征版本失败: {e}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[FeatureVersion]:
        """获取特征版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            Optional[FeatureVersion]: 特征版本对象
        """
        try:
            version_key = f"{self.version_prefix}:{version_id}"
            version_data = await self.redis.get(version_key)
            
            if version_data:
                data = json.loads(version_data)
                # 转换时间戳
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                return FeatureVersion(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"获取特征版本失败 version_id={version_id}: {e}")
            return None
    
    async def get_current_version(self) -> Optional[FeatureVersion]:
        """获取当前激活的特征版本
        
        Returns:
            Optional[FeatureVersion]: 当前版本
        """
        try:
            current_id = await self.redis.get(self.current_version_key)
            if current_id:
                return await self.get_version(current_id.decode() if isinstance(current_id, bytes) else current_id)
            return None
            
        except Exception as e:
            logger.error(f"获取当前版本失败: {e}")
            return None
    
    async def activate_version(self, version_id: str) -> bool:
        """激活特征版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            bool: 是否激活成功
        """
        try:
            # 检查版本是否存在
            version = await self.get_version(version_id)
            if not version:
                logger.error(f"版本不存在: {version_id}")
                return False
            
            # 停用当前版本
            current_version = await self.get_current_version()
            if current_version:
                current_version.is_active = False
                await self._save_version(current_version)
            
            # 激活新版本
            version.is_active = True
            await self._save_version(version)
            
            # 设置为当前版本
            await self.redis.set(self.current_version_key, version_id)
            
            # 记录激活事件
            await self._log_version_event(version_id, "activated")
            
            logger.info(f"激活特征版本: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"激活版本失败 version_id={version_id}: {e}")
            return False
    
    async def rollback_to_version(self, version_id: str) -> bool:
        """回滚到指定版本
        
        Args:
            version_id: 要回滚到的版本ID
            
        Returns:
            bool: 是否回滚成功
        """
        try:
            # 检查目标版本是否存在
            target_version = await self.get_version(version_id)
            if not target_version:
                logger.error(f"目标版本不存在: {version_id}")
                return False
            
            # 获取当前版本（用于记录）
            current_version = await self.get_current_version()
            
            # 激活目标版本
            success = await self.activate_version(version_id)
            
            if success:
                # 记录回滚事件
                await self._log_version_event(
                    version_id,
                    "rollback",
                    {
                        "from_version": current_version.version_id if current_version else None,
                        "to_version": version_id
                    }
                )
                
                logger.info(f"回滚到版本: {version_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"回滚版本失败 version_id={version_id}: {e}")
            return False
    
    async def list_versions(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[FeatureVersion]:
        """列出特征版本
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[FeatureVersion]: 版本列表
        """
        try:
            # 从历史记录中获取版本ID
            version_ids = await self.redis.zrevrange(
                self.version_history_key,
                offset,
                offset + limit - 1
            )
            
            # 获取版本详情
            versions = []
            for version_id in version_ids:
                version_id_str = version_id.decode() if isinstance(version_id, bytes) else version_id
                version = await self.get_version(version_id_str)
                if version:
                    versions.append(version)
            
            return versions
            
        except Exception as e:
            logger.error(f"列出版本失败: {e}")
            return []
    
    async def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """比较两个版本的差异
        
        Args:
            version_id_1: 版本1 ID
            version_id_2: 版本2 ID
            
        Returns:
            Dict[str, Any]: 差异信息
        """
        try:
            version_1 = await self.get_version(version_id_1)
            version_2 = await self.get_version(version_id_2)
            
            if not version_1 or not version_2:
                return {"error": "版本不存在"}
            
            changes = self._calculate_changes(
                version_1.feature_schema,
                version_2.feature_schema
            )
            
            return {
                "version_1": version_id_1,
                "version_2": version_id_2,
                "changes": changes,
                "added_features": list(
                    set(version_2.feature_schema.keys()) - 
                    set(version_1.feature_schema.keys())
                ),
                "removed_features": list(
                    set(version_1.feature_schema.keys()) - 
                    set(version_2.feature_schema.keys())
                ),
                "modified_features": [
                    key for key in version_1.feature_schema.keys()
                    if key in version_2.feature_schema and
                    version_1.feature_schema[key] != version_2.feature_schema[key]
                ]
            }
            
        except Exception as e:
            logger.error(f"比较版本失败: {e}")
            return {"error": str(e)}
    
    async def validate_version(self, version_id: str) -> Tuple[bool, List[str]]:
        """验证特征版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            Tuple[bool, List[str]]: 是否有效，错误列表
        """
        errors = []
        
        try:
            version = await self.get_version(version_id)
            if not version:
                return False, ["版本不存在"]
            
            # 验证特征模式
            if not version.feature_schema:
                errors.append("特征模式为空")
            
            # 验证特征类型
            for feature_name, feature_config in version.feature_schema.items():
                if not isinstance(feature_config, dict):
                    errors.append(f"特征 {feature_name} 配置格式错误")
                    continue
                
                # 检查必需字段
                if "type" not in feature_config:
                    errors.append(f"特征 {feature_name} 缺少类型定义")
                
                if "description" not in feature_config:
                    errors.append(f"特征 {feature_name} 缺少描述")
            
            # 如果有父版本，验证兼容性
            if version.parent_version:
                parent = await self.get_version(version.parent_version)
                if parent:
                    # 检查是否有破坏性变更
                    removed_features = set(parent.feature_schema.keys()) - set(version.feature_schema.keys())
                    if removed_features:
                        errors.append(f"删除了特征: {', '.join(removed_features)}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"验证版本失败 version_id={version_id}: {e}")
            return False, [str(e)]
    
    async def export_version(self, version_id: str) -> Dict[str, Any]:
        """导出特征版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            Dict[str, Any]: 导出的版本数据
        """
        try:
            version = await self.get_version(version_id)
            if not version:
                return {}
            
            # 转换为可序列化格式
            export_data = asdict(version)
            export_data["created_at"] = version.created_at.isoformat()
            
            # 添加额外信息
            export_data["export_time"] = utc_now().isoformat()
            export_data["export_format"] = "1.0"
            
            return export_data
            
        except Exception as e:
            logger.error(f"导出版本失败 version_id={version_id}: {e}")
            return {}
    
    async def import_version(
        self,
        version_data: Dict[str, Any],
        override_existing: bool = False
    ) -> Optional[FeatureVersion]:
        """导入特征版本
        
        Args:
            version_data: 版本数据
            override_existing: 是否覆盖已存在的版本
            
        Returns:
            Optional[FeatureVersion]: 导入的版本
        """
        try:
            version_id = version_data.get("version_id")
            
            # 检查版本是否已存在
            if await self._version_exists(version_id) and not override_existing:
                logger.warning(f"版本已存在且不允许覆盖: {version_id}")
                return None
            
            # 转换时间字段
            if "created_at" in version_data:
                version_data["created_at"] = datetime.fromisoformat(version_data["created_at"])
            
            # 创建版本对象
            version = FeatureVersion(**{
                k: v for k, v in version_data.items()
                if k in FeatureVersion.__dataclass_fields__
            })
            
            # 保存版本
            await self._save_version(version)
            
            # 添加到历史
            await self.redis.zadd(
                self.version_history_key,
                {version_id: version.created_at.timestamp()}
            )
            
            logger.info(f"导入特征版本: {version_id}")
            return version
            
        except Exception as e:
            logger.error(f"导入版本失败: {e}")
            return None
    
    def _generate_version_id(self, feature_schema: Dict[str, Any]) -> str:
        """生成版本ID
        
        Args:
            feature_schema: 特征模式
            
        Returns:
            str: 版本ID
        """
        # 基于特征模式内容生成哈希
        schema_str = json.dumps(feature_schema, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:8]
        
        # 添加时间戳
        timestamp = utc_now().strftime("%Y%m%d%H%M%S")
        
        return f"v_{timestamp}_{schema_hash}"
    
    async def _version_exists(self, version_id: str) -> bool:
        """检查版本是否存在
        
        Args:
            version_id: 版本ID
            
        Returns:
            bool: 是否存在
        """
        version_key = f"{self.version_prefix}:{version_id}"
        return await self.redis.exists(version_key) > 0
    
    async def _save_version(self, version: FeatureVersion):
        """保存版本到Redis
        
        Args:
            version: 版本对象
        """
        version_key = f"{self.version_prefix}:{version.version_id}"
        version_data = asdict(version)
        version_data["created_at"] = version.created_at.isoformat()
        
        await self.redis.set(
            version_key,
            json.dumps(version_data, default=str)
        )
    
    def _calculate_changes(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any]
    ) -> List[str]:
        """计算模式变更
        
        Args:
            old_schema: 旧模式
            new_schema: 新模式
            
        Returns:
            List[str]: 变更列表
        """
        changes = []
        
        # 新增的特征
        added = set(new_schema.keys()) - set(old_schema.keys())
        for feature in added:
            changes.append(f"添加特征: {feature}")
        
        # 删除的特征
        removed = set(old_schema.keys()) - set(new_schema.keys())
        for feature in removed:
            changes.append(f"删除特征: {feature}")
        
        # 修改的特征
        for feature in set(old_schema.keys()) & set(new_schema.keys()):
            if old_schema[feature] != new_schema[feature]:
                changes.append(f"修改特征: {feature}")
        
        return changes
    
    def _get_feature_types(self, feature_schema: Dict[str, Any]) -> Dict[str, int]:
        """获取特征类型统计
        
        Args:
            feature_schema: 特征模式
            
        Returns:
            Dict[str, int]: 类型统计
        """
        type_counts = {}
        
        for feature_config in feature_schema.values():
            if isinstance(feature_config, dict):
                feature_type = feature_config.get("type", "unknown")
                type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        
        return type_counts
    
    async def _log_version_event(
        self,
        version_id: str,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录版本事件
        
        Args:
            version_id: 版本ID
            event_type: 事件类型
            metadata: 事件元数据
        """
        try:
            event_key = f"{self.version_prefix}:events:{version_id}"
            event_data = {
                "type": event_type,
                "timestamp": utc_now().isoformat(),
                "metadata": metadata or {}
            }
            
            await self.redis.rpush(
                event_key,
                json.dumps(event_data)
            )
            
            # 保留最近100条事件
            await self.redis.ltrim(event_key, -100, -1)
            
        except Exception as e:
            logger.error(f"记录版本事件失败: {e}")
