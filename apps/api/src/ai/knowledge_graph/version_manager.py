"""
版本管理器 - 知识图谱版本控制和快照管理
"""

import asyncio
import json
import hashlib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import gzip
from src.core.utils import secure_pickle as pickle
from src.core.logging import get_logger, setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    from rdflib import Graph, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    Graph = URIRef = Literal = BNode = None
    RDF = RDFS = OWL = None

class VersionType(Enum):
    """版本类型"""
    MAJOR = "major"        # 主要版本 - 重大结构变更
    MINOR = "minor"        # 次要版本 - 功能增加
    PATCH = "patch"        # 补丁版本 - 错误修复
    SNAPSHOT = "snapshot"  # 快照版本 - 临时保存

class ChangeType(Enum):
    """变更类型"""
    ADD_TRIPLE = "add_triple"
    REMOVE_TRIPLE = "remove_triple"
    MODIFY_TRIPLE = "modify_triple"
    ADD_ENTITY = "add_entity"
    REMOVE_ENTITY = "remove_entity"
    MODIFY_ENTITY = "modify_entity"
    ADD_RELATION = "add_relation"
    REMOVE_RELATION = "remove_relation"
    MODIFY_RELATION = "modify_relation"
    BULK_IMPORT = "bulk_import"
    BULK_DELETE = "bulk_delete"

@dataclass
class VersionInfo:
    """版本信息"""
    version_id: str
    version_number: str
    version_type: VersionType
    created_at: datetime
    created_by: str
    description: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    graph_hash: Optional[str] = None
    triple_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    size_bytes: int = 0

@dataclass
class Change:
    """单个变更记录"""
    change_id: str
    change_type: ChangeType
    timestamp: datetime
    user_id: str
    affected_subjects: List[str]
    affected_predicates: List[str]
    affected_objects: List[str]
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChangeSet:
    """变更集"""
    changeset_id: str
    version_from: str
    version_to: str
    changes: List[Change]
    created_at: datetime
    description: str
    statistics: Optional[Dict[str, Any]] = None

@dataclass
class DiffResult:
    """差异对比结果"""
    added_triples: List[Dict[str, str]]
    removed_triples: List[Dict[str, str]]
    modified_triples: List[Dict[str, Any]]
    statistics: Dict[str, int]

class VersionManager:
    """知识图谱版本管理器"""
    
    def __init__(self, storage_path: str = "/tmp/kg_versions", max_versions: int = 100):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_versions = max_versions
        
        # 版本存储
        self.versions: Dict[str, VersionInfo] = {}
        self.changesets: Dict[str, ChangeSet] = {}
        self.current_version: Optional[str] = None
        
        # 版本图数据存储
        self.version_graphs: Dict[str, Any] = {}
        
        self._setup_logging()
        self._load_metadata()
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    def _load_metadata(self):
        """加载版本元数据"""
        try:
            metadata_file = self.storage_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 加载版本信息
                for version_data in data.get("versions", []):
                    version_info = VersionInfo(**version_data)
                    version_info.created_at = datetime.fromisoformat(version_data["created_at"])
                    version_info.version_type = VersionType(version_data["version_type"])
                    self.versions[version_info.version_id] = version_info
                
                # 加载变更集
                for changeset_data in data.get("changesets", []):
                    changeset = ChangeSet(**changeset_data)
                    changeset.created_at = datetime.fromisoformat(changeset_data["created_at"])
                    
                    # 重建Change对象
                    changes = []
                    for change_data in changeset_data.get("changes", []):
                        change = Change(**change_data)
                        change.timestamp = datetime.fromisoformat(change_data["timestamp"])
                        change.change_type = ChangeType(change_data["change_type"])
                        changes.append(change)
                    changeset.changes = changes
                    
                    self.changesets[changeset.changeset_id] = changeset
                
                self.current_version = data.get("current_version")
                
                self.logger.info(f"已加载 {len(self.versions)} 个版本和 {len(self.changesets)} 个变更集")
        
        except Exception as e:
            self.logger.warning(f"加载版本元数据失败: {e}")
    
    def _save_metadata(self):
        """保存版本元数据"""
        try:
            # 准备序列化数据
            versions_data = []
            for version in self.versions.values():
                version_dict = asdict(version)
                version_dict["created_at"] = version.created_at.isoformat()
                version_dict["version_type"] = version.version_type.value
                versions_data.append(version_dict)
            
            changesets_data = []
            for changeset in self.changesets.values():
                changeset_dict = asdict(changeset)
                changeset_dict["created_at"] = changeset.created_at.isoformat()
                
                # 序列化变更
                changes_data = []
                for change in changeset.changes:
                    change_dict = asdict(change)
                    change_dict["timestamp"] = change.timestamp.isoformat()
                    change_dict["change_type"] = change.change_type.value
                    changes_data.append(change_dict)
                changeset_dict["changes"] = changes_data
                
                changesets_data.append(changeset_dict)
            
            metadata = {
                "versions": versions_data,
                "changesets": changesets_data,
                "current_version": self.current_version,
                "last_updated": utc_now().isoformat()
            }
            
            metadata_file = self.storage_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存版本元数据失败: {e}")
            raise
    
    async def create_version(
        self,
        graph_data: Any,
        version_type: VersionType = VersionType.MINOR,
        description: str = "",
        created_by: str = "system",
        tags: List[str] = None
    ) -> VersionInfo:
        """
        创建新版本
        
        Args:
            graph_data: 图数据
            version_type: 版本类型
            description: 版本描述
            created_by: 创建者
            tags: 版本标签
            
        Returns:
            VersionInfo: 版本信息
        """
        try:
            # 生成版本ID和版本号
            version_id = self._generate_version_id()
            version_number = self._generate_version_number(version_type)
            
            # 计算图哈希
            graph_hash = await self._calculate_graph_hash(graph_data)
            
            # 统计信息
            stats = await self._calculate_graph_statistics(graph_data)
            
            # 创建版本信息
            version_info = VersionInfo(
                version_id=version_id,
                version_number=version_number,
                version_type=version_type,
                created_at=utc_now(),
                created_by=created_by,
                description=description,
                parent_version=self.current_version,
                tags=tags or [],
                graph_hash=graph_hash,
                triple_count=stats["triple_count"],
                entity_count=stats["entity_count"],
                relation_count=stats["relation_count"],
                size_bytes=stats["size_bytes"]
            )
            
            # 保存图数据
            await self._save_version_data(version_id, graph_data)
            
            # 如果有父版本，创建变更集
            if self.current_version:
                changeset = await self._create_changeset(
                    self.current_version,
                    version_id,
                    f"更新到版本 {version_number}"
                )
                self.changesets[changeset.changeset_id] = changeset
            
            # 更新版本信息
            self.versions[version_id] = version_info
            self.current_version = version_id
            
            # 清理旧版本
            await self._cleanup_old_versions()
            
            # 保存元数据
            self._save_metadata()
            
            self.logger.info(f"已创建版本 {version_number} (ID: {version_id})")
            return version_info
            
        except Exception as e:
            self.logger.error(f"创建版本失败: {e}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """获取版本信息"""
        return self.versions.get(version_id)
    
    async def get_version_data(self, version_id: str) -> Optional[Any]:
        """获取版本数据"""
        if version_id not in self.versions:
            return None
        
        try:
            return await self._load_version_data(version_id)
        except Exception as e:
            self.logger.error(f"加载版本数据失败 {version_id}: {e}")
            return None
    
    async def list_versions(
        self,
        limit: int = 50,
        offset: int = 0,
        version_type: Optional[VersionType] = None,
        tag: Optional[str] = None
    ) -> List[VersionInfo]:
        """列出版本"""
        versions = list(self.versions.values())
        
        # 过滤
        if version_type:
            versions = [v for v in versions if v.version_type == version_type]
        
        if tag:
            versions = [v for v in versions if tag in (v.tags or [])]
        
        # 排序 - 按创建时间倒序
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        # 分页
        return versions[offset:offset + limit]
    
    async def delete_version(self, version_id: str) -> bool:
        """删除版本"""
        if version_id not in self.versions:
            return False
        
        try:
            # 检查是否为当前版本
            if self.current_version == version_id:
                # 设置父版本为当前版本
                version_info = self.versions[version_id]
                if version_info.parent_version:
                    self.current_version = version_info.parent_version
                else:
                    self.current_version = None
            
            # 删除版本数据文件
            await self._delete_version_data(version_id)
            
            # 删除相关变更集
            changesets_to_delete = [
                cid for cid, cs in self.changesets.items()
                if cs.version_from == version_id or cs.version_to == version_id
            ]
            
            for changeset_id in changesets_to_delete:
                del self.changesets[changeset_id]
            
            # 删除版本信息
            del self.versions[version_id]
            
            # 保存元数据
            self._save_metadata()
            
            self.logger.info(f"已删除版本 {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除版本失败 {version_id}: {e}")
            return False
    
    async def compare_versions(self, version_from: str, version_to: str) -> DiffResult:
        """比较两个版本的差异"""
        try:
            # 加载两个版本的数据
            data_from = await self._load_version_data(version_from)
            data_to = await self._load_version_data(version_to)
            
            if not data_from or not data_to:
                raise ValueError("无法加载版本数据")
            
            # 转换为三元组集合进行比较
            triples_from = await self._extract_triples(data_from)
            triples_to = await self._extract_triples(data_to)
            
            # 计算差异
            added_triples = []
            removed_triples = []
            modified_triples = []
            
            # 使用哈希进行快速比较
            hash_from = {self._triple_hash(t): t for t in triples_from}
            hash_to = {self._triple_hash(t): t for t in triples_to}
            
            # 新增的三元组
            for h, triple in hash_to.items():
                if h not in hash_from:
                    added_triples.append(triple)
            
            # 删除的三元组
            for h, triple in hash_from.items():
                if h not in hash_to:
                    removed_triples.append(triple)
            
            # 统计信息
            statistics = {
                "total_triples_from": len(triples_from),
                "total_triples_to": len(triples_to),
                "added_count": len(added_triples),
                "removed_count": len(removed_triples),
                "modified_count": len(modified_triples),
                "unchanged_count": len(set(hash_from.keys()) & set(hash_to.keys()))
            }
            
            return DiffResult(
                added_triples=added_triples,
                removed_triples=removed_triples,
                modified_triples=modified_triples,
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"版本比较失败 {version_from} -> {version_to}: {e}")
            raise
    
    async def revert_to_version(self, version_id: str) -> bool:
        """回滚到指定版本"""
        if version_id not in self.versions:
            return False
        
        try:
            # 将指定版本设为当前版本
            old_current = self.current_version
            self.current_version = version_id
            
            # 创建回滚变更记录
            if old_current:
                changeset = await self._create_changeset(
                    old_current,
                    version_id,
                    f"回滚到版本 {self.versions[version_id].version_number}"
                )
                self.changesets[changeset.changeset_id] = changeset
            
            # 保存元数据
            self._save_metadata()
            
            self.logger.info(f"已回滚到版本 {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"回滚到版本失败 {version_id}: {e}")
            return False
    
    async def create_branch(self, branch_name: str, from_version: str = None) -> str:
        """创建分支（通过标签实现）"""
        base_version = from_version or self.current_version
        if not base_version or base_version not in self.versions:
            raise ValueError("基础版本不存在")
        
        # 给版本添加分支标签
        version_info = self.versions[base_version]
        if not version_info.tags:
            version_info.tags = []
        
        branch_tag = f"branch:{branch_name}"
        if branch_tag not in version_info.tags:
            version_info.tags.append(branch_tag)
        
        self._save_metadata()
        
        self.logger.info(f"已创建分支 {branch_name} 基于版本 {base_version}")
        return base_version
    
    async def merge_versions(self, source_version: str, target_version: str) -> VersionInfo:
        """合并版本（简化实现）"""
        # 这是一个简化的合并实现
        # 实际项目中需要更复杂的三路合并算法
        
        source_data = await self._load_version_data(source_version)
        target_data = await self._load_version_data(target_version)
        
        if not source_data or not target_data:
            raise ValueError("无法加载版本数据")
        
        # 简单合并：取两个版本的所有唯一三元组
        source_triples = await self._extract_triples(source_data)
        target_triples = await self._extract_triples(target_data)
        
        # 使用集合去重
        all_triples = list(set(source_triples + target_triples))
        
        # 创建合并后的版本
        merged_version = await self.create_version(
            graph_data={"triples": all_triples},
            version_type=VersionType.MINOR,
            description=f"合并版本 {source_version} 和 {target_version}",
            tags=[f"merge:{source_version}:{target_version}"]
        )
        
        return merged_version
    
    def _generate_version_id(self) -> str:
        """生成版本ID"""
        timestamp = utc_now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(f"{timestamp}_{len(self.versions)}".encode()).hexdigest()[:8]
        return f"v_{timestamp}_{random_suffix}"
    
    def _generate_version_number(self, version_type: VersionType) -> str:
        """生成版本号"""
        if not self.versions:
            return "1.0.0"
        
        # 获取最新版本号
        latest_versions = sorted(
            [v.version_number for v in self.versions.values()],
            key=lambda x: [int(part) for part in x.split('.')],
            reverse=True
        )
        
        if not latest_versions:
            return "1.0.0"
        
        latest = latest_versions[0]
        parts = [int(p) for p in latest.split('.')]
        
        if version_type == VersionType.MAJOR:
            return f"{parts[0] + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{parts[0]}.{parts[1] + 1}.0"
        else:  # PATCH或SNAPSHOT
            return f"{parts[0]}.{parts[1]}.{parts[2] + 1}"
    
    async def _calculate_graph_hash(self, graph_data: Any) -> str:
        """计算图数据哈希"""
        try:
            # 将图数据序列化为一致的字符串
            if isinstance(graph_data, dict):
                content = json.dumps(graph_data, sort_keys=True, ensure_ascii=False)
            else:
                content = str(graph_data)
            
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception:
            return hashlib.md5(str(graph_data).encode('utf-8')).hexdigest()
    
    async def _calculate_graph_statistics(self, graph_data: Any) -> Dict[str, int]:
        """计算图统计信息"""
        stats = {
            "triple_count": 0,
            "entity_count": 0,
            "relation_count": 0,
            "size_bytes": 0
        }
        
        try:
            if isinstance(graph_data, dict) and 'triples' in graph_data:
                triples = graph_data['triples']
                stats["triple_count"] = len(triples)
                
                # 统计唯一实体和关系
                subjects = set()
                objects = set()
                predicates = set()
                
                for triple in triples:
                    if isinstance(triple, dict):
                        subjects.add(triple.get('subject', ''))
                        predicates.add(triple.get('predicate', ''))
                        objects.add(triple.get('object', ''))
                
                stats["entity_count"] = len(subjects | objects)
                stats["relation_count"] = len(predicates)
            
            # 计算数据大小
            content = json.dumps(graph_data, ensure_ascii=False) if isinstance(graph_data, dict) else str(graph_data)
            stats["size_bytes"] = len(content.encode('utf-8'))
            
        except Exception as e:
            self.logger.warning(f"计算图统计信息失败: {e}")
        
        return stats
    
    async def _save_version_data(self, version_id: str, graph_data: Any):
        """保存版本数据到文件"""
        try:
            version_file = self.storage_path / f"{version_id}.pkl.gz"
            
            with gzip.open(version_file, 'wb') as f:
                pickle.dump(graph_data, f)
            
        except Exception as e:
            self.logger.error(f"保存版本数据失败 {version_id}: {e}")
            raise
    
    async def _load_version_data(self, version_id: str) -> Optional[Any]:
        """从文件加载版本数据"""
        try:
            version_file = self.storage_path / f"{version_id}.pkl.gz"
            
            if not version_file.exists():
                return None
            
            with gzip.open(version_file, 'rb') as f:
                return pickle.load(f)
            
        except Exception as e:
            self.logger.error(f"加载版本数据失败 {version_id}: {e}")
            return None
    
    async def _delete_version_data(self, version_id: str):
        """删除版本数据文件"""
        try:
            version_file = self.storage_path / f"{version_id}.pkl.gz"
            if version_file.exists():
                version_file.unlink()
        except Exception as e:
            self.logger.warning(f"删除版本数据文件失败 {version_id}: {e}")
    
    async def _extract_triples(self, graph_data: Any) -> List[Dict[str, str]]:
        """从图数据中提取三元组"""
        triples = []
        
        if isinstance(graph_data, dict) and 'triples' in graph_data:
            for triple in graph_data['triples']:
                if isinstance(triple, dict):
                    triples.append({
                        'subject': triple.get('subject', ''),
                        'predicate': triple.get('predicate', ''),
                        'object': triple.get('object', '')
                    })
        
        return triples
    
    def _triple_hash(self, triple: Dict[str, str]) -> str:
        """计算三元组哈希"""
        content = f"{triple.get('subject', '')}|{triple.get('predicate', '')}|{triple.get('object', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def _create_changeset(self, version_from: str, version_to: str, description: str) -> ChangeSet:
        """创建变更集"""
        changeset_id = f"cs_{utc_now().strftime('%Y%m%d_%H%M%S')}_{len(self.changesets)}"
        
        # 比较版本差异
        diff_result = await self.compare_versions(version_from, version_to)
        
        # 创建变更记录
        changes = []
        timestamp = utc_now()
        
        # 添加的三元组
        for i, triple in enumerate(diff_result.added_triples):
            change = Change(
                change_id=f"{changeset_id}_add_{i}",
                change_type=ChangeType.ADD_TRIPLE,
                timestamp=timestamp,
                user_id="system",
                affected_subjects=[triple.get('subject', '')],
                affected_predicates=[triple.get('predicate', '')],
                affected_objects=[triple.get('object', '')],
                new_value=triple
            )
            changes.append(change)
        
        # 删除的三元组
        for i, triple in enumerate(diff_result.removed_triples):
            change = Change(
                change_id=f"{changeset_id}_remove_{i}",
                change_type=ChangeType.REMOVE_TRIPLE,
                timestamp=timestamp,
                user_id="system",
                affected_subjects=[triple.get('subject', '')],
                affected_predicates=[triple.get('predicate', '')],
                affected_objects=[triple.get('object', '')],
                old_value=triple
            )
            changes.append(change)
        
        return ChangeSet(
            changeset_id=changeset_id,
            version_from=version_from,
            version_to=version_to,
            changes=changes,
            created_at=timestamp,
            description=description,
            statistics=diff_result.statistics
        )
    
    async def _cleanup_old_versions(self):
        """清理旧版本"""
        if len(self.versions) <= self.max_versions:
            return
        
        # 按创建时间排序，保留最新的版本
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        
        # 删除超过限制的旧版本
        for version in sorted_versions[self.max_versions:]:
            await self.delete_version(version.version_id)
    
    async def get_version_statistics(self) -> Dict[str, Any]:
        """获取版本统计信息"""
        if not self.versions:
            return {"total_versions": 0}
        
        versions = list(self.versions.values())
        
        return {
            "total_versions": len(versions),
            "current_version": self.current_version,
            "version_types": {
                vt.value: len([v for v in versions if v.version_type == vt])
                for vt in VersionType
            },
            "oldest_version": min(versions, key=lambda x: x.created_at).version_number,
            "newest_version": max(versions, key=lambda x: x.created_at).version_number,
            "total_changesets": len(self.changesets),
            "storage_size_bytes": sum(v.size_bytes for v in versions),
            "average_triple_count": sum(v.triple_count for v in versions) // len(versions) if versions else 0
        }

# 便捷函数
async def create_knowledge_graph_version(
    graph_data: Any,
    version_type: VersionType = VersionType.MINOR,
    description: str = "",
    created_by: str = "system",
    storage_path: str = "/tmp/kg_versions"
) -> VersionInfo:
    """
    创建知识图谱版本的便捷函数
    
    Args:
        graph_data: 图数据
        version_type: 版本类型
        description: 版本描述
        created_by: 创建者
        storage_path: 存储路径
        
    Returns:
        VersionInfo: 版本信息
    """
    manager = VersionManager(storage_path)
    return await manager.create_version(
        graph_data=graph_data,
        version_type=version_type,
        description=description,
        created_by=created_by
    )

if __name__ == "__main__":
    # 测试版本管理器
    async def test_version_manager():
        setup_logging()
        logger.info("测试版本管理器")
        
        manager = VersionManager("/tmp/test_kg_versions")
        
        # 创建初始版本
        initial_data = {
            "triples": [
                {"subject": "ex:John", "predicate": "rdf:type", "object": "ex:Person"},
                {"subject": "ex:John", "predicate": "ex:name", "object": "John Doe"}
            ]
        }
        
        version1 = await manager.create_version(
            graph_data=initial_data,
            version_type=VersionType.MAJOR,
            description="初始版本",
            created_by="test_user"
        )
        
        logger.info("创建版本1", version_number=version1.version_number)
        
        # 创建第二个版本
        updated_data = {
            "triples": [
                {"subject": "ex:John", "predicate": "rdf:type", "object": "ex:Person"},
                {"subject": "ex:John", "predicate": "ex:name", "object": "John Doe"},
                {"subject": "ex:John", "predicate": "ex:age", "object": "30"}
            ]
        }
        
        version2 = await manager.create_version(
            graph_data=updated_data,
            version_type=VersionType.MINOR,
            description="添加年龄信息",
            created_by="test_user"
        )
        
        logger.info("创建版本2", version_number=version2.version_number)
        
        # 比较版本差异
        diff = await manager.compare_versions(version1.version_id, version2.version_id)
        logger.info(
            "版本差异",
            added_triples=len(diff.added_triples),
            removed_triples=len(diff.removed_triples),
        )
        
        # 列出所有版本
        versions = await manager.list_versions()
        logger.info("版本数量", total=len(versions))
        
        # 获取统计信息
        stats = await manager.get_version_statistics()
        logger.info("版本统计", stats=stats)
        logger.info("版本管理器测试完成")
    
    asyncio.run(test_version_manager())
