"""文档版本控制与变更跟踪系统"""

import hashlib
import logging
import difflib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """变更类型枚举"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    MOVED = "moved"


@dataclass
class DocumentVersion:
    """文档版本数据类"""
    version_id: str
    doc_id: str
    version_number: int
    content: str
    content_hash: str
    created_at: datetime
    created_by: Optional[str] = None
    change_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version_id: Optional[str] = None
    is_current: bool = True


@dataclass
class DocumentChange:
    """文档变更数据类"""
    change_id: str
    doc_id: str
    from_version: Optional[str]
    to_version: str
    change_type: ChangeType
    change_details: Dict[str, Any]
    changed_at: datetime
    changed_by: Optional[str] = None
    impact_analysis: Optional[Dict[str, Any]] = None


@dataclass
class VersionBranch:
    """版本分支数据类"""
    branch_id: str
    branch_name: str
    base_version_id: str
    head_version_id: str
    created_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentVersionManager:
    """文档版本管理器
    
    实现文档版本管理、变更跟踪和影响分析
    """
    
    def __init__(
        self,
        max_versions_per_doc: int = 20,
        enable_auto_merge: bool = False,
        track_minor_changes: bool = True
    ):
        """初始化版本管理器
        
        Args:
            max_versions_per_doc: 每个文档最大版本数
            enable_auto_merge: 是否启用自动合并
            track_minor_changes: 是否跟踪细微变更
        """
        self.max_versions_per_doc = max_versions_per_doc
        self.enable_auto_merge = enable_auto_merge
        self.track_minor_changes = track_minor_changes
        
        # 存储版本历史（实际应使用数据库）
        self.versions: Dict[str, List[DocumentVersion]] = {}
        self.changes: List[DocumentChange] = []
        self.branches: Dict[str, VersionBranch] = {}
    
    async def create_version(
        self,
        doc_id: str,
        content: str,
        created_by: Optional[str] = None,
        change_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentVersion:
        """创建新版本
        
        Args:
            doc_id: 文档ID
            content: 文档内容
            created_by: 创建者
            change_summary: 变更摘要
            metadata: 元数据
            
        Returns:
            新版本对象
        """
        # 计算内容哈希
        content_hash = self._calculate_hash(content)
        
        # 获取当前版本历史
        doc_versions = self.versions.get(doc_id, [])
        
        # 检查是否与最新版本相同
        if doc_versions:
            latest_version = doc_versions[-1]
            if latest_version.content_hash == content_hash:
                logger.info(f"Content unchanged for doc {doc_id}, skipping version creation")
                return latest_version
            
            # 标记旧版本为非当前版本
            for v in doc_versions:
                v.is_current = False
        
        # 生成版本号
        version_number = len(doc_versions) + 1
        
        # 生成版本ID
        version_id = f"v_{doc_id}_{version_number}_{content_hash[:8]}"
        
        # 创建新版本
        new_version = DocumentVersion(
            version_id=version_id,
            doc_id=doc_id,
            version_number=version_number,
            content=content,
            content_hash=content_hash,
            created_at=datetime.utcnow(),
            created_by=created_by,
            change_summary=change_summary,
            metadata=metadata or {},
            parent_version_id=doc_versions[-1].version_id if doc_versions else None,
            is_current=True
        )
        
        # 添加到版本历史
        if doc_id not in self.versions:
            self.versions[doc_id] = []
        self.versions[doc_id].append(new_version)
        
        # 创建变更记录
        if doc_versions:
            change = await self._create_change_record(
                doc_id,
                doc_versions[-1],
                new_version,
                created_by
            )
            self.changes.append(change)
        else:
            # 首次创建
            change = DocumentChange(
                change_id=f"change_{doc_id}_{version_number}",
                doc_id=doc_id,
                from_version=None,
                to_version=version_id,
                change_type=ChangeType.CREATED,
                change_details={"initial_creation": True},
                changed_at=datetime.utcnow(),
                changed_by=created_by
            )
            self.changes.append(change)
        
        # 清理旧版本
        await self._cleanup_old_versions(doc_id)
        
        return new_version
    
    async def get_version(
        self,
        doc_id: str,
        version_id: Optional[str] = None,
        version_number: Optional[int] = None
    ) -> Optional[DocumentVersion]:
        """获取特定版本
        
        Args:
            doc_id: 文档ID
            version_id: 版本ID
            version_number: 版本号
            
        Returns:
            版本对象或None
        """
        doc_versions = self.versions.get(doc_id, [])
        
        if not doc_versions:
            return None
        
        # 如果未指定版本，返回当前版本
        if not version_id and not version_number:
            for v in reversed(doc_versions):
                if v.is_current:
                    return v
            return doc_versions[-1]
        
        # 按版本ID查找
        if version_id:
            for v in doc_versions:
                if v.version_id == version_id:
                    return v
        
        # 按版本号查找
        if version_number:
            for v in doc_versions:
                if v.version_number == version_number:
                    return v
        
        return None
    
    async def get_version_history(
        self,
        doc_id: str,
        limit: Optional[int] = None
    ) -> List[DocumentVersion]:
        """获取版本历史
        
        Args:
            doc_id: 文档ID
            limit: 限制返回数量
            
        Returns:
            版本列表
        """
        doc_versions = self.versions.get(doc_id, [])
        
        if limit:
            return doc_versions[-limit:]
        
        return doc_versions
    
    async def compare_versions(
        self,
        doc_id: str,
        version1_id: str,
        version2_id: str
    ) -> Dict[str, Any]:
        """比较两个版本
        
        Args:
            doc_id: 文档ID
            version1_id: 版本1 ID
            version2_id: 版本2 ID
            
        Returns:
            比较结果
        """
        version1 = await self.get_version(doc_id, version_id=version1_id)
        version2 = await self.get_version(doc_id, version_id=version2_id)
        
        if not version1 or not version2:
            raise ValueError("Version not found")
        
        # 生成diff
        diff = self._generate_diff(version1.content, version2.content)
        
        # 计算变更统计
        stats = self._calculate_change_stats(diff)
        
        # 检测变更类型
        change_types = self._detect_change_types(version1.content, version2.content)
        
        return {
            "version1": {
                "id": version1.version_id,
                "number": version1.version_number,
                "created_at": version1.created_at.isoformat()
            },
            "version2": {
                "id": version2.version_id,
                "number": version2.version_number,
                "created_at": version2.created_at.isoformat()
            },
            "diff": diff,
            "stats": stats,
            "change_types": change_types
        }
    
    async def rollback_version(
        self,
        doc_id: str,
        target_version_id: str,
        rollback_by: Optional[str] = None
    ) -> DocumentVersion:
        """回滚到指定版本
        
        Args:
            doc_id: 文档ID
            target_version_id: 目标版本ID
            rollback_by: 回滚操作者
            
        Returns:
            新创建的回滚版本
        """
        target_version = await self.get_version(doc_id, version_id=target_version_id)
        
        if not target_version:
            raise ValueError(f"Target version {target_version_id} not found")
        
        # 创建新版本（内容为目标版本）
        rollback_version = await self.create_version(
            doc_id=doc_id,
            content=target_version.content,
            created_by=rollback_by,
            change_summary=f"Rollback to version {target_version.version_number}",
            metadata={
                "rollback_from": target_version.version_id,
                "rollback_at": datetime.utcnow().isoformat()
            }
        )
        
        return rollback_version
    
    async def create_branch(
        self,
        doc_id: str,
        branch_name: str,
        base_version_id: Optional[str] = None
    ) -> VersionBranch:
        """创建版本分支
        
        Args:
            doc_id: 文档ID
            branch_name: 分支名称
            base_version_id: 基础版本ID
            
        Returns:
            分支对象
        """
        # 获取基础版本
        if not base_version_id:
            base_version = await self.get_version(doc_id)
            if not base_version:
                raise ValueError(f"No version found for doc {doc_id}")
            base_version_id = base_version.version_id
        
        # 生成分支ID
        branch_id = f"branch_{doc_id}_{branch_name}_{datetime.utcnow().timestamp()}"
        
        # 创建分支
        branch = VersionBranch(
            branch_id=branch_id,
            branch_name=branch_name,
            base_version_id=base_version_id,
            head_version_id=base_version_id,
            created_at=datetime.utcnow()
        )
        
        self.branches[branch_id] = branch
        
        return branch
    
    async def merge_branches(
        self,
        source_branch_id: str,
        target_branch_id: str,
        merge_by: Optional[str] = None
    ) -> DocumentVersion:
        """合并分支
        
        Args:
            source_branch_id: 源分支ID
            target_branch_id: 目标分支ID
            merge_by: 合并操作者
            
        Returns:
            合并后的新版本
        """
        source_branch = self.branches.get(source_branch_id)
        target_branch = self.branches.get(target_branch_id)
        
        if not source_branch or not target_branch:
            raise ValueError("Branch not found")
        
        # 获取分支头版本
        source_version = await self.get_version(
            doc_id=source_branch.branch_name.split("_")[0],
            version_id=source_branch.head_version_id
        )
        target_version = await self.get_version(
            doc_id=target_branch.branch_name.split("_")[0],
            version_id=target_branch.head_version_id
        )
        
        if not source_version or not target_version:
            raise ValueError("Branch head version not found")
        
        # 执行合并（简化版本，实际应处理冲突）
        if self.enable_auto_merge:
            merged_content = self._auto_merge(
                source_version.content,
                target_version.content
            )
        else:
            # 简单地使用源分支内容
            merged_content = source_version.content
        
        # 创建合并版本
        merge_version = await self.create_version(
            doc_id=source_version.doc_id,
            content=merged_content,
            created_by=merge_by,
            change_summary=f"Merge {source_branch.branch_name} into {target_branch.branch_name}",
            metadata={
                "merge_source": source_branch_id,
                "merge_target": target_branch_id,
                "merge_at": datetime.utcnow().isoformat()
            }
        )
        
        # 更新目标分支头
        target_branch.head_version_id = merge_version.version_id
        
        return merge_version
    
    async def analyze_impact(
        self,
        doc_id: str,
        version_id: str,
        related_docs: List[str]
    ) -> Dict[str, Any]:
        """分析变更影响
        
        Args:
            doc_id: 文档ID
            version_id: 版本ID
            related_docs: 相关文档ID列表
            
        Returns:
            影响分析结果
        """
        version = await self.get_version(doc_id, version_id=version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # 获取父版本
        parent_version = None
        if version.parent_version_id:
            parent_version = await self.get_version(
                doc_id,
                version_id=version.parent_version_id
            )
        
        impacts = {
            "direct_changes": [],
            "potential_impacts": [],
            "risk_level": "low",
            "affected_documents": []
        }
        
        # 分析直接变更
        if parent_version:
            diff = self._generate_diff(parent_version.content, version.content)
            change_stats = self._calculate_change_stats(diff)
            
            impacts["direct_changes"] = {
                "lines_added": change_stats["added"],
                "lines_removed": change_stats["removed"],
                "lines_modified": change_stats["modified"]
            }
            
            # 评估风险级别
            total_changes = sum(change_stats.values())
            if total_changes > 100:
                impacts["risk_level"] = "high"
            elif total_changes > 50:
                impacts["risk_level"] = "medium"
        
        # 分析对相关文档的潜在影响
        for related_doc_id in related_docs:
            # 这里应该进行更复杂的依赖分析
            impacts["potential_impacts"].append({
                "doc_id": related_doc_id,
                "impact_type": "dependency",
                "confidence": 0.5
            })
            impacts["affected_documents"].append(related_doc_id)
        
        return impacts
    
    async def _create_change_record(
        self,
        doc_id: str,
        from_version: DocumentVersion,
        to_version: DocumentVersion,
        changed_by: Optional[str]
    ) -> DocumentChange:
        """创建变更记录
        
        Args:
            doc_id: 文档ID
            from_version: 起始版本
            to_version: 目标版本
            changed_by: 变更者
            
        Returns:
            变更记录
        """
        # 生成diff
        diff = self._generate_diff(from_version.content, to_version.content)
        
        # 计算变更统计
        stats = self._calculate_change_stats(diff)
        
        # 检测变更类型
        if stats["added"] > 0 and stats["removed"] == 0:
            change_type = ChangeType.MODIFIED
        elif stats["removed"] > 0 and stats["added"] == 0:
            change_type = ChangeType.MODIFIED
        else:
            change_type = ChangeType.MODIFIED
        
        # 创建变更记录
        change = DocumentChange(
            change_id=f"change_{doc_id}_{to_version.version_number}",
            doc_id=doc_id,
            from_version=from_version.version_id,
            to_version=to_version.version_id,
            change_type=change_type,
            change_details={
                "stats": stats,
                "summary": to_version.change_summary
            },
            changed_at=to_version.created_at,
            changed_by=changed_by
        )
        
        return change
    
    def _calculate_hash(self, content: str) -> str:
        """计算内容哈希
        
        Args:
            content: 内容
            
        Returns:
            哈希值
        """
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_diff(self, content1: str, content2: str) -> List[str]:
        """生成内容差异
        
        Args:
            content1: 内容1
            content2: 内容2
            
        Returns:
            差异列表
        """
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        
        differ = difflib.unified_diff(
            lines1,
            lines2,
            lineterm='',
            n=3  # 上下文行数
        )
        
        return list(differ)
    
    def _calculate_change_stats(self, diff: List[str]) -> Dict[str, int]:
        """计算变更统计
        
        Args:
            diff: 差异列表
            
        Returns:
            统计信息
        """
        stats = {
            "added": 0,
            "removed": 0,
            "modified": 0
        }
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                stats["added"] += 1
            elif line.startswith('-') and not line.startswith('---'):
                stats["removed"] += 1
        
        # 估算修改行数（简化版）
        stats["modified"] = min(stats["added"], stats["removed"])
        
        return stats
    
    def _detect_change_types(
        self,
        content1: str,
        content2: str
    ) -> List[str]:
        """检测变更类型
        
        Args:
            content1: 内容1
            content2: 内容2
            
        Returns:
            变更类型列表
        """
        change_types = []
        
        # 检测结构变更
        if len(content1.splitlines()) != len(content2.splitlines()):
            change_types.append("structural")
        
        # 检测格式变更
        if content1.count(' ') != content2.count(' '):
            change_types.append("formatting")
        
        # 检测内容变更
        if content1 != content2:
            change_types.append("content")
        
        return change_types
    
    def _auto_merge(self, content1: str, content2: str) -> str:
        """自动合并内容（简化版）
        
        Args:
            content1: 内容1
            content2: 内容2
            
        Returns:
            合并后的内容
        """
        # 这里应该实现更复杂的合并逻辑
        # 简化版：如果内容不同，使用较长的版本
        if len(content1) >= len(content2):
            return content1
        else:
            return content2
    
    async def _cleanup_old_versions(self, doc_id: str):
        """清理旧版本
        
        Args:
            doc_id: 文档ID
        """
        doc_versions = self.versions.get(doc_id, [])
        
        if len(doc_versions) > self.max_versions_per_doc:
            # 保留最新的版本
            versions_to_keep = doc_versions[-self.max_versions_per_doc:]
            self.versions[doc_id] = versions_to_keep
            
            logger.info(
                f"Cleaned up old versions for doc {doc_id}, "
                f"kept {len(versions_to_keep)} versions"
            )