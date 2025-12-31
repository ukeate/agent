"""
训练数据版本管理系统

这个模块提供完整的数据版本控制功能，包括：
- 数据版本创建和管理
- 版本间比较分析
- 数据回滚和恢复
- 版本合并和分支管理
"""

import json
import hashlib
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
import shutil
import os
from dataclasses import dataclass
from enum import Enum
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
from .models import DataVersionModel, DataRecordModel
from .core import DataVersion, VersionComparison, DataRecord, DataFilter, ExportFormat

class VersionOperation(Enum):
    """版本操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"

class ConflictResolution(Enum):
    """冲突解决策略"""
    AUTO_MERGE = "auto_merge"
    MANUAL = "manual"
    LATEST_WINS = "latest_wins"
    OLDEST_WINS = "oldest_wins"

@dataclass
class VersionMetrics:
    """版本指标"""
    version_id: str
    record_count: int
    size_bytes: int
    quality_score: float
    creation_time: datetime
    data_hash: str
    parent_version: Optional[str] = None

@dataclass
class VersionDiff:
    """版本差异"""
    version1_id: str
    version2_id: str
    added_records: List[Dict[str, Any]]
    removed_records: List[Dict[str, Any]]
    modified_records: List[Dict[str, Any]]
    summary: Dict[str, int]

@dataclass
class MergeResult:
    """合并结果"""
    new_version_id: str
    conflicts: List[Dict[str, Any]]
    auto_resolved: int
    manual_resolution_needed: int
    success: bool

class DataVersionManager:
    """数据版本管理器"""
    
    def __init__(self, db_session: AsyncSession, storage_path: str = "./data_versions"):
        self.db = db_session
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    async def create_version(
        self,
        dataset_name: str,
        version_number: str,
        description: str,
        created_by: str,
        data_filter: Optional[DataFilter] = None,
        parent_version: Optional[str] = None
    ) -> str:
        """创建新版本"""
        version_id = f"{dataset_name}_{version_number}_{int(utc_now().timestamp())}"
        
        # 获取数据记录
        records = await self._get_filtered_records(data_filter)
        
        # 保存数据到文件
        data_path, data_hash, size_bytes = await self._save_version_data(
            version_id, records
        )
        
        # 计算变化摘要
        changes_summary = await self._calculate_changes_summary(
            records, parent_version
        )
        
        # 创建版本记录
        version_model = DataVersionModel(
            version_id=version_id,
            dataset_name=dataset_name,
            version_number=version_number,
            description=description,
            created_by=created_by,
            parent_version=parent_version,
            changes_summary=changes_summary,
            data_path=str(data_path),
            data_hash=data_hash,
            record_count=len(records),
            size_bytes=size_bytes
        )
        
        self.db.add(version_model)
        await self.db.commit()
        
        return version_id
    
    async def _get_filtered_records(
        self,
        data_filter: Optional[DataFilter]
    ) -> List[DataRecord]:
        """根据过滤条件获取数据记录"""
        query = select(DataRecordModel)
        
        if data_filter:
            if data_filter.source_id:
                query = query.where(DataRecordModel.source_id == data_filter.source_id)
            
            if data_filter.status:
                query = query.where(DataRecordModel.status == data_filter.status.value)
            
            if data_filter.min_quality_score:
                query = query.where(DataRecordModel.quality_score >= data_filter.min_quality_score)
            
            if data_filter.date_from:
                query = query.where(DataRecordModel.created_at >= data_filter.date_from)
            
            if data_filter.date_to:
                query = query.where(DataRecordModel.created_at <= data_filter.date_to)
            
            if data_filter.limit:
                query = query.limit(data_filter.limit)
            
            query = query.offset(data_filter.offset)
        
        result = await self.db.execute(query)
        models = result.scalars().all()
        
        # 转换为DataRecord对象
        records = []
        for model in models:
            record = DataRecord(
                record_id=model.record_id,
                source_id=model.source_id,
                raw_data=model.raw_data or {},
                processed_data=model.processed_data,
                metadata=model.metadata or {},
                quality_score=model.quality_score,
                status=model.status,
                created_at=model.created_at,
                processed_at=model.processed_at
            )
            records.append(record)
        
        return records
    
    async def _save_version_data(
        self,
        version_id: str,
        records: List[DataRecord]
    ) -> Tuple[Path, str, int]:
        """保存版本数据到文件"""
        version_dir = self.storage_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # 保存为JSONL格式
        data_file = version_dir / "data.jsonl"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            for record in records:
                record_dict = {
                    'record_id': record.record_id,
                    'source_id': record.source_id,
                    'raw_data': record.raw_data,
                    'processed_data': record.processed_data,
                    'metadata': record.metadata,
                    'quality_score': record.quality_score,
                    'status': record.status,
                    'created_at': record.created_at.isoformat() if record.created_at else None,
                    'processed_at': record.processed_at.isoformat() if record.processed_at else None
                }
                f.write(json.dumps(record_dict, ensure_ascii=False) + '\n')
        
        # 计算文件哈希
        data_hash = await self._calculate_file_hash(data_file)
        
        # 获取文件大小
        size_bytes = data_file.stat().st_size
        
        # 保存元数据
        metadata_file = version_dir / "metadata.json"
        metadata = {
            'version_id': version_id,
            'record_count': len(records),
            'data_hash': data_hash,
            'size_bytes': size_bytes,
            'created_at': utc_now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return data_file, data_hash, size_bytes
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    async def _calculate_changes_summary(
        self,
        current_records: List[DataRecord],
        parent_version: Optional[str]
    ) -> Dict[str, Any]:
        """计算变化摘要"""
        if not parent_version:
            return {
                'total_records': len(current_records),
                'added_records': len(current_records),
                'modified_records': 0,
                'removed_records': 0
            }
        
        # 获取父版本数据
        parent_records = await self._load_version_data(parent_version)
        
        # 创建记录ID集合
        current_ids = {r.record_id for r in current_records}
        parent_ids = {r.record_id for r in parent_records}
        
        # 计算变化
        added_ids = current_ids - parent_ids
        removed_ids = parent_ids - current_ids
        common_ids = current_ids & parent_ids
        
        # 检查修改的记录
        modified_count = 0
        current_dict = {r.record_id: r for r in current_records}
        parent_dict = {r.record_id: r for r in parent_records}
        
        for record_id in common_ids:
            current_record = current_dict[record_id]
            parent_record = parent_dict[record_id]
            
            # 比较数据内容
            if (current_record.processed_data != parent_record.processed_data or
                current_record.quality_score != parent_record.quality_score or
                current_record.status != parent_record.status):
                modified_count += 1
        
        return {
            'total_records': len(current_records),
            'added_records': len(added_ids),
            'modified_records': modified_count,
            'removed_records': len(removed_ids)
        }
    
    async def _load_version_data(self, version_id: str) -> List[DataRecord]:
        """加载版本数据"""
        version_dir = self.storage_path / version_id
        data_file = version_dir / "data.jsonl"
        
        if not data_file.exists():
            return []
        
        records = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record_dict = json.loads(line)
                    
                    created_at = None
                    if record_dict.get('created_at'):
                        created_at = datetime.fromisoformat(record_dict['created_at'].replace('Z', '+00:00'))
                    
                    processed_at = None
                    if record_dict.get('processed_at'):
                        processed_at = datetime.fromisoformat(record_dict['processed_at'].replace('Z', '+00:00'))
                    
                    record = DataRecord(
                        record_id=record_dict['record_id'],
                        source_id=record_dict['source_id'],
                        raw_data=record_dict.get('raw_data', {}),
                        processed_data=record_dict.get('processed_data'),
                        metadata=record_dict.get('metadata', {}),
                        quality_score=record_dict.get('quality_score'),
                        status=record_dict.get('status', 'raw'),
                        created_at=created_at,
                        processed_at=processed_at
                    )
                    records.append(record)
        
        return records
    
    async def compare_versions(
        self,
        version1_id: str,
        version2_id: str
    ) -> VersionComparison:
        """比较两个版本"""
        # 加载两个版本的数据
        records1 = await self._load_version_data(version1_id)
        records2 = await self._load_version_data(version2_id)
        
        # 创建记录字典
        dict1 = {r.record_id: r for r in records1}
        dict2 = {r.record_id: r for r in records2}
        
        # 计算差异
        ids1 = set(dict1.keys())
        ids2 = set(dict2.keys())
        
        added_ids = ids2 - ids1
        removed_ids = ids1 - ids2
        common_ids = ids1 & ids2
        
        added_records = [self._record_to_dict(dict2[rid]) for rid in added_ids]
        removed_records = [self._record_to_dict(dict1[rid]) for rid in removed_ids]
        
        # 检查修改的记录
        modified_records = []
        for record_id in common_ids:
            record1 = dict1[record_id]
            record2 = dict2[record_id]
            
            if (record1.processed_data != record2.processed_data or
                record1.quality_score != record2.quality_score or
                record1.status != record2.status):
                
                modified_records.append({
                    'record_id': record_id,
                    'before': self._record_to_dict(record1),
                    'after': self._record_to_dict(record2),
                    'changes': self._calculate_record_changes(record1, record2)
                })
        
        # 生成摘要
        summary = {
            'total_records_v1': len(records1),
            'total_records_v2': len(records2),
            'added_count': len(added_records),
            'removed_count': len(removed_records),
            'modified_count': len(modified_records),
            'unchanged_count': len(common_ids) - len(modified_records)
        }
        
        return VersionComparison(
            version1_id=version1_id,
            version2_id=version2_id,
            summary=summary,
            added_records=added_records,
            removed_records=removed_records,
            modified_records=modified_records
        )
    
    def _record_to_dict(self, record: DataRecord) -> Dict[str, Any]:
        """将DataRecord转换为字典"""
        return {
            'record_id': record.record_id,
            'source_id': record.source_id,
            'raw_data': record.raw_data,
            'processed_data': record.processed_data,
            'metadata': record.metadata,
            'quality_score': record.quality_score,
            'status': record.status,
            'created_at': record.created_at.isoformat() if record.created_at else None,
            'processed_at': record.processed_at.isoformat() if record.processed_at else None
        }
    
    def _calculate_record_changes(
        self,
        record1: DataRecord,
        record2: DataRecord
    ) -> Dict[str, Any]:
        """计算记录变化"""
        changes = {}
        
        if record1.processed_data != record2.processed_data:
            changes['processed_data'] = {
                'before': record1.processed_data,
                'after': record2.processed_data
            }
        
        if record1.quality_score != record2.quality_score:
            changes['quality_score'] = {
                'before': record1.quality_score,
                'after': record2.quality_score
            }
        
        if record1.status != record2.status:
            changes['status'] = {
                'before': record1.status,
                'after': record2.status
            }
        
        return changes
    
    async def rollback_to_version(
        self,
        target_version_id: str,
        new_version_number: str,
        created_by: str
    ) -> str:
        """回滚到指定版本"""
        # 获取目标版本信息
        stmt = select(DataVersionModel).where(
            DataVersionModel.version_id == target_version_id
        )
        result = await self.db.execute(stmt)
        target_version = result.scalar_one()
        
        # 加载目标版本数据
        records = await self._load_version_data(target_version_id)
        
        # 创建新版本（基于目标版本的数据）
        new_version_id = await self.create_version(
            dataset_name=target_version.dataset_name,
            version_number=new_version_number,
            description=f"Rollback to version {target_version.version_number}",
            created_by=created_by,
            parent_version=target_version_id
        )
        
        # 更新数据库中的记录
        await self._restore_records_to_database(records)
        
        return new_version_id
    
    async def _restore_records_to_database(self, records: List[DataRecord]) -> None:
        """将记录恢复到数据库"""
        if not records:
            return

        record_ids = [record.record_id for record in records]
        result = await self.db.execute(
            select(DataRecordModel).where(DataRecordModel.record_id.in_(record_ids))
        )
        existing_records = {model.record_id: model for model in result.scalars().all()}

        for record in records:
            status_value = record.status.value if hasattr(record.status, "value") else str(record.status)
            model = existing_records.get(record.record_id)
            if model:
                model.source_id = record.source_id
                model.raw_data = record.raw_data or {}
                model.processed_data = record.processed_data
                model.metadata = record.metadata or {}
                model.quality_score = record.quality_score
                model.status = status_value
                model.created_at = record.created_at
                model.processed_at = record.processed_at
                model.updated_at = utc_now()
            else:
                self.db.add(
                    DataRecordModel(
                        record_id=record.record_id,
                        source_id=record.source_id,
                        raw_data=record.raw_data or {},
                        processed_data=record.processed_data,
                        metadata=record.metadata or {},
                        quality_score=record.quality_score,
                        status=status_value,
                        created_at=record.created_at,
                        processed_at=record.processed_at,
                    )
                )

        await self.db.commit()
    
    async def merge_versions(
        self,
        version1_id: str,
        version2_id: str,
        merge_strategy: ConflictResolution,
        created_by: str,
        new_version_number: str
    ) -> MergeResult:
        """合并两个版本"""
        # 加载两个版本的数据
        records1 = await self._load_version_data(version1_id)
        records2 = await self._load_version_data(version2_id)
        
        # 创建记录字典
        dict1 = {r.record_id: r for r in records1}
        dict2 = {r.record_id: r for r in records2}
        
        # 查找冲突
        conflicts = []
        merged_records = {}
        auto_resolved = 0
        manual_resolution_needed = 0
        
        # 处理共同的记录（可能有冲突）
        common_ids = set(dict1.keys()) & set(dict2.keys())
        for record_id in common_ids:
            record1 = dict1[record_id]
            record2 = dict2[record_id]
            
            if self._records_conflict(record1, record2):
                conflict_info = {
                    'record_id': record_id,
                    'version1_data': self._record_to_dict(record1),
                    'version2_data': self._record_to_dict(record2)
                }
                
                if merge_strategy == ConflictResolution.AUTO_MERGE:
                    resolved_record = self._auto_merge_records(record1, record2)
                    merged_records[record_id] = resolved_record
                    conflict_info['resolution'] = 'auto_merged'
                    conflict_info['merged_data'] = self._record_to_dict(resolved_record)
                    auto_resolved += 1
                elif merge_strategy == ConflictResolution.LATEST_WINS:
                    # 选择创建时间较晚的记录
                    if record2.created_at >= record1.created_at:
                        merged_records[record_id] = record2
                    else:
                        merged_records[record_id] = record1
                    conflict_info['resolution'] = 'latest_wins'
                    auto_resolved += 1
                elif merge_strategy == ConflictResolution.OLDEST_WINS:
                    # 选择创建时间较早的记录
                    if record1.created_at <= record2.created_at:
                        merged_records[record_id] = record1
                    else:
                        merged_records[record_id] = record2
                    conflict_info['resolution'] = 'oldest_wins'
                    auto_resolved += 1
                else:  # MANUAL
                    conflict_info['resolution'] = 'manual_required'
                    manual_resolution_needed += 1
                
                conflicts.append(conflict_info)
            else:
                # 没有冲突，选择任一记录
                merged_records[record_id] = record1
        
        # 添加只在version1中存在的记录
        only_v1 = set(dict1.keys()) - set(dict2.keys())
        for record_id in only_v1:
            merged_records[record_id] = dict1[record_id]
        
        # 添加只在version2中存在的记录
        only_v2 = set(dict2.keys()) - set(dict1.keys())
        for record_id in only_v2:
            merged_records[record_id] = dict2[record_id]
        
        # 如果需要手动解决冲突，返回结果但不创建版本
        if manual_resolution_needed > 0 and merge_strategy == ConflictResolution.MANUAL:
            return MergeResult(
                new_version_id="",
                conflicts=conflicts,
                auto_resolved=auto_resolved,
                manual_resolution_needed=manual_resolution_needed,
                success=False
            )
        
        # 创建合并后的新版本
        records_list = list(merged_records.values())
        
        # 获取数据集名称
        stmt = select(DataVersionModel).where(
            DataVersionModel.version_id == version1_id
        )
        result = await self.db.execute(stmt)
        version1_info = result.scalar_one()
        
        new_version_id = await self.create_version(
            dataset_name=version1_info.dataset_name,
            version_number=new_version_number,
            description=f"Merge of versions {version1_id} and {version2_id}",
            created_by=created_by,
            parent_version=version1_id
        )
        
        return MergeResult(
            new_version_id=new_version_id,
            conflicts=conflicts,
            auto_resolved=auto_resolved,
            manual_resolution_needed=manual_resolution_needed,
            success=True
        )
    
    def _records_conflict(self, record1: DataRecord, record2: DataRecord) -> bool:
        """检查两个记录是否冲突"""
        return (
            record1.processed_data != record2.processed_data or
            record1.quality_score != record2.quality_score or
            record1.status != record2.status
        )
    
    def _auto_merge_records(self, record1: DataRecord, record2: DataRecord) -> DataRecord:
        """自动合并两个记录"""
        # 简单的合并策略：选择质量分数更高的记录
        if record2.quality_score and record1.quality_score:
            if record2.quality_score >= record1.quality_score:
                return record2
            else:
                return record1
        elif record2.quality_score:
            return record2
        elif record1.quality_score:
            return record1
        else:
            # 选择较新的记录
            if record2.created_at >= record1.created_at:
                return record2
            else:
                return record1
    
    async def export_version(
        self,
        version_id: str,
        export_format: ExportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """导出版本数据"""
        records = await self._load_version_data(version_id)
        
        if not output_path:
            output_path = f"{version_id}.{export_format.value}"
        
        if export_format == ExportFormat.JSON:
            await self._export_to_json(records, output_path)
        elif export_format == ExportFormat.JSONL:
            await self._export_to_jsonl(records, output_path)
        elif export_format == ExportFormat.CSV:
            await self._export_to_csv(records, output_path)
        elif export_format == ExportFormat.PARQUET:
            await self._export_to_parquet(records, output_path)
        elif export_format == ExportFormat.EXCEL:
            await self._export_to_excel(records, output_path)
        
        return output_path
    
    async def _export_to_json(self, records: List[DataRecord], output_path: str) -> None:
        """导出为JSON格式"""
        data = [self._record_to_dict(record) for record in records]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def _export_to_jsonl(self, records: List[DataRecord], output_path: str) -> None:
        """导出为JSONL格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(self._record_to_dict(record), ensure_ascii=False) + '\n')
    
    async def _export_to_csv(self, records: List[DataRecord], output_path: str) -> None:
        """导出为CSV格式"""
        if not records:
            return
        
        # 转换为DataFrame
        data = []
        for record in records:
            row = {
                'record_id': record.record_id,
                'source_id': record.source_id,
                'quality_score': record.quality_score,
                'status': record.status,
                'created_at': record.created_at,
                'processed_at': record.processed_at
            }
            
            # 展开processed_data
            if record.processed_data:
                for key, value in record.processed_data.items():
                    row[f'processed_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    async def _export_to_parquet(self, records: List[DataRecord], output_path: str) -> None:
        """导出为Parquet格式"""
        if not records:
            return
        
        data = [self._record_to_dict(record) for record in records]
        df = pd.json_normalize(data)
        df.to_parquet(output_path, index=False)
    
    async def _export_to_excel(self, records: List[DataRecord], output_path: str) -> None:
        """导出为Excel格式"""
        if not records:
            return
        
        data = [self._record_to_dict(record) for record in records]
        df = pd.json_normalize(data)
        df.to_excel(output_path, index=False)
    
    async def get_version_history(self, dataset_name: str) -> List[VersionMetrics]:
        """获取数据集的版本历史"""
        stmt = select(DataVersionModel).where(
            DataVersionModel.dataset_name == dataset_name
        ).order_by(desc(DataVersionModel.created_at))
        
        result = await self.db.execute(stmt)
        versions = result.scalars().all()
        
        metrics_list = []
        for version in versions:
            metrics = VersionMetrics(
                version_id=version.version_id,
                record_count=version.record_count,
                size_bytes=version.size_bytes,
                quality_score=0.0,  # 需要计算
                creation_time=version.created_at,
                data_hash=version.data_hash or "",
                parent_version=version.parent_version
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    async def delete_version(self, version_id: str) -> bool:
        """删除版本"""
        try:
            # 删除数据库记录
            stmt = select(DataVersionModel).where(
                DataVersionModel.version_id == version_id
            )
            result = await self.db.execute(stmt)
            version = result.scalar_one()
            
            await self.db.delete(version)
            await self.db.commit()
            
            # 删除文件
            version_dir = self.storage_path / version_id
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            return True
        except Exception:
            return False
