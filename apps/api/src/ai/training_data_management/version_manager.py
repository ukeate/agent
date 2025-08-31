"""
数据版本管理系统

提供数据版本控制、变更追踪、增量更新等功能
"""

import hashlib
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from .models import Base, DataVersion, DataVersionModel


class DataVersionManager:
    """数据版本管理器"""
    
    def __init__(self, database_url: str, storage_path: str = "./data_versions"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def create_version(
        self, 
        dataset_name: str,
        version_number: str,
        data_records: List[Dict[str, Any]],
        description: str,
        created_by: str,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建数据版本"""
        
        import uuid
        version_id = str(uuid.uuid4())
        
        # 创建版本目录
        version_dir = self.storage_path / dataset_name / version_number
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据文件
        data_file = version_dir / "data.jsonl"
        with open(data_file, 'w', encoding='utf-8') as f:
            for record in data_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # 计算数据哈希
        data_hash = self._calculate_file_hash(data_file)
        file_size = data_file.stat().st_size
        
        # 计算变更摘要
        changes_summary = {}
        if parent_version:
            changes_summary = self._calculate_changes(dataset_name, parent_version, data_records)
        
        # 创建版本记录
        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            version_number=version_number,
            description=description,
            created_by=created_by,
            parent_version=parent_version,
            changes_summary=changes_summary,
            metadata=metadata or {}
        )
        
        with self.SessionLocal() as db:
            # 检查版本号是否已存在
            existing = db.query(DataVersionModel).filter(
                DataVersionModel.dataset_name == dataset_name,
                DataVersionModel.version_number == version_number
            ).first()
            
            if existing:
                raise ValueError(f"Version {version_number} already exists for dataset {dataset_name}")
            
            db_version = DataVersionModel(
                version_id=version.version_id,
                dataset_name=version.dataset_name,
                version_number=version.version_number,
                description=version.description,
                created_by=version.created_by,
                parent_version=version.parent_version,
                changes_summary=version.changes_summary,
                metadata=version.metadata,
                data_path=str(data_file),
                data_hash=data_hash,
                record_count=len(data_records),
                size_bytes=file_size
            )
            
            db.add(db_version)
            db.commit()
            db.refresh(db_version)
        
        self.logger.info(f"Created data version: {dataset_name} v{version_number}")
        return version_id
    
    def get_version_data(self, version_id: str) -> List[Dict[str, Any]]:
        """获取版本数据"""
        
        with self.SessionLocal() as db:
            version = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == version_id
            ).first()
            
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            data_file = Path(version.data_path)
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {version.data_path}")
            
            data_records = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_records.append(json.loads(line))
            
            return data_records
    
    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """列出数据集的所有版本"""
        
        with self.SessionLocal() as db:
            versions = db.query(DataVersionModel).filter(
                DataVersionModel.dataset_name == dataset_name
            ).order_by(DataVersionModel.created_at.desc()).all()
            
            return [
                {
                    'version_id': version.version_id,
                    'version_number': version.version_number,
                    'description': version.description,
                    'created_by': version.created_by,
                    'parent_version': version.parent_version,
                    'record_count': version.record_count,
                    'size_bytes': version.size_bytes,
                    'data_hash': version.data_hash,
                    'created_at': version.created_at,
                    'changes_summary': version.changes_summary,
                    'metadata': version.metadata
                }
                for version in versions
            ]
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有数据集"""
        
        with self.SessionLocal() as db:
            # 按数据集分组统计
            dataset_stats = db.query(
                DataVersionModel.dataset_name,
                func.count(DataVersionModel.id).label('version_count'),
                func.max(DataVersionModel.created_at).label('latest_version_date'),
                func.sum(DataVersionModel.record_count).label('total_records'),
                func.sum(DataVersionModel.size_bytes).label('total_size')
            ).group_by(DataVersionModel.dataset_name).all()
            
            datasets = []
            for stat in dataset_stats:
                # 获取最新版本信息
                latest_version = db.query(DataVersionModel).filter(
                    DataVersionModel.dataset_name == stat.dataset_name
                ).order_by(DataVersionModel.created_at.desc()).first()
                
                datasets.append({
                    'dataset_name': stat.dataset_name,
                    'version_count': stat.version_count,
                    'latest_version': {
                        'version_number': latest_version.version_number,
                        'version_id': latest_version.version_id,
                        'created_at': latest_version.created_at,
                        'created_by': latest_version.created_by
                    } if latest_version else None,
                    'total_records': stat.total_records or 0,
                    'total_size_bytes': stat.total_size or 0,
                    'latest_activity': stat.latest_version_date
                })
            
            return datasets
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本"""
        
        data1 = self.get_version_data(version_id1)
        data2 = self.get_version_data(version_id2)
        
        # 获取版本信息
        with self.SessionLocal() as db:
            v1 = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == version_id1
            ).first()
            v2 = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == version_id2
            ).first()
        
        # 创建记录索引
        records1 = {self._record_hash(record): record for record in data1}
        records2 = {self._record_hash(record): record for record in data2}
        
        # 计算差异
        added_hashes = set(records2.keys()) - set(records1.keys())
        removed_hashes = set(records1.keys()) - set(records2.keys())
        common_hashes = set(records1.keys()) & set(records2.keys())
        
        # 检查修改的记录
        modified_records = []
        for hash_key in common_hashes:
            if records1[hash_key] != records2[hash_key]:
                modified_records.append({
                    'hash': hash_key,
                    'original': records1[hash_key],
                    'modified': records2[hash_key]
                })
        
        comparison = {
            'version1': {
                'version_id': version_id1,
                'version_number': v1.version_number if v1 else 'unknown',
                'created_at': v1.created_at.isoformat() if v1 and v1.created_at else None
            },
            'version2': {
                'version_id': version_id2,
                'version_number': v2.version_number if v2 else 'unknown',
                'created_at': v2.created_at.isoformat() if v2 and v2.created_at else None
            },
            'summary': {
                'total_records_v1': len(data1),
                'total_records_v2': len(data2),
                'added': len(added_hashes),
                'removed': len(removed_hashes),
                'modified': len(modified_records),
                'unchanged': len(common_hashes) - len(modified_records)
            },
            'changes': {
                'added_records': [records2[h] for h in added_hashes],
                'removed_records': [records1[h] for h in removed_hashes],
                'modified_records': modified_records
            }
        }
        
        return comparison
    
    def create_incremental_version(
        self,
        base_version_id: str,
        changes: Dict[str, List[Dict[str, Any]]],
        version_number: str,
        description: str,
        created_by: str
    ) -> str:
        """创建增量版本
        
        Args:
            base_version_id: 基础版本ID
            changes: 变更内容，包含 'added', 'removed', 'modified' 键
            version_number: 新版本号
            description: 版本描述
            created_by: 创建者
        """
        
        # 获取基础版本数据
        base_data = self.get_version_data(base_version_id)
        
        # 获取基础版本信息
        with self.SessionLocal() as db:
            base_version = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == base_version_id
            ).first()
            
            if not base_version:
                raise ValueError(f"Base version {base_version_id} not found")
        
        # 创建记录索引
        base_records = {self._record_hash(record): record for record in base_data}
        
        # 应用变更
        # 1. 移除记录
        for removed_record in changes.get('removed', []):
            removed_hash = self._record_hash(removed_record)
            base_records.pop(removed_hash, None)
        
        # 2. 修改记录
        for modified_record in changes.get('modified', []):
            if 'original' in modified_record and 'modified' in modified_record:
                old_hash = self._record_hash(modified_record['original'])
                new_hash = self._record_hash(modified_record['modified'])
                base_records.pop(old_hash, None)
                base_records[new_hash] = modified_record['modified']
        
        # 3. 添加记录
        for added_record in changes.get('added', []):
            hash_key = self._record_hash(added_record)
            base_records[hash_key] = added_record
        
        # 创建新版本
        new_data = list(base_records.values())
        
        return self.create_version(
            dataset_name=base_version.dataset_name,
            version_number=version_number,
            data_records=new_data,
            description=description,
            created_by=created_by,
            parent_version=base_version_id
        )
    
    def rollback_to_version(self, dataset_name: str, target_version_id: str, created_by: str) -> str:
        """回滚到指定版本"""
        
        # 获取目标版本数据
        target_data = self.get_version_data(target_version_id)
        
        with self.SessionLocal() as db:
            target_version = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == target_version_id
            ).first()
            
            if not target_version:
                raise ValueError(f"Target version {target_version_id} not found")
            
            # 获取最新版本号并生成回滚版本号
            latest_version = db.query(DataVersionModel).filter(
                DataVersionModel.dataset_name == dataset_name
            ).order_by(DataVersionModel.created_at.desc()).first()
            
            if latest_version:
                # 解析版本号生成新版本
                try:
                    parts = latest_version.version_number.split('.')
                    major, minor = int(parts[0]), int(parts[1])
                    new_version_number = f"{major}.{minor + 1}"
                except (ValueError, IndexError):
                    # 如果版本号格式不标准，使用时间戳
                    import time
                    new_version_number = f"rollback-{int(time.time())}"
            else:
                new_version_number = "1.0"
        
        # 创建回滚版本
        return self.create_version(
            dataset_name=dataset_name,
            version_number=new_version_number,
            data_records=target_data,
            description=f"Rollback to version {target_version.version_number}",
            created_by=created_by,
            parent_version=target_version_id,
            metadata={
                'rollback': True,
                'rollback_target': target_version_id,
                'rollback_target_version': target_version.version_number
            }
        )
    
    def _calculate_changes(
        self, 
        dataset_name: str, 
        parent_version_id: str, 
        current_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算相对于父版本的变更"""
        
        try:
            parent_data = self.get_version_data(parent_version_id)
            
            # 创建记录索引
            parent_records = {self._record_hash(record): record for record in parent_data}
            current_records = {self._record_hash(record): record for record in current_data}
            
            # 计算差异
            added = len(set(current_records.keys()) - set(parent_records.keys()))
            removed = len(set(parent_records.keys()) - set(current_records.keys()))
            common = len(set(parent_records.keys()) & set(current_records.keys()))
            
            # 检查修改的记录数
            modified = 0
            for hash_key in set(parent_records.keys()) & set(current_records.keys()):
                if parent_records[hash_key] != current_records[hash_key]:
                    modified += 1
            
            return {
                'added_records': added,
                'removed_records': removed,
                'modified_records': modified,
                'unchanged_records': common - modified,
                'total_changes': added + removed + modified,
                'parent_version': parent_version_id,
                'change_percentage': ((added + removed + modified) / len(parent_records) * 100) if parent_records else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating changes: {e}")
            return {
                'error': str(e),
                'parent_version': parent_version_id
            }
    
    def _record_hash(self, record: Dict[str, Any]) -> str:
        """计算记录哈希"""
        # 移除可能变化的元数据字段
        cleaned_record = {}
        skip_fields = ['_metadata', '_analysis', '_structure', 'processed_at', 'updated_at']
        
        for key, value in record.items():
            if key not in skip_fields:
                cleaned_record[key] = value
        
        content = json.dumps(cleaned_record, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def export_version(self, version_id: str, export_path: str, format: str = 'jsonl') -> str:
        """导出版本数据"""
        
        data = self.get_version_data(version_id)
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(export_file, 'w', encoding='utf-8') as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        elif format == 'json':
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            
            # 展开嵌套字典
            flattened_data = []
            for record in data:
                flat_record = self._flatten_dict(record)
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(export_file, index=False, encoding='utf-8')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported version {version_id} to {export_file}")
        return str(export_file)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """展开嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # 对列表进行简单处理
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_version_history(self, dataset_name: str, version_id: str) -> List[Dict[str, Any]]:
        """获取版本历史链"""
        
        with self.SessionLocal() as db:
            history = []
            current_id = version_id
            
            while current_id:
                version = db.query(DataVersionModel).filter(
                    DataVersionModel.version_id == current_id
                ).first()
                
                if not version:
                    break
                
                history.append({
                    'version_id': version.version_id,
                    'version_number': version.version_number,
                    'description': version.description,
                    'created_by': version.created_by,
                    'created_at': version.created_at,
                    'parent_version': version.parent_version,
                    'record_count': version.record_count,
                    'size_bytes': version.size_bytes,
                    'changes_summary': version.changes_summary
                })
                
                current_id = version.parent_version
            
            return history
    
    def delete_version(self, version_id: str, remove_files: bool = True) -> bool:
        """删除版本"""
        
        with self.SessionLocal() as db:
            version = db.query(DataVersionModel).filter(
                DataVersionModel.version_id == version_id
            ).first()
            
            if not version:
                return False
            
            # 检查是否有子版本依赖
            children = db.query(DataVersionModel).filter(
                DataVersionModel.parent_version == version_id
            ).all()
            
            if children:
                raise ValueError(f"Cannot delete version {version_id}: has {len(children)} dependent versions")
            
            # 删除文件
            if remove_files and version.data_path:
                data_path = Path(version.data_path)
                if data_path.exists():
                    data_path.unlink()
                    
                    # 尝试删除空的版本目录
                    version_dir = data_path.parent
                    try:
                        if version_dir.exists() and not any(version_dir.iterdir()):
                            version_dir.rmdir()
                    except OSError:
                        pass  # 目录可能不为空
            
            # 删除数据库记录
            db.delete(version)
            db.commit()
            
            self.logger.info(f"Deleted version {version_id}")
            return True
    
    def get_version_statistics(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """获取版本统计信息"""
        
        with self.SessionLocal() as db:
            query = db.query(DataVersionModel)
            
            if dataset_name:
                query = query.filter(DataVersionModel.dataset_name == dataset_name)
            
            versions = query.all()
            
            if not versions:
                return {
                    'total_versions': 0,
                    'total_datasets': 0,
                    'total_records': 0,
                    'total_size_bytes': 0
                }
            
            # 基础统计
            total_versions = len(versions)
            total_datasets = len(set(v.dataset_name for v in versions))
            total_records = sum(v.record_count or 0 for v in versions)
            total_size = sum(v.size_bytes or 0 for v in versions)
            
            # 按数据集分组
            by_dataset = {}
            for version in versions:
                dataset = version.dataset_name
                if dataset not in by_dataset:
                    by_dataset[dataset] = {
                        'version_count': 0,
                        'total_records': 0,
                        'total_size': 0,
                        'latest_version': None
                    }
                
                by_dataset[dataset]['version_count'] += 1
                by_dataset[dataset]['total_records'] += version.record_count or 0
                by_dataset[dataset]['total_size'] += version.size_bytes or 0
                
                if (not by_dataset[dataset]['latest_version'] or 
                    version.created_at > by_dataset[dataset]['latest_version']['created_at']):
                    by_dataset[dataset]['latest_version'] = {
                        'version_id': version.version_id,
                        'version_number': version.version_number,
                        'created_at': version.created_at
                    }
            
            # 时间统计
            creation_dates = [v.created_at for v in versions if v.created_at]
            time_stats = {}
            if creation_dates:
                time_stats = {
                    'earliest_version': min(creation_dates),
                    'latest_version': max(creation_dates),
                    'versions_last_30_days': len([
                        d for d in creation_dates 
                        if (utc_now() - d.replace(tzinfo=timezone.utc)).days <= 30
                    ])
                }
            
            return {
                'total_versions': total_versions,
                'total_datasets': total_datasets,
                'total_records': total_records,
                'total_size_bytes': total_size,
                'average_records_per_version': total_records / total_versions if total_versions > 0 else 0,
                'average_size_per_version': total_size / total_versions if total_versions > 0 else 0,
                'by_dataset': by_dataset,
                'time_statistics': time_stats
            }