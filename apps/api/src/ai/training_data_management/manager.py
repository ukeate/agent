"""
数据管理器模块

提供数据收集、预处理的统一管理接口
"""

import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import (
    Base, DataSource, DataRecord, 
    DataSourceModel, DataRecordModel
)
from .collectors import CollectorFactory
from .preprocessor import DataPreprocessor


class DataCollectionManager:
    """数据收集管理器"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.preprocessor = DataPreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def register_data_source(self, source: DataSource) -> str:
        """注册数据源"""
        
        with self.SessionLocal() as db:
            # 检查是否已存在
            existing = db.query(DataSourceModel).filter(
                DataSourceModel.source_id == source.source_id
            ).first()
            
            if existing:
                # 更新现有数据源
                existing.name = source.name
                existing.description = source.description
                existing.config = source.config
                existing.is_active = source.is_active
                existing.updated_at = utc_now()
                db.commit()
                db.refresh(existing)
                
                self.logger.info(f"Updated existing data source: {source.name}")
                return str(existing.id)
            else:
                # 创建新数据源
                db_source = DataSourceModel(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    name=source.name,
                    description=source.description,
                    config=source.config,
                    is_active=source.is_active
                )
                
                db.add(db_source)
                db.commit()
                db.refresh(db_source)
                
                self.logger.info(f"Registered new data source: {source.name}")
                return str(db_source.id)
    
    async def collect_from_source(
        self, 
        source_id: str,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """从指定数据源收集数据"""
        
        with self.SessionLocal() as db:
            db_source = db.query(DataSourceModel).filter(
                DataSourceModel.source_id == source_id
            ).first()
            
            if not db_source or not db_source.is_active:
                raise ValueError(f"Data source {source_id} not found or inactive")
            
            # 构造数据源对象
            source = DataSource(
                source_id=db_source.source_id,
                source_type=db_source.source_type,
                name=db_source.name,
                description=db_source.description,
                config=db_source.config,
                is_active=db_source.is_active,
                created_at=db_source.created_at
            )
            
            # 创建收集器
            collector = CollectorFactory.create_collector(source)
            
            # 初始化统计信息
            collection_stats = {
                'source_id': source_id,
                'source_name': source.name,
                'total_collected': 0,
                'total_processed': 0,
                'total_stored': 0,
                'errors': 0,
                'start_time': utc_now(),
                'end_time': None,
                'processing_errors': []
            }
            
            try:
                self.logger.info(f"Starting data collection from source: {source.name}")
                
                collected_records = []
                batch_size = source.config.get('batch_size', 100)
                
                # 收集数据
                async for record in collector.collect_data():
                    collected_records.append(record)
                    collection_stats['total_collected'] += 1
                    
                    # 批量处理
                    if len(collected_records) >= batch_size:
                        await self._process_and_store_batch(
                            collected_records, 
                            collection_stats,
                            preprocessing_config
                        )
                        collected_records = []
                
                # 处理剩余数据
                if collected_records:
                    await self._process_and_store_batch(
                        collected_records, 
                        collection_stats,
                        preprocessing_config
                    )
                
                collection_stats['end_time'] = utc_now()
                collection_stats['duration'] = (
                    collection_stats['end_time'] - collection_stats['start_time']
                ).total_seconds()
                
                self.logger.info(f"Data collection completed for source: {source.name}")
                self.logger.info(f"Stats: collected={collection_stats['total_collected']}, "
                               f"processed={collection_stats['total_processed']}, "
                               f"stored={collection_stats['total_stored']}, "
                               f"errors={collection_stats['errors']}")
                
                return collection_stats
                
            except Exception as e:
                collection_stats['end_time'] = utc_now()
                collection_stats['errors'] += 1
                collection_stats['processing_errors'].append({
                    'error': str(e),
                    'timestamp': utc_now().isoformat()
                })
                self.logger.error(f"Error during data collection from {source.name}: {e}")
                raise
    
    async def _process_and_store_batch(
        self, 
        records: List[DataRecord], 
        stats: Dict[str, Any],
        preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """处理并存储数据批次"""
        
        try:
            # 预处理数据
            preprocessing_rules = None
            if preprocessing_config:
                preprocessing_rules = preprocessing_config.get('rules')
            
            processed_records = await self.preprocessor.preprocess_data(
                records, 
                rules=preprocessing_rules,
                custom_config=preprocessing_config
            )
            stats['total_processed'] += len(processed_records)
            
            # 存储到数据库
            with self.SessionLocal() as db:
                for record in processed_records:
                    try:
                        # 检查是否已存在相同record_id
                        existing = db.query(DataRecordModel).filter(
                            DataRecordModel.record_id == record.record_id
                        ).first()
                        
                        if existing:
                            # 更新现有记录
                            existing.processed_data = record.processed_data
                            existing.metadata = record.metadata
                            existing.quality_score = record.quality_score
                            existing.status = record.status
                            existing.processed_at = record.processed_at
                            existing.updated_at = utc_now()
                        else:
                            # 创建新记录
                            db_record = DataRecordModel(
                                record_id=record.record_id,
                                source_id=record.source_id,
                                raw_data=record.raw_data,
                                processed_data=record.processed_data,
                                metadata=record.metadata,
                                quality_score=record.quality_score,
                                status=record.status,
                                created_at=record.created_at,
                                processed_at=record.processed_at
                            )
                            db.add(db_record)
                        
                        stats['total_stored'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error storing record {record.record_id}: {e}")
                        stats['errors'] += 1
                        stats['processing_errors'].append({
                            'record_id': record.record_id,
                            'error': str(e),
                            'timestamp': utc_now().isoformat()
                        })
                
                db.commit()
        
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            stats['errors'] += len(records)
            stats['processing_errors'].append({
                'batch_error': str(e),
                'batch_size': len(records),
                'timestamp': utc_now().isoformat()
            })
    
    def get_data_records(
        self, 
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取数据记录"""
        
        with self.SessionLocal() as db:
            query = db.query(DataRecordModel)
            
            if source_id:
                query = query.filter(DataRecordModel.source_id == source_id)
            
            if status:
                query = query.filter(DataRecordModel.status == status)
            
            if min_quality_score is not None:
                query = query.filter(DataRecordModel.quality_score >= min_quality_score)
            
            query = query.order_by(DataRecordModel.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            records = query.all()
            
            return [
                {
                    'id': str(record.id),
                    'record_id': record.record_id,
                    'source_id': record.source_id,
                    'raw_data': record.raw_data,
                    'processed_data': record.processed_data,
                    'metadata': record.metadata,
                    'quality_score': record.quality_score,
                    'status': record.status,
                    'created_at': record.created_at,
                    'processed_at': record.processed_at,
                    'updated_at': record.updated_at
                }
                for record in records
            ]
    
    def get_collection_statistics(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """获取收集统计信息"""
        
        with self.SessionLocal() as db:
            query = db.query(DataRecordModel)
            
            if source_id:
                query = query.filter(DataRecordModel.source_id == source_id)
            
            total_records = query.count()
            
            # 按状态统计
            status_stats = {}
            for status in ['raw', 'processed', 'validated', 'rejected', 'error']:
                count = query.filter(DataRecordModel.status == status).count()
                status_stats[status] = count
            
            # 质量分数统计
            from sqlalchemy import func
            quality_stats = db.query(
                func.avg(DataRecordModel.quality_score).label('average'),
                func.min(DataRecordModel.quality_score).label('minimum'),
                func.max(DataRecordModel.quality_score).label('maximum'),
                func.count(DataRecordModel.quality_score).label('count')
            ).filter(DataRecordModel.quality_score.isnot(None))
            
            if source_id:
                quality_stats = quality_stats.filter(DataRecordModel.source_id == source_id)
            
            quality_result = quality_stats.first()
            
            # 按数据源统计
            source_stats = {}
            if not source_id:  # 只有在不指定source_id时才统计各数据源
                source_query = db.query(
                    DataRecordModel.source_id,
                    func.count(DataRecordModel.id).label('count'),
                    func.avg(DataRecordModel.quality_score).label('avg_quality')
                ).group_by(DataRecordModel.source_id).all()
                
                for source_stat in source_query:
                    source_stats[source_stat.source_id] = {
                        'record_count': source_stat.count,
                        'average_quality': float(source_stat.avg_quality) if source_stat.avg_quality else 0.0
                    }
            
            # 时间统计
            time_stats = db.query(
                func.min(DataRecordModel.created_at).label('first_record'),
                func.max(DataRecordModel.created_at).label('latest_record')
            )
            
            if source_id:
                time_stats = time_stats.filter(DataRecordModel.source_id == source_id)
            
            time_result = time_stats.first()
            
            return {
                'total_records': total_records,
                'status_distribution': status_stats,
                'quality_stats': {
                    'average': float(quality_result.average) if quality_result.average else 0.0,
                    'minimum': float(quality_result.minimum) if quality_result.minimum else 0.0,
                    'maximum': float(quality_result.maximum) if quality_result.maximum else 0.0,
                    'records_with_quality_score': quality_result.count or 0
                },
                'source_distribution': source_stats,
                'time_range': {
                    'first_record': time_result.first_record.isoformat() if time_result.first_record else None,
                    'latest_record': time_result.latest_record.isoformat() if time_result.latest_record else None
                }
            }
    
    def list_data_sources(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """列出数据源"""
        try:
            with self.SessionLocal() as db:
                query = db.query(DataSourceModel)
                
                if active_only:
                    query = query.filter(DataSourceModel.is_active == True)
                
                sources = query.order_by(DataSourceModel.created_at.desc()).all()
                
                return [
                    {
                        'id': str(source.id),
                        'source_id': source.source_id,
                        'source_type': source.source_type,
                        'name': source.name,
                        'description': source.description,
                        'config': source.config,
                        'is_active': source.is_active,
                        'created_at': source.created_at,
                        'updated_at': source.updated_at
                    }
                    for source in sources
                ]
        except Exception as e:
            self.logger.error(f"Error listing data sources: {e}")
            return []
    
    def update_data_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """更新数据源配置"""
        
        with self.SessionLocal() as db:
            db_source = db.query(DataSourceModel).filter(
                DataSourceModel.source_id == source_id
            ).first()
            
            if not db_source:
                return False
            
            # 更新允许的字段
            allowed_fields = ['name', 'description', 'config', 'is_active']
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(db_source, field, value)
            
            db_source.updated_at = utc_now()
            db.commit()
            
            self.logger.info(f"Updated data source: {source_id}")
            return True
    
    def delete_data_source(self, source_id: str) -> bool:
        """删除数据源（软删除）"""
        
        with self.SessionLocal() as db:
            db_source = db.query(DataSourceModel).filter(
                DataSourceModel.source_id == source_id
            ).first()
            
            if not db_source:
                return False
            
            # 软删除：设置为不活跃
            db_source.is_active = False
            db_source.updated_at = utc_now()
            db.commit()
            
            self.logger.info(f"Deactivated data source: {source_id}")
            return True
    
    def get_processing_queue_status(self) -> Dict[str, Any]:
        """获取处理队列状态"""
        
        with self.SessionLocal() as db:
            # 统计各状态的记录数
            status_counts = {}
            for status in ['raw', 'processed', 'validated', 'rejected', 'error']:
                count = db.query(DataRecordModel).filter(
                    DataRecordModel.status == status
                ).count()
                status_counts[status] = count
            
            # 获取最近的处理活动
            recent_records = db.query(DataRecordModel).filter(
                DataRecordModel.processed_at.isnot(None)
            ).order_by(
                DataRecordModel.processed_at.desc()
            ).limit(10).all()
            
            return {
                'queue_status': status_counts,
                'pending_processing': status_counts.get('raw', 0),
                'recent_activity': [
                    {
                        'record_id': record.record_id,
                        'source_id': record.source_id,
                        'status': record.status,
                        'quality_score': record.quality_score,
                        'processed_at': record.processed_at.isoformat() if record.processed_at else None
                    }
                    for record in recent_records
                ],
                'total_records': sum(status_counts.values())
            }
    
    async def reprocess_records(
        self, 
        record_ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """重新处理记录"""
        
        with self.SessionLocal() as db:
            query = db.query(DataRecordModel)
            
            if record_ids:
                query = query.filter(DataRecordModel.record_id.in_(record_ids))
            if source_id:
                query = query.filter(DataRecordModel.source_id == source_id)
            if status_filter:
                query = query.filter(DataRecordModel.status == status_filter)
            
            db_records = query.all()
            
            # 转换为DataRecord对象
            records_to_process = []
            for db_record in db_records:
                record = DataRecord(
                    record_id=db_record.record_id,
                    source_id=db_record.source_id,
                    raw_data=db_record.raw_data,
                    processed_data=db_record.processed_data,
                    metadata=db_record.metadata,
                    quality_score=db_record.quality_score,
                    status='raw',  # 重置状态进行重新处理
                    created_at=db_record.created_at
                )
                records_to_process.append(record)
            
            # 重新处理
            stats = {
                'total_reprocessed': 0,
                'successful': 0,
                'errors': 0,
                'start_time': utc_now()
            }
            
            if records_to_process:
                await self._process_and_store_batch(
                    records_to_process, 
                    stats,
                    preprocessing_config
                )
            
            stats['end_time'] = utc_now()
            stats['total_reprocessed'] = len(records_to_process)
            stats['successful'] = stats['total_stored']
            
            self.logger.info(f"Reprocessed {stats['total_reprocessed']} records, "
                           f"{stats['successful']} successful, {stats['errors']} errors")
            
            return stats