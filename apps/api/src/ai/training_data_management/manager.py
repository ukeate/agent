"""
数据管理器模块

提供数据收集、预处理的统一管理接口
"""

from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Any, Optional, Callable, AsyncContextManager
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import (
    DataSource, DataRecord,
    DataSourceModel, DataRecordModel
)
from .collectors import CollectorFactory
from .preprocessor import DataPreprocessor
from src.core.database import get_db_session

from src.core.logging import get_logger
class DataCollectionManager:
    """数据收集管理器"""
    
    def __init__(
        self,
        session_factory: Callable[[], AsyncContextManager[AsyncSession]] = get_db_session,
    ):
        self.session_factory = session_factory
        self.preprocessor = DataPreprocessor()
        self.logger = get_logger(__name__)
    
    async def register_data_source(self, source: DataSource) -> str:
        """注册数据源"""
        
        async with self.session_factory() as db:
            # 检查是否已存在
            result = await db.execute(
                select(DataSourceModel).where(DataSourceModel.source_id == source.source_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新现有数据源
                existing.name = source.name
                existing.description = source.description
                existing.config = source.config
                existing.is_active = source.is_active
                existing.updated_at = utc_now()
                await db.commit()
                await db.refresh(existing)
                
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
                await db.commit()
                await db.refresh(db_source)
                
                self.logger.info(f"Registered new data source: {source.name}")
                return str(db_source.id)
    
    async def collect_from_source(
        self, 
        source_id: str,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """从指定数据源收集数据"""
        
        async with self.session_factory() as db:
            result = await db.execute(
                select(DataSourceModel).where(DataSourceModel.source_id == source_id)
            )
            db_source = result.scalar_one_or_none()
            
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
            
            stats.setdefault("total_processed", 0)
            stats.setdefault("total_stored", 0)
            stats.setdefault("errors", 0)
            stats.setdefault("processing_errors", [])

            processed_records = await self.preprocessor.preprocess_data(
                records,
                rules=preprocessing_rules,
                custom_config=preprocessing_config,
            )
            stats["total_processed"] += len(processed_records)

            async with self.session_factory() as db:
                record_ids = [record.record_id for record in processed_records]
                existing_map: Dict[str, DataRecordModel] = {}
                if record_ids:
                    result = await db.execute(
                        select(DataRecordModel).where(DataRecordModel.record_id.in_(record_ids))
                    )
                    existing_map = {item.record_id: item for item in result.scalars().all()}

                for record in processed_records:
                    try:
                        existing = existing_map.get(record.record_id)
                        if existing:
                            existing.processed_data = record.processed_data
                            existing.record_metadata = record.metadata
                            existing.quality_score = record.quality_score
                            existing.status = record.status
                            existing.processed_at = record.processed_at
                            existing.updated_at = utc_now()
                        else:
                            db_record = DataRecordModel(
                                record_id=record.record_id,
                                source_id=record.source_id,
                                raw_data=record.raw_data,
                                processed_data=record.processed_data,
                                record_metadata=record.metadata,
                                quality_score=record.quality_score,
                                status=record.status,
                                created_at=record.created_at,
                                processed_at=record.processed_at,
                            )
                            db.add(db_record)
                        stats["total_stored"] += 1
                    except Exception as e:
                        self.logger.error(f"Error storing record {record.record_id}: {e}")
                        stats["errors"] += 1
                        stats["processing_errors"].append(
                            {
                                "record_id": record.record_id,
                                "error": str(e),
                                "timestamp": utc_now().isoformat(),
                            }
                        )

                await db.commit()
        
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            stats['errors'] += len(records)
            stats['processing_errors'].append({
                'batch_error': str(e),
                'batch_size': len(records),
                'timestamp': utc_now().isoformat()
            })
    
    async def get_data_records(
        self, 
        record_ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取数据记录"""
        
        async with self.session_factory() as db:
            stmt = select(DataRecordModel)

            if record_ids:
                stmt = stmt.where(DataRecordModel.record_id.in_(record_ids))
            
            if source_id:
                stmt = stmt.where(DataRecordModel.source_id == source_id)
            
            if status:
                stmt = stmt.where(DataRecordModel.status == status)
            
            if min_quality_score is not None:
                stmt = stmt.where(DataRecordModel.quality_score >= min_quality_score)
            
            stmt = stmt.order_by(DataRecordModel.created_at.desc()).offset(offset).limit(limit)
            
            result = await db.execute(stmt)
            records = result.scalars().all()
            
            return [
                {
                    'id': str(record.id),
                    'record_id': record.record_id,
                    'source_id': record.source_id,
                    'raw_data': record.raw_data,
                    'processed_data': record.processed_data,
                    'metadata': record.record_metadata,
                    'quality_score': record.quality_score,
                    'status': record.status,
                    'created_at': record.created_at,
                    'processed_at': record.processed_at,
                    'updated_at': record.updated_at
                }
                for record in records
            ]
    
    async def get_collection_statistics(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """获取收集统计信息"""
        
        async with self.session_factory() as db:
            filters = []
            if source_id:
                filters.append(DataRecordModel.source_id == source_id)

            total_stmt = select(func.count(DataRecordModel.id))
            if filters:
                total_stmt = total_stmt.where(*filters)
            total_result = await db.execute(total_stmt)
            total_records = total_result.scalar_one() or 0

            status_stats: Dict[str, int] = {}
            for status in ['raw', 'processed', 'validated', 'rejected', 'error']:
                count_result = await db.execute(
                    select(func.count(DataRecordModel.id)).where(
                        DataRecordModel.status == status,
                        *filters,
                    )
                )
                status_stats[status] = int(count_result.scalar_one() or 0)

            quality_stmt = select(
                func.avg(DataRecordModel.quality_score).label('average'),
                func.min(DataRecordModel.quality_score).label('minimum'),
                func.max(DataRecordModel.quality_score).label('maximum'),
                func.count(DataRecordModel.quality_score).label('count'),
            ).where(DataRecordModel.quality_score.isnot(None))
            if source_id:
                quality_stmt = quality_stmt.where(DataRecordModel.source_id == source_id)
            quality_result = (await db.execute(quality_stmt)).first()

            source_stats: Dict[str, Dict[str, Any]] = {}
            if not source_id:
                source_query = await db.execute(
                    select(
                        DataRecordModel.source_id,
                        func.count(DataRecordModel.id).label('count'),
                        func.avg(DataRecordModel.quality_score).label('avg_quality'),
                    ).group_by(DataRecordModel.source_id)
                )
                for source_stat in source_query.all():
                    source_stats[source_stat.source_id] = {
                        'record_count': source_stat.count,
                        'average_quality': float(source_stat.avg_quality) if source_stat.avg_quality else 0.0,
                    }

            time_stmt = select(
                func.min(DataRecordModel.created_at).label('first_record'),
                func.max(DataRecordModel.created_at).label('latest_record'),
            )
            if source_id:
                time_stmt = time_stmt.where(DataRecordModel.source_id == source_id)
            time_result = (await db.execute(time_stmt)).first()

            average = quality_result.average if quality_result else 0
            minimum = quality_result.minimum if quality_result else 0
            maximum = quality_result.maximum if quality_result else 0
            count = quality_result.count if quality_result else 0

            return {
                'total_records': total_records,
                'status_distribution': status_stats,
                'quality_stats': {
                    'average': float(average) if average else 0.0,
                    'minimum': float(minimum) if minimum else 0.0,
                    'maximum': float(maximum) if maximum else 0.0,
                    'records_with_quality_score': count or 0,
                },
                'source_distribution': source_stats,
                'time_range': {
                    'first_record': time_result.first_record.isoformat() if time_result and time_result.first_record else None,
                    'latest_record': time_result.latest_record.isoformat() if time_result and time_result.latest_record else None,
                },
            }
    
    async def list_data_sources(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """列出数据源"""
        try:
            async with self.session_factory() as db:
                stmt = select(DataSourceModel)
                if active_only:
                    stmt = stmt.where(DataSourceModel.is_active == True)
                stmt = stmt.order_by(DataSourceModel.created_at.desc())
                result = await db.execute(stmt)
                sources = result.scalars().all()
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
                        'updated_at': source.updated_at,
                    }
                    for source in sources
                ]
        except Exception as e:
            self.logger.error(f"Error listing data sources: {e}")
            return []
    
    async def update_data_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """更新数据源配置"""
        
        async with self.session_factory() as db:
            result = await db.execute(
                select(DataSourceModel).where(DataSourceModel.source_id == source_id)
            )
            db_source = result.scalar_one_or_none()
            
            if not db_source:
                return False
            
            # 更新允许的字段
            allowed_fields = ['name', 'description', 'config', 'is_active']
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(db_source, field, value)
            
            db_source.updated_at = utc_now()
            await db.commit()
            
            self.logger.info(f"Updated data source: {source_id}")
            return True
    
    async def delete_data_source(self, source_id: str) -> bool:
        """删除数据源（软删除）"""
        
        async with self.session_factory() as db:
            result = await db.execute(
                select(DataSourceModel).where(DataSourceModel.source_id == source_id)
            )
            db_source = result.scalar_one_or_none()
            
            if not db_source:
                return False
            
            # 软删除：设置为不活跃
            db_source.is_active = False
            db_source.updated_at = utc_now()
            await db.commit()
            
            self.logger.info(f"Deactivated data source: {source_id}")
            return True
    
    async def get_processing_queue_status(self) -> Dict[str, Any]:
        """获取处理队列状态"""
        
        async with self.session_factory() as db:
            status_counts: Dict[str, int] = {}
            for status in ['raw', 'processed', 'validated', 'rejected', 'error']:
                count_result = await db.execute(
                    select(func.count(DataRecordModel.id)).where(DataRecordModel.status == status)
                )
                status_counts[status] = int(count_result.scalar_one() or 0)

            recent_result = await db.execute(
                select(DataRecordModel)
                .where(DataRecordModel.processed_at.isnot(None))
                .order_by(DataRecordModel.processed_at.desc())
                .limit(10)
            )
            recent_records = recent_result.scalars().all()

            return {
                'queue_status': status_counts,
                'pending_processing': status_counts.get('raw', 0),
                'recent_activity': [
                    {
                        'record_id': record.record_id,
                        'source_id': record.source_id,
                        'status': record.status,
                        'quality_score': record.quality_score,
                        'processed_at': record.processed_at.isoformat() if record.processed_at else None,
                    }
                    for record in recent_records
                ],
                'total_records': sum(status_counts.values()),
            }
    
    async def reprocess_records(
        self, 
        record_ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        status_filter: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """重新处理记录"""
        
        async with self.session_factory() as db:
            stmt = select(DataRecordModel)
            if record_ids:
                stmt = stmt.where(DataRecordModel.record_id.in_(record_ids))
            if source_id:
                stmt = stmt.where(DataRecordModel.source_id == source_id)
            if status_filter:
                stmt = stmt.where(DataRecordModel.status == status_filter)
            result = await db.execute(stmt)
            db_records = result.scalars().all()
            
            # 转换为DataRecord对象
            records_to_process = []
            for db_record in db_records:
                record = DataRecord(
                    record_id=db_record.record_id,
                    source_id=db_record.source_id,
                    raw_data=db_record.raw_data,
                    processed_data=db_record.processed_data,
                    metadata=db_record.record_metadata,
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
