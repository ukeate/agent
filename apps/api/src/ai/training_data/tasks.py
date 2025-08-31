"""
训练数据管理系统Celery任务

这个模块定义了所有的后台任务，包括：
- 数据收集任务
- 数据预处理任务
- 批量标注任务
- 版本创建任务
- 数据导出任务
"""

import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import List, Dict, Any, Optional
from celery import Celery
from celery.exceptions import Retry
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ...core.config import get_settings
from .models import DataSourceModel, DataRecordModel, AnnotationTaskModel
from .core import DataSource, DataFilter, ExportFormat, AssignmentStrategy
from .collectors import CollectorFactory, CollectionStats
from .preprocessing import DataPreprocessor
from .annotation import AnnotationManager
from .version_manager import DataVersionManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Celery应用
celery_app = Celery(
    "training_data",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["ai.training_data.tasks"]
)

# 配置Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30分钟
    task_soft_time_limit=25 * 60,  # 25分钟
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# 创建异步数据库引擎
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True
)

async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db_session():
    """获取数据库会话"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


def run_async_task(async_func):
    """运行异步任务的装饰器"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def collect_data_task(self, source_id: str, processing_rules: List[str] = None, batch_size: int = 100):
    """数据收集任务"""
    @run_async_task
    async def _collect_data():
        stats = CollectionStats(start_time=utc_now())
        
        try:
            async for session in get_db_session():
                # 获取数据源信息
                from sqlalchemy import select
                stmt = select(DataSourceModel).where(DataSourceModel.source_id == source_id)
                result = await session.execute(stmt)
                source_model = result.scalar_one()
                
                # 创建DataSource对象
                from .models import SourceType
                data_source = DataSource(
                    source_id=source_model.source_id,
                    source_type=SourceType(source_model.source_type),
                    name=source_model.name,
                    description=source_model.description or "",
                    config=source_model.config
                )
                
                # 创建收集器和预处理器
                collector = CollectorFactory.create_collector(data_source)
                preprocessor = DataPreprocessor(session) if processing_rules else None
                
                records_batch = []
                
                # 收集数据
                async for record in collector.collect_data():
                    records_batch.append(record)
                    stats.total_collected += 1
                    
                    # 更新任务进度
                    if stats.total_collected % 50 == 0:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'current': stats.total_collected,
                                'status': f'Collected {stats.total_collected} records'
                            }
                        )
                    
                    if len(records_batch) >= batch_size:
                        # 预处理
                        if preprocessor and processing_rules:
                            try:
                                records_batch = await preprocessor.preprocess_records(
                                    records_batch, processing_rules
                                )
                                stats.total_processed += len(records_batch)
                            except Exception as e:
                                logger.error(f"Preprocessing error: {e}")
                                stats.errors += 1
                        
                        # 保存到数据库
                        for record in records_batch:
                            try:
                                record_model = DataRecordModel(
                                    record_id=record.record_id,
                                    source_id=record.source_id,
                                    raw_data=record.raw_data,
                                    processed_data=record.processed_data,
                                    metadata=record.metadata,
                                    quality_score=record.quality_score,
                                    status=record.status,
                                    processed_at=record.processed_at
                                )
                                session.add(record_model)
                                stats.total_stored += 1
                            except Exception as e:
                                logger.error(f"Database save error: {e}")
                                stats.errors += 1
                        
                        await session.commit()
                        records_batch = []
                
                # 处理剩余记录
                if records_batch:
                    if preprocessor and processing_rules:
                        try:
                            records_batch = await preprocessor.preprocess_records(
                                records_batch, processing_rules
                            )
                            stats.total_processed += len(records_batch)
                        except Exception as e:
                            logger.error(f"Final preprocessing error: {e}")
                            stats.errors += 1
                    
                    for record in records_batch:
                        try:
                            record_model = DataRecordModel(
                                record_id=record.record_id,
                                source_id=record.source_id,
                                raw_data=record.raw_data,
                                processed_data=record.processed_data,
                                metadata=record.metadata,
                                quality_score=record.quality_score,
                                status=record.status,
                                processed_at=record.processed_at
                            )
                            session.add(record_model)
                            stats.total_stored += 1
                        except Exception as e:
                            logger.error(f"Final database save error: {e}")
                            stats.errors += 1
                    
                    await session.commit()
                
                stats.end_time = utc_now()
                
                return {
                    'source_id': source_id,
                    'total_collected': stats.total_collected,
                    'total_processed': stats.total_processed,
                    'total_stored': stats.total_stored,
                    'errors': stats.errors,
                    'duration_seconds': (stats.end_time - stats.start_time).total_seconds(),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Collection task failed: {e}")
            stats.errors += 1
            raise self.retry(countdown=60, exc=e)
    
    return _collect_data()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def preprocess_records_task(
    self,
    record_ids: List[str],
    processing_rules: List[str]
):
    """批量预处理记录任务"""
    @run_async_task
    async def _preprocess_records():
        try:
            async for session in get_db_session():
                # 获取记录
                from sqlalchemy import select
                stmt = select(DataRecordModel).where(
                    DataRecordModel.record_id.in_(record_ids)
                )
                result = await session.execute(stmt)
                record_models = result.scalars().all()
                
                # 转换为DataRecord对象
                from .core import DataRecord
                records = []
                for model in record_models:
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
                
                # 预处理
                preprocessor = DataPreprocessor(session)
                processed_records = await preprocessor.preprocess_records(
                    records, processing_rules
                )
                
                # 更新数据库
                processed_count = 0
                for processed_record in processed_records:
                    for model in record_models:
                        if model.record_id == processed_record.record_id:
                            model.processed_data = processed_record.processed_data
                            model.quality_score = processed_record.quality_score
                            model.status = processed_record.status
                            model.processed_at = processed_record.processed_at
                            processed_count += 1
                            break
                    
                    # 更新进度
                    if processed_count % 10 == 0:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'current': processed_count,
                                'total': len(processed_records),
                                'status': f'Processed {processed_count}/{len(processed_records)} records'
                            }
                        )
                
                await session.commit()
                
                return {
                    'processed_count': processed_count,
                    'total_records': len(record_ids),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Preprocessing task failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _preprocess_records()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def create_annotation_task_task(
    self,
    task_data: Dict[str, Any],
    assignment_strategy: str = "balanced"
):
    """创建标注任务的后台任务"""
    @run_async_task
    async def _create_annotation_task():
        try:
            async for session in get_db_session():
                # 转换任务数据
                from .core import AnnotationTask
                from .models import AnnotationTaskType, AnnotationStatus
                
                annotation_task = AnnotationTask(
                    task_id=task_data['task_id'],
                    name=task_data['name'],
                    description=task_data['description'],
                    task_type=AnnotationTaskType(task_data['task_type']),
                    data_records=task_data['data_records'],
                    annotation_schema=task_data['annotation_schema'],
                    guidelines=task_data['guidelines'],
                    assignees=task_data.get('assignees', []),
                    created_by=task_data['created_by'],
                    status=AnnotationStatus.PENDING,
                    deadline=task_data.get('deadline')
                )
                
                # 创建任务
                annotation_manager = AnnotationManager(session)
                strategy = AssignmentStrategy(assignment_strategy)
                task_id = await annotation_manager.create_task(annotation_task, strategy)
                
                return {
                    'task_id': task_id,
                    'assigned_records': len(task_data['data_records']),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Annotation task creation failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _create_annotation_task()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def create_version_task(
    self,
    dataset_name: str,
    version_number: str,
    description: str,
    created_by: str,
    data_filter: Optional[Dict[str, Any]] = None,
    parent_version: Optional[str] = None
):
    """创建数据版本任务"""
    @run_async_task
    async def _create_version():
        try:
            async for session in get_db_session():
                version_manager = DataVersionManager(session)
                
                # 转换过滤条件
                filter_obj = None
                if data_filter:
                    from .core import DataFilter, DataStatus
                    filter_obj = DataFilter(
                        source_id=data_filter.get('source_id'),
                        status=DataStatus(data_filter['status']) if data_filter.get('status') else None,
                        min_quality_score=data_filter.get('min_quality_score'),
                        date_from=data_filter.get('date_from'),
                        date_to=data_filter.get('date_to'),
                        limit=data_filter.get('limit', 1000),
                        offset=data_filter.get('offset', 0)
                    )
                
                version_id = await version_manager.create_version(
                    dataset_name=dataset_name,
                    version_number=version_number,
                    description=description,
                    created_by=created_by,
                    data_filter=filter_obj,
                    parent_version=parent_version
                )
                
                # 获取版本信息
                from sqlalchemy import select
                from .models import DataVersionModel
                stmt = select(DataVersionModel).where(
                    DataVersionModel.version_id == version_id
                )
                result = await session.execute(stmt)
                version = result.scalar_one()
                
                return {
                    'version_id': version_id,
                    'dataset_name': dataset_name,
                    'record_count': version.record_count,
                    'size_bytes': version.size_bytes,
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _create_version()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def export_version_task(
    self,
    version_id: str,
    export_format: str,
    output_filename: Optional[str] = None
):
    """导出版本数据任务"""
    @run_async_task
    async def _export_version():
        try:
            async for session in get_db_session():
                version_manager = DataVersionManager(session)
                
                format_enum = ExportFormat(export_format)
                output_path = await version_manager.export_version(
                    version_id=version_id,
                    export_format=format_enum,
                    output_path=output_filename
                )
                
                # 获取文件大小
                from pathlib import Path
                file_path = Path(output_path)
                file_size = file_path.stat().st_size if file_path.exists() else 0
                
                return {
                    'version_id': version_id,
                    'output_path': output_path,
                    'export_format': export_format,
                    'file_size_bytes': file_size,
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Export task failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _export_version()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def quality_assessment_task(self, task_id: str):
    """质量评估任务"""
    @run_async_task
    async def _quality_assessment():
        try:
            async for session in get_db_session():
                quality_controller = QualityController(session)
                report = await quality_controller.generate_quality_report(task_id)
                
                return {
                    'task_id': task_id,
                    'overall_score': report.overall_score,
                    'agreement_metrics': report.agreement_metrics,
                    'consistency_metrics': report.consistency_metrics,
                    'recommendations_count': len(report.recommendations),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _quality_assessment()


@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3})
def batch_annotation_import_task(
    self,
    task_id: str,
    annotations_data: List[Dict[str, Any]]
):
    """批量导入标注结果任务"""
    @run_async_task
    async def _batch_annotation_import():
        try:
            async for session in get_db_session():
                annotation_manager = AnnotationManager(session)
                
                imported_count = 0
                for annotation_data in annotations_data:
                    try:
                        from .core import Annotation
                        annotation = Annotation(
                            annotation_id=annotation_data['annotation_id'],
                            task_id=task_id,
                            record_id=annotation_data['record_id'],
                            annotator_id=annotation_data['annotator_id'],
                            annotation_data=annotation_data['annotation_data'],
                            confidence=annotation_data.get('confidence'),
                            time_spent=annotation_data.get('time_spent'),
                            status=annotation_data.get('status', 'submitted')
                        )
                        
                        await annotation_manager.submit_annotation(annotation)
                        imported_count += 1
                        
                        # 更新进度
                        if imported_count % 50 == 0:
                            self.update_state(
                                state='PROGRESS',
                                meta={
                                    'current': imported_count,
                                    'total': len(annotations_data),
                                    'status': f'Imported {imported_count}/{len(annotations_data)} annotations'
                                }
                            )
                            
                    except Exception as e:
                        logger.error(f"Failed to import annotation {annotation_data.get('annotation_id')}: {e}")
                
                return {
                    'task_id': task_id,
                    'imported_count': imported_count,
                    'total_annotations': len(annotations_data),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Batch annotation import failed: {e}")
            raise self.retry(countdown=60, exc=e)
    
    return _batch_annotation_import()


@celery_app.task(bind=True)
def cleanup_old_versions_task(self, days_old: int = 30):
    """清理旧版本数据任务"""
    @run_async_task
    async def _cleanup_old_versions():
        try:
            from datetime import timedelta
            cutoff_date = utc_now() - timedelta(days=days_old)
            
            async for session in get_db_session():
                version_manager = DataVersionManager(session)
                
                # 获取旧版本
                from sqlalchemy import select
                from .models import DataVersionModel
                stmt = select(DataVersionModel).where(
                    DataVersionModel.created_at < cutoff_date
                )
                result = await session.execute(stmt)
                old_versions = result.scalars().all()
                
                deleted_count = 0
                for version in old_versions:
                    try:
                        success = await version_manager.delete_version(version.version_id)
                        if success:
                            deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete version {version.version_id}: {e}")
                
                return {
                    'deleted_count': deleted_count,
                    'total_old_versions': len(old_versions),
                    'cutoff_date': cutoff_date.isoformat(),
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    return _cleanup_old_versions()


# 定期任务配置
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-old-versions': {
        'task': 'ai.training_data.tasks.cleanup_old_versions_task',
        'schedule': crontab(hour=2, minute=0),  # 每天凌晨2点执行
        'args': (30,)  # 清理30天前的版本
    },
}

celery_app.conf.timezone = 'UTC'


# 任务监控和健康检查

@celery_app.task
def health_check_task():
    """健康检查任务"""
    return {
        'status': 'healthy',
        'timestamp': utc_now().isoformat(),
        'celery_version': celery_app.version
    }


@celery_app.task
def get_task_stats():
    """获取任务统计信息"""
    inspect = celery_app.control.inspect()
    
    # 获取活跃任务
    active_tasks = inspect.active()
    
    # 获取调度任务
    scheduled_tasks = inspect.scheduled()
    
    # 获取保留任务
    reserved_tasks = inspect.reserved()
    
    stats = {
        'active_tasks': len(active_tasks.values()) if active_tasks else 0,
        'scheduled_tasks': len(scheduled_tasks.values()) if scheduled_tasks else 0,
        'reserved_tasks': len(reserved_tasks.values()) if reserved_tasks else 0,
        'timestamp': utc_now().isoformat()
    }
    
    return stats