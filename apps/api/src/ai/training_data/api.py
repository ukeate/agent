"""
训练数据管理系统FastAPI路由

这个模块提供完整的REST API端点，包括：
- 数据源管理
- 数据收集和处理
- 标注任务管理
- 版本控制
- 数据导出
"""

from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db, get_db_session
from .models import SourceType, DataStatus, AnnotationTaskType, AnnotationStatus
from .core import (

    DataSource, DataRecord, AnnotationTask, Annotation, DataVersion,
    DataFilter, ExportFormat, AssignmentStrategy, ConflictResolution
)
from .collectors import CollectorFactory, CollectionStats
from .preprocessing import DataPreprocessor
from .annotation import AnnotationManager, QualityController
from .version_manager import DataVersionManager

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/training-data", tags=["training-data"])

# Pydantic模型定义

class DataSourceCreate(BaseModel):
    source_type: SourceType
    name: str
    description: str
    config: Dict[str, Any]

class DataSourceResponse(BaseModel):
    source_id: str
    source_type: SourceType
    name: str
    description: str
    config: Dict[str, Any]
    is_active: bool
    created_at: datetime

class DataRecordResponse(BaseModel):
    record_id: str
    source_id: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    status: DataStatus
    created_at: datetime
    processed_at: Optional[datetime] = None

class AnnotationTaskCreate(BaseModel):
    name: str
    description: str
    task_type: AnnotationTaskType
    data_records: List[str]
    annotation_schema: Dict[str, Any]
    guidelines: str
    assignees: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None

class AnnotationTaskResponse(BaseModel):
    task_id: str
    name: str
    description: str
    task_type: AnnotationTaskType
    data_records: List[str]
    annotation_schema: Dict[str, Any]
    guidelines: str
    assignees: List[str]
    created_by: str
    status: AnnotationStatus
    created_at: datetime
    deadline: Optional[datetime] = None

class AnnotationCreate(BaseModel):
    task_id: str
    record_id: str
    annotation_data: Dict[str, Any]
    confidence: Optional[float] = None
    time_spent: Optional[int] = None

class AnnotationResponse(BaseModel):
    annotation_id: str
    task_id: str
    record_id: str
    annotator_id: str
    annotation_data: Dict[str, Any]
    confidence: Optional[float] = None
    time_spent: Optional[int] = None
    status: str
    created_at: datetime

class VersionCreate(BaseModel):
    dataset_name: str
    version_number: str
    description: str
    data_filter: Optional[Dict[str, Any]] = None
    parent_version: Optional[str] = None

class VersionResponse(BaseModel):
    version_id: str
    dataset_name: str
    version_number: str
    description: str
    created_by: str
    parent_version: Optional[str] = None
    changes_summary: Dict[str, Any]
    metadata: Dict[str, Any]
    record_count: int
    size_bytes: int
    created_at: datetime

class CollectionJobCreate(BaseModel):
    source_id: str
    processing_rules: List[str] = Field(default_factory=list)
    batch_size: int = 100

class ExportJobCreate(BaseModel):
    version_id: str
    export_format: ExportFormat
    output_filename: Optional[str] = None

# 数据源管理端点

@router.post("/sources", response_model=DataSourceResponse)
async def create_data_source(
    source_data: DataSourceCreate,
    db: AsyncSession = Depends(get_db)
):
    """创建数据源"""
    import uuid
    from .collectors import CollectorFactory
    
    source_id = f"src_{uuid.uuid4().hex[:8]}"
    
    data_source = DataSource(
        source_id=source_id,
        source_type=source_data.source_type,
        name=source_data.name,
        description=source_data.description,
        config=source_data.config
    )
    
    # 验证配置
    try:
        collector = CollectorFactory.create_collector(data_source)
        # 这里可以添加配置验证逻辑
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid source configuration: {str(e)}")
    
    # 保存到数据库
    from .models import DataSourceModel
    source_model = DataSourceModel(
        source_id=source_id,
        source_type=source_data.source_type.value,
        name=source_data.name,
        description=source_data.description,
        config=source_data.config
    )
    
    db.add(source_model)
    await db.commit()
    
    return DataSourceResponse(
        source_id=source_id,
        source_type=source_data.source_type,
        name=source_data.name,
        description=source_data.description,
        config=source_data.config,
        is_active=True,
        created_at=source_model.created_at
    )

@router.get("/sources", response_model=List[DataSourceResponse])
async def list_data_sources(
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(True, description="Only return active sources")
):
    """获取数据源列表"""
    from sqlalchemy import select
    from .models import DataSourceModel
    
    query = select(DataSourceModel)
    if active_only:
        query = query.where(DataSourceModel.is_active == True)
    
    result = await db.execute(query)
    sources = result.scalars().all()
    
    return [
        DataSourceResponse(
            source_id=source.source_id,
            source_type=SourceType(source.source_type),
            name=source.name,
            description=source.description or "",
            config=source.config,
            is_active=source.is_active,
            created_at=source.created_at
        )
        for source in sources
    ]

@router.get("/sources/{source_id}", response_model=DataSourceResponse)
async def get_data_source(
    source_id: str,
    db: AsyncSession = Depends(get_db)
):
    """获取特定数据源"""
    from sqlalchemy import select
    from .models import DataSourceModel
    
    stmt = select(DataSourceModel).where(DataSourceModel.source_id == source_id)
    result = await db.execute(stmt)
    source = result.scalar_one_or_none()
    
    if not source:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    return DataSourceResponse(
        source_id=source.source_id,
        source_type=SourceType(source.source_type),
        name=source.name,
        description=source.description or "",
        config=source.config,
        is_active=source.is_active,
        created_at=source.created_at
    )

# 数据收集端点

@router.post("/collect/{source_id}")
async def start_collection_job(
    source_id: str,
    job_data: CollectionJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """启动数据收集任务"""
    from sqlalchemy import select
    from .models import DataSourceModel
    
    # 验证数据源存在
    stmt = select(DataSourceModel).where(DataSourceModel.source_id == source_id)
    result = await db.execute(stmt)
    source_model = result.scalar_one_or_none()
    
    if not source_model:
        raise HTTPException(status_code=404, detail="Data source not found")
    
    # 创建DataSource对象
    data_source = DataSource(
        source_id=source_model.source_id,
        source_type=SourceType(source_model.source_type),
        name=source_model.name,
        description=source_model.description or "",
        config=source_model.config
    )
    
    # 添加后台任务
    background_tasks.add_task(
        _run_collection_job,
        data_source,
        job_data.processing_rules,
        job_data.batch_size
    )
    
    return {"message": "Collection job started", "source_id": source_id}

async def _run_collection_job(
    data_source: DataSource,
    processing_rules: List[str],
    batch_size: int
):
    """运行数据收集任务"""
    from .models import DataRecordModel
    
    try:
        async with get_db_session() as db:
            # 创建收集器
            collector = CollectorFactory.create_collector(data_source)
            
            # 创建预处理器
            preprocessor = DataPreprocessor(db)
            
            # 收集数据
            records_batch = []
            async for record in collector.collect_data():
                records_batch.append(record)
                
                if len(records_batch) >= batch_size:
                    # 预处理
                    if processing_rules:
                        records_batch = await preprocessor.preprocess_records(
                            records_batch, processing_rules
                        )
                    
                    # 保存到数据库
                    for record in records_batch:
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
                        db.add(record_model)
                    
                    await db.commit()
                    records_batch = []
            
            # 处理剩余的记录
            if records_batch:
                if processing_rules:
                    records_batch = await preprocessor.preprocess_records(
                        records_batch, processing_rules
                    )
                
                for record in records_batch:
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
                    db.add(record_model)
                
                await db.commit()
    
    except Exception as e:
        logger.error("数据采集任务失败", error=str(e), exc_info=True)

@router.get("/records", response_model=List[DataRecordResponse])
async def list_records(
    db: AsyncSession = Depends(get_db),
    source_id: Optional[str] = Query(None),
    status: Optional[DataStatus] = Query(None),
    min_quality_score: Optional[float] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取数据记录列表"""
    from sqlalchemy import select
    from .models import DataRecordModel
    
    query = select(DataRecordModel)
    
    if source_id:
        query = query.where(DataRecordModel.source_id == source_id)
    
    if status:
        query = query.where(DataRecordModel.status == status.value)
    
    if min_quality_score:
        query = query.where(DataRecordModel.quality_score >= min_quality_score)
    
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    return [
        DataRecordResponse(
            record_id=record.record_id,
            source_id=record.source_id,
            raw_data=record.raw_data or {},
            processed_data=record.processed_data,
            metadata=record.metadata or {},
            quality_score=record.quality_score,
            status=DataStatus(record.status),
            created_at=record.created_at,
            processed_at=record.processed_at
        )
        for record in records
    ]

# 标注管理端点

@router.post("/annotation-tasks", response_model=AnnotationTaskResponse)
async def create_annotation_task(
    task_data: AnnotationTaskCreate,
    current_user: str = "system",  # 这里应该从认证中获取
    db: AsyncSession = Depends(get_db)
):
    """创建标注任务"""
    import uuid
    
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    annotation_task = AnnotationTask(
        task_id=task_id,
        name=task_data.name,
        description=task_data.description,
        task_type=task_data.task_type,
        data_records=task_data.data_records,
        annotation_schema=task_data.annotation_schema,
        guidelines=task_data.guidelines,
        assignees=task_data.assignees,
        created_by=current_user,
        deadline=task_data.deadline
    )
    
    # 使用AnnotationManager创建任务
    annotation_manager = AnnotationManager(db)
    created_task_id = await annotation_manager.create_task(annotation_task)
    
    return AnnotationTaskResponse(
        task_id=created_task_id,
        name=task_data.name,
        description=task_data.description,
        task_type=task_data.task_type,
        data_records=task_data.data_records,
        annotation_schema=task_data.annotation_schema,
        guidelines=task_data.guidelines,
        assignees=task_data.assignees,
        created_by=current_user,
        status=AnnotationStatus.PENDING,
        created_at=utc_now(),
        deadline=task_data.deadline
    )

@router.get("/annotation-tasks", response_model=List[AnnotationTaskResponse])
async def list_annotation_tasks(
    db: AsyncSession = Depends(get_db),
    status: Optional[AnnotationStatus] = Query(None),
    created_by: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取标注任务列表"""
    from sqlalchemy import select
    from .models import AnnotationTaskModel
    
    query = select(AnnotationTaskModel)
    
    if status:
        query = query.where(AnnotationTaskModel.status == status.value)
    
    if created_by:
        query = query.where(AnnotationTaskModel.created_by == created_by)
    
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return [
        AnnotationTaskResponse(
            task_id=task.task_id,
            name=task.name,
            description=task.description or "",
            task_type=AnnotationTaskType(task.task_type),
            data_records=task.data_records,
            annotation_schema=task.annotation_schema,
            guidelines=task.guidelines or "",
            assignees=task.assignees,
            created_by=task.created_by,
            status=AnnotationStatus(task.status),
            created_at=task.created_at,
            deadline=task.deadline
        )
        for task in tasks
    ]

@router.post("/annotations", response_model=AnnotationResponse)
async def submit_annotation(
    annotation_data: AnnotationCreate,
    current_user: str = "annotator",  # 这里应该从认证中获取
    db: AsyncSession = Depends(get_db)
):
    """提交标注结果"""
    import uuid
    
    annotation_id = f"ann_{uuid.uuid4().hex[:8]}"
    
    annotation = Annotation(
        annotation_id=annotation_id,
        task_id=annotation_data.task_id,
        record_id=annotation_data.record_id,
        annotator_id=current_user,
        annotation_data=annotation_data.annotation_data,
        confidence=annotation_data.confidence,
        time_spent=annotation_data.time_spent
    )
    
    # 使用AnnotationManager提交标注
    annotation_manager = AnnotationManager(db)
    created_annotation_id = await annotation_manager.submit_annotation(annotation)
    
    return AnnotationResponse(
        annotation_id=created_annotation_id,
        task_id=annotation_data.task_id,
        record_id=annotation_data.record_id,
        annotator_id=current_user,
        annotation_data=annotation_data.annotation_data,
        confidence=annotation_data.confidence,
        time_spent=annotation_data.time_spent,
        status="submitted",
        created_at=utc_now()
    )

@router.get("/annotation-tasks/{task_id}/progress")
async def get_task_progress(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """获取标注任务进度"""
    annotation_manager = AnnotationManager(db)
    progress = await annotation_manager.get_task_progress(task_id)
    
    return {
        "task_id": progress.task_id,
        "total_records": progress.total_records,
        "annotated_records": progress.annotated_records,
        "progress_percentage": progress.progress_percentage,
        "status_distribution": progress.status_distribution,
        "annotator_performance": progress.annotator_performance,
        "estimated_completion": progress.estimated_completion
    }

@router.get("/annotation-tasks/{task_id}/quality-report")
async def get_quality_report(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """获取标注质量报告"""
    quality_controller = QualityController(db)
    report = await quality_controller.generate_quality_report(task_id)
    
    return {
        "task_id": report.task_id,
        "overall_score": report.overall_score,
        "agreement_metrics": report.agreement_metrics,
        "consistency_metrics": report.consistency_metrics,
        "annotator_performance": report.annotator_performance,
        "recommendations": report.recommendations
    }

# 版本管理端点

@router.post("/versions", response_model=VersionResponse)
async def create_version(
    version_data: VersionCreate,
    current_user: str = "system",  # 这里应该从认证中获取
    db: AsyncSession = Depends(get_db)
):
    """创建数据版本"""
    version_manager = DataVersionManager(db)
    
    # 转换过滤条件
    data_filter = None
    if version_data.data_filter:
        data_filter = DataFilter(**version_data.data_filter)
    
    version_id = await version_manager.create_version(
        dataset_name=version_data.dataset_name,
        version_number=version_data.version_number,
        description=version_data.description,
        created_by=current_user,
        data_filter=data_filter,
        parent_version=version_data.parent_version
    )
    
    # 获取创建的版本信息
    from sqlalchemy import select
    from .models import DataVersionModel
    
    stmt = select(DataVersionModel).where(DataVersionModel.version_id == version_id)
    result = await db.execute(stmt)
    version = result.scalar_one()
    
    return VersionResponse(
        version_id=version.version_id,
        dataset_name=version.dataset_name,
        version_number=version.version_number,
        description=version.description or "",
        created_by=version.created_by,
        parent_version=version.parent_version,
        changes_summary=version.changes_summary,
        metadata=version.metadata,
        record_count=version.record_count,
        size_bytes=version.size_bytes,
        created_at=version.created_at
    )

@router.get("/versions", response_model=List[VersionResponse])
async def list_versions(
    db: AsyncSession = Depends(get_db),
    dataset_name: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """获取版本列表"""
    from sqlalchemy import select, desc
    from .models import DataVersionModel
    
    query = select(DataVersionModel).order_by(desc(DataVersionModel.created_at))
    
    if dataset_name:
        query = query.where(DataVersionModel.dataset_name == dataset_name)
    
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    versions = result.scalars().all()
    
    return [
        VersionResponse(
            version_id=version.version_id,
            dataset_name=version.dataset_name,
            version_number=version.version_number,
            description=version.description or "",
            created_by=version.created_by,
            parent_version=version.parent_version,
            changes_summary=version.changes_summary,
            metadata=version.metadata,
            record_count=version.record_count,
            size_bytes=version.size_bytes,
            created_at=version.created_at
        )
        for version in versions
    ]

@router.get("/versions/{version1_id}/compare/{version2_id}")
async def compare_versions(
    version1_id: str,
    version2_id: str,
    db: AsyncSession = Depends(get_db)
):
    """比较两个版本"""
    version_manager = DataVersionManager(db)
    comparison = await version_manager.compare_versions(version1_id, version2_id)
    
    return {
        "version1_id": comparison.version1_id,
        "version2_id": comparison.version2_id,
        "summary": comparison.summary,
        "added_records": comparison.added_records,
        "removed_records": comparison.removed_records,
        "modified_records": comparison.modified_records
    }

@router.post("/versions/{target_version_id}/rollback")
async def rollback_version(
    target_version_id: str,
    new_version_number: str,
    current_user: str = "system",  # 这里应该从认证中获取
    db: AsyncSession = Depends(get_db)
):
    """回滚到指定版本"""
    version_manager = DataVersionManager(db)
    new_version_id = await version_manager.rollback_to_version(
        target_version_id=target_version_id,
        new_version_number=new_version_number,
        created_by=current_user
    )
    
    return {"new_version_id": new_version_id, "message": "Rollback completed"}

@router.post("/versions/{version1_id}/merge/{version2_id}")
async def merge_versions(
    version1_id: str,
    version2_id: str,
    new_version_number: str,
    current_user: str = "system",  # 这里应该从认证中获取
    merge_strategy: ConflictResolution = ConflictResolution.AUTO_MERGE,
    db: AsyncSession = Depends(get_db)
):
    """合并两个版本"""
    version_manager = DataVersionManager(db)
    merge_result = await version_manager.merge_versions(
        version1_id=version1_id,
        version2_id=version2_id,
        merge_strategy=merge_strategy,
        created_by=current_user,
        new_version_number=new_version_number
    )
    
    return {
        "new_version_id": merge_result.new_version_id,
        "conflicts": merge_result.conflicts,
        "auto_resolved": merge_result.auto_resolved,
        "manual_resolution_needed": merge_result.manual_resolution_needed,
        "success": merge_result.success
    }

# 数据导出端点

@router.post("/export")
async def export_version(
    export_data: ExportJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """导出版本数据"""
    # 添加后台任务
    background_tasks.add_task(
        _run_export_job,
        export_data.version_id,
        export_data.export_format,
        export_data.output_filename
    )
    
    return {"message": "Export job started", "version_id": export_data.version_id}

async def _run_export_job(
    version_id: str,
    export_format: ExportFormat,
    output_filename: Optional[str]
):
    """运行导出任务"""
    try:
        async with get_db_session() as db:
            version_manager = DataVersionManager(db)
            output_path = await version_manager.export_version(
                version_id=version_id,
                export_format=export_format,
                output_path=output_filename
            )
        logger.info("数据导出完成", output_path=output_path)
    except Exception as e:
        logger.error("数据导出失败", error=str(e), exc_info=True)

@router.get("/export/{filename}")
async def download_export_file(filename: str):
    """下载导出文件"""
    file_path = f"./{filename}"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# 统计和监控端点

@router.get("/stats/overview")
async def get_overview_stats(
    db: AsyncSession = Depends(get_db)
):
    """获取系统概览统计"""
    from sqlalchemy import select, func
    from .models import DataSourceModel, DataRecordModel, AnnotationTaskModel, AnnotationModel
    
    # 数据源统计
    sources_stmt = select(func.count(DataSourceModel.id))
    sources_result = await db.execute(sources_stmt)
    total_sources = sources_result.scalar() or 0
    
    active_sources_stmt = select(func.count(DataSourceModel.id)).where(
        DataSourceModel.is_active == True
    )
    active_sources_result = await db.execute(active_sources_stmt)
    active_sources = active_sources_result.scalar() or 0
    
    # 数据记录统计
    records_stmt = select(func.count(DataRecordModel.id))
    records_result = await db.execute(records_stmt)
    total_records = records_result.scalar() or 0
    
    processed_records_stmt = select(func.count(DataRecordModel.id)).where(
        DataRecordModel.status == DataStatus.PROCESSED.value
    )
    processed_records_result = await db.execute(processed_records_stmt)
    processed_records = processed_records_result.scalar() or 0
    
    # 标注任务统计
    tasks_stmt = select(func.count(AnnotationTaskModel.id))
    tasks_result = await db.execute(tasks_stmt)
    total_tasks = tasks_result.scalar() or 0
    
    completed_tasks_stmt = select(func.count(AnnotationTaskModel.id)).where(
        AnnotationTaskModel.status == AnnotationStatus.COMPLETED.value
    )
    completed_tasks_result = await db.execute(completed_tasks_stmt)
    completed_tasks = completed_tasks_result.scalar() or 0
    
    # 标注统计
    annotations_stmt = select(func.count(AnnotationModel.id))
    annotations_result = await db.execute(annotations_stmt)
    total_annotations = annotations_result.scalar() or 0
    
    return {
        "sources": {
            "total": total_sources,
            "active": active_sources
        },
        "records": {
            "total": total_records,
            "processed": processed_records,
            "processing_rate": processed_records / total_records if total_records > 0 else 0
        },
        "tasks": {
            "total": total_tasks,
            "completed": completed_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
        },
        "annotations": {
            "total": total_annotations
        }
    }

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": utc_now()}
