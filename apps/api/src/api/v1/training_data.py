"""
训练数据管理API接口
"""

import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, UploadFile, File, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import asyncio
import logging

from ...ai.training_data_management.models import DataSource, AnnotationTask, Annotation, AnnotationTaskType, AnnotationStatus
from ...ai.training_data_management.manager import DataCollectionManager
from ...ai.training_data_management.annotation import AnnotationManager
from ...ai.training_data_management.version_manager import DataVersionManager
from ...core.config import get_settings

router = APIRouter(prefix="/training-data", tags=["training-data"])
logger = logging.getLogger(__name__)

# 初始化安全认证
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """验证访问令牌"""
    token = credentials.credentials
    # 简化的令牌验证逻辑，生产环境应使用更安全的验证方式
    if not token or len(token) < 10:
        raise HTTPException(
            status_code=401,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# 延迟初始化管理器
data_manager = None
annotation_manager = None
version_manager = None

def get_data_manager() -> DataCollectionManager:
    """获取数据管理器实例"""
    global data_manager
    if data_manager is None:
        settings = get_settings()
        data_manager = DataCollectionManager(settings.DATABASE_URL)
    return data_manager

def get_annotation_manager() -> AnnotationManager:
    """获取标注管理器实例"""
    global annotation_manager
    if annotation_manager is None:
        settings = get_settings()
        annotation_manager = AnnotationManager(settings.DATABASE_URL)
    return annotation_manager

def get_version_manager() -> DataVersionManager:
    """获取版本管理器实例"""
    global version_manager
    if version_manager is None:
        settings = get_settings()
        version_manager = DataVersionManager(settings.DATABASE_URL, "./data_versions")
    return version_manager

# Pydantic模型定义

class DataSourceCreate(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=255, description="数据源ID")
    source_type: str = Field(..., pattern="^(api|file|web|database)$", description="数据源类型: api, file, web, database")
    name: str = Field(..., min_length=1, max_length=255, description="数据源名称")
    description: str = Field("", max_length=1000, description="数据源描述")
    config: Dict[str, Any] = Field(..., description="数据源配置")
    
    class Config:
        # 防止额外字段注入
        extra = "forbid"

class DataSourceResponse(BaseModel):
    id: str
    source_id: str
    source_type: str
    name: str
    description: str
    config: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime

class DataCollectionRequest(BaseModel):
    source_id: str = Field(..., description="数据源ID")
    preprocessing_config: Optional[Dict[str, Any]] = Field(None, description="预处理配置")

class AnnotationTaskCreate(BaseModel):
    name: str = Field(..., description="任务名称")
    description: str = Field("", description="任务描述")
    task_type: str = Field(..., description="任务类型")
    data_records: List[str] = Field(..., description="数据记录ID列表")
    annotation_schema: Dict[str, Any] = Field(..., description="标注模式")
    guidelines: str = Field("", description="标注指南")
    assignees: List[str] = Field([], description="分配的用户ID列表")
    created_by: str = Field(..., description="创建者ID")
    deadline: Optional[datetime] = Field(None, description="截止时间")

class AnnotationSubmit(BaseModel):
    task_id: str = Field(..., description="任务ID")
    record_id: str = Field(..., description="记录ID")
    annotation_data: Dict[str, Any] = Field(..., description="标注数据")
    confidence: Optional[float] = Field(None, description="置信度", ge=0, le=1)
    time_spent: Optional[int] = Field(None, description="标注耗时(秒)")

class DataVersionCreate(BaseModel):
    dataset_name: str = Field(..., description="数据集名称")
    version_number: str = Field(..., description="版本号")
    description: str = Field(..., description="版本描述")
    data_record_ids: Optional[List[str]] = Field(None, description="数据记录ID列表")
    parent_version: Optional[str] = Field(None, description="父版本ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="版本元数据")

# 数据源管理接口

@router.post("/sources", response_model=Dict[str, str])
async def create_data_source(
    source_data: DataSourceCreate,
    token: str = Depends(verify_token)
):
    """创建数据源"""
    try:
        logger.info(f"Creating data source: {source_data.source_id} by token: {token[:10]}...")
        
        # 验证配置安全性
        if 'password' in source_data.config or 'secret' in source_data.config:
            logger.warning(f"Data source {source_data.source_id} contains sensitive information")
        
        source = DataSource(
            source_id=source_data.source_id,
            source_type=source_data.source_type,
            name=source_data.name,
            description=source_data.description,
            config=source_data.config
        )
        
        db_id = get_data_manager().register_data_source(source)
        logger.info(f"Successfully created data source: {source_data.source_id}")
        
        return {
            "id": db_id,
            "source_id": source_data.source_id,
            "message": "数据源创建成功"
        }
    except Exception as e:
        logger.error(f"Failed to create data source {source_data.source_id}: {e}")
        raise HTTPException(status_code=400, detail=f"创建数据源失败: {str(e)}")

@router.get("/sources", response_model=List[DataSourceResponse])
async def list_data_sources(active_only: bool = Query(True, description="只显示活跃的数据源")):
    """列出数据源"""
    try:
        sources = get_data_manager().list_data_sources(active_only=active_only)
        return sources
    except Exception as e:
        logger.error(f"获取数据源列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据源列表失败")

@router.put("/sources/{source_id}")
async def update_data_source(
    source_id: str, 
    updates: Dict[str, Any],
    token: str = Depends(verify_token)
):
    """更新数据源配置"""
    try:
        success = get_data_manager().update_data_source(source_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="数据源未找到")
        
        return {"message": "数据源更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新数据源失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/sources/{source_id}")
async def delete_data_source(
    source_id: str,
    token: str = Depends(verify_token)
):
    """删除数据源（软删除）"""
    try:
        success = get_data_manager().delete_data_source(source_id)
        if not success:
            raise HTTPException(status_code=404, detail="数据源未找到")
        
        return {"message": "数据源删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据源失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# 数据收集接口

@router.post("/collect")
async def collect_data(
    request: DataCollectionRequest, 
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """启动数据收集任务"""
    try:
        # 在后台任务中执行数据收集
        background_tasks.add_task(
            _collect_data_background,
            request.source_id,
            request.preprocessing_config
        )
        
        return {
            "message": "数据收集任务已启动",
            "source_id": request.source_id
        }
    except Exception as e:
        logger.error(f"启动数据收集失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def _collect_data_background(source_id: str, preprocessing_config: Optional[Dict[str, Any]]):
    """后台数据收集任务"""
    try:
        stats = await get_data_manager().collect_from_source(source_id, preprocessing_config)
        logger.info(f"数据收集完成: {stats}")
    except Exception as e:
        logger.error(f"后台数据收集失败: {e}")

@router.get("/records")
async def get_data_records(
    source_id: Optional[str] = Query(None, description="数据源ID"),
    status: Optional[str] = Query(None, description="记录状态"),
    min_quality_score: Optional[float] = Query(None, description="最小质量分数"),
    limit: int = Query(100, description="返回数量限制", le=1000),
    offset: int = Query(0, description="偏移量")
):
    """获取数据记录"""
    try:
        records = get_data_manager().get_data_records(
            source_id=source_id,
            status=status,
            min_quality_score=min_quality_score,
            limit=limit,
            offset=offset
        )
        return {
            "records": records,
            "count": len(records),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"获取数据记录失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据记录失败")

@router.get("/statistics")
async def get_collection_statistics(source_id: Optional[str] = Query(None, description="数据源ID")):
    """获取收集统计信息"""
    try:
        stats = get_data_manager().get_collection_statistics(source_id=source_id)
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")

@router.post("/reprocess")
async def reprocess_records(
    record_ids: Optional[List[str]] = None,
    source_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None
):
    """重新处理记录"""
    try:
        stats = await get_data_manager().reprocess_records(
            record_ids=record_ids,
            source_id=source_id,
            status_filter=status_filter,
            preprocessing_config=preprocessing_config
        )
        return stats
    except Exception as e:
        logger.error(f"重新处理记录失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# 标注管理接口

@router.post("/annotation-tasks")
async def create_annotation_task(task_data: AnnotationTaskCreate):
    """创建标注任务"""
    try:
        task = AnnotationTask(
            task_id=str(uuid.uuid4()),
            name=task_data.name,
            description=task_data.description,
            task_type=task_data.task_type,  # 直接使用字符串
            record_ids=task_data.data_records,
            schema=task_data.annotation_schema,
            annotators=task_data.assignees  # 使用annotators而不是assignees
        )
        
        db_id = get_annotation_manager().create_annotation_task(task)
        
        return {
            "id": db_id,
            "task_id": task.task_id,
            "message": "标注任务创建成功"
        }
    except Exception as e:
        logger.error(f"创建标注任务失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/annotation-tasks")
async def list_annotation_tasks(
    assignee_id: Optional[str] = Query(None, description="分配者ID"),
    status: Optional[str] = Query(None, description="任务状态"),
    created_by: Optional[str] = Query(None, description="创建者ID"),
    limit: int = Query(100, description="返回数量限制"),
    offset: int = Query(0, description="偏移量")
):
    """获取标注任务列表"""
    try:
        tasks = get_annotation_manager().get_annotation_tasks(
            assignee_id=assignee_id,
            status=status,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
        return {
            "tasks": tasks,
            "count": len(tasks),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"获取标注任务失败: {e}")
        raise HTTPException(status_code=500, detail="获取标注任务失败")

@router.get("/annotation-tasks/{task_id}")
async def get_annotation_task_details(task_id: str):
    """获取标注任务详情"""
    try:
        task = get_annotation_manager().get_annotation_task_details(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="标注任务未找到")
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取标注任务详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取标注任务详情失败")

@router.get("/annotation-tasks/{task_id}/progress")
async def get_annotation_progress(task_id: str):
    """获取标注进度"""
    try:
        progress = get_annotation_manager().get_annotation_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="标注任务未找到")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取标注进度失败: {e}")
        raise HTTPException(status_code=500, detail="获取标注进度失败")

@router.post("/annotation-tasks/{task_id}/assign")
async def assign_annotation_task(task_id: str, user_ids: List[str]):
    """分配标注任务"""
    try:
        success = get_annotation_manager().assign_task(task_id, user_ids)
        if not success:
            raise HTTPException(status_code=404, detail="标注任务未找到")
        return {"message": "任务分配成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分配标注任务失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/annotations")
async def submit_annotation(annotation_data: AnnotationSubmit, annotator_id: str = Query(..., description="标注者ID")):
    """提交标注结果"""
    try:
        annotation = Annotation(
            annotation_id=str(uuid.uuid4()),
            task_id=annotation_data.task_id,
            record_id=annotation_data.record_id,
            annotator_id=annotator_id,
            annotation_data=annotation_data.annotation_data,
            confidence=annotation_data.confidence,
            time_spent=annotation_data.time_spent
        )
        
        db_id = get_annotation_manager().submit_annotation(annotation)
        
        return {
            "id": db_id,
            "annotation_id": annotation.annotation_id,
            "message": "标注提交成功"
        }
    except Exception as e:
        logger.error(f"提交标注失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/annotation-tasks/{task_id}/agreement")
async def get_inter_annotator_agreement(task_id: str):
    """获取标注者间一致性"""
    try:
        agreement = get_annotation_manager().calculate_inter_annotator_agreement(task_id)
        return agreement
    except Exception as e:
        logger.error(f"计算标注一致性失败: {e}")
        raise HTTPException(status_code=500, detail="计算标注一致性失败")

@router.get("/annotation-tasks/{task_id}/quality-report")
async def get_quality_control_report(task_id: str):
    """获取质量控制报告"""
    try:
        report = get_annotation_manager().get_quality_control_report(task_id)
        return report
    except Exception as e:
        logger.error(f"生成质量报告失败: {e}")
        raise HTTPException(status_code=500, detail="生成质量报告失败")

@router.get("/annotations")
async def get_user_annotations(
    annotator_id: str = Query(..., description="标注者ID"),
    task_id: Optional[str] = Query(None, description="任务ID"),
    status: Optional[str] = Query(None, description="状态"),
    limit: int = Query(100, description="返回数量限制"),
    offset: int = Query(0, description="偏移量")
):
    """获取用户标注记录"""
    try:
        annotations = get_annotation_manager().get_user_annotations(
            annotator_id=annotator_id,
            task_id=task_id,
            status=status,
            limit=limit,
            offset=offset
        )
        return {
            "annotations": annotations,
            "count": len(annotations),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"获取用户标注失败: {e}")
        raise HTTPException(status_code=500, detail="获取用户标注失败")

# 版本管理接口

@router.post("/versions")
async def create_data_version(version_data: DataVersionCreate, created_by: str = Query(..., description="创建者ID")):
    """创建数据版本"""
    try:
        # 如果指定了数据记录ID，获取数据记录
        if version_data.data_record_ids:
            records = get_data_manager().get_data_records(
                record_ids=version_data.data_record_ids,
                limit=len(version_data.data_record_ids)
            )
            data_records = [record['processed_data'] or record['raw_data'] for record in records]
        else:
            # 获取所有已处理的数据记录
            all_records = get_data_manager().get_data_records(status='processed', limit=10000)
            data_records = [record['processed_data'] for record in all_records]
        
        if not data_records:
            raise HTTPException(status_code=400, detail="没有找到可用的数据记录")
        
        version_id = get_version_manager().create_version(
            dataset_name=version_data.dataset_name,
            version_number=version_data.version_number,
            data_records=data_records,
            description=version_data.description,
            created_by=created_by,
            parent_version=version_data.parent_version,
            metadata=version_data.metadata
        )
        
        return {
            "version_id": version_id,
            "dataset_name": version_data.dataset_name,
            "version_number": version_data.version_number,
            "record_count": len(data_records),
            "message": "数据版本创建成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建数据版本失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/datasets")
async def list_datasets():
    """列出所有数据集"""
    try:
        datasets = get_version_manager().list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"获取数据集列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据集列表失败")

@router.get("/datasets/{dataset_name}/versions")
async def list_dataset_versions(dataset_name: str):
    """列出数据集的所有版本"""
    try:
        versions = get_version_manager().list_versions(dataset_name)
        return {
            "dataset_name": dataset_name,
            "versions": versions,
            "count": len(versions)
        }
    except Exception as e:
        logger.error(f"获取版本列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取版本列表失败")

@router.get("/versions/{version_id}")
async def get_version_data(version_id: str):
    """获取版本数据"""
    try:
        data = get_version_manager().get_version_data(version_id)
        return {
            "version_id": version_id,
            "data": data,
            "record_count": len(data)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取版本数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取版本数据失败")

@router.get("/versions/{version_id}/history")
async def get_version_history(version_id: str):
    """获取版本历史"""
    try:
        # 首先获取版本信息以确定数据集名称
        vm = get_version_manager()
        with vm.SessionLocal() as db:
            version = db.query(vm.DataVersionModel).filter(
                vm.DataVersionModel.version_id == version_id
            ).first()
            
            if not version:
                raise HTTPException(status_code=404, detail="版本未找到")
        
        history = vm.get_version_history(version.dataset_name, version_id)
        return {
            "version_id": version_id,
            "history": history,
            "count": len(history)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取版本历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取版本历史失败")

@router.post("/versions/{version_id1}/compare/{version_id2}")
async def compare_versions(version_id1: str, version_id2: str):
    """比较两个版本"""
    try:
        comparison = get_version_manager().compare_versions(version_id1, version_id2)
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"比较版本失败: {e}")
        raise HTTPException(status_code=500, detail="比较版本失败")

@router.post("/versions/{version_id}/export")
async def export_version(
    version_id: str, 
    format: str = Query("jsonl", description="导出格式: jsonl, json, csv")
):
    """导出版本数据"""
    try:
        if format not in ['jsonl', 'json', 'csv']:
            raise HTTPException(status_code=400, detail="不支持的导出格式")
        
        export_path = f"./exports/{version_id}.{format}"
        file_path = get_version_manager().export_version(version_id, export_path, format)
        
        return {
            "version_id": version_id,
            "export_path": file_path,
            "format": format,
            "message": "数据导出成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"导出版本数据失败: {e}")
        raise HTTPException(status_code=500, detail="导出版本数据失败")

@router.post("/datasets/{dataset_name}/rollback")
async def rollback_dataset(
    dataset_name: str, 
    target_version_id: str, 
    created_by: str = Query(..., description="创建者ID")
):
    """回滚数据集到指定版本"""
    try:
        new_version_id = get_version_manager().rollback_to_version(dataset_name, target_version_id, created_by)
        
        return {
            "dataset_name": dataset_name,
            "target_version_id": target_version_id,
            "new_version_id": new_version_id,
            "message": "数据集回滚成功"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"回滚数据集失败: {e}")
        raise HTTPException(status_code=500, detail="回滚数据集失败")

@router.get("/version-statistics")
async def get_version_statistics(dataset_name: Optional[str] = Query(None, description="数据集名称")):
    """获取版本统计信息"""
    try:
        stats = get_version_manager().get_version_statistics(dataset_name)
        return stats
    except Exception as e:
        logger.error(f"获取版本统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取版本统计失败")

@router.delete("/versions/{version_id}")
async def delete_version(version_id: str, remove_files: bool = Query(True, description="是否删除文件")):
    """删除版本"""
    try:
        success = get_version_manager().delete_version(version_id, remove_files)
        if not success:
            raise HTTPException(status_code=404, detail="版本未找到")
        
        return {"message": "版本删除成功"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"删除版本失败: {e}")
        raise HTTPException(status_code=500, detail="删除版本失败")

# 通用工具接口

@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": utc_now(),
        "services": {
            "data_collection": "active",
            "annotation": "active", 
            "version_management": "active"
        }
    }

@router.get("/queue-status")
async def get_queue_status():
    """获取处理队列状态"""
    try:
        status = get_data_manager().get_processing_queue_status()
        return status
    except Exception as e:
        logger.error(f"获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取队列状态失败")