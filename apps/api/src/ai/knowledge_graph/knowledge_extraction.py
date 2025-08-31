"""
知识图谱抽取API端点

提供RESTful API接口用于：
- 单文档知识抽取
- 批量文档处理
- 实时知识查询
- 系统监控和管理
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import json

from .data_models import (
    ExtractionRequest, ExtractionResponse,
    BatchProcessingRequest, BatchProcessingResponse,
    EntityModel, RelationModel,
    Entity, Relation, KnowledgeGraph
)
from .entity_recognizer import MultiModelEntityRecognizer
from .relation_extractor import RelationExtractor
from .entity_linker import EntityLinker
from .multilingual_processor import MultilingualProcessor
from .batch_processor import BatchProcessor, BatchConfig


# 创建路由器
router = APIRouter(prefix="/knowledge", tags=["knowledge-extraction"])

# 全局组件实例
entity_recognizer: Optional[MultiModelEntityRecognizer] = None
relation_extractor: Optional[RelationExtractor] = None
entity_linker: Optional[EntityLinker] = None
multilingual_processor: Optional[MultilingualProcessor] = None
batch_processor: Optional[BatchProcessor] = None

# 日志配置
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = "healthy"
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class SystemMetrics(BaseModel):
    """系统指标模型"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    entities_extracted: int = 0
    relations_extracted: int = 0
    batch_jobs_completed: int = 0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)


# 全局指标收集器
class MetricsCollector:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.entities_extracted = 0
        self.relations_extracted = 0
        self.batch_jobs_completed = 0
        self.start_time = time.time()
    
    def record_request(self, success: bool, response_time: float, 
                      entities: int = 0, relations: int = 0):
        self.total_requests += 1
        self.total_response_time += response_time
        self.entities_extracted += entities
        self.relations_extracted += relations
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def record_batch_completion(self):
        self.batch_jobs_completed += 1
    
    def get_metrics(self) -> SystemMetrics:
        uptime = time.time() - self.start_time
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        # 获取内存使用情况
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0
        
        # 获取缓存命中率
        cache_hit_rate = 0.0
        if batch_processor and batch_processor.result_cache:
            cache_hit_rate = batch_processor.result_cache.get_hit_rate()
        
        return SystemMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            average_response_time=avg_response_time,
            entities_extracted=self.entities_extracted,
            relations_extracted=self.relations_extracted,
            batch_jobs_completed=self.batch_jobs_completed,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=memory_mb,
            uptime_seconds=uptime
        )


# 全局指标收集器实例
metrics_collector = MetricsCollector()


async def get_components():
    """依赖注入：获取已初始化的组件"""
    global entity_recognizer, relation_extractor, entity_linker
    global multilingual_processor, batch_processor
    
    if not all([entity_recognizer, relation_extractor, entity_linker, 
                multilingual_processor, batch_processor]):
        raise HTTPException(
            status_code=503, 
            detail="知识抽取服务尚未初始化完成"
        )
    
    return {
        "entity_recognizer": entity_recognizer,
        "relation_extractor": relation_extractor,
        "entity_linker": entity_linker,
        "multilingual_processor": multilingual_processor,
        "batch_processor": batch_processor
    }


@router.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global entity_recognizer, relation_extractor, entity_linker
    global multilingual_processor, batch_processor
    
    try:
        logger.info("正在初始化知识抽取组件...")
        
        # 初始化各个组件
        entity_recognizer = MultiModelEntityRecognizer()
        await entity_recognizer.initialize()
        
        relation_extractor = RelationExtractor()
        await relation_extractor.initialize()
        
        entity_linker = EntityLinker()
        await entity_linker.initialize()
        
        multilingual_processor = MultilingualProcessor()
        await multilingual_processor.initialize()
        
        # 初始化批处理器
        batch_config = BatchConfig(
            max_concurrent_tasks=20,
            max_concurrent_per_model=5,
            worker_pool_size=4,
            memory_limit_mb=1024,
            cache_size_limit=5000
        )
        batch_processor = BatchProcessor(batch_config)
        await batch_processor.initialize()
        await batch_processor.start_workers()
        
        logger.info("知识抽取组件初始化完成")
        
    except Exception as e:
        logger.error(f"知识抽取组件初始化失败: {e}")
        raise


@router.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global batch_processor
    
    if batch_processor:
        await batch_processor.shutdown()
    
    logger.info("知识抽取组件已关闭")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        components = {}
        
        # 检查组件状态
        if entity_recognizer:
            components["entity_recognizer"] = "healthy"
        else:
            components["entity_recognizer"] = "not_initialized"
            
        if relation_extractor:
            components["relation_extractor"] = "healthy"
        else:
            components["relation_extractor"] = "not_initialized"
            
        if entity_linker:
            components["entity_linker"] = "healthy"
        else:
            components["entity_linker"] = "not_initialized"
            
        if multilingual_processor:
            components["multilingual_processor"] = "healthy"
        else:
            components["multilingual_processor"] = "not_initialized"
            
        if batch_processor:
            components["batch_processor"] = "healthy"
        else:
            components["batch_processor"] = "not_initialized"
        
        # 计算运行时间
        uptime = time.time() - metrics_collector.start_time
        
        # 获取内存使用情况
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0
        
        status = "healthy" if all(
            status == "healthy" for status in components.values()
        ) else "unhealthy"
        
        return HealthResponse(
            status=status,
            components=components,
            uptime_seconds=uptime,
            memory_usage_mb=memory_mb
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(
            status="unhealthy",
            components={"error": str(e)}
        )


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """获取系统指标"""
    return metrics_collector.get_metrics()


@router.post("/extract", response_model=ExtractionResponse)
async def extract_knowledge(
    request: ExtractionRequest,
    components: Dict = Depends(get_components)
):
    """单文档知识抽取"""
    start_time = time.time()
    document_id = str(uuid4())
    
    try:
        # 检查文本长度
        if len(request.text) > 50000:
            raise HTTPException(
                status_code=400,
                detail="文本长度超过限制（最大50000字符）"
            )
        
        # 执行知识抽取
        if request.language == "auto":
            # 使用多语言处理器
            result = await components["multilingual_processor"].process_multilingual_text(
                request.text
            )
            entities = result.entities
            relations = result.relations
            detected_language = result.detected_language.value
            
            # 实体链接（如果需要）
            if request.link_entities:
                linked_entities = await components["entity_linker"].link_entities(entities)
                # 更新实体信息
                entity_map = {e.entity_id: le for e in entities for le in linked_entities 
                             if e.entity_id == le.entity_id}
                for entity in entities:
                    if entity.entity_id in entity_map:
                        linked = entity_map[entity.entity_id]
                        entity.linked_entity = linked.linked_entity
                        entity.canonical_form = linked.canonical_form
        else:
            # 使用指定语言处理
            detected_language = request.language
            
            # 实体识别
            if request.extract_entities:
                entities = await components["entity_recognizer"].extract_entities(
                    request.text, 
                    request.language,
                    request.confidence_threshold
                )
            else:
                entities = []
            
            # 关系抽取
            if request.extract_relations and entities:
                relations = await components["relation_extractor"].extract_relations(
                    request.text, 
                    entities,
                    request.confidence_threshold
                )
            else:
                relations = []
            
            # 实体链接
            if request.link_entities and entities:
                linked_entities = await components["entity_linker"].link_entities(entities)
                # 更新实体信息
                entity_map = {e.entity_id: le for e in entities for le in linked_entities 
                             if e.entity_id == le.entity_id}
                for entity in entities:
                    if entity.entity_id in entity_map:
                        linked = entity_map[entity.entity_id]
                        entity.linked_entity = linked.linked_entity
                        entity.canonical_form = linked.canonical_form
        
        processing_time = time.time() - start_time
        
        # 转换为API模型
        entity_models = [
            EntityModel(
                text=e.text,
                label=e.label.value,
                start=e.start,
                end=e.end,
                confidence=e.confidence,
                canonical_form=e.canonical_form,
                linked_entity=e.linked_entity,
                language=e.language,
                metadata=e.metadata
            ) for e in entities
        ]
        
        relation_models = [
            RelationModel(
                subject=EntityModel(
                    text=r.subject.text,
                    label=r.subject.label.value,
                    start=r.subject.start,
                    end=r.subject.end,
                    confidence=r.subject.confidence,
                    canonical_form=r.subject.canonical_form,
                    linked_entity=r.subject.linked_entity,
                    language=r.subject.language,
                    metadata=r.subject.metadata
                ),
                predicate=r.predicate.value,
                object=EntityModel(
                    text=r.object.text,
                    label=r.object.label.value,
                    start=r.object.start,
                    end=r.object.end,
                    confidence=r.object.confidence,
                    canonical_form=r.object.canonical_form,
                    linked_entity=r.object.linked_entity,
                    language=r.object.language,
                    metadata=r.object.metadata
                ),
                confidence=r.confidence,
                context=r.context,
                source_sentence=r.source_sentence,
                evidence=r.evidence,
                metadata=r.metadata
            ) for r in relations
        ]
        
        # 构建响应
        response = ExtractionResponse(
            document_id=document_id,
            text=request.text,
            language=detected_language,
            entities=entity_models,
            relations=relation_models,
            processing_time=processing_time,
            model_versions={
                "entity_recognizer": "1.0.0",
                "relation_extractor": "1.0.0", 
                "entity_linker": "1.0.0",
                "multilingual_processor": "1.0.0"
            },
            statistics={
                "entity_count": len(entities),
                "relation_count": len(relations),
                "unique_entity_types": len(set(e.label for e in entities)),
                "unique_relation_types": len(set(r.predicate for r in relations)),
                "linked_entities": len([e for e in entities if e.linked_entity])
            },
            metadata=request.extraction_config
        )
        
        # 记录指标
        metrics_collector.record_request(
            success=True,
            response_time=processing_time,
            entities=len(entities),
            relations=len(relations)
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        # 记录失败指标
        metrics_collector.record_request(
            success=False,
            response_time=processing_time
        )
        
        logger.error(f"知识抽取失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"知识抽取处理失败: {str(e)}"
        )


@router.post("/batch", response_model=BatchProcessingResponse)
async def process_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    components: Dict = Depends(get_components)
):
    """批量文档处理"""
    try:
        # 验证文档数量
        if len(request.documents) > 1000:
            raise HTTPException(
                status_code=400,
                detail="单次批处理文档数量不能超过1000"
            )
        
        # 验证文档格式
        for i, doc in enumerate(request.documents):
            if "text" not in doc:
                raise HTTPException(
                    status_code=400,
                    detail=f"文档 {i} 缺少 'text' 字段"
                )
            if len(doc["text"]) > 50000:
                raise HTTPException(
                    status_code=400,
                    detail=f"文档 {i} 文本长度超过限制（最大50000字符）"
                )
        
        # 提交批处理任务
        batch_id = await components["batch_processor"].process_batch(
            request.documents,
            request.priority
        )
        
        # 初始响应
        response = BatchProcessingResponse(
            batch_id=batch_id,
            status="processing",
            total_documents=len(request.documents),
            processed_documents=0,
            successful_documents=0,
            failed_documents=0
        )
        
        # 添加后台任务来跟踪批处理进度
        background_tasks.add_task(
            track_batch_progress,
            batch_id,
            components["batch_processor"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批处理提交失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"批处理提交失败: {str(e)}"
        )


async def track_batch_progress(batch_id: str, batch_processor: BatchProcessor):
    """后台任务：跟踪批处理进度"""
    try:
        # 等待批处理完成
        result = await batch_processor.wait_for_batch_completion(batch_id)
        
        # 记录批处理完成
        metrics_collector.record_batch_completion()
        
        logger.info(f"批处理完成: {batch_id}, 成功: {result.successful_documents}, 失败: {result.failed_documents}")
        
    except Exception as e:
        logger.error(f"批处理跟踪失败: {batch_id}, 错误: {e}")


@router.get("/batch/{batch_id}", response_model=BatchProcessingResponse)
async def get_batch_status(
    batch_id: str = Path(..., description="批处理任务ID"),
    components: Dict = Depends(get_components)
):
    """获取批处理状态"""
    try:
        # 获取批处理状态
        status = components["batch_processor"].get_processing_status()
        
        # 检查特定批次的任务
        completed_tasks = list(components["batch_processor"].task_scheduler.completed_tasks.values())
        failed_tasks = list(components["batch_processor"].task_scheduler.failed_tasks.values())
        active_tasks = list(components["batch_processor"].task_scheduler.active_tasks.values())
        
        batch_tasks = [
            task for task in completed_tasks + failed_tasks + active_tasks
            if task.task_id.startswith(batch_id)
        ]
        
        if not batch_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"未找到批处理任务: {batch_id}"
            )
        
        # 统计状态
        total_documents = len(batch_tasks)
        completed_docs = len([t for t in batch_tasks if t.status.value == "completed"])
        failed_docs = len([t for t in batch_tasks if t.status.value == "failed"])
        processing_docs = len([t for t in batch_tasks if t.status.value == "processing"])
        
        # 判断整体状态
        if completed_docs + failed_docs == total_documents:
            batch_status = "completed"
        elif processing_docs > 0:
            batch_status = "processing"
        else:
            batch_status = "pending"
        
        # 收集结果
        results = []
        errors = []
        
        for task in batch_tasks:
            if task.status.value == "completed" and task.result:
                results.append(ExtractionResponse(**task.result))
            elif task.status.value == "failed" and task.error:
                errors.append({
                    "document_id": task.document_id,
                    "error": task.error
                })
        
        response = BatchProcessingResponse(
            batch_id=batch_id,
            status=batch_status,
            total_documents=total_documents,
            processed_documents=completed_docs + failed_docs,
            successful_documents=completed_docs,
            failed_documents=failed_docs,
            success_rate=completed_docs / total_documents if total_documents > 0 else 0,
            results=results,
            errors=errors,
            processing_time=status["metrics"]["average_processing_time"],
            metrics=status["metrics"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取批处理状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取批处理状态失败: {str(e)}"
        )


@router.get("/processing-status")
async def get_processing_status(
    components: Dict = Depends(get_components)
):
    """获取整体处理状态"""
    try:
        status = components["batch_processor"].get_processing_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"获取处理状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取处理状态失败: {str(e)}"
        )


@router.post("/entities/search")
async def search_entities(
    query: str = Query(..., description="搜索查询"),
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    limit: int = Query(100, description="结果数量限制", ge=1, le=1000),
    components: Dict = Depends(get_components)
):
    """实体搜索"""
    try:
        # 这里可以实现基于已抽取实体的搜索功能
        # 当前版本返回空结果
        return {
            "query": query,
            "entity_type": entity_type,
            "results": [],
            "total": 0,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"实体搜索失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"实体搜索失败: {str(e)}"
        )


@router.post("/relations/search")
async def search_relations(
    subject: Optional[str] = Query(None, description="主语实体"),
    predicate: Optional[str] = Query(None, description="关系类型"),
    object: Optional[str] = Query(None, description="宾语实体"),
    limit: int = Query(100, description="结果数量限制", ge=1, le=1000),
    components: Dict = Depends(get_components)
):
    """关系搜索"""
    try:
        # 这里可以实现基于已抽取关系的搜索功能
        # 当前版本返回空结果
        return {
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "results": [],
            "total": 0,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"关系搜索失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"关系搜索失败: {str(e)}"
        )


@router.delete("/cache")
async def clear_cache(
    components: Dict = Depends(get_components)
):
    """清空缓存"""
    try:
        if components["batch_processor"].result_cache:
            components["batch_processor"].result_cache.clear()
        
        return {"message": "缓存已清空"}
        
    except Exception as e:
        logger.error(f"清空缓存失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"清空缓存失败: {str(e)}"
        )


# 导出路由器
__all__ = ["router"]