"""
知识图谱抽取API端点

提供知识图谱实体识别、关系抽取、批量处理等RESTful API接口
"""

import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import Field
import json
from src.ai.knowledge_graph.data_models import (
    ExtractionRequest, ExtractionResponse,
    BatchProcessingRequest, BatchProcessingResponse,
    EntityModel, RelationModel,
    Entity, Relation, KnowledgeGraph
)
from src.ai.knowledge_graph.batch_processor import BatchProcessor, BatchConfig
from src.api.base_model import ApiBaseModel
from src.ai.knowledge_graph.entity_recognizer import MultiModelEntityRecognizer
from src.ai.knowledge_graph.relation_extractor import RelationExtractor
from src.ai.knowledge_graph.entity_linker import EntityLinker
from src.ai.knowledge_graph.multilingual_processor import MultilingualProcessor

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/knowledge", tags=["knowledge-extraction"])

# 全局组件实例
entity_recognizer: Optional[MultiModelEntityRecognizer] = None
relation_extractor: Optional[RelationExtractor] = None
entity_linker: Optional[EntityLinker] = None
multilingual_processor: Optional[MultilingualProcessor] = None
batch_processor: Optional[BatchProcessor] = None

# 日志配置

class HealthResponse(ApiBaseModel):
    """健康检查响应模型"""
    status: str = "healthy"
    version: str = "1.0.0"
    components: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemMetrics(ApiBaseModel):
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
            if self.total_requests > 0 else 0
        )
        
        return SystemMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            average_response_time=avg_response_time,
            entities_extracted=self.entities_extracted,
            relations_extracted=self.relations_extracted,
            batch_jobs_completed=self.batch_jobs_completed,
            uptime_seconds=uptime,
            last_updated=utc_now()
        )

# 全局指标收集器实例
metrics_collector = MetricsCollector()

def get_entity_recognizer() -> MultiModelEntityRecognizer:
    """获取实体识别器实例"""
    global entity_recognizer
    if entity_recognizer is None:
        entity_recognizer = MultiModelEntityRecognizer()
    return entity_recognizer

def get_relation_extractor() -> RelationExtractor:
    """获取关系抽取器实例"""
    global relation_extractor
    if relation_extractor is None:
        relation_extractor = RelationExtractor()
    return relation_extractor

def get_entity_linker() -> EntityLinker:
    """获取实体链接器实例"""
    global entity_linker
    if entity_linker is None:
        entity_linker = EntityLinker()
    return entity_linker

def get_multilingual_processor() -> MultilingualProcessor:
    """获取多语言处理器实例"""
    global multilingual_processor
    if multilingual_processor is None:
        multilingual_processor = MultilingualProcessor()
    return multilingual_processor

def get_batch_processor() -> BatchProcessor:
    """获取批处理器实例"""
    global batch_processor
    if batch_processor is None:
        batch_config = BatchConfig(
            max_concurrent_tasks=100,
            task_timeout_seconds=300,
            max_retries=3,
            worker_pool_size=4,
            memory_limit_mb=1000,
        )
        batch_processor = BatchProcessor(batch_config)
    return batch_processor

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        components = {}
        
        # 检查各组件状态
        try:
            recognizer = get_entity_recognizer()
            components["entity_recognizer"] = "healthy"
        except Exception:
            components["entity_recognizer"] = "unhealthy"
        
        try:
            extractor = get_relation_extractor()
            components["relation_extractor"] = "healthy"
        except Exception:
            components["relation_extractor"] = "unhealthy"
        
        try:
            linker = get_entity_linker()
            components["entity_linker"] = "healthy"
        except Exception:
            components["entity_linker"] = "unhealthy"
        
        # 计算运行时间
        uptime = time.time() - metrics_collector.start_time
        
        overall_status = "healthy" if all(
            status == "healthy" for status in components.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components=components,
            uptime_seconds=uptime,
            timestamp=utc_now()
        )
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="健康检查失败"
        )

@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """获取系统指标"""
    return metrics_collector.get_metrics()

@router.post("/extract", response_model=ExtractionResponse)
async def extract_knowledge(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks
):
    """
    单文档知识抽取
    
    从单个文档中抽取实体和关系
    """
    start_time = time.time()
    document_id = str(uuid4())
    
    try:
        # 获取处理器实例
        recognizer = get_entity_recognizer()
        extractor = get_relation_extractor()
        linker = get_entity_linker()
        multilingual = get_multilingual_processor()
        
        # 语言检测
        language = request.language
        if language == "auto":
            language = await multilingual.detect_language(request.text)
        
        entities = []
        relations = []
        
        # 实体识别
        if request.extract_entities:
            raw_entities = await recognizer.extract_entities(
                request.text, 
                language=language,
                confidence_threshold=request.confidence_threshold
            )
            entities = raw_entities
        
        # 关系抽取
        if request.extract_relations and entities:
            raw_relations = await extractor.extract_relations(
                request.text,
                entities,
                language=language,
                confidence_threshold=request.confidence_threshold
            )
            relations = raw_relations
        
        # 实体链接
        if request.link_entities and entities:
            linked_entities = await linker.link_entities(entities, language=language)
            entities = linked_entities
        
        # 计算处理时间
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
            )
            for e in entities
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
            )
            for r in relations
        ]
        
        # 构建响应
        response = ExtractionResponse(
            document_id=document_id,
            text=request.text,
            language=language,
            entities=entity_models,
            relations=relation_models,
            processing_time=processing_time,
            model_versions={
                "entity_recognizer": "1.0.0",
                "relation_extractor": "1.0.0",
                "entity_linker": "1.0.0"
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
        
    except Exception as e:
        processing_time = time.time() - start_time
        metrics_collector.record_request(
            success=False,
            response_time=processing_time
        )
        
        logger.error(f"知识抽取失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识抽取失败: {str(e)}"
        )

@router.post("/batch/submit", response_model=BatchProcessingResponse)
async def submit_batch_job(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    提交批量处理任务
    
    提交多个文档进行批量知识抽取
    """
    try:
        batch_id = str(uuid4())
        processor = get_batch_processor()
        
        # 提交批处理任务
        result = await processor.submit_batch(
            batch_id=batch_id,
            documents=[doc["text"] for doc in request.documents],
            config={
                "language": request.language,
                "extract_entities": request.extract_entities,
                "extract_relations": request.extract_relations,
                "link_entities": request.link_entities,
                "confidence_threshold": request.confidence_threshold,
                **request.batch_settings
            }
        )
        
        return BatchProcessingResponse(
            batch_id=batch_id,
            status="submitted",
            total_documents=len(request.documents),
            created_at=utc_now()
        )
        
    except Exception as e:
        logger.error(f"批处理任务提交失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批处理任务提交失败: {str(e)}"
        )

@router.get("/batch", summary="批处理任务列表")
async def list_batch_jobs(
    limit: int = Query(100, ge=1, le=1000, description="返回任务数量限制")
):
    """列出批处理任务摘要"""
    try:
        processor = get_batch_processor()
        batches = await processor.list_batches(limit)
        return {
            "batches": batches,
            "total": len(batches),
            "timestamp": utc_now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取批处理任务列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取批处理任务列表失败: {str(e)}"
        )

@router.get("/batch/{batch_id}/status", response_model=BatchProcessingResponse)
async def get_batch_status(batch_id: str = Path(..., description="批处理任务ID")):
    """
    获取批处理任务状态
    
    查询指定批处理任务的执行状态和结果
    """
    try:
        processor = get_batch_processor()
        
        # 获取批处理状态
        status_info = await processor.get_batch_status(batch_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="批处理任务不存在"
            )
        
        return BatchProcessingResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取批处理状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取批处理状态失败: {str(e)}"
        )

@router.get("/batch/{batch_id}/results")
async def get_batch_results(batch_id: str = Path(..., description="批处理任务ID")):
    """
    获取批处理任务结果
    
    下载指定批处理任务的完整结果
    """
    try:
        processor = get_batch_processor()
        
        # 获取批处理结果
        results = await processor.get_batch_results(batch_id)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="批处理任务结果不存在"
            )
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取批处理结果失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取批处理结果失败: {str(e)}"
        )

@router.delete("/batch/{batch_id}")
async def cancel_batch_job(batch_id: str = Path(..., description="批处理任务ID")):
    """
    取消批处理任务
    
    取消正在执行或等待中的批处理任务
    """
    try:
        processor = get_batch_processor()
        
        # 取消批处理任务
        success = await processor.cancel_batch(batch_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="批处理任务不存在或已完成"
            )
        
        return {"message": "批处理任务已取消", "batch_id": batch_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消批处理任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消批处理任务失败: {str(e)}"
        )

@router.get("/overview")
async def get_extraction_overview():
    """
    获取知识抽取系统总览
    
    返回系统运行状态和统计信息
    """
    try:
        metrics = metrics_collector.get_metrics()
        processor = get_batch_processor()
        
        return {
            "total_tasks": metrics.total_requests,
            "active_tasks": processor.task_scheduler.get_active_count(),
            "completed_tasks": metrics.successful_requests,
            "failed_tasks": metrics.failed_requests,
            "total_documents": metrics.total_requests,
            "total_entities": metrics.entities_extracted,
            "total_relations": metrics.relations_extracted,
            "average_accuracy": (
                (metrics.successful_requests / metrics.total_requests) * 100.0
                if metrics.total_requests > 0
                else 0.0
            ),
            "uptime_seconds": metrics.uptime_seconds,
            "last_updated": metrics.last_updated
        }
        
    except Exception as e:
        logger.error(f"获取系统总览失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统总览失败: {str(e)}"
        )
