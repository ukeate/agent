"""
知识图谱存储系统API
提供图谱CRUD、查询、质量管理、性能监控等功能的REST API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, status
from pydantic import Field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import json
import uuid
from src.ai.knowledge_graph.graph_database import (
    get_graph_database,
    Neo4jGraphDatabase,
    GraphConnectionError,
)
from src.api.base_model import ApiBaseModel
from src.ai.knowledge_graph.graph_operations import GraphOperations
from src.ai.knowledge_graph.schema import SchemaManager
from src.ai.knowledge_graph.incremental_updater import IncrementalUpdater, ConflictResolutionStrategy
from src.ai.knowledge_graph.quality_manager import QualityManager
from src.ai.knowledge_graph.performance_optimizer import PerformanceOptimizer
from src.ai.knowledge_graph.migration_tools import MigrationManager, DataExportImportTool
from src.ai.knowledge_graph.data_models import Entity, Relation, EntityType, RelationType
from src.core.redis import get_redis

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])

# 增量更新作业记录（Redis）
_KG_UPDATE_JOB_KEY_PREFIX = "knowledge_graph:update:job:"
_KG_QUERY_TEMPLATE_INDEX_KEY = "knowledge_graph:query_template:index"
_KG_QUERY_TEMPLATE_KEY_PREFIX = "knowledge_graph:query_template:"

# 全局实例（在应用启动时初始化）
_graph_ops: Optional[GraphOperations] = None
_schema_manager: Optional[SchemaManager] = None
_incremental_updater: Optional[IncrementalUpdater] = None
_quality_manager: Optional[QualityManager] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None
_migration_manager: Optional[MigrationManager] = None
_export_import_tool: Optional[DataExportImportTool] = None

def _query_template_key(template_id: str) -> str:
    return f"{_KG_QUERY_TEMPLATE_KEY_PREFIX}{template_id}"

async def _load_query_templates(redis) -> List[Dict[str, Any]]:
    ids = await redis.smembers(_KG_QUERY_TEMPLATE_INDEX_KEY)
    if not ids:
        return []
    keys = [_query_template_key(template_id) for template_id in ids]
    raw_list = await redis.mget(keys)
    templates: List[Dict[str, Any]] = []
    for raw in raw_list:
        if not raw:
            continue
        try:
            templates.append(json.loads(raw))
        except Exception:
            continue
    return templates

# Pydantic模型
class CreateEntityRequest(ApiBaseModel):
    """创建实体请求"""
    canonical_form: str = Field(..., description="规范化形式")
    entity_type: str = Field(..., description="实体类型")
    text: Optional[str] = Field(None, description="原始文本")
    confidence: float = Field(1.0, ge=0, le=1, description="置信度")
    language: Optional[str] = Field(None, description="语言")
    linked_entity: Optional[str] = Field(None, description="链接实体URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class UpdateEntityRequest(ApiBaseModel):
    """更新实体请求"""
    canonical_form: Optional[str] = Field(None, description="规范化形式")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="置信度")
    language: Optional[str] = Field(None, description="语言")
    linked_entity: Optional[str] = Field(None, description="链接实体URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class CreateRelationRequest(ApiBaseModel):
    """创建关系请求"""
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    confidence: float = Field(1.0, ge=0, le=1, description="置信度")
    context: str = Field(..., description="关系上下文")
    source_sentence: str = Field(..., description="源句子")
    evidence: List[str] = Field(default_factory=list, description="证据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class EntitySearchRequest(ApiBaseModel):
    """实体搜索请求"""
    canonical_form_contains: Optional[str] = Field(None, description="包含的规范化形式")
    entity_type: Optional[str] = Field(None, description="实体类型")
    confidence_gte: Optional[float] = Field(None, ge=0, le=1, description="最小置信度")
    confidence_lte: Optional[float] = Field(None, ge=0, le=1, description="最大置信度")
    created_after: Optional[datetime] = Field(None, description="创建时间之后")
    created_before: Optional[datetime] = Field(None, description="创建时间之前")
    limit: int = Field(100, ge=1, le=1000, description="返回数量限制")
    skip: int = Field(0, ge=0, description="跳过数量")

class CustomQueryRequest(ApiBaseModel):
    """自定义查询请求"""
    query: str = Field(..., description="Cypher查询语句")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    read_only: bool = Field(True, description="是否只读查询")

class QueryTemplateRequest(ApiBaseModel):
    """查询模板请求"""
    name: str = Field(..., description="模板名称")
    description: str = Field("", description="模板描述")
    query: str = Field(..., description="查询语句")
    category: str = Field("cypher", description="模板分类")
    parameters: List[str] = Field(default_factory=list, description="参数列表")

class BatchUpsertRequest(ApiBaseModel):
    """批量更新请求"""
    entities: List[Dict[str, Any]] = Field(..., description="实体数据列表")
    conflict_strategy: str = Field("merge_highest_confidence", description="冲突解决策略")

# 依赖注入
async def get_graph_components() -> tuple:
    """获取图数据库相关组件"""
    global _graph_ops, _schema_manager, _incremental_updater, _quality_manager
    global _performance_optimizer, _migration_manager, _export_import_tool
    
    if any(
        component is None
        for component in (
            _graph_ops,
            _schema_manager,
            _incremental_updater,
            _quality_manager,
            _performance_optimizer,
            _migration_manager,
            _export_import_tool,
        )
    ):
        _graph_ops = None
        _schema_manager = None
        _incremental_updater = None
        _quality_manager = None
        _performance_optimizer = None
        _migration_manager = None
        _export_import_tool = None
        # 初始化组件
        try:
            graph_db = await get_graph_database()
        except GraphConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
        _schema_manager = SchemaManager(graph_db)
        _graph_ops = GraphOperations(graph_db, _schema_manager)
        _incremental_updater = IncrementalUpdater(graph_db, _graph_ops, _schema_manager)
        _quality_manager = QualityManager(graph_db, _graph_ops)
        _performance_optimizer = PerformanceOptimizer(graph_db)
        _migration_manager = MigrationManager(graph_db, _schema_manager)
        _export_import_tool = DataExportImportTool(graph_db)
        
        # 初始化性能优化器
        await _performance_optimizer.initialize()
        
    return (_graph_ops, _schema_manager, _incremental_updater, _quality_manager,
            _performance_optimizer, _migration_manager, _export_import_tool)

# 实体管理API
@router.post("/entities", summary="创建实体")
async def create_entity(
    request: CreateEntityRequest,
    components: tuple = Depends(get_graph_components)
):
    """创建新实体"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        entity_id = str(uuid.uuid4())
        result = await graph_ops.create_entity(
            entity_id=entity_id,
            entity_type=request.entity_type,
            canonical_form=request.canonical_form,
            properties={
                "text": request.text,
                "confidence": request.confidence,
                "language": request.language,
                "linked_entity": request.linked_entity,
                **request.metadata
            }
        )
        
        if result.success:
            return {
                "success": True,
                "entity_id": entity_id,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建实体失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}", summary="获取实体")
async def get_entity(
    entity_id: str = Path(..., description="实体ID"),
    components: tuple = Depends(get_graph_components)
):
    """根据ID获取实体详情"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.get_entity(entity_id)
        
        if result.success:
            if result.data:
                return {
                    "success": True,
                    "data": result.data[0],
                    "execution_time_ms": result.execution_time_ms
                }
            else:
                raise HTTPException(status_code=404, detail="实体未找到")
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实体失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/entities/{entity_id}", summary="更新实体")
async def update_entity(
    entity_id: str,
    request: UpdateEntityRequest,
    components: tuple = Depends(get_graph_components)
):
    """更新实体信息"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        # 构建更新数据
        updates = {}
        if request.canonical_form is not None:
            updates["canonical_form"] = request.canonical_form
        if request.confidence is not None:
            updates["confidence"] = request.confidence
        if request.language is not None:
            updates["language"] = request.language
        if request.linked_entity is not None:
            updates["linked_entity"] = request.linked_entity
        updates.update(request.metadata)
        
        result = await graph_ops.update_entity(entity_id, updates)
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except Exception as e:
        logger.error(f"更新实体失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/entities/{entity_id}", summary="删除实体")
async def delete_entity(
    entity_id: str,
    cascade: bool = Query(False, description="是否级联删除关系"),
    components: tuple = Depends(get_graph_components)
):
    """删除实体"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.delete_entity(entity_id, cascade=cascade)
        
        if result.success:
            return {
                "success": True,
                "deleted_count": result.affected_count,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except Exception as e:
        logger.error(f"删除实体失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/entities/search", summary="搜索实体")
async def search_entities(
    request: EntitySearchRequest,
    components: tuple = Depends(get_graph_components)
):
    """搜索实体"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        # 构建搜索过滤器
        filters = {}
        if request.canonical_form_contains:
            filters["canonical_form_contains"] = request.canonical_form_contains
        if request.entity_type:
            filters["type"] = request.entity_type
        if request.confidence_gte is not None:
            filters["confidence_gte"] = request.confidence_gte
        if request.confidence_lte is not None:
            filters["confidence_lte"] = request.confidence_lte
        if request.created_after:
            filters["created_after"] = request.created_after.isoformat()
        if request.created_before:
            filters["created_before"] = request.created_before.isoformat()
        
        result = await graph_ops.find_entities(
            filters=filters,
            limit=request.limit,
            skip=request.skip
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data),
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索实体失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 关系管理API
@router.post("/relations", summary="创建关系")
async def create_relation(
    request: CreateRelationRequest,
    components: tuple = Depends(get_graph_components)
):
    """创建实体间关系"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.create_relationship(
            source_entity_id=request.source_entity_id,
            target_entity_id=request.target_entity_id,
            relation_type=request.relation_type,
            properties={
                "confidence": request.confidence,
                "context": request.context,
                "source_sentence": request.source_sentence,
                "evidence": request.evidence,
                **request.metadata
            }
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建关系失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities/{entity_id}/relations", summary="获取实体关系")
async def get_entity_relations(
    entity_id: str,
    direction: str = Query("both", regex="^(both|incoming|outgoing)$", description="关系方向"),
    relation_types: Optional[List[str]] = Query(None, description="关系类型过滤"),
    limit: int = Query(50, ge=1, le=500, description="返回数量限制"),
    components: tuple = Depends(get_graph_components)
):
    """获取实体的所有关系"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.get_entity_relationships(
            entity_id=entity_id,
            direction=direction,
            relation_types=relation_types,
            limit=limit
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data),
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实体关系失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 图查询API
@router.get("/path", summary="查找最短路径")
async def find_shortest_path(
    source_id: str = Query(..., description="源实体ID"),
    target_id: str = Query(..., description="目标实体ID"),
    max_depth: int = Query(5, ge=1, le=10, description="最大搜索深度")
):
    """查找两个实体间的最短路径"""
    try:
        graph_ops, _, _, _, _, _, _ = await get_graph_components()
        result = await graph_ops.find_shortest_path(
            source_entity_id=source_id,
            target_entity_id=target_id,
            max_depth=max_depth
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查找最短路径失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subgraph/{entity_id}", summary="获取子图")
async def get_subgraph(
    entity_id: str,
    depth: int = Query(2, ge=1, le=5, description="子图深度"),
    max_nodes: int = Query(100, ge=1, le=500, description="最大节点数"),
    components: tuple = Depends(get_graph_components)
):
    """获取以指定实体为中心的子图"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.get_subgraph(
            center_entity_id=entity_id,
            depth=depth,
            max_nodes=max_nodes
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取子图失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", summary="执行自定义查询")
async def execute_custom_query(
    request: CustomQueryRequest,
    components: tuple = Depends(get_graph_components)
):
    """执行自定义Cypher查询"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
        result = await graph_ops.execute_custom_query(
            query=request.query,
            parameters=request.parameters,
            read_only=request.read_only
        )
        
        if result.success:
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data),
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行自定义查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query/templates", summary="获取查询模板")
async def list_query_templates(
    category: Optional[str] = Query(None, description="模板分类过滤"),
):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法获取模板")
    templates = await _load_query_templates(redis_client)
    if category:
        templates = [t for t in templates if t.get("category") == category]
    templates.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"templates": templates, "total": len(templates)}

@router.post("/query/templates", status_code=status.HTTP_201_CREATED, summary="创建查询模板")
async def create_query_template(request: QueryTemplateRequest):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法保存模板")
    template_id = f"tpl_{uuid.uuid4().hex[:8]}"
    template = {
        "id": template_id,
        "name": request.name,
        "description": request.description,
        "query": request.query,
        "category": request.category,
        "parameters": request.parameters,
        "created_at": utc_now().isoformat(),
        "usage_count": 0,
    }
    await redis_client.set(_query_template_key(template_id), json.dumps(template, ensure_ascii=False))
    await redis_client.sadd(_KG_QUERY_TEMPLATE_INDEX_KEY, template_id)
    return template

@router.delete("/query/templates/{template_id}", summary="删除查询模板")
async def delete_query_template(template_id: str = Path(..., description="模板ID")):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法删除模板")
    await redis_client.delete(_query_template_key(template_id))
    await redis_client.srem(_KG_QUERY_TEMPLATE_INDEX_KEY, template_id)
    return {"success": True}

# 增量更新API
@router.post("/update/start", summary="启动增量更新作业")
async def start_incremental_update(
    components: tuple = Depends(get_graph_components),
):
    """启动一次增量更新检查作业（记录到Redis）"""
    _, _, incremental_updater, _, _, _, _ = components
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法记录作业")

    job_id = str(uuid.uuid4())
    started_at = utc_now()
    key = f"{_KG_UPDATE_JOB_KEY_PREFIX}{job_id}"

    record = {
        "job_id": job_id,
        "status": "running",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "updated_nodes": 0,
    }
    await redis_client.set(key, json.dumps(record, ensure_ascii=False))

    try:
        await incremental_updater.detect_and_resolve_conflicts()
        finished_at = utc_now()
        record["status"] = "completed"
        record["finished_at"] = finished_at.isoformat()
        await redis_client.set(key, json.dumps(record, ensure_ascii=False))
        return {"success": True, **record}
    except Exception as e:
        finished_at = utc_now()
        record["status"] = "failed"
        record["finished_at"] = finished_at.isoformat()
        await redis_client.set(key, json.dumps(record, ensure_ascii=False))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/update/jobs", summary="获取增量更新作业列表")
async def list_incremental_update_jobs():
    """从Redis读取最近的增量更新作业"""
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法读取作业")

    jobs: List[Dict[str, Any]] = []
    async for key in redis_client.scan_iter(match=f"{_KG_UPDATE_JOB_KEY_PREFIX}*"):
        raw = await redis_client.get(key)
        if not raw:
            continue
        try:
            jobs.append(json.loads(raw))
        except Exception:
            continue

    jobs.sort(key=lambda j: j.get("started_at") or "", reverse=True)
    return {"jobs": jobs}

@router.get("/update/conflicts", summary="获取增量更新冲突列表")
async def list_incremental_update_conflicts(
    components: tuple = Depends(get_graph_components),
):
    """实时检测冲突（不写入任何模拟数据）"""
    _, _, incremental_updater, _, _, _, _ = components
    conflicts = await incremental_updater.detect_and_resolve_conflicts()
    return {
        "conflicts": [
            {
                "id": c.conflict_id,
                "status": c.conflict_type,
                "description": c.description,
                "resolved_at": None,
            }
            for c in conflicts
        ]
    }

@router.get("/update/metrics", summary="获取增量更新指标")
async def get_incremental_update_metrics(
    components: tuple = Depends(get_graph_components),
):
    """返回当前图谱的节点/边计数及查询时间戳"""
    graph_ops, _, _, _, _, _, _ = components
    result = await graph_ops.get_graph_statistics()
    if not result.success or not result.data:
        raise HTTPException(status_code=500, detail=result.error_message or "获取统计失败")

    statistics = result.data[0]["statistics"]
    total_nodes = statistics.get("entity_count", [{"count": 0}])[0]["count"]
    total_edges = statistics.get("relation_count", [{"count": 0}])[0]["count"]
    return {
        "last_update": utc_now().isoformat(),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }

@router.post("/upsert-entity", summary="智能实体更新")
async def upsert_entity(
    request: CreateEntityRequest,
    conflict_strategy: str = Query("merge_highest_confidence", description="冲突解决策略"),
    components: tuple = Depends(get_graph_components)
):
    """智能实体更新（自动处理重复和冲突）"""
    _, _, incremental_updater, _, _, _, _ = components
    
    try:
        # 创建Entity对象
        entity = Entity(
            text=request.text or request.canonical_form,
            label=EntityType(request.entity_type),
            start=0,
            end=len(request.text or request.canonical_form),
            confidence=request.confidence,
            canonical_form=request.canonical_form,
            linked_entity=request.linked_entity,
            language=request.language,
            metadata=request.metadata
        )
        
        strategy = ConflictResolutionStrategy(conflict_strategy)
        result = await incremental_updater.upsert_entity(entity, strategy)
        
        return {
            "success": result.success,
            "operation": result.operation.value,
            "entity_id": result.entity_id,
            "merged_entities": result.merged_entities,
            "conflicts_detected": [c.to_dict() for c in result.conflicts_detected],
            "changes_made": result.changes_made,
            "execution_time_ms": result.execution_time_ms,
            "error_message": result.error_message
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"智能实体更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upsert", summary="批量智能更新")
async def batch_upsert_entities(
    request: BatchUpsertRequest,
    background_tasks: BackgroundTasks,
    components: tuple = Depends(get_graph_components)
):
    """批量智能更新实体"""
    _, _, incremental_updater, _, _, _, _ = components
    
    try:
        # 转换为Entity对象
        entities = []
        for entity_data in request.entities:
            entity = Entity(
                text=entity_data.get("text", entity_data["canonical_form"]),
                label=EntityType(entity_data["entity_type"]),
                start=0,
                end=len(entity_data.get("text", entity_data["canonical_form"])),
                confidence=entity_data.get("confidence", 1.0),
                canonical_form=entity_data["canonical_form"],
                linked_entity=entity_data.get("linked_entity"),
                language=entity_data.get("language"),
                metadata=entity_data.get("metadata", {})
            )
            entities.append(entity)
        
        strategy = ConflictResolutionStrategy(request.conflict_strategy)
        
        # 异步处理批量更新
        async def process_batch():
            return await incremental_updater.batch_upsert_entities(entities, strategy)
        
        # 启动后台任务
        background_tasks.add_task(process_batch)
        
        return {
            "success": True,
            "message": f"已启动批量更新任务，共 {len(entities)} 个实体",
            "batch_size": len(entities),
            "conflict_strategy": request.conflict_strategy
        }
        
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"缺少必填字段: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量智能更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 质量管理API
@router.get("/quality/metrics", summary="获取质量指标")
async def get_quality_metrics(
):
    """获取图谱质量指标"""
    try:
        _, _, _, quality_manager, _, _, _ = await get_graph_components()
        metrics = await quality_manager.calculate_quality_score()
        return {
            "success": True,
            "metrics": metrics.to_dict()
        }

    except Exception as e:
        return {"success": False, "available": False, "detail": str(e)}

@router.get("/quality/issues", summary="检测质量问题")
async def detect_quality_issues(
):
    """检测图谱质量问题"""
    try:
        _, _, _, quality_manager, _, _, _ = await get_graph_components()
        issues = await quality_manager.detect_quality_issues()
        return {
            "success": True,
            "total_issues": len(issues),
            "issues": [issue.to_dict() for issue in issues]
        }
        
    except Exception as e:
        return {"success": False, "available": False, "issues": [], "detail": str(e)}

@router.get("/quality/report", summary="生成质量报告")
async def generate_quality_report(
):
    """生成完整的图谱质量报告"""
    try:
        _, _, _, quality_manager, _, _, _ = await get_graph_components()
        report = await quality_manager.generate_quality_report()
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        return {"success": False, "available": False, "detail": str(e)}

# 性能监控API
@router.get("/performance/stats", summary="获取性能统计")
async def get_performance_stats(
):
    """获取系统性能统计"""
    try:
        _, _, _, _, performance_optimizer, _, _ = await get_graph_components()
        stats = await performance_optimizer.get_performance_stats()
        return {
            "success": True,
            "stats": stats.to_dict()
        }
        
    except Exception as e:
        return {"success": False, "available": False, "detail": str(e)}

@router.get("/performance/slow-queries", summary="获取慢查询")
async def get_slow_queries(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制")
):
    """获取慢查询列表"""
    try:
        _, _, _, _, performance_optimizer, _, _ = await get_graph_components()
        slow_queries = await performance_optimizer.get_slow_queries(limit)
        return {
            "success": True,
            "slow_queries": [q.to_dict() for q in slow_queries]
        }
        
    except Exception as e:
        return {"success": False, "available": False, "slow_queries": [], "detail": str(e)}

@router.delete("/performance/cache", summary="清理查询缓存")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="缓存键匹配模式"),
    components: tuple = Depends(get_graph_components)
):
    """清理查询缓存"""
    _, _, _, _, performance_optimizer, _, _ = components
    
    try:
        await performance_optimizer.invalidate_cache(pattern)
        return {
            "success": True,
            "message": f"缓存清理完成，模式: {pattern or '全部'}"
        }
        
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 统计信息API
@router.get("/statistics", summary="获取图谱统计")
async def get_graph_statistics(
):
    """获取图谱基础统计信息"""
    try:
        graph_ops, _, _, _, _, _, _ = await get_graph_components()
        result = await graph_ops.get_graph_statistics()
        
        if result.success:
            return {
                "success": True,
                "statistics": result.data[0]["statistics"],
                "execution_time_ms": result.execution_time_ms
            }
        else:
            raise HTTPException(status_code=400, detail=result.error_message)
            
    except Exception as e:
        return {"success": False, "available": False, "detail": str(e)}

# 健康检查API
@router.get("/health", summary="系统健康检查")
async def health_check(
):
    """系统健康检查"""
    try:
        graph_db = await get_graph_database()
        health_status = await graph_db.check_health()
        
        return {
            "success": True,
            "health": health_status,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 迁移管理API
@router.get("/migrations", summary="获取迁移列表")
async def list_migrations(
):
    """返回所有已注册迁移及其当前状态"""
    try:
        _, _, _, _, _, migration_manager, _ = await get_graph_components()

        applied = await migration_manager.get_applied_migrations()
        status_map = {
            record.migration_id: record.status.value
            for record in migration_manager.migration_records
        }

        migrations = []
        for migration_id, migration in migration_manager.migrations.items():
            status = status_map.get(
                migration_id,
                "completed" if migration_id in applied else "pending",
            )
            migration_dict = migration.to_dict()
            migration_dict["status"] = status
            migrations.append(migration_dict)

        return {"migrations": migrations}
    except Exception as e:
        return {"migrations": [], "available": False, "detail": str(e)}

@router.get("/migrations/records", summary="获取迁移记录")
async def get_migration_records(
):
    """返回迁移执行记录"""
    try:
        _, _, _, _, _, migration_manager, _ = await get_graph_components()
        return {"records": [record.to_dict() for record in migration_manager.migration_records]}
    except Exception as e:
        return {"records": [], "available": False, "detail": str(e)}

@router.post("/migrations/{migration_id}/apply", summary="执行指定迁移")
async def apply_migration(
    migration_id: str,
    components: tuple = Depends(get_graph_components)
):
    """执行单个迁移"""
    _, _, _, _, _, migration_manager, _ = components
    migration = migration_manager.migrations.get(migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail="Migration not found")
    
    record = await migration_manager.apply_migration(migration)
    return record.to_dict()

@router.post("/migrations/apply-all", summary="执行所有待迁移")
async def apply_all_migrations(
    components: tuple = Depends(get_graph_components)
):
    """顺序执行所有待处理的迁移"""
    _, _, _, _, _, migration_manager, _ = components
    records = await migration_manager.apply_all_pending_migrations()
    return {"records": [record.to_dict() for record in records]}

@router.post("/migrations/{migration_id}/rollback", summary="回滚迁移")
async def rollback_migration(
    migration_id: str,
    components: tuple = Depends(get_graph_components)
):
    """回滚指定迁移"""
    _, _, _, _, _, migration_manager, _ = components
    migration = migration_manager.migrations.get(migration_id)
    if not migration:
        raise HTTPException(status_code=404, detail="Migration not found")
    
    record = await migration_manager.rollback_migration(migration_id)
    return record.to_dict()

# 模式管理API（管理员功能）
@router.post("/admin/schema/initialize", summary="初始化图谱模式")
async def initialize_schema(
    components: tuple = Depends(get_graph_components)
):
    """初始化图谱模式（管理员功能）"""
    _, schema_manager, _, _, _, _, _ = components
    
    try:
        result = await schema_manager.initialize_schema()
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"初始化图谱模式失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/schema/status", summary="获取模式状态")
async def get_schema_status(
):
    """获取图谱模式状态"""
    try:
        _, schema_manager, _, _, _, _, _ = await get_graph_components()
        status = await schema_manager.validate_schema()
        return {
            "success": True,
            "schema_status": status
        }
        
    except Exception as e:
        return {"success": False, "available": False, "detail": str(e)}
