"""
知识图谱存储系统API
提供图谱CRUD、查询、质量管理、性能监控等功能的REST API接口
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import logging
import uuid

from ai.knowledge_graph.graph_database import get_graph_database, Neo4jGraphDatabase
from ai.knowledge_graph.graph_operations import GraphOperations
from ai.knowledge_graph.schema import SchemaManager
from ai.knowledge_graph.incremental_updater import IncrementalUpdater, ConflictResolutionStrategy
from ai.knowledge_graph.quality_manager import QualityManager
from ai.knowledge_graph.performance_optimizer import PerformanceOptimizer
from ai.knowledge_graph.migration_tools import MigrationManager, DataExportImportTool
from ai.knowledge_graph.data_models import Entity, Relation, EntityType, RelationType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])

# 全局实例（在应用启动时初始化）
_graph_ops: Optional[GraphOperations] = None
_schema_manager: Optional[SchemaManager] = None
_incremental_updater: Optional[IncrementalUpdater] = None
_quality_manager: Optional[QualityManager] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None
_migration_manager: Optional[MigrationManager] = None
_export_import_tool: Optional[DataExportImportTool] = None


# Pydantic模型
class CreateEntityRequest(BaseModel):
    """创建实体请求"""
    canonical_form: str = Field(..., description="规范化形式")
    entity_type: str = Field(..., description="实体类型")
    text: Optional[str] = Field(None, description="原始文本")
    confidence: float = Field(1.0, ge=0, le=1, description="置信度")
    language: Optional[str] = Field(None, description="语言")
    linked_entity: Optional[str] = Field(None, description="链接实体URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class UpdateEntityRequest(BaseModel):
    """更新实体请求"""
    canonical_form: Optional[str] = Field(None, description="规范化形式")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="置信度")
    language: Optional[str] = Field(None, description="语言")
    linked_entity: Optional[str] = Field(None, description="链接实体URI")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class CreateRelationRequest(BaseModel):
    """创建关系请求"""
    source_entity_id: str = Field(..., description="源实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    confidence: float = Field(1.0, ge=0, le=1, description="置信度")
    context: str = Field(..., description="关系上下文")
    source_sentence: str = Field(..., description="源句子")
    evidence: List[str] = Field(default_factory=list, description="证据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class EntitySearchRequest(BaseModel):
    """实体搜索请求"""
    canonical_form_contains: Optional[str] = Field(None, description="包含的规范化形式")
    entity_type: Optional[str] = Field(None, description="实体类型")
    confidence_gte: Optional[float] = Field(None, ge=0, le=1, description="最小置信度")
    confidence_lte: Optional[float] = Field(None, ge=0, le=1, description="最大置信度")
    created_after: Optional[datetime] = Field(None, description="创建时间之后")
    created_before: Optional[datetime] = Field(None, description="创建时间之前")
    limit: int = Field(100, ge=1, le=1000, description="返回数量限制")
    skip: int = Field(0, ge=0, description="跳过数量")


class CustomQueryRequest(BaseModel):
    """自定义查询请求"""
    query: str = Field(..., description="Cypher查询语句")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    read_only: bool = Field(True, description="是否只读查询")


class BatchUpsertRequest(BaseModel):
    """批量更新请求"""
    entities: List[Dict[str, Any]] = Field(..., description="实体数据列表")
    conflict_strategy: str = Field("merge_highest_confidence", description="冲突解决策略")


# 依赖注入
async def get_graph_components() -> tuple:
    """获取图数据库相关组件"""
    global _graph_ops, _schema_manager, _incremental_updater, _quality_manager
    global _performance_optimizer, _migration_manager, _export_import_tool
    
    if _graph_ops is None:
        # 初始化组件
        graph_db = await get_graph_database()
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
            
    except Exception as e:
        logger.error(f"获取实体关系失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 图查询API
@router.get("/path", summary="查找最短路径")
async def find_shortest_path(
    source_id: str = Query(..., description="源实体ID"),
    target_id: str = Query(..., description="目标实体ID"),
    max_depth: int = Query(5, ge=1, le=10, description="最大搜索深度"),
    components: tuple = Depends(get_graph_components)
):
    """查找两个实体间的最短路径"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
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
            
    except Exception as e:
        logger.error(f"执行自定义查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 增量更新API
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
        
    except Exception as e:
        logger.error(f"批量智能更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 质量管理API
@router.get("/quality/metrics", summary="获取质量指标")
async def get_quality_metrics(
    components: tuple = Depends(get_graph_components)
):
    """获取图谱质量指标"""
    _, _, _, quality_manager, _, _, _ = components
    
    try:
        metrics = await quality_manager.calculate_quality_score()
        return {
            "success": True,
            "metrics": metrics.to_dict()
        }
        
    except Exception as e:
        logger.error(f"获取质量指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality/issues", summary="检测质量问题")
async def detect_quality_issues(
    components: tuple = Depends(get_graph_components)
):
    """检测图谱质量问题"""
    _, _, _, quality_manager, _, _, _ = components
    
    try:
        issues = await quality_manager.detect_quality_issues()
        return {
            "success": True,
            "total_issues": len(issues),
            "issues": [issue.to_dict() for issue in issues]
        }
        
    except Exception as e:
        logger.error(f"检测质量问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality/report", summary="生成质量报告")
async def generate_quality_report(
    components: tuple = Depends(get_graph_components)
):
    """生成完整的图谱质量报告"""
    _, _, _, quality_manager, _, _, _ = components
    
    try:
        report = await quality_manager.generate_quality_report()
        return {
            "success": True,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"生成质量报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 性能监控API
@router.get("/performance/stats", summary="获取性能统计")
async def get_performance_stats(
    components: tuple = Depends(get_graph_components)
):
    """获取系统性能统计"""
    _, _, _, _, performance_optimizer, _, _ = components
    
    try:
        stats = await performance_optimizer.get_performance_stats()
        return {
            "success": True,
            "stats": stats.to_dict()
        }
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/slow-queries", summary="获取慢查询")
async def get_slow_queries(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    components: tuple = Depends(get_graph_components)
):
    """获取慢查询列表"""
    _, _, _, _, performance_optimizer, _, _ = components
    
    try:
        slow_queries = await performance_optimizer.get_slow_queries(limit)
        return {
            "success": True,
            "slow_queries": [q.to_dict() for q in slow_queries]
        }
        
    except Exception as e:
        logger.error(f"获取慢查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
    components: tuple = Depends(get_graph_components)
):
    """获取图谱基础统计信息"""
    graph_ops, _, _, _, _, _, _ = components
    
    try:
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
        logger.error(f"获取图谱统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查API
@router.get("/health", summary="系统健康检查")
async def health_check(
    components: tuple = Depends(get_graph_components)
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
    components: tuple = Depends(get_graph_components)
):
    """获取图谱模式状态"""
    _, schema_manager, _, _, _, _, _ = components
    
    try:
        status = await schema_manager.validate_schema()
        return {
            "success": True,
            "schema_status": status
        }
        
    except Exception as e:
        logger.error(f"获取模式状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))