"""
知识图谱管理API

提供完整的知识图谱管理RESTful API接口：
- 实体和关系CRUD操作
- 批量操作和事务支持
- 图谱结构验证和一致性检查
- API文档和OpenAPI规范集成
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from typing import Dict, List, Any, Optional, Union
from pydantic import Field, field_validator
from enum import Enum
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import uuid
from src.ai.knowledge_graph.sparql_engine import (
    default_formatter,
    ResultFormat
)
from src.ai.knowledge_graph.performance_monitor import (
    default_performance_monitor

)

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/kg", tags=["Knowledge Graph Management"])

class EntityType(str, Enum):
    """实体类型"""
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"
    DOCUMENT = "document"
    CUSTOM = "custom"

class RelationType(str, Enum):
    """关系类型"""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    CAUSED_BY = "caused_by"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CUSTOM = "custom"

class OperationType(str, Enum):
    """操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VALIDATE = "validate"

# Pydantic模型
class EntityBase(ApiBaseModel):
    """实体基础模型"""
    name: str = Field(..., description="实体名称")
    entity_type: EntityType = Field(..., description="实体类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class EntityCreate(EntityBase):
    """创建实体模型"""
    uri: Optional[str] = Field(None, description="实体URI，为空时自动生成")
    
    @field_validator('uri')
    def validate_uri(cls, v):
        if v and not v.startswith('http'):
            raise ValueError('URI必须是有效的HTTP(S) URL')
        return v

class EntityUpdate(ApiBaseModel):
    """更新实体模型"""
    name: Optional[str] = Field(None, description="实体名称")
    entity_type: Optional[EntityType] = Field(None, description="实体类型")
    properties: Optional[Dict[str, Any]] = Field(None, description="实体属性")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class EntityResponse(EntityBase):
    """实体响应模型"""
    uri: str = Field(..., description="实体URI")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    version: int = Field(..., description="版本号")

class RelationBase(ApiBaseModel):
    """关系基础模型"""
    subject_uri: str = Field(..., description="主语实体URI")
    predicate: str = Field(..., description="关系谓词")
    object_uri: str = Field(..., description="客语实体URI")
    relation_type: RelationType = Field(..., description="关系类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class RelationCreate(RelationBase):
    """创建关系模型"""
    ...

class RelationUpdate(ApiBaseModel):
    """更新关系模型"""
    predicate: Optional[str] = Field(None, description="关系谓词")
    relation_type: Optional[RelationType] = Field(None, description="关系类型")
    properties: Optional[Dict[str, Any]] = Field(None, description="关系属性")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class RelationResponse(RelationBase):
    """关系响应模型"""
    relation_id: str = Field(..., description="关系ID")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    version: int = Field(..., description="版本号")

class BatchOperation(ApiBaseModel):
    """批量操作模型"""
    operation_type: OperationType = Field(..., description="操作类型")
    entities: Optional[List[Union[EntityCreate, EntityUpdate]]] = Field(None, description="实体列表")
    relations: Optional[List[Union[RelationCreate, RelationUpdate]]] = Field(None, description="关系列表")
    target_uris: Optional[List[str]] = Field(None, description="目标URI列表（用于删除操作）")
    transaction_id: Optional[str] = Field(None, description="事务ID")

class BatchOperationResult(ApiBaseModel):
    """批量操作结果"""
    operation_id: str = Field(..., description="操作ID")
    total_items: int = Field(..., description="总项目数")
    successful_items: int = Field(..., description="成功项目数")
    failed_items: int = Field(..., description="失败项目数")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="错误列表")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="结果列表")
    execution_time_ms: float = Field(..., description="执行时间（毫秒）")

class ValidationRule(ApiBaseModel):
    """验证规则"""
    rule_id: str = Field(..., description="规则ID")
    rule_type: str = Field(..., description="规则类型")
    rule_query: str = Field(..., description="验证查询")
    error_message: str = Field(..., description="错误消息")
    enabled: bool = Field(True, description="是否启用")

class ValidationResult(ApiBaseModel):
    """验证结果"""
    valid: bool = Field(..., description="是否通过验证")
    violations: List[Dict[str, Any]] = Field(default_factory=list, description="违规列表")
    checked_rules: int = Field(..., description="检查的规则数")
    execution_time_ms: float = Field(..., description="执行时间")

# 实体管理API
@router.get(
    "/entities",
    response_model=Dict[str, Any],
    summary="查询实体列表",
    description="根据条件查询实体列表，支持分页和过滤"
)
async def list_entities(
    entity_type: Optional[EntityType] = Query(None, description="实体类型过滤"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    format: ResultFormat = Query(ResultFormat.JSON, description="返回格式")
):
    """查询实体列表"""
    try:
        # 构建SPARQL查询
        filters = []
        
        if entity_type:
            filters.append(f'?entity rdf:type <{entity_type.value}>')
        
        if search:
            filters.append(f'FILTER(CONTAINS(LCASE(?name), "{search.lower()}"))')
        
        where_clause = " . ".join([
            "?entity rdf:type ?type",
            "?entity rdfs:label ?name"
        ] + filters)
        
        query_text = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?type ?name
        WHERE {{
            {where_clause}
        }}
        ORDER BY ?name
        LIMIT {limit}
        OFFSET {offset}
        """
        
        # 执行查询
        result = await execute_sparql_query(
            query_text,
            QueryType.SELECT,
            timeout_seconds=30
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"查询失败: {result.error_message}"
            )
        
        # 格式化结果
        formatted_result = default_formatter.format_results(
            result.results,
            result.result_type,
            format
        )
        
        return {
            "entities": formatted_result["data"],
            "metadata": {
                "total_count": result.row_count,
                "limit": limit,
                "offset": offset,
                "execution_time_ms": result.execution_time_ms,
                "format": format
            }
        }
        
    except Exception as e:
        logger.error(f"查询实体列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/entities",
    response_model=EntityResponse,
    status_code=201,
    summary="创建实体",
    description="创建新的知识图谱实体"
)
async def create_entity(entity: EntityCreate):
    """创建新实体"""
    try:
        # 生成URI（如果未提供）
        if not entity.uri:
            entity.uri = f"http://example.org/entity/{uuid.uuid4()}"
        
        # 检查实体是否已存在
        check_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        ASK WHERE {{ <{entity.uri}> ?p ?o }}
        """
        
        check_result = await execute_sparql_query(
            check_query,
            QueryType.ASK
        )
        
        if check_result.success and check_result.results and check_result.results[0].get("result"):
            raise HTTPException(
                status_code=409,
                detail=f"实体已存在: {entity.uri}"
            )
        
        # 创建实体的SPARQL UPDATE
        current_time = utc_now()
        
        # 构建属性三元组
        property_triples = []
        for key, value in entity.properties.items():
            property_triples.append(
                f'<{entity.uri}> <http://example.org/property/{key}> "{value}" .'
            )
        
        insert_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX ex: <http://example.org/>
        
        INSERT DATA {{
            <{entity.uri}> rdf:type <http://example.org/type/{entity.entity_type.value}> .
            <{entity.uri}> rdfs:label "{entity.name}" .
            <{entity.uri}> ex:createdAt "{current_time.isoformat()}"^^xsd:dateTime .
            <{entity.uri}> ex:updatedAt "{current_time.isoformat()}"^^xsd:dateTime .
            <{entity.uri}> ex:version "1"^^xsd:integer .
            {' '.join(property_triples)}
        }}
        """
        
        # 执行插入
        insert_result = await execute_sparql_query(
            insert_query,
            QueryType.UPDATE
        )
        
        if not insert_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"创建实体失败: {insert_result.error_message}"
            )
        
        # 返回创建的实体
        return EntityResponse(
            uri=entity.uri,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties,
            metadata=entity.metadata,
            created_at=current_time,
            updated_at=current_time,
            version=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建实体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/entities/{entity_uri:path}",
    response_model=EntityResponse,
    summary="获取实体详情",
    description="根据URI获取实体的详细信息"
)
async def get_entity(entity_uri: str):
    """获取实体详情"""
    try:
        query_text = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ex: <http://example.org/>
        
        SELECT ?type ?name ?createdAt ?updatedAt ?version ?property ?value
        WHERE {{
            <{entity_uri}> rdf:type ?type .
            <{entity_uri}> rdfs:label ?name .
            OPTIONAL {{ <{entity_uri}> ex:createdAt ?createdAt }}
            OPTIONAL {{ <{entity_uri}> ex:updatedAt ?updatedAt }}
            OPTIONAL {{ <{entity_uri}> ex:version ?version }}
            OPTIONAL {{ 
                <{entity_uri}> ?property ?value .
                FILTER(STRSTARTS(STR(?property), "http://example.org/property/"))
            }}
        }}
        """
        
        result = await execute_sparql_query(
            query_text,
            QueryType.SELECT
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"查询失败: {result.error_message}"
            )
        
        if not result.results:
            raise HTTPException(
                status_code=404,
                detail=f"实体不存在: {entity_uri}"
            )
        
        # 解析结果
        entity_data = result.results[0]
        
        # 收集属性
        properties = {}
        for row in result.results:
            if row.get("property") and row.get("value"):
                prop_name = row["property"].split("/")[-1]
                properties[prop_name] = row["value"]
        
        # 解析实体类型
        entity_type_uri = entity_data.get("type", "")
        entity_type_name = entity_type_uri.split("/")[-1] if entity_type_uri else "custom"
        
        try:
            entity_type = EntityType(entity_type_name)
        except ValueError:
            entity_type = EntityType.CUSTOM
        
        return EntityResponse(
            uri=entity_uri,
            name=entity_data.get("name", ""),
            entity_type=entity_type,
            properties=properties,
            metadata={},
            created_at=datetime.fromisoformat(
                entity_data.get("createdAt", utc_now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                entity_data.get("updatedAt", utc_now().isoformat())
            ),
            version=int(entity_data.get("version", 1))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实体详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put(
    "/entities/{entity_uri:path}",
    response_model=EntityResponse,
    summary="更新实体",
    description="更新现有实体的信息"
)
async def update_entity(entity_uri: str, update_data: EntityUpdate):
    """更新实体"""
    try:
        # 检查实体是否存在
        existing_entity = await get_entity(entity_uri)
        
        # 构建更新查询
        delete_clauses = []
        insert_clauses = []
        current_time = utc_now()
        
        if update_data.name is not None:
            delete_clauses.append(f"<{entity_uri}> rdfs:label ?oldName")
            insert_clauses.append(f'<{entity_uri}> rdfs:label "{update_data.name}"')
        
        if update_data.entity_type is not None:
            delete_clauses.append(f"<{entity_uri}> rdf:type ?oldType")
            insert_clauses.append(
                f'<{entity_uri}> rdf:type <http://example.org/type/{update_data.entity_type.value}>'
            )
        
        if update_data.properties is not None:
            # 删除旧属性
            delete_clauses.append(
                f"<{entity_uri}> ?property ?oldValue . "
                f"FILTER(STRSTARTS(STR(?property), \"http://example.org/property/\"))"
            )
            
            # 插入新属性
            for key, value in update_data.properties.items():
                insert_clauses.append(
                    f'<{entity_uri}> <http://example.org/property/{key}> "{value}"'
                )
        
        # 更新时间戳和版本
        delete_clauses.append(f"<{entity_uri}> ex:updatedAt ?oldUpdatedAt")
        delete_clauses.append(f"<{entity_uri}> ex:version ?oldVersion")
        
        insert_clauses.append(f'<{entity_uri}> ex:updatedAt "{current_time.isoformat()}"^^xsd:dateTime')
        insert_clauses.append(f'<{entity_uri}> ex:version "{existing_entity.version + 1}"^^xsd:integer')
        
        if delete_clauses and insert_clauses:
            update_query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX ex: <http://example.org/>
            
            DELETE {{
                {' . '.join(delete_clauses)}
            }}
            INSERT {{
                {' . '.join(insert_clauses)}
            }}
            WHERE {{
                {' . '.join(delete_clauses)}
            }}
            """
            
            result = await execute_sparql_query(
                update_query,
                QueryType.UPDATE
            )
            
            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"更新实体失败: {result.error_message}"
                )
        
        # 返回更新后的实体
        return await get_entity(entity_uri)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新实体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    "/entities/{entity_uri:path}",
    status_code=204,
    summary="删除实体",
    description="删除指定的实体及其相关关系"
)
async def delete_entity(entity_uri: str):
    """删除实体"""
    try:
        # 检查实体是否存在
        await get_entity(entity_uri)
        
        # 删除实体及其所有相关数据
        delete_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        DELETE {{
            <{entity_uri}> ?p ?o .
            ?s ?p2 <{entity_uri}> .
        }}
        WHERE {{
            {{
                <{entity_uri}> ?p ?o .
            }} UNION {{
                ?s ?p2 <{entity_uri}> .
            }}
        }}
        """
        
        result = await execute_sparql_query(
            delete_query,
            QueryType.UPDATE
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"删除实体失败: {result.error_message}"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除实体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 关系管理API
@router.get(
    "/relations",
    response_model=Dict[str, Any],
    summary="查询关系列表",
    description="根据条件查询关系列表"
)
async def list_relations(
    subject_uri: Optional[str] = Query(None, description="主语实体URI"),
    object_uri: Optional[str] = Query(None, description="客语实体URI"),
    predicate: Optional[str] = Query(None, description="关系谓词"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    format: ResultFormat = Query(ResultFormat.JSON, description="返回格式")
):
    """查询关系列表"""
    try:
        # 构建查询条件
        filters = []
        
        subject_var = "?subject"
        if subject_uri:
            subject_var = f"<{subject_uri}>"
        
        object_var = "?object"
        if object_uri:
            object_var = f"<{object_uri}>"
        
        predicate_var = "?predicate"
        if predicate:
            predicate_var = f"<{predicate}>"
        
        query_text = f"""
        SELECT {subject_var} {predicate_var} {object_var}
        WHERE {{
            {subject_var} {predicate_var} {object_var} .
            FILTER(?predicate != rdf:type)
            FILTER(?predicate != rdfs:label)
        }}
        ORDER BY ?subject ?predicate ?object
        LIMIT {limit}
        OFFSET {offset}
        """
        
        # 执行查询
        result = await execute_sparql_query(
            query_text,
            QueryType.SELECT,
            timeout_seconds=30
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"查询失败: {result.error_message}"
            )
        
        # 格式化结果
        formatted_result = default_formatter.format_results(
            result.results,
            result.result_type,
            format
        )
        
        return {
            "relations": formatted_result["data"],
            "metadata": {
                "total_count": result.row_count,
                "limit": limit,
                "offset": offset,
                "execution_time_ms": result.execution_time_ms,
                "format": format
            }
        }
        
    except Exception as e:
        logger.error(f"查询关系列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/relations",
    response_model=RelationResponse,
    status_code=201,
    summary="创建关系",
    description="在两个实体之间创建新的关系"
)
async def create_relation(relation: RelationCreate):
    """创建关系"""
    try:
        # 检查主语和客语实体是否存在
        for entity_uri in [relation.subject_uri, relation.object_uri]:
            try:
                await get_entity(entity_uri)
            except HTTPException as e:
                if e.status_code == 404:
                    raise HTTPException(
                        status_code=400,
                        detail=f"实体不存在: {entity_uri}"
                    )
                raise
        
        # 生成关系ID
        relation_id = str(uuid.uuid4())
        current_time = utc_now()
        
        # 创建关系的SPARQL UPDATE
        insert_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ex: <http://example.org/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        INSERT DATA {{
            <{relation.subject_uri}> <{relation.predicate}> <{relation.object_uri}> .
            <http://example.org/relation/{relation_id}> rdf:type ex:Relation .
            <http://example.org/relation/{relation_id}> ex:subject <{relation.subject_uri}> .
            <http://example.org/relation/{relation_id}> ex:predicate <{relation.predicate}> .
            <http://example.org/relation/{relation_id}> ex:object <{relation.object_uri}> .
            <http://example.org/relation/{relation_id}> ex:relationType <{relation.relation_type.value}> .
            <http://example.org/relation/{relation_id}> ex:createdAt "{current_time.isoformat()}"^^xsd:dateTime .
            <http://example.org/relation/{relation_id}> ex:updatedAt "{current_time.isoformat()}"^^xsd:dateTime .
            <http://example.org/relation/{relation_id}> ex:version "1"^^xsd:integer .
        }}
        """
        
        # 执行插入
        result = await execute_sparql_query(
            insert_query,
            QueryType.UPDATE
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"创建关系失败: {result.error_message}"
            )
        
        return RelationResponse(
            relation_id=relation_id,
            subject_uri=relation.subject_uri,
            predicate=relation.predicate,
            object_uri=relation.object_uri,
            relation_type=relation.relation_type,
            properties=relation.properties,
            metadata=relation.metadata,
            created_at=current_time,
            updated_at=current_time,
            version=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建关系失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 批量操作API
@router.post(
    "/batch",
    response_model=BatchOperationResult,
    summary="批量操作",
    description="执行批量实体和关系操作"
)
async def batch_operations(
    operations: BatchOperation,
    background_tasks: BackgroundTasks
):
    """批量操作"""
    try:
        import time
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        total_items = 0
        successful_items = 0
        failed_items = 0
        errors = []
        results = []
        
        # 实体操作
        if operations.entities:
            total_items += len(operations.entities)
            
            for i, entity in enumerate(operations.entities):
                try:
                    if operations.operation_type == OperationType.CREATE:
                        result = await create_entity(entity)
                        results.append({"type": "entity", "operation": "create", "uri": result.uri})
                        successful_items += 1
                    elif operations.operation_type == OperationType.UPDATE and hasattr(entity, 'uri'):
                        result = await update_entity(entity.uri, entity)
                        results.append({"type": "entity", "operation": "update", "uri": result.uri})
                        successful_items += 1
                    
                except Exception as e:
                    failed_items += 1
                    errors.append({
                        "index": i,
                        "type": "entity",
                        "error": str(e)
                    })
        
        # 关系操作
        if operations.relations:
            total_items += len(operations.relations)
            
            for i, relation in enumerate(operations.relations):
                try:
                    if operations.operation_type == OperationType.CREATE:
                        result = await create_relation(relation)
                        results.append({"type": "relation", "operation": "create", "id": result.relation_id})
                        successful_items += 1
                    
                except Exception as e:
                    failed_items += 1
                    errors.append({
                        "index": i,
                        "type": "relation", 
                        "error": str(e)
                    })
        
        # 删除操作
        if operations.operation_type == OperationType.DELETE and operations.target_uris:
            total_items += len(operations.target_uris)
            
            for i, uri in enumerate(operations.target_uris):
                try:
                    await delete_entity(uri)
                    results.append({"type": "entity", "operation": "delete", "uri": uri})
                    successful_items += 1
                    
                except Exception as e:
                    failed_items += 1
                    errors.append({
                        "index": i,
                        "type": "entity",
                        "error": str(e)
                    })
        
        execution_time = (time.time() - start_time) * 1000
        
        return BatchOperationResult(
            operation_id=operation_id,
            total_items=total_items,
            successful_items=successful_items,
            failed_items=failed_items,
            errors=errors,
            results=results,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"批量操作失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 图谱验证API
@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="图谱验证",
    description="验证知识图谱的结构和一致性"
)
async def validate_graph(
    validation_rules: Optional[List[ValidationRule]] = None
):
    """验证图谱"""
    try:
        import time
        start_time = time.time()
        
        violations = []
        checked_rules = 0
        
        # 默认验证规则
        default_rules = [
            ValidationRule(
                rule_id="orphaned_entities",
                rule_type="structural",
                rule_query="""
                SELECT ?entity WHERE {
                    ?entity rdf:type ?type .
                    FILTER NOT EXISTS { ?entity ?p ?o . FILTER(?p != rdf:type) }
                    FILTER NOT EXISTS { ?s ?p2 ?entity . FILTER(?p2 != rdf:type) }
                }
                """,
                error_message="发现孤立实体"
            ),
            ValidationRule(
                rule_id="missing_labels",
                rule_type="completeness",
                rule_query="""
                SELECT ?entity WHERE {
                    ?entity rdf:type ?type .
                    FILTER NOT EXISTS { ?entity rdfs:label ?label }
                }
                """,
                error_message="实体缺少标签"
            )
        ]
        
        # 使用提供的规则或默认规则
        rules_to_check = validation_rules if validation_rules else default_rules
        
        for rule in rules_to_check:
            if not rule.enabled:
                continue
            
            try:
                result = await execute_sparql_query(
                    rule.rule_query,
                    QueryType.SELECT
                )
                
                checked_rules += 1
                
                if result.success and result.results:
                    violations.append({
                        "rule_id": rule.rule_id,
                        "rule_type": rule.rule_type,
                        "message": rule.error_message,
                        "count": len(result.results),
                        "details": result.results[:10]  # 限制详情数量
                    })
                
            except Exception as e:
                logger.error(f"验证规则 {rule.rule_id} 执行失败: {e}")
        
        execution_time = (time.time() - start_time) * 1000
        
        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations,
            checked_rules=checked_rules,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"图谱验证失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 图谱模式API
@router.get(
    "/schema",
    response_model=Dict[str, Any],
    summary="获取图谱模式",
    description="获取知识图谱的模式信息"
)
async def get_graph_schema():
    """获取图谱模式"""
    try:
        # 查询实体类型
        types_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT DISTINCT ?type (COUNT(?entity) as ?count)
        WHERE {
            ?entity rdf:type ?type .
            FILTER(?type != rdf:type)
        }
        GROUP BY ?type
        ORDER BY DESC(?count)
        """
        
        # 查询谓词
        predicates_query = """
        SELECT DISTINCT ?predicate (COUNT(?s) as ?count)
        WHERE {
            ?s ?predicate ?o .
            FILTER(?predicate != rdf:type)
            FILTER(?predicate != rdfs:label)
        }
        GROUP BY ?predicate
        ORDER BY DESC(?count)
        """
        
        # 执行查询
        types_result = await execute_sparql_query(types_query, QueryType.SELECT)
        predicates_result = await execute_sparql_query(predicates_query, QueryType.SELECT)
        
        schema_info = {
            "entity_types": types_result.results if types_result.success else [],
            "predicates": predicates_result.results if predicates_result.success else [],
            "total_types": len(types_result.results) if types_result.success else 0,
            "total_predicates": len(predicates_result.results) if predicates_result.success else 0,
            "generated_at": utc_now().isoformat()
        }
        
        return schema_info
        
    except Exception as e:
        logger.error(f"获取图谱模式失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 性能统计API
@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="获取性能统计",
    description="获取知识图谱API的性能统计信息"
)
async def get_performance_stats():
    """获取性能统计"""
    try:
        # 获取SPARQL引擎统计
        engine_stats = default_sparql_engine.get_statistics()
        
        # 获取性能监控统计
        performance_stats = default_performance_monitor.get_performance_summary()
        
        return {
            "sparql_engine": engine_stats,
            "performance_monitor": performance_stats,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查API
@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="健康检查",
    description="检查知识图谱服务的健康状态"
)
async def health_check():
    """健康检查"""
    try:
        # 执行简单查询测试
        test_query = "ASK WHERE { ?s ?p ?o } LIMIT 1"
        
        result = await execute_sparql_query(
            test_query,
            QueryType.ASK,
            timeout_seconds=5
        )
        
        return {
            "status": "healthy" if result.success else "unhealthy",
            "timestamp": utc_now().isoformat(),
            "sparql_engine_available": result.success,
            "execution_time_ms": result.execution_time_ms if result.success else None,
            "error": result.error_message if not result.success else None
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": utc_now().isoformat(),
            "error": str(e)
        }
