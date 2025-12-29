"""
实体管理API（Redis持久化）

提供实体CRUD与统计接口，避免静态假数据。
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field
from typing import Any, Dict, List, Optional
import json
import uuid
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/entities", tags=["entities"])

_ENTITY_INDEX_KEY = "entities:index"
_ENTITY_KEY_PREFIX = "entities:"

class EntityCreateRequest(ApiBaseModel):
    uri: str = Field(..., description="实体URI")
    type: str = Field(..., description="实体类型")
    label: str = Field(..., description="实体标签")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")

class EntityUpdateRequest(ApiBaseModel):
    uri: Optional[str] = Field(None, description="实体URI")
    type: Optional[str] = Field(None, description="实体类型")
    label: Optional[str] = Field(None, description="实体标签")
    properties: Optional[Dict[str, Any]] = Field(None, description="实体属性")
    status: Optional[str] = Field(None, description="实体状态")

class EntityResponse(ApiBaseModel):
    id: str
    uri: str
    type: str
    label: str
    properties: Dict[str, Any]
    created: str
    updated: str
    status: str

class EntityListResponse(ApiBaseModel):
    entities: List[EntityResponse]
    total: int
    limit: int
    offset: int

def _entity_key(entity_id: str) -> str:
    return f"{_ENTITY_KEY_PREFIX}{entity_id}"

async def _require_redis():
    redis = get_redis()
    if not redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis未初始化，实体服务不可用",
        )
    return redis

async def _load_entities(redis) -> List[Dict[str, Any]]:
    ids = await redis.smembers(_ENTITY_INDEX_KEY)
    if not ids:
        return []
    keys = [_entity_key(entity_id) for entity_id in ids]
    raw_list = await redis.mget(keys)
    entities: List[Dict[str, Any]] = []
    for raw in raw_list:
        if not raw:
            continue
        try:
            entities.append(json.loads(raw))
        except Exception:
            continue
    return entities

@router.get("", response_model=EntityListResponse, summary="获取实体列表")
async def list_entities(
    entity_type: Optional[str] = Query(None, alias="type"),
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    redis = await _require_redis()
    entities = await _load_entities(redis)

    if entity_type:
        entities = [e for e in entities if e.get("type") == entity_type]
    if status_filter:
        entities = [e for e in entities if e.get("status") == status_filter]
    if search:
        keyword = search.lower()
        entities = [
            e for e in entities
            if keyword in (e.get("label") or "").lower()
            or keyword in (e.get("uri") or "").lower()
        ]

    total = len(entities)
    sliced = entities[offset:offset + limit]
    return {
        "entities": sliced,
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@router.get("/types", summary="获取实体类型列表")
async def get_entity_types():
    redis = await _require_redis()
    entities = await _load_entities(redis)
    types = sorted({e.get("type") for e in entities if e.get("type")})
    return types

@router.get("/stats", summary="获取实体统计信息")
async def get_entity_stats():
    redis = await _require_redis()
    entities = await _load_entities(redis)
    total = len(entities)
    active = sum(1 for e in entities if e.get("status") == "active")
    inactive = sum(1 for e in entities if e.get("status") == "inactive")
    pending = sum(1 for e in entities if e.get("status") == "pending")
    type_distribution: Dict[str, int] = {}
    for entity in entities:
        entity_type = entity.get("type")
        if not entity_type:
            continue
        type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1
    return {
        "total": total,
        "activeCount": active,
        "inactiveCount": inactive,
        "pendingCount": pending,
        "typeDistribution": type_distribution,
    }

@router.get("/{entity_id}", response_model=EntityResponse, summary="获取实体详情")
async def get_entity(entity_id: str):
    redis = await _require_redis()
    raw = await redis.get(_entity_key(entity_id))
    if not raw:
        raise HTTPException(status_code=404, detail="实体不存在")
    try:
        return json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="实体数据损坏")

@router.post("", response_model=EntityResponse, status_code=status.HTTP_201_CREATED, summary="创建实体")
async def create_entity(request: EntityCreateRequest):
    redis = await _require_redis()
    entity_id = f"ent_{uuid.uuid4().hex}"
    now = utc_now().isoformat()
    entity = {
        "id": entity_id,
        "uri": request.uri,
        "type": request.type,
        "label": request.label,
        "properties": request.properties or {},
        "created": now,
        "updated": now,
        "status": "active",
    }
    await redis.set(_entity_key(entity_id), json.dumps(entity, ensure_ascii=False))
    await redis.sadd(_ENTITY_INDEX_KEY, entity_id)
    return entity

@router.put("/{entity_id}", response_model=EntityResponse, summary="更新实体")
async def update_entity(entity_id: str, request: EntityUpdateRequest):
    redis = await _require_redis()
    raw = await redis.get(_entity_key(entity_id))
    if not raw:
        raise HTTPException(status_code=404, detail="实体不存在")
    try:
        entity = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="实体数据损坏")

    if request.uri is not None:
        entity["uri"] = request.uri
    if request.type is not None:
        entity["type"] = request.type
    if request.label is not None:
        entity["label"] = request.label
    if request.properties is not None:
        entity["properties"] = request.properties
    if request.status is not None:
        entity["status"] = request.status
    entity["updated"] = utc_now().isoformat()

    await redis.set(_entity_key(entity_id), json.dumps(entity, ensure_ascii=False))
    await redis.sadd(_ENTITY_INDEX_KEY, entity_id)
    return entity

@router.delete("/{entity_id}", summary="删除实体")
async def delete_entity(entity_id: str):
    redis = await _require_redis()
    deleted = await redis.delete(_entity_key(entity_id))
    await redis.srem(_ENTITY_INDEX_KEY, entity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="实体不存在")
    return {"message": "实体删除成功", "id": entity_id}
