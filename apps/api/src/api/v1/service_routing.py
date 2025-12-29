"""
服务路由管理API（Redis持久化）
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field
from typing import Any, Dict, List, Optional
import json
import uuid
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/service-routing", tags=["service-routing"])

_ROUTING_INDEX_KEY = "service_routing:index"
_ROUTING_KEY_PREFIX = "service_routing:rule:"

class RoutingConditions(ApiBaseModel):
    capability: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: str = ""
    region: List[str] = Field(default_factory=list)
    environment: List[str] = Field(default_factory=list)
    customRules: str = ""

class RoutingTargets(ApiBaseModel):
    agentTypes: List[str] = Field(default_factory=list)
    endpoints: List[str] = Field(default_factory=list)
    loadBalanceStrategy: str = "round_robin"
    failoverEnabled: bool = False
    circuitBreakerEnabled: bool = False

class RoutingMetrics(ApiBaseModel):
    requestCount: int = 0
    successRate: float = 0
    avgResponseTime: float = 0
    errorCount: int = 0
    lastUsed: str = ""

class RoutingRuleCreate(ApiBaseModel):
    name: str
    description: str = ""
    status: str = "active"
    priority: int = 50
    conditions: RoutingConditions = Field(default_factory=RoutingConditions)
    targets: RoutingTargets = Field(default_factory=RoutingTargets)
    owner: str = "system"

class RoutingRuleUpdate(ApiBaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[int] = None
    conditions: Optional[RoutingConditions] = None
    targets: Optional[RoutingTargets] = None
    owner: Optional[str] = None

class RoutingStatusUpdate(ApiBaseModel):
    status: str

async def _require_redis():
    redis = get_redis()
    if not redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis未初始化，服务路由不可用",
        )
    return redis

def _rule_key(rule_id: str) -> str:
    return f"{_ROUTING_KEY_PREFIX}{rule_id}"

async def _load_rules(redis) -> List[Dict[str, Any]]:
    ids = await redis.smembers(_ROUTING_INDEX_KEY)
    if not ids:
        return []
    keys = [_rule_key(rule_id) for rule_id in ids]
    raw_list = await redis.mget(keys)
    rules: List[Dict[str, Any]] = []
    for raw in raw_list:
        if not raw:
            continue
        try:
            rules.append(json.loads(raw))
        except Exception:
            continue
    return rules

@router.get("/rules")
async def list_routing_rules(
    status_filter: Optional[str] = Query(None, alias="status"),
):
    redis = await _require_redis()
    rules = await _load_rules(redis)
    if status_filter:
        rules = [rule for rule in rules if rule.get("status") == status_filter]
    rules.sort(key=lambda item: item.get("priority", 0), reverse=True)
    return {"rules": rules}

@router.post("/rules", status_code=status.HTTP_201_CREATED)
async def create_routing_rule(request: RoutingRuleCreate):
    redis = await _require_redis()
    rule_id = f"rule_{uuid.uuid4().hex[:8]}"
    now = utc_now().isoformat()
    rule = {
        "id": rule_id,
        "name": request.name,
        "description": request.description,
        "status": request.status,
        "priority": request.priority,
        "conditions": request.conditions.model_dump(),
        "targets": request.targets.model_dump(),
        "metrics": RoutingMetrics().model_dump(),
        "created": now,
        "updated": now,
        "owner": request.owner,
    }
    await redis.set(_rule_key(rule_id), json.dumps(rule, ensure_ascii=False))
    await redis.sadd(_ROUTING_INDEX_KEY, rule_id)
    return rule

@router.put("/rules/{rule_id}")
async def update_routing_rule(rule_id: str, request: RoutingRuleUpdate):
    redis = await _require_redis()
    raw = await redis.get(_rule_key(rule_id))
    if not raw:
        raise HTTPException(status_code=404, detail="路由规则不存在")
    try:
        rule = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="路由规则数据损坏")

    if request.name is not None:
        rule["name"] = request.name
    if request.description is not None:
        rule["description"] = request.description
    if request.status is not None:
        rule["status"] = request.status
    if request.priority is not None:
        rule["priority"] = request.priority
    if request.conditions is not None:
        rule["conditions"] = request.conditions.model_dump()
    if request.targets is not None:
        rule["targets"] = request.targets.model_dump()
    if request.owner is not None:
        rule["owner"] = request.owner

    rule["updated"] = utc_now().isoformat()
    await redis.set(_rule_key(rule_id), json.dumps(rule, ensure_ascii=False))
    await redis.sadd(_ROUTING_INDEX_KEY, rule_id)
    return rule

@router.patch("/rules/{rule_id}/status")
async def update_routing_rule_status(rule_id: str, request: RoutingStatusUpdate):
    redis = await _require_redis()
    raw = await redis.get(_rule_key(rule_id))
    if not raw:
        raise HTTPException(status_code=404, detail="路由规则不存在")
    try:
        rule = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="路由规则数据损坏")

    rule["status"] = request.status
    rule["updated"] = utc_now().isoformat()
    await redis.set(_rule_key(rule_id), json.dumps(rule, ensure_ascii=False))
    await redis.sadd(_ROUTING_INDEX_KEY, rule_id)
    return rule

@router.delete("/rules/{rule_id}")
async def delete_routing_rule(rule_id: str):
    redis = await _require_redis()
    deleted = await redis.delete(_rule_key(rule_id))
    await redis.srem(_ROUTING_INDEX_KEY, rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="路由规则不存在")
    return {"message": "路由规则已删除", "id": rule_id}
