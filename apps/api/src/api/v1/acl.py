"""
ACL (Access Control List) 协议管理 API
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import Response
from pydantic import Field
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/acl", tags=["ACL"])

_RULE_KEY_PREFIX = "acl:rule:"
_RULE_INDEX_KEY = "acl:rules"
_METRICS_KEY = "acl:metrics"

def _rule_key(rule_id: str) -> str:
    return f"{_RULE_KEY_PREFIX}{rule_id}"

def _get_redis():
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化")
    return redis_client

async def _load_rule(rule_id: str) -> Optional["ACLRule"]:
    redis_client = _get_redis()
    raw = await redis_client.get(_rule_key(rule_id))
    if not raw:
        return None
    data = json.loads(raw)
    return ACLRule(**data)

async def _save_rule(rule: "ACLRule") -> None:
    redis_client = _get_redis()
    await redis_client.set(_rule_key(rule.id), json.dumps(rule.model_dump(), ensure_ascii=False))
    await redis_client.zadd(_RULE_INDEX_KEY, {rule.id: float(rule.priority)})

async def _list_rules() -> List["ACLRule"]:
    redis_client = _get_redis()
    rule_ids = await redis_client.zrange(_RULE_INDEX_KEY, 0, -1)
    if not rule_ids:
        return []
    pipeline = redis_client.pipeline()
    for rule_id in rule_ids:
        pipeline.get(_rule_key(rule_id))
    raws = await pipeline.execute()
    rules: List[ACLRule] = []
    for raw in raws:
        if not raw:
            continue
        rules.append(ACLRule(**json.loads(raw)))
    return rules

async def _get_metrics() -> "SecurityMetrics":
    redis_client = _get_redis()
    rules = await _list_rules()
    active_rules = sum(1 for r in rules if r.status == "active")
    metrics_raw = await redis_client.hgetall(_METRICS_KEY)
    blocked_requests = int(metrics_raw.get("blocked_requests", 0) or 0)
    allowed_requests = int(metrics_raw.get("allowed_requests", 0) or 0)
    violation_count = int(metrics_raw.get("violation_count", 0) or 0)
    last_violation_time = metrics_raw.get("last_violation_time") or None
    return SecurityMetrics(
        total_rules=len(rules),
        active_rules=active_rules,
        blocked_requests=blocked_requests,
        allowed_requests=allowed_requests,
        violation_count=violation_count,
        last_violation_time=last_violation_time,
    )

def _match_target(rule_target: str, target: str) -> bool:
    if rule_target.endswith(".>"):
        return target.startswith(rule_target[:-2])
    if rule_target.endswith(">"):
        return target.startswith(rule_target[:-1])
    return target == rule_target

# 数据模型
class CreateACLRuleRequest(ApiBaseModel):
    name: str = Field(..., description="规则名称")
    description: str = Field(..., description="规则描述") 
    source: str = Field(..., description="源")
    target: str = Field(..., description="目标")
    action: str = Field(..., description="动作：allow/deny")
    conditions: List[str] = Field(default_factory=list, description="条件列表")
    priority: int = Field(default=100, description="优先级")

class UpdateACLRuleRequest(ApiBaseModel):
    name: Optional[str] = Field(None, description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    source: Optional[str] = Field(None, description="源")
    target: Optional[str] = Field(None, description="目标")
    action: Optional[str] = Field(None, description="动作：allow/deny")
    conditions: Optional[List[str]] = Field(None, description="条件列表")
    priority: Optional[int] = Field(None, description="优先级")
    status: Optional[str] = Field(None, description="状态：active/inactive")

class ACLRule(ApiBaseModel):
    id: str
    name: str
    description: str
    source: str
    target: str
    action: str
    conditions: List[str]
    priority: int
    status: str
    created_at: str
    updated_at: str

class SecurityMetrics(ApiBaseModel):
    total_rules: int
    active_rules: int
    blocked_requests: int
    allowed_requests: int
    violation_count: int
    last_violation_time: Optional[str]

class ValidationResult(ApiBaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    conflicts: List[ACLRule]

@router.get("/rules", response_model=List[ACLRule])
async def list_rules():
    """获取ACL规则列表"""
    return await _list_rules()

@router.get("/rules/{rule_id}", response_model=ACLRule)
async def get_rule(rule_id: str):
    """获取单个ACL规则"""
    rule = await _load_rule(rule_id)
    if rule:
        return rule
    raise HTTPException(status_code=404, detail="规则未找到")

@router.post("/rules", response_model=ACLRule)
async def create_rule(data: CreateACLRuleRequest):
    """创建新的ACL规则"""
    now = utc_now().isoformat()
    new_rule = ACLRule(
        id=f"rule-{uuid.uuid4().hex[:8]}",
        name=data.name,
        description=data.description,
        source=data.source,
        target=data.target,
        action=data.action,
        conditions=data.conditions,
        priority=data.priority,
        status="active",
        created_at=now,
        updated_at=now,
    )
    await _save_rule(new_rule)
    return new_rule

@router.put("/rules/{rule_id}", response_model=ACLRule)
async def update_rule(rule_id: str, data: UpdateACLRuleRequest):
    """更新ACL规则"""
    rule = await _load_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则未找到")
    updated_rule = rule.copy()
    if data.name is not None:
        updated_rule.name = data.name
    if data.description is not None:
        updated_rule.description = data.description
    if data.source is not None:
        updated_rule.source = data.source
    if data.target is not None:
        updated_rule.target = data.target
    if data.action is not None:
        updated_rule.action = data.action
    if data.conditions is not None:
        updated_rule.conditions = data.conditions
    if data.priority is not None:
        updated_rule.priority = data.priority
    if data.status is not None:
        updated_rule.status = data.status
    updated_rule.updated_at = utc_now().isoformat()
    await _save_rule(updated_rule)
    return updated_rule

@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """删除ACL规则"""
    redis_client = _get_redis()
    if not await redis_client.delete(_rule_key(rule_id)):
        raise HTTPException(status_code=404, detail="规则未找到")
    await redis_client.zrem(_RULE_INDEX_KEY, rule_id)
    return {"message": "规则删除成功"}

@router.get("/metrics", response_model=SecurityMetrics)
async def get_security_metrics():
    """获取安全指标"""
    return await _get_metrics()

@router.post("/validate", response_model=ValidationResult)
async def validate_rule(data: CreateACLRuleRequest):
    """验证ACL规则"""
    errors = []
    warnings = []
    conflicts = []
    
    # 基本验证
    if not data.name.strip():
        errors.append("规则名称不能为空")
    
    if not data.source.strip():
        errors.append("源不能为空")
        
    if not data.target.strip():
        errors.append("目标不能为空")
        
    if data.action not in ["allow", "deny"]:
        errors.append("动作必须是allow或deny")
    
    # 检查冲突
    for rule in await _list_rules():
        if rule.name == data.name:
            conflicts.append(rule)
            warnings.append(f"规则名称与现有规则 {rule.id} 冲突")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        conflicts=conflicts
    )

@router.post("/rules/{rule_id}/test")
async def test_rule(rule_id: str, test_data: Dict[str, Any]):
    """测试ACL规则"""
    redis_client = _get_redis()
    rule = await _load_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则未找到")
    if rule.status != "active":
        return {"allowed": False, "matched_rule": rule, "reason": "规则未启用"}
    source = str(test_data.get("source", ""))
    target = str(test_data.get("target", ""))
    if source != rule.source or not _match_target(rule.target, target):
        return {"allowed": False, "matched_rule": rule, "reason": "输入不匹配规则条件"}
    allowed = rule.action == "allow"
    if allowed:
        await redis_client.hincrby(_METRICS_KEY, "allowed_requests", 1)
    else:
        await redis_client.hincrby(_METRICS_KEY, "blocked_requests", 1)
        await redis_client.hincrby(_METRICS_KEY, "violation_count", 1)
        await redis_client.hset(_METRICS_KEY, "last_violation_time", utc_now().isoformat())
    return {"allowed": allowed, "matched_rule": rule, "reason": f"规则 {rule.name} 匹配，动作为 {rule.action}"}

@router.get("/export")
async def export_rules(format: str = Query("json")):
    """导出ACL规则"""
    if format == "json":
        rules = [rule.model_dump() for rule in await _list_rules()]
        content = json.dumps({"rules": rules}, ensure_ascii=False, indent=2)
        return Response(
            content=content.encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=acl_rules.json"},
        )
    raise HTTPException(status_code=400, detail="仅支持json")

@router.post("/import")
async def import_rules(file: UploadFile = File(...)):
    """导入ACL规则"""
    redis_client = _get_redis()
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="仅支持JSON文件")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise HTTPException(status_code=400, detail="JSON格式错误：缺少rules数组")
    imported = 0
    failed = 0
    errors: List[str] = []
    for idx, item in enumerate(rules):
        if not isinstance(item, dict):
            failed += 1
            errors.append(f"第{idx}条规则格式错误")
            continue
        if not item.get("id"):
            failed += 1
            errors.append(f"第{idx}条规则缺少id")
            continue
        try:
            rule = ACLRule(**item)
        except Exception as e:
            failed += 1
            errors.append(f"第{idx}条规则解析失败: {e}")
            continue
        await redis_client.set(_rule_key(rule.id), json.dumps(rule.model_dump(), ensure_ascii=False))
        await redis_client.zadd(_RULE_INDEX_KEY, {rule.id: float(rule.priority)})
        imported += 1
    return {"imported": imported, "failed": failed, "errors": errors}

@router.get("/health")
async def health_check():
    """健康检查"""
    metrics = await _get_metrics()
    return {
        "status": "healthy",
        "total_rules": metrics.total_rules,
        "active_rules": metrics.active_rules,
        "timestamp": utc_now().isoformat(),
    }
