"""
服务配置管理API（Redis持久化）
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field
from typing import Any, Dict, List, Optional
import json
import uuid
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/service-config", tags=["service-config"])

_CONFIG_INDEX_KEY = "service_config:index"
_CONFIG_KEY_PREFIX = "service_config:item:"
_TEMPLATE_INDEX_KEY = "service_config:template:index"
_TEMPLATE_KEY_PREFIX = "service_config:template:"
_HISTORY_KEY = "service_config:history"

class ConfigItemBase(ApiBaseModel):
    category: str
    key: str
    value: Any
    type: str = Field(..., description="string|number|boolean|json|array")
    description: str = ""
    required: bool = False
    sensitive: bool = False
    defaultValue: Any = None
    validation: Optional[Dict[str, Any]] = None
    modifiedBy: str = "system"

class ConfigItemCreate(ConfigItemBase):
    ...

class ConfigItemUpdate(ApiBaseModel):
    category: Optional[str] = None
    key: Optional[str] = None
    value: Optional[Any] = None
    type: Optional[str] = None
    description: Optional[str] = None
    required: Optional[bool] = None
    sensitive: Optional[bool] = None
    defaultValue: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None
    modifiedBy: Optional[str] = None

class ConfigTemplateCreate(ApiBaseModel):
    name: str
    description: str = ""
    category: str
    configs: List[Dict[str, Any]] = Field(default_factory=list)
    version: str = "1.0.0"
    author: str = "system"

async def _require_redis():
    redis = get_redis()
    if not redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis未初始化，服务配置不可用",
        )
    return redis

def _config_key(config_id: str) -> str:
    return f"{_CONFIG_KEY_PREFIX}{config_id}"

def _template_key(template_id: str) -> str:
    return f"{_TEMPLATE_KEY_PREFIX}{template_id}"

async def _load_configs(redis) -> List[Dict[str, Any]]:
    ids = await redis.smembers(_CONFIG_INDEX_KEY)
    if not ids:
        return []
    keys = [_config_key(config_id) for config_id in ids]
    raw_list = await redis.mget(keys)
    configs: List[Dict[str, Any]] = []
    for raw in raw_list:
        if not raw:
            continue
        try:
            configs.append(json.loads(raw))
        except Exception:
            continue
    return configs

async def _load_templates(redis) -> List[Dict[str, Any]]:
    ids = await redis.smembers(_TEMPLATE_INDEX_KEY)
    if not ids:
        return []
    keys = [_template_key(template_id) for template_id in ids]
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

async def _append_history(
    redis,
    config_key: str,
    old_value: Any,
    new_value: Any,
    user: str,
    reason: str = "",
    environment: str = "default",
    sensitive: bool = False,
):
    if sensitive:
        old_value = "***" if old_value is not None else None
        new_value = "***" if new_value is not None else None
    entry = {
        "id": f"hist_{uuid.uuid4().hex[:8]}",
        "configKey": config_key,
        "oldValue": old_value,
        "newValue": new_value,
        "timestamp": utc_now().isoformat(),
        "user": user,
        "reason": reason,
        "environment": environment,
    }
    await redis.lpush(_HISTORY_KEY, json.dumps(entry, ensure_ascii=False))
    await redis.ltrim(_HISTORY_KEY, 0, 199)

@router.get("/configs")
async def list_configs(category: Optional[str] = Query(None)):
    redis = await _require_redis()
    configs = await _load_configs(redis)
    if category and category != "all":
        configs = [cfg for cfg in configs if cfg.get("category") == category]
    configs.sort(key=lambda item: item.get("key", ""))
    return {"configs": configs}

@router.post("/configs", status_code=status.HTTP_201_CREATED)
async def create_config(request: ConfigItemCreate):
    redis = await _require_redis()
    configs = await _load_configs(redis)
    if any(cfg.get("key") == request.key for cfg in configs):
        raise HTTPException(status_code=409, detail="配置项已存在")

    config_id = f"cfg_{uuid.uuid4().hex[:8]}"
    now = utc_now().isoformat()
    config = {
        "id": config_id,
        "category": request.category,
        "key": request.key,
        "value": request.value,
        "type": request.type,
        "description": request.description,
        "required": request.required,
        "sensitive": request.sensitive,
        "defaultValue": request.defaultValue,
        "validation": request.validation,
        "lastModified": now,
        "modifiedBy": request.modifiedBy or "system",
    }
    await redis.set(_config_key(config_id), json.dumps(config, ensure_ascii=False))
    await redis.sadd(_CONFIG_INDEX_KEY, config_id)
    await _append_history(
        redis,
        config_key=request.key,
        old_value=None,
        new_value=request.value,
        user=config["modifiedBy"],
        reason="create",
        sensitive=request.sensitive,
    )
    return config

@router.put("/configs/{config_id}")
async def update_config(config_id: str, request: ConfigItemUpdate):
    redis = await _require_redis()
    raw = await redis.get(_config_key(config_id))
    if not raw:
        raise HTTPException(status_code=404, detail="配置项不存在")
    try:
        config = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="配置项数据损坏")

    old_value = config.get("value")
    if request.key is not None and request.key != config.get("key"):
        configs = await _load_configs(redis)
        if any(
            cfg.get("key") == request.key and cfg.get("id") != config_id
            for cfg in configs
        ):
            raise HTTPException(status_code=409, detail="配置键已存在")
    if request.category is not None:
        config["category"] = request.category
    if request.key is not None:
        config["key"] = request.key
    if request.value is not None:
        config["value"] = request.value
    if request.type is not None:
        config["type"] = request.type
    if request.description is not None:
        config["description"] = request.description
    if request.required is not None:
        config["required"] = request.required
    if request.sensitive is not None:
        config["sensitive"] = request.sensitive
    if request.defaultValue is not None:
        config["defaultValue"] = request.defaultValue
    if request.validation is not None:
        config["validation"] = request.validation
    if request.modifiedBy is not None:
        config["modifiedBy"] = request.modifiedBy

    config["lastModified"] = utc_now().isoformat()
    await redis.set(_config_key(config_id), json.dumps(config, ensure_ascii=False))
    await redis.sadd(_CONFIG_INDEX_KEY, config_id)
    await _append_history(
        redis,
        config_key=config.get("key", config_id),
        old_value=old_value,
        new_value=config.get("value"),
        user=config.get("modifiedBy") or "system",
        reason="update",
        sensitive=bool(config.get("sensitive")),
    )
    return config

@router.delete("/configs/{config_id}")
async def delete_config(config_id: str):
    redis = await _require_redis()
    raw = await redis.get(_config_key(config_id))
    if not raw:
        raise HTTPException(status_code=404, detail="配置项不存在")
    try:
        config = json.loads(raw)
    except Exception:
        config = {"key": config_id, "value": None, "modifiedBy": "system"}
    await redis.delete(_config_key(config_id))
    await redis.srem(_CONFIG_INDEX_KEY, config_id)
    await _append_history(
        redis,
        config_key=config.get("key", config_id),
        old_value=config.get("value"),
        new_value=None,
        user=config.get("modifiedBy") or "system",
        reason="delete",
        sensitive=bool(config.get("sensitive")),
    )
    return {"message": "配置项已删除", "id": config_id}

@router.get("/templates")
async def list_templates():
    redis = await _require_redis()
    templates = await _load_templates(redis)
    templates.sort(key=lambda item: item.get("name", ""))
    return {"templates": templates}

@router.post("/templates", status_code=status.HTTP_201_CREATED)
async def create_template(request: ConfigTemplateCreate):
    redis = await _require_redis()
    template_id = f"tpl_{uuid.uuid4().hex[:8]}"
    now = utc_now().isoformat()
    template = {
        "id": template_id,
        "name": request.name,
        "description": request.description,
        "category": request.category,
        "configs": request.configs,
        "version": request.version,
        "created": now,
        "author": request.author,
    }
    await redis.set(_template_key(template_id), json.dumps(template, ensure_ascii=False))
    await redis.sadd(_TEMPLATE_INDEX_KEY, template_id)
    return template

@router.get("/history")
async def list_history(limit: int = Query(50, ge=1, le=200)):
    redis = await _require_redis()
    raw_list = await redis.lrange(_HISTORY_KEY, 0, limit - 1)
    history: List[Dict[str, Any]] = []
    for raw in raw_list:
        try:
            history.append(json.loads(raw))
        except Exception:
            continue
    return {"history": history}
