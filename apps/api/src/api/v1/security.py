"""
安全管理API端点
"""

import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import List, Optional
from pydantic import Field
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.security.auth import User, get_current_active_user, require_permission
from src.core.security.audit import audit_logger
from src.core.security.mcp_security import (
    ToolCallRequest,
    ToolPermission,
    mcp_security_manager,
)
from src.api.base_model import ApiBaseModel
from src.core.security.monitoring import security_monitor
from src.core.database import get_db
from src.core.redis import get_redis
from src.models.database.api_key import APIKey as DBAPIKey

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/security", tags=["Security"])

class APIKeyCreate(ApiBaseModel):
    """API密钥创建请求"""
    name: str
    description: Optional[str] = None
    expires_in_days: Optional[int] = 30
    permissions: List[str] = Field(default_factory=list)

class APIKeyResponse(ApiBaseModel):
    """API密钥响应"""
    id: str
    name: str
    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    permissions: List[str]
    description: Optional[str] = None
    status: str

@router.get("/config")
async def get_security_config():
    """
    获取当前安全配置
    
    需要 system:read 权限
    """
    from src.core.config import get_settings
    settings = get_settings()
    
    return {
        "force_https": settings.FORCE_HTTPS,
        "csp_header": settings.CSP_HEADER,
        "security_threshold": settings.SECURITY_THRESHOLD,
        "auto_block_threshold": settings.AUTO_BLOCK_THRESHOLD,
        "max_requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
        "max_request_size": settings.MAX_REQUEST_SIZE,
        "default_rate_limit": settings.DEFAULT_RATE_LIMIT,
        "jwt_algorithm": settings.JWT_ALGORITHM,
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "refresh_token_expire_days": settings.REFRESH_TOKEN_EXPIRE_DAYS
    }

@router.put("/config")
async def update_security_config(
    config_updates: dict,
    current_user: User = Depends(require_permission("system:admin"))
):
    """
    更新安全配置
    
    需要 system:admin 权限
    """
    mapping = {
        "force_https": "FORCE_HTTPS",
        "csp_header": "CSP_HEADER",
        "security_threshold": "SECURITY_THRESHOLD",
        "auto_block_threshold": "AUTO_BLOCK_THRESHOLD",
        "max_requests_per_minute": "MAX_REQUESTS_PER_MINUTE",
        "max_request_size": "MAX_REQUEST_SIZE",
        "default_rate_limit": "DEFAULT_RATE_LIMIT",
        "jwt_algorithm": "JWT_ALGORITHM",
        "access_token_expire_minutes": "ACCESS_TOKEN_EXPIRE_MINUTES",
        "refresh_token_expire_days": "REFRESH_TOKEN_EXPIRE_DAYS",
    }
    allowed = set(mapping.keys())
    unknown = [k for k in config_updates.keys() if k not in allowed]
    if unknown:
        raise HTTPException(status_code=400, detail=f"不支持的配置项: {unknown}")

    parsed: dict[str, object] = {}
    for k, v in config_updates.items():
        if k in ["force_https"]:
            parsed[k] = bool(v)
        elif k in ["security_threshold", "auto_block_threshold"]:
            parsed[k] = float(v)
        elif k in ["max_requests_per_minute", "max_request_size", "access_token_expire_minutes", "refresh_token_expire_days"]:
            parsed[k] = int(v)
        else:
            parsed[k] = str(v)

    redis = get_redis()
    if redis:
        await redis.hset(
            "security:config_overrides",
            mapping={k: json.dumps(v, ensure_ascii=False) for k, v in parsed.items()},
        )

    for k, env_key in mapping.items():
        if k in parsed:
            os.environ[env_key] = str(parsed[k])

    from src.core.config import get_settings
    get_settings.cache_clear()
    
    logger.info(
        "Security configuration updated",
        user_id=current_user.id,
        updates=parsed
    )
    
    return {
        "message": "Security configuration updated",
        "updated_fields": list(parsed.keys())
    }

@router.get("/api-keys")
async def list_api_keys(
    current_user: User = Depends(require_permission("system:read")),
    db: AsyncSession = Depends(get_db),
):
    """
    获取API密钥列表
    
    需要 system:read 权限
    """
    now = utc_now()
    keys = (await db.execute(select(DBAPIKey).order_by(DBAPIKey.created_at.desc()))).scalars().all()
    return {
        "api_keys": [
            APIKeyResponse(
                id=str(k.id),
                name=k.name,
                key=f"{k.key_prefix}...",
                created_at=k.created_at,
                expires_at=k.expires_at,
                permissions=list(k.permissions or []),
                description=k.description,
                status="revoked" if k.is_revoked else "expired" if k.expires_at and k.expires_at <= now else "active",
            ).model_dump()
            for k in keys
        ],
        "total": len(keys),
    }

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(require_permission("system:write")),
    db: AsyncSession = Depends(get_db),
):
    """
    创建新API密钥
    
    需要 system:write 权限
    """
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    key_prefix = api_key[:8]
    
    # 计算过期时间
    expires_at = None
    if api_key_data.expires_in_days:
        expires_at = utc_now() + timedelta(days=api_key_data.expires_in_days)

    db_key = DBAPIKey(
        name=api_key_data.name,
        description=api_key_data.description,
        key_hash=key_hash,
        key_prefix=key_prefix,
        permissions=api_key_data.permissions,
        is_revoked=False,
        expires_at=expires_at,
    )
    db.add(db_key)
    await db.commit()
    await db.refresh(db_key)

    record = APIKeyResponse(
        id=str(db_key.id),
        name=db_key.name,
        key=api_key,
        created_at=db_key.created_at,
        expires_at=db_key.expires_at,
        permissions=list(db_key.permissions or []),
        description=db_key.description,
        status="active",
    )
    
    logger.info(
        "API key created",
        user_id=current_user.id,
        key_id=str(db_key.id),
        key_name=api_key_data.name
    )
    
    return record

@router.get("/api-keys/permissions")
async def list_api_key_permissions(
    current_user: User = Depends(require_permission("system:read")),
):
    """
    获取可用权限列表
    
    需要 system:read 权限
    """
    from src.core.security.auth import rbac_manager

    permissions = set()
    for perms in rbac_manager.ROLE_PERMISSIONS.values():
        permissions.update(perms)
    return {"permissions": sorted(permissions)}

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(require_permission("system:write")),
    db: AsyncSession = Depends(get_db),
):
    """
    撤销API密钥
    
    需要 system:write 权限
    """
    try:
        key_uuid = uuid.UUID(key_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的API key id")

    db_key = (await db.execute(select(DBAPIKey).where(DBAPIKey.id == key_uuid))).scalar_one_or_none()
    if not db_key:
        raise HTTPException(status_code=404, detail="API key not found")

    db_key.is_revoked = True
    db_key.revoked_at = utc_now()
    await db.commit()
    
    logger.info(
        "API key revoked",
        user_id=current_user.id,
        key_id=key_id
    )
    
    return {"message": "API key revoked successfully"}

# MCP工具安全审计
@router.get("/mcp-tools/audit")
async def get_mcp_tool_audit_logs(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    tool_name: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    current_user: User = Depends(require_permission("system:read"))
):
    """
    获取MCP工具调用审计日志
    
    需要 system:read 权限
    """
    logs = await audit_logger.query_logs(
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        event_type="mcp:tool_call",
        limit=limit
    )
    
    # 过滤工具名称
    if tool_name:
        logs = [log for log in logs if log.resource == f"tool:{tool_name}"]
    
    return {
        "logs": [log.model_dump() for log in logs],
        "total": len(logs)
    }

@router.post("/mcp-tools/whitelist")
async def update_tool_whitelist(
    tool_names: List[str],
    action: str = "add",  # add or remove
    current_user: User = Depends(require_permission("system:admin"))
):
    """
    更新工具白名单
    
    需要 system:admin 权限
    """
    if action == "add":
        for tool_name in tool_names:
            mcp_security_manager.add_to_whitelist(tool_name)
        message = f"Added {len(tool_names)} tools to whitelist"
    elif action == "remove":
        for tool_name in tool_names:
            mcp_security_manager.tool_whitelist.discard(tool_name)
        message = f"Removed {len(tool_names)} tools from whitelist"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid action. Use 'add' or 'remove'"
        )
    
    logger.info(
        "Tool whitelist updated",
        user_id=current_user.id,
        action=action,
        tools=tool_names
    )
    
    return {
        "message": message,
        "current_whitelist": list(mcp_security_manager.tool_whitelist)
    }

@router.get("/mcp-tools/permissions")
async def get_tool_permissions(
    tool_name: Optional[str] = None,
    current_user: User = Depends(require_permission("tools:read"))
):
    """
    获取工具权限配置
    
    需要 tools:read 权限
    """
    if tool_name:
        permission = mcp_security_manager.tool_permissions.get(tool_name)
        if not permission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {tool_name} not found"
            )
        return permission.model_dump()
    
    return {
        "permissions": {
            name: perm.model_dump()
            for name, perm in mcp_security_manager.tool_permissions.items()
        }
    }

@router.get("/mcp-tools/whitelist")
async def get_tool_whitelist(
    current_user: User = Depends(require_permission("tools:read"))
):
    """
    获取工具白名单列表

    需要 tools:read 权限
    """
    return {
        "whitelist": list(mcp_security_manager.tool_whitelist)
    }

@router.put("/mcp-tools/permissions")
async def update_tool_permissions(
    tool_name: str,
    permission: ToolPermission,
    current_user: User = Depends(require_permission("system:admin"))
):
    """
    更新工具权限
    
    需要 system:admin 权限
    """
    mcp_security_manager.update_tool_permission(tool_name, permission)
    
    logger.info(
        "Tool permission updated",
        user_id=current_user.id,
        tool_name=tool_name,
        permission=permission.model_dump()
    )
    
    return {
        "message": "Tool permission updated successfully",
        "tool_name": tool_name,
        "permission": permission.model_dump()
    }

# 安全监控和告警
@router.get("/alerts")
async def get_security_alerts(
    status: Optional[str] = None,
):
    """
    获取安全告警列表
    
    需要 system:read 权限
    """
    alerts = await security_monitor.get_active_alerts()
    
    if status:
        alerts = [a for a in alerts if a.status == status]
    
    return {
        "alerts": [alert.model_dump() for alert in alerts],
        "total": len(alerts)
    }

@router.post("/alerts/{alert_id}/resolve")
async def resolve_security_alert(
    alert_id: str,
    resolution: str = "resolved",
    current_user: User = Depends(require_permission("system:write"))
):
    """
    标记告警为已解决
    
    需要 system:write 权限
    """
    await security_monitor.resolve_alert(alert_id, resolution)
    
    logger.info(
        "Security alert resolved",
        user_id=current_user.id,
        alert_id=alert_id,
        resolution=resolution
    )
    
    return {
        "message": "Alert resolved successfully",
        "alert_id": alert_id,
        "resolution": resolution
    }

@router.get("/metrics")
async def get_security_metrics():
    """
    获取安全指标和统计
    
    需要 system:read 权限
    """
    metrics = await security_monitor.get_security_metrics()
    tool_stats = mcp_security_manager.get_tool_statistics("public")
    
    return {
        "security_metrics": metrics,
        "tool_statistics": tool_stats
    }

@router.get("/risk-assessment")
async def get_risk_assessment(
    current_user: User = Depends(require_permission("system:read"))
):
    """
    获取实时风险评估
    
    需要 system:read 权限
    """
    assessment = await security_monitor.perform_risk_assessment()
    
    return assessment

# 合规报告
@router.get("/compliance-report")
async def generate_compliance_report(
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(require_permission("system:admin"))
):
    """
    生成合规报告
    
    需要 system:admin 权限
    """
    report = await audit_logger.generate_compliance_report(
        start_time=start_date,
        end_time=end_date
    )
    
    logger.info(
        "Compliance report generated",
        user_id=current_user.id,
        period_start=start_date.isoformat(),
        period_end=end_date.isoformat()
    )
    
    return report

# 工具调用审批
@router.get("/mcp-tools/pending-approvals")
async def get_pending_approvals(
    current_user: User = Depends(require_permission("tools:write"))
):
    """
    获取待审批的工具调用请求
    
    需要 tools:write 权限
    """
    pending = [
        request.model_dump()
        for request in mcp_security_manager.pending_approvals.values()
    ]
    
    return {
        "pending_approvals": pending,
        "total": len(pending)
    }

@router.post("/mcp-tools/approve/{request_id}")
async def approve_tool_call(
    request_id: str,
    approved: bool,
    reason: Optional[str] = None,
    current_user: User = Depends(require_permission("tools:write"))
):
    """
    审批工具调用请求
    
    需要 tools:write 权限
    """
    try:
        approval = await mcp_security_manager.process_approval(
            request_id=request_id,
            approved=approved,
            approver_id=current_user.id,
            reason=reason
        )
        
        return {
            "message": "Approval processed successfully",
            "approval": approval.model_dump()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
