"""
安全管理API端点
"""

from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from src.core.security.auth import User, get_current_active_user, require_permission
from src.core.security.audit import audit_logger
from src.core.security.mcp_security import (
    ToolCallRequest,
    ToolPermission,
    mcp_security_manager,
)
from src.core.security.monitoring import security_monitor

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/security", tags=["Security"])


class APIKeyCreate(BaseModel):
    """API密钥创建请求"""
    name: str
    description: Optional[str] = None
    expires_in_days: Optional[int] = 30
    permissions: List[str] = []


class APIKeyResponse(BaseModel):
    """API密钥响应"""
    id: str
    name: str
    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    permissions: List[str]


@router.get("/config")
async def get_security_config(
    current_user: User = Depends(require_permission("system:read"))
):
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
    # TODO: 实现配置更新逻辑
    
    logger.info(
        "Security configuration updated",
        user_id=current_user.id,
        updates=config_updates
    )
    
    return {
        "message": "Security configuration updated",
        "updated_fields": list(config_updates.keys())
    }


@router.get("/api-keys")
async def list_api_keys(
    current_user: User = Depends(require_permission("system:read"))
):
    """
    获取API密钥列表
    
    需要 system:read 权限
    """
    # TODO: 从数据库获取API密钥列表
    
    return {
        "api_keys": [],
        "total": 0
    }


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(require_permission("system:write"))
):
    """
    创建新API密钥
    
    需要 system:write 权限
    """
    import secrets
    import uuid
    
    # 生成API密钥
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    key_id = str(uuid.uuid4())
    
    # 计算过期时间
    expires_at = None
    if api_key_data.expires_in_days:
        expires_at = utc_now() + timedelta(days=api_key_data.expires_in_days)
    
    # TODO: 保存到数据库
    
    logger.info(
        "API key created",
        user_id=current_user.id,
        key_id=key_id,
        key_name=api_key_data.name
    )
    
    return APIKeyResponse(
        id=key_id,
        name=api_key_data.name,
        key=api_key,
        created_at=utc_now(),
        expires_at=expires_at,
        permissions=api_key_data.permissions
    )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(require_permission("system:write"))
):
    """
    撤销API密钥
    
    需要 system:write 权限
    """
    # TODO: 从数据库删除API密钥
    
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
        "logs": [log.dict() for log in logs],
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
        return permission.dict()
    
    return {
        "permissions": {
            name: perm.dict()
            for name, perm in mcp_security_manager.tool_permissions.items()
        }
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
        permission=permission.dict()
    )
    
    return {
        "message": "Tool permission updated successfully",
        "tool_name": tool_name,
        "permission": permission.dict()
    }


# 安全监控和告警
@router.get("/alerts")
async def get_security_alerts(
    status: Optional[str] = None,
    current_user: User = Depends(require_permission("system:read"))
):
    """
    获取安全告警列表
    
    需要 system:read 权限
    """
    alerts = await security_monitor.get_active_alerts()
    
    if status:
        alerts = [a for a in alerts if a.status == status]
    
    return {
        "alerts": [alert.dict() for alert in alerts],
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
async def get_security_metrics(
    current_user: User = Depends(require_permission("system:read"))
):
    """
    获取安全指标和统计
    
    需要 system:read 权限
    """
    metrics = await security_monitor.get_security_metrics()
    tool_stats = mcp_security_manager.get_tool_statistics(current_user.id)
    
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
        request.dict()
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
            "approval": approval.dict()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )