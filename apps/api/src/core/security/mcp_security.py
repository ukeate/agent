"""
MCP工具安全管理系统
"""

import asyncio
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import get_settings
from src.core.redis import get_redis
from src.core.security.audit import AuditLogger

from src.core.logging import get_logger
logger = get_logger(__name__)

settings = get_settings()

class ToolRiskLevel(str, Enum):
    """工具风险级别"""
    LOW = "low"  # 只读操作
    MEDIUM = "medium"  # 有限写操作
    HIGH = "high"  # 系统级操作
    CRITICAL = "critical"  # 危险操作

class ApprovalStatus(str, Enum):
    """审批状态"""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"

class ToolPermission(BaseModel):
    """工具权限模型"""
    tool_name: str
    risk_level: ToolRiskLevel
    allowed_roles: List[str]
    requires_approval: bool
    auto_approve_for_roles: List[str] = []
    max_calls_per_hour: Optional[int] = None
    restricted_params: Dict[str, Any] = {}

class ToolCallRequest(BaseModel):
    """工具调用请求"""
    id: str
    user_id: str
    tool_name: str
    tool_params: Dict[str, Any]
    risk_score: float
    request_time: datetime
    client_ip: str
    user_agent: str

class ToolCallApproval(BaseModel):
    """工具调用审批"""
    request_id: str
    status: ApprovalStatus
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

class SecurityCheck(BaseModel):
    """安全检查结果"""
    allowed: bool
    risk_score: float
    requires_approval: bool
    denial_reason: Optional[str] = None
    warnings: List[str] = []

class MCPToolSecurityManager:
    """MCP工具安全管理器"""
    
    # 默认工具权限配置
    DEFAULT_TOOL_PERMISSIONS = {
        # 文件系统工具
        "read_file": ToolPermission(
            tool_name="read_file",
            risk_level=ToolRiskLevel.LOW,
            allowed_roles=["user", "developer", "admin"],
            requires_approval=False
        ),
        "write_file": ToolPermission(
            tool_name="write_file",
            risk_level=ToolRiskLevel.MEDIUM,
            allowed_roles=["developer", "admin"],
            requires_approval=False,
            auto_approve_for_roles=["admin"]
        ),
        "delete_file": ToolPermission(
            tool_name="delete_file",
            risk_level=ToolRiskLevel.HIGH,
            allowed_roles=["admin"],
            requires_approval=True,
            auto_approve_for_roles=["admin"]
        ),
        
        # 系统工具
        "execute_command": ToolPermission(
            tool_name="execute_command",
            risk_level=ToolRiskLevel.CRITICAL,
            allowed_roles=["admin"],
            requires_approval=True,
            max_calls_per_hour=10
        ),
        "system_info": ToolPermission(
            tool_name="system_info",
            risk_level=ToolRiskLevel.LOW,
            allowed_roles=["user", "developer", "admin"],
            requires_approval=False
        ),
        
        # 数据库工具
        "database_query": ToolPermission(
            tool_name="database_query",
            risk_level=ToolRiskLevel.MEDIUM,
            allowed_roles=["developer", "admin"],
            requires_approval=False,
            restricted_params={"allow_write": False}
        ),
        "database_write": ToolPermission(
            tool_name="database_write",
            risk_level=ToolRiskLevel.HIGH,
            allowed_roles=["admin"],
            requires_approval=True
        ),
        
        # AI工具
        "llm_completion": ToolPermission(
            tool_name="llm_completion",
            risk_level=ToolRiskLevel.LOW,
            allowed_roles=["user", "developer", "admin"],
            requires_approval=False,
            max_calls_per_hour=100
        ),
        "agent_execution": ToolPermission(
            tool_name="agent_execution",
            risk_level=ToolRiskLevel.MEDIUM,
            allowed_roles=["developer", "admin"],
            requires_approval=False,
            max_calls_per_hour=50
        )
    }
    
    def __init__(self):
        self.tool_whitelist: Set[str] = set()
        self.tool_blacklist: Set[str] = set()
        self.tool_permissions: Dict[str, ToolPermission] = self.DEFAULT_TOOL_PERMISSIONS.copy()
        self.audit_logger = AuditLogger()
        self.pending_approvals: Dict[str, ToolCallRequest] = {}
        self.call_history: Dict[str, List[datetime]] = {}  # 用于频率限制
        
        # 初始化白名单
        self._initialize_whitelist()
    
    def _initialize_whitelist(self):
        """初始化工具白名单"""
        # 添加默认允许的工具
        self.tool_whitelist.update([
            "read_file", "write_file", "system_info",
            "database_query", "llm_completion", "agent_execution"
        ])
        
        # 添加危险工具到黑名单
        self.tool_blacklist.update([
            "rm_rf", "format_disk", "shutdown_system"
        ])
    
    async def authorize_tool_call(
        self,
        user_id: str,
        user_roles: List[str],
        tool_name: str,
        tool_params: Dict[str, Any],
        client_ip: str = "unknown",
        user_agent: str = "unknown"
    ) -> SecurityCheck:
        """
        授权工具调用
        
        Args:
            user_id: 用户ID
            user_roles: 用户角色列表
            tool_name: 工具名称
            tool_params: 工具参数
            client_ip: 客户端IP
            user_agent: 用户代理
            
        Returns:
            SecurityCheck: 安全检查结果
        """
        
        # 检查黑名单
        if tool_name in self.tool_blacklist:
            await self.audit_logger.log_security_event(
                event_type="tool_blacklisted",
                user_id=user_id,
                details={
                    "tool_name": tool_name,
                    "reason": "Tool is blacklisted"
                }
            )
            return SecurityCheck(
                allowed=False,
                risk_score=1.0,
                requires_approval=False,
                denial_reason="Tool is blacklisted"
            )
        
        # 检查白名单
        if self.tool_whitelist and tool_name not in self.tool_whitelist:
            await self.audit_logger.log_security_event(
                event_type="tool_not_whitelisted",
                user_id=user_id,
                details={
                    "tool_name": tool_name,
                    "reason": "Tool not in whitelist"
                }
            )
            return SecurityCheck(
                allowed=False,
                risk_score=0.8,
                requires_approval=False,
                denial_reason="Tool not in whitelist"
            )
        
        # 获取工具权限配置
        tool_permission = self.tool_permissions.get(tool_name)
        if not tool_permission:
            # 未知工具，默认高风险
            return SecurityCheck(
                allowed=False,
                risk_score=0.9,
                requires_approval=True,
                denial_reason="Unknown tool"
            )
        
        # 检查角色权限
        has_permission = any(role in tool_permission.allowed_roles for role in user_roles)
        if not has_permission:
            await self.audit_logger.log_security_event(
                event_type="insufficient_permissions",
                user_id=user_id,
                details={
                    "tool_name": tool_name,
                    "user_roles": user_roles,
                    "required_roles": tool_permission.allowed_roles
                }
            )
            return SecurityCheck(
                allowed=False,
                risk_score=0.7,
                requires_approval=False,
                denial_reason=f"Insufficient permissions. Required roles: {tool_permission.allowed_roles}"
            )
        
        # 检查频率限制
        if tool_permission.max_calls_per_hour:
            if not await self._check_rate_limit(user_id, tool_name, tool_permission.max_calls_per_hour):
                await self.audit_logger.log_security_event(
                    event_type="rate_limit_exceeded",
                    user_id=user_id,
                    details={
                        "tool_name": tool_name,
                        "limit": tool_permission.max_calls_per_hour
                    }
                )
                return SecurityCheck(
                    allowed=False,
                    risk_score=0.6,
                    requires_approval=False,
                    denial_reason=f"Rate limit exceeded. Max {tool_permission.max_calls_per_hour} calls per hour"
                )
        
        # 分析参数安全性
        param_check = await self._analyze_tool_params(tool_name, tool_params, tool_permission)
        if not param_check["safe"]:
            return SecurityCheck(
                allowed=False,
                risk_score=param_check["risk_score"],
                requires_approval=True,
                denial_reason=param_check["reason"],
                warnings=param_check.get("warnings", [])
            )
        
        # 计算风险分数
        risk_score = self._calculate_risk_score(tool_permission.risk_level, tool_params)
        
        # 检查是否需要审批
        requires_approval = tool_permission.requires_approval
        if requires_approval and any(role in tool_permission.auto_approve_for_roles for role in user_roles):
            requires_approval = False  # 自动审批
        
        # 记录审计日志
        await self.audit_logger.log_tool_call(
            user_id=user_id,
            tool_name=tool_name,
            tool_params=tool_params,
            risk_score=risk_score,
            approval_required=requires_approval,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        return SecurityCheck(
            allowed=True if not requires_approval else False,
            risk_score=risk_score,
            requires_approval=requires_approval,
            warnings=param_check.get("warnings", [])
        )
    
    async def _check_rate_limit(self, user_id: str, tool_name: str, limit: int) -> bool:
        """检查频率限制"""
        key = f"{user_id}:{tool_name}"
        now = utc_now()
        
        # 获取历史调用记录
        if key not in self.call_history:
            self.call_history[key] = []
        
        # 清理一小时前的记录
        hour_ago = utc_now().replace(microsecond=0) - timedelta(hours=1)
        self.call_history[key] = [
            call_time for call_time in self.call_history[key]
            if call_time > hour_ago
        ]
        
        # 检查是否超限
        if len(self.call_history[key]) >= limit:
            return False
        
        # 记录本次调用
        self.call_history[key].append(now)
        return True
    
    async def _analyze_tool_params(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_permission: ToolPermission
    ) -> Dict[str, Any]:
        """分析工具参数安全性"""
        result = {
            "safe": True,
            "risk_score": 0.0,
            "reason": None,
            "warnings": []
        }
        
        # 检查受限参数
        if tool_permission.restricted_params:
            for param, allowed_value in tool_permission.restricted_params.items():
                if param in tool_params and tool_params[param] != allowed_value:
                    result["safe"] = False
                    result["risk_score"] = 0.8
                    result["reason"] = f"Restricted parameter {param} has invalid value"
                    return result
        
        # 特定工具的参数检查
        if tool_name == "execute_command":
            # 检查危险命令
            dangerous_commands = ["rm -rf", "format", "dd if=", ":(){ :|:& };:"]
            command = tool_params.get("command", "")
            for dangerous in dangerous_commands:
                if dangerous in command:
                    result["safe"] = False
                    result["risk_score"] = 1.0
                    result["reason"] = f"Dangerous command detected: {dangerous}"
                    return result
        
        elif tool_name == "database_write":
            # 检查SQL注入风险
            query = tool_params.get("query", "")
            if any(keyword in query.lower() for keyword in ["drop", "truncate", "delete from"]):
                result["warnings"].append("Destructive SQL operation detected")
                result["risk_score"] = 0.7
        
        elif tool_name in ["read_file", "write_file"]:
            # 检查路径遍历
            file_path = tool_params.get("path", "")
            if "../" in file_path or file_path.startswith("/etc/") or file_path.startswith("/sys/"):
                result["safe"] = False
                result["risk_score"] = 0.9
                result["reason"] = "Potential path traversal or system file access"
                return result
        
        return result
    
    def _calculate_risk_score(self, risk_level: ToolRiskLevel, params: Dict[str, Any]) -> float:
        """计算风险分数"""
        base_scores = {
            ToolRiskLevel.LOW: 0.2,
            ToolRiskLevel.MEDIUM: 0.5,
            ToolRiskLevel.HIGH: 0.7,
            ToolRiskLevel.CRITICAL: 0.9
        }
        
        base_score = base_scores.get(risk_level, 0.5)
        
        # 根据参数调整分数
        # 例如：参数越多，风险越高
        param_factor = min(len(params) * 0.02, 0.1)
        
        return min(base_score + param_factor, 1.0)
    
    async def request_approval(
        self,
        request: ToolCallRequest
    ) -> str:
        """请求工具调用审批"""
        request_id = request.id
        self.pending_approvals[request_id] = request
        
        # 发送审批通知（实际实现中应该通过消息队列或通知系统）
        await self._send_approval_notification(request)
        
        # 记录审批请求
        await self.audit_logger.log_security_event(
            event_type="approval_requested",
            user_id=request.user_id,
            details={
                "request_id": request_id,
                "tool_name": request.tool_name,
                "risk_score": request.risk_score
            }
        )
        
        return request_id
    
    async def _send_approval_notification(self, request: ToolCallRequest):
        """发送审批通知"""
        payload = {
            "type": "mcp_tool_approval_requested",
            "request_id": request.id,
            "user_id": request.user_id,
            "tool_name": request.tool_name,
            "risk_score": request.risk_score,
            "request_time": request.request_time.isoformat(),
        }
        redis = get_redis()
        if redis:
            await redis.lpush("mcp:approvals:notifications", json.dumps(payload, ensure_ascii=False))
            await redis.ltrim("mcp:approvals:notifications", 0, 999)
        logger.info(
            "Approval notification sent",
            request_id=request.id,
            tool_name=request.tool_name,
            user_id=request.user_id
        )
    
    async def process_approval(
        self,
        request_id: str,
        approved: bool,
        approver_id: str,
        reason: Optional[str] = None
    ) -> ToolCallApproval:
        """处理审批"""
        if request_id not in self.pending_approvals:
            raise ValueError(f"Approval request {request_id} not found")
        
        request = self.pending_approvals[request_id]
        
        approval = ToolCallApproval(
            request_id=request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            approved_by=approver_id,
            approved_at=utc_now(),
            rejection_reason=reason if not approved else None
        )
        
        # 记录审批结果
        await self.audit_logger.log_security_event(
            event_type="approval_processed",
            user_id=approver_id,
            details={
                "request_id": request_id,
                "approved": approved,
                "tool_name": request.tool_name,
                "original_user": request.user_id,
                "reason": reason
            }
        )
        
        # 清理待审批记录
        del self.pending_approvals[request_id]
        
        return approval
    
    def update_tool_permission(self, tool_name: str, permission: ToolPermission):
        """更新工具权限配置"""
        self.tool_permissions[tool_name] = permission
        logger.info("Tool permission updated", tool_name=tool_name)
    
    def add_to_whitelist(self, tool_name: str):
        """添加到白名单"""
        self.tool_whitelist.add(tool_name)
        if tool_name in self.tool_blacklist:
            self.tool_blacklist.remove(tool_name)
        logger.info("Tool added to whitelist", tool_name=tool_name)
    
    def add_to_blacklist(self, tool_name: str):
        """添加到黑名单"""
        self.tool_blacklist.add(tool_name)
        if tool_name in self.tool_whitelist:
            self.tool_whitelist.remove(tool_name)
        logger.info("Tool added to blacklist", tool_name=tool_name)
    
    def get_tool_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取工具使用统计"""
        stats = {
            "total_tools": len(self.tool_permissions),
            "whitelisted_tools": len(self.tool_whitelist),
            "blacklisted_tools": len(self.tool_blacklist),
            "pending_approvals": len(self.pending_approvals)
        }
        
        if user_id:
            # 获取用户特定的统计
            user_calls = sum(
                len(calls) for key, calls in self.call_history.items()
                if key.startswith(f"{user_id}:")
            )
            stats["user_calls_last_hour"] = user_calls
        
        return stats

# 全局实例
mcp_security_manager = MCPToolSecurityManager()
