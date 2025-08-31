"""
安全审计日志系统
"""

import json
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Any, Dict, List, Optional

import structlog
from fastapi import Request, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.redis import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


class AuditLogLevel(str):
    """审计日志级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLog(BaseModel):
    """审计日志模型"""
    id: str
    timestamp: datetime
    level: str
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # success, failure, blocked
    details: Dict[str, Any]
    request_id: Optional[str]
    response_status: Optional[int]
    response_time_ms: Optional[float]


class MCPToolAuditLog(BaseModel):
    """MCP工具审计日志"""
    id: str
    timestamp: datetime
    user_id: str
    api_key: Optional[str]
    tool_name: str
    tool_params: Dict[str, Any]
    execution_result: str  # success, error, blocked
    risk_score: float
    approval_required: bool
    approved_by: Optional[str]
    approved_at: Optional[datetime]
    execution_time_ms: float
    client_ip: str
    user_agent: str
    request_signature: str
    security_alerts: List[str]
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    network_io: Optional[float]


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self.redis = None
        self.buffer: List[AuditLog] = []
        self.buffer_size = 100
        self.flush_interval = 60  # 秒
    
    async def initialize(self):
        """初始化审计日志器"""
        self.redis = await get_redis()
    
    async def log_api_call(
        self,
        request: Request,
        response: Response,
        process_time: float,
        request_id: str
    ):
        """记录API调用"""
        user_id = None
        session_id = None
        
        # 尝试从请求状态获取用户信息
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)
        if hasattr(request.state, "session_id"):
            session_id = request.state.session_id
        
        # 安全地构建查询参数字典
        try:
            query_params = dict(request.query_params) if request.query_params else {}
        except Exception:
            query_params = {}
        
        log_entry = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            level=self._determine_log_level(response.status_code),
            event_type="api_call",
            user_id=user_id,
            session_id=session_id,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent"),
            resource=str(request.url.path),
            action=request.method,
            result="success" if response.status_code < 400 else "failure",
            details={
                "method": request.method,
                "path": str(request.url.path),
                "query_params": query_params,
                "status_code": response.status_code
            },
            request_id=request_id,
            response_status=response.status_code,
            response_time_ms=process_time * 1000
        )
        
        await self._write_log(log_entry)
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        request: Optional[Request] = None,
        details: Optional[Dict[str, Any]] = None,
        level: str = AuditLogLevel.WARNING,
        risk_score: Optional[float] = None,
        **kwargs
    ):
        """记录安全事件"""
        ip_address = "system"
        user_agent = None
        
        if request:
            ip_address = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent")
        
        log_entry = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            level=level,
            event_type=f"security:{event_type}",
            user_id=user_id,
            session_id=kwargs.get("session_id"),
            ip_address=ip_address,
            user_agent=user_agent,
            resource=kwargs.get("resource"),
            action=kwargs.get("action"),
            result=kwargs.get("result", "blocked"),
            details={
                **(details or {}),
                **({'risk_score': risk_score} if risk_score is not None else {})
            },
            request_id=kwargs.get("request_id"),
            response_status=None,
            response_time_ms=None
        )
        
        await self._write_log(log_entry)
        
        # 对于严重安全事件，立即刷新缓冲区
        if level in [AuditLogLevel.ERROR, AuditLogLevel.CRITICAL]:
            await self.flush_buffer()
    
    async def log_tool_call(
        self,
        user_id: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        risk_score: float,
        approval_required: bool,
        client_ip: str = "unknown",
        user_agent: str = "unknown",
        **kwargs
    ):
        """记录MCP工具调用"""
        import time
        start_time = time.time()
        
        # 生成请求签名（用于防重放）
        import hashlib
        signature_data = f"{user_id}:{tool_name}:{json.dumps(tool_params, sort_keys=True)}:{utc_now().isoformat()}"
        request_signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        tool_log = MCPToolAuditLog(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            user_id=user_id,
            api_key=kwargs.get("api_key"),
            tool_name=tool_name,
            tool_params=tool_params,
            execution_result=kwargs.get("execution_result", "pending"),
            risk_score=risk_score,
            approval_required=approval_required,
            approved_by=kwargs.get("approved_by"),
            approved_at=kwargs.get("approved_at"),
            execution_time_ms=(time.time() - start_time) * 1000,
            client_ip=client_ip,
            user_agent=user_agent,
            request_signature=request_signature,
            security_alerts=kwargs.get("security_alerts", []),
            cpu_usage=kwargs.get("cpu_usage"),
            memory_usage=kwargs.get("memory_usage"),
            network_io=kwargs.get("network_io")
        )
        
        # 转换为通用审计日志
        log_entry = AuditLog(
            id=tool_log.id,
            timestamp=tool_log.timestamp,
            level=self._determine_tool_log_level(risk_score),
            event_type="mcp:tool_call",
            user_id=user_id,
            session_id=None,
            ip_address=client_ip,
            user_agent=user_agent,
            resource=f"tool:{tool_name}",
            action="execute",
            result=tool_log.execution_result,
            details=tool_log.dict(),
            request_id=None,
            response_status=None,
            response_time_ms=tool_log.execution_time_ms
        )
        
        await self._write_log(log_entry)
    
    async def log_authentication(
        self,
        event_type: str,  # login, logout, failed_login, token_refresh
        user_id: Optional[str],
        username: Optional[str],
        ip_address: str,
        user_agent: Optional[str],
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录认证事件"""
        log_entry = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            level=AuditLogLevel.INFO if success else AuditLogLevel.WARNING,
            event_type=f"auth:{event_type}",
            user_id=user_id,
            session_id=None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action=event_type,
            result="success" if success else "failure",
            details={
                "username": username,
                **((details or {}) if isinstance(details or {}, dict) else {})
            },
            request_id=None,
            response_status=None,
            response_time_ms=None
        )
        
        await self._write_log(log_entry)
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,  # database, file, api
        resource_id: str,
        action: str,  # read, write, delete
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录数据访问"""
        log_entry = AuditLog(
            id=str(uuid.uuid4()),
            timestamp=utc_now(),
            level=AuditLogLevel.INFO,
            event_type="data:access",
            user_id=user_id,
            session_id=None,
            ip_address="internal",
            user_agent=None,
            resource=f"{resource_type}:{resource_id}",
            action=action,
            result="success" if success else "failure",
            details=details or {},
            request_id=None,
            response_status=None,
            response_time_ms=None
        )
        
        await self._write_log(log_entry)
    
    def _determine_log_level(self, status_code: int) -> str:
        """根据状态码确定日志级别"""
        if status_code < 400:
            return AuditLogLevel.INFO
        elif status_code < 500:
            return AuditLogLevel.WARNING
        else:
            return AuditLogLevel.ERROR
    
    def _determine_tool_log_level(self, risk_score: float) -> str:
        """根据风险分数确定日志级别"""
        if risk_score >= 0.8:
            return AuditLogLevel.CRITICAL
        elif risk_score >= 0.6:
            return AuditLogLevel.ERROR
        elif risk_score >= 0.4:
            return AuditLogLevel.WARNING
        else:
            return AuditLogLevel.INFO
    
    async def _write_log(self, log_entry: AuditLog):
        """写入日志"""
        # 添加到缓冲区
        self.buffer.append(log_entry)
        
        # 同时写入结构化日志
        logger.info(
            "Audit log",
            audit_id=log_entry.id,
            event_type=log_entry.event_type,
            user_id=log_entry.user_id,
            result=log_entry.result,
            level=log_entry.level
        )
        
        # 如果缓冲区满了，刷新到存储
        if len(self.buffer) >= self.buffer_size:
            await self.flush_buffer()
    
    async def flush_buffer(self):
        """刷新缓冲区到持久存储"""
        if not self.buffer:
            return
        
        try:
            # 批量写入Redis（作为临时存储）
            if self.redis:
                pipeline = self.redis.pipeline()
                for log in self.buffer:
                    key = f"audit:log:{log.id}"
                    value = log.json()
                    # 保留7天
                    pipeline.setex(key, 604800, value)
                    
                    # 添加到时间索引
                    score = log.timestamp.timestamp()
                    pipeline.zadd("audit:timeline", {log.id: score})
                
                await pipeline.execute()
            
            # TODO: 批量写入数据库
            # await self._write_to_database(self.buffer)
            
            # 清空缓冲区
            self.buffer.clear()
            
        except Exception as e:
            logger.error("Failed to flush audit buffer", error=str(e))
    
    async def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """查询审计日志"""
        logs = []
        
        if not self.redis:
            return logs
        
        try:
            # 从Redis时间索引获取日志ID
            if start_time and end_time:
                start_score = start_time.timestamp()
                end_score = end_time.timestamp()
                log_ids = await self.redis.zrangebyscore(
                    "audit:timeline",
                    start_score,
                    end_score,
                    start=0,
                    num=limit
                )
            else:
                # 获取最近的日志
                log_ids = await self.redis.zrevrange(
                    "audit:timeline",
                    0,
                    limit - 1
                )
            
            # 批量获取日志内容
            if log_ids:
                pipeline = self.redis.pipeline()
                for log_id in log_ids:
                    pipeline.get(f"audit:log:{log_id}")
                
                results = await pipeline.execute()
                
                for result in results:
                    if result:
                        log = AuditLog.parse_raw(result)
                        
                        # 过滤条件
                        if user_id and log.user_id != user_id:
                            continue
                        if event_type and not log.event_type.startswith(event_type):
                            continue
                        
                        logs.append(log)
            
        except Exception as e:
            logger.error("Failed to query audit logs", error=str(e))
        
        return logs[:limit]
    
    async def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """生成合规报告"""
        logs = await self.query_logs(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(logs),
            "event_breakdown": {},
            "security_events": 0,
            "failed_authentications": 0,
            "data_access_events": 0,
            "high_risk_operations": 0,
            "unique_users": set(),
            "unique_ips": set()
        }
        
        for log in logs:
            # 事件类型统计
            event_category = log.event_type.split(":")[0]
            if event_category not in report["event_breakdown"]:
                report["event_breakdown"][event_category] = 0
            report["event_breakdown"][event_category] += 1
            
            # 安全事件统计
            if log.event_type.startswith("security:"):
                report["security_events"] += 1
            
            # 失败认证统计
            if log.event_type == "auth:failed_login":
                report["failed_authentications"] += 1
            
            # 数据访问统计
            if log.event_type.startswith("data:"):
                report["data_access_events"] += 1
            
            # 高风险操作统计
            if log.level in [AuditLogLevel.ERROR, AuditLogLevel.CRITICAL]:
                report["high_risk_operations"] += 1
            
            # 唯一用户和IP统计
            if log.user_id:
                report["unique_users"].add(log.user_id)
            if log.ip_address:
                report["unique_ips"].add(log.ip_address)
        
        # 转换集合为计数
        report["unique_users"] = len(report["unique_users"])
        report["unique_ips"] = len(report["unique_ips"])
        
        return report


# 全局实例
audit_logger = AuditLogger()