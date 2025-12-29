"""
智能体集群管理API

提供集群管理的RESTful API接口，包括智能体生命周期管理、
监控数据查询、扩缩容控制等功能。
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pydantic import Field
from src.ai.cluster import (
    ClusterStateManager, LifecycleManager, MetricsCollector, AutoScaler,
    AgentInfo, AgentStatus, AgentGroup, ScalingPolicy, AgentCapability, ResourceSpec
)
from src.api.base_model import ApiBaseModel
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

def _rethrow_http_exception(error: Exception) -> None:
    if isinstance(error, HTTPException):
        raise error

# Pydantic models for API
class AgentInfoCreate(ApiBaseModel):
    """创建智能体的请求模型"""
    name: str = Field(..., description="智能体名称")
    host: str = Field(..., description="智能体主机地址")
    port: int = Field(..., description="智能体端口")
    capabilities: List[str] = Field(default=[], description="智能体能力")
    version: str = Field(default="1.0.0", description="智能体版本")
    config: Dict[str, Any] = Field(default={}, description="智能体配置")
    labels: Dict[str, str] = Field(default={}, description="智能体标签")
    resource_spec: Dict[str, Any] = Field(default={}, description="资源规格")

class AgentGroupCreate(ApiBaseModel):
    """创建智能体分组的请求模型"""
    name: str = Field(..., description="分组名称")
    description: str = Field(default="", description="分组描述")
    max_agents: Optional[int] = Field(None, description="最大智能体数量")
    min_agents: int = Field(default=0, description="最小智能体数量")
    labels: Dict[str, str] = Field(default={}, description="分组标签")

class AgentGroupUpdate(ApiBaseModel):
    """更新智能体分组的请求模型"""
    name: Optional[str] = Field(None, description="分组名称")
    description: Optional[str] = Field(None, description="分组描述")
    max_agents: Optional[int] = Field(None, description="最大智能体数量")
    min_agents: Optional[int] = Field(None, description="最小智能体数量")
    labels: Optional[Dict[str, str]] = Field(None, description="分组标签")

class ScalingPolicyCreate(ApiBaseModel):
    """创建扩缩容策略的请求模型"""
    name: str = Field(..., description="策略名称")
    target_cpu_percent: float = Field(default=70.0, description="目标CPU使用率")
    target_memory_percent: float = Field(default=75.0, description="目标内存使用率")
    scale_up_cpu_threshold: float = Field(default=80.0, description="CPU扩容阈值")
    scale_up_memory_threshold: float = Field(default=85.0, description="内存扩容阈值")
    scale_down_cpu_threshold: float = Field(default=30.0, description="CPU缩容阈值")
    scale_down_memory_threshold: float = Field(default=35.0, description="内存缩容阈值")
    min_instances: int = Field(default=1, description="最小实例数")
    max_instances: int = Field(default=10, description="最大实例数")
    cooldown_period_seconds: int = Field(default=180, description="冷却时间")
    enabled: bool = Field(default=True, description="是否启用")

class ScalingPolicyUpdate(ApiBaseModel):
    """更新扩缩容策略的请求模型"""
    name: Optional[str] = Field(None, description="策略名称")
    target_cpu_percent: Optional[float] = Field(None, description="目标CPU使用率")
    target_memory_percent: Optional[float] = Field(None, description="目标内存使用率")
    scale_up_cpu_threshold: Optional[float] = Field(None, description="CPU扩容阈值")
    scale_up_memory_threshold: Optional[float] = Field(None, description="内存扩容阈值")
    scale_down_cpu_threshold: Optional[float] = Field(None, description="CPU缩容阈值")
    scale_down_memory_threshold: Optional[float] = Field(None, description="内存缩容阈值")
    min_instances: Optional[int] = Field(None, description="最小实例数")
    max_instances: Optional[int] = Field(None, description="最大实例数")
    cooldown_period_seconds: Optional[int] = Field(None, description="冷却时间")
    enabled: Optional[bool] = Field(None, description="是否启用")

class ManualScalingRequest(ApiBaseModel):
    """手动扩缩容请求模型"""
    target_instances: int = Field(..., description="目标实例数")
    reason: str = Field(default="Manual scaling", description="扩缩容原因")

class MetricsQueryRequest(ApiBaseModel):
    """指标查询请求模型"""
    metric_names: Optional[List[str]] = Field(None, description="指标名称列表")
    duration_seconds: int = Field(default=3600, description="查询时长（秒）")
    agent_id: Optional[str] = Field(None, description="智能体ID")

class LoadBalancingStrategyCreate(ApiBaseModel):
    """负载均衡策略创建模型"""
    name: str = Field(..., description="策略名称")
    algorithm: str = Field(..., description="算法类型")
    weights: Optional[Dict[str, float]] = Field(default=None, description="权重配置")
    health_check_settings: Dict[str, Any] = Field(default_factory=dict, description="健康检查设置")
    failover_settings: Dict[str, Any] = Field(default_factory=dict, description="故障转移设置")
    is_active: bool = Field(default=True, description="是否启用")

class CapacityForecastRequest(ApiBaseModel):
    """容量预测请求"""
    forecast_horizon_days: int = Field(default=30, ge=1, le=365)
    scenarios: List[str] = Field(default_factory=lambda: ["moderate"])
    include_recommendations: bool = Field(default=True)

class AnomalyDetectionRequest(ApiBaseModel):
    """异常检测请求"""
    detection_window_hours: int = Field(default=24, ge=1, le=168)
    sensitivity: str = Field(default="medium")
    include_predictions: bool = Field(default=True)

class WorkflowCreate(ApiBaseModel):
    """自动化工作流创建模型"""
    name: str = Field(..., description="工作流名称")
    workflow_type: str = Field(default="maintenance", description="工作流类型")
    trigger_type: str = Field(default="manual", description="触发类型")
    is_enabled: bool = Field(default=True, description="是否启用")

class SecurityAuditRequest(ApiBaseModel):
    """安全审计请求"""
    audit_scope: str = Field(default="cluster")
    target_id: Optional[str] = None
    audit_frameworks: List[str] = Field(default_factory=list)
    security_domains: List[str] = Field(default_factory=list)
    audit_depth: str = Field(default="standard")

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.agent_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if agent_id:
            if agent_id not in self.agent_connections:
                self.agent_connections[agent_id] = []
            self.agent_connections[agent_id].append(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if agent_id and agent_id in self.agent_connections:
            if websocket in self.agent_connections[agent_id]:
                self.agent_connections[agent_id].remove(websocket)
            if not self.agent_connections[agent_id]:
                del self.agent_connections[agent_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.active_connections.remove(conn)

    async def broadcast_to_agent(self, message: str, agent_id: str):
        if agent_id in self.agent_connections:
            disconnected = []
            for connection in self.agent_connections[agent_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.append(connection)
            
            # 清理断开的连接
            for conn in disconnected:
                self.agent_connections[agent_id].remove(conn)

# 全局连接管理器
manager = ConnectionManager()

# 增强集群管理数据存储（内存态，避免静态假数据）
_load_balancing_strategies: Dict[str, Dict[str, Any]] = {}
_security_audits: List[Dict[str, Any]] = []
_automation_workflows: Dict[str, Dict[str, Any]] = {}

# 依赖注入 - 这些应该从应用配置中获取
async def get_cluster_manager(request: Request):
    # 使用Request对象获取当前应用实例的状态
    return request.app.state.cluster_manager

async def get_lifecycle_manager(request: Request):
    return request.app.state.lifecycle_manager

async def get_metrics_collector(request: Request):
    return request.app.state.metrics_collector

async def get_auto_scaler(request: Request):
    return request.app.state.auto_scaler

router = APIRouter(prefix="/cluster", tags=["cluster"])

def _group_to_dict(group: AgentGroup) -> Dict[str, Any]:
    return {
        "group_id": group.group_id,
        "name": group.name,
        "description": group.description,
        "agent_ids": list(group.agent_ids),
        "agent_count": group.agent_count,
        "min_agents": group.min_agents,
        "max_agents": group.max_agents,
        "is_full": group.is_full,
        "can_scale_down": group.can_scale_down,
        "labels": group.labels,
        "created_at": group.created_at,
        "updated_at": group.updated_at
    }

def _policy_to_dict(policy: ScalingPolicy) -> Dict[str, Any]:
    created_at = getattr(policy, "created_at", time.time())
    updated_at = getattr(policy, "updated_at", created_at)
    return {
        "policy_id": policy.policy_id,
        "name": policy.name,
        "target_cpu_percent": policy.target_cpu_percent,
        "target_memory_percent": policy.target_memory_percent,
        "scale_up_cpu_threshold": policy.scale_up_cpu_threshold,
        "scale_up_memory_threshold": policy.scale_up_memory_threshold,
        "scale_down_cpu_threshold": policy.scale_down_cpu_threshold,
        "scale_down_memory_threshold": policy.scale_down_memory_threshold,
        "min_instances": policy.min_instances,
        "max_instances": policy.max_instances,
        "cooldown_period_seconds": policy.cooldown_period_seconds,
        "enabled": policy.enabled,
        "created_at": created_at,
        "updated_at": updated_at
    }

def _iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat()

def _build_performance_trends(cluster_metrics: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[int, Dict[str, Any]] = {}
    mapping = {
        "cluster_cpu_usage": "cpu_trend",
        "cluster_memory_usage": "memory_trend",
        "cluster_avg_response_time": "response_time_trend",
    }
    for metric_name, points in cluster_metrics.items():
        key_name = mapping.get(metric_name)
        if not key_name:
            continue
        for point in points:
            ts = int(getattr(point, "timestamp", 0))
            if ts <= 0:
                continue
            entry = bucket.setdefault(ts, {"timestamp": _iso_from_ts(ts)})
            entry[key_name] = float(getattr(point, "value", 0) or 0)

    if not bucket:
        return []

    ordered = [bucket[k] for k in sorted(bucket.keys())]
    return ordered[-20:]

def _build_performance_profile(agent: AgentInfo) -> Dict[str, Any]:
    usage = agent.resource_usage
    cpu = float(usage.cpu_usage_percent or 0)
    memory = float(usage.memory_usage_percent or 0)
    response_time = float(usage.avg_response_time or 0)
    error_rate = float(usage.error_rate or 0) * 100

    bottlenecks: List[str] = []
    if cpu >= 80:
        bottlenecks.append("CPU压力过高")
    if memory >= 80:
        bottlenecks.append("内存压力过高")
    if response_time >= 1000:
        bottlenecks.append("响应时间过长")
    if error_rate >= 5:
        bottlenecks.append("错误率偏高")

    score_penalty = (cpu * 0.4) + (memory * 0.4) + min(response_time / 20, 20) + min(error_rate * 2, 20)
    overall_score = max(0, min(100, round(100 - score_penalty)))

    recommendations = []
    if "CPU压力过高" in bottlenecks:
        recommendations.append("优化CPU密集型任务或增加实例")
    if "内存压力过高" in bottlenecks:
        recommendations.append("优化内存使用或扩展内存资源")
    if "响应时间过长" in bottlenecks:
        recommendations.append("优化请求路径或启用缓存策略")
    if "错误率偏高" in bottlenecks:
        recommendations.append("检查错误日志并修复异常")

    expected_improvement = min(30, len(recommendations) * 10)

    return {
        "agent_id": agent.agent_id,
        "overall_performance_score": overall_score,
        "bottlenecks": bottlenecks,
        "optimization_recommendations": recommendations,
        "expected_improvement": expected_improvement,
    }

def _build_scaling_suggestions(stats: Dict[str, Any]) -> List[str]:
    usage = stats.get("resource_usage") or {}
    cpu = float(usage.get("cpu_usage_percent") or 0)
    memory = float(usage.get("memory_usage_percent") or 0)
    suggestions: List[str] = []
    if cpu >= 75 or memory >= 75:
        suggestions.append("资源利用率偏高，建议增加实例或扩容资源")
    if cpu <= 30 and memory <= 30 and stats.get("total_agents", 0) > 1:
        suggestions.append("资源利用率偏低，可考虑缩容以节省成本")
    if not suggestions:
        suggestions.append("当前资源利用率稳定，保持现有规模")
    return suggestions

# 集群状态API
@router.get("/status")
async def get_cluster_status(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取集群状态概览"""
    try:
        stats = await cluster_manager.get_cluster_stats()
        return JSONResponse(content={"success": True, "data": stats})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting cluster status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topology")
async def get_cluster_topology(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取集群拓扑"""
    try:
        topology = await cluster_manager.get_cluster_topology()
        return JSONResponse(content={"success": True, "data": topology.to_dict()})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting cluster topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_cluster_health(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取集群健康状态"""
    try:
        topology = await cluster_manager.get_cluster_topology()
        
        health_info = {
            "cluster_id": topology.cluster_id,
            "health_score": topology.cluster_health_score,
            "total_agents": topology.total_agents,
            "healthy_agents": topology.healthy_agents,
            "running_agents": topology.running_agents,
            "failed_agents": len(topology.get_agents_by_status(AgentStatus.FAILED)),
            "resource_usage": topology.cluster_resource_usage.to_dict() if hasattr(topology.cluster_resource_usage, 'to_dict') else {}
        }
        
        return JSONResponse(content={"success": True, "data": health_info})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting cluster health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 智能体管理API
@router.post("/agents")
async def create_agent(
    agent_data: AgentInfoCreate,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """创建并注册新的智能体"""
    try:
        # 创建智能体信息
        capabilities = {AgentCapability(cap) for cap in agent_data.capabilities if cap in [c.value for c in AgentCapability]}
        
        resource_spec = ResourceSpec(**agent_data.resource_spec) if agent_data.resource_spec else ResourceSpec()
        
        agent = AgentInfo(
            name=agent_data.name,
            host=agent_data.host,
            port=agent_data.port,
            capabilities=capabilities,
            version=agent_data.version,
            config=agent_data.config,
            labels=agent_data.labels,
            resource_spec=resource_spec
        )
        
        # 注册智能体
        result = await lifecycle_manager.register_agent(agent, auto_start=True)
        
        if result.success:
            return JSONResponse(content={
                "success": True,
                "data": {"agent_id": agent.agent_id, "result": jsonable_encoder(result)}
            })
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_agents(
    status: Optional[str] = None,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取智能体列表"""
    try:
        if status:
            try:
                agent_status = AgentStatus(status)
                agents = await cluster_manager.get_agents_by_status(agent_status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        else:
            topology = await cluster_manager.get_cluster_topology()
            agents = list(topology.agents.values())
        
        agents_data = []
        for agent in agents:
            agent_dict = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "endpoint": agent.endpoint,
                "status": agent.status.value,
                "capabilities": [cap.value for cap in agent.capabilities],
                "is_healthy": agent.is_healthy,
                "uptime": agent.uptime_seconds,
                "resource_usage": {
                    "cpu_usage": agent.resource_usage.cpu_usage_percent,
                    "memory_usage": agent.resource_usage.memory_usage_percent,
                    "active_tasks": agent.resource_usage.active_tasks,
                    "error_rate": agent.resource_usage.error_rate
                },
                "labels": agent.labels,
                "created_at": agent.created_at,
                "updated_at": agent.updated_at
            }
            agents_data.append(agent_dict)
        
        return JSONResponse(content={"success": True, "data": agents_data})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent_details(
    agent_id: str,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取智能体详细信息"""
    try:
        agent = await cluster_manager.get_agent_info(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_data = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "host": agent.host,
            "port": agent.port,
            "endpoint": agent.endpoint,
            "status": agent.status.value,
            "capabilities": [cap.value for cap in agent.capabilities],
            "version": agent.version,
            "is_healthy": agent.is_healthy,
            "uptime": agent.uptime_seconds,
            "resource_spec": {
                "cpu_cores": agent.resource_spec.cpu_cores,
                "memory_gb": agent.resource_spec.memory_gb,
                "storage_gb": agent.resource_spec.storage_gb,
                "gpu_count": agent.resource_spec.gpu_count
            },
            "resource_usage": {
                "cpu_usage_percent": agent.resource_usage.cpu_usage_percent,
                "memory_usage_percent": agent.resource_usage.memory_usage_percent,
                "storage_usage_percent": agent.resource_usage.storage_usage_percent,
                "gpu_usage_percent": agent.resource_usage.gpu_usage_percent,
                "network_io_mbps": agent.resource_usage.network_io_mbps,
                "active_tasks": agent.resource_usage.active_tasks,
                "total_requests": agent.resource_usage.total_requests,
                "failed_requests": agent.resource_usage.failed_requests,
                "error_rate": agent.resource_usage.error_rate,
                "avg_response_time": agent.resource_usage.avg_response_time
            },
            "health": {
                "is_healthy": agent.health.is_healthy,
                "is_responsive": agent.health.is_responsive,
                "needs_restart": agent.health.needs_restart,
                "consecutive_failures": agent.health.consecutive_failures,
                "last_heartbeat": agent.health.last_heartbeat
            },
            "config": agent.config,
            "labels": agent.labels,
            "metadata": agent.metadata,
            "group_id": agent.group_id,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
            "started_at": agent.started_at
        }
        
        return JSONResponse(content={"success": True, "data": agent_data})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting agent details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/start")
async def start_agent(
    agent_id: str,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """启动智能体"""
    try:
        result = await lifecycle_manager.start_agent(agent_id)
        
        if result.success:
            # 通过WebSocket广播状态变更
            await manager.broadcast(json.dumps({
                "type": "agent_status_change",
                "agent_id": agent_id,
                "status": "running",
                "timestamp": time.time()
            }))
            
            return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error starting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/stop")
async def stop_agent(
    agent_id: str,
    graceful: bool = True,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """停止智能体"""
    try:
        result = await lifecycle_manager.stop_agent(agent_id, graceful=graceful)
        
        if result.success:
            await manager.broadcast(json.dumps({
                "type": "agent_status_change",
                "agent_id": agent_id,
                "status": "stopped",
                "timestamp": time.time()
            }))
            
            return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error stopping agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/restart")
async def restart_agent(
    agent_id: str,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """重启智能体"""
    try:
        result = await lifecycle_manager.restart_agent(agent_id)
        
        if result.success:
            await manager.broadcast(json.dumps({
                "type": "agent_status_change",
                "agent_id": agent_id,
                "status": "running",
                "timestamp": time.time()
            }))
            
            return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error restarting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str,
    force: bool = False,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """删除智能体"""
    try:
        result = await lifecycle_manager.unregister_agent(agent_id, force=force)
        
        if result.success:
            await manager.broadcast(json.dumps({
                "type": "agent_removed",
                "agent_id": agent_id,
                "timestamp": time.time()
            }))
            
            return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 分组管理API
@router.post("/groups")
async def create_group(
    group_data: AgentGroupCreate,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """创建智能体分组"""
    try:
        group = AgentGroup(
            name=group_data.name,
            description=group_data.description,
            max_agents=group_data.max_agents,
            min_agents=group_data.min_agents,
            labels=group_data.labels
        )
        
        success = await cluster_manager.create_group(group)
        
        if success:
            return JSONResponse(content={
                "success": True,
                "data": _group_to_dict(group)
            })
        else:
            raise HTTPException(status_code=400, detail="Failed to create group")
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error creating group: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/groups")
async def list_groups(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取分组列表"""
    try:
        topology = await cluster_manager.get_cluster_topology()
        
        groups_data = []
        for group in topology.groups.values():
            groups_data.append(_group_to_dict(group))
        
        return JSONResponse(content={"success": True, "data": groups_data})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error listing groups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/groups/{group_id}")
async def get_group_details(
    group_id: str,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """获取分组详情"""
    try:
        topology = await cluster_manager.get_cluster_topology()
        group = topology.groups.get(group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        return JSONResponse(content={"success": True, "data": _group_to_dict(group)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/groups/{group_id}")
async def update_group(
    group_id: str,
    group_data: AgentGroupUpdate,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """更新分组信息"""
    try:
        updated = await cluster_manager.update_group(group_id, group_data.model_dump(exclude_unset=True))
        if not updated:
            raise HTTPException(status_code=404, detail="Group not found")
        return JSONResponse(content={"success": True, "data": _group_to_dict(updated)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/groups/{group_id}")
async def delete_group(
    group_id: str,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """删除分组"""
    try:
        success = await cluster_manager.delete_group(group_id)
        if not success:
            raise HTTPException(status_code=404, detail="Group not found")
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/groups/{group_id}/agents/{agent_id}")
async def add_agent_to_group(
    group_id: str,
    agent_id: str,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """将智能体添加到分组"""
    try:
        success = await cluster_manager.add_agent_to_group(group_id, agent_id)
        
        if success:
            return JSONResponse(content={"success": True})
        else:
            raise HTTPException(status_code=400, detail="Failed to add agent to group")
            
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error adding agent to group: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/groups/{group_id}/agents/{agent_id}")
async def remove_agent_from_group(
    group_id: str,
    agent_id: str,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager)
):
    """将智能体从分组移除"""
    try:
        success = await cluster_manager.remove_agent_from_group(group_id, agent_id)
        if success:
            return JSONResponse(content={"success": True})
        raise HTTPException(status_code=400, detail="Failed to remove agent from group")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing agent from group: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 监控指标API
@router.post("/metrics/query")
async def query_metrics(
    query: MetricsQueryRequest,
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """查询指标数据"""
    try:
        if query.agent_id:
            metrics = await metrics_collector.get_agent_metrics(
                query.agent_id,
                query.metric_names,
                query.duration_seconds
            )
        else:
            metrics = await metrics_collector.get_cluster_metrics(
                query.metric_names,
                query.duration_seconds
            )
        
        # 转换指标点为JSON友好格式
        formatted_metrics = {}
        for metric_name, points in metrics.items():
            if isinstance(points, (list, tuple)):
                # 如果是点列表，转换每个点
                formatted_metrics[metric_name] = [
                    {
                        "value": point.value if hasattr(point, 'value') else point,
                        "timestamp": point.timestamp if hasattr(point, 'timestamp') else time.time(),
                        "labels": point.labels if hasattr(point, 'labels') else {}
                    }
                    for point in points
                ]
            else:
                # 如果是单个值，创建一个点
                formatted_metrics[metric_name] = [{
                    "value": points if isinstance(points, (int, float)) else 0,
                    "timestamp": time.time(),
                    "labels": {}
                }]
        
        return JSONResponse(content={"success": True, "data": formatted_metrics})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error querying metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/summary")
async def get_metrics_summary(
    agent_id: Optional[str] = None,
    duration_seconds: int = 300,
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """获取指标摘要"""
    try:
        summary = await metrics_collector.get_metrics_summary(agent_id, duration_seconds)
        return JSONResponse(content={"success": True, "data": summary})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/trends/{metric_name}")
async def analyze_metric_trend(
    metric_name: str,
    agent_id: Optional[str] = None,
    duration_seconds: int = 3600,
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """分析指标趋势"""
    try:
        trend = await metrics_collector.analyze_trend(
            metric_name, agent_id, duration_seconds
        )
        return JSONResponse(content={"success": True, "data": trend})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error analyzing trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 自动扩缩容API
@router.get("/scaling/policies")
async def list_scaling_policies(
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """获取扩缩容策略列表"""
    try:
        policies = auto_scaler.get_policies()
        return JSONResponse(content={"success": True, "data": [_policy_to_dict(p) for p in policies]})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error listing scaling policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scaling/policies/{policy_id}")
async def get_scaling_policy(
    policy_id: str,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """获取扩缩容策略详情"""
    try:
        policy = auto_scaler.policies.get(policy_id)
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        return JSONResponse(content={"success": True, "data": _policy_to_dict(policy)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scaling policy {policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scaling/policies")
async def create_scaling_policy(
    policy_data: ScalingPolicyCreate,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """创建扩缩容策略"""
    try:
        policy_id = f"policy-{uuid.uuid4().hex[:8]}"
        policy = ScalingPolicy(
            policy_id=policy_id,
            name=policy_data.name,
            target_cpu_percent=policy_data.target_cpu_percent,
            target_memory_percent=policy_data.target_memory_percent,
            scale_up_cpu_threshold=policy_data.scale_up_cpu_threshold,
            scale_up_memory_threshold=policy_data.scale_up_memory_threshold,
            scale_down_cpu_threshold=policy_data.scale_down_cpu_threshold,
            scale_down_memory_threshold=policy_data.scale_down_memory_threshold,
            min_instances=policy_data.min_instances,
            max_instances=policy_data.max_instances,
            cooldown_period_seconds=policy_data.cooldown_period_seconds,
            enabled=policy_data.enabled
        )
        policy.created_at = time.time()
        policy.updated_at = policy.created_at
        
        auto_scaler.add_policy(policy)
        
        return JSONResponse(content={
            "success": True, 
            "data": _policy_to_dict(policy)
        })
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error creating scaling policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/scaling/policies/{policy_id}")
async def update_scaling_policy(
    policy_id: str,
    policy_data: ScalingPolicyUpdate,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """更新扩缩容策略"""
    try:
        policy = auto_scaler.policies.get(policy_id)
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")

        updates = policy_data.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(policy, key, value)
        policy.updated_at = time.time()
        return JSONResponse(content={"success": True, "data": _policy_to_dict(policy)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating scaling policy {policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/scaling/policies/{policy_id}")
async def delete_scaling_policy(
    policy_id: str,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """删除扩缩容策略"""
    try:
        if policy_id not in auto_scaler.policies:
            raise HTTPException(status_code=404, detail="Policy not found")
        auto_scaler.remove_policy(policy_id)
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting scaling policy {policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scaling/recommendations")
async def get_scaling_recommendations(
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """获取扩缩容建议"""
    try:
        recommendations = await auto_scaler.get_scaling_recommendations()
        
        # 转换为JSON友好格式
        formatted_recommendations = {}
        if hasattr(recommendations, 'items'):
            # 如果是字典类型
            for group_id, decision in recommendations.items():
                if hasattr(decision, 'to_dict'):
                    formatted_recommendations[group_id] = decision.to_dict()
                else:
                    formatted_recommendations[group_id] = decision
        elif isinstance(recommendations, list):
            # 如果是列表类型
            for i, decision in enumerate(recommendations):
                if hasattr(decision, 'to_dict'):
                    formatted_recommendations[f"group_{i}"] = decision.to_dict()
                else:
                    formatted_recommendations[f"group_{i}"] = decision
        else:
            # 其他类型，直接返回
            formatted_recommendations = {"default": recommendations if isinstance(recommendations, (dict, list, str, int, float)) else str(recommendations)}
        
        return JSONResponse(content={"success": True, "data": formatted_recommendations})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting scaling recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scaling/groups/{group_id}/manual")
async def manual_scale_group(
    group_id: str,
    scaling_request: ManualScalingRequest,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """手动扩缩容分组"""
    try:
        result = await auto_scaler.manual_scale(
            group_id,
            scaling_request.target_instances,
            scaling_request.reason
        )
        
        # 广播扩缩容事件
        await manager.broadcast(json.dumps({
            "type": "scaling_event",
            "group_id": group_id,
            "action": result.decision.action.value,
            "target_instances": scaling_request.target_instances,
            "timestamp": time.time()
        }))
        
        return JSONResponse(content={
            "success": True, 
            "data": {
                "event_id": result.event_id,
                "success": result.success,
                "duration": result.duration_seconds
            }
        })
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error manual scaling group {group_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scaling/history")
async def get_scaling_history(
    group_id: Optional[str] = None,
    limit: int = 100,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """获取扩缩容历史"""
    try:
        history = auto_scaler.get_scaling_history(group_id, limit)
        
        history_data = []
        for event in history:
            event_data = {
                "event_id": event.event_id,
                "group_id": event.group_id,
                "decision": event.decision.to_dict(),
                "success": event.success,
                "duration_seconds": event.duration_seconds,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "error_message": event.error_message
            }
            history_data.append(event_data)
        
        return JSONResponse(content={"success": True, "data": history_data})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error getting scaling history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket实时推送
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket实时数据推送"""
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接并监听客户端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                if message.get("type") == "subscribe":
                    # 处理订阅请求
                    subscription = message.get("subscription", {})
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "subscription": subscription
                    }))
                elif message.get("type") == "ping":
                    # 心跳响应
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@router.websocket("/ws/agent/{agent_id}")
async def agent_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """智能体专用WebSocket连接"""
    await manager.connect(websocket, agent_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理智能体特定的消息
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, agent_id)

# 批量操作API
@router.post("/agents/batch/start")
async def batch_start_agents(
    agent_ids: List[str],
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """批量启动智能体"""
    try:
        from src.ai.cluster.lifecycle_manager import AgentOperation
        
        result = await lifecycle_manager.batch_operation(
            agent_ids, AgentOperation.START
        )
        
        # 广播批量操作结果
        await manager.broadcast(json.dumps({
            "type": "batch_operation_completed",
            "operation": "start",
            "agent_ids": agent_ids,
            "success_count": result.success_count,
            "failed_count": result.failed_count,
            "timestamp": time.time()
        }))
        
        return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error in batch start operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/batch/stop")
async def batch_stop_agents(
    agent_ids: List[str],
    graceful: bool = True,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """批量停止智能体"""
    try:
        from src.ai.cluster.lifecycle_manager import AgentOperation
        
        result = await lifecycle_manager.batch_operation(
            agent_ids, AgentOperation.STOP, {"graceful": graceful}
        )
        
        await manager.broadcast(json.dumps({
            "type": "batch_operation_completed",
            "operation": "stop",
            "agent_ids": agent_ids,
            "success_count": result.success_count,
            "failed_count": result.failed_count,
            "timestamp": time.time()
        }))
        
        return JSONResponse(content={"success": True, "data": jsonable_encoder(result)})
    except Exception as e:
        _rethrow_http_exception(e)
        logger.error(f"Error in batch stop operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 增强集群管理接口 ==========
@router.get("/load-balancing/strategies")
async def list_load_balancing_strategies():
    return {"strategies": list(_load_balancing_strategies.values())}

@router.post("/load-balancing/strategies")
async def create_load_balancing_strategy(
    request: LoadBalancingStrategyCreate,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    stats = await cluster_manager.get_cluster_stats()
    health_score = float(stats.get("health_score") or 0)
    strategy_id = f"lbs_{uuid.uuid4().hex[:8]}"
    now = utc_now().isoformat()
    improvement = max(0, round(100 - health_score))
    strategy = {
        "strategy_id": strategy_id,
        "name": request.name,
        "algorithm": request.algorithm,
        "weights": request.weights,
        "health_check_settings": request.health_check_settings,
        "failover_settings": request.failover_settings,
        "estimated_performance_improvement": improvement,
        "is_active": request.is_active,
        "created_at": now,
        "updated_at": now,
    }
    _load_balancing_strategies[strategy_id] = strategy
    return {
        "strategy_id": strategy_id,
        "created_at": now,
        "estimated_performance_improvement": improvement,
    }

@router.get("/health/deep-analysis")
async def get_deep_health_analysis(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    stats = await cluster_manager.get_cluster_stats()
    usage = stats.get("resource_usage") or {}
    health_score = float(stats.get("health_score") or 0)
    unhealthy = max(0, stats.get("total_agents", 0) - stats.get("healthy_agents", 0))
    summary = "系统运行正常" if health_score >= 80 else "系统存在风险，需要关注"

    upcoming: List[Dict[str, Any]] = []
    cpu = float(usage.get("cpu_usage_percent") or 0)
    memory = float(usage.get("memory_usage_percent") or 0)
    if cpu >= 80:
        upcoming.append({
            "description": "CPU负载持续偏高，建议扩容或优化负载",
            "estimated_time": utc_now().date().isoformat(),
            "urgency": "high" if cpu >= 90 else "medium",
        })
    if memory >= 80:
        upcoming.append({
            "description": "内存压力偏高，建议检查内存占用",
            "estimated_time": utc_now().date().isoformat(),
            "urgency": "high" if memory >= 90 else "medium",
        })
    if unhealthy > 0:
        upcoming.append({
            "description": f"{unhealthy} 个智能体健康异常，建议排查",
            "estimated_time": utc_now().date().isoformat(),
            "urgency": "medium",
        })

    cluster_metrics = await metrics_collector.get_cluster_metrics(
        metric_names=["cluster_cpu_usage", "cluster_memory_usage", "cluster_avg_response_time"],
        duration_seconds=3600,
    )
    performance_trends = _build_performance_trends(cluster_metrics)

    return {
        "overall_health_score": round(health_score),
        "health_summary": summary,
        "predictive_maintenance": {"upcoming_maintenance": upcoming},
        "performance_trends": performance_trends,
    }

@router.get("/performance/profiles")
async def get_performance_profiles(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    topology = await cluster_manager.get_cluster_topology()
    profiles = [_build_performance_profile(agent) for agent in topology.agents.values()]
    return {"profiles": profiles}

@router.post("/capacity/forecast")
async def generate_capacity_forecast(
    request: CapacityForecastRequest,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    stats = await cluster_manager.get_cluster_stats()
    usage = stats.get("resource_usage") or {}
    total_agents = int(stats.get("total_agents") or 0)
    cpu = float(usage.get("cpu_usage_percent") or 0)
    memory = float(usage.get("memory_usage_percent") or 0)
    pressure = max(cpu, memory) / 100 if max(cpu, memory) > 0 else 0

    horizon = request.forecast_horizon_days
    base = total_agents
    forecast_data: List[Dict[str, Any]] = []
    for i in range(1, horizon + 1):
        factor = i / horizon if horizon > 0 else 0
        conservative = round(base * (1 + pressure * factor * 0.05))
        moderate = round(base * (1 + pressure * factor * 0.1))
        aggressive = round(base * (1 + pressure * factor * 0.2))
        forecast_data.append({
            "date": (utc_now().date() + timedelta(days=i)).isoformat(),
            "conservative": conservative,
            "moderate": moderate,
            "aggressive": aggressive,
        })

    recommendations = {
        "resource_scaling_suggestions": _build_scaling_suggestions(stats) if request.include_recommendations else []
    }
    result: Dict[str, Any] = {
        "forecast_data": forecast_data,
        "recommendations": recommendations,
    }
    return result

@router.get("/security/audits")
async def list_security_audits():
    return {"audits": _security_audits}

@router.post("/security/audit")
async def create_security_audit(
    request: SecurityAuditRequest,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    stats = await cluster_manager.get_cluster_stats()
    health_score = float(stats.get("health_score") or 0)
    usage = stats.get("resource_usage") or {}
    cpu = float(usage.get("cpu_usage_percent") or 0)
    memory = float(usage.get("memory_usage_percent") or 0)
    error_rate = float(usage.get("error_rate") or 0) * 100
    findings: List[Dict[str, Any]] = []
    if cpu >= 85:
        findings.append({
            "category": "resource",
            "severity": "high",
            "description": "CPU负载过高",
            "affected_components": ["cluster"],
            "remediation_steps": ["扩容或优化负载"],
            "compliance_impact": request.audit_frameworks or [],
        })
    if memory >= 85:
        findings.append({
            "category": "resource",
            "severity": "high",
            "description": "内存占用过高",
            "affected_components": ["cluster"],
            "remediation_steps": ["检查内存泄漏或扩容"],
            "compliance_impact": request.audit_frameworks or [],
        })
    if error_rate >= 5:
        findings.append({
            "category": "reliability",
            "severity": "medium",
            "description": "错误率偏高",
            "affected_components": ["cluster"],
            "remediation_steps": ["排查失败请求来源"],
            "compliance_impact": request.audit_frameworks or [],
        })

    risk_score = max(0, round(100 - health_score))
    compliance_status = "compliant" if risk_score < 30 and not findings else "partial"
    audit_id = f"aud_{uuid.uuid4().hex[:8]}"
    audit = {
        "audit_id": audit_id,
        "audit_types": list({*request.audit_frameworks, *request.security_domains}),
        "overall_risk_score": risk_score,
        "findings": findings,
        "compliance_status": compliance_status,
        "created_at": utc_now().isoformat(),
    }
    _security_audits.insert(0, audit)
    _security_audits[:] = _security_audits[:200]
    return audit

@router.post("/anomaly-detection/detect")
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    topology = await cluster_manager.get_cluster_topology()
    sensitivity = request.sensitivity.lower()
    threshold = 80
    if sensitivity == "high":
        threshold = 70
    elif sensitivity == "low":
        threshold = 90

    anomalies: List[Dict[str, Any]] = []
    now = utc_now().isoformat()
    for agent in topology.agents.values():
        usage = agent.resource_usage
        cpu = float(usage.cpu_usage_percent or 0)
        memory = float(usage.memory_usage_percent or 0)
        error_rate = float(usage.error_rate or 0)
        response_time = float(usage.avg_response_time or 0)

        if cpu >= threshold:
            anomalies.append({
                "anomaly_type": "cpu_usage",
                "severity": "high" if cpu >= 90 else "medium",
                "confidence": min(1.0, cpu / 100),
                "affected_agents": [agent.agent_id],
                "description": f"CPU使用率达到 {cpu:.1f}%",
                "detected_at": now,
            })
        if memory >= threshold:
            anomalies.append({
                "anomaly_type": "memory_usage",
                "severity": "high" if memory >= 90 else "medium",
                "confidence": min(1.0, memory / 100),
                "affected_agents": [agent.agent_id],
                "description": f"内存使用率达到 {memory:.1f}%",
                "detected_at": now,
            })
        if error_rate >= 0.05:
            anomalies.append({
                "anomaly_type": "error_rate",
                "severity": "medium",
                "confidence": min(1.0, error_rate * 10),
                "affected_agents": [agent.agent_id],
                "description": f"错误率达到 {error_rate * 100:.1f}%",
                "detected_at": now,
            })
        if response_time >= 1000:
            anomalies.append({
                "anomaly_type": "response_time",
                "severity": "medium",
                "confidence": min(1.0, response_time / 2000),
                "affected_agents": [agent.agent_id],
                "description": f"响应时间达到 {response_time:.1f}ms",
                "detected_at": now,
            })

    return {"anomalies": anomalies}

@router.get("/automation/workflows")
async def list_automation_workflows():
    return {"workflows": list(_automation_workflows.values())}

@router.post("/automation/workflows")
async def create_automation_workflow(request: WorkflowCreate):
    workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
    workflow = {
        "workflow_id": workflow_id,
        "name": request.name,
        "workflow_type": request.workflow_type,
        "trigger_type": request.trigger_type,
        "is_enabled": request.is_enabled,
        "execution_count": 0,
        "success_rate": 0.0,
        "created_at": utc_now().isoformat(),
    }
    _automation_workflows[workflow_id] = workflow
    return workflow

@router.get("/reports")
async def get_comprehensive_reports(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
):
    stats = await cluster_manager.get_cluster_stats()
    health_score = float(stats.get("health_score") or 0)
    total_agents = int(stats.get("total_agents") or 0)
    healthy_agents = int(stats.get("healthy_agents") or 0)
    reports: List[Dict[str, Any]] = []

    reports.append({
        "report_id": f"rpt_health_{uuid.uuid4().hex[:6]}",
        "name": "集群健康报告",
        "description": f"健康评分 {health_score:.0f}，健康/总数 {healthy_agents}/{total_agents}",
        "report_type": "health",
    })

    if _security_audits:
        latest = _security_audits[0]
        reports.append({
            "report_id": f"rpt_security_{uuid.uuid4().hex[:6]}",
            "name": "安全审计报告",
            "description": f"最新风险评分 {latest.get('overall_risk_score', 0)}",
            "report_type": "security",
        })

    return {"reports": reports}

# 操作历史API
@router.get("/operations/history")
async def get_operation_history(
    agent_id: Optional[str] = None,
    operation_type: Optional[str] = None,
    limit: int = 100,
    lifecycle_manager: LifecycleManager = Depends(get_lifecycle_manager)
):
    """获取操作历史"""
    try:
        from src.ai.cluster.lifecycle_manager import AgentOperation
        
        operation_enum = None
        if operation_type:
            try:
                operation_enum = AgentOperation(operation_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation type: {operation_type}")
        
        history = await lifecycle_manager.get_operation_history(
            agent_id, operation_enum, limit
        )
        
        history_data = [jsonable_encoder(result) for result in history]
        
        return JSONResponse(content={"success": True, "data": history_data})
    except Exception as e:
        logger.error(f"Error getting operation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 后台任务：定期推送实时数据
async def broadcast_real_time_data(
    cluster_manager: ClusterStateManager,
    metrics_collector: MetricsCollector
):
    """定期广播实时数据"""
    while True:
        try:
            # 获取集群状态
            stats = await cluster_manager.get_cluster_stats()
            
            # 获取指标摘要
            metrics_summary = await metrics_collector.get_metrics_summary()
            
            # 广播实时数据
            await manager.broadcast(json.dumps({
                "type": "realtime_update",
                "data": {
                    "cluster_stats": stats,
                    "metrics_summary": metrics_summary,
                    "timestamp": time.time()
                }
            }))
            
            await asyncio.sleep(10)  # 每10秒更新一次
            
        except Exception as e:
            logger.error(f"Error in real-time data broadcast: {e}")
            await asyncio.sleep(30)  # 出错时等待更长时间
