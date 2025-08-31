"""
智能体集群管理API

提供集群管理的RESTful API接口，包括智能体生命周期管理、
监控数据查询、扩缩容控制等功能。
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ...ai.cluster import (
    ClusterStateManager, LifecycleManager, MetricsCollector, AutoScaler,
    AgentInfo, AgentStatus, AgentGroup, ScalingPolicy, AgentCapability, ResourceSpec
)

logger = logging.getLogger(__name__)

# Pydantic models for API
class AgentInfoCreate(BaseModel):
    """创建智能体的请求模型"""
    name: str = Field(..., description="智能体名称")
    host: str = Field(..., description="智能体主机地址")
    port: int = Field(..., description="智能体端口")
    capabilities: List[str] = Field(default=[], description="智能体能力")
    version: str = Field(default="1.0.0", description="智能体版本")
    config: Dict[str, Any] = Field(default={}, description="智能体配置")
    labels: Dict[str, str] = Field(default={}, description="智能体标签")
    resource_spec: Dict[str, Any] = Field(default={}, description="资源规格")


class AgentGroupCreate(BaseModel):
    """创建智能体分组的请求模型"""
    name: str = Field(..., description="分组名称")
    description: str = Field(default="", description="分组描述")
    max_agents: Optional[int] = Field(None, description="最大智能体数量")
    min_agents: int = Field(default=0, description="最小智能体数量")
    labels: Dict[str, str] = Field(default={}, description="分组标签")


class ScalingPolicyCreate(BaseModel):
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


class ManualScalingRequest(BaseModel):
    """手动扩缩容请求模型"""
    target_instances: int = Field(..., description="目标实例数")
    reason: str = Field(default="Manual scaling", description="扩缩容原因")


class MetricsQueryRequest(BaseModel):
    """指标查询请求模型"""
    metric_names: Optional[List[str]] = Field(None, description="指标名称列表")
    duration_seconds: int = Field(default=3600, description="查询时长（秒）")
    agent_id: Optional[str] = Field(None, description="智能体ID")


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
                "data": {"agent_id": agent.agent_id, "result": result.__dict__}
            })
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
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
            
            return JSONResponse(content={"success": True, "data": result.__dict__})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
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
            
            return JSONResponse(content={"success": True, "data": result.__dict__})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
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
            
            return JSONResponse(content={"success": True, "data": result.__dict__})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
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
            
            return JSONResponse(content={"success": True, "data": result.__dict__})
        else:
            raise HTTPException(status_code=400, detail=result.message)
            
    except Exception as e:
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
                "data": {"group_id": group.group_id}
            })
        else:
            raise HTTPException(status_code=400, detail="Failed to create group")
            
    except Exception as e:
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
            group_data = {
                "group_id": group.group_id,
                "name": group.name,
                "description": group.description,
                "agent_count": group.agent_count,
                "min_agents": group.min_agents,
                "max_agents": group.max_agents,
                "is_full": group.is_full,
                "can_scale_down": group.can_scale_down,
                "labels": group.labels,
                "created_at": group.created_at,
                "updated_at": group.updated_at
            }
            groups_data.append(group_data)
        
        return JSONResponse(content={"success": True, "data": groups_data})
    except Exception as e:
        logger.error(f"Error listing groups: {e}")
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
        logger.error(f"Error adding agent to group: {e}")
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
        logger.error(f"Error analyzing trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 自动扩缩容API
@router.post("/scaling/policies")
async def create_scaling_policy(
    policy_data: ScalingPolicyCreate,
    auto_scaler: AutoScaler = Depends(get_auto_scaler)
):
    """创建扩缩容策略"""
    try:
        policy = ScalingPolicy(
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
        
        auto_scaler.add_policy(policy)
        
        return JSONResponse(content={
            "success": True, 
            "data": {"policy_id": policy.policy_id}
        })
    except Exception as e:
        logger.error(f"Error creating scaling policy: {e}")
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
        from ...ai.cluster.lifecycle_manager import AgentOperation
        
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
        
        return JSONResponse(content={"success": True, "data": result.__dict__})
    except Exception as e:
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
        from ...ai.cluster.lifecycle_manager import AgentOperation
        
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
        
        return JSONResponse(content={"success": True, "data": result.__dict__})
    except Exception as e:
        logger.error(f"Error in batch stop operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        from ...ai.cluster.lifecycle_manager import AgentOperation
        
        operation_enum = None
        if operation_type:
            try:
                operation_enum = AgentOperation(operation_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation type: {operation_type}")
        
        history = await lifecycle_manager.get_operation_history(
            agent_id, operation_enum, limit
        )
        
        history_data = [result.__dict__ for result in history]
        
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