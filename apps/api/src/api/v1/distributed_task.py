"""分布式任务协调API端点"""

import asyncio
import os
import socket
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import Field
from src.ai.distributed_task import (
    DistributedTaskCoordinationEngine,
    TaskPriority,
    TaskStatus,
    ConflictType
)
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/distributed-task", tags=["distributed_task"])

# 全局引擎实例（实际应该通过依赖注入）
coordination_engine: Optional[DistributedTaskCoordinationEngine] = None
_engine_lock = asyncio.Lock()

class TaskSubmitRequest(ApiBaseModel):
    """任务提交请求"""
    task_type: str = Field(..., description="任务类型")
    task_data: Dict[str, Any] = Field(..., description="任务数据")
    requirements: Optional[Dict[str, Any]] = Field(None, description="任务需求")
    priority: str = Field("medium", description="任务优先级")
    decomposition_strategy: Optional[str] = Field(None, description="分解策略")
    assignment_strategy: Optional[str] = Field("capability_based", description="分配策略")

class TaskResponse(ApiBaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str

class SystemStatsResponse(ApiBaseModel):
    """系统统计响应"""
    node_id: str
    raft_state: str
    leader_id: Optional[str]
    active_tasks: int
    completed_tasks: int
    queued_tasks: int
    stats: Dict[str, Any]
    state_summary: Dict[str, Any]

async def get_engine() -> DistributedTaskCoordinationEngine:
    """获取协调引擎实例"""
    global coordination_engine
    if coordination_engine:
        return coordination_engine
    async with _engine_lock:
        if coordination_engine:
            return coordination_engine
        redis_client = get_redis()
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis未初始化，无法启动分布式任务协调")
        service_discovery = AgentServiceDiscoverySystem(
            redis_client=redis_client,
            prefix=default_config.redis_prefix,
            ttl_seconds=default_config.agent_ttl_seconds,
        )
        await service_discovery.initialize()
        node_id = os.getenv("DISTRIBUTED_TASK_NODE_ID") or socket.gethostname()
        coordination_engine = DistributedTaskCoordinationEngine(
            node_id=node_id,
            cluster_nodes=[node_id],
            service_registry=service_discovery.registry,
            load_balancer=service_discovery.load_balancer,
        )
        await coordination_engine.start()
        return coordination_engine

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_engine(
    node_id: str = Query(..., description="节点ID"),
    cluster_nodes: List[str] = Body(..., description="集群节点列表")
):
    """初始化协调引擎"""
    
    global coordination_engine
    
    try:
        if coordination_engine:
            await coordination_engine.stop()

        redis_client = get_redis()
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis未初始化，无法启动分布式任务协调")
        service_discovery = AgentServiceDiscoverySystem(
            redis_client=redis_client,
            prefix=default_config.redis_prefix,
            ttl_seconds=default_config.agent_ttl_seconds,
        )
        await service_discovery.initialize()
        
        coordination_engine = DistributedTaskCoordinationEngine(
            node_id=node_id,
            cluster_nodes=cluster_nodes,
            service_registry=service_discovery.registry,
            load_balancer=service_discovery.load_balancer,
        )
        
        await coordination_engine.start()
        
        return {"status": "initialized", "node_id": node_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit", response_model=TaskResponse)
async def submit_task(
    request: TaskSubmitRequest,
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """提交任务"""
    
    try:
        # 解析优先级
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
            "background": TaskPriority.BACKGROUND
        }
        
        priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
        
        # 添加策略到requirements
        requirements = request.requirements or {}
        if request.decomposition_strategy:
            requirements["decomposition_strategy"] = request.decomposition_strategy
            requirements["decompose"] = True
        if request.assignment_strategy:
            requirements["assignment_strategy"] = request.assignment_strategy
        
        # 提交任务
        task_id = await engine.submit_task(
            task_type=request.task_type,
            task_data=request.task_data,
            requirements=requirements,
            priority=priority
        )
        
        if task_id:
            return TaskResponse(
                task_id=task_id,
                status="submitted",
                message="Task submitted successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to submit task")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str,
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """获取任务状态"""
    
    try:
        status = await engine.get_task_status(task_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel/{task_id}", response_model=Dict[str, str])
async def cancel_task(
    task_id: str,
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """取消任务"""
    
    try:
        success = await engine.cancel_task(task_id)
        
        if success:
            return {"status": "cancelled", "task_id": task_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel task")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """获取系统统计"""
    
    try:
        stats = await engine.get_system_stats()
        
        return SystemStatsResponse(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conflicts", response_model=List[Dict[str, Any]])
async def detect_conflicts(
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """检测冲突"""
    
    try:
        conflicts = await engine.conflict_resolver.detect_conflicts()
        
        return [conflict.to_dict() for conflict in conflicts]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conflicts/resolve/{conflict_id}", response_model=Dict[str, str])
async def resolve_conflict(
    conflict_id: str,
    strategy: str = Query("priority_based", description="解决策略"),
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """解决冲突"""
    
    try:
        # 查找冲突
        conflicts = await engine.conflict_resolver.detect_conflicts()
        conflict = next((c for c in conflicts if c.conflict_id == conflict_id), None)
        
        if not conflict:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        # 解决冲突
        success = await engine.conflict_resolver.resolve_conflict(conflict, strategy)
        
        if success:
            return {"status": "resolved", "conflict_id": conflict_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to resolve conflict")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/checkpoint/create", response_model=Dict[str, str])
async def create_checkpoint(
    name: str = Query(..., description="检查点名称"),
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """创建状态检查点"""
    
    try:
        success = await engine.state_manager.create_checkpoint(name)
        
        if success:
            return {"status": "created", "checkpoint_name": name}
        else:
            raise HTTPException(status_code=400, detail="Failed to create checkpoint")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/checkpoint/rollback", response_model=Dict[str, str])
async def rollback_checkpoint(
    name: str = Query(..., description="检查点名称"),
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """回滚到检查点"""
    
    try:
        success = await engine.state_manager.rollback_state(name)
        
        if success:
            return {"status": "rolled_back", "checkpoint_name": name}
        else:
            raise HTTPException(status_code=400, detail="Failed to rollback to checkpoint")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown", response_model=Dict[str, str])
async def shutdown_engine(
    engine: DistributedTaskCoordinationEngine = Depends(get_engine)
):
    """关闭协调引擎"""
    
    global coordination_engine
    
    try:
        await engine.stop()
        coordination_engine = None
        
        return {"status": "shutdown"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
