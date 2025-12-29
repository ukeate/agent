"""记忆管理API端点"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now
from fastapi import APIRouter, Depends, HTTPException, Header, Query
import json
import io
import networkx as nx
from src.ai.memory.models import (
    MemoryCreateRequest,
    MemoryUpdateRequest,
    MemoryResponse,
    MemoryFilters,
    MemoryAnalytics,
    MemoryType,
    MemoryStatus,
    ImportResult
)
from src.services.memory_service import memory_service
from src.core.dependencies import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/memories", tags=["Memory Management"])

@router.post("/", response_model=MemoryResponse)
async def create_memory(
    request: MemoryCreateRequest,
    session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Depends(get_current_user)
) -> MemoryResponse:
    """创建新记忆"""
    try:
        # 确保MemoryType为枚举类型
        memory_type = request.type
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
            
        memory = await memory_service.create_memory(
            content=request.content,
            memory_type=memory_type,
            session_id=session_id,
            user_id=user_id,
            metadata=request.metadata,
            importance=request.importance,
            tags=request.tags,
            source=request.source
        )
        return MemoryResponse.from_memory(memory)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"请求参数无效: {str(e)}")
    except Exception as e:
        logger.error(f"创建记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

# 注意：具体路径路由必须放在路径参数路由之前，已移动到文件末尾

@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    request: MemoryUpdateRequest,
    user_id: Optional[str] = Depends(get_current_user)
) -> MemoryResponse:
    """更新记忆"""
    # 检查记忆是否存在
    memory = await memory_service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    # 检查权限
    if memory.user_id and memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权更新该记忆")
    
    updated_memory = await memory_service.update_memory(memory_id, request)
    if not updated_memory:
        raise HTTPException(status_code=500, detail="更新记忆失败")
    
    return MemoryResponse.from_memory(updated_memory)

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: Optional[str] = Depends(get_current_user)
):
    """删除记忆"""
    # 检查记忆是否存在
    memory = await memory_service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    # 检查权限
    if memory.user_id and memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权删除该记忆")
    
    success = await memory_service.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=500, detail="删除记忆失败")
    
    return {"message": "记忆已删除"}

@router.get("/search", response_model=List[MemoryResponse])
async def search_memories(
    query: str,
    memory_types: Optional[List[MemoryType]] = Query(None),
    status: Optional[List[MemoryStatus]] = Query(None),
    min_importance: Optional[float] = Query(None, ge=0, le=1),
    max_importance: Optional[float] = Query(None, ge=0, le=1),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Depends(get_current_user)
) -> List[MemoryResponse]:
    """搜索记忆"""
    filters = MemoryFilters(
        memory_types=memory_types,
        status=status,
        min_importance=min_importance,
        max_importance=max_importance,
        tags=tags,
        session_id=session_id,
        user_id=user_id
    )
    
    results = await memory_service.search_memories(query, filters, limit)
    
    return [
        MemoryResponse.from_memory(memory, relevance_score=score)
        for memory, score in results
    ]

@router.get("/session/{session_id}", response_model=List[MemoryResponse])
async def get_session_memories(
    session_id: str,
    memory_type: Optional[MemoryType] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = Depends(get_current_user)
) -> List[MemoryResponse]:
    """获取会话的所有记忆"""
    memories = await memory_service.get_session_memories(
        session_id,
        memory_type,
        limit
    )
    
    # 过滤用户权限
    filtered_memories = [
        m for m in memories 
        if not m.user_id or m.user_id == user_id
    ]
    
    return [MemoryResponse.from_memory(m) for m in filtered_memories]

@router.post("/{memory_id}/associate")
async def associate_memories(
    memory_id: str,
    target_memory_id: str,
    weight: float = Query(0.5, ge=0, le=1),
    association_type: str = Query("related"),
    user_id: Optional[str] = Depends(get_current_user)
):
    """关联两个记忆"""
    # 检查权限
    memory1 = await memory_service.get_memory(memory_id)
    memory2 = await memory_service.get_memory(target_memory_id)
    
    if not memory1 or not memory2:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    if (memory1.user_id and memory1.user_id != user_id) or \
       (memory2.user_id and memory2.user_id != user_id):
        raise HTTPException(status_code=403, detail="无权关联这些记忆")
    
    await memory_service.associate_memories(
        memory_id,
        target_memory_id,
        weight,
        association_type
    )
    
    return {"message": "记忆已关联"}

@router.get("/{memory_id}/related", response_model=List[MemoryResponse])
async def get_related_memories(
    memory_id: str,
    depth: int = Query(2, ge=1, le=5),
    limit: int = Query(10, ge=1, le=50),
    user_id: Optional[str] = Depends(get_current_user)
) -> List[MemoryResponse]:
    """获取相关记忆"""
    # 检查权限
    memory = await memory_service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    if memory.user_id and memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该记忆")
    
    related = await memory_service.get_related_memories(
        memory_id,
        depth,
        limit
    )
    
    return [
        MemoryResponse.from_memory(memory, relevance_score=score)
        for memory, score in related
    ]

@router.post("/consolidate/{session_id}")
async def consolidate_session_memories(
    session_id: str,
    user_id: Optional[str] = Depends(get_current_user)
):
    """巩固会话记忆"""
    try:
        await memory_service.consolidate_memories(session_id)
        return {"message": "记忆巩固完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记忆巩固失败: {str(e)}")

@router.get("/analytics", response_model=MemoryAnalytics)
async def get_memory_analytics(
    days_back: int = Query(7, ge=1, le=365),
    session_id: Optional[str] = Query(None)
) -> MemoryAnalytics:
    """获取记忆分析统计"""
    start_time = utc_now() - timedelta(days=days_back)
    end_time = utc_now()
    return await memory_service.get_memory_analytics(
        session_id=session_id,
        start_time=start_time,
        end_time=end_time
    )

@router.get("/analytics/patterns")
async def get_memory_patterns(
    days_back: int = Query(7, ge=1, le=365)
):
    """获取记忆模式分析"""
    start_time = utc_now() - timedelta(days=days_back)
    memories = await memory_service.export_memories()
    hourly_distribution: Dict[str, int] = {str(i): 0 for i in range(24)}
    daily_distribution: Dict[str, int] = {}
    tag_frequency: Dict[str, int] = {}
    type_distribution: Dict[str, int] = {}
    for memory in memories:
        created_at = memory.get("created_at")
        if not created_at:
            continue
        created_dt = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at)
        if created_dt < start_time:
            continue
        hour_key = str(created_dt.hour)
        hourly_distribution[hour_key] = hourly_distribution.get(hour_key, 0) + 1
        day_key = created_dt.date().isoformat()
        daily_distribution[day_key] = daily_distribution.get(day_key, 0) + 1
        for tag in memory.get("tags", []):
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
        mem_type = memory.get("type")
        if mem_type:
            type_distribution[mem_type if isinstance(mem_type, str) else mem_type.value] = \
                type_distribution.get(mem_type if isinstance(mem_type, str) else mem_type.value, 0) + 1
    peak_hours = sorted(hourly_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    most_active_days = sorted(daily_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        "time_patterns": {
            "hourly_distribution": hourly_distribution,
            "daily_distribution": daily_distribution
        },
        "content_patterns": {
            "tag_frequency": tag_frequency,
            "type_distribution": type_distribution
        },
        "usage_patterns": {
            "peak_hours": peak_hours,
            "most_active_days": most_active_days
        }
    }

@router.get("/analytics/trends")
async def get_memory_trends(
    days: int = Query(30, ge=7, le=365)
):
    """获取记忆趋势分析"""
    start_time = utc_now() - timedelta(days=days)
    memories = await memory_service.export_memories()
    daily_trends: Dict[str, Dict[str, Any]] = {}
    for memory in memories:
        created_at = memory.get("created_at")
        if not created_at:
            continue
        created_dt = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at)
        if created_dt < start_time:
            continue
        day_key = created_dt.date().isoformat()
        entry = daily_trends.setdefault(day_key, {
            "memory_count": 0,
            "avg_importance": 0.0,
            "total_access": 0,
            "type_distribution": {}
        })
        entry["memory_count"] += 1
        entry["total_access"] += memory.get("access_count", 0)
        importance = memory.get("importance", 0)
        # 动态平均
        count = entry["memory_count"]
        entry["avg_importance"] = ((entry["avg_importance"] * (count - 1)) + importance) / count
        mem_type = memory.get("type")
        if mem_type:
            type_key = mem_type if isinstance(mem_type, str) else mem_type.value
            entry["type_distribution"][type_key] = entry["type_distribution"].get(type_key, 0) + 1
    total_memories = sum(d["memory_count"] for d in daily_trends.values())
    avg_daily_creation = total_memories / max(len(daily_trends), 1)
    growth_rate = total_memories / max(days, 1)
    return {
        "period": {
            "start_date": start_time.isoformat(),
            "end_date": utc_now().isoformat(),
            "total_days": days
        },
        "daily_trends": daily_trends,
        "summary": {
            "total_memories": total_memories,
            "avg_daily_creation": avg_daily_creation,
            "growth_rate": growth_rate
        }
    }

@router.get("/analytics/graph/stats")
async def get_memory_graph_stats():
    """获取记忆关联图统计"""
    await memory_service.initialize()
    graph = memory_service.association_graph.graph
    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()
    isolated = list(nx.isolates(graph)) if total_nodes else []
    degree_sequence = [deg for _, deg in graph.degree()] if total_nodes else []
    max_connections = max(degree_sequence) if degree_sequence else 0
    avg_connections = sum(degree_sequence) / total_nodes if total_nodes else 0
    type_counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        mem_type = data.get("memory_type")
        if mem_type:
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
    return {
        "graph_overview": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "density": nx.density(graph) if total_nodes else 0,
            "connected_components": nx.number_connected_components(graph.to_undirected()) if total_nodes else 0
        },
        "node_statistics": {
            "isolated_nodes": len(isolated),
            "connected_nodes": total_nodes - len(isolated),
            "max_connections": max_connections,
            "avg_connections": avg_connections
        },
        "connectivity_distribution": {
            "0_connections": len([d for d in degree_sequence if d == 0]),
            "1-2_connections": len([d for d in degree_sequence if 1 <= d <= 2]),
            "3-5_connections": len([d for d in degree_sequence if 3 <= d <= 5]),
            "6+_connections": len([d for d in degree_sequence if d >= 6])
        },
        "memory_types_in_graph": type_counts
    }

@router.post("/cleanup")
async def cleanup_old_memories(
    days_old: int = Query(30, ge=7, le=365),
    min_importance: float = Query(0.3, ge=0, le=1),
    user_id: Optional[str] = Depends(get_current_user)
):
    """清理旧记忆"""
    try:
        await memory_service.cleanup_old_memories(days_old, min_importance)
        return {"message": "旧记忆清理完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> MemoryResponse:
    """获取单个记忆"""
    memory = await memory_service.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    # 检查权限
    if memory.user_id and memory.user_id != user_id:
        raise HTTPException(status_code=403, detail="无权访问该记忆")
    
    return MemoryResponse.from_memory(memory)
from src.core.logging import get_logger
