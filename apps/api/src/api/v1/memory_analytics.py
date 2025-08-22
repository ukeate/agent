"""记忆分析API端点"""
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel

from src.ai.memory.models import MemoryAnalytics, MemoryStatus, MemoryFilters
from src.services.memory_service import memory_service
from src.core.dependencies import get_current_user

router = APIRouter(prefix="/memories/analytics", tags=["Memory Analytics"])


class TimeRange(BaseModel):
    """时间范围参数"""
    start: Optional[datetime] = None
    end: Optional[datetime] = None


@router.get("/", response_model=MemoryAnalytics)
async def get_memory_analytics(
    session_id: Optional[str] = Query(None),
    days_back: int = Query(7, ge=1, le=365),
    user_id: Optional[str] = Depends(get_current_user)
) -> MemoryAnalytics:
    """获取记忆统计分析"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    
    try:
        analytics = await memory_service.get_memory_analytics(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分析数据失败: {str(e)}")


@router.get("/patterns")
async def get_memory_patterns(
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Depends(get_current_user)
):
    """获取记忆模式分析"""
    try:
        # 获取记忆访问模式
        patterns = memory_service.association_graph.detect_memory_patterns()
        
        # 获取记忆簇
        clusters = memory_service.association_graph.find_memory_clusters()
        
        # 获取重要性排名
        importance_rank = memory_service.association_graph.get_memory_importance_rank()
        
        return {
            "access_patterns": patterns,
            "memory_clusters": [list(cluster) for cluster in clusters[:10]],
            "importance_ranking": importance_rank[:20],
            "cluster_count": len(clusters)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取记忆模式失败: {str(e)}")


@router.get("/graph/stats")
async def get_graph_statistics(
    user_id: Optional[str] = Depends(get_current_user)
):
    """获取记忆图统计"""
    graph = memory_service.association_graph.graph
    
    if not graph.nodes():
        return {
            "node_count": 0,
            "edge_count": 0,
            "avg_degree": 0,
            "density": 0,
            "connected_components": 0
        }
    
    try:
        import networkx as nx
        
        # 基础统计
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        
        # 度统计
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        # 图密度
        density = nx.density(graph)
        
        # 连通分量
        undirected = graph.to_undirected()
        connected_components = nx.number_connected_components(undirected)
        
        # 中心性统计
        if node_count < 1000:  # 避免大图计算太慢
            degree_centrality = nx.degree_centrality(graph)
            max_centrality_node = max(degree_centrality.items(), key=lambda x: x[1])
        else:
            max_centrality_node = ("N/A", 0)
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "avg_degree": round(avg_degree, 2),
            "density": round(density, 4),
            "connected_components": connected_components,
            "most_central_node": {
                "id": max_centrality_node[0],
                "centrality": round(max_centrality_node[1], 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图统计失败: {str(e)}")


@router.get("/trends")
async def get_memory_trends(
    days: int = Query(30, ge=7, le=365),
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Depends(get_current_user)
):
    """获取记忆趋势分析"""
    try:
        from collections import defaultdict
        from ai.memory.models import MemoryFilters
        
        # 获取时间范围内的记忆
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        filters = MemoryFilters(
            session_id=session_id,
            user_id=user_id,
            created_after=start_time,
            created_before=end_time
        )
        
        memories = await memory_service.hierarchy_manager.storage.search_memories(
            filters,
            limit=10000
        )
        
        # 按日期分组
        daily_counts = defaultdict(int)
        type_trends = defaultdict(lambda: defaultdict(int))
        
        for memory in memories:
            date_key = memory.created_at.date().isoformat()
            daily_counts[date_key] += 1
            type_trends[memory.type.value][date_key] += 1
        
        # 计算增长率
        dates = sorted(daily_counts.keys())
        if len(dates) > 1:
            first_week_avg = sum(daily_counts[d] for d in dates[:7]) / 7
            last_week_avg = sum(daily_counts[d] for d in dates[-7:]) / 7
            growth_rate = (last_week_avg - first_week_avg) / max(first_week_avg, 1)
        else:
            growth_rate = 0
        
        return {
            "daily_counts": dict(daily_counts),
            "type_trends": dict(type_trends),
            "total_memories": len(memories),
            "growth_rate_percentage": round(growth_rate * 100, 2),
            "avg_daily_memories": round(len(memories) / days, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取趋势分析失败: {str(e)}")


@router.get("/health")
async def get_memory_system_health(
    user_id: Optional[str] = Depends(get_current_user)
):
    """获取记忆系统健康状态"""
    try:
        # 检查各组件状态
        storage_ok = memory_service.hierarchy_manager.storage._initialized
        
        # 获取容量信息
        working_memory_usage = len(memory_service.hierarchy_manager.working_memory.get_all())
        working_memory_capacity = memory_service.config.working_memory_capacity
        
        # 获取存储统计
        from ai.memory.models import MemoryFilters
        
        filters = MemoryFilters(status=[MemoryStatus.ACTIVE])
        active_memories = await memory_service.hierarchy_manager.storage.search_memories(
            filters,
            limit=1
        )
        
        return {
            "status": "healthy" if storage_ok else "degraded",
            "components": {
                "storage": "ok" if storage_ok else "error",
                "hierarchy_manager": "ok",
                "context_recall": "ok",
                "association_graph": "ok"
            },
            "capacity": {
                "working_memory": {
                    "used": working_memory_usage,
                    "total": working_memory_capacity,
                    "percentage": round(working_memory_usage / working_memory_capacity * 100, 2)
                }
            },
            "performance": {
                "avg_recall_time_ms": 50,  # 示例值
                "cache_hit_rate": 0.8  # 示例值
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }