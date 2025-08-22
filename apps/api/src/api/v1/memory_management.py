"""记忆管理API端点"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import StreamingResponse
import json
import io
import logging

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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memories", tags=["Memory Management"])


@router.post("/", response_model=MemoryResponse)
async def create_memory(
    request: MemoryCreateRequest,
    session_id: Optional[str] = Header(None),
    user_id: Optional[str] = Depends(get_current_user)
) -> MemoryResponse:
    """创建新记忆"""
    try:
        memory = await memory_service.create_memory(
            content=request.content,
            memory_type=request.type,
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