"""链式思考(CoT)推理API端点"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from src.models.schemas.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningChain,
    ReasoningStreamChunk,
    ReasoningValidation
)
from src.services.reasoning_service import ReasoningService
from src.core.dependencies import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/reasoning", tags=["reasoning"])

# 创建服务实例
reasoning_service = ReasoningService()

@router.post("/chain", response_model=ReasoningResponse)
async def create_reasoning_chain(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> ReasoningResponse:
    """

    创建新的推理链
    
    Args:
        request: 推理请求参数
        background_tasks: 后台任务
        current_user: 当前用户
    
    Returns:
        推理响应
    """
    try:
        logger.info(
            f"用户 {current_user.get('id', 'unknown')} "
            f"创建推理链: {request.problem[:50]}..."
        )
        
        # 如果不是流式响应，直接执行
        if not request.stream:
            result = await reasoning_service.execute_reasoning(request, current_user)
            return result
        else:
            # 流式响应在另一个端点处理
            raise HTTPException(
                status_code=400,
                detail="请使用 /reasoning/stream 端点进行流式推理"
            )
            
    except Exception as e:
        logger.error(f"创建推理链失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def create_reasoning_stream(
    request: ReasoningRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    创建流式推理链
    
    Args:
        request: 推理请求参数
        current_user: 当前用户
    
    Returns:
        流式响应
    """
    try:
        logger.info(
            f"用户 {current_user.get('id', 'unknown')} "
            f"创建流式推理: {request.problem[:50]}..."
        )
        
        async def generate():
            """生成流式响应"""
            async for chunk in reasoning_service.stream_reasoning(request, current_user):
                # 将每个块转换为SSE格式
                data = json.dumps(chunk.model_dump(), ensure_ascii=False)
                yield f"data: {data}\n\n"
            
            # 发送结束信号
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"创建流式推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chain/{chain_id}", response_model=ReasoningChain)
async def get_reasoning_chain(
    chain_id: UUID,
    current_user: dict = Depends(get_current_user)
) -> ReasoningChain:
    """
    获取推理链详情
    
    Args:
        chain_id: 推理链ID
        current_user: 当前用户
    
    Returns:
        推理链对象
    """
    try:
        chain = await reasoning_service.get_chain(
            chain_id=chain_id,
            user_id=current_user.get('id')
        )
        
        if not chain:
            raise HTTPException(status_code=404, detail="推理链不存在")
        
        return chain
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取推理链失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[ReasoningChain])
async def get_reasoning_history(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> List[ReasoningChain]:
    """
    获取推理历史
    
    Args:
        limit: 限制数量
        offset: 偏移量
        current_user: 当前用户
    
    Returns:
        推理链列表
    """
    try:
        chains = await reasoning_service.get_user_history(
            user_id=current_user.get('id'),
            limit=limit,
            offset=offset
        )
        
        return chains
        
    except Exception as e:
        logger.error(f"获取推理历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chain/{chain_id}/validate")
async def validate_reasoning_chain(
    chain_id: UUID,
    step_number: Optional[int] = None,
    current_user: dict = Depends(get_current_user)
) -> ReasoningValidation:
    """
    验证推理链或特定步骤
    
    Args:
        chain_id: 推理链ID
        step_number: 可选，特定步骤号
        current_user: 当前用户
    
    Returns:
        验证结果
    """
    try:
        validation = await reasoning_service.validate_chain(
            chain_id=chain_id,
            step_number=step_number,
            user_id=current_user.get('id')
        )
        
        if not validation:
            raise HTTPException(status_code=404, detail="无法验证")
        
        return validation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"验证推理链失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chain/{chain_id}/branch")
async def create_reasoning_branch(
    chain_id: UUID,
    parent_step_number: int,
    reason: str,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    创建推理分支
    
    Args:
        chain_id: 推理链ID
        parent_step_number: 父步骤号
        reason: 分支原因
        current_user: 当前用户
    
    Returns:
        分支ID
    """
    try:
        branch_id = await reasoning_service.create_branch(
            chain_id=chain_id,
            parent_step_number=parent_step_number,
            reason=reason,
            user_id=current_user.get('id')
        )
        
        if not branch_id:
            raise HTTPException(status_code=400, detail="创建分支失败")
        
        return {"branch_id": str(branch_id)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建推理分支失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chain/{chain_id}/recover")
async def recover_reasoning_chain(
    chain_id: UUID,
    strategy: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    恢复推理链
    
    Args:
        chain_id: 推理链ID
        strategy: 恢复策略 (backtrack, branch, restart, refine, alternative)
        current_user: 当前用户
    
    Returns:
        恢复结果
    """
    try:
        success = await reasoning_service.recover_chain(
            chain_id=chain_id,
            strategy=strategy,
            user_id=current_user.get('id')
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="恢复失败")
        
        return {
            "success": success,
            "message": "推理链已恢复"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复推理链失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_reasoning_stats(
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    获取推理统计信息
    
    Args:
        current_user: 当前用户
    
    Returns:
        统计信息
    """
    try:
        stats = await reasoning_service.get_user_stats(
            user_id=current_user.get('id')
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"获取推理统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chain/{chain_id}")
async def delete_reasoning_chain(
    chain_id: UUID,
    current_user: dict = Depends(get_current_user)
) -> dict:
    """
    删除推理链
    
    Args:
        chain_id: 推理链ID
        current_user: 当前用户
    
    Returns:
        删除结果
    """
    try:
        success = await reasoning_service.delete_chain(
            chain_id=chain_id,
            user_id=current_user.get('id')
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="推理链不存在")
        
        return {
            "success": True,
            "message": "推理链已删除"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除推理链失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
from src.core.logging import get_logger
