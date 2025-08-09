"""
工作流管理API路由
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import JSONResponse

from ...models.schemas.workflow import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowExecuteRequest, WorkflowControlRequest,
    CheckpointResponse
)
from ...services.workflow_service import workflow_service

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow_data: WorkflowCreate):
    """创建新工作流"""
    try:
        return await workflow_service.create_workflow(workflow_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    status: Optional[str] = Query(None, description="工作流状态过滤"),
    limit: int = Query(100, description="返回数量限制"),
    offset: int = Query(0, description="偏移量")
):
    """列出工作流"""
    try:
        return await workflow_service.list_workflows(status, limit, offset)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str = Path(..., description="工作流ID")
):
    """获取工作流详情"""
    try:
        return await workflow_service.get_workflow_status(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{workflow_id}/start", response_model=WorkflowResponse)
async def start_workflow(
    workflow_id: str = Path(..., description="工作流ID"),
    execute_data: Optional[WorkflowExecuteRequest] = None
):
    """启动工作流执行"""
    try:
        input_data = execute_data.input_data if execute_data else None
        return await workflow_service.start_workflow(workflow_id, input_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/status", response_model=WorkflowResponse)
async def get_workflow_status(
    workflow_id: str = Path(..., description="工作流ID")
):
    """查询工作流状态"""
    try:
        return await workflow_service.get_workflow_status(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{workflow_id}/control")
async def control_workflow(
    workflow_id: str = Path(..., description="工作流ID"),
    control_data: WorkflowControlRequest = ...
):
    """控制工作流执行 (暂停/恢复/取消)"""
    try:
        if control_data.action == "pause":
            success = await workflow_service.pause_workflow(workflow_id)
            if success:
                return {"message": "工作流已暂停", "workflow_id": workflow_id}
            else:
                raise HTTPException(status_code=400, detail="暂停工作流失败")
        
        elif control_data.action == "resume":
            success = await workflow_service.resume_workflow(workflow_id)
            if success:
                return {"message": "工作流已恢复", "workflow_id": workflow_id}
            else:
                raise HTTPException(status_code=400, detail="恢复工作流失败")
        
        elif control_data.action == "cancel":
            success = await workflow_service.cancel_workflow(workflow_id)
            if success:
                return {"message": "工作流已取消", "workflow_id": workflow_id}
            else:
                raise HTTPException(status_code=400, detail="取消工作流失败")
        
        else:
            raise HTTPException(status_code=400, detail=f"不支持的操作: {control_data.action}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/checkpoints", response_model=List[CheckpointResponse])
async def get_workflow_checkpoints(
    workflow_id: str = Path(..., description="工作流ID")
):
    """获取工作流检查点列表"""
    try:
        checkpoints = await workflow_service.get_workflow_checkpoints(workflow_id)
        return [
            CheckpointResponse(
                id=cp["id"],
                workflow_id=cp["workflow_id"],
                created_at=cp["created_at"],
                version=cp["version"],
                metadata=cp["metadata"]
            )
            for cp in checkpoints
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str = Path(..., description="工作流ID")
):
    """删除工作流"""
    try:
        # 先取消工作流（如果正在运行）
        await workflow_service.cancel_workflow(workflow_id)
        
        # 删除工作流（软删除）
        result = await workflow_service.delete_workflow(workflow_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="工作流不存在")
            
        return {"message": "工作流已删除", "workflow_id": workflow_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 健康检查端点
@router.get("/health/check")
async def health_check():
    """工作流服务健康检查"""
    return {
        "status": "healthy",
        "service": "workflow_service",
        "timestamp": "2025-01-01T00:00:00Z"
    }