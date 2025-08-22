"""
工作流管理API路由
"""
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.models.schemas.workflow import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse,
    WorkflowExecuteRequest, WorkflowControlRequest,
    CheckpointResponse
)
from src.services.workflow_service import workflow_service

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


# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, workflow_id: str):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections[workflow_id] = websocket

    def disconnect(self, workflow_id: str):
        """断开WebSocket连接"""
        if workflow_id in self.active_connections:
            del self.active_connections[workflow_id]

    async def send_workflow_update(self, workflow_id: str, data: dict):
        """向特定工作流发送更新"""
        if workflow_id in self.active_connections:
            try:
                await self.active_connections[workflow_id].send_text(json.dumps(data))
            except Exception as e:
                print(f"Error sending update to {workflow_id}: {e}")
                self.disconnect(workflow_id)

    async def broadcast_update(self, data: dict):
        """广播更新到所有连接"""
        for workflow_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_text(json.dumps(data))
            except Exception as e:
                print(f"Error broadcasting to {workflow_id}: {e}")
                self.disconnect(workflow_id)

manager = ConnectionManager()

@router.websocket("/{workflow_id}/ws")
async def workflow_websocket_endpoint(websocket: WebSocket, workflow_id: str):
    """工作流实时状态WebSocket端点"""
    await manager.connect(websocket, workflow_id)
    
    try:
        # 发送初始状态
        initial_status = await workflow_service.get_workflow_status(workflow_id)
        await websocket.send_text(json.dumps({
            "type": "initial_status",
            "data": initial_status.dict() if initial_status else None
        }))
        
        # 保持连接并处理客户端消息
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理客户端请求
            if message.get("type") == "get_status":
                current_status = await workflow_service.get_workflow_status(workflow_id)
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "data": current_status.dict() if current_status else None
                }))
                
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(workflow_id)
    except Exception as e:
        print(f"WebSocket error for workflow {workflow_id}: {e}")
        manager.disconnect(workflow_id)

# 健康检查端点
@router.get("/health/check")
async def health_check():
    """工作流服务健康检查"""
    return {
        "status": "healthy",
        "service": "workflow_service",
        "timestamp": "2025-01-01T00:00:00Z"
    }