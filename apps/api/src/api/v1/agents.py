"""
智能体API端点
提供ReAct智能体的REST API接口
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import Field
from typing import Dict, List, Optional, Any
import json
from src.services.agent_service import get_agent_service, AgentService
from src.core.dependencies import get_current_user
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# 请求模型
class CreateAgentSessionRequest(ApiBaseModel):
    """创建智能体会话请求"""
    agent_type: str = Field(default="react", description="智能体类型")
    conversation_title: Optional[str] = Field(None, description="对话标题")
    agent_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="智能体配置")

class ChatRequest(ApiBaseModel):
    """对话请求"""
    message: str = Field(..., description="用户消息")
    stream: bool = Field(default=False, description="是否启用流式响应")

class TaskRequest(ApiBaseModel):
    """任务请求"""
    task_description: str = Field(..., description="任务描述")
    task_type: str = Field(default="general", description="任务类型")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="任务上下文")

class UpdateConversationTitleRequest(ApiBaseModel):
    """更新对话标题请求"""
    title: str = Field(..., description="对话标题")

# 响应模型
class AgentSessionResponse(ApiBaseModel):
    """智能体会话响应"""
    conversation_id: str
    agent_id: str
    agent_type: str
    status: str = "created"

class ChatResponse(ApiBaseModel):
    """对话响应"""
    conversation_id: str
    response: str
    steps: int
    tool_calls: List[Dict[str, Any]]
    completed: bool
    session_summary: Optional[Dict[str, Any]] = None

class ConversationHistoryResponse(ApiBaseModel):
    """对话历史响应"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    summary: Dict[str, Any]

class AgentStatusResponse(ApiBaseModel):
    """智能体状态响应"""
    conversation_id: str
    status: str
    session_summary: Optional[Dict[str, Any]] = None
    agent_type: str

@router.post("/sessions", response_model=AgentSessionResponse)
async def create_agent_session(
    request: CreateAgentSessionRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """创建新的智能体会话"""
    try:
        result = await agent_service.create_agent_session(
            user_id=current_user,
            agent_type=request.agent_type,
            agent_config=request.agent_config,
            conversation_title=request.conversation_title
        )
        
        return AgentSessionResponse(
            conversation_id=result["conversation_id"],
            agent_id=result["agent_id"],
            agent_type=result["agent_type"]
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(
            "创建智能体会话失败",
            error=str(e),
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建智能体会话失败: {str(e)}"
        )

@router.post("/react/chat/{conversation_id}", response_model=ChatResponse)
async def chat_with_react_agent(
    conversation_id: str,
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """与ReAct智能体对话"""
    try:
        if request.stream:
            # 流式响应
            async def generate_stream():
                async for step_data in agent_service.chat_with_agent(
                    conversation_id=conversation_id,
                    user_input=request.message,
                    user_id=current_user,
                    stream=True
                ):
                    yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"
                
                # 发送结束标记
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 普通响应
            result = await agent_service.chat_with_agent(
                conversation_id=conversation_id,
                user_input=request.message,
                user_id=current_user,
                stream=False
            )
            
            return ChatResponse(**result)
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "智能体对话失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"智能体对话失败: {str(e)}"
        )

@router.post("/react/task/{conversation_id}")
async def assign_task_to_react_agent(
    conversation_id: str,
    request: TaskRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """为ReAct智能体分配任务"""
    try:
        # 将任务转换为对话消息
        task_message = f"任务: {request.task_description}"
        if request.context:
            task_message += f"\n上下文: {json.dumps(request.context, ensure_ascii=False)}"
        
        result = await agent_service.chat_with_agent(
            conversation_id=conversation_id,
            user_input=task_message,
            user_id=current_user,
            stream=False
        )
        
        return {
            "conversation_id": conversation_id,
            "task_type": request.task_type,
            "task_description": request.task_description,
            "result": result["response"],
            "steps": result["steps"],
            "tool_calls": result["tool_calls"],
            "completed": result["completed"]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "任务分配失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务分配失败: {str(e)}"
        )

@router.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    limit: Optional[int] = None,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """获取对话历史"""
    try:
        result = await agent_service.get_conversation_history(
            conversation_id=conversation_id,
            user_id=current_user,
            limit=limit
        )
        
        return ConversationHistoryResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "获取对话历史失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话历史失败: {str(e)}"
        )

@router.put("/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    request: UpdateConversationTitleRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """更新对话标题"""
    try:
        normalized = request.title.strip()
        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="对话标题不能为空"
            )
        result = await agent_service.update_conversation_title(
            conversation_id=conversation_id,
            user_id=current_user,
            title=normalized
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(
            "更新对话标题失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新对话标题失败: {str(e)}"
        )

@router.get("/conversations/{conversation_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    conversation_id: str,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """获取智能体状态"""
    try:
        result = await agent_service.get_agent_status(conversation_id, current_user)
        
        return AgentStatusResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "获取智能体状态失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取智能体状态失败: {str(e)}"
        )

@router.delete("/conversations/{conversation_id}")
async def close_agent_session(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """关闭智能体会话"""
    try:
        # 在后台任务中执行清理
        background_tasks.add_task(
            agent_service.close_agent_session,
            conversation_id,
            current_user
        )
        
        return {
            "conversation_id": conversation_id,
            "status": "closing",
            "message": "会话正在关闭"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "关闭智能体会话失败",
            error=str(e),
            conversation_id=conversation_id,
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关闭智能体会话失败: {str(e)}"
        )

@router.get("/conversations")
async def list_user_conversations(
    limit: int = 20,
    offset: int = 0,
    query: Optional[str] = None,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
):
    """列出用户的对话会话"""
    try:
        conversations, total = await agent_service.list_user_conversations(
            user_id=current_user,
            limit=limit,
            offset=offset,
            query=query
        )
        
        return {
            "conversations": conversations,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(
            "列出用户对话失败",
            error=str(e),
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出用户对话失败: {str(e)}"
        )

@router.get("/performance")
async def get_agent_performance_metrics(
    agent_service: AgentService = Depends(get_agent_service)
):
    """获取智能体性能指标"""
    try:
        # 简单的性能统计
        active_agents = len(agent_service.agents)
        active_conversations = len(agent_service.conversation_service.active_sessions) if agent_service.conversation_service else 0
        
        return {
            "active_agents": active_agents,
            "active_conversations": active_conversations,
            "agent_types": ["react"],
            "status": "healthy"
        }
        
    except Exception as e:
        logger.error(
            "获取性能指标失败",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )

# WebSocket支持（需要FastAPI WebSocket）
@router.websocket("/ws/{conversation_id}")
async def websocket_chat(
    websocket,  # WebSocket类型
    conversation_id: str,
    agent_service: AgentService = Depends(get_agent_service)
):
    """WebSocket实时对话接口"""
    await websocket.accept()
    
    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_input = message_data.get("message", "")
            user_id = message_data.get("user_id", "anonymous")
            
            if not user_input:
                await websocket.send_text(json.dumps({
                    "error": "消息内容不能为空"
                }))
                continue
            
            # 流式处理智能体响应
            async for step_data in agent_service.chat_with_agent(
                conversation_id=conversation_id,
                user_input=user_input,
                user_id=user_id,
                stream=True
            ):
                await websocket.send_text(json.dumps(step_data, ensure_ascii=False))
                
    except Exception as e:
        logger.error(
            "WebSocket对话失败",
            error=str(e),
            conversation_id=conversation_id
        )
        await websocket.send_text(json.dumps({
            "error": f"对话失败: {str(e)}"
        }))
    finally:
        await websocket.close()
