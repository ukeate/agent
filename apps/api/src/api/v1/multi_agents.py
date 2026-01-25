"""
多智能体协作API路由
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import Response
from pydantic import Field
import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.services.multi_agent_service import MultiAgentService
from src.ai.autogen.config import AgentRole, ConversationConfig
from src.core.constants import ConversationConstants
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])

# 全局服务实例（单例模式）
_multi_agent_service_instance = None
_agent_configs_loaded_at = utc_now().isoformat()

# 依赖注入
async def get_multi_agent_service() -> MultiAgentService:
    """获取多智能体服务实例（单例）"""
    global _multi_agent_service_instance
    
    # 懒加载创建单例实例
    if _multi_agent_service_instance is None:
        _multi_agent_service_instance = MultiAgentService()
        logger.info("MultiAgentService单例实例创建成功")
    
    return _multi_agent_service_instance

# Pydantic模型
class CreateConversationRequest(ApiBaseModel):
    """创建对话请求"""
    message: str = Field(..., description="初始消息", min_length=1, max_length=5000)
    agent_roles: Optional[List[AgentRole]] = Field(
        default=None, 
        description="参与的智能体角色列表"
    )
    user_context: Optional[str] = Field(
        default=None, 
        description="用户上下文信息",
        max_length=2000
    )
    max_rounds: Optional[int] = Field(
        default=10, 
        description="最大对话轮数", 
        ge=1, 
        le=50
    )
    timeout_seconds: Optional[int] = Field(
        default=300, 
        description="超时时间(秒)", 
        ge=30, 
        le=1800
    )
    auto_reply: Optional[bool] = Field(
        default=True, 
        description="是否自动回复"
    )

class ConversationResponse(ApiBaseModel):
    """对话响应"""
    conversation_id: str
    status: str
    participants: List[Dict[str, Any]]
    created_at: str
    config: Dict[str, Any]
    initial_status: Dict[str, Any]

class ConversationStatusResponse(ApiBaseModel):
    """对话状态响应"""
    conversation_id: str
    status: str
    created_at: str
    updated_at: str
    message_count: int
    round_count: int
    participants: List[Dict[str, Any]]
    config: Dict[str, Any]
    real_time_stats: Optional[Dict[str, Any]] = None

class TerminateConversationRequest(ApiBaseModel):
    """终止对话请求"""
    reason: Optional[str] = Field(
        default="用户终止", 
        description="终止原因",
        max_length=500
    )

class MessagesResponse(ApiBaseModel):
    """消息响应"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    total_count: int
    returned_count: int
    offset: int

class ConversationSummaryResponse(ApiBaseModel):
    """对话摘要响应"""
    conversation_id: str
    message_count: int
    round_count: int
    key_points: List[str]
    decisions_made: List[str]
    action_items: List[str]
    participants_summary: Dict[str, str]

class ConversationAnalysisResponse(ApiBaseModel):
    """对话分析响应"""
    conversation_id: str
    sentiment_analysis: Dict[str, float]
    topic_distribution: Dict[str, float]
    interaction_patterns: List[Dict[str, Any]]
    recommendations: List[str]

# API路由
@router.post(
    "/conversation", 
    response_model=ConversationResponse,
    summary="创建多智能体对话",
    description="启动一个新的多智能体协作对话会话"
)
async def create_conversation(
    request: CreateConversationRequest,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> ConversationResponse:
    """创建多智能体对话"""
    try:
        # 构建对话配置
        conversation_config = ConversationConfig(
            max_rounds=request.max_rounds or 10,
            timeout_seconds=request.timeout_seconds or 300,
            auto_reply=request.auto_reply if request.auto_reply is not None else True,
        )
        
        # 定义WebSocket回调函数
        async def websocket_callback(data):
            """WebSocket回调函数，向连接的客户端推送消息"""
            session_id = data.get("session_id")
            if session_id and session_id in manager.active_connections:
                try:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": data.get("type"),
                            "data": data,
                            "timestamp": utc_now().isoformat()
                        }), 
                        session_id
                    )
                except Exception as e:
                    logger.warning(f"WebSocket推送失败: {e}")
        
        # 创建对话
        result = await service.create_multi_agent_conversation(
            initial_message=request.message,
            agent_roles=request.agent_roles,
            conversation_config=conversation_config,
            user_context=request.user_context,
            websocket_callback=websocket_callback,
        )
        
        logger.info(
            "多智能体对话创建成功",
            conversation_id=result["conversation_id"],
            participant_count=len(result["participants"]),
        )
        
        return ConversationResponse(**result)
        
    except Exception as e:
        logger.error(
            "创建多智能体对话失败",
            error=str(e),
            message_length=len(request.message),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建对话失败: {str(e)}"
        )

@router.get(
    "/conversation/{conversation_id}/status",
    response_model=ConversationStatusResponse,
    summary="获取对话状态",
    description="获取指定对话的详细状态信息"
)
async def get_conversation_status(
    conversation_id: str,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> ConversationStatusResponse:
    """获取对话状态"""
    try:
        status_data = await service.get_conversation_status(conversation_id)
        return ConversationStatusResponse(**status_data)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "获取对话状态失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取状态失败: {str(e)}"
        )

@router.post(
    "/conversation/{conversation_id}/pause",
    summary="暂停对话",
    description="暂停正在进行的多智能体对话"
)
async def pause_conversation(
    conversation_id: str,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """暂停对话"""
    try:
        result = await service.pause_conversation(conversation_id)
        
        logger.info(
            "对话暂停成功",
            conversation_id=conversation_id,
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "暂停对话失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停对话失败: {str(e)}"
        )

@router.post(
    "/conversation/{conversation_id}/resume",
    summary="恢复对话",
    description="恢复已暂停的多智能体对话"
)
async def resume_conversation(
    conversation_id: str,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """恢复对话"""
    try:
        # 定义WebSocket回调函数，向连接的客户端推送消息
        async def websocket_callback(data):
            """WebSocket回调函数，向连接的客户端推送消息"""
            target_session_id = service.get_ws_session_id(conversation_id) or conversation_id
            if target_session_id in manager.active_connections:
                try:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": data.get("type"),
                            "data": data,
                            "timestamp": utc_now().isoformat()
                        }), 
                        target_session_id
                    )
                except Exception as e:
                    logger.warning(f"WebSocket推送失败: {e}")
        
        result = await service.resume_conversation(conversation_id, websocket_callback)
        
        logger.info(
            "对话恢复成功",
            conversation_id=conversation_id,
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "恢复对话失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复对话失败: {str(e)}"
        )

@router.post(
    "/conversation/{conversation_id}/terminate",
    summary="终止对话",
    description="终止指定的多智能体对话并生成总结"
)
async def terminate_conversation(
    conversation_id: str,
    request: TerminateConversationRequest = TerminateConversationRequest(),
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """终止对话"""
    try:
        result = await service.terminate_conversation(
            conversation_id=conversation_id,
            reason=request.reason,
        )
        
        logger.info(
            "对话终止成功",
            conversation_id=conversation_id,
            reason=request.reason,
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "终止对话失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"终止对话失败: {str(e)}"
        )

@router.get(
    "/conversation/{conversation_id}/messages",
    response_model=MessagesResponse,
    summary="获取对话消息",
    description="获取指定对话的消息历史"
)
async def get_conversation_messages(
    conversation_id: str,
    limit: Optional[int] = None,
    offset: int = 0,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> MessagesResponse:
    """获取对话消息"""
    try:
        # 验证参数
        if limit is not None and limit <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit必须大于0"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="offset不能为负数"
            )
        
        result = await service.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
        
        return MessagesResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "获取对话消息失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取消息失败: {str(e)}"
        )

@router.get(
    "/conversation/{conversation_id}/summary",
    response_model=ConversationSummaryResponse,
    summary="获取对话摘要",
    description="生成并返回对话摘要信息"
)
async def get_conversation_summary(
    conversation_id: str,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> ConversationSummaryResponse:
    """获取对话摘要"""
    try:
        result = await service.get_conversation_summary(conversation_id)
        return ConversationSummaryResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "获取对话摘要失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话摘要失败: {str(e)}"
        )

@router.post(
    "/conversation/{conversation_id}/analyze",
    response_model=ConversationAnalysisResponse,
    summary="分析对话内容",
    description="分析对话内容并返回建议与统计"
)
async def analyze_conversation(
    conversation_id: str,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> ConversationAnalysisResponse:
    """分析对话内容"""
    try:
        result = await service.analyze_conversation(conversation_id)
        return ConversationAnalysisResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "分析对话内容失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"分析对话内容失败: {str(e)}"
        )

@router.get(
    "/conversation/{conversation_id}/export",
    summary="导出对话内容",
    description="导出对话内容为JSON文件"
)
async def export_conversation(
    conversation_id: str,
    format: str = Query("json"),
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Response:
    """导出对话内容"""
    if format.lower() != "json":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持json格式导出"
        )
    try:
        payload = await service.export_conversation(conversation_id)
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        filename = f"conversation_{conversation_id}.json"
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "导出对话内容失败",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出对话内容失败: {str(e)}"
        )

@router.get(
    "/conversations",
    summary="列出活跃对话",
    description="获取所有活跃的多智能体对话列表"
)
async def list_conversations(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> List[Dict[str, Any]]:
    """列出活跃对话"""
    try:
        conversations = await service.list_active_conversations(
            limit=limit,
            offset=offset,
        )
        
        logger.info(
            "获取活跃对话列表成功",
            conversation_count=len(conversations),
        )
        
        return conversations
        
    except Exception as e:
        logger.error(
            "获取活跃对话列表失败",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取对话列表失败: {str(e)}"
        )

@router.get(
    "/statistics",
    summary="获取智能体统计",
    description="获取多智能体系统的使用统计信息"
)
async def get_agent_statistics(
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """获取智能体统计信息"""
    try:
        stats = await service.get_agent_statistics()
        
        logger.info(
            "获取智能体统计成功",
            total_sessions=stats.get("total_active_sessions", 0),
        )
        
        return stats
        
    except Exception as e:
        logger.error(
            "获取智能体统计失败",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )

@router.post(
    "/cleanup",
    summary="清理非活跃会话",
    description="清理已完成或终止的会话，释放系统资源"
)
async def cleanup_sessions(
    background_tasks: BackgroundTasks,
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """清理非活跃会话"""
    try:
        # 在后台任务中执行清理
        background_tasks.add_task(service.cleanup_inactive_sessions)
        
        return {
            "message": "会话清理任务已启动",
            "status": "accepted",
        }
        
    except Exception as e:
        logger.error(
            "启动会话清理失败",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动清理任务失败: {str(e)}"
        )

# 智能体管理端点
@router.get(
    "/agents",
    summary="获取智能体列表",
    description="获取所有可用的智能体信息"
)
async def get_agents(
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """获取智能体列表"""
    try:
        # 从配置中获取智能体信息
        from src.ai.autogen.config import AGENT_CONFIGS
        
        agents = []
        for role, config in AGENT_CONFIGS.items():
            agents.append({
                "id": config.role,
                "name": config.name,
                "role": config.role,
                "status": "active",
                "capabilities": config.capabilities,
                "configuration": {
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "tools": config.tools,
                    "system_prompt": config.system_prompt,
                },
                "created_at": _agent_configs_loaded_at,
                "updated_at": _agent_configs_loaded_at,
            })
        
        logger.info(f"返回 {len(agents)} 个智能体")
        
        return {
            "success": True,
            "data": {
                "agents": agents,
                "total": len(agents)
            }
        }
        
    except Exception as e:
        logger.error(f"获取智能体列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取智能体列表失败: {str(e)}"
        )

# 系统状态端点
@router.get(
    "/health",
    summary="多智能体系统健康检查",
    description="检查多智能体系统的健康状态"
)
async def health_check(
    service: MultiAgentService = Depends(get_multi_agent_service),
) -> Dict[str, Any]:
    """多智能体系统健康检查"""
    try:
        # 检查服务状态
        active_sessions = len(service._active_sessions)
        
        # 简单的健康状态判断
        healthy = True
        issues = []
        
        if active_sessions > 100:  # 假设超过100个活跃会话可能存在问题
            issues.append("活跃会话数量过多")
            healthy = False
        
        return {
            "healthy": healthy,
            "timestamp": utc_now().isoformat(),
            "active_sessions": active_sessions,
            "issues": issues,
            "service_info": {
                "name": "MultiAgentService",
                "version": "1.0.0",
                "autogen_integrated": True,
            }
        }
        
    except Exception as e:
        logger.error(
            "多智能体系统健康检查失败",
            error=str(e),
        )
        return {
            "healthy": False,
            "timestamp": utc_now().isoformat(),
            "error": str(e),
            "service_info": {
                "name": "MultiAgentService",
                "version": "1.0.0",
                "autogen_integrated": False,
            }
        }

# WebSocket连接管理
class ConnectionManager:
    """WebSocket连接管理器 - 支持一个会话多个连接"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """接受连接 - 支持多个连接到同一会话"""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        connection_count = len(self.active_connections[session_id])
        logger.info(f"WebSocket连接已建立: {session_id} (第{connection_count}个连接)")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """断开特定连接"""
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].remove(websocket)
                remaining_count = len(self.active_connections[session_id])
                logger.info(f"WebSocket连接已断开: {session_id} (剩余{remaining_count}个连接)")
                
                # 如果没有连接了，删除整个会话
                if remaining_count == 0:
                    del self.active_connections[session_id]
                    logger.info(f"会话{session_id}所有连接已断开")
            except ValueError:
                logger.warning(f"尝试断开不存在的连接: {session_id}")
    
    async def send_personal_message(self, message: str, session_id: str):
        """向会话的所有连接广播消息"""
        if session_id not in self.active_connections:
            logger.warning(f"尝试向不存在的会话发送消息: {session_id}")
            return
            
        connections = self.active_connections[session_id].copy()  # 复制列表避免并发修改
        if not connections:
            logger.warning(f"会话没有活跃连接: {session_id}")
            return
            
        failed_connections = []
        
        for websocket in connections:
            try:
                # 检查WebSocket状态
                if websocket.client_state == 3:  # CLOSED
                    logger.warning(f"WebSocket连接已关闭，跳过发送: {session_id}")
                    failed_connections.append(websocket)
                    continue
                
                await websocket.send_text(message)
                logger.debug(f"消息发送成功到: {session_id}")
            except Exception as e:
                logger.error(f"向连接发送消息失败 (session: {session_id}): {type(e).__name__}: {e}")
                failed_connections.append(websocket)
        
        # 移除失败的连接
        for failed_ws in failed_connections:
            self.disconnect(failed_ws, session_id)
        
        successful_count = len(connections) - len(failed_connections)
        if successful_count > 0:
            logger.debug(f"消息已广播到{successful_count}个连接: {session_id}")
        else:
            logger.warning(f"没有成功发送到任何连接: {session_id}")

# 全局连接管理器
manager = ConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """多智能体WebSocket端点"""
    logger.info(f"WebSocket连接请求: {session_id}")
    
    try:
        await manager.connect(websocket, session_id)
        logger.info(f"WebSocket连接已建立: {session_id}")
        
        # 发送连接成功消息
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "data": {"session_id": session_id},
                "timestamp": utc_now().isoformat()
            }),
            session_id
        )
        logger.info(f"连接成功消息已发送: {session_id}")
        
        # 延迟创建服务实例，避免初始化问题
        service = None
        
        while True:
            # 接收前端消息
            logger.info(f"等待接收消息: {session_id}")
            data = await websocket.receive_text()
            logger.info(f"收到原始数据: {data}")
            message_data = json.loads(data)
            
            logger.info(f"收到WebSocket消息: {message_data}")
            logger.info(f"消息类型: {message_data.get('type')}")
            
            # 处理不同类型的消息
            if message_data.get("type") == "start_conversation":
                # 启动多智能体对话
                try:
                    # 使用单例服务实例，确保与REST API共享同一个实例
                    if service is None:
                        logger.info(f"获取MultiAgentService单例实例: {session_id}")
                        service = await get_multi_agent_service()
                        logger.info(f"MultiAgentService单例实例获取成功: {session_id}")
                    # 获取消息数据
                    msg_data = message_data.get("data", {})
                    initial_message = msg_data.get("message", "").strip()
                    participants = msg_data.get("participants", [])
                    
                    # 验证必要参数
                    if not initial_message:
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "data": {"message": "初始消息不能为空"},
                                "timestamp": utc_now().isoformat()
                            }),
                            session_id
                        )
                        continue
                    
                    logger.info(f"启动对话 - 消息: '{initial_message}' (长度: {len(initial_message)}), 参与者: {participants}")
                    
                    if not isinstance(participants, list):
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "data": {"message": "参与者格式不正确"},
                                "timestamp": utc_now().isoformat()
                            }),
                            session_id
                        )
                        continue

                    agent_roles = []
                    invalid_roles = []
                    for participant_id in participants:
                        try:
                            agent_roles.append(AgentRole(participant_id))
                        except ValueError:
                            invalid_roles.append(participant_id)

                    if invalid_roles:
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "data": {"message": f"未知智能体角色: {', '.join(invalid_roles)}"},
                                "timestamp": utc_now().isoformat()
                            }),
                            session_id
                        )
                        continue

                    if not agent_roles:
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "error",
                                "data": {"message": "未指定参与的智能体"},
                                "timestamp": utc_now().isoformat()
                            }),
                            session_id
                        )
                        continue
                    
                    logger.info(f"转换后的角色: {agent_roles}")
                    
                    # WebSocket回调函数，只向原始session_id推送消息，保持连接稳定
                    async def websocket_callback(data):
                        """WebSocket回调函数，向连接的客户端推送消息"""
                        try:
                            message = json.dumps({
                                "type": data.get("type", "message"),
                                "data": data,
                                "timestamp": utc_now().isoformat()
                            })
                            
                            # 只向原始session_id发送消息，保持连接稳定性
                            if session_id in manager.active_connections:
                                await manager.send_personal_message(message, session_id)
                                logger.debug(f"WebSocket消息已推送到: {session_id}")
                            else:
                                logger.warning(f"WebSocket连接不存在: {session_id}")
                                
                        except Exception as e:
                            logger.warning(f"WebSocket推送失败: {e}")
                    
                    # 调用服务创建对话，立即传递WebSocket回调
                    result = await service.create_multi_agent_conversation(
                        initial_message=initial_message,
                        agent_roles=agent_roles,
                        user_context=f"WebSocket会话: {session_id}",
                        conversation_config=ConversationConfig(
                            max_rounds=ConversationConstants.DEFAULT_MAX_ROUNDS,
                            timeout_seconds=ConversationConstants.DEFAULT_TIMEOUT_SECONDS,
                            auto_reply=ConversationConstants.DEFAULT_AUTO_REPLY
                        ),
                        websocket_callback=websocket_callback  # 立即传递回调！
                    )
                    
                    # 获取conversation_id，保持原有连接稳定
                    conversation_id = result["conversation_id"]
                    logger.info(f"对话创建成功，conversation_id: {conversation_id}")
                    service.bind_ws_session(conversation_id, session_id)
                    
                    # 对话已经在create_multi_agent_conversation中启动，WebSocket回调已设置
                    logger.info(f"多智能体对话已启动，WebSocket回调已就绪: {conversation_id}")
                    
                    # 发送创建成功消息到原始session_id，保持连接稳定
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "conversation_created",
                            "data": result,
                            "timestamp": utc_now().isoformat()
                        }),
                        session_id
                    )
                    
                    # 开始对话流程 - 发送到原始session_id
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "conversation_started", 
                            "data": {"session_id": conversation_id},
                            "timestamp": utc_now().isoformat()
                        }),
                        session_id
                    )
                    
                    logger.info(f"多智能体对话已启动: {result['conversation_id']}")
                    
                except Exception as e:
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "error",
                            "data": {"message": f"启动对话失败: {str(e)}"},
                            "timestamp": utc_now().isoformat()
                        }),
                        session_id
                    )
                    logger.error(f"启动对话失败: {e}")
            
            elif message_data.get("type") == "ping":
                # 心跳检测
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": utc_now().isoformat()
                    }),
                    session_id
                )
            
            else:
                # 未知消息类型
                logger.warning(f"收到未知消息类型: {message_data.get('type')}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "data": {"message": f"未知消息类型: {message_data.get('type')}"},
                        "timestamp": utc_now().isoformat()
                    }),
                    session_id
                )
                    
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket客户端主动断开连接: {session_id}, 代码: {e.code}")
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket处理异常: {session_id}, 错误: {str(e)}", exc_info=True)
        # 发送错误消息给客户端
        try:
            await manager.send_personal_message(
                json.dumps({
                    "type": "error",
                    "data": {"message": f"服务器错误: {str(e)}"},
                    "timestamp": utc_now().isoformat()
                }),
                session_id
            )
        except Exception as send_error:
            logger.error(f"发送错误消息失败: {send_error}")
        manager.disconnect(websocket, session_id)
    finally:
        logger.info(f"WebSocket连接处理结束: {session_id}")
