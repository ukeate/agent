"""
异步多智能体系统API路由
集成AutoGen v0.7.x异步事件驱动架构
"""

import asyncio
import json
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from pydantic import Field, ValidationError
from src.ai.autogen.async_manager import AsyncAgentManager, AgentTask, TaskStatus
from src.ai.autogen.events import (
    EventBus, MessageQueue, StateManager, Event, EventType, EventPriority,
    LoggingEventHandler, MetricsEventHandler
)
from src.api.base_model import ApiBaseModel
from src.ai.autogen.langgraph_bridge import AutoGenLangGraphBridge
from src.ai.autogen.config import AgentConfig, AgentRole, AGENT_CONFIGS
from src.ai.langgraph.context import AgentContext, SessionContext

from src.core.logging import get_logger
logger = get_logger(__name__)

fastapi_logger = get_logger(__name__)

router = APIRouter(prefix="/async-agents", tags=["async-agents"])

# 全局组件实例
_event_bus: Optional[EventBus] = None
_message_queue: Optional[MessageQueue] = None
_state_manager: Optional[StateManager] = None
_agent_manager: Optional[AsyncAgentManager] = None
_langgraph_bridge: Optional[AutoGenLangGraphBridge] = None

async def get_event_bus() -> EventBus:
    """获取事件总线实例"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus(max_queue_size=10000)
        
        # 注册默认事件处理器
        logging_handler = LoggingEventHandler()
        metrics_handler = MetricsEventHandler()
        
        _event_bus.subscribe_all(logging_handler)
        _event_bus.subscribe_all(metrics_handler)
        
        # 启动事件总线
        await _event_bus.start(worker_count=3)
        
        logger.info("事件总线初始化完成")
    
    return _event_bus

async def get_message_queue() -> MessageQueue:
    """获取消息队列实例"""
    global _message_queue
    if _message_queue is None:
        _message_queue = MessageQueue()
        logger.info("消息队列初始化完成")
    
    return _message_queue

async def get_state_manager() -> StateManager:
    """获取状态管理器实例"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
        logger.info("状态管理器初始化完成")
    
    return _state_manager

async def get_agent_manager() -> AsyncAgentManager:
    """获取异步智能体管理器实例"""
    global _agent_manager
    if _agent_manager is None:
        event_bus = await get_event_bus()
        message_queue = await get_message_queue()
        state_manager = await get_state_manager()
        
        _agent_manager = AsyncAgentManager(
            event_bus=event_bus,
            message_queue=message_queue,
            state_manager=state_manager,
            max_concurrent_tasks=10
        )
        
        # 启动管理器
        await _agent_manager.start()
        
        logger.info("异步智能体管理器初始化完成")
    
    return _agent_manager

async def get_langgraph_bridge() -> AutoGenLangGraphBridge:
    """获取LangGraph桥接器实例"""
    global _langgraph_bridge
    if _langgraph_bridge is None:
        agent_manager = await get_agent_manager()
        event_bus = await get_event_bus()
        
        _langgraph_bridge = AutoGenLangGraphBridge(
            agent_manager=agent_manager,
            event_bus=event_bus
        )
        
        logger.info("LangGraph桥接器初始化完成")
    
    return _langgraph_bridge

# Pydantic模型
class CreateAgentRequest(ApiBaseModel):
    """创建智能体请求"""
    role: AgentRole = Field(..., description="智能体角色")
    name: Optional[str] = Field(None, description="自定义名称")
    model: Optional[str] = Field("gpt-4o-mini", description="使用的模型")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="温度参数")
    max_tokens: Optional[int] = Field(2000, gt=0, description="最大token数")
    custom_prompt: Optional[str] = Field(None, description="自定义系统提示词")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")

class SubmitTaskRequest(ApiBaseModel):
    """提交任务请求"""
    agent_id: str = Field(..., description="目标智能体ID")
    task_type: str = Field(..., description="任务类型")
    description: str = Field(..., description="任务描述", min_length=1, max_length=5000)
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    priority: int = Field(0, description="任务优先级")
    timeout_seconds: int = Field(300, ge=30, le=1800, description="超时时间(秒)")

class CreateWorkflowRequest(ApiBaseModel):
    """创建工作流请求"""
    name: str = Field(..., description="工作流名称")
    description: str = Field(..., description="工作流描述")
    agents: List[Dict[str, Any]] = Field(..., description="智能体配置列表")
    tasks: List[Dict[str, Any]] = Field(..., description="任务配置列表")
    dependencies: Optional[Dict[str, List[str]]] = Field(None, description="任务依赖关系")
    context: Optional[Dict[str, Any]] = Field(None, description="工作流上下文")

class AgentResponse(ApiBaseModel):
    """智能体响应"""
    agent_id: str
    name: str
    role: str
    status: str
    created_at: str
    config: Dict[str, Any]

class TaskResponse(ApiBaseModel):
    """任务响应"""
    task_id: str
    agent_id: str
    task_type: str
    description: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]

# 智能体管理端点
@router.post(
    "/agents",
    response_model=AgentResponse,
    summary="创建智能体",
    description="创建新的异步智能体"
)
async def create_agent(
    request: CreateAgentRequest,
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> AgentResponse:
    """创建异步智能体"""
    try:
        # 构建智能体配置
        base_config = AGENT_CONFIGS.get(request.role)
        if base_config is None:
            base_config = AgentConfig(
                name=request.name or request.role.value,
                role=request.role,
                model=request.model or "gpt-4o-mini",
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 2000,
                system_prompt=request.custom_prompt or "你是一个通用智能体。",
                tools=[],
                capabilities=[],
            )
        
        config = AgentConfig(
            name=request.name or base_config.name,
            role=request.role,
            model=request.model or base_config.model,
            temperature=request.temperature or base_config.temperature,
            max_tokens=request.max_tokens or base_config.max_tokens,
            system_prompt=request.custom_prompt or base_config.system_prompt,
            tools=base_config.tools,
            capabilities=base_config.capabilities
        )
        
        session_id = str(uuid.uuid4())
        # 创建上下文
        context = AgentContext(
            user_id=request.context.get("user_id", "system") if request.context else "system",
            session_id=session_id,
            conversation_id=request.context.get("conversation_id") if request.context else None,
            session_context=SessionContext(session_id=session_id),
            metadata=request.context or {},
        )
        
        # 通过桥接器创建上下文智能体
        agent_id = await bridge.create_contextual_agent(config, context)
        
        # 获取智能体信息
        agent_info = await bridge.agent_manager.get_agent_info(agent_id)
        
        logger.info("异步智能体创建成功", agent_id=agent_id, role=request.role)
        
        return AgentResponse(
            agent_id=agent_id,
            name=agent_info["name"],
            role=agent_info["role"],
            status=agent_info["status"],
            created_at=agent_info["created_at"],
            config={
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "capabilities": config.capabilities
            }
        )
        
    except Exception as e:
        logger.error("创建智能体失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建智能体失败: {str(e)}"
        )

@router.get(
    "/agents",
    summary="列出智能体",
    description="获取所有智能体列表"
)
async def list_agents(
    manager: AsyncAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """列出所有智能体"""
    try:
        agents = await manager.list_agents()
        
        return {
            "success": True,
            "data": {
                "agents": agents,
                "total": len(agents)
            }
        }
        
    except Exception as e:
        logger.error("获取智能体列表失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取智能体列表失败: {str(e)}"
        )

@router.get(
    "/agents/{agent_id}",
    summary="获取智能体信息",
    description="获取指定智能体的详细信息"
)
async def get_agent(
    agent_id: str,
    manager: AsyncAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """获取智能体信息"""
    try:
        agent_info = await manager.get_agent_info(agent_id)
        if not agent_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"智能体不存在: {agent_id}"
            )
        
        return {
            "success": True,
            "data": agent_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取智能体信息失败", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取智能体信息失败: {str(e)}"
        )

@router.delete(
    "/agents/{agent_id}",
    summary="销毁智能体",
    description="销毁指定的智能体"
)
async def destroy_agent(
    agent_id: str,
    manager: AsyncAgentManager = Depends(get_agent_manager),
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> Dict[str, Any]:
    """销毁智能体"""
    try:
        # 清理上下文
        await bridge.cleanup_agent_context(agent_id)
        
        # 销毁智能体
        success = await manager.destroy_agent(agent_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"智能体不存在: {agent_id}"
            )
        
        return {
            "success": True,
            "message": f"智能体 {agent_id} 已销毁"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("销毁智能体失败", agent_id=agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"销毁智能体失败: {str(e)}"
        )

# 任务管理端点
@router.post(
    "/tasks",
    response_model=TaskResponse,
    summary="提交任务",
    description="向智能体提交新任务"
)
async def submit_task(
    request: SubmitTaskRequest,
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> TaskResponse:
    """提交任务给智能体"""
    try:
        task_id = await bridge.execute_contextual_task(
            agent_id=request.agent_id,
            task_type=request.task_type,
            description=request.description,
            input_data=request.input_data,
            priority=request.priority
        )
        
        # 获取任务信息
        task_info = await bridge.agent_manager.get_task_info(task_id)
        
        logger.info("任务提交成功", task_id=task_id, agent_id=request.agent_id)
        
        return TaskResponse(**task_info)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("任务提交失败", agent_id=request.agent_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"任务提交失败: {str(e)}"
        )

@router.get(
    "/tasks",
    summary="列出任务",
    description="获取任务列表"
)
async def list_tasks(
    agent_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    manager: AsyncAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """列出任务"""
    try:
        # 解析状态过滤器
        task_status = None
        if status_filter:
            try:
                task_status = TaskStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"无效的任务状态: {status_filter}"
                )
        
        tasks = await manager.list_tasks(agent_id=agent_id, status=task_status)
        
        # 应用分页
        total = len(tasks)
        paginated_tasks = tasks[offset:offset + limit]
        
        return {
            "success": True,
            "data": {
                "tasks": paginated_tasks,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取任务列表失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务列表失败: {str(e)}"
        )

@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    summary="获取任务信息",
    description="获取指定任务的详细信息"
)
async def get_task(
    task_id: str,
    manager: AsyncAgentManager = Depends(get_agent_manager),
) -> TaskResponse:
    """获取任务信息"""
    try:
        task_info = await manager.get_task_info(task_id)
        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {task_id}"
            )
        
        return TaskResponse(**task_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取任务信息失败", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取任务信息失败: {str(e)}"
        )

@router.delete(
    "/tasks/{task_id}",
    summary="取消任务",
    description="取消指定的任务"
)
async def cancel_task(
    task_id: str,
    manager: AsyncAgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """取消任务"""
    try:
        success = await manager.cancel_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务不存在: {task_id}"
            )
        
        return {
            "success": True,
            "message": f"任务 {task_id} 已取消"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("取消任务失败", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"取消任务失败: {str(e)}"
        )

# 工作流管理端点
@router.post(
    "/workflows",
    summary="创建工作流",
    description="创建多智能体协作工作流"
)
async def create_workflow(
    request: CreateWorkflowRequest,
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> Dict[str, Any]:
    """创建多智能体工作流"""
    try:
        # 创建工作流上下文
        session_id = str(uuid.uuid4())
        context = AgentContext(
            user_id=request.context.get("user_id", "system") if request.context else "system",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id),
            additional_context=request.context
        )
        
        # 构建工作流配置
        workflow_config = {
            "name": request.name,
            "description": request.description,
            "agents": request.agents,
            "tasks": request.tasks,
            "dependencies": request.dependencies or {}
        }
        
        # 创建工作流
        workflow_id = await bridge.create_agent_workflow(workflow_config, context)
        
        logger.info("工作流创建成功", workflow_id=workflow_id)
        
        return {
            "success": True,
            "data": {
                "workflow_id": workflow_id,
                "name": request.name,
                "description": request.description,
                "created_at": utc_now().isoformat()
            }
        }
    except HTTPException:
        raise
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("创建工作流失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建工作流失败: {str(e)}"
        )

@router.post(
    "/collaborative-tasks",
    summary="执行协作任务",
    description="在多个智能体间执行协作任务"
)
async def execute_collaborative_task(
    agent_ids: List[str],
    task_description: str,
    coordination_strategy: str = "sequential",
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> Dict[str, Any]:
    """执行协作任务"""
    try:
        if not agent_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="必须提供至少一个智能体ID"
            )
        
        if coordination_strategy not in ["sequential", "parallel"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="协调策略必须是 'sequential' 或 'parallel'"
            )
        
        task_ids = await bridge.execute_collaborative_task(
            agent_ids=agent_ids,
            task_description=task_description,
            coordination_strategy=coordination_strategy
        )
        
        logger.info(
            "协作任务执行成功",
            agent_count=len(agent_ids),
            task_count=len(task_ids),
            strategy=coordination_strategy
        )
        
        return {
            "success": True,
            "data": {
                "task_ids": task_ids,
                "agent_ids": agent_ids,
                "coordination_strategy": coordination_strategy,
                "task_description": task_description,
                "started_at": utc_now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("执行协作任务失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"执行协作任务失败: {str(e)}"
        )

# 系统状态和监控端点
@router.get(
    "/stats",
    summary="获取系统统计",
    description="获取异步智能体系统的统计信息"
)
async def get_system_stats(
    manager: AsyncAgentManager = Depends(get_agent_manager),
    event_bus: EventBus = Depends(get_event_bus),
    bridge: AutoGenLangGraphBridge = Depends(get_langgraph_bridge),
) -> Dict[str, Any]:
    """获取系统统计信息"""
    try:
        manager_stats = manager.get_manager_stats()
        event_stats = event_bus.get_stats()
        bridge_stats = bridge.get_bridge_stats()
        
        return {
            "success": True,
            "data": {
                "agent_manager": manager_stats,
                "event_bus": event_stats,
                "langgraph_bridge": bridge_stats,
                "timestamp": utc_now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error("获取系统统计失败", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统统计失败: {str(e)}"
        )

@router.get(
    "/health",
    summary="健康检查",
    description="检查异步智能体系统的健康状态"
)
async def health_check(
    manager: AsyncAgentManager = Depends(get_agent_manager),
    event_bus: EventBus = Depends(get_event_bus),
) -> Dict[str, Any]:
    """健康检查"""
    try:
        manager_stats = manager.get_manager_stats()
        event_stats = event_bus.get_stats()
        
        # 简单的健康检查逻辑
        healthy = True
        issues = []
        
        if not manager_stats["running"]:
            healthy = False
            issues.append("智能体管理器未运行")
        
        if not event_stats["running"]:
            healthy = False
            issues.append("事件总线未运行")
        
        if manager_stats["tasks"]["failed"] > manager_stats["tasks"]["completed"] * 0.1:
            issues.append("任务失败率过高")
        
        return {
            "healthy": healthy,
            "timestamp": utc_now().isoformat(),
            "issues": issues,
            "stats": {
                "agents": manager_stats["agents"],
                "tasks": manager_stats["tasks"],
                "events": {
                    "processed": event_stats["processed_events"],
                    "failed": event_stats["failed_events"],
                    "queue_size": event_stats["queue_size"]
                }
            }
        }
        
    except Exception as e:
        logger.error("健康检查失败", error=str(e))
        return {
            "healthy": False,
            "timestamp": utc_now().isoformat(),
            "error": str(e)
        }

# WebSocket连接管理
class AsyncConnectionManager:
    """异步WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.event_handlers: Dict[str, Callable] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        
        logger.info(
            "WebSocket连接建立",
            session_id=session_id,
            connection_count=len(self.active_connections[session_id])
        )
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].remove(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
                
                logger.info("WebSocket连接断开", session_id=session_id)
            except ValueError:
                logger.warning("捕获到ValueError，已继续执行", exc_info=True)
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """发送消息到指定会话"""
        if session_id in self.active_connections:
            message_text = json.dumps(message)
            connections = self.active_connections[session_id].copy()
            
            for websocket in connections:
                try:
                    await websocket.send_text(message_text)
                except Exception as e:
                    logger.error("发送WebSocket消息失败", error=str(e))
                    self.disconnect(websocket, session_id)

# 全局连接管理器
connection_manager = AsyncConnectionManager()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str
):
    """异步智能体WebSocket端点"""
    try:
        await connection_manager.connect(websocket, session_id)
        
        # 发送连接确认
        await connection_manager.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": utc_now().isoformat()
        })
        
        # 获取组件实例
        manager = await get_agent_manager()
        bridge = await get_langgraph_bridge()
        
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            message_data = message.get("data", {})
            
            if message_type == "create_agent":
                # 创建智能体
                try:
                    role = AgentRole(message_data.get("role"))
                    config = AGENT_CONFIGS[role]
                    
                    context = AgentContext(
                        user_id=message_data.get("user_id", "websocket_user"),
                        session_id=session_id
                    )
                    
                    agent_id = await bridge.create_contextual_agent(config, context)
                    
                    await connection_manager.send_message(session_id, {
                        "type": "agent_created",
                        "data": {"agent_id": agent_id, "role": role.value},
                        "timestamp": utc_now().isoformat()
                    })
                    
                except Exception as e:
                    await connection_manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": f"创建智能体失败: {str(e)}"},
                        "timestamp": utc_now().isoformat()
                    })
            
            elif message_type == "submit_task":
                # 提交任务
                try:
                    task_id = await bridge.execute_contextual_task(
                        agent_id=message_data.get("agent_id"),
                        task_type=message_data.get("task_type", "general"),
                        description=message_data.get("description", ""),
                        input_data=message_data.get("input_data", {}),
                        priority=message_data.get("priority", 0)
                    )
                    
                    await connection_manager.send_message(session_id, {
                        "type": "task_submitted",
                        "data": {"task_id": task_id},
                        "timestamp": utc_now().isoformat()
                    })
                    
                except Exception as e:
                    await connection_manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": f"提交任务失败: {str(e)}"},
                        "timestamp": utc_now().isoformat()
                    })
            
            elif message_type == "ping":
                # 心跳检测
                await connection_manager.send_message(session_id, {
                    "type": "pong",
                    "timestamp": utc_now().isoformat()
                })
            
            else:
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "data": {"message": f"未知消息类型: {message_type}"},
                    "timestamp": utc_now().isoformat()
                })
    
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, session_id)
        logger.info("WebSocket客户端断开连接", session_id=session_id)
    
    except Exception as e:
        logger.error("WebSocket处理异常", session_id=session_id, error=str(e))
        connection_manager.disconnect(websocket, session_id)

# 清理函数（在应用关闭时调用）
async def cleanup():
    """清理资源"""
    global _agent_manager, _event_bus
    
    try:
        if _agent_manager:
            await _agent_manager.stop()
        
        if _event_bus:
            await _event_bus.stop()
        
        logger.info("异步智能体系统组件清理完成")
    
    except Exception as e:
        logger.error("清理异步智能体系统组件失败", error=str(e))
