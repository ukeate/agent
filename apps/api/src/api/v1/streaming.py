"""
流式处理API路由

提供SSE和WebSocket流式响应接口。
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from typing import Optional, Dict, Any
import uuid
from src.core.utils.timezone_utils import utc_now
from src.ai.streaming import StreamProcessor
from src.core.dependencies import get_current_user
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/streaming", tags=["streaming"])

# 延迟初始化流式处理器实例
stream_processor = None

def get_stream_processor():
    """获取流式处理器实例"""
    global stream_processor
    if stream_processor is None:
        stream_processor = StreamProcessor()
    return stream_processor

class StreamingRequest(ApiBaseModel):
    """流式处理请求"""
    agent_id: str
    message: str
    session_id: Optional[str] = None
    buffer_size: Optional[int] = None

class StreamingSessionResponse(ApiBaseModel):
    """流式处理响应"""
    session_id: str
    status: str
    message: str

@router.post("/start", response_model=StreamingSessionResponse)
async def start_streaming_session(
    request: StreamingRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    启动流式处理会话
    
    创建新的流式处理会话，用于后续的流式通信。
    """
    try:
        # 生成会话ID（如果未提供）
        session_id = request.session_id or str(uuid.uuid4())
        
        # 创建流式处理会话
        session = await get_stream_processor().create_session(
            session_id=session_id,
            agent_id=request.agent_id,
            buffer_size=request.buffer_size
        )
        
        logger.info(f"启动流式会话: {session_id} (用户: {current_user})")
        
        return StreamingSessionResponse(
            session_id=session_id,
            status="created",
            message="流式处理会话已创建"
        )
        
    except Exception as e:
        logger.error(f"创建流式会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sse/{session_id}")
async def stream_sse(
    session_id: str,
    message: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    SSE流式端点
    
    提供Server-Sent Events流式响应。
    """
    try:
        # 检查会话是否存在
        session = await get_stream_processor().get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        logger.info(f"启动SSE流式响应: {session_id} (用户: {current_user})")
        
        # 获取响应处理器
        response_handler = get_stream_processor().get_response_handler()
        
        # 设置Token流处理器
        token_streamer = get_stream_processor().token_stream_manager.get_streamer(session_id)
        response_handler.token_streamer = token_streamer
        
        # 返回SSE流式响应
        return await response_handler.handle_sse(
            agent_id=session.agent_id,
            message=message,
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SSE流式处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket流式端点
    
    提供双向WebSocket流式通信。
    """
    try:
        # 检查会话是否存在
        session = await get_stream_processor().get_session(session_id)
        if not session:
            await websocket.close(code=4004, reason="会话不存在")
            return
        
        logger.info(f"启动WebSocket连接: {session_id}")
        
        # 获取响应处理器
        response_handler = get_stream_processor().get_response_handler()
        
        # 设置Token流处理器
        token_streamer = get_stream_processor().token_stream_manager.get_streamer(session_id)
        response_handler.token_streamer = token_streamer
        
        # 处理WebSocket连接
        await response_handler.handle_websocket(websocket, session_id)
        
    except Exception as e:
        logger.error(f"WebSocket处理失败: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            logger.exception("关闭WebSocket失败", exc_info=True)

@router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(
    session_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    获取会话指标
    
    返回指定会话的详细性能指标。
    """
    try:
        metrics = await get_stream_processor().get_session_metrics(session_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        
        return {
            "session_metrics": metrics,
            "timestamp": utc_now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_sessions(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    列出所有活跃会话
    
    返回所有活跃流式处理会话的信息。
    """
    try:
        all_metrics = await get_stream_processor().get_all_session_metrics()
        
        return {
            "sessions": all_metrics,
            "total_sessions": len(all_metrics),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_system_metrics(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    获取系统级流式处理指标
    
    返回整个流式处理系统的性能指标。
    """
    try:
        system_metrics = await get_stream_processor().get_system_metrics()
        
        return {
            "system_metrics": system_metrics,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backpressure/status")
async def get_backpressure_status(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    获取背压控制状态
    
    返回当前的背压管理状态和限流信息。
    """
    try:
        backpressure_status = get_stream_processor().get_backpressure_status()
        
        if backpressure_status is None:
            return {
                "backpressure_enabled": False,
                "message": "背压控制未启用",
                "timestamp": utc_now().isoformat()
            }
        
        return {
            "backpressure_enabled": True,
            "backpressure_status": backpressure_status,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取背压状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/flow-control/metrics")
async def get_flow_control_metrics(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    获取流量控制指标
    
    返回速率限制、熔断器和队列监控等流量控制组件的详细指标。
    """
    try:
        flow_metrics = get_stream_processor().get_flow_control_metrics()
        
        return {
            "flow_control_metrics": flow_metrics,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取流量控制指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/status")
async def get_queue_status(
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    获取队列状态
    
    返回所有监控队列的状态和指标。
    """
    try:
        from src.ai.streaming.queue_monitor import queue_monitor_manager
        
        queue_metrics = queue_monitor_manager.get_all_metrics()
        system_summary = queue_monitor_manager.get_system_summary()
        overloaded_queues = queue_monitor_manager.get_overloaded_queues()
        
        return {
            "queue_metrics": {
                name: {
                    "name": metrics.name,
                    "current_size": metrics.current_size,
                    "max_size": metrics.max_size,
                    "utilization": metrics.utilization,
                    "enqueue_rate": metrics.enqueue_rate,
                    "dequeue_rate": metrics.dequeue_rate,
                    "average_wait_time": metrics.average_wait_time,
                    "oldest_item_age": metrics.oldest_item_age,
                    "is_overloaded": metrics.is_overloaded,
                    "throughput_ratio": metrics.throughput_ratio
                }
                for name, metrics in queue_metrics.items()
            },
            "system_summary": system_summary,
            "overloaded_queues": overloaded_queues,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取队列状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def stop_session(
    session_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    停止流式处理会话
    
    停止并清理指定的流式处理会话。
    """
    try:
        session = await get_stream_processor().get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        # 停止流式处理
        await get_stream_processor().stop_streaming(session_id)
        
        # 移除会话
        await get_stream_processor().remove_session(session_id)
        
        logger.info(f"停止流式会话: {session_id} (用户: {current_user})")
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "message": "流式处理会话已停止"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/cleanup")
async def cleanup_session(
    session_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    清理会话资源
    
    手动清理指定会话的所有相关资源。
    """
    try:
        await get_stream_processor().remove_session(session_id)
        
        logger.info(f"清理流式会话: {session_id} (用户: {current_user})")
        
        return {
            "session_id": session_id,
            "status": "cleaned",
            "message": "会话资源已清理"
        }
        
    except Exception as e:
        logger.error(f"清理会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查端点
@router.get("/health")
async def health_check():
    """
    流式处理服务健康检查
    
    返回服务状态和基本指标。
    """
    try:
        system_metrics = await get_stream_processor().get_system_metrics()
        
        return {
            "status": "healthy",
            "service": "streaming",
            "active_sessions": system_metrics.get("active_sessions", 0),
            "total_sessions": system_metrics.get("total_sessions_created", 0),
            "uptime": system_metrics.get("uptime", 0),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": utc_now().isoformat()
        }

# 应用启动时初始化
async def initialize_streaming():
    """初始化流式处理服务"""
    logger.info("初始化流式处理服务...")

# 应用关闭时清理资源
async def shutdown_streaming():
    """关闭流式处理服务"""
    logger.info("关闭流式处理服务...")
    await get_stream_processor().shutdown()
