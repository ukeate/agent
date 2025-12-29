"""
流式响应处理器

处理SSE和WebSocket流式响应，提供统一的流式输出接口。
"""

from typing import AsyncIterator, Dict, Any, Optional, Callable
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from .token_streamer import TokenStreamer, StreamEvent, StreamType
from .stream_buffer import StreamBuffer
from src.ai.openai_client import get_openai_client

from src.core.logging import get_logger
logger = get_logger(__name__)

class StreamingResponseHandler:
    """流式响应处理器"""
    
    def __init__(
        self, 
        token_streamer: TokenStreamer,
        buffer_manager: Optional[Dict[str, StreamBuffer]] = None
    ):
        self.token_streamer = token_streamer
        self.buffer_manager = buffer_manager or {}
        self.active_connections: Dict[str, WebSocket] = {}
        self._message_processors: Dict[str, Callable] = {}
    
    def register_message_processor(self, message_type: str, processor: Callable):
        """注册消息处理器"""
        self._message_processors[message_type] = processor
    
    async def handle_sse(
        self, 
        agent_id: str, 
        message: str,
        session_id: Optional[str] = None
    ) -> StreamingResponse:
        """
        处理Server-Sent Events流式响应
        
        Args:
            agent_id: 智能体ID
            message: 用户消息
            session_id: 会话ID
            
        Returns:
            StreamingResponse: SSE流式响应
        """
        
        async def event_generator():
            # 创建订阅队列
            queue = await self.token_streamer.subscribe()
            
            try:
                # 启动智能体处理任务
                processing_task = asyncio.create_task(
                    self._process_agent_message(agent_id, message, session_id)
                )
                
                # 发送初始连接事件
                yield self._format_sse_event({
                    "type": "connection",
                    "data": {"status": "connected", "session_id": session_id},
                    "timestamp": utc_now().isoformat()
                })
                
                # 流式发送事件
                while True:
                    try:
                        # 等待事件，带超时
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                        
                        # 格式化并发送事件
                        if event.type == StreamType.COMPLETE:
                            yield self._format_sse_event({
                                "type": "complete",
                                "data": event.data,
                                "metadata": event.metadata,
                                "sequence": event.sequence
                            })
                            break
                        elif event.type == StreamType.TOKEN:
                            yield self._format_sse_event({
                                "type": "token",
                                "data": event.data,
                                "metadata": event.metadata,
                                "sequence": event.sequence
                            })
                        elif event.type == StreamType.PARTIAL:
                            yield self._format_sse_event({
                                "type": "partial",
                                "data": event.data,
                                "metadata": event.metadata,
                                "sequence": event.sequence
                            })
                        elif event.type == StreamType.ERROR:
                            yield self._format_sse_event({
                                "type": "error",
                                "data": event.data,
                                "metadata": event.metadata,
                                "sequence": event.sequence
                            })
                            break
                        elif event.type == StreamType.HEARTBEAT:
                            yield self._format_sse_event({
                                "type": "heartbeat",
                                "data": event.data,
                                "sequence": event.sequence
                            })
                            
                    except asyncio.TimeoutError:
                        # 发送心跳保持连接
                        yield self._format_sse_event({
                            "type": "heartbeat",
                            "data": {"timestamp": utc_now().isoformat()}
                        })
                        continue
                    
                # 等待处理任务完成
                await processing_task
                    
            except Exception as e:
                logger.error(f"SSE处理出错: {e}")
                yield self._format_sse_event({
                    "type": "error",
                    "data": {"message": str(e), "error_type": type(e).__name__}
                })
            finally:
                self.token_streamer.unsubscribe(queue)
                yield self._format_sse_event({
                    "type": "connection",
                    "data": {"status": "disconnected"}
                })
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def handle_websocket(
        self, 
        websocket: WebSocket, 
        session_id: str
    ):
        """
        处理WebSocket流式响应
        
        Args:
            websocket: WebSocket连接
            session_id: 会话ID
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # 创建订阅队列
        queue = await self.token_streamer.subscribe()
        
        try:
            # 发送连接确认
            await websocket.send_json({
                "type": "connection",
                "data": {"status": "connected", "session_id": session_id},
                "timestamp": utc_now().isoformat()
            })
            
            # 启动接收和发送任务
            receive_task = asyncio.create_task(
                self._handle_websocket_receive(websocket, session_id)
            )
            send_task = asyncio.create_task(
                self._handle_websocket_send(websocket, queue)
            )
            
            # 等待任一任务完成
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消剩余任务
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    raise
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket连接断开: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket处理出错: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": str(e), "error_type": type(e).__name__}
                })
            except Exception:
                logger.exception("发送WebSocket错误消息失败", exc_info=True)
        finally:
            # 清理资源
            self.active_connections.pop(session_id, None)
            self.token_streamer.unsubscribe(queue)
            try:
                await websocket.close()
            except Exception:
                logger.exception("关闭WebSocket失败", exc_info=True)
    
    async def _handle_websocket_receive(self, websocket: WebSocket, session_id: str):
        """处理WebSocket接收消息"""
        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()
                
                message_type = data.get("type")
                message_data = data.get("data", {})
                
                # 处理不同类型的消息
                if message_type == "chat":
                    agent_id = message_data.get("agent_id")
                    message = message_data.get("message")
                    
                    if agent_id and message:
                        # 启动智能体处理
                        asyncio.create_task(
                            self._process_agent_message(agent_id, message, session_id)
                        )
                elif message_type == "ping":
                    # 响应ping
                    await websocket.send_json({
                        "type": "pong",
                        "data": {"timestamp": utc_now().isoformat()}
                    })
                elif message_type in self._message_processors:
                    # 使用注册的处理器
                    processor = self._message_processors[message_type]
                    await processor(websocket, session_id, message_data)
                else:
                    logger.warning(f"未知的WebSocket消息类型: {message_type}")
        except WebSocketDisconnect:
            logger.info("WebSocket连接断开", session_id=session_id)
        except Exception as e:
            logger.error(f"WebSocket接收处理出错: {e}")
    
    async def _handle_websocket_send(self, websocket: WebSocket, queue: asyncio.Queue):
        """处理WebSocket发送消息"""
        try:
            while True:
                # 等待事件
                event = await queue.get()
                
                # 发送事件
                await websocket.send_json({
                    "type": event.type.value,
                    "data": event.data,
                    "metadata": event.metadata,
                    "timestamp": event.timestamp,
                    "sequence": event.sequence,
                    "session_id": event.session_id
                })
                
                # 检查是否是结束事件
                if event.type in [StreamType.COMPLETE, StreamType.ERROR]:
                    break
        except WebSocketDisconnect:
            logger.info("WebSocket连接断开")
        except Exception as e:
            logger.error(f"WebSocket发送处理出错: {e}")
    
    async def _process_agent_message(
        self, 
        agent_id: str, 
        message: str,
        session_id: Optional[str] = None
    ):
        """
        处理智能体消息
        
        通过OpenAI流式接口生成回复，并写入TokenStreamer。
        """
        if not self.token_streamer:
            raise RuntimeError("token_streamer 未设置")

        async def token_iter():
            client = await get_openai_client()
            messages = client.format_messages_for_openai(
                system_prompt=f"你是一个名为{agent_id}的智能助手。",
                user_message=message,
            )
            async for chunk in client.create_streaming_completion(messages=messages, temperature=0.7, max_tokens=1000):
                if chunk.get("content"):
                    yield chunk["content"]

        try:
            async for _ in self.token_streamer.stream_tokens(token_iter(), session_id=session_id):
                continue
        except Exception as e:
            logger.error(f"智能体流式处理失败: {e}")
    
    def _format_sse_event(self, data: Dict[str, Any]) -> str:
        """格式化SSE事件"""
        json_data = json.dumps(data, ensure_ascii=False)
        return f"data: {json_data}\n\n"
    
    async def broadcast_to_session(
        self, 
        session_id: str, 
        event_type: str, 
        data: Any
    ):
        """向特定会话广播消息"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json({
                    "type": event_type,
                    "data": data,
                    "timestamp": utc_now().isoformat()
                })
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                # 移除失效连接
                self.active_connections.pop(session_id, None)
    
    async def get_active_connections(self) -> Dict[str, str]:
        """获取活跃连接信息"""
        return {
            session_id: "websocket" 
            for session_id in self.active_connections.keys()
        }
    
    async def close_connection(self, session_id: str):
        """关闭指定会话的连接"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.close()
            except Exception:
                logger.exception("关闭WebSocket失败", exc_info=True)
            finally:
                self.active_connections.pop(session_id, None)
    
    def get_connection_count(self) -> int:
        """获取连接数量"""
        return len(self.active_connections)
