"""
Token流式处理器

负责处理LLM响应的Token流输出，支持实时流式响应和事件广播。
"""

from typing import AsyncIterator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import time
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

class StreamType(str, Enum):
    """流式事件类型"""
    TOKEN = "token"
    PARTIAL = "partial"
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    METADATA = "metadata"

@dataclass
class StreamEvent:
    """流式事件数据结构"""
    type: StreamType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    sequence: int = 0

class TokenStreamer:
    """Token流式处理器"""
    
    def __init__(self, buffer_size: int = 100, heartbeat_interval: float = 30.0):
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.subscribers: List[asyncio.Queue] = []
        self.streaming = False
        self.heartbeat_interval = heartbeat_interval
        self._sequence_counter = 0
        self._session_metrics: Dict[str, Dict] = {}
        
    async def stream_tokens(
        self, 
        llm_response: AsyncIterator[str], 
        session_id: str = None
    ) -> AsyncIterator[StreamEvent]:
        """
        流式处理LLM响应
        
        Args:
            llm_response: LLM异步响应迭代器
            session_id: 会话ID
            
        Yields:
            StreamEvent: 流式事件
        """
        self.streaming = True
        partial_response = ""
        token_count = 0
        start_time = time.time()
        
        # 初始化会话指标
        if session_id:
            self._session_metrics[session_id] = {
                "start_time": start_time,
                "token_count": 0,
                "event_count": 0,
                "error_count": 0
            }
        
        try:
            # 发送开始事件
            start_event = self._create_event(
                StreamType.METADATA,
                {"status": "started", "session_id": session_id},
                session_id=session_id
            )
            await self._broadcast(start_event)
            yield start_event
            
            # 启动心跳任务
            heartbeat_task = asyncio.create_task(
                self._heartbeat_sender(session_id)
            ) if session_id else None
            
            # 流式处理Token
            async for token in llm_response:
                partial_response += token
                token_count += 1
                
                # 创建Token事件
                token_event = self._create_event(
                    StreamType.TOKEN,
                    token,
                    metadata={
                        "partial": partial_response,
                        "token_count": token_count,
                        "elapsed_time": time.time() - start_time
                    },
                    session_id=session_id
                )
                
                # 广播并生成事件
                await self._broadcast(token_event)
                yield token_event
                
                # 更新指标
                if session_id:
                    self._session_metrics[session_id]["token_count"] = token_count
                    self._session_metrics[session_id]["event_count"] += 1
            
            # 停止心跳
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    raise
            
            # 发送完成事件
            end_time = time.time()
            complete_event = self._create_event(
                StreamType.COMPLETE,
                partial_response,
                metadata={
                    "token_count": token_count,
                    "duration": end_time - start_time,
                    "tokens_per_second": token_count / (end_time - start_time) if end_time > start_time else 0,
                    "final_response": partial_response
                },
                session_id=session_id
            )
            await self._broadcast(complete_event)
            yield complete_event
            
        except Exception as e:
            logger.error(f"流式处理出错 (session: {session_id}): {e}")
            
            # 更新错误指标
            if session_id and session_id in self._session_metrics:
                self._session_metrics[session_id]["error_count"] += 1
            
            # 发送错误事件
            error_event = self._create_event(
                StreamType.ERROR,
                str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "partial_response": partial_response,
                    "token_count": token_count
                },
                session_id=session_id
            )
            await self._broadcast(error_event)
            yield error_event
            raise
        
        finally:
            self.streaming = False
            # 清理会话指标（可选，或者保留用于分析）
            if session_id and session_id in self._session_metrics:
                logger.info(f"会话 {session_id} 流式处理完成: {self._session_metrics[session_id]}")
    
    def _create_event(
        self,
        event_type: StreamType,
        data: Any,
        metadata: Dict[str, Any] = None,
        session_id: str = None
    ) -> StreamEvent:
        """创建流式事件"""
        self._sequence_counter += 1
        return StreamEvent(
            type=event_type,
            data=data,
            metadata=metadata or {},
            timestamp=time.time(),
            session_id=session_id,
            sequence=self._sequence_counter
        )
    
    async def _broadcast(self, event: StreamEvent):
        """广播事件给所有订阅者"""
        if not self.subscribers:
            return
            
        # 移除已关闭的订阅者
        active_subscribers = []
        
        for subscriber in self.subscribers:
            try:
                subscriber.put_nowait(event)
                active_subscribers.append(subscriber)
            except asyncio.QueueFull:
                logger.warning("订阅者队列已满，跳过事件广播")
            except Exception as e:
                logger.error(f"广播事件失败: {e}")
        
        self.subscribers = active_subscribers
    
    async def _heartbeat_sender(self, session_id: str):
        """发送心跳事件"""
        try:
            while self.streaming:
                await asyncio.sleep(self.heartbeat_interval)
                if self.streaming:  # 再次检查状态
                    heartbeat_event = self._create_event(
                        StreamType.HEARTBEAT,
                        {"timestamp": utc_now().isoformat()},
                        session_id=session_id
                    )
                    await self._broadcast(heartbeat_event)
        except asyncio.CancelledError:
            raise
    
    async def subscribe(self, queue_size: int = 100) -> asyncio.Queue:
        """订阅流式事件"""
        queue = asyncio.Queue(maxsize=queue_size)
        self.subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        """取消订阅"""
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict]:
        """获取会话指标"""
        return self._session_metrics.get(session_id)
    
    def get_active_subscribers_count(self) -> int:
        """获取活跃订阅者数量"""
        return len(self.subscribers)
    
    def clear_session_metrics(self, session_id: str = None):
        """清理会话指标"""
        if session_id:
            self._session_metrics.pop(session_id, None)
        else:
            self._session_metrics.clear()

class TokenStreamManager:
    """Token流管理器，管理多个Token流处理器"""
    
    def __init__(self):
        self.streamers: Dict[str, TokenStreamer] = {}
        
    def get_streamer(self, session_id: str, buffer_size: int = 100) -> TokenStreamer:
        """获取或创建Token流处理器"""
        if session_id not in self.streamers:
            self.streamers[session_id] = TokenStreamer(buffer_size=buffer_size)
        return self.streamers[session_id]
    
    def remove_streamer(self, session_id: str):
        """移除Token流处理器"""
        self.streamers.pop(session_id, None)
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """获取所有会话指标"""
        metrics = {}
        for session_id, streamer in self.streamers.items():
            session_metrics = streamer.get_session_metrics(session_id)
            if session_metrics:
                metrics[session_id] = session_metrics
        return metrics
    
    def cleanup_inactive_streamers(self):
        """清理不活跃的流处理器"""
        inactive_sessions = [
            session_id for session_id, streamer in self.streamers.items()
            if not streamer.streaming and not streamer.subscribers
        ]
        
        for session_id in inactive_sessions:
            self.remove_streamer(session_id)
        
        logger.info(f"清理了 {len(inactive_sessions)} 个不活跃的流处理器")
