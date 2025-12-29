"""
流式处理引擎

提供统一的流式处理接口，管理Token流、缓冲区和状态追踪。
"""

from typing import AsyncIterator, Dict, Any, Optional, List, Callable
import asyncio
import time
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
from enum import Enum
from .token_streamer import TokenStreamer, TokenStreamManager, StreamEvent, StreamType
from .stream_buffer import StreamBuffer, MultiStreamBuffer, BufferMetrics
from .response_handler import StreamingResponseHandler
from .backpressure import BackpressureManager, RateLimiter, CircuitBreaker
from .queue_monitor import QueueMonitor, queue_monitor_manager

from src.core.logging import get_logger
logger = get_logger(__name__)

class ProcessingStatus(str, Enum):
    """处理状态"""
    IDLE = "idle"
    STARTING = "starting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class StreamSession:
    """流式处理会话"""
    session_id: str
    agent_id: str
    status: ProcessingStatus = ProcessingStatus.IDLE
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    token_count: int = 0
    event_count: int = 0
    error_count: int = 0
    last_activity: float = 0

class StreamProcessor:
    """流式处理引擎"""
    
    def __init__(
        self,
        default_buffer_size: int = 1000,
        session_timeout: float = 3600.0,  # 1小时会话超时
        cleanup_interval: float = 300.0,   # 5分钟清理间隔
        max_concurrent_sessions: int = 100,
        enable_backpressure: bool = True
    ):
        self.token_stream_manager = TokenStreamManager()
        self.buffer_manager = MultiStreamBuffer(default_max_size=default_buffer_size)
        self.response_handler = StreamingResponseHandler(
            token_streamer=None  # 将在需要时动态设置
        )
        
        self.sessions: Dict[str, StreamSession] = {}
        self.session_timeout = session_timeout
        self.cleanup_interval = cleanup_interval
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # 背压和流量控制
        self.enable_backpressure = enable_backpressure
        self.backpressure_manager = None
        self.rate_limiter = None
        self.circuit_breaker = None
        self.queue_monitor = None
        
        if enable_backpressure:
            self._init_flow_control()
        
        # 性能指标
        self._total_sessions = 0
        self._total_tokens = 0
        self._total_events = 0
        
        # 启动清理任务
        self._cleanup_task = None
        self._cleanup_started = False
    
    def _init_flow_control(self):
        """初始化流量控制组件"""
        # 背压管理器
        self.backpressure_manager = BackpressureManager(
            max_buffer_size=self.max_concurrent_sessions,
            high_watermark=0.8,
            critical_watermark=0.95
        )
        
        # 速率限制器：每秒最多100个请求
        self.rate_limiter = RateLimiter(rate=100, per=1.0, burst=150)
        
        # 熔断器：5次失败后熔断，60秒恢复
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # 注册队列监控
        self.queue_monitor = queue_monitor_manager.register_queue(
            name="stream_sessions",
            max_size=self.max_concurrent_sessions,
            monitoring_window=60.0
        )
        
        # 设置背压回调
        self.backpressure_manager.add_throttle_callback(self._on_throttle_applied)
        self.backpressure_manager.add_release_callback(self._on_throttle_released)
        
        logger.info("流量控制组件已初始化")
    
    async def _on_throttle_applied(self, level):
        """背压限流应用回调"""
        logger.warning(f"流式处理应用背压限流: {level.value}")
        # 可以在这里实现具体的限流逻辑
    
    async def _on_throttle_released(self):
        """背压限流释放回调"""
        logger.info("流式处理背压限流已释放")
    
    async def create_session(
        self, 
        session_id: str, 
        agent_id: str,
        buffer_size: Optional[int] = None
    ) -> StreamSession:
        """
        创建流式处理会话
        
        Args:
            session_id: 会话ID
            agent_id: 智能体ID
            buffer_size: 缓冲区大小
            
        Returns:
            StreamSession: 创建的会话
        """
        # 首次使用时启动清理任务
        if not self._cleanup_started:
            await self._ensure_cleanup_task()
            self._cleanup_started = True
        
        if session_id in self.sessions:
            logger.warning(f"会话已存在: {session_id}")
            return self.sessions[session_id]
        
        # 背压检查
        if self.enable_backpressure:
            # 检查速率限制
            if not await self.rate_limiter.acquire():
                raise Exception("请求频率过高，请稍后重试")
            
            # 检查会话数量限制
            if len(self.sessions) >= self.max_concurrent_sessions:
                raise Exception("并发会话数已达上限，请稍后重试")
            
            # 更新背压管理器的缓冲区使用情况
            self.backpressure_manager.update_buffer_usage(len(self.sessions))
        
        # 创建会话
        session = StreamSession(
            session_id=session_id,
            agent_id=agent_id,
            last_activity=time.time()
        )
        
        # 创建Token流处理器
        streamer = self.token_stream_manager.get_streamer(
            session_id, 
            buffer_size or 100
        )
        
        # 创建缓冲区
        await self.buffer_manager.get_buffer(session_id, buffer_size)
        
        self.sessions[session_id] = session
        self._total_sessions += 1
        
        # 记录队列操作
        if self.queue_monitor:
            self.queue_monitor.record_enqueue(session_id)
        
        logger.info(f"创建流式处理会话: {session_id} (智能体: {agent_id})")
        return session
    
    async def start_streaming(
        self,
        session_id: str,
        llm_response: AsyncIterator[str],
        on_event: Optional[Callable] = None
    ) -> AsyncIterator[StreamEvent]:
        """
        启动流式处理
        
        Args:
            session_id: 会话ID
            llm_response: LLM异步响应迭代器
            on_event: 事件回调函数
            
        Yields:
            StreamEvent: 流式事件
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        # 更新会话状态
        session.status = ProcessingStatus.STARTING
        session.start_time = time.time()
        session.last_activity = time.time()
        
        # 获取Token流处理器
        streamer = self.token_stream_manager.get_streamer(session_id)
        
        try:
            session.status = ProcessingStatus.PROCESSING
            
            # 流式处理
            async for event in streamer.stream_tokens(llm_response, session_id):
                # 更新会话统计
                session.token_count += 1 if event.type == StreamType.TOKEN else 0
                session.event_count += 1
                session.last_activity = time.time()
                
                # 更新全局统计
                if event.type == StreamType.TOKEN:
                    self._total_tokens += 1
                self._total_events += 1
                
                # 执行事件回调
                if on_event:
                    try:
                        if asyncio.iscoroutinefunction(on_event):
                            await on_event(event)
                        else:
                            on_event(event)
                    except Exception as e:
                        logger.error(f"事件回调执行失败: {e}")
                
                yield event
                
                # 检查是否为结束事件
                if event.type in [StreamType.COMPLETE, StreamType.ERROR]:
                    break
            
            # 更新完成状态
            session.status = ProcessingStatus.COMPLETED
            session.end_time = time.time()
            
        except Exception as e:
            logger.error(f"流式处理失败 (会话: {session_id}): {e}")
            session.status = ProcessingStatus.ERROR
            session.error_count += 1
            session.end_time = time.time()
            raise
    
    async def stop_streaming(self, session_id: str):
        """停止流式处理"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        session.status = ProcessingStatus.CANCELLED
        session.end_time = time.time()
        
        logger.info(f"停止流式处理: {session_id}")
    
    async def get_session(self, session_id: str) -> Optional[StreamSession]:
        """获取会话信息"""
        return self.sessions.get(session_id)
    
    async def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话指标"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # 获取Token流指标
        streamer = self.token_stream_manager.streamers.get(session_id)
        stream_metrics = streamer.get_session_metrics(session_id) if streamer else {}
        
        # 获取缓冲区指标
        buffer_metrics = None
        if session_id in self.buffer_manager.buffers:
            buffer = self.buffer_manager.buffers[session_id]
            buffer_metrics = buffer.get_metrics()
        
        # 计算处理时长
        duration = None
        if session.start_time:
            end_time = session.end_time or time.time()
            duration = end_time - session.start_time
        
        # 计算处理速率
        tokens_per_second = 0
        if duration and duration > 0:
            tokens_per_second = session.token_count / duration
        
        return {
            "session_id": session_id,
            "agent_id": session.agent_id,
            "status": session.status.value,
            "duration": duration,
            "token_count": session.token_count,
            "event_count": session.event_count,
            "error_count": session.error_count,
            "tokens_per_second": tokens_per_second,
            "last_activity": session.last_activity,
            "stream_metrics": stream_metrics,
            "buffer_metrics": buffer_metrics.__dict__ if buffer_metrics else None
        }
    
    async def get_all_session_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有会话指标"""
        metrics = {}
        for session_id in self.sessions.keys():
            session_metrics = await self.get_session_metrics(session_id)
            if session_metrics:
                metrics[session_id] = session_metrics
        return metrics
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统级指标"""
        active_sessions = len([
            s for s in self.sessions.values() 
            if s.status in [ProcessingStatus.PROCESSING, ProcessingStatus.STARTING]
        ])
        
        total_buffer_usage = 0
        buffer_metrics = await self.buffer_manager.get_all_metrics()
        for metrics in buffer_metrics.values():
            total_buffer_usage += metrics.current_size
        
        return {
            "total_sessions": self._total_sessions,
            "active_sessions": active_sessions,
            "total_sessions_created": len(self.sessions),
            "total_tokens_processed": self._total_tokens,
            "total_events_processed": self._total_events,
            "total_buffer_usage": total_buffer_usage,
            "active_streamers": len(self.token_stream_manager.streamers),
            "active_buffers": len(self.buffer_manager.buffers),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            # 检查会话是否过期
            if (current_time - session.last_activity) > self.session_timeout:
                expired_sessions.append(session_id)
        
        # 清理过期会话
        for session_id in expired_sessions:
            await self.remove_session(session_id)
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    
    async def remove_session(self, session_id: str):
        """移除会话及相关资源"""
        # 移除会话记录
        session = self.sessions.pop(session_id, None)
        if session:
            # 计算处理时间
            processing_time = None
            if session.start_time and session.end_time:
                processing_time = session.end_time - session.start_time
            
            # 记录队列操作
            if self.queue_monitor:
                self.queue_monitor.record_dequeue(session_id, processing_time)
            
            # 更新背压管理器
            if self.backpressure_manager:
                self.backpressure_manager.update_buffer_usage(len(self.sessions))
            
            logger.info(f"移除会话: {session_id}")
        
        # 清理Token流处理器
        self.token_stream_manager.remove_streamer(session_id)
        
        # 清理缓冲区
        await self.buffer_manager.remove_buffer(session_id)
    
    async def _ensure_cleanup_task(self):
        """确保清理任务已启动"""
        if self._cleanup_task is None:
            await self._start_cleanup_task()
    
    async def _start_cleanup_task(self):
        """启动清理任务"""
        self._start_time = time.time()
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_expired_sessions()
                    await self.buffer_manager.cleanup_empty_buffers()
                    self.token_stream_manager.cleanup_inactive_streamers()
                except Exception as e:
                    logger.error(f"清理任务出错: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        
        # 启动背压监控
        if self.enable_backpressure and self.backpressure_manager:
            asyncio.create_task(self.backpressure_manager.start_monitoring())
            if self.queue_monitor:
                asyncio.create_task(self.queue_monitor.start_monitoring())
    
    async def shutdown(self):
        """关闭流式处理引擎"""
        logger.info("关闭流式处理引擎...")
        
        # 停止背压监控
        if self.enable_backpressure:
            if self.backpressure_manager:
                await self.backpressure_manager.stop_monitoring()
            if self.queue_monitor:
                await self.queue_monitor.stop_monitoring()
        
        # 取消清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                raise
        
        # 清理所有会话
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self.remove_session(session_id)
        
        logger.info("流式处理引擎已关闭")
    
    def get_response_handler(self) -> StreamingResponseHandler:
        """获取响应处理器"""
        return self.response_handler
    
    def get_backpressure_status(self) -> Optional[Dict[str, Any]]:
        """获取背压状态"""
        if not self.enable_backpressure or not self.backpressure_manager:
            return None
        
        return self.backpressure_manager.get_current_status()
    
    def get_flow_control_metrics(self) -> Dict[str, Any]:
        """获取流量控制指标"""
        metrics = {
            "backpressure_enabled": self.enable_backpressure,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "current_sessions": len(self.sessions)
        }
        
        if self.enable_backpressure:
            if self.backpressure_manager:
                metrics["backpressure_status"] = self.backpressure_manager.get_current_status()
            
            if self.rate_limiter:
                metrics["rate_limiter_stats"] = self.rate_limiter.get_stats()
            
            if self.circuit_breaker:
                metrics["circuit_breaker_state"] = self.circuit_breaker.get_state()
            
            if self.queue_monitor:
                metrics["queue_metrics"] = self.queue_monitor.get_current_metrics().__dict__
        
        return metrics
