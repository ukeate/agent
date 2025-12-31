"""
流式处理容错机制

提供断线重连、错误恢复和连接状态管理功能。
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import json
import websockets
from contextlib import asynccontextmanager

from src.core.logging import get_logger
logger = get_logger(__name__)

class ConnectionState(str, Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"

class RetryStrategy(str, Enum):
    """重试策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"
    CUSTOM = "custom"

@dataclass
class ConnectionConfig:
    """连接配置"""
    max_retries: int = 5
    initial_retry_delay: float = 1.0  # 初始重试延迟（秒）
    max_retry_delay: float = 60.0  # 最大重试延迟（秒）
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    connection_timeout: float = 30.0
    heartbeat_interval: float = 30.0
    reconnect_on_error: bool = True
    preserve_session_state: bool = True
    max_buffer_size: int = 1000  # 消息缓冲区最大大小

@dataclass
class SessionState:
    """会话状态"""
    session_id: str
    agent_id: str
    last_message_id: Optional[str] = None
    last_token_position: int = 0
    connection_start_time: Optional[datetime] = None
    total_reconnections: int = 0
    message_buffer: List[Dict] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectionMetrics:
    """连接指标"""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_reconnections: int = 0
    avg_connection_duration: float = 0.0
    last_failure_reason: Optional[str] = None
    uptime_percentage: float = 100.0

class HeartbeatManager:
    """心跳管理器"""
    
    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self.last_heartbeat: Optional[float] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.missed_heartbeats = 0
        self.max_missed_heartbeats = 3
        
    async def start(self, send_heartbeat: Callable):
        """启动心跳"""
        if self.is_running:
            return
        
        self.is_running = True
        self.heartbeat_task = create_task_with_logging(
            self._heartbeat_loop(send_heartbeat)
        )
        logger.debug("心跳管理器启动")
    
    async def stop(self):
        """停止心跳"""
        self.is_running = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                raise
        logger.debug("心跳管理器停止")
    
    async def _heartbeat_loop(self, send_heartbeat: Callable):
        """心跳循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.interval)
                if self.is_running:
                    await send_heartbeat()
                    self.last_heartbeat = time.time()
                    self.missed_heartbeats = 0
            except Exception as e:
                self.missed_heartbeats += 1
                logger.warning(f"心跳发送失败: {e}")
                
                if self.missed_heartbeats >= self.max_missed_heartbeats:
                    logger.error("连续心跳失败，连接可能已断开")
                    break
    
    def is_connection_alive(self) -> bool:
        """检查连接是否存活"""
        if not self.last_heartbeat:
            return True  # 初始状态
        
        time_since_last = time.time() - self.last_heartbeat
        return time_since_last < (self.interval * 2)

class FaultTolerantConnection:
    """容错连接管理器"""
    
    def __init__(self, session_id: str, config: ConnectionConfig = None):
        self.session_id = session_id
        self.config = config or ConnectionConfig()
        
        self.state = ConnectionState.DISCONNECTED
        self.session_state = SessionState(session_id=session_id, agent_id="")
        self.metrics = ConnectionMetrics()
        
        self.connection = None
        self.retry_count = 0
        self.last_connection_attempt: Optional[float] = None
        self.heartbeat_manager = HeartbeatManager(self.config.heartbeat_interval)
        
        # 保存连接参数以供重连使用
        self._connection_factory: Optional[Callable] = None
        self._connection_kwargs: Dict[str, Any] = {}
        
        # 事件回调
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_message: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_reconnect_attempt: Optional[Callable] = None
        
        # 消息缓冲
        self._message_queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        
    async def connect(self, connection_factory: Callable, **kwargs) -> bool:
        """建立连接"""
        if self.state == ConnectionState.CONNECTED:
            return True
        
        # 保存连接参数以供重连使用
        self._connection_factory = connection_factory
        self._connection_kwargs = kwargs.copy()
        
        self.state = ConnectionState.CONNECTING
        self.last_connection_attempt = time.time()
        
        try:
            # 创建连接
            self.connection = await asyncio.wait_for(
                connection_factory(**kwargs),
                timeout=self.config.connection_timeout
            )
            
            # 连接成功
            self.state = ConnectionState.CONNECTED
            self.session_state.connection_start_time = utc_now()
            self.retry_count = 0
            
            # 更新指标
            self.metrics.total_connections += 1
            self.metrics.successful_connections += 1
            
            # 启动心跳
            await self.heartbeat_manager.start(self._send_heartbeat)
            
            # 启动发送任务
            self._send_task = create_task_with_logging(self._message_sender())
            
            # 恢复会话状态
            if self.config.preserve_session_state:
                await self._restore_session_state()
            
            # 触发连接回调
            if self.on_connected:
                await self.on_connected(self.session_id)
            
            logger.info(f"连接建立成功: {self.session_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.metrics.failed_connections += 1
            self.metrics.last_failure_reason = str(e)
            
            logger.error(f"连接失败: {self.session_id} - {e}")
            
            # 尝试重连
            if self.config.reconnect_on_error:
                create_task_with_logging(self._attempt_reconnect())
            
            return False
    
    async def disconnect(self):
        """断开连接"""
        logger.info(f"主动断开连接: {self.session_id}")
        await self._cleanup_connection()
        self.state = ConnectionState.DISCONNECTED
    
    async def _establish_connection_direct(self) -> bool:
        """直接建立连接（用于重连，避免递归调用）"""
        if not self._connection_factory or self._connection_kwargs is None:
            return False
        
        self.state = ConnectionState.CONNECTING
        self.last_connection_attempt = time.time()
        
        try:
            # 创建连接
            self.connection = await asyncio.wait_for(
                self._connection_factory(**self._connection_kwargs),
                timeout=self.config.connection_timeout
            )
            
            # 连接成功
            self.state = ConnectionState.CONNECTED
            self.session_state.connection_start_time = utc_now()
            self.retry_count = 0  # 重置重试计数
            
            # 更新指标
            self.metrics.total_connections += 1
            self.metrics.successful_connections += 1
            
            # 启动心跳
            await self.heartbeat_manager.start(self._send_heartbeat)
            
            # 启动发送任务
            self._send_task = create_task_with_logging(self._message_sender())
            
            # 恢复会话状态
            if self.config.preserve_session_state:
                await self._restore_session_state()
            
            # 触发连接回调
            if self.on_connected:
                await self.on_connected(self.session_id)
            
            return True
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.metrics.failed_connections += 1
            self.metrics.last_failure_reason = str(e)
            
            logger.error(f"直接连接失败: {self.session_id} - {e}")
            return False
    
    async def send_message(self, message: Any):
        """发送消息（加入队列）"""
        if self.state != ConnectionState.CONNECTED:
            # 检查缓冲区大小限制
            if len(self.session_state.message_buffer) >= self.config.max_buffer_size:
                # 移除最老的消息
                self.session_state.message_buffer.pop(0)
                logger.warning(f"消息缓冲区已满，丢弃最老消息: {self.session_id}")
            
            # 缓存消息等待重连
            self.session_state.message_buffer.append({
                'message': message,
                'timestamp': time.time()
            })
            logger.debug(f"连接未就绪，消息已缓存: {self.session_id} (缓冲区大小: {len(self.session_state.message_buffer)})")
            return False
        
        await self._message_queue.put(message)
        return True
    
    async def _message_sender(self):
        """消息发送协程"""
        while self.state == ConnectionState.CONNECTED:
            try:
                # 获取待发送消息
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # 发送消息
                await self._send_message_impl(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"消息发送失败: {e}")
                # 连接可能已断开，触发重连
                await self._handle_connection_error(e)
                break
    
    async def _send_message_impl(self, message: Any):
        """实际发送消息的实现"""
        if hasattr(self.connection, 'send'):
            # WebSocket连接
            if isinstance(message, dict):
                await self.connection.send(json.dumps(message))
            else:
                await self.connection.send(str(message))
        else:
            # 其他类型连接的处理
            raise ValueError("不支持的连接类型")
    
    async def _send_heartbeat(self):
        """发送心跳"""
        heartbeat_msg = {
            'type': 'heartbeat',
            'session_id': self.session_id,
            'timestamp': time.time()
        }
        await self._send_message_impl(heartbeat_msg)
    
    async def _attempt_reconnect(self):
        """尝试重连"""
        if self.retry_count >= self.config.max_retries:
            self.state = ConnectionState.PERMANENTLY_FAILED
            logger.error(f"达到最大重试次数，连接永久失败: {self.session_id}")
            return
        
        # 计算重试延迟
        delay = self._calculate_retry_delay()
        
        self.state = ConnectionState.RECONNECTING
        self.retry_count += 1
        self.metrics.total_reconnections += 1
        self.session_state.total_reconnections += 1
        
        logger.info(f"重连尝试 {self.retry_count}/{self.config.max_retries}: {self.session_id} (延迟: {delay:.1f}s)")
        
        # 触发重连回调
        if self.on_reconnect_attempt:
            await self.on_reconnect_attempt(self.session_id, self.retry_count)
        
        # 等待重试延迟
        await asyncio.sleep(delay)
        
        # 使用保存的连接参数重新连接
        if self._connection_factory and self._connection_kwargs is not None:
            try:
                # 清理旧连接
                await self._cleanup_connection()
                
                # 直接建立连接避免递归调用
                success = await self._establish_connection_direct()
                
                if success:
                    logger.info(f"重连成功: {self.session_id}")
                    return
                else:
                    logger.warning(f"重连失败: {self.session_id}")
                    
            except Exception as e:
                logger.error(f"重连过程中发生错误: {self.session_id} - {e}")
                self.metrics.last_failure_reason = str(e)
        else:
            logger.error(f"无法重连，缺少连接工厂或参数: {self.session_id}")
        
        # 如果重连失败且未达到最大重试次数，安排下次重连
        if self.retry_count < self.config.max_retries:
            create_task_with_logging(self._attempt_reconnect())
        else:
            self.state = ConnectionState.PERMANENTLY_FAILED
            logger.error(f"重连彻底失败: {self.session_id}")
    
    def _calculate_retry_delay(self) -> float:
        """计算重试延迟"""
        if self.config.retry_strategy == RetryStrategy.FIXED_INTERVAL:
            return self.config.initial_retry_delay
        
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_retry_delay * self.retry_count
            
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_retry_delay * (2 ** (self.retry_count - 1))
            
        else:  # CUSTOM或默认
            delay = self.config.initial_retry_delay * (1.5 ** (self.retry_count - 1))
        
        return min(delay, self.config.max_retry_delay)
    
    async def _handle_connection_error(self, error: Exception):
        """处理连接错误"""
        logger.warning(f"连接错误: {self.session_id} - {error}")
        
        self.metrics.last_failure_reason = str(error)
        
        # 触发错误回调
        if self.on_error:
            await self.on_error(self.session_id, error)
        
        # 清理当前连接
        await self._cleanup_connection()
        
        # 尝试重连
        if self.config.reconnect_on_error and self.retry_count < self.config.max_retries:
            create_task_with_logging(self._attempt_reconnect())
        else:
            self.state = ConnectionState.FAILED
    
    async def _cleanup_connection(self):
        """清理连接资源"""
        # 停止心跳
        await self.heartbeat_manager.stop()
        
        # 停止发送任务
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                raise
            self._send_task = None
        
        # 关闭连接
        if self.connection:
            try:
                if hasattr(self.connection, 'close'):
                    await self.connection.close()
            except Exception as e:
                logger.debug(f"关闭连接时出错: {e}")
            finally:
                self.connection = None
        
        # 保存会话状态
        if self.config.preserve_session_state:
            await self._save_session_state()
        
        # 触发断开回调
        if self.on_disconnected:
            await self.on_disconnected(self.session_id)
    
    async def _save_session_state(self):
        """保存会话状态"""
        # 这里可以实现状态持久化到数据库或缓存
        logger.debug(f"保存会话状态: {self.session_id}")
    
    async def _restore_session_state(self):
        """恢复会话状态"""
        # 这里可以实现从数据库或缓存恢复状态
        logger.debug(f"恢复会话状态: {self.session_id}")
        
        # 重发缓存的消息
        while self.session_state.message_buffer:
            buffered = self.session_state.message_buffer.pop(0)
            await self.send_message(buffered['message'])
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        uptime = 0
        if (self.session_state.connection_start_time and 
            self.state == ConnectionState.CONNECTED):
            uptime = (utc_now() - self.session_state.connection_start_time).total_seconds()
        
        return {
            'session_id': self.session_id,
            'state': self.state.value,
            'retry_count': self.retry_count,
            'uptime_seconds': uptime,
            'total_reconnections': self.session_state.total_reconnections,
            'heartbeat_alive': self.heartbeat_manager.is_connection_alive(),
            'buffered_messages': len(self.session_state.message_buffer),
            'metrics': {
                'total_connections': self.metrics.total_connections,
                'successful_connections': self.metrics.successful_connections,
                'failed_connections': self.metrics.failed_connections,
                'last_failure_reason': self.metrics.last_failure_reason
            }
        }

class ConnectionManager:
    """连接管理器"""
    
    def __init__(self, default_config: ConnectionConfig = None):
        self.default_config = default_config or ConnectionConfig()
        self.connections: Dict[str, FaultTolerantConnection] = {}
        self.connection_factory = None
        
    def set_connection_factory(self, factory: Callable):
        """设置连接工厂"""
        self.connection_factory = factory
    
    async def create_connection(self, session_id: str, agent_id: str, 
                              config: ConnectionConfig = None, **kwargs) -> FaultTolerantConnection:
        """创建容错连接"""
        if session_id in self.connections:
            return self.connections[session_id]
        
        config = config or self.default_config
        conn = FaultTolerantConnection(session_id, config)
        conn.session_state.agent_id = agent_id
        
        # 设置回调
        conn.on_connected = self._on_connection_established
        conn.on_disconnected = self._on_connection_lost
        conn.on_error = self._on_connection_error
        
        self.connections[session_id] = conn
        
        # 建立连接
        if self.connection_factory:
            await conn.connect(self.connection_factory, **kwargs)
        
        return conn
    
    async def get_connection(self, session_id: str) -> Optional[FaultTolerantConnection]:
        """获取连接"""
        return self.connections.get(session_id)
    
    async def remove_connection(self, session_id: str):
        """移除连接"""
        if session_id in self.connections:
            conn = self.connections[session_id]
            await conn.disconnect()
            del self.connections[session_id]
    
    async def _on_connection_established(self, session_id: str):
        """连接建立回调"""
        logger.info(f"连接管理器: 连接建立 {session_id}")
    
    async def _on_connection_lost(self, session_id: str):
        """连接丢失回调"""
        logger.info(f"连接管理器: 连接丢失 {session_id}")
    
    async def _on_connection_error(self, session_id: str, error: Exception):
        """连接错误回调"""
        logger.warning(f"连接管理器: 连接错误 {session_id} - {error}")
    
    async def get_all_connections_status(self) -> Dict[str, Dict]:
        """获取所有连接状态"""
        return {
            session_id: conn.get_connection_info()
            for session_id, conn in self.connections.items()
        }
    
    async def cleanup_failed_connections(self, max_age_hours: int = 24):
        """清理失败的连接"""
        cutoff_time = utc_now() - timedelta(hours=max_age_hours)
        
        sessions_to_remove = []
        for session_id, conn in self.connections.items():
            if (conn.state in [ConnectionState.FAILED, ConnectionState.PERMANENTLY_FAILED] and
                conn.session_state.connection_start_time and
                conn.session_state.connection_start_time < cutoff_time):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self.remove_connection(session_id)
        
        if sessions_to_remove:
            logger.info(f"清理了 {len(sessions_to_remove)} 个失败的连接")

# 全局连接管理器实例
connection_manager = ConnectionManager()
