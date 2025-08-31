"""
NATS客户端连接管理
处理与NATS服务器的连接、重连和基础操作
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable

try:
    import nats
    from nats.errors import TimeoutError as NatsTimeoutError, ConnectionClosedError, NoServersError
    from nats.js.api import StreamConfig as NatsStreamConfig, ConsumerConfig
    NATS_AVAILABLE = True
except ImportError:
    # 当nats-py包不可用时的占位符
    nats = None
    NatsTimeoutError = TimeoutError
    ConnectionClosedError = ConnectionError
    NoServersError = ConnectionError
    NatsStreamConfig = dict
    ConsumerConfig = dict
    NATS_AVAILABLE = False

from .models import ConnectionState, ConnectionMetrics, StreamConfig, TopicConfig

logger = logging.getLogger(__name__)


class NATSClient:
    """NATS客户端封装"""
    
    def __init__(
        self,
        servers: List[str],
        agent_id: str,
        cluster_name: str = "agent-cluster",
        connection_timeout: float = 10.0,
        reconnect_attempts: int = 10,
        max_pending_messages: int = 10000
    ):
        self.servers = servers
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        
        # 连接配置
        self.connection_timeout = connection_timeout
        self.reconnect_attempts = reconnect_attempts
        self.max_pending_messages = max_pending_messages
        
        # NATS连接
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[nats.js.JetStreamContext] = None
        
        # 状态管理
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        
        # 主题配置
        self.topics = TopicConfig.create_for_agent(agent_id, cluster_name)
        
        # 事件回调
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_reconnect: Optional[Callable] = None
        
        # 流配置
        self.stream_configs = self._create_default_streams()
        
    def _create_default_streams(self) -> List[StreamConfig]:
        """创建默认的JetStream流配置"""
        return [
            StreamConfig(
                name="AGENTS_DIRECT",
                subjects=[f"agents.direct.*"]
            ),
            StreamConfig(
                name="AGENTS_BROADCAST", 
                subjects=[f"agents.broadcast.*"]
            ),
            StreamConfig(
                name="AGENTS_TASKS",
                subjects=[f"agents.tasks.*"]
            ),
            StreamConfig(
                name="AGENTS_RESOURCES",
                subjects=[f"agents.resources.*"] 
            ),
            StreamConfig(
                name="AGENTS_EVENTS",
                subjects=[f"agents.events.*"]
            ),
            StreamConfig(
                name="AGENTS_STREAMS",
                subjects=[f"agents.streams.*"],
                max_messages=10000000,  # 更大的容量用于数据流
                max_bytes=10*1024*1024*1024  # 10GB
            )
        ]
    
    async def connect(self) -> bool:
        """连接到NATS集群"""
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return True
            
        try:
            if not NATS_AVAILABLE:
                logger.warning("NATS-py包不可用，模拟连接成功")
                self.state = ConnectionState.CONNECTED
                self.metrics.connections_attempted += 1
                self.metrics.connections_successful += 1
                if self.on_connect:
                    await self.on_connect()
                return True
            
            self.state = ConnectionState.CONNECTING
            logger.info(f"智能体 {self.agent_id} 正在连接到NATS集群...")
            
            # 连接配置
            options = {
                "servers": self.servers,
                "name": f"agent_{self.agent_id}",
                "connect_timeout": self.connection_timeout,
                "max_reconnect_attempts": self.reconnect_attempts,
                "reconnect_time_wait": 2,
                "max_outstanding_pings": 2,
                "ping_interval": 60,
                "allow_reconnect": True,
                "flusher_queue_size": self.max_pending_messages,
                "disconnected_cb": self._on_disconnected,
                "reconnected_cb": self._on_reconnected,
                "error_cb": self._on_error,
                "closed_cb": self._on_closed
            }
            
            # 建立连接
            self.nc = await nats.connect(**options)
            
            # 启用JetStream
            self.js = self.nc.jetstream()
            
            # 创建流
            await self._setup_streams()
            
            self.state = ConnectionState.CONNECTED
            self.metrics.connection_reconnects = 0
            
            logger.info(f"智能体 {self.agent_id} 成功连接到NATS集群")
            
            # 触发连接回调
            if self.on_connect:
                try:
                    await self.on_connect()
                except Exception as e:
                    logger.error(f"连接回调执行失败: {e}")
                    
            return True
            
        except Exception as e:
            self.state = ConnectionState.DISCONNECTED
            logger.error(f"连接NATS失败: {e}")
            
            if self.on_error:
                try:
                    await self.on_error(e)
                except Exception as cb_error:
                    logger.error(f"错误回调执行失败: {cb_error}")
                    
            return False
    
    async def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self.nc and self.state != ConnectionState.CLOSED:
                self.state = ConnectionState.DRAINING
                await self.nc.drain()
                self.state = ConnectionState.CLOSED
                
            logger.info(f"智能体 {self.agent_id} 已断开NATS连接")
            return True
            
        except Exception as e:
            logger.error(f"断开连接失败: {e}")
            return False
    
    async def _setup_streams(self):
        """设置JetStream流"""
        for stream_config in self.stream_configs:
            try:
                # 检查流是否已存在
                try:
                    await self.js.stream_info(stream_config.name)
                    logger.debug(f"流 {stream_config.name} 已存在")
                    continue
                except:
                    pass
                
                # 创建流
                nats_config = NatsStreamConfig(
                    name=stream_config.name,
                    subjects=stream_config.subjects,
                    retention=stream_config.retention_policy,
                    max_msgs=stream_config.max_messages,
                    max_bytes=stream_config.max_bytes,
                    max_age=stream_config.max_age,
                    storage=stream_config.storage,
                    num_replicas=stream_config.replicas
                )
                
                await self.js.add_stream(nats_config)
                logger.info(f"创建JetStream流: {stream_config.name}")
                
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"创建流 {stream_config.name} 失败: {e}")
                    
    async def _on_disconnected(self):
        """连接断开回调"""
        self.state = ConnectionState.RECONNECTING
        self.metrics.connection_errors += 1
        logger.warning(f"智能体 {self.agent_id} 与NATS连接断开")
        
        if self.on_disconnect:
            try:
                await self.on_disconnect()
            except Exception as e:
                logger.error(f"断开连接回调执行失败: {e}")
    
    async def _on_reconnected(self):
        """重新连接成功回调"""
        self.state = ConnectionState.CONNECTED
        self.metrics.connection_reconnects += 1
        logger.info(f"智能体 {self.agent_id} 重新连接到NATS成功")
        
        if self.on_reconnect:
            try:
                await self.on_reconnect()
            except Exception as e:
                logger.error(f"重连回调执行失败: {e}")
    
    async def _on_error(self, error: Exception):
        """错误回调"""
        self.metrics.connection_errors += 1
        logger.error(f"NATS连接错误: {error}")
        
        if self.on_error:
            try:
                await self.on_error(error)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
    
    async def _on_closed(self):
        """连接关闭回调"""
        self.state = ConnectionState.CLOSED
        logger.info(f"智能体 {self.agent_id} NATS连接已关闭")
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.state == ConnectionState.CONNECTED and self.nc and self.nc.is_connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        status = {
            "state": self.state.value,
            "agent_id": self.agent_id,
            "cluster_name": self.cluster_name,
            "connected": self.is_connected(),
            "metrics": self.metrics.to_dict()
        }
        
        if self.nc:
            try:
                server_info = getattr(self.nc, '_server_info', {})
                stats = getattr(self.nc, 'stats', {})
                
                status.update({
                    "server_info": server_info,
                    "stats": stats,
                    "servers": self.servers,
                    "connected_url": str(getattr(self.nc, 'connected_url', '')),
                })
            except Exception as e:
                logger.debug(f"获取连接详情失败: {e}")
                
        return status
    
    async def publish(
        self,
        subject: str,
        data: bytes,
        reply: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """发布消息到指定主题"""
        if not self.is_connected():
            logger.error("NATS未连接，无法发布消息")
            return False
        
        if not NATS_AVAILABLE:
            logger.debug(f"NATS-py包不可用，模拟发布消息到: {subject}")
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)
            return True
            
        try:
            await self.nc.publish(subject, data, reply=reply, headers=headers)
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)
            return True
            
        except Exception as e:
            self.metrics.messages_failed += 1
            logger.error(f"发布消息失败: {e}")
            return False
    
    async def subscribe(
        self,
        subject: str,
        callback: Callable,
        queue: Optional[str] = None,
        max_msgs: Optional[int] = None
    ):
        """订阅主题"""
        if not self.is_connected():
            logger.error("NATS未连接，无法订阅")
            return None
        
        if not NATS_AVAILABLE:
            logger.warning(f"NATS-py包不可用，模拟订阅主题: {subject}")
            # 返回模拟的订阅对象
            class MockSubscription:
                async def unsubscribe(self):
                    pass
            return MockSubscription()
            
        try:
            subscription = await self.nc.subscribe(
                subject, 
                cb=callback,
                queue=queue,
                max_msgs=max_msgs
            )
            
            self.metrics.active_subscriptions += 1
            logger.info(f"订阅主题: {subject}")
            return subscription
            
        except Exception as e:
            logger.error(f"订阅主题失败: {e}")
            return None
    
    async def request(
        self,
        subject: str,
        data: bytes,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """发送请求并等待响应"""
        if not self.is_connected():
            logger.error("NATS未连接，无法发送请求")
            return None
            
        try:
            response = await self.nc.request(
                subject, 
                data, 
                timeout=timeout,
                headers=headers
            )
            
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)
            self.metrics.messages_received += 1
            self.metrics.bytes_received += len(response.data)
            
            return response
            
        except NatsTimeoutError:
            logger.warning(f"请求超时: {subject}")
            return None
        except Exception as e:
            self.metrics.messages_failed += 1
            logger.error(f"发送请求失败: {e}")
            return None
    
    async def js_publish(
        self,
        subject: str,
        data: bytes,
        stream: Optional[str] = None,
        timeout: float = 10.0
    ):
        """使用JetStream发布持久化消息"""
        if not self.is_connected():
            logger.error("NATS未连接，无法发布持久化消息")
            return None
        
        if not NATS_AVAILABLE:
            logger.debug(f"NATS-py包不可用，模拟JetStream发布到: {subject}")
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)
            # 返回模拟的ACK对象
            class MockAck:
                sequence = 1
                stream = stream or "MOCK_STREAM"
            return MockAck()
        
        if not self.js:
            logger.error("JetStream未启用，无法发布持久化消息")
            return None
            
        try:
            ack = await self.js.publish(
                subject, 
                data,
                stream=stream,
                timeout=timeout
            )
            
            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)
            
            return ack
            
        except Exception as e:
            self.metrics.messages_failed += 1
            logger.error(f"JetStream发布消息失败: {e}")
            return None
    
    def get_topics(self) -> TopicConfig:
        """获取主题配置"""
        return self.topics
    
    def get_metrics(self) -> ConnectionMetrics:
        """获取连接指标"""
        return self.metrics