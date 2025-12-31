"""
分布式消息总线实现
基于NATS JetStream的智能体消息通信核心组件
"""

from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import asyncio
import uuid
import time
import hashlib
import weakref
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import timedelta
from .models import (
    Message, MessageHeader, MessageType, MessagePriority, DeliveryMode,
    MessageHandler, StreamConfig, ConnectionState, ConnectionMetrics, TopicConfig
)
from .client import NATSClient
from .request_response import RequestResponseManager
from .reliability import ReliabilityManager, RetryConfig
from .advanced_patterns import AdvancedCommunicationManager, RoutingStrategy
from .monitoring import MonitoringManager

from src.core.logging import get_logger
logger = get_logger(__name__)

class DistributedMessageBus:
    """分布式消息总线"""
    
    def __init__(
        self,
        nats_servers: List[str],
        agent_id: str,
        cluster_name: str = "agent-cluster"
    ):
        self.nats_servers = nats_servers
        self.agent_id = agent_id
        self.cluster_name = cluster_name
        
        # NATS客户端
        self.client = NATSClient(
            servers=nats_servers,
            agent_id=agent_id,
            cluster_name=cluster_name
        )
        
        # 消息处理
        self.message_handlers: Dict[MessageType, MessageHandler] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.subscriptions: Dict[str, object] = {}
        
        # 请求-响应管理器
        self.request_response_manager = RequestResponseManager(
            default_timeout=30.0,
            max_concurrent_requests=1000
        )
        
        # 可靠性管理器
        self.reliability_manager = ReliabilityManager()
        
        # 高级通信管理器
        self.advanced_comm_manager = AdvancedCommunicationManager(self.client)
        
        # 监控管理器
        self.monitoring_manager = MonitoringManager(check_interval=30.0)
        
        # 主题配置
        self.topics = TopicConfig.create_for_agent(agent_id, cluster_name)
        
        # 后台任务
        self.cleanup_tasks: Set[asyncio.Task] = set()
        
        # 配置
        self.config = {
            "request_timeout": 30.0,
            "heartbeat_interval": 30.0,
            "cleanup_interval": 60.0,
            "max_pending_requests": 1000
        }
        
        # 设置客户端回调
        self.client.on_connect = self._on_client_connected
        self.client.on_disconnect = self._on_client_disconnected
        self.client.on_error = self._on_client_error
        
        logger.info(f"分布式消息总线初始化完成: {agent_id}")
    
    async def connect(self) -> bool:
        """连接到NATS集群"""
        try:
            success = await self.client.connect()
            if success:
                # 设置可靠性管理器的发送回调
                self.reliability_manager.set_send_callback(self._reliable_send_callback)
                # 启动可靠性管理器
                await self.reliability_manager.start()
                # 启动高级通信管理器
                await self.advanced_comm_manager.start()
                # 启动监控管理器
                await self.monitoring_manager.start()
                logger.info(f"智能体 {self.agent_id} 成功连接到消息总线")
            return success
        except Exception as e:
            logger.error(f"连接消息总线失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开连接"""
        try:
            # 停止监控管理器
            await self.monitoring_manager.stop()
            
            # 停止高级通信管理器
            await self.advanced_comm_manager.stop()
            
            # 停止可靠性管理器
            await self.reliability_manager.stop()
            
            # 停止请求-响应管理器的后台任务
            await self.request_response_manager.stop_background_tasks()
            
            # 清理后台任务
            for task in self.cleanup_tasks:
                task.cancel()
            
            if self.cleanup_tasks:
                await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
            
            # 关闭所有订阅
            for subscription in self.subscriptions.values():
                if hasattr(subscription, 'unsubscribe'):
                    await subscription.unsubscribe()
            
            # 断开客户端连接
            success = await self.client.disconnect()
            
            logger.info(f"智能体 {self.agent_id} 已断开消息总线连接")
            return success
            
        except Exception as e:
            logger.error(f"断开消息总线连接失败: {e}")
            return False
    
    async def _on_client_connected(self):
        """客户端连接成功回调"""
        # 设置基础订阅
        await self._setup_subscriptions()
        
        # 启动后台任务
        await self._start_background_tasks()
        
        # 启动请求-响应管理器的后台任务
        self.request_response_manager.start_background_tasks()
        
        logger.info("消息总线订阅和后台任务已启动")
    
    async def _on_client_disconnected(self):
        """客户端断开连接回调"""
        logger.warning("消息总线连接断开")
    
    async def _on_client_error(self, error: Exception):
        """客户端错误回调"""
        logger.error(f"消息总线客户端错误: {error}")
    
    async def _setup_subscriptions(self):
        """设置基础订阅"""
        try:
            # 订阅直接消息
            direct_sub = await self.client.subscribe(
                self.topics.direct_messages,
                self._handle_direct_message
            )
            if direct_sub:
                self.subscriptions["direct"] = direct_sub
                logger.info(f"订阅直接消息: {self.topics.direct_messages}")
            
            # 订阅广播消息
            broadcast_sub = await self.client.subscribe(
                self.topics.broadcast,
                self._handle_broadcast_message
            )
            if broadcast_sub:
                self.subscriptions["broadcast"] = broadcast_sub
                logger.info(f"订阅广播消息: {self.topics.broadcast}")
            
            # 订阅系统事件
            events_sub = await self.client.subscribe(
                f"{self.topics.system_events}.*",
                self._handle_system_event
            )
            if events_sub:
                self.subscriptions["events"] = events_sub
                logger.info(f"订阅系统事件: {self.topics.system_events}.*")
            
            self.client.metrics.active_subscriptions = len(self.subscriptions)
            
        except Exception as e:
            logger.error(f"设置订阅失败: {e}")
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        try:
            # 心跳任务
            heartbeat_task = create_task_with_logging(self._heartbeat_loop())
            self.cleanup_tasks.add(heartbeat_task)
            
            # 请求超时清理任务
            timeout_task = create_task_with_logging(self._cleanup_expired_requests())
            self.cleanup_tasks.add(timeout_task)
            
            # 指标收集任务
            metrics_task = create_task_with_logging(self._metrics_collection_loop())
            self.cleanup_tasks.add(metrics_task)
            
            logger.debug("后台任务已启动")
            
        except Exception as e:
            logger.error(f"启动后台任务失败: {e}")
    
    async def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[int] = None,
        delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    ) -> bool:
        """发送点对点消息"""
        try:
            # 创建消息
            header = MessageHeader(
                message_id=str(uuid.uuid4()),
                priority=priority,
                ttl=ttl,
                delivery_mode=delivery_mode
            )
            
            message = Message(
                header=header,
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type,
                payload=payload,
                topic=f"agents.direct.{receiver_id}"
            )
            
            # 序列化消息
            message_bytes = message.to_bytes()
            
            # 发送消息
            if delivery_mode == DeliveryMode.AT_LEAST_ONCE and self.client.js:
                await self.client.js_publish(
                    subject=f"agents.direct.{receiver_id}",
                    data=message_bytes,
                    stream="AGENTS_DIRECT"
                )
            else:
                await self.client.publish(
                    subject=f"agents.direct.{receiver_id}",
                    data=message_bytes
                )
            
            # 记录监控指标
            self.record_message_sent(message_type.value, len(message_bytes))
            logger.debug(f"发送消息 {header.message_id} 到 {receiver_id}")
            return True
            
        except Exception as e:
            self.client.metrics.messages_failed += 1
            self.record_message_failed(message_type.value, str(e))
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def send_request(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
        max_retries: int = 3
    ) -> Optional[Message]:
        """发送请求并等待响应"""
        return await self.request_response_manager.send_request(
            sender_function=self._send_message_with_correlation,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timeout=timeout,
            priority=priority,
            max_retries=max_retries
        )
    
    async def _send_message_with_correlation(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: str,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """带关联ID发送消息的内部方法"""
        try:
            reply_subject = f"agents.direct.{self.agent_id}.reply.{correlation_id}"
            
            header = MessageHeader(
                message_id=str(uuid.uuid4()),
                correlation_id=correlation_id,
                reply_to=reply_subject,
                priority=priority
            )
            
            message = Message(
                header=header,
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type,
                payload=payload,
                topic=f"agents.direct.{receiver_id}"
            )
            
            # 发送消息
            message_bytes = message.to_bytes()
            if self.client.js:
                await self.client.js_publish(
                    subject=f"agents.direct.{receiver_id}",
                    data=message_bytes,
                    stream="AGENTS_DIRECT"
                )
            else:
                await self.client.publish(
                    subject=f"agents.direct.{receiver_id}",
                    data=message_bytes
                )
            
            self.client.metrics.messages_sent += 1
            return True
            
        except Exception as e:
            self.client.metrics.messages_failed += 1
            logger.error(f"发送带关联ID的消息失败: {e}")
            return False
    
    async def send_reply(
        self,
        original_message: Message,
        payload: Dict[str, Any],
        message_type: Optional[MessageType] = None
    ) -> bool:
        """发送回复消息"""
        if not original_message.header.reply_to or not original_message.header.correlation_id:
            logger.warning("无法回复：缺少reply_to或correlation_id")
            return False
        
        try:
            # 创建回复消息
            reply_type = message_type or MessageType.ACK
            
            header = MessageHeader(
                message_id=str(uuid.uuid4()),
                correlation_id=original_message.header.correlation_id
            )
            
            reply_message = Message(
                header=header,
                sender_id=self.agent_id,
                receiver_id=original_message.sender_id,
                message_type=reply_type,
                payload=payload,
                topic=original_message.header.reply_to
            )
            
            # 发送回复
            message_bytes = reply_message.to_bytes()
            await self.client.publish(
                subject=original_message.header.reply_to,
                data=message_bytes
            )
            
            logger.debug(f"发送回复: {original_message.header.message_id}")
            return True
            
        except Exception as e:
            logger.error(f"发送回复失败: {e}")
            return False
    
    async def send_reliable_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        require_ack: bool = True,
        retry_config: Optional[RetryConfig] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[float] = None
    ) -> Optional[str]:
        """发送可靠消息（支持重试和确认）"""
        try:
            # 创建消息
            header = MessageHeader(
                message_id=str(uuid.uuid4()),
                priority=priority,
                ttl=ttl,
                delivery_mode=DeliveryMode.AT_LEAST_ONCE if require_ack else DeliveryMode.AT_MOST_ONCE
            )
            
            message = Message(
                header=header,
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type,
                payload=payload,
                topic=f"agents.direct.{receiver_id}"
            )
            
            # 通过可靠性管理器发送
            message_id = await self.reliability_manager.send_reliable_message(
                message=message,
                require_ack=require_ack,
                retry_config=retry_config
            )
            
            if message_id:
                logger.debug(f"可靠消息发送成功: {message_id}")
            
            return message_id
            
        except Exception as e:
            logger.error(f"发送可靠消息失败: {e}")
            return None
    
    async def broadcast_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        group: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """广播消息"""
        try:
            # 确定广播主题
            if group:
                subject = f"agents.group.{group}"
                stream = "AGENTS_BROADCAST"
            else:
                subject = self.topics.broadcast
                stream = "AGENTS_BROADCAST"
            
            # 创建广播消息
            header = MessageHeader(
                message_id=str(uuid.uuid4()),
                priority=priority
            )
            
            message = Message(
                header=header,
                sender_id=self.agent_id,
                receiver_id=None,  # 广播消息
                message_type=message_type,
                payload=payload,
                topic=subject
            )
            
            # 发送广播
            message_bytes = message.to_bytes()
            if self.client.js:
                await self.client.js_publish(
                    subject=subject,
                    data=message_bytes,
                    stream=stream
                )
            else:
                await self.client.publish(
                    subject=subject,
                    data=message_bytes
                )
            
            logger.debug(f"广播消息: {header.message_id}")
            return True
            
        except Exception as e:
            self.client.metrics.messages_failed += 1
            logger.error(f"广播消息失败: {e}")
            return False
    
    async def subscribe_to_group(self, group_name: str) -> bool:
        """订阅组消息"""
        try:
            subject = f"agents.group.{group_name}"
            
            if subject in self.subscriptions:
                logger.debug(f"已订阅组: {group_name}")
                return True
            
            # 创建订阅
            subscription = await self.client.subscribe(
                subject,
                self._handle_group_message
            )
            
            if subscription:
                self.subscriptions[subject] = subscription
                self.client.metrics.active_subscriptions += 1
                logger.info(f"订阅组消息: {group_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"订阅组失败: {e}")
            return False
    
    async def unsubscribe_from_group(self, group_name: str) -> bool:
        """取消组消息订阅"""
        try:
            subject = f"agents.group.{group_name}"
            
            if subject not in self.subscriptions:
                logger.debug(f"未订阅组: {group_name}")
                return True
            
            # 取消订阅
            subscription = self.subscriptions[subject]
            if hasattr(subscription, 'unsubscribe'):
                await subscription.unsubscribe()
            
            del self.subscriptions[subject]
            self.client.metrics.active_subscriptions -= 1
            
            logger.info(f"取消订阅组: {group_name}")
            return True
            
        except Exception as e:
            logger.error(f"取消订阅组失败: {e}")
            return False
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any],
        is_async: bool = True,
        max_concurrent: int = 10
    ):
        """注册消息处理器"""
        message_handler = MessageHandler(
            message_type=message_type,
            handler=handler,
            is_async=is_async,
            max_concurrent=max_concurrent
        )
        
        self.message_handlers[message_type] = message_handler
        logger.info(f"注册消息处理器: {message_type.value}")
    
    def register_request_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any],
        is_async: bool = True
    ):
        """注册请求处理器"""
        self.request_response_manager.register_request_handler(
            message_type=message_type,
            handler=handler,
            is_async=is_async
        )
    
    def register_response_callback(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Any]
    ):
        """注册响应回调处理器"""
        self.request_response_manager.register_response_callback(
            message_type=message_type,
            callback=callback
        )
    
    async def _handle_direct_message(self, msg):
        """处理直接消息"""
        try:
            # 反序列化消息
            message = Message.from_bytes(msg.data)
            
            self.client.metrics.messages_received += 1
            self.client.metrics.bytes_received += len(msg.data)
            
            # 首先尝试使用RequestResponseManager处理响应
            if self.request_response_manager.handle_response(message):
                return
            
            # 检查是否需要RequestResponseManager处理请求
            if message.header.reply_to and message.header.correlation_id:
                # 这是一个请求消息，使用RequestResponseManager处理
                await self.request_response_manager.handle_request(
                    message=message,
                    reply_function=self._send_reply_message
                )
                return
            
            # 分发到消息处理器
            await self._dispatch_message(message)
            
        except Exception as e:
            logger.error(f"处理直接消息失败: {e}")
    
    async def _send_reply_message(self, original_message: Message, payload: Dict[str, Any], message_type: MessageType) -> bool:
        """发送回复消息的内部方法（供RequestResponseManager使用）"""
        return await self.send_reply(
            original_message=original_message,
            payload=payload,
            message_type=message_type
        )
    
    async def _handle_reply_message(self, msg, correlation_id: str):
        """处理回复消息"""
        try:
            message = Message.from_bytes(msg.data)
            
            if correlation_id in self.pending_requests:
                future = self.pending_requests[correlation_id]
                if not future.done():
                    future.set_result(message)
                    
        except Exception as e:
            logger.error(f"处理回复消息失败: {e}")
    
    async def _handle_broadcast_message(self, msg):
        """处理广播消息"""
        try:
            message = Message.from_bytes(msg.data)
            
            # 忽略自己发送的消息
            if message.sender_id == self.agent_id:
                return
            
            self.client.metrics.messages_received += 1
            self.client.metrics.bytes_received += len(msg.data)
            
            # 分发消息
            await self._dispatch_message(message)
            
        except Exception as e:
            logger.error(f"处理广播消息失败: {e}")
    
    async def _handle_group_message(self, msg):
        """处理组消息"""
        try:
            message = Message.from_bytes(msg.data)
            
            # 忽略自己发送的消息
            if message.sender_id == self.agent_id:
                return
            
            self.client.metrics.messages_received += 1
            self.client.metrics.bytes_received += len(msg.data)
            
            # 分发消息
            await self._dispatch_message(message)
            
        except Exception as e:
            logger.error(f"处理组消息失败: {e}")
    
    async def _handle_system_event(self, msg):
        """处理系统事件"""
        try:
            message = Message.from_bytes(msg.data)
            
            self.client.metrics.messages_received += 1
            self.client.metrics.bytes_received += len(msg.data)
            
            # 分发消息
            await self._dispatch_message(message)
            
        except Exception as e:
            logger.error(f"处理系统事件失败: {e}")
    
    async def _dispatch_message(self, message: Message):
        """分发消息到处理器"""
        message_type = message.message_type
        
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            
            try:
                await handler.handle_message(message)
                
            except Exception as e:
                logger.error(f"消息处理器执行失败 {message_type.value}: {e}")
                
                # 发送NACK (如果需要)
                if message.header.reply_to:
                    await self.send_reply(
                        message,
                        {"error": str(e), "status": "failed"},
                        MessageType.NACK
                    )
        else:
            logger.warning(f"未注册的消息类型: {message_type.value}")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while True:
            try:
                await asyncio.sleep(self.config["heartbeat_interval"])
                
                await self.broadcast_message(
                    message_type=MessageType.HEARTBEAT,
                    payload={
                        "agent_id": self.agent_id,
                        "timestamp": utc_now().isoformat(),
                        "status": "active",
                        "metrics": self.client.get_metrics().to_dict()
                    }
                )
                
                # 更新心跳时间
                self.client.metrics.last_heartbeat = utc_now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环错误: {e}")
    
    async def _cleanup_expired_requests(self):
        """清理过期的请求"""
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval"])
                
                current_time = time.time()
                expired_requests = []
                
                for correlation_id, future in self.pending_requests.items():
                    if not future.done():
                        # 简单的清理策略：限制pending请求数量
                        if len(self.pending_requests) > self.config["max_pending_requests"]:
                            future.cancel()
                            expired_requests.append(correlation_id)
                
                for correlation_id in expired_requests:
                    if correlation_id in self.pending_requests:
                        del self.pending_requests[correlation_id]
                
                self.client.metrics.pending_requests = len(self.pending_requests)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理过期请求失败: {e}")
    
    async def _metrics_collection_loop(self):
        """指标收集循环"""
        while True:
            try:
                await asyncio.sleep(10)  # 每10秒更新指标
                
                # 更新连接指标
                if self.client.nc and self.client.nc.is_connected:
                    # 更新基本指标
                    self.client.metrics.last_heartbeat = utc_now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
    
    def get_metrics(self) -> ConnectionMetrics:
        """获取指标"""
        return self.client.get_metrics()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return self.client.get_connection_status()
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.client.is_connected()
    
    async def _reliable_send_callback(self, message: Message) -> bool:
        """可靠性管理器的发送回调"""
        try:
            # 根据消息的投递模式选择发送方式
            if message.header.delivery_mode in [DeliveryMode.AT_LEAST_ONCE, DeliveryMode.EXACTLY_ONCE]:
                ack = await self.client.js_publish(
                    subject=message.topic,
                    data=message.to_bytes()
                )
                return ack is not None
            else:
                success = await self.client.publish(
                    subject=message.topic,
                    data=message.to_bytes()
                )
                return success
        except Exception as e:
            logger.error(f"可靠发送失败: {e}")
            return False
    
    def get_reliability_statistics(self) -> Dict[str, Any]:
        """获取可靠性统计信息"""
        return self.reliability_manager.get_statistics()
    
    def get_dead_letter_messages(self) -> List[Dict[str, Any]]:
        """获取死信队列消息"""
        return self.reliability_manager.get_dead_letter_messages()
    
    async def acknowledge_message(self, message_id: str) -> bool:
        """确认消息"""
        return self.reliability_manager.acknowledge_message(message_id)
    
    def reject_message(self, message_id: str, reason: str = "") -> None:
        """拒绝消息（触发重试）"""
        self.reliability_manager.nack_message(message_id, reason)
    
    # ====== 高级通信模式方法 ======
    
    def create_multicast_group(
        self,
        group_name: str,
        description: Optional[str] = None,
        max_members: int = 100
    ) -> str:
        """创建多播组"""
        return self.advanced_comm_manager.multicast_manager.create_group(
            group_name, description, max_members
        )
    
    def join_multicast_group(self, group_id: str, agent_id: Optional[str] = None) -> bool:
        """加入多播组"""
        agent_id = agent_id or self.agent_id
        return self.advanced_comm_manager.multicast_manager.join_group(group_id, agent_id)
    
    def leave_multicast_group(self, group_id: str, agent_id: Optional[str] = None) -> bool:
        """离开多播组"""
        agent_id = agent_id or self.agent_id
        return self.advanced_comm_manager.multicast_manager.leave_group(group_id, agent_id)
    
    async def send_multicast_message(
        self,
        group_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        exclude_agents: Optional[Set[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> int:
        """发送多播消息"""
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            priority=priority,
            multicast_group=group_id
        )
        
        message = Message(
            header=header,
            sender_id=self.agent_id,
            receiver_id=None,  # 多播消息没有单一接收者
            message_type=message_type,
            payload=payload
        )
        
        return await self.advanced_comm_manager.multicast_manager.send_multicast(
            group_id, message, exclude_agents
        )
    
    async def send_data_stream(
        self,
        receiver_id: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """发送数据流"""
        return await self.advanced_comm_manager.streaming_manager.send_stream(
            receiver_id, data, metadata=metadata
        )
    
    def register_agent_capability(
        self,
        capabilities: Set[str],
        load_factor: float = 0.0,
        location: Optional[str] = None,
        priority: int = 0
    ):
        """注册智能体能力"""
        self.advanced_comm_manager.smart_router.register_agent_capability(
            self.agent_id, capabilities, load_factor, location, priority
        )
    
    def update_agent_load(self, load_factor: float):
        """更新智能体负载"""
        self.advanced_comm_manager.smart_router.update_agent_load(self.agent_id, load_factor)
    
    async def route_message_by_capability(
        self,
        message: Message,
        required_capability: str,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED
    ) -> bool:
        """根据能力路由消息"""
        return await self.advanced_comm_manager.smart_router.route_message(
            message, required_capability, strategy
        )
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.advanced_comm_manager.register_event_handler(event_type, handler)
    
    def unregister_event_handler(self, event_type: str):
        """取消注册事件处理器"""
        self.advanced_comm_manager.unregister_event_handler(event_type)
    
    def get_multicast_groups(self) -> List[Dict[str, Any]]:
        """获取多播组列表"""
        return self.advanced_comm_manager.multicast_manager.list_groups()
    
    def get_active_streams(self) -> List[Dict[str, Any]]:
        """获取活跃的数据流"""
        return self.advanced_comm_manager.streaming_manager.list_active_streams()
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        return self.advanced_comm_manager.smart_router.get_routing_statistics()
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """获取高级通信统计信息"""
        return self.advanced_comm_manager.get_advanced_statistics()
    
    # ====== 监控和性能优化方法 ======
    
    def record_message_sent(self, message_type: str, size: int = 0) -> None:
        """记录发送消息指标"""
        self.monitoring_manager.record_message_metric("messages_sent", 1, {"message_type": message_type, "size": str(size)})
    
    def record_message_received(self, message_type: str, size: int = 0) -> None:
        """记录接收消息指标"""
        self.monitoring_manager.record_message_metric("messages_received", 1, {"message_type": message_type, "size": str(size)})
    
    def record_message_failed(self, message_type: str, error: str) -> None:
        """记录消息失败指标"""
        self.monitoring_manager.record_message_metric("messages_failed", 1, {"message_type": message_type, "error": error})
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板信息"""
        return self.monitoring_manager.get_monitoring_dashboard()
    
    def enable_compression(self, threshold: int = 1024) -> None:
        """启用消息压缩"""
        self.monitoring_manager.performance_optimizer.enable_compression(threshold)
    
    def disable_compression(self) -> None:
        """禁用消息压缩"""
        self.monitoring_manager.performance_optimizer.disable_compression()
    
    def enable_batching(self, batch_size: int = 10, timeout: float = 1.0) -> None:
        """启用批处理"""
        self.monitoring_manager.performance_optimizer.enable_batching(batch_size, timeout)
    
    def disable_batching(self) -> None:
        """禁用批处理"""
        self.monitoring_manager.performance_optimizer.disable_batching()
    
    def register_alert_handler(self, handler: Callable[[Any], None]) -> None:
        """注册告警处理器"""
        self.monitoring_manager.register_alert_handler(handler)
    
    def _create_message_header(self, **kwargs) -> MessageHeader:
        """创建消息头的辅助方法"""
        defaults = {
            "message_id": str(uuid.uuid4()),
        }
        defaults.update(kwargs)
        return MessageHeader(**defaults)
    
    def _create_message(self, header: MessageHeader, **kwargs) -> Message:
        """创建消息的辅助方法"""
        defaults = {
            "sender_id": self.agent_id,
        }
        defaults.update(kwargs)
        return Message(header=header, **defaults)
