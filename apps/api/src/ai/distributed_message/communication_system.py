"""
智能体通信系统主类
提供高层次的智能体间通信接口和便捷方法
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Callable

from .models import Message, MessageType, MessagePriority, DeliveryMode
from .message_bus import DistributedMessageBus
from .protocol import MessageProtocol

logger = logging.getLogger(__name__)


class AgentCommunicationSystem:
    """智能体通信系统主类"""
    
    def __init__(
        self,
        agent_id: str,
        nats_servers: List[str],
        cluster_name: str = "agent-cluster",
        agent_type: str = "generic",
        capabilities: Optional[List[str]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        
        # 创建消息总线
        self.message_bus = DistributedMessageBus(
            nats_servers=nats_servers,
            agent_id=agent_id,
            cluster_name=cluster_name
        )
        
        # 状态信息
        self.start_time = time.time()
        self.groups: List[str] = []
        self.collaborations: Dict[str, Dict[str, Any]] = {}
        
        # 注册基础消息处理器
        self._setup_basic_handlers()
        
        logger.info(f"智能体通信系统初始化完成: {agent_id}")
    
    async def initialize(self) -> bool:
        """初始化通信系统"""
        try:
            success = await self.message_bus.connect()
            if success:
                # 广播智能体加入消息
                await self._announce_agent_joined()
                logger.info(f"智能体通信系统启动成功: {self.agent_id}")
            return success
        except Exception as e:
            logger.error(f"初始化通信系统失败: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """关闭通信系统"""
        try:
            # 广播智能体离开消息
            await self._announce_agent_left()
            
            # 离开所有组
            for group in self.groups.copy():
                await self.leave_group(group)
            
            # 断开消息总线连接
            success = await self.message_bus.disconnect()
            
            logger.info(f"智能体通信系统已关闭: {self.agent_id}")
            return success
            
        except Exception as e:
            logger.error(f"关闭通信系统失败: {e}")
            return False
    
    def _setup_basic_handlers(self):
        """设置基础消息处理器"""
        
        async def handle_ping(message: Message):
            """处理ping消息"""
            try:
                pong_payload = MessageProtocol.create_pong_message(
                    message.payload.get("timestamp", ""),
                    self.agent_id
                )
                await self.message_bus.send_reply(message, pong_payload, MessageType.PONG)
                logger.debug(f"回复ping消息: {message.sender_id}")
                
            except Exception as e:
                logger.error(f"处理ping消息失败: {e}")
        
        async def handle_heartbeat(message: Message):
            """处理心跳消息"""
            logger.debug(f"收到心跳: {message.sender_id}")
            # 可以在这里记录其他智能体的状态
        
        async def handle_agent_joined(message: Message):
            """处理智能体加入消息"""
            joined_agent = message.payload.get("agent_id")
            agent_type = message.payload.get("agent_type")
            capabilities = message.payload.get("capabilities", [])
            
            logger.info(f"智能体加入: {joined_agent} (类型: {agent_type}, 能力: {capabilities})")
        
        async def handle_agent_left(message: Message):
            """处理智能体离开消息"""
            left_agent = message.payload.get("agent_id")
            reason = message.payload.get("reason", "unknown")
            
            logger.info(f"智能体离开: {left_agent} (原因: {reason})")
        
        # 注册基础处理器
        self.message_bus.register_handler(MessageType.PING, handle_ping)
        self.message_bus.register_handler(MessageType.HEARTBEAT, handle_heartbeat)
        self.message_bus.register_handler(MessageType.AGENT_JOINED, handle_agent_joined)
        self.message_bus.register_handler(MessageType.AGENT_LEFT, handle_agent_left)
    
    # 便捷方法
    async def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        **kwargs
    ) -> bool:
        """发送消息（便捷方法）"""
        return await self.message_bus.send_message(
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            **kwargs
        )
    
    async def send_request(
        self,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Message]:
        """发送请求（便捷方法）"""
        return await self.message_bus.send_request(
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timeout=timeout
        )
    
    async def broadcast(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        group: Optional[str] = None
    ) -> bool:
        """广播消息（便捷方法）"""
        return await self.message_bus.broadcast_message(
            message_type=message_type,
            payload=payload,
            group=group
        )
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any],
        **kwargs
    ):
        """注册消息处理器（便捷方法）"""
        self.message_bus.register_handler(
            message_type=message_type,
            handler=handler,
            **kwargs
        )
    
    # 高级功能方法
    async def ping_agent(self, agent_id: str, timeout: float = 10.0) -> Optional[float]:
        """Ping指定智能体，返回RTT（毫秒）"""
        try:
            start_time = time.time()
            
            ping_payload = MessageProtocol.create_ping_message(self.agent_id)
            response = await self.send_request(
                receiver_id=agent_id,
                message_type=MessageType.PING,
                payload=ping_payload,
                timeout=timeout
            )
            
            if response and response.message_type == MessageType.PONG:
                end_time = time.time()
                rtt = (end_time - start_time) * 1000  # 转换为毫秒
                logger.debug(f"Ping {agent_id}: {rtt:.2f}ms")
                return rtt
            
            return None
            
        except Exception as e:
            logger.error(f"Ping智能体失败: {e}")
            return None
    
    async def request_task_execution(
        self,
        agent_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        timeout: float = 60.0,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """请求任务执行"""
        try:
            task_id = str(uuid.uuid4())
            task_payload = MessageProtocol.create_task_request(
                task_id=task_id,
                task_type=task_type,
                task_data=task_data,
                **kwargs
            )
            
            response = await self.send_request(
                receiver_id=agent_id,
                message_type=MessageType.TASK_REQUEST,
                payload=task_payload,
                timeout=timeout
            )
            
            if response:
                return response.payload
            
            return None
            
        except Exception as e:
            logger.error(f"请求任务执行失败: {e}")
            return None
    
    async def send_task_result(
        self,
        requester_id: str,
        task_id: str,
        result: Dict[str, Any],
        status: str = "completed",
        execution_time: Optional[float] = None
    ) -> bool:
        """发送任务结果"""
        try:
            result_payload = MessageProtocol.create_task_result(
                task_id=task_id,
                result=result,
                status=status,
                execution_time=execution_time
            )
            
            return await self.send_message(
                receiver_id=requester_id,
                message_type=MessageType.TASK_RESULT,
                payload=result_payload
            )
            
        except Exception as e:
            logger.error(f"发送任务结果失败: {e}")
            return False
    
    async def join_group(self, group_name: str) -> bool:
        """加入组"""
        try:
            success = await self.message_bus.subscribe_to_group(group_name)
            
            if success:
                self.groups.append(group_name)
                
                # 广播加入组消息
                join_payload = MessageProtocol.create_agent_joined_message(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    capabilities=self.capabilities,
                    group=group_name
                )
                
                await self.broadcast(
                    message_type=MessageType.AGENT_JOINED,
                    payload=join_payload,
                    group=group_name
                )
                
                logger.info(f"加入组: {group_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"加入组失败: {e}")
            return False
    
    async def leave_group(self, group_name: str) -> bool:
        """离开组"""
        try:
            # 广播离开组消息
            leave_payload = MessageProtocol.create_agent_left_message(
                agent_id=self.agent_id,
                reason="left_group",
                group=group_name
            )
            
            await self.broadcast(
                message_type=MessageType.AGENT_LEFT,
                payload=leave_payload,
                group=group_name
            )
            
            success = await self.message_bus.unsubscribe_from_group(group_name)
            
            if success and group_name in self.groups:
                self.groups.remove(group_name)
                logger.info(f"离开组: {group_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"离开组失败: {e}")
            return False
    
    async def request_collaboration(
        self,
        collaboration_type: str,
        description: str,
        required_capabilities: List[str],
        max_participants: int = 10,
        duration_minutes: Optional[int] = None,
        target_group: Optional[str] = None
    ) -> str:
        """发起协作请求"""
        try:
            collaboration_id = str(uuid.uuid4())
            
            invite_payload = MessageProtocol.create_collaboration_invite(
                collaboration_id=collaboration_id,
                collaboration_type=collaboration_type,
                description=description,
                initiator_id=self.agent_id,
                required_capabilities=required_capabilities,
                max_participants=max_participants,
                duration_minutes=duration_minutes
            )
            
            # 记录协作信息
            self.collaborations[collaboration_id] = {
                "type": collaboration_type,
                "description": description,
                "participants": [self.agent_id],
                "status": "recruiting"
            }
            
            # 发送协作邀请
            if target_group:
                await self.broadcast(
                    message_type=MessageType.COLLABORATION_INVITE,
                    payload=invite_payload,
                    group=target_group
                )
            else:
                await self.broadcast(
                    message_type=MessageType.COLLABORATION_INVITE,
                    payload=invite_payload
                )
            
            logger.info(f"发起协作请求: {collaboration_id}")
            return collaboration_id
            
        except Exception as e:
            logger.error(f"发起协作请求失败: {e}")
            return ""
    
    async def respond_to_collaboration(
        self,
        collaboration_id: str,
        response: str,  # "accept" or "reject"
        initiator_id: str
    ) -> bool:
        """响应协作请求"""
        try:
            response_payload = MessageProtocol.create_collaboration_response(
                collaboration_id=collaboration_id,
                agent_id=self.agent_id,
                response=response,
                capabilities=self.capabilities
            )
            
            success = await self.send_message(
                receiver_id=initiator_id,
                message_type=MessageType.COLLABORATION_JOIN if response == "accept" else MessageType.COLLABORATION_LEAVE,
                payload=response_payload
            )
            
            logger.info(f"响应协作请求 {collaboration_id}: {response}")
            return success
            
        except Exception as e:
            logger.error(f"响应协作请求失败: {e}")
            return False
    
    async def request_resource(
        self,
        resource_type: str,
        requirements: Dict[str, Any],
        duration_minutes: int,
        priority: int = 5,
        exclusive: bool = False
    ) -> bool:
        """请求资源"""
        try:
            request_payload = MessageProtocol.create_resource_request(
                resource_type=resource_type,
                resource_requirements=requirements,
                duration_minutes=duration_minutes,
                requester_id=self.agent_id,
                priority=priority,
                exclusive=exclusive
            )
            
            return await self.broadcast(
                message_type=MessageType.RESOURCE_REQUEST,
                payload=request_payload
            )
            
        except Exception as e:
            logger.error(f"请求资源失败: {e}")
            return False
    
    async def offer_resource(
        self,
        resource_type: str,
        available_resources: Dict[str, Any],
        cost: Optional[float] = None,
        availability_window: Optional[Dict[str, str]] = None
    ) -> bool:
        """提供资源"""
        try:
            offer_payload = MessageProtocol.create_resource_offer(
                resource_type=resource_type,
                available_resources=available_resources,
                provider_id=self.agent_id,
                cost=cost,
                availability_window=availability_window
            )
            
            return await self.broadcast(
                message_type=MessageType.RESOURCE_OFFER,
                payload=offer_payload
            )
            
        except Exception as e:
            logger.error(f"提供资源失败: {e}")
            return False
    
    async def _announce_agent_joined(self):
        """广播智能体加入消息"""
        try:
            join_payload = MessageProtocol.create_agent_joined_message(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                capabilities=self.capabilities,
                metadata={
                    "start_time": time.time(),
                    "version": "1.0.0"
                }
            )
            
            await self.broadcast(
                message_type=MessageType.AGENT_JOINED,
                payload=join_payload
            )
            
        except Exception as e:
            logger.error(f"广播智能体加入消息失败: {e}")
    
    async def _announce_agent_left(self):
        """广播智能体离开消息"""
        try:
            uptime = time.time() - self.start_time
            leave_payload = MessageProtocol.create_agent_left_message(
                agent_id=self.agent_id,
                reason="normal_shutdown",
                final_status={
                    "uptime": uptime,
                    "messages_processed": self.message_bus.get_metrics().messages_received
                }
            )
            
            await self.broadcast(
                message_type=MessageType.AGENT_LEFT,
                payload=leave_payload
            )
            
        except Exception as e:
            logger.error(f"广播智能体离开消息失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "uptime": time.time() - self.start_time,
            "groups": self.groups,
            "collaborations": len(self.collaborations),
            "connection_status": self.message_bus.get_connection_status(),
            "metrics": self.message_bus.get_metrics().to_dict(),
            "registered_handlers": list(self.message_bus.message_handlers.keys())
        }
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.message_bus.is_connected()