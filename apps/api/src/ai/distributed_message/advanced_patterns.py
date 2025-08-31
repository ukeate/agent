"""
高级通信模式
实现多播通信、流式数据传输、事件流处理和智能路由
"""

import asyncio
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Set, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .models import Message, MessageHeader, MessageType, MessagePriority
from .client import NATSClient

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """路由策略"""
    ROUND_ROBIN = "round_robin"     # 轮询
    LEAST_LOADED = "least_loaded"   # 最少负载
    CAPABILITY_BASED = "capability_based"  # 基于能力
    GEOGRAPHIC = "geographic"       # 地理位置
    PRIORITY_BASED = "priority_based"  # 基于优先级


class StreamMode(str, Enum):
    """流模式"""
    PUSH = "push"  # 推送模式
    PULL = "pull"  # 拉取模式
    BIDIRECTIONAL = "bidirectional"  # 双向流


@dataclass
class AgentCapability:
    """智能体能力"""
    agent_id: str
    capabilities: Set[str]
    load_factor: float = 0.0  # 0.0-1.0，0表示空闲，1表示满负载
    location: Optional[str] = None
    priority: int = 0
    last_seen: datetime = field(default_factory=datetime.now)
    
    def matches_requirement(self, required_capability: str) -> bool:
        """检查是否匹配能力要求"""
        return required_capability in self.capabilities


@dataclass
class MulticastGroup:
    """多播组"""
    group_id: str
    group_name: str
    members: Set[str]
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    max_members: int = 100
    
    def add_member(self, agent_id: str) -> bool:
        """添加成员"""
        if len(self.members) >= self.max_members:
            return False
        self.members.add(agent_id)
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """移除成员"""
        if agent_id in self.members:
            self.members.remove(agent_id)
            return True
        return False


@dataclass
class StreamChunk:
    """数据流块"""
    chunk_id: str
    stream_id: str
    sequence: int
    data: bytes
    is_last: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataStream:
    """数据流"""
    stream_id: str
    sender_id: str
    receiver_id: str
    stream_mode: StreamMode
    total_chunks: Optional[int] = None
    received_chunks: int = 0
    chunks: Dict[int, StreamChunk] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def is_complete(self) -> bool:
        """检查流是否完整"""
        if self.total_chunks is None:
            return False
        return self.received_chunks >= self.total_chunks
    
    def get_missing_sequences(self) -> List[int]:
        """获取缺失的序列号"""
        if self.total_chunks is None:
            return []
        received_sequences = set(self.chunks.keys())
        expected_sequences = set(range(self.total_chunks))
        return sorted(expected_sequences - received_sequences)


class MulticastManager:
    """多播管理器"""
    
    def __init__(self, client: NATSClient):
        self.client = client
        self.groups: Dict[str, MulticastGroup] = {}
        self.agent_groups: Dict[str, Set[str]] = {}  # 智能体所属的组
        
    def create_group(
        self,
        group_name: str,
        description: Optional[str] = None,
        max_members: int = 100
    ) -> str:
        """创建多播组"""
        group_id = str(uuid.uuid4())
        group = MulticastGroup(
            group_id=group_id,
            group_name=group_name,
            members=set(),
            description=description,
            max_members=max_members
        )
        self.groups[group_id] = group
        logger.info(f"创建多播组: {group_name} ({group_id})")
        return group_id
    
    def join_group(self, group_id: str, agent_id: str) -> bool:
        """加入组"""
        if group_id not in self.groups:
            return False
        
        group = self.groups[group_id]
        if group.add_member(agent_id):
            if agent_id not in self.agent_groups:
                self.agent_groups[agent_id] = set()
            self.agent_groups[agent_id].add(group_id)
            logger.info(f"智能体 {agent_id} 加入组 {group.group_name}")
            return True
        return False
    
    def leave_group(self, group_id: str, agent_id: str) -> bool:
        """离开组"""
        if group_id not in self.groups:
            return False
        
        group = self.groups[group_id]
        if group.remove_member(agent_id):
            if agent_id in self.agent_groups:
                self.agent_groups[agent_id].discard(group_id)
                if not self.agent_groups[agent_id]:
                    del self.agent_groups[agent_id]
            logger.info(f"智能体 {agent_id} 离开组 {group.group_name}")
            return True
        return False
    
    async def send_multicast(
        self,
        group_id: str,
        message: Message,
        exclude_agents: Optional[Set[str]] = None
    ) -> int:
        """发送多播消息"""
        if group_id not in self.groups:
            logger.error(f"多播组不存在: {group_id}")
            return 0
        
        group = self.groups[group_id]
        exclude_agents = exclude_agents or set()
        target_agents = group.members - exclude_agents
        
        sent_count = 0
        for agent_id in target_agents:
            try:
                # 为每个目标智能体创建单独的消息副本
                agent_message = Message(
                    header=MessageHeader(
                        message_id=str(uuid.uuid4()),
                        correlation_id=message.header.correlation_id,
                        multicast_group=group_id
                    ),
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type=message.message_type,
                    payload=message.payload,
                    topic=f"agents.direct.{agent_id}"
                )
                
                success = await self.client.js_publish(
                    subject=f"agents.direct.{agent_id}",
                    data=agent_message.to_bytes(),
                    stream="AGENTS_DIRECT"
                )
                
                if success:
                    sent_count += 1
                    
            except Exception as e:
                logger.error(f"发送多播消息到 {agent_id} 失败: {e}")
        
        logger.info(f"多播消息发送完成: {sent_count}/{len(target_agents)} 成功")
        return sent_count
    
    def get_group_info(self, group_id: str) -> Optional[Dict[str, Any]]:
        """获取组信息"""
        if group_id not in self.groups:
            return None
        
        group = self.groups[group_id]
        return {
            "group_id": group.group_id,
            "group_name": group.group_name,
            "description": group.description,
            "member_count": len(group.members),
            "members": list(group.members),
            "max_members": group.max_members,
            "created_at": group.created_at.isoformat()
        }
    
    def list_groups(self) -> List[Dict[str, Any]]:
        """列出所有组"""
        return [self.get_group_info(group_id) for group_id in self.groups.keys()]
    
    def get_agent_groups(self, agent_id: str) -> List[str]:
        """获取智能体所属的组"""
        return list(self.agent_groups.get(agent_id, set()))


class StreamingManager:
    """流式数据传输管理器"""
    
    def __init__(self, client: NATSClient):
        self.client = client
        self.active_streams: Dict[str, DataStream] = {}
        self.chunk_size = 64 * 1024  # 64KB
        self.stream_timeout = 300.0  # 5分钟超时
        
    async def send_stream(
        self,
        receiver_id: str,
        data: bytes,
        stream_mode: StreamMode = StreamMode.PUSH,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """发送数据流"""
        stream_id = str(uuid.uuid4())
        chunks = self._split_data_into_chunks(data, stream_id)
        total_chunks = len(chunks)
        
        # 创建数据流记录
        stream = DataStream(
            stream_id=stream_id,
            sender_id=self.client.agent_id,
            receiver_id=receiver_id,
            stream_mode=stream_mode,
            total_chunks=total_chunks,
            metadata=metadata or {}
        )
        self.active_streams[stream_id] = stream
        
        try:
            # 发送流开始消息
            start_header = MessageHeader(
                message_id=str(uuid.uuid4()),
                stream_id=stream_id
            )
            start_message = Message(
                header=start_header,
                sender_id=self.client.agent_id,
                receiver_id=receiver_id,
                message_type=MessageType.DATA_STREAM_START,
                payload={
                    "stream_id": stream_id,
                    "total_chunks": total_chunks,
                    "chunk_size": self.chunk_size,
                    "metadata": metadata or {}
                },
                topic=f"agents.direct.{receiver_id}"
            )
            
            await self.client.js_publish(
                subject=f"agents.direct.{receiver_id}",
                data=start_message.to_bytes(),
                stream="AGENTS_STREAMS"
            )
            
            # 发送数据块
            for chunk in chunks:
                chunk_header = MessageHeader(
                    message_id=str(uuid.uuid4()),
                    stream_id=stream_id
                )
                chunk_message = Message(
                    header=chunk_header,
                    sender_id=self.client.agent_id,
                    receiver_id=receiver_id,
                    message_type=MessageType.DATA_STREAM_CHUNK,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "sequence": chunk.sequence,
                        "data": chunk.data.hex(),  # 十六进制编码
                        "is_last": chunk.is_last,
                        "metadata": chunk.metadata
                    },
                    topic=f"agents.direct.{receiver_id}"
                )
                
                await self.client.js_publish(
                    subject=f"agents.direct.{receiver_id}",
                    data=chunk_message.to_bytes(),
                    stream="AGENTS_STREAMS"
                )
                
                # 添加少量延迟以避免网络拥塞
                await asyncio.sleep(0.001)
            
            # 发送流结束消息
            end_header = MessageHeader(
                message_id=str(uuid.uuid4()),
                stream_id=stream_id
            )
            end_message = Message(
                header=end_header,
                sender_id=self.client.agent_id,
                receiver_id=receiver_id,
                message_type=MessageType.DATA_STREAM_END,
                payload={"stream_id": stream_id},
                topic=f"agents.direct.{receiver_id}"
            )
            
            await self.client.js_publish(
                subject=f"agents.direct.{receiver_id}",
                data=end_message.to_bytes(),
                stream="AGENTS_STREAMS"
            )
            
            logger.info(f"数据流发送完成: {stream_id}, 共 {total_chunks} 块")
            return stream_id
            
        except Exception as e:
            logger.error(f"发送数据流失败: {e}")
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            raise
    
    def _split_data_into_chunks(self, data: bytes, stream_id: str) -> List[StreamChunk]:
        """将数据分割成块"""
        chunks = []
        total_size = len(data)
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total_size)
            chunk_data = data[start:end]
            
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stream_id=stream_id,
                sequence=i,
                data=chunk_data,
                is_last=(i == num_chunks - 1),
                metadata={"size": len(chunk_data)}
            )
            chunks.append(chunk)
        
        return chunks
    
    async def handle_stream_start(self, message: Message) -> bool:
        """处理流开始消息"""
        try:
            payload = message.payload
            stream_id = payload["stream_id"]
            total_chunks = payload["total_chunks"]
            metadata = payload.get("metadata", {})
            
            stream = DataStream(
                stream_id=stream_id,
                sender_id=message.sender_id,
                receiver_id=message.receiver_id,
                stream_mode=StreamMode.PUSH,  # 接收方总是PUSH模式
                total_chunks=total_chunks,
                metadata=metadata
            )
            self.active_streams[stream_id] = stream
            
            logger.info(f"开始接收数据流: {stream_id}, 预期 {total_chunks} 块")
            return True
            
        except Exception as e:
            logger.error(f"处理流开始消息失败: {e}")
            return False
    
    async def handle_stream_chunk(self, message: Message) -> bool:
        """处理流数据块"""
        try:
            payload = message.payload
            stream_id = message.header.stream_id
            
            if stream_id not in self.active_streams:
                logger.error(f"未找到数据流: {stream_id}")
                return False
            
            stream = self.active_streams[stream_id]
            sequence = payload["sequence"]
            chunk_data = bytes.fromhex(payload["data"])
            
            chunk = StreamChunk(
                chunk_id=payload["chunk_id"],
                stream_id=stream_id,
                sequence=sequence,
                data=chunk_data,
                is_last=payload.get("is_last", False),
                metadata=payload.get("metadata", {})
            )
            
            stream.chunks[sequence] = chunk
            stream.received_chunks += 1
            
            logger.debug(f"接收数据块: {stream_id} [{sequence}], 进度: {stream.received_chunks}/{stream.total_chunks}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理流数据块失败: {e}")
            return False
    
    async def handle_stream_end(self, message: Message) -> Optional[bytes]:
        """处理流结束消息"""
        try:
            stream_id = message.payload["stream_id"]
            
            if stream_id not in self.active_streams:
                logger.error(f"未找到数据流: {stream_id}")
                return None
            
            stream = self.active_streams[stream_id]
            stream.completed_at = datetime.now()
            
            if stream.is_complete():
                # 重建完整数据
                data_parts = []
                for seq in sorted(stream.chunks.keys()):
                    data_parts.append(stream.chunks[seq].data)
                
                complete_data = b''.join(data_parts)
                logger.info(f"数据流接收完成: {stream_id}, 总大小: {len(complete_data)} 字节")
                
                # 清理
                del self.active_streams[stream_id]
                return complete_data
            else:
                missing = stream.get_missing_sequences()
                logger.warning(f"数据流不完整: {stream_id}, 缺失块: {missing}")
                return None
                
        except Exception as e:
            logger.error(f"处理流结束消息失败: {e}")
            return None
    
    def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流信息"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return {
            "stream_id": stream.stream_id,
            "sender_id": stream.sender_id,
            "receiver_id": stream.receiver_id,
            "stream_mode": stream.stream_mode.value,
            "total_chunks": stream.total_chunks,
            "received_chunks": stream.received_chunks,
            "progress": stream.received_chunks / stream.total_chunks if stream.total_chunks else 0,
            "is_complete": stream.is_complete(),
            "missing_sequences": stream.get_missing_sequences(),
            "created_at": stream.created_at.isoformat(),
            "completed_at": stream.completed_at.isoformat() if stream.completed_at else None,
            "metadata": stream.metadata
        }
    
    def list_active_streams(self) -> List[Dict[str, Any]]:
        """列出活跃的数据流"""
        return [self.get_stream_info(stream_id) for stream_id in self.active_streams.keys()]


class SmartRouter:
    """智能路由器"""
    
    def __init__(self, client: NATSClient):
        self.client = client
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.routing_stats: Dict[str, int] = {}
        self.last_route_index = 0  # 用于轮询路由
        
    def register_agent_capability(
        self,
        agent_id: str,
        capabilities: Set[str],
        load_factor: float = 0.0,
        location: Optional[str] = None,
        priority: int = 0
    ):
        """注册智能体能力"""
        self.agent_capabilities[agent_id] = AgentCapability(
            agent_id=agent_id,
            capabilities=capabilities,
            load_factor=load_factor,
            location=location,
            priority=priority
        )
        logger.info(f"注册智能体能力: {agent_id}, 能力: {capabilities}")
    
    def update_agent_load(self, agent_id: str, load_factor: float):
        """更新智能体负载"""
        if agent_id in self.agent_capabilities:
            self.agent_capabilities[agent_id].load_factor = load_factor
            self.agent_capabilities[agent_id].last_seen = datetime.now()
    
    def find_best_agent(
        self,
        required_capability: str,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        exclude_agents: Optional[Set[str]] = None
    ) -> Optional[str]:
        """找到最佳的智能体"""
        exclude_agents = exclude_agents or set()
        
        # 筛选具有所需能力的智能体
        candidates = [
            capability for capability in self.agent_capabilities.values()
            if (capability.matches_requirement(required_capability) and
                capability.agent_id not in exclude_agents)
        ]
        
        if not candidates:
            return None
        
        # 根据策略选择最佳智能体
        if strategy == RoutingStrategy.LEAST_LOADED:
            # 选择负载最低的智能体
            best = min(candidates, key=lambda x: x.load_factor)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # 轮询选择
            self.last_route_index = (self.last_route_index + 1) % len(candidates)
            best = candidates[self.last_route_index]
        elif strategy == RoutingStrategy.PRIORITY_BASED:
            # 优先级最高的智能体
            best = max(candidates, key=lambda x: x.priority)
        elif strategy == RoutingStrategy.CAPABILITY_BASED:
            # 能力最匹配的智能体（拥有最多相关能力）
            best = max(candidates, key=lambda x: len(x.capabilities))
        else:
            # 默认选择第一个
            best = candidates[0]
        
        # 更新路由统计
        if best.agent_id not in self.routing_stats:
            self.routing_stats[best.agent_id] = 0
        self.routing_stats[best.agent_id] += 1
        
        logger.debug(f"路由选择: {required_capability} -> {best.agent_id} (策略: {strategy.value})")
        return best.agent_id
    
    async def route_message(
        self,
        message: Message,
        required_capability: str,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED
    ) -> bool:
        """路由消息到最佳智能体"""
        best_agent = self.find_best_agent(required_capability, strategy)
        
        if not best_agent:
            logger.error(f"未找到具有能力 {required_capability} 的智能体")
            return False
        
        try:
            # 更新消息接收者
            routed_message = Message(
                header=message.header,
                sender_id=message.sender_id,
                receiver_id=best_agent,
                message_type=message.message_type,
                payload=message.payload,
                topic=f"agents.direct.{best_agent}"
            )
            
            success = await self.client.js_publish(
                subject=f"agents.direct.{best_agent}",
                data=routed_message.to_bytes(),
                stream="AGENTS_DIRECT"
            )
            
            if success:
                logger.info(f"消息路由成功: {message.header.message_id} -> {best_agent}")
                return True
            else:
                logger.error(f"消息路由失败: 发送到 {best_agent} 失败")
                return False
                
        except Exception as e:
            logger.error(f"消息路由异常: {e}")
            return False
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        total_routes = sum(self.routing_stats.values())
        agent_stats = []
        
        for agent_id, count in self.routing_stats.items():
            capability = self.agent_capabilities.get(agent_id)
            agent_stats.append({
                "agent_id": agent_id,
                "route_count": count,
                "route_percentage": count / total_routes * 100 if total_routes > 0 else 0,
                "load_factor": capability.load_factor if capability else 0,
                "capabilities": list(capability.capabilities) if capability else [],
                "last_seen": capability.last_seen.isoformat() if capability else None
            })
        
        return {
            "total_routes": total_routes,
            "registered_agents": len(self.agent_capabilities),
            "agent_statistics": sorted(agent_stats, key=lambda x: x["route_count"], reverse=True)
        }
    
    def cleanup_stale_agents(self, max_age_minutes: int = 30):
        """清理过期的智能体能力信息"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        stale_agents = [
            agent_id for agent_id, capability in self.agent_capabilities.items()
            if capability.last_seen < cutoff_time
        ]
        
        for agent_id in stale_agents:
            del self.agent_capabilities[agent_id]
            if agent_id in self.routing_stats:
                del self.routing_stats[agent_id]
        
        if stale_agents:
            logger.info(f"清理过期智能体: {stale_agents}")


class AdvancedCommunicationManager:
    """高级通信模式管理器"""
    
    def __init__(self, client: NATSClient):
        self.client = client
        self.multicast_manager = MulticastManager(client)
        self.streaming_manager = StreamingManager(client)
        self.smart_router = SmartRouter(client)
        
        # 事件流处理
        self.event_handlers: Dict[str, Callable] = {}
        self.event_subscriptions: Dict[str, object] = {}
        
    async def start(self):
        """启动高级通信管理器"""
        # 订阅系统事件流
        await self._subscribe_to_system_events()
        logger.info("高级通信模式管理器已启动")
    
    async def stop(self):
        """停止高级通信管理器"""
        # 取消所有事件订阅
        for subscription in self.event_subscriptions.values():
            if hasattr(subscription, 'unsubscribe'):
                await subscription.unsubscribe()
        self.event_subscriptions.clear()
        logger.info("高级通信模式管理器已停止")
    
    async def _subscribe_to_system_events(self):
        """订阅系统事件"""
        # 订阅数据流事件
        stream_subscription = await self.client.subscribe(
            subject="agents.streams.*",
            callback=self._handle_stream_message,
            queue="stream_handlers"
        )
        if stream_subscription:
            self.event_subscriptions["streams"] = stream_subscription
        
        # 订阅组管理事件
        group_subscription = await self.client.subscribe(
            subject="agents.groups.*",
            callback=self._handle_group_message,
            queue="group_handlers"
        )
        if group_subscription:
            self.event_subscriptions["groups"] = group_subscription
    
    async def _handle_stream_message(self, msg):
        """处理数据流消息"""
        try:
            message = Message.from_bytes(msg.data)
            
            if message.message_type == MessageType.DATA_STREAM_START:
                await self.streaming_manager.handle_stream_start(message)
            elif message.message_type == MessageType.DATA_STREAM_CHUNK:
                await self.streaming_manager.handle_stream_chunk(message)
            elif message.message_type == MessageType.DATA_STREAM_END:
                data = await self.streaming_manager.handle_stream_end(message)
                if data and "stream_complete" in self.event_handlers:
                    await self.event_handlers["stream_complete"](message.header.stream_id, data)
                    
        except Exception as e:
            logger.error(f"处理数据流消息失败: {e}")
    
    async def _handle_group_message(self, msg):
        """处理组管理消息"""
        try:
            message = Message.from_bytes(msg.data)
            # 处理组管理相关的消息
            if "group_message" in self.event_handlers:
                await self.event_handlers["group_message"](message)
        except Exception as e:
            logger.error(f"处理组消息失败: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        self.event_handlers[event_type] = handler
        logger.info(f"注册事件处理器: {event_type}")
    
    def unregister_event_handler(self, event_type: str):
        """取消注册事件处理器"""
        if event_type in self.event_handlers:
            del self.event_handlers[event_type]
            logger.info(f"取消注册事件处理器: {event_type}")
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """获取高级通信统计信息"""
        return {
            "multicast": {
                "groups_count": len(self.multicast_manager.groups),
                "total_members": sum(len(group.members) for group in self.multicast_manager.groups.values())
            },
            "streaming": {
                "active_streams": len(self.streaming_manager.active_streams),
                "stream_details": self.streaming_manager.list_active_streams()
            },
            "routing": self.smart_router.get_routing_statistics(),
            "event_handlers": list(self.event_handlers.keys()),
            "event_subscriptions": list(self.event_subscriptions.keys())
        }