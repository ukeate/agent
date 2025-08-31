"""
分布式事件处理系统
实现跨节点的事件分发、协调和负载均衡
"""
import asyncio
import json
import hashlib
import socket
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
from enum import Enum
import structlog
try:
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis
    except ImportError:
        aioredis = None

from .events import Event, EventType, EventPriority
from .event_processors import AsyncEventProcessingEngine, EventContext
from .event_store import EventStore

logger = structlog.get_logger(__name__)


class NodeStatus(str, Enum):
    """节点状态"""
    ACTIVE = "active"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAILED = "failed"


class NodeRole(str, Enum):
    """节点角色"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    role: NodeRole
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=lambda: utc_now())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        """检查节点是否存活"""
        time_since_heartbeat = (utc_now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat < timeout_seconds and self.status == NodeStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "port": self.port,
            "status": self.status.value,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "load": self.load,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """从字典创建"""
        return cls(
            node_id=data["node_id"],
            hostname=data["hostname"],
            ip_address=data["ip_address"],
            port=data["port"],
            status=NodeStatus(data["status"]),
            role=NodeRole(data["role"]),
            capabilities=data.get("capabilities", []),
            load=data.get("load", 0.0),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            metadata=data.get("metadata", {})
        )


class ConsistentHash:
    """一致性哈希环"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
    
    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node_id: str) -> None:
        """添加节点到哈希环"""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        
        # 添加虚拟节点
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node_id
        
        # 重新排序键
        self.sorted_keys = sorted(self.ring.keys())
        
        logger.debug(f"节点添加到哈希环", node_id=node_id, virtual_nodes=self.virtual_nodes)
    
    def remove_node(self, node_id: str) -> None:
        """从哈希环移除节点"""
        if node_id not in self.nodes:
            return
        
        self.nodes.remove(node_id)
        
        # 移除虚拟节点
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
        
        # 重新排序键
        self.sorted_keys = sorted(self.ring.keys())
        
        logger.debug(f"节点从哈希环移除", node_id=node_id)
    
    def get_node(self, key: str) -> Optional[str]:
        """获取键对应的节点"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # 二分查找
        idx = self._binary_search(hash_value)
        
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def _binary_search(self, hash_value: int) -> int:
        """二分查找哈希值位置"""
        left, right = 0, len(self.sorted_keys)
        
        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < hash_value:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def get_nodes_for_replication(self, key: str, replication_factor: int = 3) -> List[str]:
        """获取复制节点列表"""
        if not self.ring or replication_factor <= 0:
            return []
        
        hash_value = self._hash(key)
        idx = self._binary_search(hash_value)
        
        nodes = []
        seen = set()
        
        for i in range(len(self.sorted_keys)):
            actual_idx = (idx + i) % len(self.sorted_keys)
            node = self.ring[self.sorted_keys[actual_idx]]
            
            if node not in seen:
                nodes.append(node)
                seen.add(node)
                
                if len(nodes) >= replication_factor:
                    break
        
        return nodes


class DistributedEventCoordinator:
    """分布式事件协调器"""
    
    def __init__(
        self,
        node_id: str,
        redis_client=None,
        event_store: Optional[EventStore] = None,
        processing_engine: Optional[AsyncEventProcessingEngine] = None
    ):
        self.node_id = node_id
        self.redis = redis_client
        self.event_store = event_store
        self.processing_engine = processing_engine
        
        # 节点信息
        self.node_info = NodeInfo(
            node_id=node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_ip_address(),
            port=8000,  # 默认端口
            status=NodeStatus.ACTIVE,
            role=NodeRole.FOLLOWER
        )
        
        # 集群管理
        self.nodes: Dict[str, NodeInfo] = {}
        self.consistent_hash = ConsistentHash()
        
        # 领导选举
        self.leader_key = "cluster:leader"
        self.nodes_key = "cluster:nodes"
        self.election_timeout = 5  # 秒
        self.heartbeat_interval = 2  # 秒
        self.leader_lease_time = 10  # 秒
        
        # 任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._election_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        
        # 事件分发
        self.event_queue_prefix = "node:events:"
        self.pending_events: asyncio.Queue = asyncio.Queue()
        
        # 统计信息
        self.stats = {
            "events_distributed": 0,
            "events_received": 0,
            "events_forwarded": 0,
            "leader_elections": 0,
            "heartbeats_sent": 0
        }
    
    def _get_ip_address(self) -> str:
        """获取本机IP地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    async def start(self) -> None:
        """启动协调器"""
        logger.info(f"分布式事件协调器启动", node_id=self.node_id)
        
        # 注册节点
        await self.register_node()
        
        # 启动心跳
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动领导选举
        self._election_task = asyncio.create_task(self._election_loop())
        
        # 启动节点同步
        self._sync_task = asyncio.create_task(self._sync_nodes_loop())
        
        # 启动事件处理
        asyncio.create_task(self._process_events_loop())
    
    async def stop(self) -> None:
        """停止协调器"""
        logger.info(f"分布式事件协调器停止", node_id=self.node_id)
        
        # 注销节点
        await self.unregister_node()
        
        # 取消任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._election_task:
            self._election_task.cancel()
        if self._sync_task:
            self._sync_task.cancel()
    
    async def register_node(self) -> None:
        """注册节点到集群"""
        if not self.redis:
            return
        
        try:
            # 将节点信息存储到Redis
            node_data = json.dumps(self.node_info.to_dict())
            await self.redis.hset(self.nodes_key, self.node_id, node_data)
            await self.redis.expire(self.nodes_key, 3600)  # 1小时过期
            
            # 添加到哈希环
            self.consistent_hash.add_node(self.node_id)
            
            logger.info(f"节点注册成功", node_id=self.node_id)
            
        except Exception as e:
            logger.error(f"节点注册失败", node_id=self.node_id, error=str(e))
    
    async def unregister_node(self) -> None:
        """从集群注销节点"""
        if not self.redis:
            return
        
        try:
            # 从Redis删除节点信息
            await self.redis.hdel(self.nodes_key, self.node_id)
            
            # 从哈希环移除
            self.consistent_hash.remove_node(self.node_id)
            
            # 如果是领导者，释放领导权
            if self.node_info.role == NodeRole.LEADER:
                await self.redis.delete(self.leader_key)
            
            logger.info(f"节点注销成功", node_id=self.node_id)
            
        except Exception as e:
            logger.error(f"节点注销失败", node_id=self.node_id, error=str(e))
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # 更新心跳时间
                self.node_info.last_heartbeat = utc_now()
                
                # 更新负载信息
                if self.processing_engine:
                    stats = self.processing_engine.get_stats()
                    queue_size = sum(stats.get("queue_sizes", {}).values())
                    self.node_info.load = min(1.0, queue_size / 1000)  # 归一化负载
                
                # 发送心跳
                if self.redis:
                    node_data = json.dumps(self.node_info.to_dict())
                    await self.redis.hset(self.nodes_key, self.node_id, node_data)
                    await self.redis.expire(self.nodes_key, 3600)
                    
                    # 如果是领导者，续约
                    if self.node_info.role == NodeRole.LEADER:
                        await self.redis.setex(self.leader_key, self.leader_lease_time, self.node_id)
                
                self.stats["heartbeats_sent"] += 1
                
            except Exception as e:
                logger.error(f"心跳发送失败", error=str(e))
    
    async def _election_loop(self) -> None:
        """领导选举循环"""
        while True:
            try:
                await asyncio.sleep(self.election_timeout)
                
                if not self.redis:
                    continue
                
                # 检查当前领导者
                current_leader = await self.redis.get(self.leader_key)
                
                if not current_leader:
                    # 没有领导者，尝试成为领导者
                    await self._try_become_leader()
                elif current_leader.decode() == self.node_id:
                    # 自己是领导者，续约
                    await self.redis.setex(self.leader_key, self.leader_lease_time, self.node_id)
                else:
                    # 其他节点是领导者
                    if self.node_info.role != NodeRole.FOLLOWER:
                        self.node_info.role = NodeRole.FOLLOWER
                        logger.info(f"成为追随者", node_id=self.node_id, leader=current_leader.decode())
                
            except Exception as e:
                logger.error(f"领导选举失败", error=str(e))
    
    async def _try_become_leader(self) -> None:
        """尝试成为领导者"""
        if not self.redis:
            return
        
        try:
            # 使用SET NX实现分布式锁
            self.node_info.role = NodeRole.CANDIDATE
            
            result = await self.redis.set(
                self.leader_key,
                self.node_id,
                nx=True,
                ex=self.leader_lease_time
            )
            
            if result:
                self.node_info.role = NodeRole.LEADER
                self.stats["leader_elections"] += 1
                logger.info(f"成为领导者", node_id=self.node_id)
                
                # 领导者特权任务
                await self._on_become_leader()
            else:
                self.node_info.role = NodeRole.FOLLOWER
                
        except Exception as e:
            logger.error(f"尝试成为领导者失败", error=str(e))
            self.node_info.role = NodeRole.FOLLOWER
    
    async def _on_become_leader(self) -> None:
        """成为领导者时的处理"""
        # 这里可以执行一些只有领导者才能做的任务
        # 比如：重新平衡负载、清理过期数据等
        pass
    
    async def _sync_nodes_loop(self) -> None:
        """节点同步循环"""
        while True:
            try:
                await asyncio.sleep(5)  # 每5秒同步一次
                
                if not self.redis:
                    continue
                
                # 获取所有节点信息
                nodes_data = await self.redis.hgetall(self.nodes_key)
                
                self.nodes.clear()
                self.consistent_hash = ConsistentHash()
                
                for node_id_bytes, node_data_bytes in nodes_data.items():
                    try:
                        node_id = node_id_bytes.decode()
                        node_data = json.loads(node_data_bytes.decode())
                        node_info = NodeInfo.from_dict(node_data)
                        
                        # 检查节点是否存活
                        if node_info.is_alive():
                            self.nodes[node_id] = node_info
                            self.consistent_hash.add_node(node_id)
                        else:
                            # 清理死节点
                            await self.redis.hdel(self.nodes_key, node_id)
                            
                    except Exception as e:
                        logger.error(f"节点信息解析失败", node_id=node_id_bytes, error=str(e))
                
                logger.debug(f"节点同步完成", active_nodes=len(self.nodes))
                
            except Exception as e:
                logger.error(f"节点同步失败", error=str(e))
    
    async def distribute_event(self, event: Event) -> str:
        """分发事件到合适的节点"""
        self.stats["events_distributed"] += 1
        
        try:
            # 生成路由键
            routing_key = self._get_routing_key(event)
            
            # 获取目标节点
            target_node = self.consistent_hash.get_node(routing_key)
            
            if not target_node:
                # 没有可用节点，本地处理
                return await self._process_locally(event)
            
            if target_node == self.node_id:
                # 本地处理
                return await self._process_locally(event)
            else:
                # 转发到目标节点
                return await self._forward_to_node(target_node, event)
                
        except Exception as e:
            logger.error(f"事件分发失败", error=str(e))
            # 失败时本地处理
            return await self._process_locally(event)
    
    def _get_routing_key(self, event: Event) -> str:
        """生成事件路由键"""
        # 基于事件类型和源进行路由
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        source = getattr(event, 'source', 'unknown')
        conversation_id = getattr(event, 'conversation_id', '')
        
        # 优先使用会话ID，确保同一会话的事件在同一节点处理
        if conversation_id:
            return f"conversation:{conversation_id}"
        
        return f"{event_type}:{source}"
    
    async def _process_locally(self, event: Event) -> str:
        """本地处理事件"""
        if self.processing_engine:
            priority = getattr(event, 'priority', EventPriority.NORMAL)
            await self.processing_engine.submit_event(event, priority)
        
        if self.event_store:
            await self.event_store.append_event(event)
        
        return "local"
    
    async def _forward_to_node(self, target_node: str, event: Event) -> str:
        """转发事件到目标节点"""
        if not self.redis:
            return await self._process_locally(event)
        
        try:
            # 序列化事件
            event_data = {
                "id": event.id if hasattr(event, 'id') else None,
                "type": event.type.value if hasattr(event.type, 'value') else str(event.type),
                "source": getattr(event, 'source', None),
                "target": getattr(event, 'target', None),
                "data": getattr(event, 'data', {}),
                "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else utc_now().isoformat(),
                "correlation_id": getattr(event, 'correlation_id', None),
                "conversation_id": getattr(event, 'conversation_id', None),
                "session_id": getattr(event, 'session_id', None),
                "priority": event.priority.value if hasattr(event, 'priority') and hasattr(event.priority, 'value') else 'normal'
            }
            
            # 发送到目标节点的队列
            queue_key = f"{self.event_queue_prefix}{target_node}"
            await self.redis.lpush(queue_key, json.dumps(event_data))
            
            self.stats["events_forwarded"] += 1
            
            logger.debug(f"事件转发", target_node=target_node, event_type=event_data["type"])
            
            return target_node
            
        except Exception as e:
            logger.error(f"事件转发失败", target_node=target_node, error=str(e))
            return await self._process_locally(event)
    
    async def _process_events_loop(self) -> None:
        """处理接收到的事件"""
        if not self.redis:
            return
        
        queue_key = f"{self.event_queue_prefix}{self.node_id}"
        
        while True:
            try:
                # 从队列获取事件
                result = await self.redis.brpop(queue_key, timeout=1)
                
                if result:
                    _, event_data_bytes = result
                    event_data = json.loads(event_data_bytes.decode())
                    
                    # 重建事件对象
                    event = Event(
                        id=event_data.get("id"),
                        type=EventType(event_data["type"]) if "type" in event_data else EventType.MESSAGE_SENT,
                        source=event_data.get("source", ""),
                        target=event_data.get("target"),
                        data=event_data.get("data", {}),
                        timestamp=datetime.fromisoformat(event_data["timestamp"]) if "timestamp" in event_data else utc_now(),
                        correlation_id=event_data.get("correlation_id"),
                        conversation_id=event_data.get("conversation_id"),
                        session_id=event_data.get("session_id"),
                        priority=EventPriority(event_data.get("priority", "normal"))
                    )
                    
                    self.stats["events_received"] += 1
                    
                    # 本地处理
                    await self._process_locally(event)
                    
            except Exception as e:
                logger.error(f"处理接收事件失败", error=str(e))
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        return {
            "node_id": self.node_id,
            "role": self.node_info.role.value,
            "status": self.node_info.status.value,
            "load": self.node_info.load,
            "active_nodes": len(self.nodes),
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "role": node.role.value,
                    "load": node.load,
                    "is_alive": node.is_alive()
                }
                for node_id, node in self.nodes.items()
            },
            "stats": self.stats
        }
    
    async def rebalance_load(self) -> None:
        """重新平衡负载（仅领导者可执行）"""
        if self.node_info.role != NodeRole.LEADER:
            logger.warning("只有领导者可以重新平衡负载")
            return
        
        try:
            # 计算平均负载
            total_load = sum(node.load for node in self.nodes.values())
            avg_load = total_load / len(self.nodes) if self.nodes else 0
            
            # 找出高负载和低负载节点
            high_load_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.load > avg_load * 1.2  # 超过平均值20%
            ]
            
            low_load_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.load < avg_load * 0.8  # 低于平均值20%
            ]
            
            logger.info(
                "负载平衡分析",
                avg_load=avg_load,
                high_load_nodes=high_load_nodes,
                low_load_nodes=low_load_nodes
            )
            
            # TODO: 实现实际的负载迁移逻辑
            
        except Exception as e:
            logger.error(f"负载平衡失败", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


@dataclass
class DistributedEvent:
    """分布式事件数据结构"""
    event_id: str
    event_type: str
    source_node: str
    target_nodes: List[str]
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


class DistributedEventBus:
    """企业级分布式事件总线"""
    
    def __init__(self, redis_client, node_id: str):
        self.redis_client = redis_client
        self.node_id = node_id
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_handlers: Dict[str, Callable] = {}
        self.running = False
        self._listen_task: Optional[asyncio.Task] = None
        
        # 事件存储配置
        self.event_stream_prefix = "distributed_events:"
        self.max_stream_length = 10000
        
        logger.info("分布式事件总线初始化", node_id=node_id)
    
    async def start(self):
        """启动分布式事件总线"""
        self.running = True
        self._listen_task = asyncio.create_task(self._listen_for_events())
        logger.info("分布式事件总线启动", node_id=self.node_id)
    
    async def stop(self):
        """停止分布式事件总线"""
        self.running = False
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        logger.info("分布式事件总线停止", node_id=self.node_id)
    
    async def publish(self, event: DistributedEvent) -> bool:
        """发布分布式事件"""
        try:
            # 序列化事件
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "source_node": event.source_node,
                "target_nodes": event.target_nodes,
                "payload": event.payload,
                "timestamp": event.timestamp.isoformat(),
                "priority": event.priority,
                "retry_count": event.retry_count,
                "max_retries": event.max_retries
            }
            
            # 发布到Redis流
            stream_key = f"{self.event_stream_prefix}{event.event_type}"
            await self.redis_client.xadd(
                stream_key,
                event_data,
                maxlen=self.max_stream_length
            )
            
            # 通知目标节点
            for target_node in event.target_nodes:
                notification_key = f"node_notifications:{target_node}"
                await self.redis_client.lpush(
                    notification_key,
                    json.dumps({
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "stream_key": stream_key
                    })
                )
                # 设置通知过期时间
                await self.redis_client.expire(notification_key, 3600)
            
            logger.debug(
                "分布式事件发布成功",
                event_id=event.event_id,
                event_type=event.event_type,
                targets=len(event.target_nodes)
            )
            
            return True
            
        except Exception as e:
            logger.error("分布式事件发布失败", event_id=event.event_id, error=str(e))
            return False
    
    async def subscribe(
        self, 
        event_type: str, 
        handler: Callable[[DistributedEvent], Any]
    ):
        """订阅事件类型"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        
        logger.info("订阅分布式事件", event_type=event_type, node_id=self.node_id)
    
    async def _listen_for_events(self):
        """监听分布式事件"""
        notification_key = f"node_notifications:{self.node_id}"
        
        while self.running:
            try:
                # 监听通知
                result = await self.redis_client.brpop(notification_key, timeout=1)
                
                if result:
                    _, notification_data = result
                    notification = json.loads(notification_data.decode())
                    
                    # 处理事件通知
                    await self._handle_event_notification(notification)
                    
            except Exception as e:
                logger.error("监听分布式事件失败", error=str(e))
                await asyncio.sleep(1)
    
    async def _handle_event_notification(self, notification: Dict[str, Any]):
        """处理事件通知"""
        try:
            event_id = notification["event_id"]
            event_type = notification["event_type"]
            stream_key = notification["stream_key"]
            
            # 从流中读取事件
            events = await self.redis_client.xread({stream_key: "$"}, count=1, block=1000)
            
            for stream, messages in events:
                for message_id, fields in messages:
                    # 检查是否是目标事件
                    if fields.get("event_id") == event_id:
                        # 重构事件对象
                        event = DistributedEvent(
                            event_id=fields["event_id"],
                            event_type=fields["event_type"],
                            source_node=fields["source_node"],
                            target_nodes=json.loads(fields["target_nodes"]),
                            payload=json.loads(fields["payload"]),
                            timestamp=datetime.fromisoformat(fields["timestamp"]),
                            priority=int(fields.get("priority", 0)),
                            retry_count=int(fields.get("retry_count", 0)),
                            max_retries=int(fields.get("max_retries", 3))
                        )
                        
                        # 处理事件
                        await self._process_event(event)
                        break
            
        except Exception as e:
            logger.error("处理事件通知失败", notification=notification, error=str(e))
    
    async def _process_event(self, event: DistributedEvent):
        """处理分布式事件"""
        if event.event_type in self.subscribers:
            for handler in self.subscribers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(
                        "事件处理器失败",
                        event_id=event.event_id,
                        handler=handler.__name__,
                        error=str(e)
                    )
                    
                    # 重试机制
                    if event.retry_count < event.max_retries:
                        event.retry_count += 1
                        await asyncio.sleep(2 ** event.retry_count)  # 指数退避
                        await self.publish(event)
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """获取事件统计"""
        stats = {
            "node_id": self.node_id,
            "subscribed_events": len(self.subscribers),
            "event_types": list(self.subscribers.keys())
        }
        
        # 获取流统计
        for event_type in self.subscribers.keys():
            stream_key = f"{self.event_stream_prefix}{event_type}"
            try:
                info = await self.redis_client.xinfo_stream(stream_key)
                stats[f"{event_type}_stream_length"] = info["length"]
            except:
                stats[f"{event_type}_stream_length"] = 0
        
        return stats